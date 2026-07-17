#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native joystick calibration - reads /dev/input/js0 directly, no pygame/SDL.

pygame's get_axis()/get_button() can return stale cached values on Bluetooth LE
gamepads (Flydigi VADER2P etc.) because SDL's event pump doesn't always flush
joystick state. This tool reads the kernel joystick interface directly, which
is the same data pygame eventually gets but without the SDL caching layer.

Usage:
    python -m rl.teleop.calibrate_native
    (or: python rl/teleop/calibrate_native.py)

Then follow the prompts. Output is ready-to-export KUAFU_AXIS_* / KUAFU_BTN_*
lines.
"""
from __future__ import annotations

import fcntl
import os
import struct
import sys
import time

JS_DEVICE = "/dev/input/js0"
JS_EVENT_FMT = "IhBB"       # time_ms, value, type, number
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FMT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

# Axis value thresholds
AXIS_DETECT_THRESHOLD = 20000   # out of 32767, ~60% push
AXIS_DEAD_THRESHOLD = 1000      # below this = released/centered


def _open_js():
    """Open js0 non-blocking, drain INIT events, return (file, n_axes, n_btns)."""
    try:
        f = open(JS_DEVICE, "rb")
    except PermissionError:
        print(f"权限不足: 无法读取 {JS_DEVICE}")
        print("运行: sudo usermod -aG input $USER  然后重新登录")
        sys.exit(1)
    except FileNotFoundError:
        print(f"未找到 {JS_DEVICE} - 手柄未连接")
        sys.exit(1)

    fd = f.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    # Drain INIT events to discover axis/button count
    n_axes = 0
    n_btns = 0
    while True:
        data = f.read(JS_EVENT_SIZE)
        if data is None or len(data) < JS_EVENT_SIZE:
            break
        _t, _val, etype, num = struct.unpack(JS_EVENT_FMT, data)
        if etype & JS_EVENT_INIT:
            if etype & JS_EVENT_AXIS:
                n_axes = max(n_axes, num + 1)
            if etype & JS_EVENT_BUTTON:
                n_btns = max(n_btns, num + 1)

    if n_axes == 0:
        # Fallback: read from sysfs
        try:
            n_axes = int(open(
                f"/sys/class/input/js0/device/../capabilities/abs"
            ).read().strip(), 16).bit_count()
        except Exception:
            n_axes = 8

    return f, n_axes, n_btns


def _read_events(f, timeout=0.1):
    """Read all pending events, return list of (type, number, value)."""
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = f.read(JS_EVENT_SIZE)
        if data is None or len(data) < JS_EVENT_SIZE:
            time.sleep(0.005)
            continue
        _t, val, etype, num = struct.unpack(JS_EVENT_FMT, data)
        if etype & JS_EVENT_INIT:
            continue    # skip init
        events.append((etype, num, val))
    return events


def _wait_axis(f, n_axes, prompt, timeout=20.0):
    """Wait for user to push an axis past threshold.

    Returns (axis_index, peak_value) or (None, 0).
    The peak_value is captured at the moment of detection, NOT after release,
    so the sign reflects the user's intended push direction (not spring-back).
    """
    print(f"\n>> {prompt}")
    print(f"   (超时 {timeout:.0f}s 跳过)")
    axis_state = [0] * n_axes
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = _read_events(f, 0.05)
        for etype, num, val in events:
            if etype & JS_EVENT_AXIS and num < n_axes:
                axis_state[num] = val
                if abs(val) > AXIS_DETECT_THRESHOLD:
                    print(f"   ✓ 轴{num} 响应 (value={val:+d})")
                    _wait_axis_release(f, num)
                    return num, val
    print("   (超时)")
    return None, 0


def _wait_axis_release(f, axis_idx, timeout=3.0):
    """Wait for axis to return near center."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = _read_events(f, 0.02)
        for etype, num, val in events:
            if etype & JS_EVENT_AXIS and num == axis_idx:
                if abs(val) < AXIS_DEAD_THRESHOLD:
                    return
    return


def _wait_button(f, n_btns, prompt, timeout=20.0):
    """Wait for user to press a button. Return button index or None."""
    print(f"\n>> {prompt}")
    print(f"   (超时 {timeout:.0f}s 跳过)")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = _read_events(f, 0.05)
        for etype, num, val in events:
            if etype & JS_EVENT_BUTTON and num < n_btns and val == 1:
                print(f"   ✓ 按钮{num} 按下")
                _wait_button_release(f, num)
                return num
    print("   (超时)")
    return None


def _wait_button_release(f, btn_idx, timeout=3.0):
    """Wait for button release."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = _read_events(f, 0.02)
        for etype, num, val in events:
            if etype & JS_EVENT_BUTTON and num == btn_idx and val == 0:
                return
    return


def _get_axis_direction(f, axis_idx, timeout=5.0):
    """Read the peak value to determine push direction.

    Kept for backward compatibility but no longer used for V-axis invert
    detection -- _wait_axis now returns the value at detection time, which
    avoids capturing the spring-back overshoot after release.
    """
    deadline = time.monotonic() + timeout
    peak = 0
    while time.monotonic() < deadline:
        events = _read_events(f, 0.02)
        for etype, num, val in events:
            if etype & JS_EVENT_AXIS and num == axis_idx:
                if abs(val) > abs(peak):
                    peak = val
        if abs(peak) > AXIS_DETECT_THRESHOLD:
            time.sleep(0.3)
            events = _read_events(f, 0.02)
            for etype, num, val in events:
                if etype & JS_EVENT_AXIS and num == axis_idx:
                    if abs(val) > abs(peak):
                        peak = val
            break
    return peak


def main():
    print("=" * 55)
    print("  KUAFU 手柄标定 (原生 /dev/input/js0, 不依赖 pygame)")
    print("=" * 55)

    f, n_axes, n_btns = _open_js()
    print(f"  设备: {JS_DEVICE}")
    print(f"  轴数: {n_axes}  按钮数: {n_btns}")
    print("=" * 55)

    result = {}

    # ---- 轴标定 ----
    axis_steps = [
        ("KUAFU_AXIS_V",  "推左摇杆 向上到最大 (前进方向)"),
        ("KUAFU_AXIS_W",  "推右摇杆 向右到最大 (右转方向)"),
        ("KUAFU_AXIS_LT", "捏紧 LT 扳机 (降 D0)"),
        ("KUAFU_AXIS_RT", "捏紧 RT 扳机 (升 D0)"),
    ]

    invert_v = False
    for env_key, prompt in axis_steps:
        idx, peak_val = _wait_axis(f, n_axes, prompt)
        if idx is None:
            continue
        result[env_key] = str(idx)

        # For V axis: check the sign of the push at detection time.
        # kernel js convention: pushing UP on left stick Y gives -32767 on
        # most gamepads (including VADER2P). We want push-up = forward = +v,
        # so if push-up gives a negative raw value, we need INVERT=1.
        # IMPORTANT: use the detection-time value, NOT a re-read after release,
        # because the spring-back overshoot has the opposite sign.
        if env_key == "KUAFU_AXIS_V":
            invert_v = peak_val < 0
            print(f"   推上检测值={peak_val:+d} -> KUAFU_AXIS_V_INVERT={'1' if invert_v else '0'}")

    result["KUAFU_AXIS_V_INVERT"] = "1" if invert_v else "0"
    result["KUAFU_AXIS_W_INVERT"] = "0"  # right = positive by convention

    # ---- 按钮标定 ----
    btn_steps = [
        ("KUAFU_BTN_ARM",    "按 ARM 使能键 (建议 START)"),
        ("KUAFU_BTN_DISARM", "按 DISARM 卸能键 (建议 Select/Back)"),
        ("KUAFU_BTN_ESTOP",  "按 ESTOP 急停键 (建议 A)"),
    ]
    for env_key, prompt in btn_steps:
        idx = _wait_button(f, n_btns, prompt)
        if idx is None:
            continue
        result[env_key] = str(idx)

    f.close()

    # ---- 输出 ----
    print()
    print("=" * 55)
    print("  标定完成! 复制以下行到 ~/.bashrc 或启动脚本:")
    print("=" * 55)
    order = [
        "KUAFU_AXIS_V", "KUAFU_AXIS_W",
        "KUAFU_AXIS_LT", "KUAFU_AXIS_RT",
        "KUAFU_AXIS_V_INVERT", "KUAFU_AXIS_W_INVERT",
        "KUAFU_BTN_ARM", "KUAFU_BTN_DISARM", "KUAFU_BTN_ESTOP",
    ]
    lines = [f"export {k}={result[k]}" for k in order if k in result]
    for line in lines:
        print(f"  {line}")
    print()
    print(f"  一行版:")
    print(f"  {'; '.join(lines)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
