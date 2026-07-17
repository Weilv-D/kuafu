#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive gamepad calibration — guided axis/button discovery.

Reads ``/dev/input/js0`` via :class:`~rl.teleop.native_joystick.NativeJoystick`
and walks the operator through each stick, trigger, and button.  Auto-detects
v-axis and trigger invert directions, then prints ready-to-export
``KUAFU_AXIS_*`` / ``KUAFU_BTN_*`` lines.

Usage::

    python -m rl.teleop.calibrate_native
"""
from __future__ import annotations

import os
import sys
import time

from rl.teleop.native_joystick import JS_EVENT_AXIS, JS_EVENT_BUTTON, NativeJoystick

JS_DEVICE = os.environ.get("KUAFU_JS_DEVICE", "/dev/input/js0")

# Detection thresholds (raw int16 units, ±32767)
_AXIS_DETECT = 20000     # ~60 % deflection
_AXIS_RELEASED = 2000    # near-centre = released


def _drain_and_wait(joy: NativeJoystick, timeout: float = 0.1) -> list:
    """Poll the joystick for *timeout* seconds, return raw (etype, num, val) events."""
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # We need raw events, but NativeJoystick.poll() consumes them.
        # Read the underlying file directly for calibration.
        import struct
        data = joy._file.read(8)
        if data is None or len(data) < 8:
            time.sleep(0.005)
            continue
        _t, val, etype, num = struct.unpack("IhBB", data)
        if etype & 0x80:  # INIT
            continue
        events.append((etype, num, val))
        # Also update the snapshot so get_axis stays consistent
        if etype & JS_EVENT_AXIS and num < joy.n_axes:
            joy.axes[num] = val / 32767.0
        elif etype & JS_EVENT_BUTTON and num < joy.n_buttons:
            joy.buttons[num] = bool(val)
    return events


def wait_axis(joy: NativeJoystick, prompt: str, timeout: float = 20.0):
    """Wait for user to push an axis past threshold.

    Returns ``(axis_index, peak_value)`` or ``(None, 0)``.
    """
    print(f"\n>> {prompt}")
    print(f"   (超时 {timeout:.0f}s 跳过)")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for etype, num, val in _drain_and_wait(joy, 0.05):
            if etype & JS_EVENT_AXIS and abs(val) > _AXIS_DETECT:
                print(f"   ✓ 轴{num} 响应 (value={val:+d})")
                _wait_axis_release(joy, num)
                return num, val
    print("   (超时)")
    return None, 0


def _wait_axis_release(joy: NativeJoystick, axis_idx: int, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for _etype, num, val in _drain_and_wait(joy, 0.02):
            if num == axis_idx and abs(val) < _AXIS_RELEASED:
                return


def wait_button(joy: NativeJoystick, prompt: str, timeout: float = 20.0):
    """Wait for user to press a button.  Returns button index or ``None``."""
    print(f"\n>> {prompt}")
    print(f"   (超时 {timeout:.0f}s 跳过)")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for etype, num, val in _drain_and_wait(joy, 0.05):
            if etype & JS_EVENT_BUTTON and val == 1:
                print(f"   ✓ 按钮{num} 按下")
                _wait_button_release(joy, num)
                return num
    print("   (超时)")
    return None


def _wait_button_release(joy: NativeJoystick, btn_idx: int, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for etype, num, val in _drain_and_wait(joy, 0.02):
            if num == btn_idx and val == 0:
                return


def main() -> None:
    print("=" * 55)
    print("  KUAFU 手柄标定 (原生 js0, 不依赖 pygame/SDL)")
    print("=" * 55)

    joy = NativeJoystick(JS_DEVICE)
    if not joy.connected:
        print(f"未找到 {JS_DEVICE} — 手柄未连接")
        sys.exit(1)
    print(f"  设备: {JS_DEVICE}  轴: {joy.n_axes}  按钮: {joy.n_buttons}")
    print("=" * 55)

    result: dict[str, str] = {}

    # ---- 轴标定 ----
    axis_steps = [
        ("KUAFU_AXIS_V",  "推左摇杆 向上到最大 (前进方向)"),
        ("KUAFU_AXIS_W",  "推右摇杆 向右到最大 (右转方向)"),
        ("KUAFU_AXIS_LT", "捏紧 LT 扳机 (降 D0)"),
        ("KUAFU_AXIS_RT", "捏紧 RT 扳机 (升 D0)"),
    ]

    invert_v = invert_lt = invert_rt = False
    for env_key, prompt in axis_steps:
        idx, peak = wait_axis(joy, prompt)
        if idx is None:
            continue
        result[env_key] = str(idx)
        if env_key == "KUAFU_AXIS_V":
            invert_v = peak < 0
            print(f"   推上检测值={peak:+d} → V_INVERT={'1' if invert_v else '0'}")
        if env_key == "KUAFU_AXIS_LT":
            invert_lt = peak < 0
            print(f"   捏紧检测值={peak:+d} → LT_INVERT={'1' if invert_lt else '0'}")
        if env_key == "KUAFU_AXIS_RT":
            invert_rt = peak < 0
            print(f"   捏紧检测值={peak:+d} → RT_INVERT={'1' if invert_rt else '0'}")

    result["KUAFU_AXIS_V_INVERT"] = "1" if invert_v else "0"
    result["KUAFU_AXIS_W_INVERT"] = "0"
    result["KUAFU_AXIS_LT_INVERT"] = "1" if invert_lt else "0"
    result["KUAFU_AXIS_RT_INVERT"] = "1" if invert_rt else "0"

    # ---- 按钮标定 ----
    for env_key, prompt in [
        ("KUAFU_BTN_ARM",    "按 ARM 使能键 (建议 START)"),
        ("KUAFU_BTN_DISARM", "按 DISARM 卸能键 (建议 Select/Back)"),
        ("KUAFU_BTN_ESTOP",  "按 ESTOP 急停键 (建议 A)"),
    ]:
        idx = wait_button(joy, prompt)
        if idx is not None:
            result[env_key] = str(idx)

    joy.close()

    # ---- 输出 ----
    print()
    print("=" * 55)
    print("  标定完成! 复制以下行到 ~/.bashrc:")
    print("=" * 55)
    order = [
        "KUAFU_AXIS_V", "KUAFU_AXIS_W",
        "KUAFU_AXIS_LT", "KUAFU_AXIS_RT",
        "KUAFU_AXIS_V_INVERT", "KUAFU_AXIS_W_INVERT",
        "KUAFU_AXIS_LT_INVERT", "KUAFU_AXIS_RT_INVERT",
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
