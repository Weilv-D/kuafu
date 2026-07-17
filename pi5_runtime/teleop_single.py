#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-process teleop: gamepad + serial + command in one process.

When ``serial_node`` and ``teleop_node`` run as separate processes they can
compete for the same UART (``/dev/ttyAMA10``), causing heartbeat frames to be
dropped or the serial read to fail with "multiple access on port".  This module
combines the gamepad reader, the serial link, and the command arbiter into a
single process so there is exactly one serial fd and one js0 fd.

This is the recommended entry point for baseline-LQR teleop (no ONNX model).

Usage::

    cd ~/aspace/kuafu_repo
    PYTHONPATH=. python -m pi5_runtime.teleop_single

Environment variables (same as ``gamepad_source``)::

    KUAFU_AXIS_V=1 KUAFU_AXIS_W=2 KUAFU_AXIS_LT=5 KUAFU_AXIS_RT=4
    KUAFU_AXIS_V_INVERT=1 KUAFU_AXIS_W_INVERT=0
    KUAFU_AXIS_LT_INVERT=0 KUAFU_AXIS_RT_INVERT=0
    KUAFU_BTN_ARM=9 KUAFU_BTN_DISARM=8 KUAFU_BTN_ESTOP=0
    KUAFU_JS_DEVICE=/dev/input/js0
    KUAFU_SERIAL_PORT=/dev/ttyAMA10
    KUAFU_SERIAL_BAUD=921600
"""
from __future__ import annotations

import argparse
import fcntl
import os
import signal
import struct
import sys
import time

import numpy as np

from pi5_runtime.protocol import (
    StreamDecoder, TEL_HEALTH, decode_health_payload,
    hello_frame, command_frames,
)
from rl.env.contract import ACTION_DIM
from rl.teleop.shaping import normalize_trigger, shape_axis
import kuafu_physics as P

JS_EVENT_FMT = "IhBB"
JS_EVENT_SIZE = 8
JS_EVENT_INIT = 0x80
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_AXIS_MAX = 32767.0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip() not in ("0", "false", "False", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KUAFU single-process teleop (baseline LQR, no ONNX)"
    )
    parser.add_argument("--port", default=os.environ.get("KUAFU_SERIAL_PORT", "/dev/ttyAMA10"))
    parser.add_argument("--baudrate", type=int, default=int(os.environ.get("KUAFU_SERIAL_BAUD", "921600")))
    parser.add_argument("--duration", type=float, default=0,
                        help="run for N seconds (0 = until Ctrl-C)")
    args = parser.parse_args()

    # --- Axis / button mapping ---
    axis_v = _env_int("KUAFU_AXIS_V", 1)
    axis_w = _env_int("KUAFU_AXIS_W", 2)
    axis_lt = _env_int("KUAFU_AXIS_LT", 4)
    axis_rt = _env_int("KUAFU_AXIS_RT", 5)
    btn_arm = _env_int("KUAFU_BTN_ARM", 7)
    btn_disarm = _env_int("KUAFU_BTN_DISARM", 6)
    btn_estop = _env_int("KUAFU_BTN_ESTOP", 0)
    invert_v = _env_bool("KUAFU_AXIS_V_INVERT", True)
    invert_w = _env_bool("KUAFU_AXIS_W_INVERT", False)
    invert_lt = _env_bool("KUAFU_AXIS_LT_INVERT", False)
    invert_rt = _env_bool("KUAFU_AXIS_RT_INVERT", False)
    js_device = os.environ.get("KUAFU_JS_DEVICE", "/dev/input/js0")

    stick_deadzone = 0.08
    stick_gamma = 2.0
    trigger_deadzone = 0.10
    d0_rate_mm_s = 40.0

    # --- Open gamepad ---
    try:
        jf = open(js_device, "rb")
    except (FileNotFoundError, PermissionError) as exc:
        print(f"[teleop_single] cannot open {js_device}: {exc}")
        sys.exit(1)
    fcntl.fcntl(jf.fileno(), fcntl.F_SETFL,
                os.O_NONBLOCK | fcntl.fcntl(jf.fileno(), fcntl.F_GETFL))
    # Drain INIT events
    while jf.read(JS_EVENT_SIZE):
        pass
    print(f"[teleop_single] gamepad: {js_device}")

    axes: list[float] = [0.0] * 12
    buttons: list[bool] = [False] * 24
    prev_buttons: dict[int, bool] = {}

    armed = False
    estop_latched = False
    d0 = P.D0_MIN

    # --- Open serial ---
    import serial as pyserial
    s = pyserial.Serial(args.port, baudrate=args.baudrate, timeout=0)
    dec = StreamDecoder()
    seq = 0
    s.write(hello_frame(seq, int(time.monotonic() * 1000), P.model_hash()).encode())
    seq = (seq + 2) & 0xFFFF
    print(f"[teleop_single] serial: {args.port} @ {args.baudrate}, HELLO sent (hash={P.model_hash()})")

    stopping = {"flag": False}

    def _shutdown(_signum, _frame):
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[teleop_single] READY — wake gamepad (push stick), "
          f"btn{btn_arm}=ARM btn{btn_disarm}=DISARM btn{btn_estop}=ESTOP")

    start = time.monotonic()
    last_poll = start
    tick = 0
    last_health_mode = None
    last_report = 0.0
    zero_action = np.zeros(ACTION_DIM, dtype=np.float32)

    try:
        while not stopping["flag"]:
            now = time.monotonic()
            dt = now - last_poll
            last_poll = now
            tick += 1

            if args.duration > 0 and now - start > args.duration:
                break

            # --- Read gamepad events ---
            while True:
                data = jf.read(JS_EVENT_SIZE)
                if data is None or len(data) < JS_EVENT_SIZE:
                    break
                _t, val, etype, num = struct.unpack(JS_EVENT_FMT, data)
                if etype & JS_EVENT_INIT:
                    continue
                if etype & JS_EVENT_AXIS:
                    if num < len(axes):
                        axes[num] = val / JS_AXIS_MAX
                elif etype & JS_EVENT_BUTTON:
                    if num < len(buttons):
                        pressed = bool(val)
                        if pressed and not prev_buttons.get(num, False):
                            if num == btn_arm:
                                armed = True
                                estop_latched = False
                                print("[teleop_single] ARMED", flush=True)
                            elif num == btn_disarm:
                                armed = False
                                print("[teleop_single] DISARMED", flush=True)
                            elif num == btn_estop:
                                armed = False
                                estop_latched = True
                                print("[teleop_single] ESTOP", flush=True)
                        prev_buttons[num] = pressed
                        buttons[num] = pressed

            if estop_latched:
                v_cmd = 0.0
                w_cmd = 0.0
                mode = 4
            elif armed:
                vy = shape_axis(-axes[axis_v] if invert_v else axes[axis_v],
                                stick_deadzone, stick_gamma)
                wx = shape_axis(-axes[axis_w] if invert_w else axes[axis_w],
                                stick_deadzone, stick_gamma)
                v_cmd = vy * 0.5
                w_cmd = wx * 1.0
                lt = normalize_trigger(axes[axis_lt], trigger_deadzone, invert=invert_lt)
                rt = normalize_trigger(axes[axis_rt], trigger_deadzone, invert=invert_rt)
                d0 += (rt - lt) * d0_rate_mm_s * dt
                d0 = max(P.D0_MIN, min(P.D0_MAX, d0))
                mode = 2
            else:
                v_cmd = 0.0
                w_cmd = 0.0
                lt = normalize_trigger(axes[axis_lt], trigger_deadzone, invert=invert_lt)
                rt = normalize_trigger(axes[axis_rt], trigger_deadzone, invert=invert_rt)
                d0 += (rt - lt) * d0_rate_mm_s * dt
                d0 = max(P.D0_MIN, min(P.D0_MAX, d0))
                mode = 1

            # --- Send command (50 Hz) ---
            if tick % 2 == 0:
                ts = int(now * 1000)
                try:
                    hb, res = command_frames(seq, ts, mode, v_cmd, w_cmd, d0, zero_action)
                    seq = (seq + 2) & 0xFFFF
                    s.write(hb.encode() + res.encode())
                except (ValueError, OSError):
                    pass

            # --- Read telemetry ---
            for f in dec.feed(s.read(256)):
                if f.type == TEL_HEALTH:
                    h = decode_health_payload(f.payload)
                    modes = {0: "STARTUP", 1: "STAND", 2: "ACTIVE",
                             3: "CLIMB", 4: "FAULT"}
                    if h.mode != last_health_mode or now - last_report > 5.0:
                        bits = []
                        for name, bit in [("SERVO", 0x10), ("IMU", 0x20),
                                          ("WL", 0x40), ("WR", 0x80)]:
                            if h.fault_mask & bit:
                                bits.append(name)
                        fault_str = ",".join(bits) if bits else "none"
                        print(f"[teleop_single] STM32={modes.get(h.mode, h.mode)} "
                              f"faults={fault_str} "
                              f"cmd(v={v_cmd:+.3f} w={w_cmd:+.3f} d0={d0:.1f} m={mode})",
                              flush=True)
                        last_health_mode = h.mode
                        last_report = now

            time.sleep(0.01)
    finally:
        # Send one ESTOP on exit
        try:
            ts = int(time.monotonic() * 1000)
            hb, res = command_frames(seq, ts, 4, 0.0, 0.0, P.D0_MIN, zero_action)
            s.write(hb.encode() + res.encode())
        except (ValueError, OSError):
            pass
        s.close()
        jf.close()
        print("[teleop_single] stopped")


if __name__ == "__main__":
    main()
