#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-process baseline-LQR teleop (no ONNX model required).

Combines gamepad input, the STM32 serial link, and command generation into
one process.  This avoids the UART contention that occurs when
``serial_node`` and ``teleop_node`` run as separate processes (heartbeat
frames get dropped and the STM32 never transitions out of STAND).

Reading path
------------
Gamepad axes/buttons are read from ``/dev/input/js0`` via
:class:`~rl.teleop.native_joystick.NativeJoystick` (direct kernel interface,
no pygame/SDL).  The STM32 link uses the versioned 0xA5 protocol: a HELLO
frame establishes ``link_compatible``, then 50 Hz heartbeat+action frames
carry ``(mode, v, ω, D0)`` and a zero residual (baseline LQR only).

Two-state arm/disarm model
--------------------------
==============  ==============  ==================  ====================
State           Button         Wire mode           Firmware behaviour
==============  ==============  ==================  ====================
DISARMED (def)  Back / btn8    IDLE → STAND(1)     LQR holds balance,
                                                  wheels do not track
ARMED           START / btn9   MANUAL → ACTIVE(2)  LQR tracks v/ω commands
ESTOP           A / btn0       ESTOP → FAULT(4)    Latched stop
==============  ==============  ==================  ====================

Usage
-----

::

    cd ~/aspace/kuafu_repo
    PYTHONPATH=. python -m pi5_runtime.teleop_single

Environment variables (set from ``--calibrate`` output)::

    KUAFU_AXIS_V=1 KUAFU_AXIS_W=2 KUAFU_AXIS_LT=5 KUAFU_AXIS_RT=4
    KUAFU_AXIS_V_INVERT=1 KUAFU_AXIS_W_INVERT=0
    KUAFU_AXIS_LT_INVERT=0 KUAFU_AXIS_RT_INVERT=0
    KUAFU_BTN_ARM=9 KUAFU_BTN_DISARM=8 KUAFU_BTN_ESTOP=0
    KUAFU_JS_DEVICE=/dev/input/js0
    KUAFU_SERIAL_PORT=/dev/ttyAMA10
    KUAFU_SERIAL_BAUD=921600
    KUAFU_BT_MAC=F8:3B:26:8F:FE:F3   # enables BLE idle auto-reconnect
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import numpy as np

from pi5_runtime.protocol import (
    StreamDecoder, TEL_HEALTH, decode_health_payload,
    hello_frame, command_frames,
)
from rl.env.contract import ACTION_DIM
from rl.teleop.native_joystick import NativeJoystick
from rl.teleop.shaping import normalize_trigger, shape_axis
import kuafu_physics as P


# -----------------------------------------------------------------------
# config helpers
# -----------------------------------------------------------------------
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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="KUAFU single-process teleop (baseline LQR, no ONNX)"
    )
    parser.add_argument("--port",
                        default=os.environ.get("KUAFU_SERIAL_PORT", "/dev/ttyAMA10"))
    parser.add_argument("--baudrate", type=int,
                        default=int(os.environ.get("KUAFU_SERIAL_BAUD", "921600")))
    parser.add_argument("--duration", type=float, default=0.0,
                        help="run for N seconds (0 = until Ctrl-C)")
    args = parser.parse_args()

    # --- config from environment ---
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
    bt_mac = os.environ.get("KUAFU_BT_MAC", "")

    stick_deadzone = _env_float("KUAFU_STICK_DEADZONE", 0.08)
    stick_gamma = _env_float("KUAFU_STICK_GAMMA", 2.0)
    trigger_deadzone = _env_float("KUAFU_TRIGGER_DEADZONE", 0.10)
    d0_rate = _env_float("KUAFU_D0_RATE_MM_S", 40.0)
    idle_reconnect = _env_float("KUAFU_IDLE_RECONNECT", 15.0)

    # --- open gamepad ---
    joy = NativeJoystick(js_device)
    if not joy.connected:
        print(f"[teleop] ERROR: cannot open {js_device}")
        sys.exit(1)
    print(f"[teleop] gamepad: {js_device}")

    prev_buttons: dict[int, bool] = {}
    armed = False
    estop_latched = False
    d0 = P.D0_MIN

    # --- open serial ---
    import serial as pyserial
    ser = pyserial.Serial(args.port, baudrate=args.baudrate, timeout=0)
    decoder = StreamDecoder()
    seq = 0
    ser.write(hello_frame(seq, int(time.monotonic() * 1000),
                          P.model_hash()).encode())
    ser.flush()                     # wait until HELLO is fully transmitted
    time.sleep(0.3)                 # let STM32 process HELLO before heartbeats
    seq = (seq + 2) & 0xFFFF
    print(f"[teleop] serial: {args.port} @ {args.baudrate} "
          f"(hash={P.model_hash()})")

    # --- signal handling ---
    stopping = {"flag": False}

    def _shutdown(_signum, _frame):
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[teleop] READY — wake gamepad (push stick), "
          f"btn{btn_arm}=ARM  btn{btn_disarm}=DISARM  btn{btn_estop}=ESTOP")

    # --- main loop ---
    start = time.monotonic()
    last_poll = start
    tick = 0
    last_mode = None
    last_report = 0.0
    last_reconnect = 0.0
    was_idle = False
    zero_action = np.zeros(ACTION_DIM, dtype=np.float32)
    v_cmd = w_cmd = 0.0
    mode = 1

    try:
        while not stopping["flag"]:
            now = time.monotonic()
            dt = now - last_poll
            last_poll = now
            tick += 1

            if args.duration > 0 and now - start > args.duration:
                break

            # --- gamepad ---
            if not joy.connected:
                if now - last_reconnect > 2.0:
                    if joy.reconnect():
                        print(f"[teleop] gamepad reconnected")
                        last_reconnect = now
                    else:
                        last_reconnect = now
                if not joy.connected:
                    mode = 4  # ESTOP if no gamepad
            else:
                joy.poll()

                # BLE idle detection
                if joy.is_idle:
                    if not was_idle:
                        print("[teleop] ⚠️  gamepad idle — push a stick to wake")
                        was_idle = True
                    elif int(now) % 5 == 0 and now - last_report > 4.5:
                        print(f"[teleop] still idle ({joy.idle_seconds:.0f}s)")
                        last_report = now
                    if (bt_mac and joy.idle_seconds > idle_reconnect
                            and now - last_reconnect > 15.0):
                        print(f"[teleop] idle {joy.idle_seconds:.0f}s, "
                              f"reconnecting {bt_mac} ...")
                        from rl.teleop.bt_wakeup import bt_reconnect
                        bt_reconnect(bt_mac)
                        joy.reconnect()
                        last_reconnect = now
                elif was_idle:
                    print("[teleop] ✅ gamepad awake")
                    was_idle = False

                # button edges (rising-edge only, estop prints once)
                for btn, action in (
                    (btn_arm, "arm"),
                    (btn_disarm, "disarm"),
                    (btn_estop, "estop"),
                ):
                    pressed = joy.get_button(btn)
                    if pressed and not prev_buttons.get(btn, False):
                        if action == "arm":
                            armed = True
                            estop_latched = False
                            print("[teleop] ARMED", flush=True)
                        elif action == "disarm":
                            armed = False
                            print("[teleop] DISARMED", flush=True)
                        elif action == "estop" and not estop_latched:
                            armed = False
                            estop_latched = True
                            print("[teleop] ESTOP", flush=True)
                    prev_buttons[btn] = pressed

                # sticks
                if estop_latched:
                    v_cmd = w_cmd = 0.0
                    mode = 4
                elif armed:
                    vy = shape_axis(
                        -joy.get_axis(axis_v) if invert_v else joy.get_axis(axis_v),
                        stick_deadzone, stick_gamma)
                    wx = shape_axis(
                        -joy.get_axis(axis_w) if invert_w else joy.get_axis(axis_w),
                        stick_deadzone, stick_gamma)
                    v_cmd = vy * 0.5
                    w_cmd = wx * 1.0
                    mode = 2
                else:
                    v_cmd = w_cmd = 0.0
                    mode = 1

                # D0 rate (works in all states)
                lt = normalize_trigger(joy.get_axis(axis_lt),
                                       trigger_deadzone, invert=invert_lt)
                rt = normalize_trigger(joy.get_axis(axis_rt),
                                       trigger_deadzone, invert=invert_rt)
                d0 += (rt - lt) * d0_rate * dt
                d0 = max(P.D0_MIN, min(P.D0_MAX, d0))

            # --- send command (50 Hz) ---
            if tick % 2 == 0:
                ts = int(now * 1000)
                try:
                    hb, res = command_frames(seq, ts, mode, v_cmd, w_cmd,
                                             d0, zero_action)
                    seq = (seq + 2) & 0xFFFF
                    ser.write(hb.encode() + res.encode())
                except (ValueError, OSError):
                    pass

            # --- read telemetry ---
            for frame in decoder.feed(ser.read(256)):
                if frame.type == TEL_HEALTH:
                    h = decode_health_payload(frame.payload)
                    labels = {0: "STARTUP", 1: "STAND", 2: "ACTIVE",
                              3: "CLIMB", 4: "FAULT"}
                    if h.mode != last_mode or now - last_report > 5.0:
                        bits = []
                        for name, bit in [("SERVO", 0x10), ("IMU", 0x20),
                                          ("WL", 0x40), ("WR", 0x80)]:
                            if h.fault_mask & bit:
                                bits.append(name)
                        print(f"[teleop] STM32={labels.get(h.mode, h.mode)} "
                              f"faults={','.join(bits) or 'none'} "
                              f"cmd(v={v_cmd:+.3f} w={w_cmd:+.3f} "
                              f"d0={d0:.1f} m={mode})",
                              flush=True)
                        last_mode = h.mode
                        last_report = now

            time.sleep(0.01)
    finally:
        # Send ESTOP on exit
        try:
            ts = int(time.monotonic() * 1000)
            hb, res = command_frames(seq, ts, 4, 0.0, 0.0,
                                     P.D0_MIN, zero_action)
            ser.write(hb.encode() + res.encode())
        except (ValueError, OSError):
            pass
        ser.close()
        joy.close()
        print("[teleop] stopped")


if __name__ == "__main__":
    main()
