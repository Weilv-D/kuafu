#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bluetooth gamepad wakeup daemon for Flydigi VADER2P and similar BLE HID devices.

Problem
-------
Some BLE gamepads (Flydigi VADER2P, certain 8BitDo models) enter a low-power
idle state after Bluetooth connection and stop sending HID input reports until
physically operated (stick move or button press).  The kernel ``/dev/input/js0``
device exists and INIT events are received, but no real-time events flow.

Solution
--------
1. ``is_gamepad_alive()`` — check whether events are flowing.
2. ``bt_reconnect(mac)`` — disconnect + reconnect via ``bluetoothctl``.
3. ``watch_loop(mac)`` — background monitor; reconnect when silent too long.

Usage
-----
Standalone::

    python -m rl.teleop.bt_wakeup F8:3B:26:8F:FE:F3

Library (used by ``teleop_single``)::

    from rl.teleop.bt_wakeup import bt_reconnect
    bt_reconnect("F8:3B:26:8F:FE:F3")

Environment::

    KUAFU_BT_MAC            default MAC
    KUAFU_JS_DEVICE         js device path (default /dev/input/js0)
    KUAFU_BT_IDLE_THRESHOLD seconds of silence before reconnect (default 10)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from rl.teleop.native_joystick import NativeJoystick


def is_gamepad_alive(device: str = "/dev/input/js0", timeout: float = 1.0) -> bool:
    """Return ``True`` if the gamepad sent at least one real-time event in *timeout* s."""
    joy = NativeJoystick(device)
    if not joy.connected:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        joy.poll()
        if joy._event_count > 0:
            joy.close()
            return True
        time.sleep(0.01)
    joy.close()
    return False


def count_events(device: str = "/dev/input/js0", timeout: float = 1.0) -> int:
    """Count real-time events in *timeout* seconds."""
    joy = NativeJoystick(device)
    if not joy.connected:
        return -1
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        joy.poll()
        time.sleep(0.01)
    count = joy._event_count
    joy.close()
    return count


def bt_reconnect(mac: str, settle: float = 2.0) -> bool:
    """Disconnect and reconnect a Bluetooth device via bluetoothctl.

    Returns ``True`` if the device reports ``Connected: yes`` after reconnect.
    """
    print(f"[bt_wakeup] disconnecting {mac} ...")
    subprocess.run(["bluetoothctl", "disconnect", mac],
                   capture_output=True, timeout=10)
    time.sleep(1.0)
    print(f"[bt_wakeup] reconnecting {mac} ...")
    result = subprocess.run(["bluetoothctl", "connect", mac],
                            capture_output=True, timeout=15, text=True)
    ok = "Connection successful" in result.stdout
    if ok:
        print(f"[bt_wakeup] connected, waiting {settle:.0f}s for services ...")
        time.sleep(settle)
    else:
        print(f"[bt_wakeup] reconnect failed: {result.stderr.strip()}")
    return ok


def watch_loop(mac: str | None = None, device: str | None = None,
               idle_threshold: float | None = None) -> None:
    """Monitor gamepad event flow; reconnect when silent for too long."""
    mac = mac or os.environ.get("KUAFU_BT_MAC", "")
    device = device or os.environ.get("KUAFU_JS_DEVICE", "/dev/input/js0")
    idle_threshold = idle_threshold or float(
        os.environ.get("KUAFU_BT_IDLE_THRESHOLD", "10"))
    reconnect_enabled = bool(mac)

    print(f"[bt_wakeup] monitoring {device} "
          f"(idle={idle_threshold:.0f}s, reconnect={'on' if reconnect_enabled else 'off'})")

    joy = NativeJoystick(device)
    last_event_time = time.monotonic()
    last_reconnect = 0.0
    was_alive = bool(joy.connected and joy._event_count > 0)

    while True:
        if not joy.connected:
            joy.reconnect()
            time.sleep(2.0)
            continue

        joy.poll()
        if joy._event_count > 0 or not joy.is_idle:
            last_event_time = joy._last_event_time
            if not was_alive:
                print("[bt_wakeup] ✅ gamepad awake — events flowing")
                was_alive = True
        else:
            silent = joy.idle_seconds
            if was_alive:
                print(f"[bt_wakeup] ⚠️  gamepad silent ({silent:.0f}s)")
                was_alive = False
            if silent > idle_threshold and reconnect_enabled:
                now = time.monotonic()
                if now - last_reconnect > 15.0:
                    print(f"[bt_wakeup] silent {silent:.0f}s > {idle_threshold:.0f}s; "
                          f"reconnecting ...")
                    bt_reconnect(mac)
                    joy.reconnect()
                    last_reconnect = now
                    last_event_time = time.monotonic()
                else:
                    print(f"[bt_wakeup] push a stick to wake "
                          f"(next reconnect in {15.0 - (now - last_reconnect):.0f}s)")

        time.sleep(2.0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="BLE gamepad wakeup daemon")
    parser.add_argument("mac", nargs="?", default=None,
                        help="Bluetooth MAC (default: KUAFU_BT_MAC)")
    parser.add_argument("--device", default=None,
                        help="js device (default: KUAFU_JS_DEVICE)")
    parser.add_argument("--idle", type=float, default=None,
                        help="idle threshold seconds (default: 10)")
    parser.add_argument("--check", action="store_true",
                        help="one-shot alive check and exit")
    args = parser.parse_args()

    mac = args.mac or os.environ.get("KUAFU_BT_MAC", "")
    device = args.device or os.environ.get("KUAFU_JS_DEVICE", "/dev/input/js0")

    if args.check:
        alive = is_gamepad_alive(device, timeout=2.0)
        events = count_events(device, timeout=2.0)
        print(f"device={device}  alive={alive}  events_in_2s={events}")
        sys.exit(0 if alive else 1)

    if not mac:
        print("error: MAC required (arg or KUAFU_BT_MAC)")
        sys.exit(1)

    watch_loop(mac=mac, device=device, idle_threshold=args.idle)


if __name__ == "__main__":
    main()
