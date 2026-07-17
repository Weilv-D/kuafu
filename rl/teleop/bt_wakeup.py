#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bluetooth gamepad wakeup daemon for Flydigi VADER2P and similar BLE HID devices.

Problem
-------
Some BLE gamepads (Flydigi VADER2P, certain 8BitDo models) enter a low-power
idle state after Bluetooth connection and stop sending HID input reports until
physically operated (stick move or button press).  The kernel ``/dev/input/js0``
device exists and INIT events are received, but no real-time events flow.

This manifests as: ``bluetoothctl`` shows ``Connected: yes``, pygame/SDL sees
the joystick, but ``get_axis()`` / ``get_button()`` return stale values and
``--show-axes`` shows nothing.

Solution
--------
This module provides:

1. ``is_gamepad_alive()`` — read ``/dev/input/js0`` for 1 second and report
   whether any real-time events arrived.
2. ``bt_reconnect(mac)`` — disconnect + reconnect via ``bluetoothctl``.
3. ``watch_loop(mac)`` — background loop that monitors event flow; when the
   gamepad goes silent for ``idle_threshold`` seconds, it attempts a reconnect
   and prints a wake-up reminder.

Usage
-----
Standalone (blocks):

    python -m rl.teleop.bt_wakeup F8:3B:26:8F:FE:F3

As a library (non-blocking checker):

    from rl.teleop.bt_wakeup import is_gamepad_alive, bt_reconnect
    if not is_gamepad_alive():
        bt_reconnect("F8:3B:26:8F:FE:F3")

Environment
-----------
``KUAFU_BT_MAC`` — default MAC if not passed as argument.
``KUAFU_JS_DEVICE`` — js device path (default ``/dev/input/js0``).
``KUAFU_BT_IDLE_THRESHOLD`` — seconds of silence before reconnect (default 10).
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import time

JS_EVENT_FMT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FMT)
JS_EVENT_INIT = 0x80


def is_gamepad_alive(device: str = "/dev/input/js0", timeout: float = 1.0) -> bool:
    """Return True if the gamepad sent at least one real-time event in *timeout* seconds."""
    import fcntl

    try:
        f = open(device, "rb")
    except (FileNotFoundError, PermissionError):
        return False

    fd = f.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    # Drain INIT events
    while True:
        data = f.read(JS_EVENT_SIZE)
        if data is None or len(data) < JS_EVENT_SIZE:
            break

    deadline = time.monotonic() + timeout
    alive = False
    while time.monotonic() < deadline:
        data = f.read(JS_EVENT_SIZE)
        if data is not None and len(data) == JS_EVENT_SIZE:
            _t, _v, etype, _n = struct.unpack(JS_EVENT_FMT, data)
            if not (etype & JS_EVENT_INIT):
                alive = True
                break
        time.sleep(0.01)

    f.close()
    return alive


def bt_reconnect(mac: str, settle: float = 2.0) -> bool:
    """Disconnect and reconnect a Bluetooth device via bluetoothctl.

    Returns True if the device reports Connected: yes after reconnect.
    """
    print(f"[bt_wakeup] disconnecting {mac} ...")
    subprocess.run(
        ["bluetoothctl", "disconnect", mac],
        capture_output=True, timeout=10,
    )
    time.sleep(1.0)

    print(f"[bt_wakeup] reconnecting {mac} ...")
    result = subprocess.run(
        ["bluetoothctl", "connect", mac],
        capture_output=True, timeout=15, text=True,
    )
    ok = "Connection successful" in result.stdout
    if ok:
        print(f"[bt_wakeup] connected, waiting {settle:.0f}s for services ...")
        time.sleep(settle)
    else:
        print(f"[bt_wakeup] reconnect failed: {result.stderr.strip()}")
    return ok


def count_events(device: str = "/dev/input/js0", timeout: float = 1.0) -> int:
    """Count real-time events in *timeout* seconds (non-blocking)."""
    import fcntl

    try:
        f = open(device, "rb")
    except (FileNotFoundError, PermissionError):
        return -1

    fd = f.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    while f.read(JS_EVENT_SIZE):
        pass  # drain

    deadline = time.monotonic() + timeout
    count = 0
    while time.monotonic() < deadline:
        data = f.read(JS_EVENT_SIZE)
        if data is not None and len(data) == JS_EVENT_SIZE:
            _t, _v, etype, _n = struct.unpack(JS_EVENT_FMT, data)
            if not (etype & JS_EVENT_INIT):
                count += 1
        time.sleep(0.01)

    f.close()
    return count


def watch_loop(mac: str | None = None, device: str | None = None,
               idle_threshold: float | None = None) -> None:
    """Monitor gamepad event flow; reconnect when silent for too long.

    This is a blocking loop intended to run in a separate process or thread.
    When the gamepad goes silent (0 events for *idle_threshold* seconds), it
    attempts a Bluetooth reconnect and reminds the operator to physically
    operate the controller.
    """
    mac = mac or os.environ.get("KUAFU_BT_MAC", "")
    device = device or os.environ.get("KUAFU_JS_DEVICE", "/dev/input/js0")
    idle_threshold = idle_threshold or float(
        os.environ.get("KUAFU_BT_IDLE_THRESHOLD", "10")
    )

    if not mac:
        print("[bt_wakeup] KUAFU_BT_MAC not set; cannot reconnect. "
              "Monitoring only.")
        reconnect_enabled = False
    else:
        reconnect_enabled = True

    print(f"[bt_wakeup] monitoring {device} "
          f"(idle_threshold={idle_threshold:.0f}s, "
          f"reconnect={'on' if reconnect_enabled else 'off'})")

    last_event_time = time.monotonic()
    last_reconnect_time = 0.0
    was_alive = True
    check_interval = 2.0

    while True:
        events = count_events(device, timeout=check_interval)

        if events > 0:
            last_event_time = time.monotonic()
            if not was_alive:
                print("[bt_wakeup] ✅ gamepad awake — events flowing")
                was_alive = True
        else:
            silent_for = time.monotonic() - last_event_time
            if was_alive:
                print(f"[bt_wakeup] ⚠️  gamepad went silent "
                      f"({silent_for:.0f}s without events)")
                was_alive = False

            if silent_for > idle_threshold and reconnect_enabled:
                now = time.monotonic()
                # Don't reconnect more than once per 15s
                if now - last_reconnect_time > 15.0:
                    print(f"[bt_wakeup] silent for {silent_for:.0f}s > "
                          f"{idle_threshold:.0f}s threshold; reconnecting ...")
                    bt_reconnect(mac)
                    last_reconnect_time = now
                    last_event_time = time.monotonic()
                else:
                    remaining = 15.0 - (now - last_reconnect_time)
                    if remaining > 0:
                        print(f"[bt_wakeup] push a stick/button to wake the "
                              f"gamepad (next reconnect in {remaining:.0f}s)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Bluetooth gamepad wakeup daemon"
    )
    parser.add_argument("mac", nargs="?", default=None,
                        help="Bluetooth MAC (default: KUAFU_BT_MAC env)")
    parser.add_argument("--device", default=None,
                        help="js device path (default: KUAFU_JS_DEVICE or /dev/input/js0)")
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
        print("error: MAC address required (positional arg or KUAFU_BT_MAC)")
        sys.exit(1)

    watch_loop(mac=mac, device=device, idle_threshold=args.idle)


if __name__ == "__main__":
    main()
