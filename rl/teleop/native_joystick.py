# -*- coding: utf-8 -*-
"""Native /dev/input/jsX reader — no pygame/SDL dependency.

Reads the Linux kernel joystick interface directly via non-blocking file I/O.
This avoids the stale-cache and event-dropping issues that pygame/SDL exhibit
on Bluetooth LE gamepads (Flydigi VADER2P, etc.).

The reader maintains an up-to-date snapshot of all axes and buttons.  Hot-plug
is detected by checking the file descriptor validity; the caller can attempt
``reconnect()`` when the device disappears.
"""
from __future__ import annotations

import fcntl
import os
import struct
import time

_JS_EVENT_FMT = "IhBB"           # time_ms, value, type, number
JS_EVENT_SIZE = struct.calcsize(_JS_EVENT_FMT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
_AXIS_MAX = 32767.0


class NativeJoystick:
    """Non-blocking reader for a single ``/dev/input/jsX`` device.

    Parameters
    ----------
    device : str
        Kernel joystick device path (default ``/dev/input/js0``).
    n_axes : int
        Maximum axis index to track (default 12).
    n_buttons : int
        Maximum button index to track (default 24).
    """

    def __init__(self, device: str = "/dev/input/js0",
                 n_axes: int = 12, n_buttons: int = 24):
        self.device = device
        self.n_axes = n_axes
        self.n_buttons = n_buttons
        self.axes: list[float] = [0.0] * n_axes
        self.buttons: list[bool] = [False] * n_buttons
        self._file = None
        self._event_count = 0
        self._last_event_time = 0.0
        self._open()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _open(self) -> bool:
        """Open the device non-blocking and drain INIT events."""
        self.close()
        self._event_count = 0
        self._last_event_time = time.monotonic()
        try:
            self._file = open(self.device, "rb")
        except (FileNotFoundError, PermissionError, OSError):
            self._file = None
            return False
        fd = self._file.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        self._drain_init()
        return True

    def _drain_init(self) -> None:
        """Consume INIT events to populate resting axis/button values."""
        if self._file is None:
            return
        while True:
            data = self._file.read(JS_EVENT_SIZE)
            if data is None or len(data) < JS_EVENT_SIZE:
                break
            _t, val, etype, num = struct.unpack(_JS_EVENT_FMT, data)
            if not (etype & JS_EVENT_INIT):
                continue
            if etype & JS_EVENT_AXIS and num < self.n_axes:
                self.axes[num] = val / _AXIS_MAX
            elif etype & JS_EVENT_BUTTON and num < self.n_buttons:
                self.buttons[num] = bool(val)

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    def reconnect(self) -> bool:
        """Re-open after disconnect.  Returns ``True`` if reconnected."""
        return self._open()

    # ------------------------------------------------------------------
    # polling
    # ------------------------------------------------------------------
    def poll(self) -> None:
        """Read all pending kernel events into the axes/buttons arrays."""
        if self._file is None:
            return
        try:
            while True:
                data = self._file.read(JS_EVENT_SIZE)
                if data is None or len(data) < JS_EVENT_SIZE:
                    break
                _t, val, etype, num = struct.unpack(_JS_EVENT_FMT, data)
                if etype & JS_EVENT_INIT:
                    continue
                self._event_count += 1
                self._last_event_time = time.monotonic()
                if etype & JS_EVENT_AXIS and num < self.n_axes:
                    self.axes[num] = val / _AXIS_MAX
                elif etype & JS_EVENT_BUTTON and num < self.n_buttons:
                    self.buttons[num] = bool(val)
        except OSError:
            self.close()

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    def get_axis(self, idx: int) -> float:
        """Return axis value in ``[-1, 1]`` (0 if index out of range)."""
        if 0 <= idx < self.n_axes:
            return self.axes[idx]
        return 0.0

    def get_button(self, idx: int) -> bool:
        """Return ``True`` if button *idx* is currently pressed."""
        if 0 <= idx < self.n_buttons:
            return self.buttons[idx]
        return False

    @property
    def connected(self) -> bool:
        """``True`` if the device file descriptor is open."""
        return self._file is not None

    @property
    def is_idle(self) -> bool:
        """``True`` if no real-time events have arrived since open.

        Indicates a BLE gamepad in low-power idle that needs physical
        operation (stick/button press) to wake its HID report stream.
        """
        return self._event_count == 0

    @property
    def idle_seconds(self) -> float:
        """Seconds since the last real-time event (or since open)."""
        return time.monotonic() - self._last_event_time
