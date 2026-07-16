"""Shared command IPC for the teleop -> serial_node link.

The teleop process (``pi5_runtime.teleop_node``) reads a gamepad/keyboard and
sends raw ``Command`` values over a Unix domain socket. The serial_node process
receives them through :class:`rl.teleop.ipc_source.IPCCommandSource` and feeds
the project :class:`~rl.teleop.arbiter.CommandArbiter`, which owns the safety
layer (ramp / limit / estop / timeout). Keeping the safety layer inside
serial_node means a teleop-process crash or a dropped Bluetooth link degrades
to the arbiter's safe default instead of holding the last command.

Wire format: a 4-byte big-endian length prefix followed by UTF-8 JSON::

    {"v": float, "omega": float, "d0": float, "mode": int, "stamp": float}

Only the standard library is used so no new dependency is added. ``mode`` uses
the integer codes of the firmware/protocol RobotMode (0=INIT, 1=STAND,
2=ACTIVE, 3=CLIMB, 4=FAULT). The teleop source only ever sends MANUAL-bearing
fields; ESTOP is signalled by ``mode == 4``.
"""

from __future__ import annotations

import json
import math
import os
import socket
import struct
import time
from typing import Any

COMMAND_SOCKET_PATH = "/tmp/kuafu-cmd.sock"
LENGTH_PREFIX = struct.Struct(">I")  # 4-byte big-endian unsigned length

# Bounds mirror rl/teleop/command.py V_CMD_RANGE / W_CMD_RANGE / D0_CMD_RANGE.
_V_RANGE = (-0.5, 0.5)
_W_RANGE = (-1.0, 1.0)
_D0_RANGE = (58.0, 207.0)
_VALID_MODES = (0, 1, 2, 3, 4)
_MAX_PAYLOAD = 256  # a command frame is well under 100 bytes


def encode_command(v: float, omega: float, d0_mm: float, mode: int,
                   stamp: float | None = None) -> bytes:
    """Encode one command frame as length-prefix + JSON."""
    if stamp is None:
        stamp = time.monotonic()
    payload = json.dumps(
        {"v": float(v), "omega": float(omega), "d0": float(d0_mm),
         "mode": int(mode), "stamp": float(stamp)},
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _MAX_PAYLOAD:
        raise ValueError("command payload exceeds safety limit")
    return LENGTH_PREFIX.pack(len(payload)) + payload


def _validate(obj: Any) -> dict[str, Any]:
    """Strict schema check. Raises ValueError on any violation."""
    if not isinstance(obj, dict):
        raise ValueError("command frame is not an object")
    for key in ("v", "omega", "d0", "mode", "stamp"):
        if key not in obj:
            raise ValueError(f"command frame missing key: {key}")
    v, omega, d0, stamp = obj["v"], obj["omega"], obj["d0"], obj["stamp"]
    mode = obj["mode"]
    for name, value in (("v", v), ("omega", omega), ("d0", d0), ("stamp", stamp)):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"command field {name!r} is not finite: {value!r}")
    if not isinstance(mode, int) or mode not in _VALID_MODES:
        raise ValueError(f"command mode out of range: {mode!r}")
    if not _V_RANGE[0] <= v <= _V_RANGE[1]:
        raise ValueError(f"v out of range: {v}")
    if not _W_RANGE[0] <= omega <= _W_RANGE[1]:
        raise ValueError(f"omega out of range: {omega}")
    if not _D0_RANGE[0] <= d0 <= _D0_RANGE[1]:
        raise ValueError(f"d0 out of range: {d0}")
    return obj


def _recv_exactly(conn: socket.socket, n: int) -> bytes | None:
    """Read exactly ``n`` bytes or return None if the peer closed."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class CommandSocketServer:
    """Listening side, owned by the serial_node process.

    ``recv_command`` drains the receive buffer and returns the most recent valid
    frame. Stale/partial/corrupt frames are discarded so the caller never sees a
    torn read; a closed or never-connected client yields ``None``.
    """

    def __init__(self, path: str = COMMAND_SOCKET_PATH, backlog: int = 1) -> None:
        self.path = path
        self._sock: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._buf = bytearray()
        self._backlog = backlog

    def bind(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.setblocking(False)
        self._sock.bind(self.path)
        self._sock.listen(self._backlog)

    def _accept_pending(self) -> None:
        if self._conn is not None:
            return
        if self._sock is None:
            return
        import select

        try:
            conn, _addr = self._sock.accept()
        except BlockingIOError:
            return
        conn.setblocking(False)
        self._conn = conn
        self._buf.clear()

    def recv_command(self) -> dict[str, Any] | None:
        """Return the newest fully-decoded valid frame, or None."""
        import select

        self._accept_pending()
        if self._conn is None:
            return None
        # Drain everything currently available without blocking.
        while True:
            rlist, _, _ = select.select([self._conn], [], [], 0.0)
            if not rlist:
                break
            try:
                chunk = self._conn.recv(128)
            except BlockingIOError:
                break
            except OSError:
                self._reset_conn()
                return None
            if not chunk:
                # Client closed; drop it so a reconnect can be accepted.
                self._reset_conn()
                return None
            self._buf.extend(chunk)

        # Decode as many length-prefixed frames as are present, keep the last.
        latest: dict[str, Any] | None = None
        while len(self._buf) >= LENGTH_PREFIX.size:
            (length,) = LENGTH_PREFIX.unpack_from(self._buf, 0)
            if length > _MAX_PAYLOAD:
                self._reset_conn()
                return None
            total = LENGTH_PREFIX.size + length
            if len(self._buf) < total:
                break  # wait for the rest on the next call
            payload = bytes(self._buf[LENGTH_PREFIX.size:total])
            del self._buf[:total]
            try:
                latest = _validate(json.loads(payload.decode("utf-8")))
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                continue  # drop a single bad frame, keep going
        return latest

    def close(self) -> None:
        self._reset_conn()
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def _reset_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
        self._buf.clear()


class CommandSocketClient:
    """Sending side, owned by the teleop process.

    ``connect`` blocks until the server appears, retrying once per second, so the
    teleop process can be started before serial_node. ``send_command`` raises on
    a broken link; the caller should reconnect.
    """

    def __init__(self, path: str = COMMAND_SOCKET_PATH) -> None:
        self.path = path
        self._sock: socket.socket | None = None

    def connect(self, retry_interval: float = 1.0) -> None:
        while True:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.path)
                self._sock = sock
                return
            except (FileNotFoundError, ConnectionRefusedError):
                sock.close()
                time.sleep(retry_interval)

    def connected(self) -> bool:
        return self._sock is not None

    def send_command(self, v: float, omega: float, d0_mm: float, mode: int) -> None:
        if self._sock is None:
            raise ConnectionError("client not connected")
        frame = encode_command(v, omega, d0_mm, mode)
        try:
            self._sock.sendall(frame)
        except (BrokenPipeError, ConnectionResetError):
            self._sock = None
            raise

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
