"""Versioned Pi5 <-> STM32 protocol shared by runtime tests and firmware docs."""

from __future__ import annotations

from dataclasses import dataclass
import struct
import math
from typing import Iterable

from rl.env.contract import ACTION_DIM, ProtocolFrameSpec

HEADER = 0xA5
FOOTER = 0x5A
VERSION = ProtocolFrameSpec.version
CMD_HEARTBEAT = 0x01
CMD_ACTION = 0x02
CMD_HELLO = 0x03
TEL_IMU = 0x81
TEL_JOINTS = 0x82
TEL_DIAG = 0x83
TEL_HEALTH = 0x84
TEL_FAULT = 0x8F


def crc8_maxim(data: bytes) -> int:
    crc = 0
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8C if crc & 1 else crc >> 1
    return crc & 0xFF


@dataclass(frozen=True)
class Frame:
    type: int
    sequence: int
    timestamp_ms: int
    payload: bytes

    def encode(self) -> bytes:
        if not 0 <= self.sequence <= 0xFFFF:
            raise ValueError("sequence outside uint16")
        if len(self.payload) > 64:
            raise ValueError("payload exceeds protocol maximum")
        prefix = struct.pack(
            ">BBBBHI", HEADER, VERSION, self.type, len(self.payload), self.sequence, self.timestamp_ms & 0xFFFFFFFF
        )
        checksum = crc8_maxim(prefix[1:] + self.payload)
        return prefix + self.payload + bytes((checksum, FOOTER))


@dataclass(frozen=True)
class FirmwareHealth:
    fault_mask: int
    mode: int
    reset_cause: int
    imu_age_ms: int
    wheel_age_ms: tuple[int, int]
    servo_age_ms: tuple[int, int, int, int]
    imu_errors: int
    wheel_errors: tuple[int, int]
    servo_errors: tuple[int, int, int, int]
    wheel_error_breakdown: tuple  # ((L_timeout,L_checksum,L_protocol),(R_timeout,R_checksum,R_protocol))


def decode_health_payload(payload: bytes) -> FirmwareHealth:
    if len(payload) != 46:
        raise ValueError("health telemetry payload must be 46 bytes")
    values = struct.unpack(">IBB20H", payload)
    return FirmwareHealth(
        fault_mask=values[0], mode=values[1], reset_cause=values[2],
        imu_age_ms=values[3], wheel_age_ms=(values[4], values[5]),
        servo_age_ms=tuple(values[6:10]), imu_errors=values[10],
        wheel_errors=(values[11], values[12]),
        servo_errors=tuple(values[13:17]),
        wheel_error_breakdown=(
            (values[17], values[18], values[19]),  # L: timeout, checksum, protocol
            (values[20], values[21], values[22]),  # R: timeout, checksum, protocol
        ),
    )


class StreamDecoder:
    """Loss-tolerant incremental decoder matching ``pi_link_parse_packet``."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._last_sequence: int | None = None

    def feed(self, chunk: bytes) -> list[Frame]:
        self._buffer.extend(chunk)
        frames: list[Frame] = []
        while len(self._buffer) >= 12:
            if self._buffer[0] != HEADER:
                del self._buffer[0]
                continue
            version, msg_type, length = self._buffer[1:4]
            total = 12 + length
            if version != VERSION or length > 64:
                del self._buffer[0]
                continue
            if len(self._buffer) < total:
                break
            raw = bytes(self._buffer[:total])
            if raw[-1] != FOOTER or crc8_maxim(raw[1:-2]) != raw[-2]:
                # Match STM32 resynchronization: retain bytes after the header
                # candidate so a valid frame embedded after corruption survives.
                del self._buffer[0]
                continue
            del self._buffer[:total]
            sequence = struct.unpack(">H", raw[4:6])[0]
            if (msg_type != CMD_HELLO and self._last_sequence is not None and
                    not (0 < ((sequence - self._last_sequence) & 0xFFFF) < 0x8000)):
                continue
            self._last_sequence = sequence
            timestamp_ms = struct.unpack(">I", raw[6:10])[0]
            frames.append(Frame(msg_type, sequence, timestamp_ms, raw[10:-2]))
        return frames


def _i16(value: float, scale: float) -> int:
    if not math.isfinite(value):
        raise ValueError("non-finite protocol value")
    raw = round(value * scale)
    if not -32768 <= raw <= 32767:
        raise ValueError(f"scaled value overflows int16: {value} * {scale}")
    return raw


def command_frames(sequence: int, timestamp_ms: int, mode: int, vx: float, wz: float,
                   d0_mm: float, action: Iterable[float]) -> tuple[Frame, Frame]:
    if mode not in (0, 1, 2, 3, 4):
        raise ValueError(f"invalid robot mode: {mode}")
    if not -0.5 <= vx <= 0.5:
        raise ValueError("vx outside [-0.5, 0.5] m/s")
    if not -1.0 <= wz <= 1.0:
        raise ValueError("wz outside [-1.0, 1.0] rad/s")
    d0_limit = 120.0 if abs(vx) > 0.3 or abs(wz) > 0.6 else 207.0
    if not 58.0 <= d0_mm <= d0_limit:
        raise ValueError(f"D0 outside gated range [58, {d0_limit}] mm")
    values = tuple(action)
    if len(values) != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} actions, got {len(values)}")
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values):
        raise ValueError("action outside finite [-1, 1] contract")
    heartbeat = bytes((mode,)) + struct.pack(
        ">hhh", _i16(vx, 1000), _i16(wz, 1000), _i16(d0_mm, 1)
    )
    actions = struct.pack(">" + "h" * ACTION_DIM, *(_i16(value, 10000) for value in values))
    return (
        Frame(CMD_HEARTBEAT, sequence & 0xFFFF, timestamp_ms, heartbeat),
        Frame(CMD_ACTION, (sequence + 1) & 0xFFFF, timestamp_ms, actions),
    )


def hello_frame(sequence: int, timestamp_ms: int, model_hash: str) -> Frame:
    if len(model_hash) != 16 or any(character not in "0123456789abcdef" for character in model_hash):
        raise ValueError("model hash must be a 16-character lowercase hex string")
    return Frame(CMD_HELLO, sequence & 0xFFFF, timestamp_ms, model_hash.encode("ascii"))
