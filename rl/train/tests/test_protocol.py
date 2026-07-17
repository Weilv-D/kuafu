import struct

import numpy as np
import pytest

from pi5_runtime.protocol import (
    CMD_ACTION, CMD_HELLO, StreamDecoder, command_frames,
    decode_health_payload, hello_frame,
)
from pi5_runtime.serial_node import decode_joint_payload, projected_gravity_from_euler
import kuafu_physics as P
from rl.verify.release_gate import wilson_lower_bound


def test_versioned_frames_survive_fragmentation_and_replay():
    heartbeat, action = command_frames(10, 1234, 2, 0.5, -1.0, 120.0, [0.1] * 6)
    decoder = StreamDecoder()
    raw = heartbeat.encode() + action.encode()
    frames = decoder.feed(raw[:5])
    frames += decoder.feed(raw[5:19])
    frames += decoder.feed(raw[19:])
    assert [frame.type for frame in frames] == [heartbeat.type, CMD_ACTION]
    assert decoder.feed(action.encode()) == []


def test_crc_corruption_resynchronizes_to_following_frame():
    heartbeat, action = command_frames(20, 1234, 2, 0.0, 0.0, 58.0, [0.0] * 6)
    corrupted = bytearray(heartbeat.encode())
    corrupted[-2] ^= 0x01
    frames = StreamDecoder().feed(bytes(corrupted) + action.encode())
    assert [frame.type for frame in frames] == [CMD_ACTION]


def test_action_payload_and_full_speed_range_are_representable():
    _heartbeat, action = command_frames(0, 0, 2, 0.5, 1.0, 120.0, [-1.0, 1.0, 0, 0, 0, 0])
    values = struct.unpack(">6h", action.payload)
    assert values[:2] == (-10000, 10000)


def test_protocol_rejects_int16_overflow():
    with pytest.raises(ValueError, match="outside"):
        command_frames(0, 0, 2, 40.0, 0.0, 58.0, [0.0] * 6)


def test_protocol_rejects_out_of_contract_action_and_hello_round_trip():
    with pytest.raises(ValueError, match="action"):
        command_frames(0, 0, 2, 0.0, 0.0, 58.0, [1.01] * 6)
    hello = hello_frame(0, 0, P.model_hash())
    decoded = StreamDecoder().feed(hello.encode())
    assert decoded[0].type == CMD_HELLO
    assert decoded[0].payload.decode("ascii") == P.model_hash()


def test_hello_starts_a_new_sequence_session():
    decoder = StreamDecoder()
    assert decoder.feed(hello_frame(500, 0, P.model_hash()).encode())
    assert decoder.feed(hello_frame(1, 1, P.model_hash()).encode())


def test_relative_fivebar_fk_matches_dwell_and_extension():
    dwell = P.fivebar_fk_relative(0.0, 0.0)
    extended = P.fivebar_fk_relative(*P.fivebar_ik_cmd(P.D0_MAX))
    assert abs(dwell[0]) < 1e-6 and abs(dwell[1] + P.D0_MIN) < 1e-6
    assert abs(extended[0]) < 1e-3 and abs(extended[1] + P.D0_MAX) < 1e-3


def test_pi_gravity_matches_shared_quaternion_convention():
    roll, pitch, yaw = 0.12, -0.18, 0.31
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
    gravity = np.array([0.0, 0.0, -1.0])
    xyz = -q[1:]
    expected = gravity + 2.0 * (q[0] * np.cross(xyz, gravity) +
                                np.cross(xyz, np.cross(xyz, gravity)))
    assert np.allclose(projected_gravity_from_euler(roll, pitch, yaw), expected, atol=1e-6)


def test_joint_telemetry_scales_wheel_speed_once():
    raw = [1000, 12000, 10000, -1000, -12000, -10000] + [0] * 12
    payload = struct.pack(">18h", *raw)
    values = decode_joint_payload(payload)
    assert values[1] == pytest.approx(12.0)
    assert values[2] == pytest.approx(1.0)
    assert values[4] == pytest.approx(-12.0)
    assert values[5] == pytest.approx(-1.0)


def test_health_telemetry_is_big_endian_and_complete():
    # Layout: fault_mask:u32 | mode:u8 | reset:u8 | imu_age | wheel_age[2] |
    # servo_age[4] | imu_errors | wheel_errors[2] | servo_errors[4] |
    # wheel_l_{timeout,checksum,protocol} | wheel_r_{timeout,checksum,protocol}
    # = IBB + 20 H = 46 bytes.
    payload = struct.pack(">IBB20H", 0x12345678, 3, 0x21,
                          1, 2, 3, 4, 5, 6, 7,            # ages (imu, wheel×2, servo×4)
                          8, 9, 10, 11, 12, 13, 14,        # errors (imu, wheel×2, servo×4)
                          15, 16, 17, 18, 19, 20)          # wheel breakdown (L×3, R×3)
    health = decode_health_payload(payload)
    assert health.fault_mask == 0x12345678
    assert health.mode == 3 and health.reset_cause == 0x21
    assert health.imu_age_ms == 1
    assert health.wheel_age_ms == (2, 3)
    assert health.servo_age_ms == (4, 5, 6, 7)
    assert health.imu_errors == 8
    assert health.wheel_errors == (9, 10)
    assert health.servo_errors == (11, 12, 13, 14)
    assert health.wheel_error_breakdown == ((15, 16, 17), (18, 19, 20))
    with pytest.raises(ValueError, match="46 bytes"):
        decode_health_payload(payload[:-1])


def test_wilson_gate_is_conservative_for_small_samples():
    assert wilson_lower_bound(9, 10) < 0.80
    assert wilson_lower_bound(96, 100) > 0.80
