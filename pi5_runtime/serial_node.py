"""Serial adapter for the Pi5 Actor runtime (ONNX policy mode).

Decodes STM32 IMU/joint frames, runs ONNX inference at 50 Hz, and sends
paired heartbeat/action frames.  For baseline-LQR teleop without an ONNX
model, use ``pi5_runtime.teleop_single`` instead — it combines gamepad
input and serial in a single process to avoid UART contention.
"""

from __future__ import annotations

import argparse
import struct
import time

import numpy as np

import kuafu_physics as P
from pi5_runtime.protocol import (
    StreamDecoder, TEL_HEALTH, TEL_IMU, TEL_JOINTS, decode_health_payload,
)
from pi5_runtime.runtime import PolicyRuntime, Telemetry


def _i16s(payload: bytes) -> tuple[int, ...]:
    return struct.unpack(">" + "h" * (len(payload) // 2), payload)


def projected_gravity_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Match the MJX world-gravity vector in body coordinates."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    q = np.asarray([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)
    q_xyz = -q[1:]
    gravity = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    uv = np.cross(q_xyz, gravity)
    uuv = np.cross(q_xyz, uv)
    return gravity + 2.0 * (q[0] * uv + uuv)


def decode_joint_payload(payload: bytes) -> np.ndarray:
    if len(payload) != 36:
        raise ValueError("joint telemetry payload must contain 18 int16 values")
    raw = np.asarray(_i16s(payload), dtype=np.float32)
    values = raw / 1000.0
    values[[2, 5]] = raw[[2, 5]] / 10000.0
    return values


class SerialPolicyNode:
    def __init__(self, model: str, port: str, baudrate: int = 921600) -> None:
        import serial

        self.serial = serial.Serial(port, baudrate=baudrate, timeout=0)
        print(f"[serial] opened {port} @ {baudrate} baud")
        self.runtime = PolicyRuntime(model)
        self.serial.write(self.runtime.hello())
        print("[serial] ONNX policy loaded")
        self.decoder = StreamDecoder()
        self.command = (0.0, 0.0, P.D0_MIN, 2)
        self.imu = None
        self.joints = None
        self.health = None
        self.last_imu = self.last_joints = 0.0

    def set_command(self, vx: float, wz: float, d0_mm: float, mode: int = 2) -> None:
        if mode != self.command[3]:
            self.runtime.reset()
        d0_upper = P.D0_GATE_MAX_HIGH if abs(vx) > P.D0_GATE_V_THRESH or abs(wz) > P.D0_GATE_W_THRESH else P.D0_MAX
        self.command = (float(np.clip(vx, -0.5, 0.5)), float(np.clip(wz, -1.0, 1.0)),
                        float(np.clip(d0_mm, P.D0_MIN, d0_upper)), mode)

    def poll(self) -> None:
        chunk = self.serial.read(512)
        now = time.monotonic()
        for frame in self.decoder.feed(chunk):
            if frame.type == TEL_IMU and len(frame.payload) == 12:
                self.imu = np.asarray(_i16s(frame.payload), dtype=np.float32) / 1000.0
                self.last_imu = now
            elif frame.type == TEL_JOINTS and len(frame.payload) == 36:
                self.joints = decode_joint_payload(frame.payload)
                self.last_joints = now
            elif frame.type == TEL_HEALTH:
                self.health = decode_health_payload(frame.payload)

    def _telemetry(self) -> Telemetry | None:
        if self.imu is None or self.joints is None:
            return None
        now = time.monotonic()
        imu_age = (now - self.last_imu) * 1000.0
        joint_age = (now - self.last_joints) * 1000.0
        if imu_age > 100.0 or joint_age > 100.0:
            return None
        roll, pitch, yaw, gx, gy, gz = self.imu
        # Match MJX's exact quaternion-conjugate world-gravity rotation rather
        # than using an Euler approximation with a different sign convention.
        gravity = projected_gravity_from_euler(roll, pitch, yaw)
        joint = self.joints
        wheel_speed = np.asarray([joint[1], joint[4]], dtype=np.float32)
        hip_pos = np.asarray([joint[6], joint[12], joint[9], joint[15]], dtype=np.float32)
        hip_vel = np.asarray([joint[7], joint[13], joint[10], joint[16]], dtype=np.float32)
        left = P.fivebar_fk_relative(float(hip_pos[0]), float(hip_pos[1]))
        right = P.fivebar_fk_relative(float(hip_pos[2]), float(hip_pos[3]))
        return Telemetry(
            proj_gravity=gravity,
            body_gyro=(gx, gy, gz),
            est_vx=float(np.mean(wheel_speed) * P.R),
            est_wz=float(gz),
            est_d0_mm=float((-left[1] - right[1]) * 0.5),
            est_roll=float(roll),
            wheel_speed=wheel_speed,
            hip_pos=hip_pos,
            hip_vel=hip_vel,
            sensor_age_ms=(imu_age, imu_age, imu_age, joint_age, joint_age, joint_age),
        )

    def tick(self) -> bool:
        self.poll()
        telemetry = self._telemetry()
        if telemetry is None:
            return False
        vx, wz, d0, mode = self.command
        _action, heartbeat, residual = self.runtime.tick(telemetry, vx, wz, d0, mode)
        self.serial.write(heartbeat + residual)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="KUAFU 50 Hz Pi5 ONNX serial policy node")
    parser.add_argument("--model", required=True,
                        help="path to ONNX model file")
    parser.add_argument("--port", default="/dev/ttyAMA10",
                        help="serial device; on the Pi5 this is the SoC PL011 "
                             "behind the 3-pin JST debug connector")
    parser.add_argument("--baudrate", type=int, default=921600)
    args = parser.parse_args()
    node = SerialPolicyNode(args.model, args.port, args.baudrate)

    period = 0.02
    deadline = time.monotonic()
    tick_count = 0
    telemetry_ok = False
    try:
        while True:
            ok = node.tick()
            if ok and not telemetry_ok:
                print("[serial] first telemetry received → STM32 link active")
                telemetry_ok = True
            tick_count += 1
            if tick_count % 250 == 0:  # ~5s heartbeat log
                print(f"[serial] alive  ticks={tick_count}  "
                      f"telemetry={'OK' if telemetry_ok else 'waiting'}  "
                      f"cmd={node.command}")
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
