"""50 Hz ONNX Actor runtime with schema validation and UART freshness separation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import hashlib
import os
import time
from typing import Sequence

import numpy as np

import kuafu_physics as P
from pi5_runtime.protocol import command_frames, hello_frame
from rl.env.contract import ACTION_DIM, ACTION_NAMES, SCHEMA_VERSION, obs_dim


@dataclass
class Telemetry:
    proj_gravity: Sequence[float]
    body_gyro: Sequence[float]
    est_vx: float
    est_wz: float
    est_d0_mm: float
    est_roll: float
    wheel_speed: Sequence[float]
    hip_pos: Sequence[float]
    hip_vel: Sequence[float]
    sensor_age_ms: Sequence[float]


class PolicyRuntime:
    """Runtime-independent policy loop; serial/ROS adapters can call ``tick``."""

    def __init__(self, model_path: str, calibration_table_path: str | None = None) -> None:
        import onnxruntime as ort

        manifest_path = f"{model_path}.manifest.json"
        with open(manifest_path, encoding="utf-8") as source:
            manifest = json.load(source)
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise RuntimeError("ONNX schema does not match firmware/runtime schema")
        if manifest["model_hash"] != P.model_hash():
            raise RuntimeError("ONNX model hash does not match local physical model")
        if manifest.get("transform") != "tanh(actor(obs))":
            raise RuntimeError("ONNX manifest transform is not the supported tanh transform")
        if manifest.get("onnx_sha256") != _sha256(model_path):
            raise RuntimeError("ONNX file digest does not match manifest")
        if calibration_table_path is None:
            calibration_table_path = os.path.join(os.path.dirname(model_path), "fivebar_ik_table.json")
        if not os.path.exists(calibration_table_path):
            raise RuntimeError("calibration table required beside the ONNX artifact")
        if manifest.get("calibration_table_sha256") != _sha256(calibration_table_path):
            raise RuntimeError("calibration table digest does not match manifest")
        expected_dim = obs_dim() * 4
        if manifest["input"]["shape"][-1] != expected_dim:
            raise RuntimeError("ONNX input dimension does not match actor contract")
        if manifest["output"]["shape"][-1] != ACTION_DIM or tuple(manifest["output"].get("names", ())) != ACTION_NAMES:
            raise RuntimeError("ONNX output metadata does not match action contract")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        if self.session.get_inputs()[0].name != "obs" or self.session.get_outputs()[0].name != "action":
            raise RuntimeError("ONNX input/output names do not match runtime contract")
        self.history: deque[np.ndarray] = deque(maxlen=4)
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.sequence = 0

    def hello(self, timestamp_ms: int | None = None) -> bytes:
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        frame = hello_frame(self.sequence, timestamp_ms, P.model_hash()).encode()
        self.sequence = (self.sequence + 1) & 0xFFFF
        return frame

    def reset(self) -> None:
        self.history.clear()
        self.previous_action.fill(0.0)

    def _frame(self, telemetry: Telemetry, vx_cmd: float, wz_cmd: float, d0_cmd_mm: float) -> np.ndarray:
        values = np.concatenate([
            np.asarray([vx_cmd / 0.5, wz_cmd, (d0_cmd_mm - 132.5) / 74.5], np.float32),
            np.asarray(telemetry.proj_gravity, np.float32),
            np.asarray(telemetry.body_gyro, np.float32) / 10.0,
            np.asarray([telemetry.est_vx / 0.5, telemetry.est_wz,
                        (telemetry.est_d0_mm - 132.5) / 74.5, telemetry.est_roll], np.float32),
            np.asarray(telemetry.wheel_speed, np.float32) / 33.0,
            np.asarray(telemetry.hip_pos, np.float32) / 3.3,
            np.asarray(telemetry.hip_vel, np.float32) / P.SERVO_MAX_SPEED,
            self.previous_action,
            np.asarray(telemetry.sensor_age_ms, np.float32) / 100.0,
        ])
        if values.shape != (obs_dim(),):
            raise ValueError(f"telemetry maps to {values.shape}, expected {(obs_dim(),)}")
        if not np.isfinite(values).all():
            raise ValueError("telemetry contains NaN/Inf")
        return values

    def tick(self, telemetry: Telemetry, vx_cmd: float, wz_cmd: float, d0_cmd_mm: float,
             mode: int = 2, timestamp_ms: int | None = None) -> tuple[np.ndarray, bytes, bytes]:
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        frame = self._frame(telemetry, vx_cmd, wz_cmd, d0_cmd_mm)
        while len(self.history) < 4:
            self.history.append(np.zeros_like(frame))
        self.history.append(frame)
        observation = np.concatenate(tuple(self.history))[None, :]
        action = self.session.run(["action"], {"obs": observation})[0][0].astype(np.float32)
        if not np.isfinite(action).all():
            raise RuntimeError("ONNX policy returned NaN/Inf")
        action = np.clip(action, -1.0, 1.0)
        self.previous_action = action
        heartbeat, residual = command_frames(self.sequence, timestamp_ms, mode, vx_cmd, wz_cmd, d0_cmd_mm, action)
        self.sequence = (self.sequence + 2) & 0xFFFF
        return action, heartbeat.encode(), residual.encode()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="KUAFU Pi5 ONNX policy runtime")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    PolicyRuntime(os.path.abspath(args.model))
    print("model manifest validated; use pi5_runtime.serial_node for the UART loop")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
