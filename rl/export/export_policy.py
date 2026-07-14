# -*- coding: utf-8 -*-
"""Export the schema-compatible KUAFU Actor to ONNX plus a deployment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

import kuafu_physics as P
from rl.env.contract import ACTION_NAMES, SCHEMA_VERSION


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_onnx(onnx_path: str, inputs, expected, action_dim: int) -> None:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual = session.run(None, inputs)[0]
    if actual.shape != expected.shape or actual.shape[-1] != action_dim:
        raise RuntimeError(f"ONNX action shape mismatch: {actual.shape}, expected {expected.shape}")
    if not np.isfinite(actual).all() or np.max(np.abs(actual)) > 1.000001:
        raise RuntimeError("ONNX action contains NaN/Inf or exceeds tanh bounds")
    max_error = float(np.max(np.abs(actual - expected)))
    if max_error >= 1e-5:
        raise RuntimeError(f"Torch/ONNX parity failed: max error {max_error:.3e}")
    print(f"ONNX parity: max_abs_error={max_error:.3e}, action range=[{actual.min():.3f}, {actual.max():.3f}]")


def export_actor(ckpt_path: str, out_path: str) -> None:
    import numpy as np
    import torch

    from rl.env.kuafu_mjx_env import ACTION_DIM, ACTOR_OBS_DIM
    from rl.train.teacher_model import TeacherInferenceModel

    model = TeacherInferenceModel.from_checkpoint(ckpt_path, obs_dim=ACTOR_OBS_DIM, action_dim=ACTION_DIM)
    torch.manual_seed(0)
    dummy_obs = torch.randn(3, ACTOR_OBS_DIM)
    with torch.inference_mode():
        expected = model(dummy_obs).numpy()
    torch.onnx.export(
        model,
        dummy_obs,
        out_path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    _verify_onnx(out_path, {"obs": dummy_obs.numpy()}, expected, ACTION_DIM)

    table_path = os.path.join(PROJ_ROOT, "rl", "env", "fivebar_ik_table.json")
    if not os.path.exists(table_path):
        raise RuntimeError("five-bar calibration table is required for deployment export")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_hash": P.model_hash(),
        "calibration_table_sha256": _sha256(table_path),
        "checkpoint": os.path.abspath(ckpt_path),
        "onnx": os.path.basename(out_path),
        "onnx_sha256": _sha256(out_path),
        "input": {"name": "obs", "shape": ["batch", ACTOR_OBS_DIM]},
        "output": {"name": "action", "shape": ["batch", ACTION_DIM], "names": list(ACTION_NAMES)},
        "normalization": "fixed physical scales in rl.env.contract / Pi5 runtime",
        "transform": "tanh(actor(obs))",
    }
    manifest_path = f"{out_path}.manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(f"exported {out_path}")
    print(f"wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export KUAFU deployable Actor")
    parser.add_argument("--ckpt", required=True, help="schema-compatible PPO checkpoint")
    parser.add_argument("--out", default="policy.onnx", help="output ONNX path")
    args = parser.parse_args()
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    export_actor(args.ckpt, args.out)


if __name__ == "__main__":
    main()
