"""Generate and validate the deployment five-bar table from canonical geometry.

The generated angles are dwell-relative actuator commands, not the absolute circle
angles returned by the geometry solver.  The generator rejects branch jumps,
unreachable endpoints, sign violations, and joint-limit violations before writing an
artifact.  Mechanical servo zero offsets remain a P9 bench-calibration input and are
applied by firmware, never folded into this model-level table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import kuafu_physics as P
from rl.env.contract import SCHEMA_VERSION


def build_table(points: int) -> dict:
    table = P.fivebar_ik_table(points)
    grid = P.fivebar_ik_grid(n_d0=points, qx_limit_mm=P.QX_RESIDUAL_SCALE, n_qx=33)
    d0 = table["d0"]
    qA = table["qA"]
    qB = table["qB"]

    if not (np.isclose(d0[0], P.D0_MIN) and np.isclose(d0[-1], P.D0_MAX)):
        raise RuntimeError("five-bar table does not include both D0 endpoints")
    if not np.all(np.diff(d0) > 0.0):
        raise RuntimeError("D0 grid is not strictly ascending")
    if not np.all(np.diff(qA) < 0.0):
        raise RuntimeError("five-bar guard failed: dqA/dD0 must be negative")
    if not np.all(np.diff(qB) > 0.0):
        raise RuntimeError("five-bar guard failed: dqB/dD0 must be positive")
    if max(np.max(np.abs(qA)), np.max(np.abs(qB))) > 3.3:
        raise RuntimeError("five-bar command exceeds the XML/servo joint limit")

    max_fk_error_mm = 0.0
    for depth in d0:
        raw = P.fivebar_ik(float(depth))
        if raw is None:
            raise RuntimeError(f"unreachable D0={depth:.3f}mm")
        q = P.fivebar_fk(raw[0], raw[1])
        max_fk_error_mm = max(max_fk_error_mm, abs(q[0]), abs(q[1] + depth))
    if max_fk_error_mm > 1e-6:
        raise RuntimeError(f"FK/IK closure error {max_fk_error_mm:.6f}mm")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_hash": P.model_hash(),
        "source": "kuafu_physics.fivebar_ik_table",
        "d0_unit": "mm",
        "joint_unit": "rad",
        "d0": d0.tolist(),
        "qA": qA.tolist(),
        "qB": qB.tolist(),
        "qx": grid["qx"].tolist(),
        "qA_grid": grid["qA"].tolist(),
        "qB_grid": grid["qB"].tolist(),
        "qx_limit_mm": float(P.QX_RESIDUAL_SCALE),
        "d0_min": float(P.D0_MIN),
        "d0_max": float(P.D0_MAX),
        "max_fk_error_mm": max_fk_error_mm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate KUAFU five-bar deployment table")
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "rl", "env", "fivebar_ik_table.json"),
        help="output JSON path",
    )
    args = parser.parse_args()
    if args.points < 3:
        raise SystemExit("--points must be at least 3")

    table = build_table(args.points)
    with open(args.out, "w", encoding="utf-8") as output:
        json.dump(table, output, indent=2)
        output.write("\n")

    qA = np.asarray(table["qA"])
    qB = np.asarray(table["qB"])
    print(f"wrote {args.out}")
    print(f"D0={table['d0'][0]:.1f}..{table['d0'][-1]:.1f} mm, {args.points} points")
    print(f"qA={qA[0]:+.4f}..{qA[-1]:+.4f}, qB={qB[0]:+.4f}..{qB[-1]:+.4f} rad")
    print("guards: dqA/dD0<0, dqB/dD0>0, FK/IK closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
