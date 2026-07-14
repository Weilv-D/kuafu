"""Cross-layer controller constant and direction golden-vector checks."""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np

import kuafu_physics as P
from rl.env.contract import wheels_from_tau


def _macro(text: str, name: str) -> float:
    match = re.search(rf"^#define {name} ([^\s]+)", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"missing generated macro {name}")
    return float(match.group(1).rstrip("f"))


def main() -> int:
    header = os.path.join(ROOT, "stm32_firmware", "Core", "Inc", "kuafu_generated.h")
    with open(header, encoding="utf-8") as source:
        text = source.read()
    generated = np.asarray([_macro(text, f"KUAFU_LQR_K{i}") for i in range(4)])
    if not np.allclose(generated, P.LQR_K_DT4, atol=1e-7):
        raise RuntimeError("generated C LQR gains diverge from Python source")
    if not np.isclose(_macro(text, "KUAFU_LQI_KI"), P.LQI_KI_DT4, atol=1e-7):
        raise RuntimeError("generated C LQI gain diverges from Python source")

    # Fixed state vector is used in Python/JAX/C integration tests as a direction
    # witness. Positive common torque maps to both positive wheel torques; positive
    # yaw maps to right greater than left.
    state = np.asarray([0.02, 0.03, -0.10, 0.20])
    force = float(-(P.LQR_K_DT4 @ state))
    left, right = wheels_from_tau(force * P.R * 0.5, 0.10)
    if not (right > left):
        raise RuntimeError("positive yaw golden vector does not produce right > left")
    print(f"controller golden: F={force:+.6f}N, tauL={left:+.6f}, tauR={right:+.6f}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
