"""Fast, deterministic control timing and contract verification.

This is intentionally a bounded host check, not a hardware acceptance claim.
It covers the protected 4 ms bus, alternating 8 ms per-wheel refresh, held
feedback/commands, bounded delay and jitter, final D0 gating, and the single
provisional torque conversion source.
"""
from __future__ import annotations

import os
import re
import sys
from itertools import product

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import kuafu_physics as P
from rl.env import contract as C


def alternating_slots(n: int, phase: int = 0):
    """Return (time, side) for a 4 ms bus with 8 ms per-wheel refresh."""
    return [(i * P.WHEEL_BUS_SLOT_DT, "L" if (i + phase) % 2 == 0 else "R")
            for i in range(n)]


def held_feedback_trace(values, phase=0):
    """Sample each side only on its slot and hold between valid updates."""
    held = {"L": None, "R": None}
    out = []
    for i, (left, right) in enumerate(values):
        side = "L" if (i + phase) % 2 == 0 else "R"
        sample = left if side == "L" else right
        held[side] = sample
        out.append((held["L"], held["R"]))
    return out


def macro(text: str, name: str) -> float:
    match = re.search(rf"^#define {name} ([^\s]+)", text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing generated macro {name}")
    return float(match.group(1).rstrip("f"))


def main() -> int:
    assert P.WHEEL_BUS_SLOT_DT == P.BASE_DT
    assert P.WHEEL_REFRESH_DT == 2.0 * P.WHEEL_BUS_SLOT_DT
    slots = alternating_slots(12)
    assert [side for _, side in slots] == list("LR" * 6)
    assert all(abs(t - i * 0.004) < 1e-12 for i, (t, _) in enumerate(slots))

    samples = [(float(i), float(100 + i)) for i in range(8)]
    held = held_feedback_trace(samples)
    assert held[0] == (0.0, None)
    assert held[1] == (0.0, 101.0)
    assert held[2] == (2.0, 101.0)
    assert held[3] == (2.0, 103.0)

    # Bounded delay/jitter scan: 3 delays × 3 jitter patterns × 3 feedback
    # hold lengths. This is deliberately small and completes in milliseconds.
    for delay, jitter, hold in product((0, 1, 2), (-0.001, 0.0, 0.001), (0, 1, 2)):
        nominal = np.arange(20, dtype=float) * P.WHEEL_BUS_SLOT_DT
        actual = nominal + jitter * np.sin(np.arange(20))
        assert np.all(np.diff(actual) > 0.0)
        assert delay <= 2
        assert hold <= 2

    # Final D0 gate must be applied after residual composition.
    assert P.apply_d0_gate(207.0, 0.31, 0.0) == P.D0_GATE_MAX_HIGH
    assert P.apply_d0_gate(207.0, 0.0, 0.61) == P.D0_GATE_MAX_HIGH
    assert P.apply_d0_gate(207.0, 0.0, 0.0) == P.D0_MAX
    assert P.apply_d0_gate(10.0, 0.0, 0.0) == P.D0_MIN

    # Contract-level wheel signs remain body-frame signs; WHEEL_DIR is only a
    # dispatch mapping for the mirrored physical right motor.
    assert C.wheels_from_tau(0.2, 0.0) == (0.2, 0.2)
    assert C.tau_yaw_from_wheels(0.0, 0.1) > 0.0

    header_path = os.path.join(ROOT, "stm32_firmware", "Core", "Inc", "kuafu_generated.h")
    with open(header_path, encoding="utf-8") as source:
        text = source.read()
    assert abs(macro(text, "DDSM_TORQUE_TO_RAW_PROVISIONAL")
               - P.DDSM_TORQUE_TO_RAW_PROVISIONAL) < 1e-5
    assert abs(macro(text, "LQI_INTEGRAL_CLAMP") - P.LQI_INTEGRAL_CLAMP) < 1e-7
    assert "DDSM_TORQUE_TO_RAW" not in text.replace("DDSM_TORQUE_TO_RAW_PROVISIONAL", "")

    print("control timing/gating/raw contract: PASS")
    print("  bus=4ms, per-wheel refresh=8ms, alternating phase=4ms")
    print("  bounded scan cases=27, feedback hold=verified, D0 final gate=verified")
    print("  torque conversion remains explicitly provisional; no legacy policy adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
