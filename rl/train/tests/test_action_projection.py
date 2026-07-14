# -*- coding: utf-8 -*-
"""
KUAFU action projection contract tests.

Verifies the 6-dim residual action layout, its tanh bounds, and the
deployment D0 high-speed gate constants that the safe-projection layer
relies on (kuafu_physics single source of truth).

Run: rl/.venv/bin/python -m pytest rl/train/tests/test_action_projection.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pytest

import kuafu_physics as P
from rl.env import contract as C


def test_action_bounds():
    assert len(C.ACTION_BOUNDS) == C.ACTION_DIM
    for lo, hi in C.ACTION_BOUNDS:
        assert (lo, hi) == (-1.0, 1.0)


def test_action_names_order():
    assert C.ACTION_NAMES == ("dtau_common", "dtau_yaw", "dQx_L", "dD0_L", "dQx_R", "dD0_R")


def test_d0_gate_constant_exists():
    assert hasattr(P, "D0_GATE_V_THRESH")
    assert hasattr(P, "D0_GATE_W_THRESH")
    assert hasattr(P, "D0_GATE_MAX_HIGH")
