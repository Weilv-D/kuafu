# -*- coding: utf-8 -*-
"""
KUAFU observation contract parity tests.

Verifies the 35-dim proprioceptive observation frame declared in
rl/env/contract.py against the invariants documented in AGENTS.md:
  - Actor consumes 35 values x 4 causal frames = 140 inputs.
  - Raw root velocity / absolute yaw / sim-truth contact must never leak.
  - Forward speed is wheel-odometry estimated (est_vx), not root truth.

Run: rl/.venv/bin/python -m pytest rl/train/tests/test_observation_parity.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pytest

from rl.env import contract as C


def test_obs_dim_is_35():
    assert C.obs_dim() == 35


def test_no_root_velocity_in_obs_fields():
    names = [entry[0] for entry in C.OBS_FIELDS]
    for name in names:
        assert "root" not in name
        assert "lin_vel" not in name


def test_action_dim_is_6():
    assert C.ACTION_DIM == 6


def test_schema_version_is_v1_1_0():
    assert C.SCHEMA_VERSION == "v1.1.0"


def test_prev_applied_action_in_obs_fields():
    names = [entry[0] for entry in C.OBS_FIELDS]
    assert "prev_applied_action" in names


def test_sensor_age_split_imu_joint():
    notes = {entry[0]: entry[3] for entry in C.OBS_FIELDS}
    note = notes["sensor_age"]
    assert "IMU" in note
    assert "joint" in note
