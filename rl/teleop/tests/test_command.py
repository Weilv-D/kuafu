# -*- coding: utf-8 -*-
"""Tests for the Command contract and ArbiterConfig defaults."""
from __future__ import annotations

import pytest

from rl.teleop.command import (
    ArbiterConfig, Command, D0_CMD_RANGE, Mode, V_CMD_RANGE, W_CMD_RANGE,
)


class TestMode:
    def test_idle_exists(self):
        assert Mode.IDLE == Mode.IDLE

    def test_all_modes_distinct(self):
        modes = {Mode.MANUAL, Mode.IDLE, Mode.AUTONOMOUS, Mode.ASSISTED, Mode.ESTOP}
        assert len(modes) == 5


class TestCommandRanges:
    def test_ranges_match_physics(self):
        import kuafu_physics as P
        assert D0_CMD_RANGE == (P.D0_MIN, P.D0_MAX)
        assert V_CMD_RANGE == (-0.5, 0.5)
        assert W_CMD_RANGE == (-1.0, 1.0)


class TestArbiterConfigDefaults:
    def test_input_shaping_defaults(self):
        cfg = ArbiterConfig()
        assert cfg.stick_deadzone == 0.08
        assert cfg.stick_gamma == 2.0
        assert cfg.trigger_deadzone == 0.10
        assert cfg.d0_rate_mm_s == 40.0

    def test_safety_defaults(self):
        cfg = ArbiterConfig()
        assert cfg.max_smoothing_dt == 0.10
        assert cfg.ramp_time == pytest.approx(0.30)
        assert cfg.stale_time == pytest.approx(0.50)
        assert cfg.handoff_time == pytest.approx(1.5)

    def test_backward_compatible_construction(self):
        # No-arg construction must still work (old call sites).
        ArbiterConfig()


class TestCommandImmutability:
    def test_as_array(self):
        c = Command(0.1, 0.2, 100.0, Mode.MANUAL)
        assert c.as_array() == [0.1, 0.2, 100.0]

    def test_stamp_auto_set(self):
        c = Command(0.0, 0.0, 58.0, Mode.IDLE)
        assert c.stamp > 0.0
