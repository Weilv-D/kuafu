# -*- coding: utf-8 -*-
"""Unit tests for the pure input-shaping functions."""
from __future__ import annotations

import math

import pytest

from rl.teleop.shaping import (
    apply_curve, apply_deadzone, normalize_trigger, shape_axis,
)


class TestApplyDeadzone:
    def test_below_deadzone_is_zero(self):
        assert apply_deadzone(0.05, 0.08) == 0.0
        assert apply_deadzone(-0.079, 0.08) == 0.0

    def test_at_deadzone_edge_maps_to_zero(self):
        # Just outside the deadzone the re-mapped output is near zero.
        assert apply_deadzone(0.081, 0.08) == pytest.approx(0.001 / 0.92, abs=1e-6)

    def test_full_deflection_reaches_one(self):
        assert apply_deadzone(1.0, 0.08) == pytest.approx(1.0)
        assert apply_deadzone(-1.0, 0.08) == pytest.approx(-1.0)

    def test_sign_preserved(self):
        assert apply_deadzone(0.5, 0.08) > 0.0
        assert apply_deadzone(-0.5, 0.08) < 0.0

    def test_symmetric(self):
        assert apply_deadzone(0.3, 0.08) == pytest.approx(-apply_deadzone(-0.3, 0.08))

    def test_zero_deadzone_is_identity(self):
        for x in (-1.0, -0.3, 0.0, 0.3, 1.0):
            assert apply_deadzone(x, 0.0) == pytest.approx(x)


class TestApplyCurve:
    def test_zero_stays_zero(self):
        assert apply_curve(0.0, 2.0) == 0.0

    def test_sign_preserved(self):
        assert apply_curve(0.5, 2.0) > 0.0
        assert apply_curve(-0.5, 2.0) < 0.0

    def test_unit_magnitude_preserved(self):
        assert apply_curve(1.0, 2.0) == pytest.approx(1.0)
        assert apply_curve(-1.0, 2.0) == pytest.approx(-1.0)

    def test_gamma_one_is_linear(self):
        for x in (-0.9, -0.2, 0.1, 0.6):
            assert apply_curve(x, 1.0) == pytest.approx(x)

    def test_gamma_two_is_squared(self):
        assert apply_curve(0.5, 2.0) == pytest.approx(0.25)
        assert apply_curve(-0.4, 2.0) == pytest.approx(-0.16)

    def test_low_deflection_attenuated(self):
        # Squared curve makes small stick movements produce much smaller output.
        assert abs(apply_curve(0.2, 2.0)) < 0.2


class TestShapeAxis:
    def test_deadzone_inside_curve(self):
        # A value within the deadzone is zeroed before the curve sees it.
        assert shape_axis(0.05, 0.08, 2.0) == 0.0

    def test_pipeline_matches_manual(self):
        raw = 0.5
        manual = apply_curve(apply_deadzone(raw, 0.08), 2.0)
        assert shape_axis(raw, 0.08, 2.0) == pytest.approx(manual)

    def test_known_value(self):
        # 0.2 -> deadzone -> (0.2-0.08)/0.92 = 0.13043 -> squared = 0.01702
        assert shape_axis(0.2, 0.08, 2.0) == pytest.approx(
            ((0.2 - 0.08) / 0.92) ** 2, abs=1e-6
        )


class TestNormalizeTrigger:
    def test_rest_position_is_dead(self):
        # Rest = raw -1 (no pull). Even with a noisy -0.95 it is below deadzone.
        assert normalize_trigger(-1.0, 0.10) == 0.0
        assert normalize_trigger(-0.95, 0.10) == pytest.approx(
            max(0.0, (0.025 - 0.10) / 0.90), abs=1e-6
        ) or normalize_trigger(-0.95, 0.10) == 0.0

    def test_full_pull_reaches_one(self):
        assert normalize_trigger(1.0, 0.10) == pytest.approx(1.0)

    def test_small_pull_below_deadzone_is_zero(self):
        # raw -0.85 -> val 0.075 < 0.10 deadzone
        assert normalize_trigger(-0.85, 0.10) == 0.0

    def test_partial_pull_remap(self):
        # raw 0.0 -> val 0.5, past deadzone -> (0.5-0.1)/0.9
        assert normalize_trigger(0.0, 0.10) == pytest.approx(0.4 / 0.9, abs=1e-6)

    def test_monotonic(self):
        prev = -1.0
        for raw in (-0.8, -0.4, 0.0, 0.4, 0.8, 1.0):
            cur = normalize_trigger(raw, 0.10)
            assert cur >= prev - 1e-9
            prev = cur
