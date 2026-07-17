# -*- coding: utf-8 -*-
"""Pure input-shaping functions shared by gamepad/keyboard sources.

These are kept side-effect-free and dependency-free so they can be unit-tested
without pygame, and so both input sources apply identical deadzone/curve logic.
"""
from __future__ import annotations


def apply_deadzone(x: float, deadzone: float = 0.08) -> float:
    """Hard deadzone with linear re-map to [-1, 1].

    ``|x| < deadzone`` maps to ``0``; outside the deadzone the value is shifted
    so the deadzone edge maps to ``0`` and full deflection still maps to ``±1``.
    """
    if abs(x) < deadzone:
        return 0.0
    sign = 1.0 if x > 0.0 else -1.0
    return sign * (abs(x) - deadzone) / (1.0 - deadzone)


def apply_curve(x: float, gamma: float = 2.0) -> float:
    """Power response curve ``sign(x) * |x|**gamma``.

    Preserves sign, keeps ``0`` at ``0``, and keeps ``±1`` at ``±1`` so full
    deflection still reaches the command limit. ``gamma=1`` is linear.
    """
    if x == 0.0:
        return 0.0
    sign = 1.0 if x > 0.0 else -1.0
    return sign * (abs(x) ** gamma)


def shape_axis(raw: float, deadzone: float = 0.08, gamma: float = 2.0) -> float:
    """Canonical stick pipeline: deadzone then curve."""
    return apply_curve(apply_deadzone(raw, deadzone), gamma)


def normalize_trigger(raw: float, deadzone: float = 0.10) -> float:
    """Map a pygame trigger in ``[-1, 1]`` to ``[0, 1]`` with a soft deadzone.

    Resting the finger on a trigger previously caused ``d0`` to drift because
    the raw value sat slightly above ``-1``. The deadzone zeroes small pulls and
    re-maps the rest so a full pull still reaches ``1.0``.
    """
    val = (raw + 1.0) * 0.5            # [-1, 1] -> [0, 1]
    if val < deadzone:
        return 0.0
    return (val - deadzone) / (1.0 - deadzone)
