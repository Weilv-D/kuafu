# -*- coding: utf-8 -*-
"""Unit tests for CommandArbiter safety rules (no pygame / no socket)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pytest

from rl.teleop.arbiter import CommandArbiter
from rl.teleop.command import ArbiterConfig, Command, D0_CMD_RANGE, Mode


@dataclass
class FakeSource:
    """A controllable CommandSource that returns a queued Command (or None)."""
    name: str = "fake"
    _cmd: Optional[Command] = None

    def set(self, cmd: Optional[Command]) -> None:
        # Re-stamp so the command is always fresh within stale_time.
        if cmd is not None:
            cmd.stamp = time.monotonic()
        self._cmd = cmd

    def poll(self) -> Optional[Command]:
        return self._cmd


def _arbiter(*sources, **cfg_kwargs) -> CommandArbiter:
    cfg = ArbiterConfig(**cfg_kwargs) if cfg_kwargs else ArbiterConfig()
    # Use a tiny ramp_time so smoothing is near-instant in tests; tests that
    # care about smoothing pass their own ramp_time.
    return CommandArbiter(list(sources), cfg)


def _settle(arb: CommandArbiter, n: int = 5) -> Command:
    """Poll a few times so the one-pole smoother converges to the target."""
    out = Command(0.0, 0.0, D0_CMD_RANGE[0], Mode.ESTOP)
    for _ in range(n):
        out = arb.poll()
    return out


class TestEstopPriority:
    def test_estop_from_any_source_latches(self):
        src_a = FakeSource("a")
        src_b = FakeSource("b")
        arb = _arbiter(src_a, src_b)
        src_a.set(Command(0.2, 0.0, 100.0, Mode.MANUAL))
        src_b.set(Command(0.0, 0.0, 100.0, Mode.ESTOP))
        out = _settle(arb)
        assert out.mode == Mode.ESTOP
        assert arb.estop_locked

    def test_estop_zeros_velocity(self):
        src = FakeSource()
        arb = _arbiter(src, ramp_time=0.0)
        src.set(Command(0.4, 0.3, 120.0, Mode.MANUAL))
        _settle(arb)
        src.set(Command(0.0, 0.0, 120.0, Mode.ESTOP))
        out = arb.poll()
        assert out.v == pytest.approx(0.0)
        assert out.omega == pytest.approx(0.0)


class TestManualPreemption:
    def test_manual_overtakes_autonomous(self):
        man = FakeSource("man")
        auto = FakeSource("auto")
        arb = _arbiter(man, auto, ramp_time=0.0)
        auto.set(Command(0.3, 0.0, 100.0, Mode.AUTONOMOUS))
        _settle(arb)
        man.set(Command(0.1, 0.0, 100.0, Mode.MANUAL))
        out = arb.poll()
        assert out.mode == Mode.MANUAL


class TestIdleArbitration:
    def test_idle_requests_stand(self):
        src = FakeSource()
        arb = _arbiter(src, ramp_time=0.0)
        # First arm and drive, then disarm.
        src.set(Command(0.2, 0.0, 100.0, Mode.MANUAL))
        _settle(arb)
        src.set(Command(0.0, 0.0, 100.0, Mode.IDLE))
        out = arb.poll()
        assert out.mode == Mode.IDLE
        assert out.v == pytest.approx(0.0)

    def test_idle_keeps_d0(self):
        src = FakeSource()
        arb = _arbiter(src, ramp_time=0.0)
        src.set(Command(0.0, 0.0, 150.0, Mode.IDLE))
        out = arb.poll()
        assert out.d0 == pytest.approx(150.0)

    def test_manual_rearms_over_idle(self):
        """A DISARMED source that starts driving again resumes MANUAL."""
        src = FakeSource()
        arb = _arbiter(src, ramp_time=0.0)
        src.set(Command(0.0, 0.0, 100.0, Mode.IDLE))
        _settle(arb)
        # The same source now reports MANUAL with stick deflection.
        src.set(Command(0.2, 0.0, 100.0, Mode.MANUAL))
        out = arb.poll()
        assert out.mode == Mode.MANUAL

    def test_idle_unlocks_estop(self):
        src = FakeSource()
        arb = _arbiter(src, ramp_time=0.0)
        # Latch ESTOP by sending none (rule 7), then offer IDLE.
        src.set(None)
        _settle(arb)
        assert arb.estop_locked
        src.set(Command(0.0, 0.0, 80.0, Mode.IDLE))
        out = arb.poll()
        assert out.mode == Mode.IDLE
        assert not arb.estop_locked


class TestSmoothingDtClamp:
    def test_large_dt_does_not_jump_to_target(self):
        """Inject a huge dt gap; output must not reach target in one step."""
        src = FakeSource()
        arb = _arbiter(src, ramp_time=0.30, max_smoothing_dt=0.10)
        # Start from a known zero output.
        src.set(Command(0.0, 0.0, 58.0, Mode.MANUAL))
        _settle(arb)
        prev = arb._last_output
        # Now request a big target and simulate a long pause before next poll.
        src.set(Command(0.5, 0.0, 58.0, Mode.MANUAL))
        # Force the internal last_cmd_time far into the past.
        arb._last_cmd_time = time.monotonic() - 10.0
        out = arb.poll()
        # With dt clamped to 0.10 and ramp_time 0.30, alpha = 1/3, so output
        # moves only a third of the way toward 0.5.
        assert out.v < 0.5
        assert out.v == pytest.approx(prev.v + (0.5 - prev.v) * (0.10 / 0.30), abs=1e-6)


class TestD0HighSpeedGate:
    def test_d0_clamped_at_high_speed(self):
        src = FakeSource()
        arb = _arbiter(src, ramp_time=0.0)
        # |v| > D0_GATE_V_THRESH(0.3) => d0 capped at 120 mm even if source says 207.
        src.set(Command(0.4, 0.0, 207.0, Mode.MANUAL))
        out = arb.poll()
        import kuafu_physics as P
        assert out.d0 <= P.D0_GATE_MAX_HIGH + 1e-6

    def test_d0_full_range_at_low_speed(self):
        src = FakeSource()
        arb = _arbiter(src, ramp_time=0.0)
        src.set(Command(0.1, 0.1, 207.0, Mode.MANUAL))
        out = arb.poll()
        assert out.d0 == pytest.approx(207.0)


class TestStaleDegradation:
    def test_stale_command_is_dropped(self):
        src = FakeSource()
        arb = _arbiter(src, ramp_time=0.0, stale_time=0.001)
        # Stamp the command in the distant past.
        cmd = Command(0.2, 0.0, 100.0, Mode.MANUAL)
        cmd.stamp = time.monotonic() - 1.0
        src._cmd = cmd  # bypass set() which re-stamps
        out = arb.poll()
        assert out.mode == Mode.ESTOP  # rule 7 safe default
        assert arb.estop_locked


class TestSafeDefault:
    def test_no_source_latches_estop(self):
        arb = _arbiter(FakeSource())
        out = arb.poll()
        assert out.mode == Mode.ESTOP
        assert arb.estop_locked

    def test_none_poll_returns_estop(self):
        src = FakeSource()
        arb = _arbiter(src, ramp_time=0.0)
        src.set(None)
        out = arb.poll()
        assert out.mode == Mode.ESTOP
