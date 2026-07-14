"""KUAFU 7-axis independent curriculum state machine.

Axes: command, D0, dynamics_DR, latency_noise, slope, step, push.
Each axis has independent levels (0-4), evaluated every 25 PPO updates.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

AXES = ("command", "d0", "dr", "latency", "slope", "step", "push")
NUM_AXES = len(AXES)

@dataclass
class AxisState:
    level: int = 0
    max_level: int = 4
    streak: int = 0  # consecutive passes
    fail_streak: int = 0
    eval_episodes: list = field(default_factory=list)

@dataclass
class Curriculum:
    """7-axis independent curriculum."""
    axes: dict = field(default_factory=lambda: {ax: AxisState() for ax in AXES})
    eval_interval: int = 25  # PPO updates between evaluations
    min_episodes: int = 256
    last_eval_update: int = 0

    def should_evaluate(self, completed_updates: int) -> bool:
        return completed_updates - self.last_eval_update >= self.eval_interval

    def update_axis(self, axis: str, episodes: list) -> str | None:
        """Update one axis from evaluation episodes. Returns 'up', 'down', or None."""
        if len(episodes) < self.min_episodes:
            return None
        ax = self.axes[axis]
        survival_rate = np.mean([e["survived"] for e in episodes])
        track_pass = np.mean([e.get("track_pass", False) for e in episodes])

        if survival_rate >= 0.90 and track_pass >= 0.80:
            ax.streak += 1
            ax.fail_streak = 0
            if ax.streak >= 2 and ax.level < ax.max_level:
                ax.level += 1
                ax.streak = 0
                return "up"
        elif survival_rate < 0.70 or track_pass < 0.50:
            ax.fail_streak += 1
            ax.streak = 0
            if ax.fail_streak >= 2 and ax.level > 0:
                ax.level -= 1
                ax.fail_streak = 0
                return "down"
        return None

    def difficulty_vector(self) -> np.ndarray:
        """Return 7-element difficulty vector for environment sampling."""
        return np.array([self.axes[ax].level / self.axes[ax].max_level for ax in AXES])

    def state_dict(self) -> dict:
        return {ax: {"level": s.level, "streak": s.streak, "fail_streak": s.fail_streak}
                for ax, s in self.axes.items()}

    def load_state_dict(self, state: dict):
        for ax, s in state.items():
            if ax in self.axes:
                self.axes[ax].level = s["level"]
                self.axes[ax].streak = s["streak"]
                self.axes[ax].fail_streak = s["fail_streak"]
