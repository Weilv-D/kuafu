"""KUAFU 8-axis independent curriculum state machine.

Axes: command, d0, dr, latency, slope, step, rough, push.
Each axis has an independent level (0-4) and advances/falls back per the
AXIS_CONFIG gate. Terrain/perturbation axes (dr, latency, slope, step, rough,
push) use a pure survival gate; only d0 keeps a tracking anti-cheat gate
(best practice: never block terrain progress on velocity tracking that is
physically impossible on rough ground).

The command axis is pinned at max level (fixed full-range velocity commands,
legged_gym / MuJoCo Playground / RMA standard) and is never re-evaluated, so
there is no tracking gate and no oscillation risk on the velocity-command axis.

Per-axis episode bucketing is driven by done-env count in the trainer
(min_episodes gate), NOT by PPO-update count, so no RSL-RL callback is needed.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

# Canonical axis order. The env imports DIFF_INDICES from here as the single
# source of truth (no longer defined privately in kuafu_mjx_env.py).
AXES = ("command", "d0", "dr", "latency", "slope", "step", "rough", "push")
NUM_AXES = len(AXES)
DIFF_INDICES = {name: i for i, name in enumerate(AXES)}


@dataclass
class AxisConfig:
    """Per-axis curriculum gate.

    survival_thresh: required mean survival rate to advance.
    track_thresh: required mean track_pass rate to advance. None => no tracking
        gate (pure survival). Used only to prevent "do-nothing" policies from
        advancing on the d0 axis.
    track_metric: which tracking error feeds the gate ("linvel_yaw" or "d0").
    pinned: when True the axis is fixed at max_level and never re-evaluated
        (command axis: fixed full-range velocity commands, legged_gym / MuJoCo
        Playground / RMA standard). update_axis returns None for pinned axes.
    """
    survival_thresh: float
    track_thresh: float | None = None
    track_metric: str | None = None
    track_err: dict | None = None  # per-episode 误差门槛, 如 {"lin_vel":0.10,"yaw":0.15} / {"d0":12.0}
    pinned: bool = False


# 门控设计: terrain/扰动轴 (dr/latency/slope/step/rough/push) 纯存活门 —
# 粗糙地形上无法精确跟踪速度, 故不应用跟踪门 (ETH legged_gym 实践)。
# d0 保留跟踪反作弊软门, 阻止静止策略仅凭存活升级; track_err 取略高于
# 实测新手策略噪声地板的宽松值 (lin_vel~0.18, yaw~0.59, d0~1.3mm), 不要求精确跟踪。
# command 轴 pinned 于 max_level: 速度命令固定全范围 (legged_gym / MuJoCo
# Playground / RMA 共识), 不做课程缩放, 彻底消除命令轴阶跃振荡。track_metric/
# track_err 保留仅用于训练期 diag_track_command_pass 监控, 不参与门控。
AXIS_CONFIG: dict[str, AxisConfig] = {
    # command: pinned, 固定全范围速度命令; 无跟踪门, 无振荡。
    "command":  AxisConfig(0.90, pinned=True, track_metric="linvel_yaw",
                           track_err={"lin_vel": 0.25, "yaw": 0.50}),
    "d0":       AxisConfig(0.90, track_thresh=0.80, track_metric="d0",
                           track_err={"d0": 15.0}),
    "dr":       AxisConfig(0.90),
    "latency":  AxisConfig(0.85),
    "slope":    AxisConfig(0.85),
    "step":     AxisConfig(0.80),
    "rough":    AxisConfig(0.85),
    "push":     AxisConfig(0.80),
}


@dataclass
class AxisState:
    level: int = 0
    max_level: int = 4
    streak: int = 0  # consecutive passes
    fail_streak: int = 0


@dataclass
class Curriculum:
    """8-axis independent curriculum."""
    axes: dict = field(default_factory=lambda: {
        ax: AxisState(level=4 if AXIS_CONFIG[ax].pinned else 0) for ax in AXES})
    min_episodes: int = 256

    def update_axis(self, axis: str, episodes: list) -> str | None:
        """Update one axis from evaluation episodes. Returns 'up', 'down', or None.

        episodes: list of dicts, each with keys:
          "survived"   (bool/number)  - episode did not fall / survived long enough
          "track_pass" (bool/number)  - tracking anti-cheat passed (optional, only
                                         needed if axis has a track_thresh)

        Pinned axes (command) always return None without modifying state.
        """
        if AXIS_CONFIG[axis].pinned:
            return None
        if len(episodes) < self.min_episodes:
            return None
        cfg = AXIS_CONFIG[axis]
        ax = self.axes[axis]
        survival_rate = np.mean([e["survived"] for e in episodes])
        if cfg.track_thresh is not None:
            track_pass_rate = np.mean([e.get("track_pass", False) for e in episodes])
        else:
            track_pass_rate = 1.0  # no gate: always satisfied

        track_ok = cfg.track_thresh is None or track_pass_rate >= cfg.track_thresh
        if survival_rate >= cfg.survival_thresh and track_ok:
            ax.streak += 1
            ax.fail_streak = 0
            if ax.streak >= 2 and ax.level < ax.max_level:
                ax.level += 1
                ax.streak = 0
                return "up"
        elif survival_rate < cfg.survival_thresh or (
            cfg.track_thresh is not None and track_pass_rate < cfg.track_thresh / 2.0
        ):
            ax.fail_streak += 1
            ax.streak = 0
            if ax.fail_streak >= 2 and ax.level > 0:
                ax.level -= 1
                ax.fail_streak = 0
                return "down"
        return None

    def difficulty_vector(self) -> np.ndarray:
        """Return 8-element difficulty vector (level/max_level) for env sampling."""
        return np.array([self.axes[ax].level / self.axes[ax].max_level for ax in AXES])

    def state_dict(self) -> dict:
        return {ax: {"level": s.level, "streak": s.streak, "fail_streak": s.fail_streak}
                for ax, s in self.axes.items()}

    def load_state_dict(self, state: dict):
        for ax, s in state.items():
            if ax in self.axes:
                if AXIS_CONFIG[ax].pinned:
                    # Pinned axes stay at max_level regardless of saved state.
                    continue
                self.axes[ax].level = s["level"]
                self.axes[ax].streak = s["streak"]
                self.axes[ax].fail_streak = s["fail_streak"]
