# -*- coding: utf-8 -*-
"""
统一命令接口 - 遥控与自主共用

策略(policy)已训练成跟踪 3 维命令 [v, ω, d0]。所有命令源(手柄/键盘/自主规划器)
实现 CommandSource, 产出 Command, 经 CommandArbiter 仲裁后注入策略 obs 的
第 30-32 维(见 rl/env/kuafu_env.py OBS_SPEC)。新增命令源只写一个类, 仲裁器与
策略都不碰。
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

# 复用项目物理常量真源(d0 范围)
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
import kuafu_physics as P


# ============================================================
# 命令速度范围 - 与训练环境 kuafu_mjx_env.py V/W/D0_CMD_RANGE 对齐
# (此处自包含定义, 避免把 jax 拖进 teleop 包)
# ============================================================
V_CMD_RANGE = (-0.5, 0.5)                  # m/s (轮缘额定 0.82 m/s, 留余量)
W_CMD_RANGE = (-1.0, 1.0)                  # rad/s
D0_CMD_RANGE = (P.D0_MIN, P.D0_MAX)        # (58, 207) mm


class Mode(Enum):
    """命令源工作模式。

    MANUAL     - 手柄/键盘直接控制(人全程操控)
    AUTONOMOUS - 自主规划器控制(机器自己走)
    ASSISTED   - 保留: 自主给主速度 + 人叠加微调(本期不实现)
    ESTOP      - 急停: 输出归零并保持当前 d0, 锁定
    """

    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    ASSISTED = "assisted"
    ESTOP = "estop"


@dataclass
class Command:
    """一条高层命令, 适配策略 obs 的 command 分量。

    Attributes:
        v:     前向线速度 m/s   (V_CMD_RANGE)
        omega: 偏航角速度 rad/s (W_CMD_RANGE)
        d0:    足端下垂量 mm     (D0_CMD_RANGE)
        mode:  命令源当前模式
        stamp: time.monotonic() 时间戳, 仲裁器判超时用
    """

    v: float
    omega: float
    d0: float
    mode: Mode
    stamp: float = field(default_factory=time.monotonic)

    def as_array(self) -> "list[float]":
        """返回 [v, omega, d0], 供 _build_obs 的 command 参数直接使用。"""
        return [self.v, self.omega, self.d0]


@runtime_checkable
class CommandSource(Protocol):
    """命令源统一接口。实现方提供 name 属性和 poll() 方法。

    poll() 在每个控制周期(50Hz)被调用一次, 返回最新命令; 无新数据时返回 None
    (仲裁器据此判该源静默或失效)。
    """

    name: str

    def poll(self) -> Command | None: ...


@dataclass
class ArbiterConfig:
    """CommandArbiter 安全参数。所有时长单位秒。"""

    # 手柄抢占死区: |v| 或 |omega| 超过此值视为"人在操控"
    manual_deadzone_v: float = 0.05        # m/s
    manual_deadzone_w: float = 0.10        # rad/s
    # 手柄松手后多久交还自主 (人手抖动容差)
    handoff_time: float = 1.5              # s
    # 模式切换 / 急停恢复时的速度平滑过渡时长(防突跳摔机)
    ramp_time: float = 0.30                # s
    # 源失效判据: stamp 老于此值则该源降级
    stale_time: float = 0.50               # s
    # 速度限幅(与训练 V/W/D0_CMD_RANGE 一致, 此处显式给出便于 sim 调参)
    v_limit: tuple[float, float] = V_CMD_RANGE
    w_limit: tuple[float, float] = W_CMD_RANGE
    d0_limit: tuple[float, float] = D0_CMD_RANGE
