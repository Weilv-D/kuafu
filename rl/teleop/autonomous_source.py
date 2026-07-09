# -*- coding: utf-8 -*-
"""
AutonomousSource - 自主规划器命令源(stub)

本期仅做接口占位: poll() 返回 None, 表示"无自主命令"。
仲裁器据此走 MANUAL 路径。

真正的自主实现见 docs/plans/2026-07-09-自主导航与SLAM选型-design.md。
未来规划器(Nav2 local planner 等)输出 [v, ω, d0] 后, 在本类内部订阅/轮询,
包装成 Command(mode=AUTONOMOUS) 即可--仲裁器与策略无需任何改动。

设计要点:
  - 规划器只产 [v, ω, d0], 不碰底层(LQR/电机)
  - 平衡由 policy + LQR 保证, 规划器不关心
  - 手柄一动就抢占规划器(见 CommandArbiter 规则 2)
  - 规划器输出的 [v,ω] 必须在平衡包络内(见选型文档 ADR-10 局部规划约束)
"""
from __future__ import annotations

from rl.teleop.command import Command, Mode


class AutonomousSource:
    """自主规划器源 stub。name 属性 + poll() 满足 CommandSource Protocol。

    TODO(选型文档落地后实现):
      - 接入 Nav2 local planner 的 /cmd_vel (geometry_msgs/Twist)
      - 轮式里程计估计的位姿反馈给规划器
      - 将 Twist.linear.x -> v, Twist.angular.z -> omega, 附加 d0 策略
    """

    name = "autonomous"

    def __init__(self, d0: float | None = None):
        """Args:
            d0: 自主模式默认姿态 mm; None 则用驻留态 D0_MIN。
        """
        from rl.teleop.command import D0_CMD_RANGE
        self._d0 = d0 if d0 is not None else D0_CMD_RANGE[0]
        self._active = False   # 规划器未接入, 始终不活跃

    def poll(self) -> Command | None:
        # stub: 无规划器接入, 返回 None 让仲裁器走 MANUAL / 安全默认
        return None

    # 以下方法供未来实现时调用 ----------------------------------
    def set_command(self, v: float, omega: float, d0: float | None = None) -> None:
        """规划器回调: 写入最新 [v, ω, d0], 下次 poll 返回它。"""
        # TODO: 接入 ROS2 subscriber 后在此更新缓存
        self._active = True
