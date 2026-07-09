# -*- coding: utf-8 -*-
"""
GamepadSource - pygame 手柄命令源

轴映射(Xbox/通用手柄布局):
  左摇杆 Y (轴 1)  -> v_cmd  (前后, ±0.5 m/s)
  右摇杆 X (轴 2/3) -> ω_cmd  (转向, ±1.0 rad/s)
  LT/RT 扳机       -> d0     (蹲下/站起, 58~207 mm)
  A 键 (按钮 0)    -> ESTOP  (急停)
  B 键 (按钮 1)    -> 模式切换标记(由上层解读)

无手柄时 __init__ 抛 RuntimeError, 上层 fallback 到 KeyboardSource。
"""
from __future__ import annotations

import time

import pygame

from rl.teleop.command import Command, Mode, V_CMD_RANGE, W_CMD_RANGE, D0_CMD_RANGE
from rl.teleop.pygame_base import init_pygame, pump_events


class GamepadSource:
    """pygame 手柄源。name 属性 + poll() 满足 CommandSource Protocol。"""

    name = "gamepad"

    def __init__(self):
        init_pygame()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("未检测到手柄, 请接入或改用 --device keyboard")
        self._joy = pygame.joystick.Joystick(0)
        # 轴索引(通用布局; 不同手柄可能需微调, 见 _apply_axis_mapping 注释)
        self._axis_v = 1       # 左摇杆 Y
        self._axis_w = 3       # 右摇杆 X (多数 Xbox 手柄为轴 3)
        self._axis_lt = 2      # LT
        self._axis_rt = 5      # RT
        self._btn_estop = 0    # A
        self._btn_mode = 1     # B
        self._d0 = D0_CMD_RANGE[0]  # 初始姿态: 驻留态
        self._mode = Mode.MANUAL

    def poll(self) -> Command | None:
        pump_events()
        now = time.monotonic()

        # --- 急停按钮 ---
        if self._joy.get_button(self._btn_estop):
            return Command(0.0, 0.0, self._d0, Mode.ESTOP, now)

        # --- 模式切换按钮(按下则置 ESTOP 让仲裁器接管切换) ---
        # 这里简化: B 按下即请求 ESTOP, 由用户松开恢复
        if self._joy.get_button(self._btn_mode):
            return Command(0.0, 0.0, self._d0, Mode.ESTOP, now)

        # --- 速度: 摇杆(中心死区 + 反 Y 轴, 上推为正) ---
        vy = self._joy.get_axis(self._axis_v)
        wx = self._joy.get_axis(self._axis_w)
        vy = self._apply_deadzone(-vy)   # 手柄 Y 向下为正, 取反让上推=前进
        wx = self._apply_deadzone(wx)
        v = vy * V_CMD_RANGE[1]          # 归一化 -> m/s
        omega = wx * W_CMD_RANGE[1]      # 归一化 -> rad/s

        # --- d0: 扳机(LT 蹲下 / RT 站起), 持续按累积 ---
        lt = self._joy.get_axis(self._axis_lt)
        rt = self._joy.get_axis(self._axis_rt)
        # pygame 扳机范围 [-1, 1], 归一到 [0, 1]
        lt_norm = (lt + 1) / 2
        rt_norm = (rt + 1) / 2
        rate = 40.0  # mm/s 调节速率
        dt = 0.02    # 假定 50Hz poll
        self._d0 += (rt_norm - lt_norm) * rate * dt
        self._d0 = max(D0_CMD_RANGE[0], min(D0_CMD_RANGE[1], self._d0))

        return Command(v, omega, self._d0, Mode.MANUAL, now)

    @staticmethod
    def _apply_deadzone(x: float, dz: float = 0.08) -> float:
        """摇杆中心死区: |x|<dz 归零, 否则重新映射到 [0,1]。"""
        if abs(x) < dz:
            return 0.0
        return (x - dz * (1 if x > 0 else -1)) / (1.0 - dz)
