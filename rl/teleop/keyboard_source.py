# -*- coding: utf-8 -*-
"""
KeyboardSource - 键盘命令源(无手柄 fallback)

键位:
  W/S    -> v_cmd  前进/后退 (±0.25 m/s, 按住生效)
  A/D    -> ω_cmd  左转/右转 (±0.8 rad/s, 按住生效)
  Q/E    -> d0     蹲下/站起 (按住调, 58~207 mm)
  空格   -> ESTOP  急停
  R      -> 解锁(从急停恢复)

保证没手柄也能在仿真里测通整套遥控链路。
"""
from __future__ import annotations

import time

import pygame

from rl.teleop.command import Command, Mode, V_CMD_RANGE, W_CMD_RANGE, D0_CMD_RANGE
from rl.teleop.pygame_base import init_pygame, pump_events


class KeyboardSource:
    """pygame 键盘源。name 属性 + poll() 满足 CommandSource Protocol。"""

    name = "keyboard"

    def __init__(self):
        init_pygame("KUAFU teleop (keyboard)")
        self._d0 = D0_CMD_RANGE[0]  # 初始驻留态
        self._estop = False
        self._last_poll = time.monotonic()

    def poll(self) -> Command | None:
        pump_events()
        keys = pygame.key.get_pressed()
        now = time.monotonic()
        dt = now - self._last_poll  # 实测 dt (周期抖动会累积误差, 不硬编码)
        self._last_poll = now

        # 急停(空格按下) / 解锁(R 按下)
        if keys[pygame.K_SPACE]:
            self._estop = True
            return Command(0.0, 0.0, self._d0, Mode.ESTOP, now)
        if keys[pygame.K_r]:
            self._estop = False
        if self._estop:
            return Command(0.0, 0.0, self._d0, Mode.ESTOP, now)

        # 速度: WASD 离散档位
        v = 0.0
        if keys[pygame.K_w]:
            v += V_CMD_RANGE[1] * 0.5
        if keys[pygame.K_s]:
            v -= V_CMD_RANGE[1] * 0.5
        omega = 0.0
        if keys[pygame.K_a]:
            omega += W_CMD_RANGE[1] * 0.8
        if keys[pygame.K_d]:
            omega -= W_CMD_RANGE[1] * 0.8

        # d0: QE 连续调节(40 mm/s, 用实测 dt)
        rate = 40.0
        if keys[pygame.K_q]:
            self._d0 -= rate * dt
        if keys[pygame.K_e]:
            self._d0 += rate * dt
        self._d0 = max(D0_CMD_RANGE[0], min(D0_CMD_RANGE[1], self._d0))

        return Command(v, omega, self._d0, Mode.MANUAL, now)
