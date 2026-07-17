# -*- coding: utf-8 -*-
"""
KeyboardSource - 键盘命令源(无手柄 fallback)

与 GamepadSource 一致的两态使能模型(启动默认 DISARMED):
  Enter     -> ARMED  (发 Mode.MANUAL / ACTIVE, 轮出力+RL残差)
  Backspace -> DISARMED (发 Mode.IDLE / STAND, 保平衡但不跟走)
  空格      -> ESTOP 锁存(轮失能, Enter 解除)
  R         -> 解锁(从急停恢复, 等价于 ARM)

键位:
  W/S    -> v_cmd  前进/后退 (±0.25 m/s, 按住生效, 离散档位不套曲线)
  A/D    -> ω_cmd  左转/右转 (±0.8 rad/s, 按住生效)
  Q/E    -> d0     蹲下/站起 (按住调, 速率 ArbiterConfig.d0_rate_mm_s, 默认 40 mm/s)

保证没手柄也能在仿真里测通整套遥控链路。
"""
from __future__ import annotations

import time

import pygame

from rl.teleop.command import ArbiterConfig, Command, Mode, V_CMD_RANGE, W_CMD_RANGE, D0_CMD_RANGE
from rl.teleop.pygame_base import init_pygame, pump_events


class KeyboardSource:
    """pygame 键盘源。name 属性 + poll() 满足 CommandSource Protocol。"""

    name = "keyboard"

    def __init__(self, cfg: ArbiterConfig | None = None):
        self._cfg = cfg or ArbiterConfig()
        init_pygame("KUAFU teleop (keyboard)")
        self._d0 = D0_CMD_RANGE[0]       # 初始驻留态
        self._armed = False              # 启动默认 DISARMED(安全)
        self._estop = False
        self._last_poll = time.monotonic()

    def poll(self) -> Command | None:
        pump_events()
        keys = pygame.key.get_pressed()
        now = time.monotonic()
        dt = now - self._last_poll  # 实测 dt (周期抖动会累积误差, 不硬编码)
        self._last_poll = now

        # --- 使能/急停控制 ---
        if keys[pygame.K_SPACE]:
            self._estop = True
            self._armed = False
            return Command(0.0, 0.0, self._d0, Mode.ESTOP, now)
        if keys[pygame.K_RETURN]:
            self._estop = False
            self._armed = True
        elif keys[pygame.K_BACKSPACE]:
            self._armed = False
        if keys[pygame.K_r]:
            self._estop = False          # R 仅解锁急停, 不改变 ARMED
        if self._estop:
            return Command(0.0, 0.0, self._d0, Mode.ESTOP, now)

        # --- d0: QE 连续调节 (共享 d0_rate_mm_s) ---
        rate = self._cfg.d0_rate_mm_s
        if keys[pygame.K_q]:
            self._d0 -= rate * dt
        if keys[pygame.K_e]:
            self._d0 += rate * dt
        self._d0 = max(D0_CMD_RANGE[0], min(D0_CMD_RANGE[1], self._d0))

        if not self._armed:
            # DISARMED: 请求 STAND 保平衡, 速度归零
            return Command(0.0, 0.0, self._d0, Mode.IDLE, now)

        # --- 速度: WASD 离散档位 (按键是阶跃, 不套曲线) ---
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

        return Command(v, omega, self._d0, Mode.MANUAL, now)

    @property
    def armed(self) -> bool:
        return self._armed
