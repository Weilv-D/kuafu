# -*- coding: utf-8 -*-
"""
pygame 共享初始化。

GamepadSource / KeyboardSource 都需要一个 pygame display 窗口来接收输入事件。
两者通过 init_pygame(display_title) 共享同一个 display, 只初始化一次。
"""
from __future__ import annotations

import pygame

_initialized = False


def init_pygame(display_title: str = "KUAFU teleop") -> None:
    """初始化 pygame(含 joystick 子系统)并建一个最小 display。

    幂等: 重复调用不会重复初始化。display 是事件泵的前提(键盘事件依赖它)。
    """
    global _initialized
    if _initialized:
        return
    pygame.init()
    pygame.joystick.init()
    # 1x1 窗口即可, 仅用于接收键盘事件; 设为 SCALED 避免某些环境无显示报错
    try:
        pygame.display.set_mode((1, 1), pygame.SCALED)
        pygame.display.set_caption(display_title)
    except pygame.error:
        # 无头环境退化为 dummy display
        pygame.display.set_mode((1, 1))
    _initialized = True


def pump_events() -> None:
    """每个周期调用, 推进 pygame 事件队列(否则键盘/手柄状态不更新)。"""
    pygame.event.pump()
