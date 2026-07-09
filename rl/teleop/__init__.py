# -*- coding: utf-8 -*-
"""
teleop 包 - 手柄遥控 / 自主规划 命令接口

提供统一的 Command 抽象,让手柄、键盘、自主规划器等不同命令源可插拔地接入
同一套策略(策略本身零改动,它本就是 command-following)。

核心组件:
  - Command / Mode / CommandSource : 统一接口定义 (command.py)
  - CommandArbiter                  : 多源仲裁 + 安全层 (arbiter.py)
  - GamepadSource / KeyboardSource  : 遥控源 (pygame)
  - AutonomousSource                : 自主源 stub (接口占位, 实现见选型文档)
"""
from rl.teleop.command import Command, Mode, CommandSource, ArbiterConfig
from rl.teleop.arbiter import CommandArbiter

__all__ = [
    "Command",
    "Mode",
    "CommandSource",
    "ArbiterConfig",
    "CommandArbiter",
]
