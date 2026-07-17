# -*- coding: utf-8 -*-
"""
teleop 包 — 手柄遥控输入整形与原生读取

组件:
  - shaping          : 死区 / 平方曲线 / 扳机归一化 (纯函数)
  - native_joystick  : /dev/input/jsX 原生读取器 (不依赖 pygame/SDL)
  - calibrate_native : 交互式手柄标定工具
  - bt_wakeup        : BLE 手柄休眠唤醒守护

运行入口: pi5_runtime.teleop_single (单进程遥控, 基线 LQR)
"""
