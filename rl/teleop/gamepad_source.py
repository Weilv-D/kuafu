# -*- coding: utf-8 -*-
"""
GamepadSource - pygame 手柄命令源

轴映射默认按 Xbox/通用手柄布局 (SDL2 evdev ABS 升序):
  左摇杆 Y (轴 1, ABS_Y)      -> v_cmd  (前后, ±0.5 m/s)
  右摇杆 X (轴 2, ABS_Z/RX)   -> ω_cmd  (转向, ±1.0 rad/s)
  LT/RT 扳机 (轴 4/5, GAS/BRAKE) -> d0  (蹲下/站起, 58~207 mm)
  A 键 (按钮 0)               -> ESTOP  (急停)
  B 键 (按钮 1)               -> ESTOP  (急停)

不同手柄的轴编号可能不同 (Flydigi、PS、Switch 布局差异)。用环境变量覆盖:
  KUAFU_AXIS_V    左摇杆 Y 轴号 (默认 1)
  KUAFU_AXIS_W    右摇杆 X 轴号 (默认 2)
  KUAFU_AXIS_LT   LT 扳机轴号   (默认 4)
  KUAFU_AXIS_RT   RT 扳机轴号   (默认 5)
  KUAFU_BTN_ESTOP 急停按钮号    (默认 0)
  KUAFU_AXIS_V_INVERT  设为 1 则反转速度轴 (默认 1, 因 pygame Y 向下为正)

确认手柄轴映射: python -m rl.teleop.gamepad_source --show-axes

无手柄时 __init__ 抛 RuntimeError, 上层 fallback 到 KeyboardSource。
"""
from __future__ import annotations

import os
import time

import pygame

from rl.teleop.command import Command, Mode, V_CMD_RANGE, W_CMD_RANGE, D0_CMD_RANGE
from rl.teleop.pygame_base import init_pygame, pump_events


def _env_int(name: str, default: int) -> int:
    """从环境变量读取整数, 解析失败或未设则用默认值。"""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class GamepadSource:
    """pygame 手柄源。name 属性 + poll() 满足 CommandSource Protocol。"""

    name = "gamepad"

    def __init__(self):
        init_pygame()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("未检测到手柄, 请接入或改用 --device keyboard")
        self._joy = pygame.joystick.Joystick(0)
        # 轴索引: 默认 Xbox 布局, 可用环境变量覆盖以适配其他手柄
        self._axis_v = _env_int("KUAFU_AXIS_V", 1)       # 左摇杆 Y
        self._axis_w = _env_int("KUAFU_AXIS_W", 2)       # 右摇杆 X
        self._axis_lt = _env_int("KUAFU_AXIS_LT", 4)     # LT (左扳机)
        self._axis_rt = _env_int("KUAFU_AXIS_RT", 5)     # RT (右扳机)
        self._btn_estop = _env_int("KUAFU_BTN_ESTOP", 0) # A
        self._btn_mode = _env_int("KUAFU_BTN_MODE", 1)   # B
        # pygame 摇杆 Y 向下为正, 取反让上推=前进; 可用环境变量关闭反转
        self._invert_v = _env_int("KUAFU_AXIS_V_INVERT", 1) != 0
        self._d0 = D0_CMD_RANGE[0]  # 初始姿态: 驻留态
        self._mode = Mode.MANUAL
        self._last_poll = time.monotonic()

    def poll(self) -> Command | None:
        pump_events()
        now = time.monotonic()
        dt = now - self._last_poll  # 实测 dt (周期抖动会累积误差, 不硬编码)
        self._last_poll = now

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
        if self._invert_v:
            vy = -vy   # pygame Y 向下为正, 取反让上推=前进
        vy = self._apply_deadzone(vy)
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
        self._d0 += (rt_norm - lt_norm) * rate * dt  # 用实测 dt
        self._d0 = max(D0_CMD_RANGE[0], min(D0_CMD_RANGE[1], self._d0))

        return Command(v, omega, self._d0, Mode.MANUAL, now)

    @staticmethod
    def _apply_deadzone(x: float, dz: float = 0.08) -> float:
        """摇杆中心死区: |x|<dz 归零, 否则重新映射到 [0,1]。"""
        if abs(x) < dz:
            return 0.0
        return (x - dz * (1 if x > 0 else -1)) / (1.0 - dz)


def _show_axes() -> None:
    """实时打印每个轴的值, 帮助确认手柄轴映射。

    用法: SDL_VIDEODRIVER=dummy python -m rl.teleop.gamepad_source --show-axes
    然后逐个推摇杆/按扳机, 看哪个轴号响应, 据此设置 KUAFU_AXIS_* 环境变量。
    """
    import argparse
    parser = argparse.ArgumentParser(description="显示手柄实时轴值 (确认映射用)")
    parser.add_argument("--show-axes", action="store_true")
    parser.parse_args()
    init_pygame()
    if pygame.joystick.get_count() == 0:
        print("未检测到手柄")
        return
    j = pygame.joystick.Joystick(0)
    j.init()
    n = j.get_numaxes()
    print(f"手柄: {j.get_name()} ({n} 轴, {j.get_numbuttons()} 钮)")
    print("逐个操作摇杆/扳机, 看哪个轴号变化。Ctrl-C 退出。")
    print("-" * 50)
    prev = [j.get_axis(i) for i in range(n)]
    try:
        while True:
            pump_events()
            for i in range(n):
                v = j.get_axis(i)
                if abs(v - prev[i]) > 0.15:
                    print(f"  轴{i}: {prev[i]:+.2f} -> {v:+.2f}")
                    prev[i] = v
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n退出。据上面的输出设置环境变量, 例如:")
        print("  export KUAFU_AXIS_V=1 KUAFU_AXIS_W=3 KUAFU_AXIS_LT=4 KUAFU_AXIS_RT=5")


if __name__ == "__main__":
    _show_axes()
