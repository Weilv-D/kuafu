# -*- coding: utf-8 -*-
"""
GamepadSource - pygame 手柄命令源

两态使能模型(显式 ARM/DISARM, 启动默认 DISARMED 安全态):
  START / arm 键   -> ARMED  : 发 Mode.MANUAL(ACTIVE), 轮出力+RL残差, 摇杆跟踪
  Back / disarm 键 -> DISARMED: 发 Mode.IDLE(STAND), LQR 保平衡但轮不跟走, RL残差关
  A / estop 键     -> ESTOP 锁存(轮失能, 需 ARM 解除)

轴映射默认 Xbox 布局, 不同手柄(Flydigi/PS/Switch)用环境变量覆盖:
  KUAFU_AXIS_V    左摇杆 Y (v 前后)         默认 1
  KUAFU_AXIS_W    右摇杆 X (ω 转向)         默认 2
  KUAFU_AXIS_LT   LT 扳机 (蹲)              默认 4
  KUAFU_AXIS_RT   RT 扳机 (站)              默认 5
  KUAFU_AXIS_V_INVERT  反转 v 轴(默认 1)    pygame Y 向下为正
  KUAFU_AXIS_W_INVERT  反转 ω 轴(默认 0)
  KUAFU_BTN_ARM   使能键(默认 7=START)
  KUAFU_BTN_DISARM 卸能键(默认 6=Back)
  KUAFU_BTN_ESTOP  急停键(默认 0=A)
  KUAFU_RUMBLE    触觉反馈(默认 1, 设 0 关闭)

摇杆走 死区 -> 平方曲线 管道(sign·|x|², 低速段精度高); 扳机带死区防误触 d0 漂移。
D0 为 rate-based 累积(速率由 ArbiterConfig.d0_rate_mm_s 决定, 默认 40 mm/s)。

热插拔: 断连时 poll() 返回 ESTOP(降级到仲裁器安全默认), 重连后需重新 ARM。
确认手柄轴映射: python -m rl.teleop.gamepad_source --show-axes

无手柄时 __init__ 不再抛异常(支持热插拔等待); poll() 在无手柄时返回 ESTOP,
上层 arbiter 据此走安全默认。teleop_node 的 fallback 仍可用(检测 name 或 0 轴)。
"""
from __future__ import annotations

import os
import time

import pygame

from rl.teleop.command import (
    ArbiterConfig, Command, Mode, V_CMD_RANGE, W_CMD_RANGE, D0_CMD_RANGE,
)
from rl.teleop.pygame_base import init_pygame, pump_events
from rl.teleop.shaping import normalize_trigger, shape_axis


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip() not in ("0", "false", "False", "")


# Rumble profiles: (low_freq_strength, high_freq_strength, duration_ms).
# Distinct signatures so the operator can tell events apart by feel.
_RUMBLE_PROFILES = {
    "arm":       (0.4, 0.0, 120),    # 低频短促: 已使能
    "disarm":    (0.0, 0.4, 120),    # 高频短促: 已卸能
    "estop":     (1.0, 1.0, 400),    # 强烈长震: 急停
    "reconnect": (0.3, 0.3, 200),    # 双频中震: 重连
}


class GamepadSource:
    """pygame 手柄源。name 属性 + poll() 满足 CommandSource Protocol。"""

    name = "gamepad"

    def __init__(self, cfg: ArbiterConfig | None = None):
        self._cfg = cfg or ArbiterConfig()
        init_pygame("KUAFU teleop (gamepad)")
        # 轴/键索引: 默认 Xbox 布局, 环境变量覆盖
        self._axis_v = _env_int("KUAFU_AXIS_V", 1)
        self._axis_w = _env_int("KUAFU_AXIS_W", 2)
        self._axis_lt = _env_int("KUAFU_AXIS_LT", 4)
        self._axis_rt = _env_int("KUAFU_AXIS_RT", 5)
        self._btn_arm = _env_int("KUAFU_BTN_ARM", 7)      # START
        self._btn_disarm = _env_int("KUAFU_BTN_DISARM", 6) # Select/Back
        self._btn_estop = _env_int("KUAFU_BTN_ESTOP", 0)   # A
        self._invert_v = _env_bool("KUAFU_AXIS_V_INVERT", True)
        self._invert_w = _env_bool("KUAFU_AXIS_W_INVERT", False)
        self._rumble_enabled = _env_bool("KUAFU_RUMBLE", True)
        # 状态
        self._armed = False           # 启动默认 DISARMED(安全)
        self._estop_latched = False
        self._d0 = D0_CMD_RANGE[0]    # 初始姿态: 驻留态
        self._last_poll = time.monotonic()
        self._prev_buttons: dict[int, bool] = {}
        # 手柄句柄(支持热插拔; 初始可能为 None)
        self._joy: pygame.joystick.Joystick | None = None
        self._joy_instance_id: int | None = None
        self._open_first_joystick()
        if self._joy is None:
            # 兼容旧调用方: 无手柄时打印一次提示。poll() 会持续返回 ESTOP
            # 等待热插拔, 而不是抛异常阻塞启动。
            print("[gamepad] no joystick detected; waiting for hot-plug "
                  "(poll() returns ESTOP until a controller connects)")

    # ------------------------------------------------------------------
    # 手柄生命周期
    # ------------------------------------------------------------------
    def _open_first_joystick(self) -> None:
        if pygame.joystick.get_count() > 0:
            self._joy = pygame.joystick.Joystick(0)
            self._joy.init()
            self._joy_instance_id = self._joy.get_instance_id()

    def _handle_hotplug(self, events: list) -> None:
        for ev in events:
            if ev.type == pygame.JOYDEVICEADDED and self._joy is None:
                self._joy = pygame.joystick.Joystick(ev.instance_id if hasattr(ev, "instance_id") else 0)
                self._joy.init()
                self._joy_instance_id = self._joy.get_instance_id()
                self._rumble("reconnect")
            elif ev.type == pygame.JOYDEVICEREMOVED and self._joy is not None:
                removed_id = ev.instance_id if hasattr(ev, "instance_id") else self._joy_instance_id
                if removed_id == self._joy_instance_id:
                    self._joy = None
                    self._joy_instance_id = None
                    self._armed = False    # 断连自动卸能, 重连后需重新 ARM

    # ------------------------------------------------------------------
    # 按钮边沿 + rumble
    # ------------------------------------------------------------------
    def _on_button_edge(self, action: str) -> None:
        if action == "arm":
            self._armed = True
            self._estop_latched = False
            self._rumble("arm")
        elif action == "disarm":
            self._armed = False
            self._rumble("disarm")
        elif action == "estop":
            self._armed = False
            self._estop_latched = True
            self._rumble("estop")

    def _rumble(self, kind: str) -> None:
        if not self._rumble_enabled or self._joy is None:
            return
        lo, hi, dur = _RUMBLE_PROFILES[kind]
        try:
            self._joy.rumble(lo, hi, dur)
        except (pygame.error, OSError, AttributeError):
            pass    # 不支持 rumble 的手柄静默降级

    # ------------------------------------------------------------------
    # 主轮询
    # ------------------------------------------------------------------
    def poll(self) -> Command | None:
        self._handle_hotplug(pump_events())
        now = time.monotonic()
        dt = now - self._last_poll
        self._last_poll = now

        # 手柄断连 -> ESTOP(仲裁器会平滑归零并保持 d0)
        if self._joy is None:
            return Command(0.0, 0.0, self._d0, Mode.ESTOP, now)

        # --- 按钮边沿触发(上升沿, 避免按住一直切换) ---
        for btn, action in (
            (self._btn_arm, "arm"),
            (self._btn_disarm, "disarm"),
            (self._btn_estop, "estop"),
        ):
            pressed = bool(self._joy.get_button(btn))
            if pressed and not self._prev_buttons.get(btn, False):
                self._on_button_edge(action)
            self._prev_buttons[btn] = pressed

        if self._estop_latched:
            return Command(0.0, 0.0, self._d0, Mode.ESTOP, now)

        # --- 摇杆 v/omega (死区 + 平方曲线) ---
        vy = self._read_axis(self._axis_v, self._invert_v)
        wx = self._read_axis(self._axis_w, self._invert_w)
        vy = shape_axis(vy, self._cfg.stick_deadzone, self._cfg.stick_gamma)
        wx = shape_axis(wx, self._cfg.stick_deadzone, self._cfg.stick_gamma)
        v = vy * V_CMD_RANGE[1]
        omega = wx * W_CMD_RANGE[1]

        # --- D0 rate (扳机带死区, 防误触漂移) ---
        lt = normalize_trigger(self._joy.get_axis(self._axis_lt), self._cfg.trigger_deadzone)
        rt = normalize_trigger(self._joy.get_axis(self._axis_rt), self._cfg.trigger_deadzone)
        self._d0 += (rt - lt) * self._cfg.d0_rate_mm_s * dt
        self._d0 = max(D0_CMD_RANGE[0], min(D0_CMD_RANGE[1], self._d0))

        if not self._armed:
            # DISARMED: 请求 STAND 保平衡, 摇杆不跟走(v/w 强制 0), D0 仍可调
            return Command(0.0, 0.0, self._d0, Mode.IDLE, now)
        return Command(v, omega, self._d0, Mode.MANUAL, now)

    def _read_axis(self, axis: int, invert: bool) -> float:
        val = float(self._joy.get_axis(axis))   # type: ignore[union-attr]
        return -val if invert else val

    # ------------------------------------------------------------------
    # 调试辅助
    # ------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._joy is not None

    @property
    def armed(self) -> bool:
        return self._armed


def _show_axes() -> None:
    """实时打印每个轴的值, 帮助确认手柄轴映射。

    用法: SDL_VIDEODRIVER=dummy python -m rl.teleop.gamepad_source --show-axes
    然后逐个推摇杆/按扳机, 看哪个轴号响应, 据此设置 KUAFU_AXIS_* 环境变量。
    """
    import argparse
    parser = argparse.ArgumentParser(description="显示手柄实时轴值 (确认映射用)")
    parser.add_argument("--show-axes", action="store_true")
    parser.parse_args()
    init_pygame("KUAFU axes")
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
