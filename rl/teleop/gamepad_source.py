# -*- coding: utf-8 -*-
"""
GamepadSource - 手柄命令源 (原生 js0 读取 + pygame 热插拔/触觉)

两态使能模型(显式 ARM/DISARM, 启动默认 DISARMED 安全态):
  START / arm 键   -> ARMED  : 发 Mode.MANUAL(ACTIVE), 轮出力+RL残差, 摇杆跟踪
  Back / disarm 键 -> DISARMED: 发 Mode.IDLE(STAND), LQR 保平衡但轮不跟走, RL残差关
  A / estop 键     -> ESTOP 锁存(轮失能, 需 ARM 解除)

轴/按钮读取走原生 /dev/input/js0 (不依赖 pygame get_axis/get_button, 蓝牙 LE
游戏中 pygame 存在轴值更新滞后/丢事件)。pygame 仅用于 joystick 枚举、热插拔事件
(JOYDEVICEADDED/REMOVED) 和 rumble 触觉反馈。

轴映射默认 Xbox 布局, 不同手柄(Flydigi/PS/Switch)用环境变量覆盖:
  KUAFU_AXIS_V    左摇杆 Y (v 前后)         默认 1
  KUAFU_AXIS_W    右摇杆 X (ω 转向)         默认 2
  KUAFU_AXIS_LT   LT 扳机 (蹲)              默认 4
  KUAFU_AXIS_RT   RT 扳机 (站)              默认 5
  KUAFU_AXIS_V_INVERT  反转 v 轴(默认 1)    pygame Y 向下为正
  KUAFU_AXIS_W_INVERT  反转 ω 轴(默认 0)
  KUAFU_AXIS_LT_INVERT 反转 LT 扳机(默认 0)  某些手柄(如 VADER2P RT)静止为 +1
  KUAFU_AXIS_RT_INVERT 反转 RT 扳机(默认 0)
  KUAFU_BTN_ARM   使能键(默认 7=START)
  KUAFU_BTN_DISARM 卸能键(默认 6=Back)
  KUAFU_BTN_ESTOP  急停键(默认 0=A)
  KUAFU_RUMBLE    触觉反馈(默认 1, 设 0 关闭)

摇杆走 死区 -> 平方曲线 管道(sign·|x|², 低速段精度高); 扳机带死区防误触 d0 漂移。
D0 为 rate-based 累积(速率由 ArbiterConfig.d0_rate_mm_s 决定, 默认 40 mm/s)。

热插拔: 断连时 poll() 返回 ESTOP(降级到仲裁器安全默认), 重连后需重新 ARM。
标定手柄轴/按钮映射: python -m rl.teleop.gamepad_source --calibrate
(旧的无引导轴值监视: --show-axes)

无手柄时 __init__ 不再抛异常(支持热插拔等待); poll() 在无手柄时返回 ESTOP,
上层 arbiter 据此走安全默认。teleop_node 的 fallback 仍可用(检测 name 或 0 轴)。
"""
from __future__ import annotations

import fcntl
import os
import struct
import time

import pygame

from rl.teleop.command import (
    ArbiterConfig, Command, Mode, V_CMD_RANGE, W_CMD_RANGE, D0_CMD_RANGE,
)
from rl.teleop.pygame_base import init_pygame, pump_events
from rl.teleop.shaping import normalize_trigger, shape_axis

JS_DEVICE = "/dev/input/js0"
JS_EVENT_FMT = "IhBB"       # time_ms, value, type, number
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FMT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_AXIS_MAX = 32767.0


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
_RUMBLE_PROFILES = {
    "arm":       (0.4, 0.0, 120),
    "disarm":    (0.0, 0.4, 120),
    "estop":     (1.0, 1.0, 400),
    "reconnect": (0.3, 0.3, 200),
}


class _NativeJoystick:
    """Read /dev/input/js0 directly for axis/button state.

    Pygame's ``get_axis()`` / ``get_button()`` can return stale values on
    Bluetooth LE gamepads (SDL 2.28.4 caches joystick state between event pumps
    and does not always flush it).  This class reads the kernel interface
    directly — the same path ``calibrate_native.py`` uses — and maintains an
    up-to-date snapshot of every axis and button.

    The caller still drives pygame's event pump for hot-plug detection and
    rumble; only the axis/button read path changes.
    """

    def __init__(self, n_axes: int, n_btns: int):
        self.n_axes = n_axes
        self.n_btns = n_btns
        self.axes: list[float] = [0.0] * n_axes
        self.buttons: list[bool] = [False] * n_btns
        self._fd: int | None = None
        self._file = None
        self._open()

    def _open(self) -> None:
        """Open js0 non-blocking and drain INIT events."""
        self.close()
        try:
            self._file = open(JS_DEVICE, "rb")
        except (FileNotFoundError, PermissionError):
            self._fd = None
            return
        self._fd = self._file.fileno()
        flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        self._drain_init()

    def _drain_init(self) -> None:
        """Consume INIT events to discover actual axis/button range."""
        if self._file is None:
            return
        while True:
            data = self._file.read(JS_EVENT_SIZE)
            if data is None or len(data) < JS_EVENT_SIZE:
                break
            _t, val, etype, num = struct.unpack(JS_EVENT_FMT, data)
            if etype & JS_EVENT_INIT:
                if etype & JS_EVENT_AXIS and num < self.n_axes:
                    self.axes[num] = val / JS_AXIS_MAX
                elif etype & JS_EVENT_BUTTON and num < self.n_btns:
                    self.buttons[num] = bool(val)

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
            self._fd = None

    @property
    def connected(self) -> bool:
        return self._fd is not None

    def poll(self) -> None:
        """Read all pending kernel events into axes/buttons arrays."""
        if self._file is None:
            return
        try:
            while True:
                data = self._file.read(JS_EVENT_SIZE)
                if data is None or len(data) < JS_EVENT_SIZE:
                    break
                _t, val, etype, num = struct.unpack(JS_EVENT_FMT, data)
                if etype & JS_EVENT_INIT:
                    continue
                if etype & JS_EVENT_AXIS and num < self.n_axes:
                    self.axes[num] = val / JS_AXIS_MAX
                elif etype & JS_EVENT_BUTTON and num < self.n_btns:
                    self.buttons[num] = bool(val)
        except OSError:
            self.close()

    def get_axis(self, idx: int) -> float:
        if idx < self.n_axes:
            return self.axes[idx]
        return 0.0

    def get_button(self, idx: int) -> bool:
        if idx < self.n_btns:
            return self.buttons[idx]
        return False


class GamepadSource:
    """手柄源。name 属性 + poll() 满足 CommandSource Protocol。"""

    name = "gamepad"

    def __init__(self, cfg: ArbiterConfig | None = None):
        self._cfg = cfg or ArbiterConfig()
        init_pygame("KUAFU teleop (gamepad)")
        # 轴/键索引: 默认 Xbox 布局, 环境变量覆盖
        self._axis_v = _env_int("KUAFU_AXIS_V", 1)
        self._axis_w = _env_int("KUAFU_AXIS_W", 2)
        self._axis_lt = _env_int("KUAFU_AXIS_LT", 4)
        self._axis_rt = _env_int("KUAFU_AXIS_RT", 5)
        self._btn_arm = _env_int("KUAFU_BTN_ARM", 7)
        self._btn_disarm = _env_int("KUAFU_BTN_DISARM", 6)
        self._btn_estop = _env_int("KUAFU_BTN_ESTOP", 0)
        self._invert_v = _env_bool("KUAFU_AXIS_V_INVERT", True)
        self._invert_w = _env_bool("KUAFU_AXIS_W_INVERT", False)
        self._invert_lt = _env_bool("KUAFU_AXIS_LT_INVERT", False)
        self._invert_rt = _env_bool("KUAFU_AXIS_RT_INVERT", False)
        self._rumble_enabled = _env_bool("KUAFU_RUMBLE", True)
        # 状态
        self._armed = False
        self._estop_latched = False
        self._d0 = D0_CMD_RANGE[0]
        self._last_poll = time.monotonic()
        self._prev_buttons: dict[int, bool] = {}
        # pygame 句柄(仅用于热插拔和 rumble)
        self._joy: pygame.joystick.Joystick | None = None
        self._joy_instance_id: int | None = None
        self._open_first_joystick()
        # 原生读取器(轴/按钮走 js0, 绕过 SDL 缓存)
        self._native: _NativeJoystick | None = None
        if self._joy is not None:
            self._native = _NativeJoystick(self._joy.get_numaxes(),
                                           self._joy.get_numbuttons())
        else:
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
                self._joy = pygame.joystick.Joystick(
                    ev.instance_id if hasattr(ev, "instance_id") else 0)
                self._joy.init()
                self._joy_instance_id = self._joy.get_instance_id()
                self._native = _NativeJoystick(self._joy.get_numaxes(),
                                               self._joy.get_numbuttons())
                self._rumble("reconnect")
            elif ev.type == pygame.JOYDEVICEREMOVED and self._joy is not None:
                removed_id = (ev.instance_id if hasattr(ev, "instance_id")
                              else self._joy_instance_id)
                if removed_id == self._joy_instance_id:
                    self._joy = None
                    self._joy_instance_id = None
                    if self._native is not None:
                        self._native.close()
                        self._native = None
                    self._armed = False

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
            pass

    # ------------------------------------------------------------------
    # 主轮询
    # ------------------------------------------------------------------
    def poll(self) -> Command | None:
        self._handle_hotplug(pump_events())
        now = time.monotonic()
        dt = now - self._last_poll
        self._last_poll = now

        # 手柄断连 -> ESTOP
        if self._joy is None or self._native is None or not self._native.connected:
            return Command(0.0, 0.0, self._d0, Mode.ESTOP, now)

        # 从原生 js0 刷新最新轴/按钮状态
        self._native.poll()

        # --- 按钮边沿触发(上升沿) ---
        for btn, action in (
            (self._btn_arm, "arm"),
            (self._btn_disarm, "disarm"),
            (self._btn_estop, "estop"),
        ):
            pressed = self._native.get_button(btn)
            if pressed and not self._prev_buttons.get(btn, False):
                self._on_button_edge(action)
            self._prev_buttons[btn] = pressed

        if self._estop_latched:
            return Command(0.0, 0.0, self._d0, Mode.ESTOP, now)

        # --- 摇杆 v/omega ---
        vy = self._read_axis(self._axis_v, self._invert_v)
        wx = self._read_axis(self._axis_w, self._invert_w)
        vy = shape_axis(vy, self._cfg.stick_deadzone, self._cfg.stick_gamma)
        wx = shape_axis(wx, self._cfg.stick_deadzone, self._cfg.stick_gamma)
        v = vy * V_CMD_RANGE[1]
        omega = wx * W_CMD_RANGE[1]

        # --- D0 rate (扳机) ---
        lt = normalize_trigger(self._native.get_axis(self._axis_lt),
                               self._cfg.trigger_deadzone, invert=self._invert_lt)
        rt = normalize_trigger(self._native.get_axis(self._axis_rt),
                               self._cfg.trigger_deadzone, invert=self._invert_rt)
        self._d0 += (rt - lt) * self._cfg.d0_rate_mm_s * dt
        self._d0 = max(D0_CMD_RANGE[0], min(D0_CMD_RANGE[1], self._d0))

        if not self._armed:
            return Command(0.0, 0.0, self._d0, Mode.IDLE, now)
        return Command(v, omega, self._d0, Mode.MANUAL, now)

    def _read_axis(self, axis: int, invert: bool) -> float:
        val = self._native.get_axis(axis) if self._native else 0.0
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


def _calibrate() -> None:
    """交互式手柄标定: 实时显示轴+按钮, 引导逐项确认, 最后输出环境变量。

    用法:
      SDL_VIDEODRIVER=dummy python -m rl.teleop.gamepad_source --calibrate
      (有桌面环境时去掉 SDL_VIDEODRIVER=dummy)

    流程:
      1. 探测手柄型号 / 轴数 / 按钮数
      2. 引导依次推 v 轴(左摇杆 Y)、w 轴(右摇杆 X)、LT、RT
      3. 引导依次按 ARM / DISARM / ESTOP 键
      4. 自动判断 v 轴是否需要反转(pygame Y 向下为正, 推上应输出正值)
      5. 输出可直接 export 的 KUAFU_AXIS_* / KUAFU_BTN_* / KUAFU_AXIS_*_INVERT 行

    也可用 --show-axes 做无引导的实时轴值监视(旧模式, 仅轴)。
    """
    import argparse
    parser = argparse.ArgumentParser(description="KUAFU 手柄标定工具")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--calibrate", action="store_true",
                      help="交互式引导标定(轴+按钮), 输出环境变量")
    mode.add_argument("--show-axes", action="store_true",
                      help="仅实时显示轴值变化(旧模式, 无引导)")
    args = parser.parse_args()

    if args.show_axes:
        _show_axes_live()
        return

    # 默认走 --calibrate (委托给原生标定工具)
    print("调用原生标定工具 (绕过 pygame/SDL)...")
    from rl.teleop import calibrate_native
    calibrate_native.main()


def _show_axes_live() -> None:
    """旧模式: 无引导实时轴值监视(原生 js0 + pygame 双通道对比)。"""
    init_pygame("KUAFU axes")
    if pygame.joystick.get_count() == 0:
        print("未检测到手柄")
        return
    j = pygame.joystick.Joystick(0)
    j.init()
    n = j.get_numaxes()
    n_btns = j.get_numbuttons()
    native = _NativeJoystick(n, n_btns)
    if not native.connected:
        print(f"无法打开 {JS_DEVICE}; 用 pygame 回退模式")
        native = None

    print(f"手柄: {j.get_name()} ({n} 轴, {n_btns} 钮)")
    print("逐个操作摇杆/扳机/按钮, 看哪个编号响应。Ctrl-C 退出。")
    print(f"  源: native={JS_DEVICE}" + (" + pygame" if native else " (pygame only)"))
    print("-" * 55)

    prev_axes = [native.get_axis(i) if native else j.get_axis(i) for i in range(n)]
    prev_btns = [native.get_button(i) if native else j.get_button(i) for i in range(n_btns)]
    try:
        while True:
            pump_events()
            if native is not None:
                native.poll()
            for i in range(n):
                v = native.get_axis(i) if native else j.get_axis(i)
                if abs(v - prev_axes[i]) > 0.15:
                    print(f"  轴{i}: {prev_axes[i]:+.2f} -> {v:+.2f}")
                    prev_axes[i] = v
            for i in range(n_btns):
                v = native.get_button(i) if native else j.get_button(i)
                if v != prev_btns[i]:
                    print(f"  按钮{i}: {'按下' if v else '松开'}")
                    prev_btns[i] = v
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n退出。据上面的输出设置环境变量, 例如:")
        print("  export KUAFU_AXIS_V=1 KUAFU_AXIS_W=3 KUAFU_AXIS_LT=4 KUAFU_AXIS_RT=5")
        print("  export KUAFU_BTN_ARM=7 KUAFU_BTN_DISARM=6 KUAFU_BTN_ESTOP=0")
    finally:
        if native is not None:
            native.close()


if __name__ == "__main__":
    _calibrate()
