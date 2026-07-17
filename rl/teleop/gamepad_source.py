# -*- coding: utf-8 -*-
"""
GamepadSource - 手柄命令源 (纯原生 js0, 不依赖 pygame Joystick)

两态使能模型(显式 ARM/DISARM, 启动默认 DISARMED 安全态):
  START / arm 键   -> ARMED  : 发 Mode.MANUAL(ACTIVE), 轮出力+RL残差, 摇杆跟踪
  Back / disarm 键 -> DISARMED: 发 Mode.IDLE(STAND), LQR 保平衡但轮不跟走, RL残差关
  A / estop 键     -> ESTOP 锁存(轮失能, 需 ARM 解除)

轴/按钮读取完全走原生 /dev/input/js0 内核设备。pygame 不再打开 Joystick 对象
(SDL 的 event pump 会和 js0 read 争抢内核事件导致丢按钮)。pygame 仅用于初始化
display (teleop_node 需要) 和 best-effort rumble (通过 evdev EV_FF 或跳过)。

热插拔通过 os.path.exists("/dev/input/js0") 轮询检测, 无需 SDL 事件。

轴映射默认 Xbox 布局, 不同手柄用环境变量覆盖:
  KUAFU_AXIS_V    左摇杆 Y (v 前后)         默认 1
  KUAFU_AXIS_W    右摇杆 X (ω 转向)         默认 2
  KUAFU_AXIS_LT   LT 扳机 (蹲)              默认 4
  KUAFU_AXIS_RT   RT 扳机 (站)              默认 5
  KUAFU_AXIS_V_INVERT  反转 v 轴(默认 1)    大多数手柄 Y 向下为正
  KUAFU_AXIS_W_INVERT  反转 ω 轴(默认 0)
  KUAFU_AXIS_LT_INVERT 反转 LT 扳机(默认 0)
  KUAFU_AXIS_RT_INVERT 反转 RT 扳机(默认 0)
  KUAFU_BTN_ARM   使能键(默认 7=START)
  KUAFU_BTN_DISARM 卸能键(默认 6=Back)
  KUAFU_BTN_ESTOP  急停键(默认 0=A)
  KUAFU_JS_DEVICE  js 设备路径(默认 /dev/input/js0)

标定: python -m rl.teleop.gamepad_source --calibrate
"""
from __future__ import annotations

import fcntl
import os
import struct
import time

from rl.teleop.command import (
    ArbiterConfig, Command, Mode, V_CMD_RANGE, W_CMD_RANGE, D0_CMD_RANGE,
)
from rl.teleop.shaping import normalize_trigger, shape_axis

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


class _NativeJoystick:
    """Read /dev/input/js0 directly — no pygame/SDL involvement.

    Opens the kernel joystick device in non-blocking mode, drains INIT events
    to discover axis/button layout, and maintains an up-to-date snapshot.

    Hot-plug: caller checks ``connected`` each poll; if the fd is lost,
    ``reconnect()`` attempts to re-open.
    """

    def __init__(self, device: str = "/dev/input/js0", n_axes: int = 9, n_btns: int = 18):
        self.device = device
        self.n_axes = n_axes
        self.n_btns = n_btns
        self.axes: list[float] = [0.0] * n_axes
        self.buttons: list[bool] = [False] * n_btns
        self._file = None
        self._fd: int | None = None
        self._event_count = 0          # total real-time events since open
        self._last_event_time = 0.0    # monotonic time of last real event
        self._open()

    def _open(self) -> bool:
        """Open js device non-blocking, drain INIT. Return True on success."""
        self.close()
        self._event_count = 0
        self._last_event_time = time.monotonic()
        try:
            self._file = open(self.device, "rb")
        except (FileNotFoundError, PermissionError, OSError):
            self._fd = None
            return False
        self._fd = self._file.fileno()
        flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        self._drain_init()
        return True

    def _drain_init(self) -> None:
        """Consume INIT events to get axis/button layout and resting values."""
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

    def reconnect(self) -> bool:
        """Attempt to re-open after disconnect. Return True if reconnected."""
        return self._open()

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
                self._event_count += 1
                self._last_event_time = time.monotonic()
                if etype & JS_EVENT_AXIS and num < self.n_axes:
                    self.axes[num] = val / JS_AXIS_MAX
                elif etype & JS_EVENT_BUTTON and num < self.n_btns:
                    self.buttons[num] = bool(val)
        except OSError:
            self.close()

    @property
    def is_idle(self) -> bool:
        """True if the gamepad hasn't sent any real-time events since open.

        Indicates the BLE gamepad is in low-power idle and needs physical
        operation (stick/button press) to wake its HID report stream.
        """
        return self._event_count == 0

    @property
    def idle_seconds(self) -> float:
        """Seconds since the last real-time event (or since open if none yet)."""
        return time.monotonic() - self._last_event_time

    def get_axis(self, idx: int) -> float:
        if 0 <= idx < self.n_axes:
            return self.axes[idx]
        return 0.0

    def get_button(self, idx: int) -> bool:
        if 0 <= idx < self.n_btns:
            return self.buttons[idx]
        return False


class GamepadSource:
    """手柄源。name 属性 + poll() 满足 CommandSource Protocol。

    完全基于原生 /dev/input/js0, 不创建 pygame.joystick.Joystick 对象。
    """

    name = "gamepad"

    def __init__(self, cfg: ArbiterConfig | None = None):
        self._cfg = cfg or ArbiterConfig()
        # 轴/键索引
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
        self._js_device = os.environ.get("KUAFU_JS_DEVICE", "/dev/input/js0")
        # 状态
        self._armed = False
        self._estop_latched = False
        self._d0 = D0_CMD_RANGE[0]
        self._last_poll = time.monotonic()
        self._prev_buttons: dict[int, bool] = {}
        # 休眠守护
        self._bt_mac = os.environ.get("KUAFU_BT_MAC", "")
        self._idle_warn_interval = float(os.environ.get("KUAFU_IDLE_WARN_INTERVAL", "5"))
        self._idle_reconnect_threshold = float(os.environ.get("KUAFU_IDLE_RECONNECT", "15"))
        self._last_idle_warn = 0.0
        self._last_reconnect = 0.0
        self._was_idle = False
        # 原生手柄读取器
        self._native: _NativeJoystick | None = None
        self._native_connected_time = 0.0
        self._try_connect()
        if self._native is not None and self._native.connected:
            print(f"[gamepad] connected via {self._js_device}")
        else:
            print(f"[gamepad] no joystick at {self._js_device}; "
                  "waiting for hot-plug (poll returns ESTOP)")

    def _try_connect(self) -> None:
        """Attempt to open the joystick device."""
        if self._native is not None:
            self._native.close()
        self._native = _NativeJoystick(self._js_device)
        if not self._native.connected:
            self._native = None

    # ------------------------------------------------------------------
    # 主轮询
    # ------------------------------------------------------------------
    def poll(self) -> Command | None:
        now = time.monotonic()
        dt = now - self._last_poll
        self._last_poll = now

        # 热插拔检测: 设备不存在或 fd 丢失时尝试重连
        if self._native is None or not self._native.connected:
            # 每 0.5s 尝试重连一次
            if now - self._native_connected_time > 0.5:
                self._try_connect()
                if self._native is not None and self._native.connected:
                    self._native_connected_time = now
                    self._armed = False
                    print(f"[gamepad] reconnected via {self._js_device}")
            if self._native is None or not self._native.connected:
                return Command(0.0, 0.0, self._d0, Mode.ESTOP, now)

        # 从原生 js0 刷新最新轴/按钮状态
        self._native.poll()

        # 检查 fd 是否在 poll 中丢失
        if not self._native.connected:
            self._armed = False
            return Command(0.0, 0.0, self._d0, Mode.ESTOP, now)

        # --- BLE 休眠守护 ---
        # VADER2P 等蓝牙手柄连接后可能进入低功耗待机，不发 HID report。
        # 检测到长时间无事件时提醒操作者物理唤醒，并可选自动重连。
        self._check_idle(now)

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

    def _check_idle(self, now: float) -> None:
        """Detect BLE gamepad idle (no HID reports) and warn / reconnect.

        Flydigi VADER2P and similar BLE controllers enter a low-power idle
        after connection and only resume sending HID reports when physically
        operated. This method prints periodic reminders and optionally triggers
        a bluetoothctl reconnect when the silence exceeds a threshold.
        """
        if self._native is None or not self._native.connected:
            return

        idle = self._native.idle_seconds

        # Transition: was receiving events → went idle
        if self._native.is_idle and not self._was_idle:
            print(f"[gamepad] ⚠️  no input events since connect — "
                  f"BLE gamepad may be asleep. Push a stick to wake it.")
            self._was_idle = True
            self._last_idle_warn = now

        # Periodic reminder while idle
        elif self._native.is_idle and now - self._last_idle_warn > self._idle_warn_interval:
            print(f"[gamepad] still idle ({idle:.0f}s) — push a stick/button to wake")
            self._last_idle_warn = now

        # Auto-reconnect attempt when idle too long
        if (self._native.is_idle and self._bt_mac
                and idle > self._idle_reconnect_threshold
                and now - self._last_reconnect > 15.0):
            print(f"[gamepad] idle {idle:.0f}s > {self._idle_reconnect_threshold:.0f}s; "
                  f"reconnecting {self._bt_mac} ...")
            from rl.teleop.bt_wakeup import bt_reconnect
            bt_reconnect(self._bt_mac)
            self._last_reconnect = now
            self._try_connect()

        # Transition: idle → receiving events
        if not self._native.is_idle and self._was_idle:
            print("[gamepad] ✅ awake — events flowing")
            self._was_idle = False

    def _on_button_edge(self, action: str) -> None:
        if action == "arm":
            self._armed = True
            self._estop_latched = False
            print("[gamepad] ARMED")
        elif action == "disarm":
            self._armed = False
            print("[gamepad] DISARMED")
        elif action == "estop":
            self._armed = False
            self._estop_latched = True
            print("[gamepad] ESTOP latched")

    # ------------------------------------------------------------------
    # 调试辅助
    # ------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._native is not None and self._native.connected

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def _joy(self):
        """Compatibility stub for teleop_node detection (returns None).

        teleop_node checks ``src._joy`` to print device info. With the native
        reader there is no pygame Joystick object, so return a lightweight
        dummy with get_name().
        """
        if self._native is not None and self._native.connected:
            return _DummyJoy()
        return None


class _DummyJoy:
    """Minimal stand-in for pygame.joystick.Joystick for teleop_node logging."""
    def get_name(self) -> str:
        return os.environ.get("KUAFU_JS_DEVICE", "/dev/input/js0")

    def get_numaxes(self) -> int:
        return 9

    def get_numbuttons(self) -> int:
        return 18


def _calibrate() -> None:
    """Delegate to the native calibration tool."""
    from rl.teleop import calibrate_native
    calibrate_native.main()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KUAFU 手柄工具")
    parser.add_argument("--calibrate", action="store_true",
                        help="交互式引导标定")
    parser.add_argument("--show-axes", action="store_true",
                        help="实时显示轴值变化")
    args = parser.parse_args()
    if args.show_axes:
        _show_axes_live()
    else:
        _calibrate()


def _show_axes_live() -> None:
    """无引导实时轴/按钮值监视 (纯原生 js0)。"""
    device = os.environ.get("KUAFU_JS_DEVICE", "/dev/input/js0")
    nj = _NativeJoystick(device)
    if not nj.connected:
        print(f"未检测到手柄 ({device})")
        return
    print(f"手柄: {device} ({nj.n_axes} 轴, {nj.n_btns} 钮)")
    print("逐个操作摇杆/扳机/按钮, 看哪个编号响应。Ctrl-C 退出。")
    print("-" * 55)
    prev_axes = list(nj.axes)
    prev_btns = list(nj.buttons)
    try:
        while True:
            nj.poll()
            for i in range(nj.n_axes):
                v = nj.get_axis(i)
                if abs(v - prev_axes[i]) > 0.15:
                    print(f"  轴{i}: {prev_axes[i]:+.2f} -> {v:+.2f}")
                    prev_axes[i] = v
            for i in range(nj.n_btns):
                v = nj.get_button(i)
                if v != prev_btns[i]:
                    print(f"  按钮{i}: {'按下' if v else '松开'}")
                    prev_btns[i] = v
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n退出。")
    finally:
        nj.close()
