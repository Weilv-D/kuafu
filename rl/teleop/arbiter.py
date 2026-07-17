# -*- coding: utf-8 -*-
"""
CommandArbiter - 多源命令仲裁 + 安全层

仲裁规则(手柄抢占式半自动语义):

  1. 急停最高优先级: 任一源 mode=ESTOP -> 立即输出 [0,0,d0_cur] 并锁定
  2. 手柄抢占:       手柄源 |v| 或 |omega| 超死区 -> MANUAL 抢占, 自主挂起
  3. 交还自主:       手柄归零持续 handoff_time 且自主源活跃 -> 切 AUTONOMOUS
   4. 平滑:          输出对目标做 ramp_time 一阶低通(模式切换 / 手柄 idle↔active / 急停均平滑, 防突跳摔机)
  5. 限幅:           clip 到 V/W/D0 CMD_RANGE
  6. 超时降级:       源 stamp 老于 stale_time -> 该源失效
  7. 全无源:         输出安全默认 [0,0,D0_MIN] + ESTOP

策略不碰, LQR 不碰; 仲裁器只决定"喂给策略的 command 从哪来、怎么平滑"。
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

from rl.teleop.command import (
    Command, Mode, CommandSource, ArbiterConfig, D0_CMD_RANGE,
)

# 物理真源 (D0 高速门控阈值)
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
import kuafu_physics as _P


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class CommandArbiter:
    """聚合多个 CommandSource, 按优先级 + 抢占逻辑输出单一 Command。

    用法:
        arb = CommandArbiter([gamepad, autonomous], cfg)
        cmd = arb.poll()   # 每个 50Hz 周期调用一次
        base_obs = _build_obs(..., cmd.as_array())

    手柄源应排在 sources 列表前面(优先级靠手动源在前); AutonomousSource 排其后。
    """

    def __init__(self, sources: list[CommandSource], cfg: ArbiterConfig | None = None):
        if not sources:
            raise ValueError("CommandArbiter 至少需要一个 CommandSource")
        self.sources = sources
        self.cfg = cfg or ArbiterConfig()

        # --- 内部状态 ---
        self._mode: Mode = Mode.ESTOP          # 启动即急停, 等首个有效命令
        self._estop_locked: bool = True        # 急停锁定标志
        self._manual_idle_since: float | None = None  # 手柄归零计时起点
        self._last_output: Command = _safe_default()  # 上一步输出(低通起点)
        self._last_cmd_time: float = time.monotonic()

    # ------------------------------------------------------------
    # 仲裁主入口
    # ------------------------------------------------------------
    def poll(self) -> Command:
        """聚合所有源, 返回本周期要喂给策略的安全命令。50Hz 调用。"""
        now = time.monotonic()
        cfg = self.cfg

        # 每周期对每个 source 仅 poll 一次并缓存。原因: poll() 可能有副作用
        # (KeyboardSource/GamepadSource 的 poll() 会按 dt 积分 self._d0 并 pump_events),
        # 若每源被调多次(原 _poll_source_of_mode×2 + _any_estop 各扫一遍 = 3 次),
        # d0 会以 3× 速率变化、事件被 pump 3 次。缓存后只 poll 一次。
        cached: list[Command] = []
        for src in self.sources:
            cmd = src.poll()
            if cmd is not None and now - cmd.stamp <= cfg.stale_time:
                cached.append(cmd)

        manual_cmd = next((c for c in cached if c.mode == Mode.MANUAL), None)
        idle_cmd = next((c for c in cached if c.mode == Mode.IDLE), None)
        auto_cmd = next((c for c in cached if c.mode == Mode.AUTONOMOUS), None)
        estop = any(c.mode == Mode.ESTOP for c in cached)

        # --- 规则 1: 急停最高优先级 (速度平滑归零, 保持当前 d0) ---
        if estop:
            self._enter_estop(now)
            target = Command(0.0, 0.0, self._last_output.d0, Mode.ESTOP, now)
            return self._emit(target, now)

        # 若当前锁定在急停, 需要一个有效源来解锁
        if self._estop_locked:
            if manual_cmd is None and auto_cmd is None and idle_cmd is None:
                target = Command(0.0, 0.0, self._last_output.d0, Mode.ESTOP, now)
                return self._emit(target, now)  # 仍无有效源, 保持急停
            self._estop_locked = False    # 解锁

        # --- 规则 2/3: 手柄抢占式 ---
        manual_active = (manual_cmd is not None and self._is_manual_active(manual_cmd))
        if manual_active:
            # 人在操控 -> MANUAL 抢占
            self._manual_idle_since = None
            self._switch_mode(Mode.MANUAL, now)
            target = manual_cmd
        elif idle_cmd is not None:
            # 操作者显式 DISARMED(按 disarm 键) -> 请求 STAND 保平衡但不跟走。
            # 优先级低于 manual_active(摇杆一动即重新 ARMED), 高于 autonomous。
            # GamepadSource 在 DISARMED 时发 IDLE 且 v=w=0, 故不会与 manual_active 冲突。
            self._manual_idle_since = None
            self._switch_mode(Mode.IDLE, now)
            target = Command(0.0, 0.0, idle_cmd.d0, Mode.IDLE, now)
        elif auto_cmd is not None:
            # 手柄未活跃(松手/死区内): handoff_time 内保持 MANUAL 给人反应时间,
            # 超时后交还自主
            if self._mode == Mode.MANUAL and self._manual_idle_since is None:
                self._manual_idle_since = now
            if (self._mode == Mode.MANUAL
                    and self._manual_idle_since is not None
                    and now - self._manual_idle_since < cfg.handoff_time):
                # handoff 内: 维持手柄态, 速度归零, 保持 d0
                d0 = manual_cmd.d0 if manual_cmd is not None else self._last_output.d0
                target = Command(0.0, 0.0, d0, Mode.MANUAL, now)
            else:
                self._switch_mode(Mode.AUTONOMOUS, now)
                target = auto_cmd
        elif manual_cmd is not None:
            # 手柄在死区内(松手但源未失效), 无自主源: 保持当前 d0, 速度归零
            self._switch_mode(Mode.MANUAL, now)
            target = Command(0.0, 0.0, manual_cmd.d0, Mode.MANUAL, now)
        else:
            # --- 规则 7: 全无源 -> 安全默认 + ESTOP ---
            self._enter_estop(now)
            target = Command(0.0, 0.0, self._last_output.d0, Mode.ESTOP, now)
            return self._emit(target, now)

        # --- 规则 4: 平滑 (一阶低通) + 规则 5: 限幅 ---
        return self._emit(target, now)

    def _emit(self, target: Command, now: float) -> Command:
        """低通平滑 + 限幅, 更新内部状态并返回。"""
        out = self._smooth(target, now)
        out = self._clamp_cmd(out)
        self._last_output = out
        self._last_cmd_time = now
        return out

    # ------------------------------------------------------------
    # 状态切换 + 平滑
    # ------------------------------------------------------------
    def _switch_mode(self, new_mode: Mode, now: float) -> None:
        self._mode = new_mode

    def _enter_estop(self, now: float) -> None:
        self._mode = Mode.ESTOP
        self._estop_locked = True

    def _smooth(self, target: Command, now: float) -> Command:
        """一阶低通: 输出从上一周期输出向 target 过渡, 时间常数 ramp_time (防突跳)。

        对模式切换 / 手柄 idle↔active / 急停 统一平滑, 避免速度/蹲起量瞬间跳变摔机。
        """
        cfg = self.cfg
        # 限幅 dt: 首帧(_last_cmd_time=0) 或长间隔(调度抖动/GC暂停)会让 dt 很大,
        # alpha 饱和到 1.0 使输出一步跳到 target, 击穿 ramp 保护。限制到
        # max_smoothing_dt(默认 0.1s) 保证任意单步过渡不快于 ramp_time 的最小粒度。
        dt = min(now - self._last_cmd_time, cfg.max_smoothing_dt)
        alpha = min(1.0, dt / cfg.ramp_time) if cfg.ramp_time > 0 else 1.0
        v = self._last_output.v + (target.v - self._last_output.v) * alpha
        w = self._last_output.omega + (target.omega - self._last_output.omega) * alpha
        d0 = self._last_output.d0 + (target.d0 - self._last_output.d0) * alpha
        return Command(v, w, d0, target.mode, now)

    def _is_manual_active(self, cmd: Command) -> bool:
        """手柄是否在死区外(人在操控)。"""
        cfg = self.cfg
        return (abs(cmd.v) > cfg.manual_deadzone_v
                or abs(cmd.omega) > cfg.manual_deadzone_w)

    def _clamp_cmd(self, cmd: Command) -> Command:
        cfg = self.cfg
        v = _clamp(cmd.v, cfg.v_limit[0], cfg.v_limit[1])
        w = _clamp(cmd.omega, cfg.w_limit[0], cfg.w_limit[1])
        # D0 高速门控: |v| 或 |ω| 超阈值时限制 D0_max (防高速伸腿抬 COM topple)
        d0_max = cfg.d0_limit[1]
        if abs(v) > _P.D0_GATE_V_THRESH or abs(w) > _P.D0_GATE_W_THRESH:
            d0_max = _P.D0_GATE_MAX_HIGH
        d0 = _clamp(cmd.d0, cfg.d0_limit[0], d0_max)
        return Command(v, w, d0, cmd.mode, cmd.stamp)

    # ------------------------------------------------------------
    # 状态查询(供 HUD 显示)
    # ------------------------------------------------------------
    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def estop_locked(self) -> bool:
        return self._estop_locked


def _safe_default() -> Command:
    """安全默认命令: 零速 + 驻留态 + ESTOP。"""
    return Command(0.0, 0.0, D0_CMD_RANGE[0], Mode.ESTOP)
