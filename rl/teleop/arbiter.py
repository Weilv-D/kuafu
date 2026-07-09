# -*- coding: utf-8 -*-
"""
CommandArbiter - 多源命令仲裁 + 安全层

仲裁规则(手柄抢占式半自动语义):

  1. 急停最高优先级: 任一源 mode=ESTOP -> 立即输出 [0,0,d0_cur] 并锁定
  2. 手柄抢占:       手柄源 |v| 或 |omega| 超死区 -> MANUAL 抢占, 自主挂起
  3. 交还自主:       手柄归零持续 handoff_time 且自主源活跃 -> 切 AUTONOMOUS
  4. ramp 平滑:      模式切换瞬间, 输出向新源目标 ramp_time 线性过渡(防突跳摔机)
  5. 限幅:           clip 到 V/W/D0 CMD_RANGE
  6. 超时降级:       源 stamp 老于 stale_time -> 该源失效
  7. 全无源:         输出安全默认 [0,0,D0_MIN] + ESTOP

策略不碰, LQR 不碰; 仲裁器只决定"喂给策略的 command 从哪来、怎么平滑"。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from rl.teleop.command import (
    Command, Mode, CommandSource, ArbiterConfig, D0_CMD_RANGE,
)


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
        self._last_output: Command = _safe_default()  # 上一步输出(ramp 起点)
        self._ramp_t0: float | None = None     # ramp 开始时刻; None=不在 ramp
        self._ramp_from: Command = _safe_default()
        self._last_cmd_time: float = time.monotonic()

    # ------------------------------------------------------------
    # 仲裁主入口
    # ------------------------------------------------------------
    def poll(self) -> Command:
        """聚合所有源, 返回本周期要喂给策略的安全命令。50Hz 调用。"""
        now = time.monotonic()
        cfg = self.cfg

        # 收集所有源的当前命令(过滤超时源)
        manual_cmd = self._poll_source_of_mode(Mode.MANUAL, now)
        auto_cmd = self._poll_source_of_mode(Mode.AUTONOMOUS, now)
        estop = self._any_estop(now)

        # --- 规则 1: 急停最高优先级 ---
        if estop:
            self._enter_estop(now)
            return self._last_output

        # 若当前锁定在急停, 需要一个有效源来解锁
        if self._estop_locked:
            if manual_cmd is None and auto_cmd is None:
                return self._last_output  # 仍无有效源, 保持急停
            self._estop_locked = False    # 解锁

        # --- 规则 2/3: 手柄抢占式 ---
        manual_active = (manual_cmd is not None and self._is_manual_active(manual_cmd))
        if manual_active:
            # 人在操控 -> MANUAL 抢占
            self._manual_idle_since = None
            self._switch_mode(Mode.MANUAL, now)
            target = manual_cmd
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
            return self._last_output

        # --- 规则 4: ramp 平滑 ---
        out = self._apply_ramp(target, now)
        # --- 规则 5: 限幅 ---
        out = self._clamp_cmd(out)
        self._last_output = out
        self._last_cmd_time = now
        return out

    # ------------------------------------------------------------
    # 源采集
    # ------------------------------------------------------------
    def _poll_source_of_mode(self, want: Mode, now: float) -> Command | None:
        """取指定 mode 的最新有效(未超时)命令, 取第一个命中的源。"""
        cfg = self.cfg
        for src in self.sources:
            cmd = src.poll()
            if cmd is None:
                continue
            if now - cmd.stamp > cfg.stale_time:
                continue  # 该源数据陈旧
            if cmd.mode == want:
                return cmd
        return None

    def _any_estop(self, now: float) -> bool:
        cfg = self.cfg
        for src in self.sources:
            cmd = src.poll()
            if cmd is not None and cmd.mode == Mode.ESTOP \
                    and now - cmd.stamp <= cfg.stale_time:
                return True
        return False

    # ------------------------------------------------------------
    # 状态切换 + ramp
    # ------------------------------------------------------------
    def _switch_mode(self, new_mode: Mode, now: float) -> None:
        if self._mode != new_mode:
            # 切换: 启动 ramp, 从当前输出平滑到新源目标
            self._ramp_from = self._last_output
            self._ramp_t0 = now
            self._mode = new_mode

    def _enter_estop(self, now: float) -> None:
        self._mode = Mode.ESTOP
        self._estop_locked = True
        # 急停也走 ramp, 防止速度瞬间归零导致平衡失稳
        self._ramp_from = self._last_output
        self._ramp_t0 = now
        # 急停目标: 速度归零, 保持当前 d0
        self._last_output = Command(0.0, 0.0, self._last_output.d0, Mode.ESTOP, now)

    def _apply_ramp(self, target: Command, now: float) -> Command:
        """从 _ramp_from 向 target 线性插值, ramp 期内抑制突变。"""
        if self._ramp_t0 is None:
            return target
        cfg = self.cfg
        alpha = min(1.0, (now - self._ramp_t0) / cfg.ramp_time) if cfg.ramp_time > 0 else 1.0
        v = self._ramp_from.v + (target.v - self._ramp_from.v) * alpha
        w = self._ramp_from.omega + (target.omega - self._ramp_from.omega) * alpha
        d0 = self._ramp_from.d0 + (target.d0 - self._ramp_from.d0) * alpha
        if alpha >= 1.0:
            self._ramp_t0 = None  # ramp 结束
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
        d0 = _clamp(cmd.d0, cfg.d0_limit[0], cfg.d0_limit[1])
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
