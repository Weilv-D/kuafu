# -*- coding: utf-8 -*-
"""
IPCCommandSource - 把 Unix socket 收到的手柄命令包装成 CommandSource。

serial_node 进程内持有一个 IPCCommandSource 实例, 与 AutonomousSource 一起喂给
CommandArbiter。仲裁器的安全层(急停/抢占/ramp/限幅/超时降级)就在 serial_node
进程内, 紧贴执行器; 因此即使 teleop 进程崩溃或蓝牙断连, IPCCommandSource 的
poll() 返回 None -> 仲裁器判源失效(stale_time 内无新鲜命令) -> 输出安全默认
[0, 0, D0_MIN] + ESTOP, 机器人停住。

teleop 进程只发原始手柄命令(mode=MANUAL 的 v/omega/d0, 或 mode=ESTOP 急停);
它不碰仲裁器、不碰策略、不碰串口。
"""
from __future__ import annotations

import os
import sys
import time

# 复用项目 IPC 协议(纯标准库), 与 teleop_node 进程共享 wire 格式。
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from pi5_runtime.command_socket import COMMAND_SOCKET_PATH, CommandSocketServer

from rl.teleop.command import Command, Mode

# wire mode(firmware RobotMode) -> teleop Mode 的完整映射。
# 0=INIT, 1=STAND, 2=ACTIVE, 3=CLIMB, 4=FAULT。INIT/FAULT 降级为 ESTOP;
# STAND 透传为 IDLE(保平衡不跟走); ACTIVE/CLIMB 透传为 MANUAL(出力)。
_WIRE_TO_MODE = {
    0: Mode.ESTOP,
    1: Mode.IDLE,
    2: Mode.MANUAL,
    3: Mode.MANUAL,
    4: Mode.ESTOP,
}


class IPCCommandSource:
    """Unix socket 命令源。name 属性 + poll() 满足 CommandSource Protocol。"""

    name = "ipc"

    def __init__(self, path: str = COMMAND_SOCKET_PATH) -> None:
        self._server = CommandSocketServer(path)
        self._server.bind()

    def poll(self) -> Command | None:
        """拉最新一帧 socket 命令, 转成 Command; 无数据返回 None。

        返回 None 时仲裁器会按 stale_time 判该源失效并降级到安全默认,
        所以这里不需要自己再做超时判断——超时语义集中在 CommandArbiter。
        """
        frame = self._server.recv_command()
        if frame is None:
            return None
        # wire mode 完整透传: 0/4=ESTOP, 1=IDLE(STAND), 2/3=MANUAL(ACTIVE/CLIMB)。
        # 未知值降级为 ESTOP(安全侧失败)。
        mode = _WIRE_TO_MODE.get(int(frame["mode"]), Mode.ESTOP)
        # stamp 来自发送方 time.monotonic(); 与接收方同一时钟域(Pi5 单机),
        # 仲裁器用 stamp 判新鲜度。若跨机部署需改用对齐时间戳, 但本设计是同机双进程。
        return Command(
            v=float(frame["v"]),
            omega=float(frame["omega"]),
            d0=float(frame["d0"]),
            mode=mode,
            stamp=float(frame["stamp"]),
        )

    def close(self) -> None:
        self._server.close()
