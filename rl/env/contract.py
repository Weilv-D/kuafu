# -*- coding: utf-8 -*-
"""
KUAFU 端到端契约 — frame / unit / sign / obs / action / protocol 单一真源

本模块是所有层（MJX 训练、原生 MuJoCo 评估、STM32 固件、UART 协议、Pi5 runtime、
ONNX 导出）必须引用的**版本化契约**。任何跨层不一致（单位、符号、维度、协议）都先
在本模块机器可验证地定义，再由各层遵守。

设计原则：
- 所有坐标/符号约定集中定义，禁止在各层各自硬编码方向。
- 观测/动作/命令/传感器/协议的维度与单位都在此声明，并由测试强制。
- 版本号随破坏兼容的修改递增；旧 checkpoint 标 legacy-v0（见 AGENTS.md）。

单位约定：长度 m（除非注明 mm）、质量 kg、力 N、力矩 N·m、角度 rad、时间 s。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

# ============================================================
# 契约版本
# ============================================================
# 破坏性修改（维度/符号/单位/协议变化）时递增次版本号或主版本号。
SCHEMA_VERSION = "v1.1.0"

# ============================================================
# 坐标系 / 符号约定（单一真源，禁止各层各自推导）
# ============================================================
# 机身坐标系：原点在髋面中心（两髋点 A-B 中点），+X 前、+Y 左、+Z 上。
# 这是文档（KUAFU.md / design.md）与各层物理推导的共同前提。
FORWARD_AXIS = "+X"   # 前进方向
LEFT_AXIS = "+Y"      # 左向（+Y 在 +X 左侧）
UP_AXIS = "+Z"        # 上向

# 轮扭矩 → 运动映射（紧凑差速轮式，左右为对称两轮）：
#   前向（pitch 平衡）每轮等扭矩：tau_pitch = (tau_L + tau_R) / 2
#   偏航（yaw）差速：          tau_yaw   = (tau_R - tau_L) / 2
# 由右手系（+Z 上，+X→+Y 为 CCW）推导：
#   两轮同向前扭矩 → +vx（前进）。
#   右轮 > 左轮（tau_R > tau_L）→ tau_yaw > 0 → +wz（左转）。
#
# 关键护栏（由测试强制）：
#   - 同向正轮扭矩 ⇒ +vx
#   - 右轮扭矩 > 左轮 ⇒ +wz
WHEEL_LEFT = "L"   # 左轮（+Y 侧）
WHEEL_RIGHT = "R"  # 右轮（-Y 侧）

# LQR 状态/输入符号约定（倒立摆线性化）
#   状态 x_state = [x, theta, xdot, thetadot]
#     x      : 机身水平位移 (m)，+X 前
#     theta  : 俯仰角 (rad)，+theta 为机身前倾（顶部向 -X 倒）
#     xdot   : 水平速度 (m/s)
#     thetadot: 俯仰角速度 (rad/s)
#   输入 F    : 地面作用于轮的水平力 (N)，+F ⇒ +x 加速度（前进）
#   轮扭矩与 F： F = (tau_L + tau_R) / R_wheel   （R_wheel 为轮半径 m）
#   反馈律： F = -K · e，e = [x - x_ref, theta, xdot - xdot_ref, thetadot]
LQR_STATE_NAMES = ("x", "theta", "xdot", "thetadot")
LQR_INPUT_NAME = "F"


def tau_pitch_from_wheels(tau_l: float, tau_r: float) -> float:
    """前向（平衡）分量：两轮等扭矩均值。"""
    return (tau_l + tau_r) / 2.0


def tau_yaw_from_wheels(tau_l: float, tau_r: float) -> float:
    """偏航分量：右减左。tau_r > tau_l ⇒ +wz（左转）。"""
    return (tau_r - tau_l) / 2.0


def wheels_from_tau(tau_pitch: float, tau_yaw: float) -> Tuple[float, float]:
    """逆映射：给定 pitch/yaw 扭矩分量，解出左右轮扭矩。"""
    tau_l = tau_pitch - tau_yaw
    tau_r = tau_pitch + tau_yaw
    return tau_l, tau_r


# ============================================================
# 命令契约（高层 → 基层 / RL）
# ============================================================
@dataclass
class CommandSpec:
    """高层命令（Pi5 → STM32 / Actor）。单位已在字段注明。"""
    vx: float          # 前向速度命令 m/s，范围 [-0.5, 0.5]
    wz: float          # 偏航角速度命令 rad/s，范围 [-1.0, 1.0]
    d0: float          # 目标 D0（足端下垂）mm，范围 [58, 207]；低速短时全行程，高速门控到 120
    vx_range: Tuple[float, float] = (-0.5, 0.5)
    wz_range: Tuple[float, float] = (-1.0, 1.0)
    d0_range: Tuple[float, float] = (58.0, 207.0)
    d0_high_speed_max: float = 120.0  # 高速时 D0 上限 mm


# ============================================================
# 观测契约（Actor 输入，纯本体感受，可由实机获得）
# ============================================================
# Actor 禁用：root 真值速度、绝对 yaw、无限轮角、仿真真接触、特权 DR 参数。
# 前向速度用轮速/IMU 估计；姿态用 projected gravity + body gyro；含命令、传感器年龄、
# 上一帧实际动作；可选接触估计（电流/轮速估计器，非 MuJoCo contact）。
OBS_FIELDS: List[Tuple[str, int, str, str]] = [
    # (name, dim, unit, note)
    ("command_vx", 1, "m/s", "当前前向速度命令（已限幅）"),
    ("command_wz", 1, "rad/s", "当前偏航命令"),
    ("command_d0", 1, "mm", "当前 D0 命令（已门控）"),
    ("proj_gravity", 3, "1", "机体坐标系重力投影 (gx,gy,gz)"),
    ("body_gyro", 3, "rad/s", "机体角速度 (wx,wy,wz)"),
    ("est_vx", 1, "m/s", "轮速里程计估计的前向速度（非仿真 root truth）"),
    ("est_wz", 1, "rad/s", "估计偏航角速度"),
    ("est_d0", 1, "mm", "估计平均 D0"),
    ("est_roll", 1, "rad", "估计 roll"),
    ("wheel_speed", 2, "rad/s", "左右轮角速度（硬件可得）"),
    ("hip_pos", 4, "rad", "4 舵机位置（A_l,B_l,A_r,B_r）"),
    ("hip_vel", 4, "rad/s", "4 舵机速度"),
    ("prev_applied_action", 6, "1", "上一帧实际施加的 6 维动作（延迟后）"),
    ("sensor_age", 6, "ms", "IMU age(3) + joint age(3)"),
]


def obs_dim(base_per_step: int = 1) -> int:
    """Single-frame observation dimension (35 values)."""
    dim = 0
    for _name, d, _unit, _note in OBS_FIELDS:
        dim += d
    return dim


# ============================================================
# 动作契约（Actor 输出 → 安全投影 → IK → 执行器）
# ============================================================
# 动作 = 残差增量，经工作空间/速率/加速度限幅后投影为执行器目标。
#   [dtau_common, dtau_yaw, dQx_L, dD0_L, dQx_R, dD0_R]
#   dtau_common : 前向（pitch）轮扭矩残差 (N·m)
#   dtau_yaw    : 偏航轮扭矩残差 (N·m)
#   dQx_L      : 左腿髋关节 X 平移残差 (mm 或 rad，见部署)
#   dD0_L      : 左腿 D0 残差 (mm)
#   dQx_R      : 右腿髋关节 X 平移残差
#   dD0_R      : 右腿 D0 残差 (mm)
ACTION_DIM = 6
ACTION_NAMES = ("dtau_common", "dtau_yaw", "dQx_L", "dD0_L", "dQx_R", "dD0_R")
ACTION_BOUNDS: List[Tuple[float, float]] = [
    (-1.0, 1.0),  # dtau_common (由 tanh 有界，再乘比例)
    (-1.0, 1.0),  # dtau_yaw
    (-1.0, 1.0),  # dQx_L
    (-1.0, 1.0),  # dD0_L
    (-1.0, 1.0),  # dQx_R
    (-1.0, 1.0),  # dD0_R
]


# ============================================================
# 协议契约（UART Pi5 ↔ STM32）
# ============================================================
# 方向护栏：全额定范围无溢出；带版本、长度、序号、时间戳、CRC。
@dataclass
class ProtocolFrameSpec:
    header: int = 0xA5
    footer: int = 0x5A
    version: int = 1
    # 下行（Pi5→STM32）：命令 + action 残差
    down_fields: Tuple[str, ...] = (
        "vx_cmd", "wz_cmd", "d0_cmd",
        "dtau_common", "dtau_yaw",
        "dQx_L", "dD0_L", "dQx_R", "dD0_R",
        "mode", "heartbeat_seq",
    )
    # 上行（STM32→Pi5）实际分为 IMU(6×int16) 与 joints(18×int16) 两帧。
    up_fields: Tuple[str, ...] = (
        "imu_roll", "imu_pitch", "imu_yaw", "gyro_wx", "gyro_wy", "gyro_wz",
        "wheel_L_pos", "wheel_L_vel", "wheel_L_tau",
        "wheel_R_pos", "wheel_R_vel", "wheel_R_tau",
        "hip_A_l_pos", "hip_A_l_vel", "hip_A_l_current",
        "hip_A_r_pos", "hip_A_r_vel", "hip_A_r_current",
        "hip_B_l_pos", "hip_B_l_vel", "hip_B_l_current",
        "hip_B_r_pos", "hip_B_r_vel", "hip_B_r_current",
    )
    # 轮速/角速度使用 ×1000；位置/电流使用各自的 frame codec scale。
    WHEEL_SPEED_SCALE = 1000


# ============================================================
# 契约自检（被 verify_physics_source.py 调用）
# ============================================================
def check_sign_invariants() -> List[str]:
    """返回所有违反的符号护栏错误列表（空 = 通过）。"""
    errs: List[str] = []
    # 同向正扭矩 ⇒ +vx（tau_pitch>0 ⇒ 前进）
    if tau_pitch_from_wheels(0.1, 0.1) <= 0:
        errs.append("tau_pitch_from_wheels(+,+) 应为正（前进）")
    # 右轮 > 左轮 ⇒ +wz（左转）
    if tau_yaw_from_wheels(0.0, 0.1) <= 0:
        errs.append("tau_yaw_from_wheels(L<,R>) 应为正（+wz 左转）")
    if tau_yaw_from_wheels(0.1, 0.0) >= 0:
        errs.append("tau_yaw_from_wheels(L>,R<) 应为负（-wz 右转）")
    # 逆映射一致性
    tl, tr = wheels_from_tau(0.2, 0.05)
    if abs(tl - 0.15) > 1e-9 or abs(tr - 0.25) > 1e-9:
        errs.append("wheels_from_tau 逆映射不一致")
    return errs


if __name__ == "__main__":
    print(f"KUAFU contract schema {SCHEMA_VERSION}")
    print("frame: +X 前, +Y 左, +Z 上")
    print("action dim:", ACTION_DIM, ACTION_NAMES)
    errs = check_sign_invariants()
    print("sign invariants:", "OK" if not errs else errs)
