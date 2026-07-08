# -*- coding: utf-8 -*-
"""
KUAFU 残差 RL 环境 — MuJoCo MJX 封装

对应 design.md §2.1 观测空间 / §2.2 动作空间 / §2.3 Reward / §2.4 域随机化。
驻留态腿被动自锁，整机降为轮式倒立摆；RL 输出残差叠加在 LQR/PD 底层之上。

本模块定义观测/动作/reward/域随机化的规格与骨架实现，供 MuJoCo Playground /
RSL-RL 训练管线接入。本轮只交付规格与可实例化的骨架，不跑训练。

依赖: mujoco-mjx, jax, flax (训练时), 验证可降级为原生 mujoco
"""
import os
import sys
import numpy as np

# 物理真源
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import kuafu_physics as P

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kuafu.xml")


# ============================================================
# 观测空间 (design.md §2.1) — M5 student 本体感受 27 维 + 历史
# 对称步态: 只观测驱动侧 2 舵机 (hip_A_l/r), hip_B 由对称耦合确定不冗余观测
# ============================================================
OBS_SPEC = [
    # (组名, 维度, 内容, 来源)
    ("attitude",       3, "机身 roll/pitch/yaw",                 "BMI088 Mahony"),
    ("ang_vel",        3, "机身角速度 ωx/ωy/ωz (陀螺)",           "BMI088, 1kHz→50Hz 降采样"),
    ("wheel_state",    4, "左右轮位置+速度",                      "DDSM315 回传"),
    ("hip_state",      4, "驱动侧 2 舵机位置+速度",               "ST3215 回传"),
    ("wheel_torque",   2, "左右轮力矩电流→τ",                    "DDSM315, 关键"),
    ("hip_torque",     2, "驱动侧 2 舵机电流→力矩代理",          "ST3215 6.5mA/LSB, 关键"),
    ("last_action",    4, "上一步动作 [Δτ_L, Δτ_R, q_hip_A×2]",  "动作平滑性诊断"),
    ("command",        3, "[v_cmd, ω_cmd, D0_cmd]",              "高层下发"),
    ("phase_clock",    2, "sin/cos(2π t/T_phase)",               "步态相位编码"),
]
OBS_DIM_BASE = sum(d for _, d, _, _ in OBS_SPEC)     # = 27
HISTORY_STEPS = 4                                      # design.md §2.1: 堆叠 N=4
OBS_DIM = OBS_DIM_BASE * HISTORY_STEPS                # = 108

# teacher 特权观测 (仅训练, design.md §2.1)
PRIVILEGED_SPEC = [
    ("terrain_height", -1, "机身周围高度图",        "sim 注入"),
    ("friction",        1, "接触摩擦系数真值",      "sim"),
    ("mass_bias",       3, "M/COM 偏移真值",        "sim"),
    ("delay",           2, "actuator/sensor 延迟",  "sim"),
    ("external_force",  3, "外部扰动力向量",        "sim"),
]

# RMA latent (design.md §2.5)
RMA_LATENT_DIM = 5


# ============================================================
# 动作空间 (design.md §2.2) — 混合 4 维, 50Hz
# 对称步态: 仅 hip_A 有舵机 (hip_B 由 joint equality 镜像驱动)
# ============================================================
ACTION_SPEC = [
    # (维度名, 对象, 物理量, 范围, 叠加方式)
    ("dtau_L",  "左轮",    "力矩残差 Δτ_L", [-1, 1], "τ_cmd = clip(LQR + Δτ×0.55, ±1.1)"),
    ("dtau_R",  "右轮",    "力矩残差 Δτ_R", [-1, 1], "同上"),
    ("q_hip_A_l","左驱动髋","位置目标",     [-1, 1], "q_goal = action×HIP_STROKE (D0 58→207)"),
    ("q_hip_A_r","右驱动髋","位置目标",     [-1, 1], "同上"),
]
ACTION_DIM = len(ACTION_SPEC)     # = 4
WHEEL_TAU_MAX = P.TAU_WHEEL_RATED  # 0.55 Nm, 残差归一化基准
WHEEL_TAU_CLIP = P.TAU_WHEEL_STALL # 1.1 Nm, 叠加后硬限幅
HIP_RANGE = 1.0                    # 腿位置目标半幅 (归一化), 实际 rad 由 HIP_STROKE 定


# ============================================================
# Reward (design.md §2.3) — task + style + safety
# ============================================================
REWARD_TASK = {
    "lin_vel_tracking":  ("跟踪 v_cmd",            1.0),
    "ang_vel_tracking":  ("跟踪 ω_cmd",            0.5),
    "leg_height_tracking":("跟踪 D0_cmd (驻留主导)",0.3),
    "upright":           ("exp(-pitch²) + exp(-roll²)", 1.0),
}
REWARD_STYLE = {
    "action_rate":       ("-‖a_t - a_{t-1}‖²",      0.01),
    "action_smoothness": ("-‖a_t - 2a_{t-1} + a_{t-2}‖²", 0.01),
    "energy":            ("-Σ|τ·ω|",               0.001),
    "torque_limit":      ("超连续安全扭矩惩罚",      0.5),
}
REWARD_SAFETY = {
    "fall":         ("|pitch|或|roll|>阈值 → 终止 -50", None),
    "joint_limit":  ("超机械限位惩罚",             None),
    "leg_overload": ("舵机电流超连续安全 → 回锁",   None),
}


# ============================================================
# 域随机化 (design.md §2.4) — 从 kuafu_physics 取范围
# ============================================================
DOMAIN_RANDOMIZATION = {
    "mass":          P.DR_MASS,          # ±15%
    "com":           P.DR_COM,           # ±20mm
    "inertia":       P.DR_INERTIA,       # ×[0.5, 2.0]
    "friction":      P.DR_FRICTION,      # [0.3, 1.2]
    "wheel_radius":  P.DR_WHEEL_R,       # ±1mm
    "torque_const":  P.DR_TORQUE_CONST,  # ±10%
    "servo_pd":      P.DR_SERVO_PD,      # ±30%
    "deadband":      P.DR_DEADBAND,      # [0, 2°]
    "delay_act":     P.DR_DELAY_ACT,     # [0, 30]ms
    "delay_sense":   P.DR_DELAY_SENSE,   # [0, 20]ms
}


# ============================================================
# LQR 底层 (永远在环, design.md §6.4.2)
# ============================================================
LQR_K = P.LQR_K  # [-4.47, -61.18, -5.82, -4.02], 状态 [x,θ,ẋ,θ̇]


def residual_wheel_torque(lqr_F, dtau_norm):
    """RL 轮力矩残差叠加: τ_cmd = clip(LQR/2 + Δτ×τ_max, ±堵转), 每轮."""
    tau_lqr = lqr_F * P.R / 2.0          # LQR 输出分摊两轮
    tau_rl = dtau_norm * WHEEL_TAU_MAX   # 残差归一化→Nm
    return float(np.clip(tau_lqr + tau_rl, -WHEEL_TAU_CLIP, WHEEL_TAU_CLIP))


def print_spec():
    """打印观测/动作/reward 规格, 供核对与文档同步."""
    print("="*60)
    print("KUAFU RL 环境规格 (design.md §2.1-2.4)")
    print("="*60)
    print(f"\n观测空间: 基础 {OBS_DIM_BASE} 维 × {HISTORY_STEPS} 步历史 = {OBS_DIM} 维")
    for name, dim, desc, src in OBS_SPEC:
        print(f"  {name:14s} {dim:2d}  {desc}  ({src})")
    print(f"\n动作空间: {ACTION_DIM} 维 (50 Hz)")
    for name, obj, qty, rng, stack in ACTION_SPEC:
        print(f"  {name:8s} {obj:6s} {qty:16s} {rng}  {stack}")
    print(f"\nReward (task):")
    for k, (desc, w) in REWARD_TASK.items():
        print(f"  {k:20s} w={w:<5} {desc}")
    print(f"Reward (style):")
    for k, (desc, w) in REWARD_STYLE.items():
        print(f"  {k:20s} w={w:<6} {desc}")
    print(f"Reward (safety):")
    for k, (desc, _) in REWARD_SAFETY.items():
        print(f"  {k:20s} {desc}")
    print(f"\n域随机化: {len(DOMAIN_RANDOMIZATION)} 项 (见 kuafu_physics.DR_*)")
    print(f"LQR 底层增益 K = {LQR_K} (永远在环, RL 挂掉兜底)")


if __name__ == "__main__":
    print_spec()
