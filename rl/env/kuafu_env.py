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
# 观测空间 (design.md §2.1) — M5 student 本体感受 37 维 + 历史
# 2-DOF 五杆: 全观测 4 舵机 (hip_A + hip_B 各左右)
# ============================================================
OBS_SPEC = [
    # (组名, 维度, 内容, 来源)
    ("attitude",       3, "机身 roll/pitch/yaw",                 "BMI088 Mahony"),
    ("ang_vel",        3, "机身角速度 ωx/ωy/ωz (陀螺)",           "BMI088, 1kHz→50Hz 降采样"),
    ("wheel_state",    4, "左右轮位置+速度",                      "DDSM315 回传"),
    ("hip_state",      8, "4 舵机位置+速度 (hip_A/B 各左右)",     "ST3215 回传"),
    ("wheel_torque",   2, "左右轮力矩电流→τ",                    "DDSM315, 关键"),
    ("hip_torque",     4, "4 舵机电流→力矩代理 (hip_A/B 各左右)", "ST3215 6.5mA/LSB, 关键"),
    ("last_action",    6, "上一步动作 [Δτ_L/R, q_hip_A/B×2×lr]", "动作平滑性诊断"),
    ("command",        3, "[v_cmd, ω_cmd, D0_cmd]",              "高层下发"),
    ("phase_clock",    2, "sin/cos(2π t/T_phase)",               "步态相位编码"),
    ("contact",        2, "左右轮接触标志 (1=接地,0=离地)",      "MJX contact geom"),
]
OBS_DIM_BASE = sum(d for _, d, _, _ in OBS_SPEC)     # = 37
HISTORY_STEPS = 4                                      # design.md §2.1: 堆叠 N=4
OBS_DIM = OBS_DIM_BASE * HISTORY_STEPS                # = 140

# teacher 特权观测 (仅训练, design.md §2.1) 拆分为静态外因 + 瞬态扰动
# 静态环境外因 (RMA latent 监督目标, 9 维): episode 级常量, 由 RMA adapter 从
#   本体感受历史推断 (Kumar et al. RSS 2021)。与 kuafu_mjx_env.RMA_STATIC_DIM 一致。
PRIVILEGED_STATIC_SPEC = [
    ("friction",        1, "接触摩擦系数真值",      "sim"),
    ("mass_scale",      1, "整机质量缩放真值",      "sim"),
    ("com_bias",        3, "M/COM 偏移真值",        "sim"),
    ("inertia_scale",   1, "转动惯量缩放真值",      "sim"),
    ("torque_scale",    1, "电机力矩常数缩放真值",  "sim"),
    ("deadband",        1, "舵机死区真值",          "sim"),
    ("delay_steps",     1, "执行器/传感延迟步数",  "sim"),
]
# 瞬态扰动 (3 维): 每步变化的外部推力, 由 policy 本体感受(wheel/hip torque)在线
#   感知, 只留 teacher critic 特权, 不进 RMA latent。与 kuafu_mjx_env.TRANSIENT_DIM 一致。
TRANSIENT_SPEC = [
    ("active_push",     3, "实际施加的瞬态扰动力",  "sim"),
]

# RMA latent (design.md §2.5) — 仅静态环境外因 (9 维)
# 训练时 teacher actor 以 proprio(148) + z(9) = 157 维为条件 (Kumar 2021);
# critic 额外吃瞬态(3) → 160。部署时 z 由 student adapter 从历史预测。
RMA_LATENT_DIM = 9
ACTOR_OBS_DIM = OBS_DIM + RMA_LATENT_DIM          # 157
CRITIC_OBS_DIM = ACTOR_OBS_DIM + 3                # 160 (瞬态 active_push 3 维)


# ============================================================
# 动作空间 (design.md §2.2) — 混合 6 维, 50Hz
# 2-DOF 五杆: 4 个舵机 (hip_A + hip_B 各左右) 全部独立位置控制
# ============================================================
ACTION_SPEC = [
    # (维度名, 对象, 物理量, 范围, 叠加方式)
    ("dtau_L",   "左轮",    "力矩残差 Δτ_L", [-1, 1], "τ_cmd = clip(LQR + Δτ×0.55, ±1.1)"),
    ("dtau_R",   "右轮",    "力矩残差 Δτ_R", [-1, 1], "同上"),
    ("q_hip_A_l","左髋A",   "位置目标",     [-1, 1], "q_goal = action×HIP_STROKE"),
    ("q_hip_A_r","右髋A",   "位置目标",     [-1, 1], "同上"),
    ("q_hip_B_l","左髋B",   "位置目标",     [-1, 1], "同上 (2-DOF 独立曲柄)"),
    ("q_hip_B_r","右髋B",   "位置目标",     [-1, 1], "同上"),
]
ACTION_DIM = len(ACTION_SPEC)     # = 6
WHEEL_TAU_MAX = P.TAU_WHEEL_RATED  # 0.55 Nm, 残差归一化基准
WHEEL_TAU_CLIP = P.TAU_WHEEL_STALL # 1.1 Nm, 叠加后硬限幅
HIP_RANGE = 1.0                    # 腿位置目标半幅 (归一化), 实际 rad 由 HIP_STROKE 定


# ============================================================
# Reward (design.md §2.3) — task + style + safety
# ============================================================
REWARD_TASK = {
    "lin_vel_tracking":  ("跟踪 v_cmd",                    1.0),
    "ang_vel_tracking":  ("跟踪 ω_cmd",                    0.5),
    "d0_avg_tracking":   ("跟踪 D0_cmd 左右平均(不惩罚左右差)", 0.3),
    "orientation":       ("exp(-α·(gx²+gy²)) 重力向量",    1.0),
    "tilt_cost":         ("-‖g_xy‖ 线性倾角惩罚(恢复激励)",  0.5),
    "roll_leveling":     ("exp(-roll²/σ²) 机身水平奖励",    1.0),
    "alive":             ("存活奖励 (倒下时归零, 防蹭奖)",  0.1),
    "fall_penalty":      ("倒下当步终止惩罚",              1.0),
}
REWARD_STYLE = {
    "extension_cost":    ("超 d0_cmd 过度伸展惩罚(不惩罚命令内伸展)", 0.5),
    "contact_asymmetry": ("-|contact_L-R| 抑制长时间单轮卸载(限M4)", 0.3),
    "ang_vel_xy":        ("-(ωx²+ωy²) 惩罚 roll/pitch 角速度", 0.05),
    "action_rate":       ("-‖a_t - a_{t-1}‖² (一阶)",  0.01),
    "energy":            ("轮|τ·ω| + 4髋τ²(铜损)",      0.001),
    "torque_limit":      ("超连续安全扭矩惩罚 (4舵机)",   0.5),
}
REWARD_SAFETY = {
    "soft_termination":  ("连续倒下≥10步(200ms)才终止, alive门控+fall_penalty双重抑制", None),
    "joint_limit":       ("超机械限位惩罚",             None),
    "leg_overload":      ("舵机电流超连续安全 → 回锁",   None),
}
# 注: 所有 reward 项 ×scale 后统一乘 CTRL_DT (Go1/T1 标准, 保持 PPO value 尺度)
# d0_avg_tracking 仅惩罚左右平均 D0 偏离, 不惩罚左右差 → 允许腿用于 roll 调平
#   (一腿高一腿低仍水平, 机构 4 舵机独立支持)。
# roll_leveling 显式奖励机身水平, 激活腿调平能力 (原 default_pose 抑制)。
# contact_asymmetry 限制 M4 抬轮为短暂, 防单轮持续卸载失稳。
# tilt_cost 与 orientation 互补: orientation(exp) 近直立饱和; tilt_cost(线性) 全程梯度。
# fallen 阈值统一 30° (物理可恢复 ~25°, 留余量), 训练/评估/回放共用。


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


def residual_wheel_torque(lqr_F, dtau_norm, torque_scale=1.0):
    """RL 轮力矩残差叠加 (与 kuafu_mjx_env.step 一致):

    τ_cmd = clip((LQR×R/2 + Δτ×τ_max) × torque_scale, ±堵转), 每轮。
    torque_scale 模拟电机常数偏差, 对 LQR 底层与 RL 残差统一生效。
    """
    tau_lqr = lqr_F * P.R / 2.0 * torque_scale          # LQR 输出分摊两轮
    tau_rl = dtau_norm * WHEEL_TAU_MAX * torque_scale    # 残差归一化→Nm
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
