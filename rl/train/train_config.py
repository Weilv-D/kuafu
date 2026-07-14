# -*- coding: utf-8 -*-
"""
KUAFU PPO 训练配置 — design.md §2.6 训练管线 (单一真相源)

本文件是训练超参的唯一来源: train.py 的 make_train_cfg() 直接读取这里的常量,
不再内联重复定义。物理验证（verify_model.py 11/11）通过后, 配合 MuJoCo Playground /
RSL-RL 2.x 启动训练。

键名严格对齐 RSL-RL 2.x (rsl_rl/algorithms/ppo.py:104-115 与 ActorCritic):
  algorithm: clip_param / num_learning_epochs / num_mini_batches / value_loss_coef /
             entropy_coef / gamma / lam / max_grad_norm / desired_kl / schedule / learning_rate
  policy:    actor_hidden_dims / critic_hidden_dims / activation / init_noise_std
  runner:    num_steps_per_env / save_interval / empirical_normalization

收敛判据 (design.md M5): 恢复时间 < LQR baseline×0.85 或 扰动承受 > baseline×1.2
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================
# 训练规模 (design.md §2.6: MJX 4096 envs, 单 4090 PPO 60-650K steps/s)
# ============================================================
NUM_ENVS = 3072              # 并行环境数 (RTX 4070 8GB: 3072 envs 实测 ~3.3GiB)
ITERATIONS = 5000            # 训练迭代数
SEED = 42
NUM_STEPS_PER_ENV = 96       # 每次 rollout 的步数 (96×0.02s=1.92s, 覆盖完整步态+越障周期)

# ============================================================
# PPO 超参 (RSL-RL legged locomotion 事实标准, design.md §2.6)
# 键名与 rsl_rl 2.x PPO 算法严格一致
# ============================================================
ALGORITHM = {
    "class_name": "PPO",
    "num_learning_epochs": 5,
    "num_mini_batches": 8,
    "clip_param": 0.2,
    "gamma": 0.995,
    "lam": 0.95,
    "value_loss_coef": 1.0,    # RSL-RL 标准值; 价值估计权重, 1.0 比 0.5 更稳
    "entropy_coef": 0.01,     # RSL-RL 标准默认; 配合 adaptive KL(desired_kl=0.01) 自动控熵, 3072 envs 强正则下无需高探索bonus
    "learning_rate": 3e-4,
    "max_grad_norm": 1.0,
    "schedule": "adaptive",   # 按 KL 自适应调学习率
    "desired_kl": 0.01,
    "rnd_cfg": None,
    "symmetry_cfg": None,
}

# ============================================================
# 网络结构 (design.md §2.5: Pi5 ONNX <1ms 约束)
# Scaling: 隐藏层 256→512, 参数量~70万, Pi5 MLP 推理~1.5ms (< 20ms 周期)
# ============================================================
POLICY = {
    "class_name": "ActorCritic",
    "init_noise_std": 0.4,
    "noise_std_type": "scalar",
    "actor_hidden_dims": [512, 512, 512],   # 主干 MLP: actor obs 140 → action 6
    "critic_hidden_dims": [512, 512, 512],  # value head: actor 140 + privileged 12 = 152
    "activation": "elu",
}

# ============================================================
# Runner 级参数 (OnPolicyRunner 顶层键, 非 PPO 算法内部)
# ============================================================
RUN = {
    "num_envs": NUM_ENVS,
    "iterations": ITERATIONS,
    "seed": SEED,
    "num_steps_per_env": NUM_STEPS_PER_ENV,
    "save_interval": 50,
    "empirical_normalization": False,  # Actor 输入使用 contract 固定物理尺度
}

# ============================================================
# 蒸馏 (design.md §2.6 阶段 2: Student DAgger) — 单一真相源
# ============================================================
DISTILL = {
    "max_grad_norm": 1.0,        # 梯度裁剪阈值 (与 PPO max_grad_norm 对齐)
    "z_loss_weight": 5.0,        # latent MSE 相对 action MSE 的权重
    "buffer_capacity": 200_000,  # 回放缓冲上限 (ring 覆盖旧样本)
    "train_batches": 16,         # 每 iter 从缓冲训练 batch 数
    "mini_batch_size": 256,      # 训练 mini-batch
    "buffer_device": "cpu",      # 缓冲驻留设备 (默认 cpu 控显存峰值)
}

# ============================================================
# 课程 (design.md §2.6: 自动课程) — 设计参考; 实际课程由 train.py 的 Curriculum 类
# 驱动 (全局成功率滑动窗口双向调节, 初始难度 0.1 即注入 DR + 随机推力, 防卡 difficulty=0)
# ============================================================
CURRICULUM = [
    {"name": "flat_balance",    "terrain": "plane",        "difficulty": 0.0, "threshold": 0.90},
    {"name": "slope",           "terrain": "plane_tilt",   "difficulty": 0.3, "threshold": 0.85},
    {"name": "rough",           "terrain": "hfield",       "difficulty": 0.5, "threshold": 0.80},
    {"name": "stair_30mm",      "terrain": "mesh_stair",   "difficulty": 0.7, "threshold": 0.80},  # M4 验收
    {"name": "perturbation",    "terrain": "plane",        "difficulty": 1.0, "threshold": 0.80},
    # 成功率 ≥ threshold -> 升级; ≤ 40% -> 降级 (双向调节, 避免策略退化后永久卡死)
]

# ============================================================
# 收敛判据 (design.md M5 验收) — "两者结合": 原恢复时间指标保留为子集,
# 同时加入速度跟踪 / 扰动承受 / 地形成功率多维指标 (残差 RL 安全定位)
# ============================================================
CONVERGENCE = {
    # 子集指标: 残差 RL 恢复时间缩短 ≥15% (靠大扰动/地形逼入非线性区争取达成)
    "recovery_time_target": "LQR_baseline × 0.85 (subset)",
    # 扰动承受提升 ≥20% (最大可恢复推力对比 LQR)
    "perturbation_target":  "LQR_baseline × 1.20",
    # student 无特权 ≥ teacher×0.9
    "student_teacher_ratio": 0.90,
    # 新增多维验收 (M5 综合判据)
    "lin_vel_track_err_max": 0.10,    # m/s, cmd_sweep 速度跟踪误差均值上限
    "terrain_stair_success": 0.80,    # M4: 30mm 台阶成功率 ≥80%
    "latency_robustness":   "带延迟(≤40ms)+DR 评估不显著退化",
}


def print_config():
    print("=" * 60)
    print("KUAFU PPO 训练配置 (design.md §2.6, RSL-RL 2.x)")
    print("=" * 60)
    print(f"并行环境: {RUN['num_envs']} (MJX GPU)")
    print(f"迭代数: {RUN['iterations']} × {RUN['num_envs']} envs × {RUN['num_steps_per_env']} steps")
    print(f"PPO: lr={ALGORITHM['learning_rate']}, clip={ALGORITHM['clip_param']}, "
          f"schedule={ALGORITHM['schedule']}, desired_kl={ALGORITHM['desired_kl']}")
    print(f"网络: actor {POLICY['actor_hidden_dims']}, 参数 <700k, Pi5 ONNX ~1.5ms (< 20ms 周期)")
    print(f"课程: {len(CURRICULUM)} 阶段")
    for c in CURRICULUM:
        print(f"  - {c['name']:18s} {c['terrain']:14s} 解锁阈值 {c['threshold']}")
    print(f"收敛判据: {CONVERGENCE['recovery_time_target']} / {CONVERGENCE['perturbation_target']}")
    print(f"\nGPU: {__import__('jax').devices()}")


if __name__ == "__main__":
    print_config()
