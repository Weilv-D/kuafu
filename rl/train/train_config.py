# -*- coding: utf-8 -*-
"""
KUAFU PPO 训练配置 — design.md §2.6 训练管线

本轮只交付配置，不执行训练。物理验证（verify_model.py 11/11）通过后，
配合 MuJoCo Playground / RSL-RL 启动训练。

收敛判据 (design.md M5): 恢复时间 < LQR baseline×0.85 或 扰动承受 > baseline×1.2
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import kuafu_physics as P

# ============================================================
# 训练规模 (design.md §2.6: MJX 4096 envs, 单 4090 PPO 60-650K steps/s)
# ============================================================
NUM_ENVS = 1024              # 并行环境数 (RTX 4070 8GB 实测, 4090 可调 4096)
NUM_STEPS_PER_ENV = 24       # 每次 rollout 的步数
TOTAL_TIMESTEPS = 200_000_000  # 总步数 (残差 RL 收敛约 1-2 亿步)

# ============================================================
# PPO 超参 (RSL-RL legged locomotion 事实标准, design.md §2.6)
# ============================================================
PPO = {
    "learning_rate": 3e-4,
    "clip": 0.2,
    "entropy_coef": 0.005,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 4,
    "num_learning_epochs": 5,
    "schedule": "adaptive",   # 按 KL 自适应调学习率
}

# ============================================================
# 网络结构 (design.md §2.5: Pi5 ONNX <1ms 约束)
# ============================================================
NETWORK = {
    "actor":  [256, 256, 256],   # 主干 MLP: obs(~140) → 256×3 → action 6
    "critic": [256, 256, 256],   # value head
    "adapter_cnn": [32, 64, 32], # RMA: 50-step 历史 → 5 维 z
    "vision_encoder": None,      # M6 启用: CNN ~80k 参数 → 32 维
    "activation": "elu",
    "total_params_target": 200_000,  # <200k 保证 ONNX <1ms
}

# ============================================================
# 课程 (design.md §2.6: 自动课程, 按成功率递增)
# ============================================================
CURRICULUM = [
    {"name": "flat_balance",    "terrain": "plane",        "difficulty": 0.0, "threshold": 0.90},
    {"name": "slope",           "terrain": "plane_tilt",   "difficulty": 0.3, "threshold": 0.85},
    {"name": "rough",           "terrain": "hfield",       "difficulty": 0.5, "threshold": 0.80},
    {"name": "stair_30mm",      "terrain": "mesh_stair",   "difficulty": 0.7, "threshold": 0.80},  # M4 验收
    {"name": "perturbation",    "terrain": "plane",        "difficulty": 1.0, "threshold": 0.80},
    # 当前成功率 > threshold → 解锁下一关
]

# ============================================================
# 域随机化 (design.md §2.4, 从 env.kuafu_env 取)
# ============================================================
from rl.env import DOMAIN_RANDOMIZATION  # noqa: E402

# ============================================================
# 收敛判据 (design.md M5 验收)
# ============================================================
CONVERGENCE = {
    "recovery_time_target": "LQR_baseline × 0.85",   # 残差 RL 恢复时间缩短 ≥15%
    "perturbation_target":  "LQR_baseline × 1.20",   # 扰动承受提升 ≥20%
    "student_teacher_ratio": 0.90,                    # student 无特权 ≥ teacher×0.9
}


def print_config():
    print("="*60)
    print("KUAFU PPO 训练配置 (design.md §2.6)")
    print("="*60)
    print(f"并行环境: {NUM_ENVS} (MJX GPU)")
    print(f"总步数: {TOTAL_TIMESTEPS:,}")
    print(f"PPO: lr={PPO['learning_rate']}, clip={PPO['clip']}")
    print(f"网络: actor {NETWORK['actor']}, 参数 <{NETWORK['total_params_target']}")
    print(f"课程: {len(CURRICULUM)} 阶段")
    for c in CURRICULUM:
        print(f"  - {c['name']:18s} {c['terrain']:14s} 解锁阈值 {c['threshold']}")
    print(f"收敛判据: {CONVERGENCE['recovery_time_target']} / {CONVERGENCE['perturbation_target']}")
    print(f"\nGPU: {__import__('jax').devices()}")


if __name__ == "__main__":
    print_config()
