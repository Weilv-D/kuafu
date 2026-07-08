# -*- coding: utf-8 -*-
"""KUAFU RL 环境子包 — 观测/动作/reward 规格 + MJX 环境实现 (design.md §2)

规格常量 (kuafu_env.py) 无 JAX 依赖, 可在 CPU 验证模式直接导入。
MJX 环境 (kuafu_mjx_env.py) 需 JAX/GPU, 按需导入避免链式加载。
"""
from .kuafu_env import (
    OBS_SPEC, OBS_DIM_BASE, OBS_DIM, HISTORY_STEPS,
    PRIVILEGED_SPEC, RMA_LATENT_DIM,
    ACTION_SPEC, ACTION_DIM, WHEEL_TAU_MAX, WHEEL_TAU_CLIP, HIP_RANGE,
    REWARD_TASK, REWARD_STYLE, REWARD_SAFETY,
    DOMAIN_RANDOMIZATION, LQR_K,
    residual_wheel_torque, print_spec,
)

# MJX 环境按需导入 (需 JAX/GPU, 不在 __init__ 链式加载)
# 使用时: from rl.env.kuafu_mjx_env import KuafuMjxEnv
