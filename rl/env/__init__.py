# -*- coding: utf-8 -*-
"""KUAFU RL 环境子包 — 观测/动作/reward 规格 + MJX 环境实现 (design.md §2)"""
from .kuafu_env import (
    OBS_SPEC, OBS_DIM_BASE, OBS_DIM, HISTORY_STEPS,
    PRIVILEGED_SPEC, RMA_LATENT_DIM,
    ACTION_SPEC, ACTION_DIM, WHEEL_TAU_MAX, WHEEL_TAU_CLIP, HIP_RANGE,
    REWARD_TASK, REWARD_STYLE, REWARD_SAFETY,
    DOMAIN_RANDOMIZATION, LQR_K,
    residual_wheel_torque, print_spec,
)
# MJX 环境 (训练时导入, 避免 CPU 验证模式加载 JAX)
from .kuafu_mjx_env import KuafuMjxEnv, make_env
