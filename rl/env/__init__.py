# -*- coding: utf-8 -*-
"""KUAFU RL 环境子包 — 观测/动作/reward/域随机化规格 (design.md §2)"""
from .kuafu_env import (
    OBS_SPEC, OBS_DIM_BASE, OBS_DIM, HISTORY_STEPS,
    PRIVILEGED_SPEC, RMA_LATENT_DIM,
    ACTION_SPEC, ACTION_DIM, WHEEL_TAU_MAX, WHEEL_TAU_CLIP, HIP_RANGE,
    REWARD_TASK, REWARD_STYLE, REWARD_SAFETY,
    DOMAIN_RANDOMIZATION, LQR_K,
    residual_wheel_torque, print_spec,
)
