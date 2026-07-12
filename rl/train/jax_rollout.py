# -*- coding: utf-8 -*-
"""jax 侧 actor/critic MLP 前向与权重映射工具。

仅供 `rl/verify/s0_parity.py` 使用: 把 teacher `.pt` 的权重映射到 jax 的
{w0,b0,w2,b2,w4,b4,w6,b6} 约定, 并用 `mlp_forward` 复刻 RSL-RL actor/critic
`[512,512,512] elu` 前向, 与 torch `ActorCritic` 做数值对齐 (S0 护栏)。

采集路径 (jax.lax.scan 一次性 rollout) 已移除: 逐步采集在 RTX 4070 8GB 上实测
反而更快、更省显存, 且 scan 在 jax 内跑 MLP 慢于 torch 的 cuDNN 路径; 等价性
验证由 S0 护栏覆盖, 无需保留第二套采集实现。
"""

import os
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

import numpy as np
import torch

import jax.numpy as jnp
from jax import nn as jnn


def _t2j(t):
    return np.ascontiguousarray(t.detach().cpu().numpy())


def load_ckpt_weights(ckpt_path):
    sd = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    m = sd["model_state_dict"]
    _map = {"0.weight": "w0", "0.bias": "b0", "2.weight": "w2", "2.bias": "b2",
            "4.weight": "w4", "4.bias": "b4", "6.weight": "w6", "6.bias": "b6"}
    actor = {v: jnp.asarray(_t2j(m[f"actor.{k}"])) for k, v in _map.items()}
    critic = {v: jnp.asarray(_t2j(m[f"critic.{k}"])) for k, v in _map.items()}
    std = jnp.asarray(_t2j(m["std"]))
    return {"actor": actor, "critic": critic, "std": std}


def mlp_forward(p, x):
    x = jnn.elu(x @ p["w0"].T + p["b0"])
    x = jnn.elu(x @ p["w2"].T + p["b2"])
    x = jnn.elu(x @ p["w4"].T + p["b4"])
    x = x @ p["w6"].T + p["b6"]
    return x
