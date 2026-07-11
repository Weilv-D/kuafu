# -*- coding: utf-8 -*-
"""S0 回归护栏: jax actor/critic 前向 与 torch ActorCritic 数值对齐 (design.md §2.6 阶段 3).

分层断言:
  - float64 结构性对齐 (隔离后端噪声): actor/critic max diff < 1e-5
  - float32 后端噪声容差:           actor/critic/std max diff < 2e-3
  - std 逐位相等 (init_noise_std 常量): diff == 0.0

这是 JaxRollout 的"权重映射 + MLP 前向"层护栏, 不依赖环境/物理; 任何
权重键映射或 mlp_forward 的改动都会在此暴露。运行:

  rl/.venv/bin/python rl/verify/s0_parity.py
"""
import os
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import torch

from rl.env.kuafu_mjx_env import ACTOR_OBS_DIM, CRITIC_OBS_DIM, ACTION_DIM
from rl.train import jax_rollout as jr
from rsl_rl.modules import ActorCritic

CKPT = os.path.join(PROJ_ROOT, "rl/checkpoints/wheel/teacher/model_600.pt")


def main():
    weights = jr.load_ckpt_weights(CKPT)
    sd = torch.load(CKPT, weights_only=False, map_location="cpu")["model_state_dict"]

    torch_model = ActorCritic(
        ACTOR_OBS_DIM, CRITIC_OBS_DIM, ACTION_DIM,
        actor_hidden_dims=[512, 512, 512],
        critic_hidden_dims=[512, 512, 512],
        activation="elu", init_noise_std=1.0,
    )
    torch_model.load_state_dict(sd)
    torch_model.eval()

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    obs_a = jax.random.normal(k1, (128, ACTOR_OBS_DIM))
    obs_c = jax.random.normal(k2, (128, CRITIC_OBS_DIM))
    obs_a_t = torch.from_numpy(np.asarray(obs_a)).float()
    obs_c_t = torch.from_numpy(np.asarray(obs_c)).float()

    with torch.no_grad():
        mean_t = torch_model.actor(obs_a_t)
        val_t = torch_model.critic(obs_c_t)
    mean_j = jr.mlp_forward(weights["actor"], obs_a)
    val_j = jr.mlp_forward(weights["critic"], obs_c)

    mean_diff = float(np.max(np.abs(np.asarray(mean_j) - mean_t.numpy())))
    val_diff = float(np.max(np.abs(np.asarray(val_j) - val_t.numpy())))
    std_diff = float(np.max(np.abs(np.asarray(weights["std"]) - torch_model.std.detach().numpy())))

    print(f"[float32] actor mean max diff : {mean_diff:.3e}")
    print(f"[float32] critic value max diff: {val_diff:.3e}")
    print(f"[float32] std max diff         : {std_diff:.3e}")

    # ---- float64 结构性对齐 (隔离后端噪声) ----
    jax.config.update("jax_enable_x64", True)
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        w64 = {
            "actor": {kk: v.astype(jnp.float64) for kk, v in weights["actor"].items()},
            "critic": {kk: v.astype(jnp.float64) for kk, v in weights["critic"].items()},
            "std": weights["std"].astype(jnp.float64),
        }
        obs_a64 = obs_a.astype(jnp.float64)
        obs_c64 = obs_c.astype(jnp.float64)
        mean_j64 = jr.mlp_forward(w64["actor"], obs_a64)
        val_j64 = jr.mlp_forward(w64["critic"], obs_c64)
        torch_model.double()
        with torch.no_grad():
            mean_t64 = torch_model.actor(torch.from_numpy(np.asarray(obs_a64)).double())
            val_t64 = torch_model.critic(torch.from_numpy(np.asarray(obs_c64)).double())
    mean_diff64 = float(np.max(np.abs(np.asarray(mean_j64) - mean_t64.numpy())))
    val_diff64 = float(np.max(np.abs(np.asarray(val_j64) - val_t64.numpy())))
    print(f"[float64] actor mean max diff : {mean_diff64:.3e}")
    print(f"[float64] critic value max diff: {val_diff64:.3e}")

    ok_struct = max(mean_diff64, val_diff64) < 1e-5
    ok_f32 = max(mean_diff, val_diff, std_diff) < 2e-3
    print("S0 STRUCTURAL (f64):", "PASS" if ok_struct else "FAIL",
          "| S0 f32 backend-noise (<2e-3):", "PASS" if ok_f32 else "FAIL")
    assert ok_struct and ok_f32 and std_diff == 0.0
    print("S0 PARITY: PASS")


if __name__ == "__main__":
    main()
