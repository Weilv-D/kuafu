"""JAX reset/step and selective auto-reset smoke for the current Actor contract."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp
import torch

from rl.env.kuafu_mjx_env import ACTOR_OBS_DIM, CRITIC_OBS_DIM, KuafuMjxEnv
from rl.train import dlpack_utils as dlu
from rl.train.train import DirectVecEnv


def main() -> int:
    env = KuafuMjxEnv(teacher=True, num_envs=1, episode_length=2)
    state = env.reset(jax.random.PRNGKey(0))
    if state.obs["state"].shape != (ACTOR_OBS_DIM,) or state.obs["privileged_state"].shape != (12,):
        raise RuntimeError("single-env observation contract mismatch")
    state = jax.jit(env.step)(state, jp.zeros(6))
    if not jp.isfinite(state.reward):
        raise RuntimeError("JAX step returned non-finite reward")

    device = dlu.resolve_device("cuda")
    vector = DirectVecEnv(KuafuMjxEnv(teacher=True, num_envs=2, episode_length=0), 2, 0, device=device)
    action = torch.zeros((2, 6), device=device)
    actor_obs, _reward, done, info = vector.step(action)
    if actor_obs.shape != (2, ACTOR_OBS_DIM) or not bool(done.all()):
        raise RuntimeError("selective auto-reset contract mismatch")
    if info["observations"]["critic"].shape != (2, CRITIC_OBS_DIM):
        raise RuntimeError("critic observation contract mismatch")
    print("MJX reset/step/auto-reset smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
