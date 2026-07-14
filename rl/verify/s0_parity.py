"""Policy-contract parity guard for the current 140-dimensional Actor."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from rl.env.kuafu_mjx_env import ACTOR_OBS_DIM, CRITIC_OBS_DIM
from rl.train.tanh_actor_critic import TanhActorCritic


def main() -> int:
    policy = TanhActorCritic(ACTOR_OBS_DIM, CRITIC_OBS_DIM, 6,
                             actor_hidden_dims=[16], critic_hidden_dims=[16])
    obs = torch.randn(4, ACTOR_OBS_DIM)
    deterministic = policy.act_inference(obs)
    expected = torch.tanh(policy.actor(obs))
    critic = policy.evaluate(torch.cat([obs, torch.zeros(4, CRITIC_OBS_DIM - ACTOR_OBS_DIM)], dim=-1))
    if not torch.allclose(deterministic, expected) or torch.max(torch.abs(deterministic)) > 1.0:
        raise RuntimeError("Actor deterministic tanh parity failed")
    if critic.shape != (4, 1) or not torch.isfinite(critic).all():
        raise RuntimeError("Critic output parity failed")
    print(f"Actor/Critic dimensions {ACTOR_OBS_DIM}/{CRITIC_OBS_DIM}; actor tanh + critic shape: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
