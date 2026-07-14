# -*- coding: utf-8 -*-
"""KUAFU environment interface specification.

This module deliberately contains no second implementation of the observation or
action contract.  Runtime behavior lives in :mod:`kuafu_mjx_env`; cross-layer names,
units and dimensions live in :mod:`contract`.
"""

from __future__ import annotations

from rl.env.contract import ACTION_BOUNDS, ACTION_DIM, ACTION_NAMES, OBS_FIELDS, obs_dim

HISTORY_STEPS = 4
OBS_DIM_BASE = obs_dim()
OBS_DIM = OBS_DIM_BASE * HISTORY_STEPS

# PPO actor receives only causal, hardware-observable history.  The critic receives
# static domain-randomization values (9) and applied push force (3) during training.
RMA_LATENT_DIM = 0
PRIVILEGED_STATIC_DIM = 9
TRANSIENT_DIM = 3
PRIVILEGED_DIM = PRIVILEGED_STATIC_DIM + TRANSIENT_DIM
ACTOR_OBS_DIM = OBS_DIM
CRITIC_OBS_DIM = ACTOR_OBS_DIM + PRIVILEGED_DIM

OBS_SPEC = OBS_FIELDS
ACTION_SPEC = list(zip(ACTION_NAMES, ACTION_BOUNDS))


def print_spec() -> None:
    print("KUAFU actor interface")
    print(f"base observation: {OBS_DIM_BASE} x {HISTORY_STEPS} = {OBS_DIM}")
    print(f"critic observation: {CRITIC_OBS_DIM} (actor + {PRIVILEGED_DIM} privileged)")
    print("actions:", ", ".join(ACTION_NAMES))


if __name__ == "__main__":
    print_spec()
