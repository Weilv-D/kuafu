# -*- coding: utf-8 -*-
"""Deployable KUAFU Actor inference model.

The Actor consumes four causal 35-dimensional hardware-observable frames (140
values).  Inputs are normalized by fixed physical scales in the environment/runtime;
there is no training-period running normalizer or RMA latent at deployment.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

import kuafu_physics as P
from rl.env.contract import SCHEMA_VERSION


class TeacherInferenceModel(nn.Module):
    """RSL-RL actor plus the deterministic tanh action transform."""

    def __init__(self, obs_dim: int = 140, action_dim: int = 6,
                 hidden: tuple[int, ...] = (512, 512, 512)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for width in hidden:
            layers.extend((nn.Linear(in_dim, width), nn.ELU()))
            in_dim = width
        layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(obs))

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, obs_dim: int = 140,
                        action_dim: int = 6) -> "TeacherInferenceModel":
        """Load a schema-compatible RSL-RL checkpoint without silent key drops."""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Teacher checkpoint 不存在: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Teacher checkpoint 读取失败: {exc}") from exc
        state = checkpoint.get("model_state_dict")
        if state is None or "actor.0.weight" not in state:
            raise KeyError("checkpoint 缺少 RSL-RL actor 权重")
        metadata = checkpoint.get("kuafu_state")
        if metadata is None or metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("checkpoint lacks current KUAFU schema metadata")
        if metadata.get("model_hash") != P.model_hash():
            raise ValueError("checkpoint physical-model hash does not match current source")

        actual_obs_dim = state["actor.0.weight"].shape[1]
        if actual_obs_dim != obs_dim:
            raise ValueError(
                f"checkpoint actor 输入为 {actual_obs_dim} 维，当前契约要求 {obs_dim} 维；"
                "它是 legacy-v0，不能导出到新架构。"
            )
        actor_linear_keys = sorted(
            (key for key in state if key.startswith("actor.") and key.endswith(".weight")),
            key=lambda key: int(key.split(".")[1]),
        )
        hidden = tuple(state[key].shape[0] for key in actor_linear_keys[:-1])
        model = cls(obs_dim=obs_dim, action_dim=action_dim, hidden=hidden)
        actor_state = {key: value for key, value in state.items() if key.startswith("actor.")}
        missing, unexpected = model.load_state_dict(actor_state, strict=False)
        actor_missing = [key for key in missing if key.startswith("actor.")]
        if actor_missing or unexpected:
            raise RuntimeError(
                f"actor checkpoint 结构不匹配: missing={actor_missing}, unexpected={unexpected}"
            )
        model.eval()
        return model
