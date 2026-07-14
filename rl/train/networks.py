# -*- coding: utf-8 -*-
"""Standalone deployable Actor network definitions.

RMA adapters and student policies are intentionally absent.  The deployable policy
is the PPO Actor over the 140-dimensional causal hardware-observation history.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ActorPolicy(nn.Module):
    def __init__(self, observation_dim: int = 140, action_dim: int = 6,
                 hidden_dims: Tuple[int, ...] = (512, 512, 512)):
        super().__init__()
        layers: list[nn.Module] = []
        width = observation_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.ELU()))
            width = hidden
        layers.append(nn.Linear(width, action_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(observation))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
