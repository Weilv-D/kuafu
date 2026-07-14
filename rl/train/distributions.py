# -*- coding: utf-8 -*-
"""
KUAFU 动作分布 — 单一真相源。

训练 (RSL-RL ActorCritic 子类) 与部署 (PyTorch Actor / ONNX / Pi5)
必须使用同一个 tanh-squashed Gaussian 变换，否则训练策略与推理策略不是同一策略
(见 audit P0)。本模块是该变换的唯一实现：

  a = tanh(u),  u ~ N(loc, scale)
  log p(a) = log N(atanh(a); loc, scale) - Σ_i log(1 - a_i^2)

行列式 Jacobian 项 -log(1-a^2) 必须计入 log_prob，否则 PPO 的 ratio / 熵估计偏差。
"""

import torch
from torch.distributions import Normal


class TanhGaussian:
    """对角 tanh-squashed Gaussian。

    与 RSL-RL 的 DiagonalGaussian 接口对齐 (sample / log_prob / mode / mean / stddev /
    entropy)，便于直接替换 ActorCritic.update_distribution 中的分布。
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = scale
        self.normal = Normal(loc, scale)
        # RSL-RL persists ``action_mean``/``action_std`` for adaptive KL.  KL is
        # invariant under a shared bijection, so those values must remain in the
        # pre-tanh Normal space.  Deterministic control uses ``mode()`` instead.
        self._mean = loc

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def stddev(self) -> torch.Tensor:
        return self.scale

    @property
    def squashed_mean(self) -> torch.Tensor:
        """Deterministic action in the bounded environment/deployment space."""
        return torch.tanh(self.loc)

    def sample(self, sample_shape=()) -> torch.Tensor:
        u = self.normal.sample(sample_shape)
        return torch.tanh(u)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # 逆变换 atanh，裁剪避免 |a|→1 时数值爆炸
        a = torch.clamp(actions, -0.999999, 0.999999)
        u = torch.atanh(a)
        # 行列式 Jacobian: |da/du| = Π(1 - tanh^2(u)) = Π(1 - a^2)
        jac = torch.log(1.0 - a.pow(2) + 1e-6)
        return self.normal.log_prob(u) - jac

    def entropy(self) -> torch.Tensor:
        # There is no closed-form entropy for a tanh-normal.  A detached Monte
        # Carlo sample gives the correct transformed-policy estimator, including
        # the Jacobian already used by log_prob, while preserving gradients to
        # loc/scale for the PPO entropy coefficient.
        sample = torch.tanh(self.normal.sample())
        return -self.log_prob(sample)

    def mode(self) -> torch.Tensor:
        return self.squashed_mean
