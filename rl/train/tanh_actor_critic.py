# -*- coding: utf-8 -*-
"""
KUAFU 训练策略 — tanh-squashed Gaussian 版 ActorCritic。

RSL-RL 2.x 的 ActorCritic 默认用无界 DiagonalGaussian，而部署端 Actor 对动作
额外做 torch.tanh，导致训练策略 ≠ 推理策略 (audit P0)。本子类仅在 update_distribution
把分布替换为单一真相源 TanhGaussian，使训练采样与确定性动作与部署端完全一致。

接线方式 (train.py): 把 policy.class_name 设为 "TanhActorCritic" 并注入本类到
rsl_rl.runners.on_policy_runner 命名空间 (runner 用 eval() 解析 class_name)。
"""

import torch
from rsl_rl.modules import ActorCritic
from rl.train.distributions import TanhGaussian


class TanhActorCritic(ActorCritic):
    """动作分布为 TanhGaussian 的 ActorCritic (其余与 RSL-RL 原版一致)。"""

    def update_distribution(self, observations):
        mean = self.actor(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. "
                "Should be 'scalar' or 'log'"
            )
        # 唯一改动: 用 TanhGaussian 替换 Normal
        self.distribution = TanhGaussian(mean, std)

    def act_inference(self, observations):
        """Use the same deterministic tanh transform as export and Pi runtime."""
        return torch.tanh(self.actor(observations))
