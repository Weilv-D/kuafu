# -*- coding: utf-8 -*-
"""
KUAFU RL 策略网络 — PyTorch nn.Module

design.md §2.5: ActorCritic [256,256,256] + RMA Adapter [32,64,32]→5
约束: 总参数 <200k (Pi5 ONNX <1ms)

- Teacher: ActorCritic(obs+privileged → action + value)
- Student: trunk(proprio) + adapter(history→z) + policy_head(trunk+z → action)

依赖: torch, rsl-rl-lib (提供 ActorCritic 基类)
"""
import torch
import torch.nn as nn
from typing import Tuple


class KuafuActorCritic(nn.Module):
    """KUAFU Actor-Critic MLP.

    用于 RSL-RL OnPolicyRunner 的 policy 网络。
    Teacher 模式: input = proprio(140) + privileged(9) = 149
    Student 模式: input = proprio(140) + RMA z(5) = 145

    结构: [input] → 256 → 256 → 256 → [action(6) | value(1)]
    参数量: 149×256 + 256×256 + 256×256 + 256×7 ≈ 200k
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 6,
        hidden_dims: Tuple[int, ...] = (256, 256, 256),
        activation: str = "elu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        act_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        # 共享主干
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)

        # Actor head (输出 action 均值)
        self.actor_mean = nn.Linear(in_dim, action_dim)
        # 动作标准差 (可学习参数, 不经过网络)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic head (输出 value)
        self.critic = nn.Linear(in_dim, 1)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播: 返回 (action_mean, value)."""
        h = self.trunk(obs)
        action_mean = self.actor_mean(h)
        value = self.critic(h)
        return action_mean, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作: 返回 (action, log_prob, value)."""
        action_mean, value = self.forward(obs)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作: 返回 (log_prob, entropy, value)."""
        action_mean, value = self.forward(obs)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class RMAAdapter(nn.Module):
    """RMA 适配器: 从历史观测推断环境隐变量 z.

    design.md §2.5: 50 步历史 → CNN [32,64,32] → 5 维 z
    Student 部署时在线推断 z, 适应质量/摩擦/延迟变化。
    """

    def __init__(
        self,
        obs_dim: int = 35,
        history_len: int = 50,
        latent_dim: int = 5,
        hidden_dims: Tuple[int, ...] = (32, 64, 32),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.latent_dim = latent_dim

        # 1D CNN over history: (B, obs_dim, history_len) → conv → flatten → latent
        layers = []
        in_ch = obs_dim
        for out_ch in hidden_dims:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ELU())
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(in_ch, latent_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: (B, history_len, obs_dim) → z: (B, latent_dim)."""
        # Conv1d 期望 (B, channels, length)
        x = history.transpose(1, 2)  # (B, obs_dim, history_len)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (B, hidden)
        z = self.fc(x)
        return z


class StudentPolicy(nn.Module):
    """Student 策略: trunk(proprio) + adapter(history→z) + policy_head(trunk+z).

    部署时只保留 trunk + adapter + policy_mean (不含 critic)。
    """

    def __init__(
        self,
        proprio_dim: int = 140,
        history_obs_dim: int = 35,
        history_len: int = 50,
        action_dim: int = 6,
        latent_dim: int = 5,
        hidden_dims: Tuple[int, ...] = (256, 256, 256),
    ):
        super().__init__()
        self.adapter = RMAAdapter(history_obs_dim, history_len, latent_dim, (32, 64, 32))

        # trunk 输入 = proprio + latent_z
        act_fn = nn.ELU
        layers = []
        in_dim = proprio_dim + latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(in_dim, action_dim)

    def forward(self, proprio: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        z = self.adapter(history)
        x = torch.cat([proprio, z], dim=-1)
        h = self.trunk(x)
        action = torch.tanh(self.actor_mean(h))  # [-1, 1]
        return action


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 参数量检查
    teacher = KuafuActorCritic(obs_dim=149, action_dim=6)
    print(f"Teacher ActorCritic: {count_parameters(teacher):,} 参数")
    # 应 <200k

    student = StudentPolicy()
    print(f"Student Policy (trunk+adapter): {count_parameters(student):,} 参数")

    adapter = RMAAdapter()
    print(f"RMA Adapter only: {count_parameters(adapter):,} 参数")
