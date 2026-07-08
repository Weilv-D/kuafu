# -*- coding: utf-8 -*-
"""
KUAFU RL 策略网络 — PyTorch nn.Module (部署用)

design.md §2.5: RMA Adapter [32,64,32]→5 + StudentPolicy [256,256,256]
约束: 总参数 <200k (Pi5 ONNX <1ms)

注意: Teacher 训练使用 RSL-RL 内置 ActorCritic (由 train.py config 配置),
本文件仅定义部署用的 StudentPolicy 和 RMAAdapter。

Student: trunk(proprio 140 + RMA z 5) + policy_head → action 6
RMA: 50 步历史 → CNN [32,64,32] → latent z(5)

history_len 说明: 环境用 HISTORY_STEPS=4 堆叠成 140 维 obs (proprio),
但 RMA adapter 需要更长的时序历史 (50 步) 来推断环境参数。
部署时从 50 步的 140 维 obs 序列中提取 base_obs(35) 喂给 adapter。
"""
import torch
import torch.nn as nn
from typing import Tuple


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
        x = history.transpose(1, 2)  # (B, obs_dim, history_len)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (B, hidden)
        z = self.fc(x)
        return z


class StudentPolicy(nn.Module):
    """Student 策略: trunk(proprio+z) + policy_head.

    部署时只保留 trunk + adapter + policy_mean (不含 critic)。
    """

    def __init__(
        self,
        proprio_dim: int = 140,
        history_obs_dim: int = 35,
        history_len: int = 50,
        action_dim: int = 6,
        latent_dim: int = 9,
        hidden_dims: Tuple[int, ...] = (512, 512, 512),
    ):
        super().__init__()
        self.adapter = RMAAdapter(history_obs_dim, history_len, latent_dim, (32, 64, 32))

        # 注册 Normalizer Buffers (从 Teacher Checkpoint 载入)
        self.register_buffer("obs_mean", torch.zeros(1, proprio_dim))
        self.register_buffer("obs_std", torch.ones(1, proprio_dim))
        self.register_buffer("priv_mean", torch.zeros(1, latent_dim))
        self.register_buffer("priv_std", torch.ones(1, latent_dim))

        act_fn = nn.ELU
        layers = []
        in_dim = proprio_dim + latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(in_dim, action_dim)

    def forward(self, proprio: torch.Tensor, history: torch.Tensor):
        z = self.adapter(history)

        # 观测值正规化 (Normalizer)
        proprio_norm = (proprio - self.obs_mean) / (self.obs_std + 1e-8)
        z_norm = (z - self.priv_mean) / (self.priv_std + 1e-8)

        x = torch.cat([proprio_norm, z_norm], dim=-1)
        h = self.trunk(x)
        action = torch.tanh(self.actor_mean(h))

        if self.training:
            return action, z
        return action


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    student = StudentPolicy()
    print(f"Student Policy (trunk+adapter): {count_parameters(student):,} 参数")
    adapter = RMAAdapter()
    print(f"RMA Adapter only: {count_parameters(adapter):,} 参数")
