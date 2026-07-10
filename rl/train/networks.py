# -*- coding: utf-8 -*-
"""
KUAFU RL 策略网络 — PyTorch nn.Module (部署用)

design.md §2.5: RMA Adapter [32,64,32]→9 (静态环境外因) + StudentPolicy [512,512,512]
约束: 总参数 ~621k (与 teacher 隐藏层对齐, Pi5 ONNX ~1.5ms < 20ms 周期)

注意: Teacher 训练使用 RSL-RL 内置 ActorCritic (由 train.py config 配置),
本文件仅定义部署用的 StudentPolicy 和 RMAAdapter。

Student: trunk(proprio 148 + RMA z 9) + policy_head → action 6
RMA: 50 步历史 → CNN [32,64,32] → latent z(9)

history_len 说明: 环境用 HISTORY_STEPS=4 堆叠成 148 维 obs (proprio),
但 RMA adapter 需要更长的时序历史 (50 步) 来推断环境参数。
部署时从 50 步的 148 维 obs 序列中提取 base_obs(37) 喂给 adapter。
2-DOF 五杆: 4 舵机独立驱动, 故动作 6 维、obs 37 维 (含接触标志)。
"""
import torch
import torch.nn as nn
from typing import Tuple


class RMAAdapter(nn.Module):
    """RMA 适配器: 从历史观测推断环境隐变量 z.

    design.md §2.5: 50 步历史 → CNN [32,64,32] → 9 维静态环境参数 z
    Student 部署时在线推断 z, 适应质量/摩擦/延迟变化。
    """

    def __init__(
        self,
        obs_dim: int = 37,
        history_len: int = 50,
        latent_dim: int = 9,
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
        proprio_dim: int = 148,
        history_obs_dim: int = 37,
        history_len: int = 50,
        action_dim: int = 6,
        latent_dim: int = 9,
        hidden_dims: Tuple[int, ...] = (512, 512, 512),
    ):
        super().__init__()
        self.adapter = RMAAdapter(history_obs_dim, history_len, latent_dim, (32, 64, 32))

        # 注册 Normalizer Buffers (从 Teacher Checkpoint 载入, 维度 = proprio+latent = 157)
        # 与 Teacher actor 一致: EmpiricalNormalization 对 [proprio, z] 整体归一化。
        self.register_buffer("obs_mean", torch.zeros(1, proprio_dim + latent_dim))
        self.register_buffer("obs_std", torch.ones(1, proprio_dim + latent_dim))

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

        # 与 Teacher actor 对齐: 将 [proprio, z] 拼成 157 维后整体归一化,
        # 再由 trunk 处理。z 在原始空间 (与训练时真值同尺度), 不直接归一化。
        full = torch.cat([proprio, z], dim=-1)
        full_norm = (full - self.obs_mean) / (self.obs_std + 1e-8)

        h = self.trunk(full_norm)
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
