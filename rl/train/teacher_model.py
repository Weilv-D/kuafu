# -*- coding: utf-8 -*-
"""
KUAFU Teacher 推理模型 — 匹配 RSL-RL 2.x checkpoint 的真实结构

checkpoint 实际键名 (从 model_0.pt 实读):
  model_state_dict:
    std:               (6,)            动作标准差
    actor.0.weight:    (256, 140)      actor 输入=140 (仅 proprio)
    actor.0.bias:      (256,)
    actor.2.weight:    (256, 256)
    actor.4.weight:    (256, 256)
    actor.6.weight:    (6, 256)        最后一层 Linear 在 Sequential 内
    critic.0.weight:   (256, 9)        critic 输入=9 (仅 privileged)
    critic.6.weight:   (1, 256)
  obs_norm_state_dict:
    _mean: (1, 140)    EmpiricalNormalization
    _var:  (1, 140)
    _std:  (1, 140)

关键: actor 只吃 proprio(140), critic 只吃 privileged(9)。
      actor 是 4 层 Linear 全在 Sequential 内 (含输出层), 不是 trunk+head 拆分。
      obs_norm 键名是 _mean/_var/_std, 不是 obs_rms.mean。
"""
import torch
import torch.nn as nn


class TeacherInferenceModel(nn.Module):
    """Teacher actor + obs_normalizer 合并推理模型.

    精确匹配 RSL-RL ActorCritic 的 actor 部分 (4 层 Linear in Sequential) +
    EmpiricalNormalization (_mean/_var/_std)。

    用法:
      model = TeacherInferenceModel.from_checkpoint(ckpt_path)
      action = model(obs_tensor)  # obs: (B, 140) → action: (B, 6)
    """

    def __init__(self, obs_dim: int = 140, action_dim: int = 6, hidden: tuple = (256, 256, 256)):
        super().__init__()
        # actor: Linear(140,256) ELU Linear(256,256) ELU Linear(256,256) ELU Linear(256,6)
        # 键名: actor.0, actor.2, actor.4, actor.6 (Sequential 索引)
        layers = []
        in_d = obs_dim
        for h in hidden:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ELU())
            in_d = h
        layers.append(nn.Linear(in_d, action_dim))  # 输出层在 Sequential 内
        self.actor = nn.Sequential(*layers)

        # EmpiricalNormalization (RSL-RL 键名: _mean/_var/_std)
        self.register_buffer("_mean", torch.zeros(1, obs_dim))
        self.register_buffer("_std", torch.ones(1, obs_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs(140) → normalize → actor → action(6)."""
        obs_norm = (obs - self._mean) / (self._std + 1e-8)
        return torch.tanh(self.actor(obs_norm))

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, obs_dim: int = 140, action_dim: int = 6) -> "TeacherInferenceModel":
        """从 RSL-RL checkpoint 加载, 并断言权重全部命中."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = cls(obs_dim=obs_dim, action_dim=action_dim)

        # 加载 actor 权重 (键名完全匹配: actor.0.weight 等)
        model_state = ckpt.get("model_state_dict", {})
        actor_keys = {k: v for k, v in model_state.items() if k.startswith("actor.")}
        missing, unexpected = model.load_state_dict(actor_keys, strict=False)

        # 断言: actor 权重必须全部加载 (排除 _mean/_std buffer)
        actor_missing = [k for k in missing if k.startswith("actor.")]
        assert len(actor_missing) == 0, \
            f"Teacher actor 权重缺失: {actor_missing} — checkpoint 结构不匹配"

        # 加载 obs normalizer (键名: _mean/_var/_std)
        obs_norm = ckpt.get("obs_norm_state_dict", {})
        if obs_norm:
            assert "_mean" in obs_norm and "_std" in obs_norm, \
                f"obs_norm 键名不匹配: {list(obs_norm.keys())}"
            model._mean = obs_norm["_mean"]
            model._std = obs_norm["_std"]

        model.eval()
        return model
