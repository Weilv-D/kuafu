# -*- coding: utf-8 -*-
"""
KUAFU Teacher 推理模型 — 匹配 RSL-RL 2.x checkpoint 的真实结构

checkpoint 实际键名 (从 model_0.pt 实读, 2-DOF 五杆 6 维动作):
  model_state_dict:
    std:               (ACTION_DIM,)   动作标准差 (6)
    actor.0.weight:    (H, OBS_DIM)    actor 输入=OBS_DIM (仅 proprio, 140)
    actor.0.bias:      (H,)
    actor.2.weight:    (H, H)
    actor.4.weight:    (H, H)
    actor.6.weight:    (ACTION_DIM, H) 最后一层 Linear 在 Sequential 内 (6)
    critic.0.weight:   (H, OBS_DIM+PRIVILEGED_DIM)  critic 输入=152 (proprio 140 + 特权 12)
    critic.6.weight:   (1, H)
  obs_norm_state_dict:
    _mean: (1, OBS_DIM)    EmpiricalNormalization (actor 本体感受)
    _var:  (1, OBS_DIM)
    _std:  (1, OBS_DIM)
  privileged_obs_norm_state_dict:           # critic 特权归一化 (本模块不消费)
    _mean: (1, OBS_DIM+PRIVILEGED_DIM)

 关键: actor 只吃 proprio(OBS_DIM=140), critic 吃 proprio+特权(152)。
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
      action = model(obs_tensor)  # obs: (B, OBS_DIM) → action: (B, ACTION_DIM)
    """

    def __init__(self, obs_dim: int = 140, action_dim: int = 6, hidden: tuple = (512, 512, 512)):
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
        """obs(OBS_DIM) → normalize → actor → action(ACTION_DIM)."""
        obs_norm = (obs - self._mean) / (self._std + 1e-8)
        return torch.tanh(self.actor(obs_norm))

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, obs_dim: int = 140, action_dim: int = 6) -> "TeacherInferenceModel":
        """从 RSL-RL checkpoint 加载, 并断言权重全部命中."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_state = ckpt.get("model_state_dict", {})

        # 动态推断隐藏层维度
        hidden = []
        i = 0
        while f"actor.{i*2}.weight" in model_state:
            weight = model_state[f"actor.{i*2}.weight"]
            if f"actor.{(i+1)*2}.weight" in model_state:
                hidden.append(weight.shape[0])
            i += 1

        model = cls(obs_dim=obs_dim, action_dim=action_dim, hidden=tuple(hidden))

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
