# -*- coding: utf-8 -*-
"""
KUAFU 策略导出 — PyTorch → ONNX (design.md §2.6 阶段 4 / §六 部署)

支持两种模式:
  --mode teacher:  RSL-RL ActorCritic checkpoint → ONNX (actor 部分)
  --mode student:  StudentPolicy checkpoint → ONNX (trunk + adapter + policy_head)

部署链路 (design.md §六):
  WSL2 训练 → model.pt → torch.onnx.export → policy.onnx → scp → Pi5
  Pi5: ONNX Runtime aarch64, MLP <1ms 推理, 50Hz 控制循环

运行:
  rl/.venv/bin/python rl/export/export_policy.py --ckpt model_500.pt --mode teacher
  rl/.venv/bin/python rl/export/export_policy.py --ckpt model_final.pt --mode student
"""
import os
import sys
import argparse

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

import kuafu_physics as P


def export_teacher(ckpt_path: str, out_path: str):
    """导出 RSL-RL Teacher ActorCritic → ONNX (仅 actor + obs_normalizer).

    RSL-RL checkpoint 结构:
      - model_state_dict: ActorCritic 权重 (actor.* / critic.* / log_std)
      - obs_norm_state_dict: EmpiricalNormalization 权重
    导出时合并 obs_normalizer 到 actor 前端, 保证推理时 obs 归一化一致。
    """
    import torch
    from rl.env.kuafu_mjx_env import OBS_DIM, ACTION_DIM

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    model_state = checkpoint.get("model_state_dict", {})
    obs_norm_state = checkpoint.get("obs_norm_state_dict", None)

    # 构造 actor + obs_normalizer 合并的推理模型
    class TeacherInferenceModel(torch.nn.Module):
        """obs → normalizer → actor_mean → tanh → action."""
        def __init__(self, obs_dim, action_dim, hidden=(256, 256, 256)):
            super().__init__()
            layers = []
            in_d = obs_dim
            for h in hidden:
                layers.append(torch.nn.Linear(in_d, h))
                layers.append(torch.nn.ELU())
                in_d = h
            self.actor = torch.nn.Sequential(*layers)
            self.actor_mean = torch.nn.Linear(in_d, action_dim)
            # EmpiricalNormalization (mean/var 缩放)
            self.obs_mean = torch.nn.Parameter(torch.zeros(obs_dim))
            self.obs_std = torch.nn.Parameter(torch.ones(obs_dim))

        def forward(self, obs):
            obs_norm = (obs - self.obs_mean) / (self.obs_std + 1e-8)
            h = self.actor(obs_norm)
            action = torch.tanh(self.actor_mean(h))
            return action

    model = TeacherInferenceModel(OBS_DIM, ACTION_DIM)

    # 加载 actor 权重 (RSL-RL key: actor.0.weight, actor_mean.weight)
    renamed = {}
    for k, v in model_state.items():
        # 去掉 "actor." 前缀以匹配我们的 self.actor Sequential
        if k.startswith("actor."):
            renamed[k[len("actor."):]] = v
        elif k.startswith("actor_mean."):
            renamed[k] = v
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    print(f"  Actor 权重: missing={len(missing)}, unexpected={len(unexpected)}")

    # 加载 obs normalizer
    if obs_norm_state:
        # EmpiricalNormalization: obs_rms.mean / obs_rms.var
        if "obs_rms.mean" in obs_norm_state:
            model.obs_mean.data = obs_norm_state["obs_rms.mean"]
            model.obs_std.data = torch.sqrt(obs_norm_state.get("obs_rms.var", torch.ones_like(model.obs_mean.data)))
            print("  obs_normalizer: 加载成功")
        else:
            print(f"  obs_normalizer: key 不匹配 {list(obs_norm_state.keys())[:5]}")

    model.eval()

    # 导出
    dummy_obs = torch.randn(1, OBS_DIM)
    torch.onnx.export(
        model, dummy_obs, out_path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"✅ Teacher 导出: {out_path}")
    _verify_onnx(out_path, OBS_DIM, ACTION_DIM)


def export_student(ckpt_path: str, out_path: str):
    """导出 StudentPolicy → ONNX (trunk + adapter + policy_head)."""
    import torch
    from rl.train.networks import StudentPolicy
    from rl.env.kuafu_mjx_env import OBS_DIM, ACTION_DIM

    student = StudentPolicy(
        proprio_dim=OBS_DIM, history_obs_dim=35, history_len=50,
        action_dim=ACTION_DIM, latent_dim=5,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("student_state_dict", ckpt)
    student.load_state_dict(state, strict=False)
    student.eval()

    # StudentPolicy.forward(proprio, history) → action
    dummy_proprio = torch.randn(1, OBS_DIM)
    dummy_history = torch.randn(1, 50, 35)

    torch.onnx.export(
        student, (dummy_proprio, dummy_history), out_path,
        input_names=["proprio", "history"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"proprio": {0: "batch"}, "history": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"✅ Student 导出: {out_path}")

    _verify_onnx(out_path, None, ACTION_DIM, inputs={"proprio": dummy_proprio.numpy(), "history": dummy_history.numpy()})


def _verify_onnx(onnx_path: str, obs_dim: int, action_dim: int, inputs=None):
    """ONNX 验证: 维度 / NaN / 范围 (design.md §六 单元测试)."""
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    if inputs is None:
        inputs = {"obs": np.random.randn(1, obs_dim).astype(np.float32)}

    action = sess.run(None, inputs)[0]

    assert action.shape[-1] == action_dim, f"动作维度错: {action.shape}"
    assert not np.isnan(action).any(), "动作含 NaN"
    assert action.min() >= -1.01 and action.max() <= 1.01, f"动作超范围: [{action.min():.3f}, {action.max():.3f}]"

    print(f"✅ 验证通过: shape={action.shape}, range=[{action.min():.3f}, {action.max():.3f}], 无 NaN")
    print(f"   下一步: scp {onnx_path} pi5:~/kuafu/models/ → 部署")


def main():
    parser = argparse.ArgumentParser(description="KUAFU 策略 ONNX 导出")
    parser.add_argument("--ckpt", required=True, help="Checkpoint 路径")
    parser.add_argument("--mode", choices=["teacher", "student"], default="teacher", help="导出模式")
    parser.add_argument("--out", default="policy.onnx", help="输出路径")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"❌ 找不到 {args.ckpt}")
        sys.exit(1)

    if args.mode == "teacher":
        export_teacher(args.ckpt, args.out)
    else:
        export_student(args.ckpt, args.out)


if __name__ == "__main__":
    main()
