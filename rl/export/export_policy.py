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
    """导出 RSL-RL Teacher ActorCritic → ONNX (actor + obs_normalizer 合并).

    使用共享的 TeacherInferenceModel (精确匹配 checkpoint 键名)。
    """
    import torch
    from rl.env.kuafu_mjx_env import OBS_DIM, ACTION_DIM
    from rl.train.teacher_model import TeacherInferenceModel

    model = TeacherInferenceModel.from_checkpoint(ckpt_path, obs_dim=OBS_DIM)

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
    from rl.env.kuafu_mjx_env import OBS_DIM, OBS_DIM_BASE, ACTION_DIM, PRIVILEGED_DIM

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("student_state_dict", ckpt.get("model_state_dict", ckpt))

    # 动态推断 Student trunk 的维度
    hidden_dims = []
    i = 0
    while f"trunk.{i*2}.weight" in state:
        hidden_dims.append(state[f"trunk.{i*2}.weight"].shape[0])
        i += 1

    student = StudentPolicy(
        proprio_dim=OBS_DIM, history_obs_dim=OBS_DIM_BASE, history_len=50,
        action_dim=ACTION_DIM, latent_dim=PRIVILEGED_DIM,
        hidden_dims=tuple(hidden_dims),
    )

    missing, unexpected = student.load_state_dict(state, strict=False)
    assert len(missing) == 0, f"Student 权重缺失: {missing}"
    student.eval()

    # StudentPolicy.forward(proprio, history) → action
    dummy_proprio = torch.randn(1, OBS_DIM)
    dummy_history = torch.randn(1, 50, OBS_DIM_BASE)

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
