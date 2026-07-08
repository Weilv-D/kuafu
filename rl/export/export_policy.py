# -*- coding: utf-8 -*-
"""
KUAFU 策略导出 — PyTorch → ONNX (design.md §六 部署流水线)

本轮交付导出骨架。训练产出 policy.pt 后执行本脚本，导出 policy.onnx 供
树莓派 rl_policy_node (ONNX Runtime, <1ms) 部署。

部署链路 (design.md §六):
  WSL2 训练 → policy.pt → torch.onnx.export → policy.onnx → scp → Pi5

运行 (训练后): python rl/export/export_policy.py --ckpt checkpoints/policy.pt
依赖 (导出时): torch, onnx, onnxruntime (验证)
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import kuafu_physics as P


def export(ckpt_path: str, out_path: str = "policy.onnx"):
    """导出 PyTorch policy 为 ONNX.

    Args:
        ckpt_path: 训练产出的 policy.pt 路径
        out_path: 输出 policy.onnx 路径
    """
    import torch
    from rl.env import OBS_DIM, ACTION_DIM, RMA_LATENT_DIM

    # 加载训练好的 policy (结构见 train/train_config.py NETWORK)
    policy = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy.eval()

    # 构造 dummy 输入 (obs + RMA latent z)
    dummy_obs = torch.randn(1, OBS_DIM)
    dummy_z = torch.randn(1, RMA_LATENT_DIM)

    # 导出
    torch.onnx.export(
        policy, (dummy_obs, dummy_z), out_path,
        input_names=["obs", "z"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"✅ 导出: {out_path}")

    # 验证: 维度 / NaN 检查 (design.md §六 单元测试)
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    action = sess.run(None, {"obs": dummy_obs.numpy(), "z": dummy_z.numpy()})[0]
    assert action.shape == (1, ACTION_DIM), f"动作维度错: {action.shape}"
    assert not (action != action).any(), "动作含 NaN"
    assert action.min() >= -1.01 and action.max() <= 1.01, "动作超 [-1,1]"
    print(f"✅ 验证通过: shape={action.shape}, range=[{action.min():.3f}, {action.max():.3f}], 无 NaN")
    print(f"   下一步: scp {out_path} pi5:~/kuafu/models/ → M5 部署")


def main():
    parser = argparse.ArgumentParser(description="KUAFU 策略 ONNX 导出")
    parser.add_argument("--ckpt", required=True, help="policy.pt 路径")
    parser.add_argument("--out", default="policy.onnx", help="输出路径")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"❌ 找不到 {args.ckpt} — 训练产出 policy.pt 后再执行")
        sys.exit(1)
    export(args.ckpt, args.out)


if __name__ == "__main__":
    main()
