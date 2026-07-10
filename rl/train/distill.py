# -*- coding: utf-8 -*-
"""
KUAFU Student 蒸馏 — design.md §2.6 阶段 2

Teacher (特权信息) → Student (仅本体感受 + RMA latent):
  1. policy 对齐: DAgger (student 在环境中执行, teacher 给参考动作)
  2. adapter z 监督: MSE(student_z, teacher_z) [阶段 2 后期加入]

运行:
  rl/.venv/bin/python rl/train/distill.py \
    --run_name garlic --teacher_ckpt rl/checkpoints/garlic/teacher/model_3999.pt

产出:
  rl/checkpoints/<run_name>/student/model_final.pt
"""
import os
import sys
import argparse
import time
import re

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ---- DLPack 零拷贝辅助函数 ----
def to_torch(x):
    """JAX DeviceArray → torch.Tensor (DLPack 零拷贝)."""
    return torch.utils.dlpack.from_dlpack(x)


def to_jax(t):
    """torch.Tensor → JAX DeviceArray (DLPack 零拷贝)."""
    return jax.dlpack.from_dlpack(t.contiguous())


def distill(
    teacher_ckpt: str,
    num_envs: int = 1024,
    iterations: int = 1000,
    log_dir: str = "rl/checkpoints",
    smoke_test: bool = False,
    run_name: str = None,
):
    """Student DAgger 蒸馏.

    design.md §2.5:
      Student = trunk(proprio + z) + adapter(history→z) + policy_head(trunk+z→action)
      Teacher 已训练好 (actor 仅吃 proprio), student 用历史推测特权 z, 拟合 teacher 动作。
    """
    from rl.env.kuafu_mjx_env import KuafuMjxEnv, OBS_DIM, OBS_DIM_BASE, PRIVILEGED_DIM, ACTION_DIM
    from rl.train.networks import StudentPolicy, count_parameters
    from rl.train.teacher_model import TeacherInferenceModel

    print("=" * 60)
    print("KUAFU Student 蒸馏 (design.md §2.6 阶段 2)")
    print("=" * 60)

    # ---- 加载 Teacher (actor 只吃 proprio, 不含特权) ----
    print(f"  加载 Teacher: {teacher_ckpt}")
    teacher = TeacherInferenceModel.from_checkpoint(teacher_ckpt, obs_dim=OBS_DIM).cuda()
    teacher.eval()
    print(f"  Teacher 推理模型就绪 (actor {OBS_DIM}维 proprio + obs_normalizer)")

    # ---- 创建 Student 网络 (动态匹配 Teacher 隐藏层维度) ----
    teacher_hidden = [layer.out_features for layer in teacher.actor[:-1] if isinstance(layer, nn.Linear)]
    student = StudentPolicy(
        proprio_dim=OBS_DIM,
        history_obs_dim=OBS_DIM_BASE,
        history_len=50,
        action_dim=ACTION_DIM,
        latent_dim=PRIVILEGED_DIM,
        hidden_dims=tuple(teacher_hidden),
    ).cuda()

    # 载入 Teacher actor 的 proprio normalizer 到 Student (z 的归一化由 RMA 自适应学习)
    student.obs_mean.copy_(teacher._mean)
    student.obs_std.copy_(teacher._std)
    print(f"  Student 参数: {count_parameters(student):,}")

    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    # ---- 环境 (teacher 模式, 获取特权 obs 供 teacher 推理) ----
    env = KuafuMjxEnv(teacher=True, num_envs=num_envs)
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))

    state = reset_fn(jax.random.split(jax.random.PRNGKey(42), num_envs))

    n_iter = 5 if smoke_test else iterations
    # 历史缓冲放在 GPU 显存上 (RMA adapter 消费 50 步 base_obs)
    history_buffer = torch.zeros(num_envs, 50, OBS_DIM_BASE, device="cuda")

    print(f"\n开始 DAgger 蒸馏: {n_iter} iterations")
    t0 = time.time()

    for it in range(n_iter):
        # 1. 采集当前环境状态 (JAX GPU → PyTorch GPU 零拷贝)
        obs = state.obs
        proprio_jax = obs["state"]               # (num_envs, OBS_DIM)
        privileged_jax = obs["privileged_state"] # (num_envs, PRIVILEGED_DIM)

        proprio_torch = to_torch(proprio_jax)
        privileged_torch = to_torch(privileged_jax)

        # 时序对齐: 推理动作前先更新 history
        current_base = proprio_torch[:, -OBS_DIM_BASE:]
        history_buffer = torch.roll(history_buffer, shifts=-1, dims=1)
        history_buffer[:, -1, :] = current_base

        # student 推理动作 (在 eval 模式下只返回 action)
        student.eval()
        with torch.no_grad():
            student_action = student(proprio_torch, history_buffer)

        # 环境物理步进
        jax_action = to_jax(student_action)
        state = step_fn(state, jax_action)

        # 检查 done 环境并进行 Selective Auto-Reset (与 train.py 一致)
        done_jax = state.done
        done_any = bool(jax.device_get(done_jax.any()))
        if done_any:
            reset_state = reset_fn(jax.random.split(jax.random.PRNGKey(it * 1000), num_envs))
            done_mask = done_jax.astype(jp.bool_)
            state = jax.tree_util.tree_map(
                lambda cur, new: jp.where(
                    done_mask.reshape((-1,) + (1,) * (cur.ndim - 1)), new, cur),
                state, reset_state)

            # 同步重置 done 环境的 history_buffer (避免跨 episode 污染)
            done_mask_torch = to_torch(done_jax).bool()
            history_buffer[done_mask_torch] = 0.0

        # 2. Teacher 给参考动作 (Teacher actor 只吃 proprio, 无特权泄漏)
        with torch.no_grad():
            teacher_action = teacher(proprio_torch)

        # 3. Student 训练 (将 1024 样本乱序切分为大小为 256 的 mini-batches)
        student.train()

        indices = np.arange(num_envs)
        np.random.shuffle(indices)

        mini_batch_size = 256
        action_loss_sum = 0.0
        z_loss_sum = 0.0

        for start_idx in range(0, num_envs, mini_batch_size):
            end_idx = start_idx + mini_batch_size
            batch_idx = indices[start_idx:end_idx]

            p_b = proprio_torch[batch_idx]
            h_b = history_buffer[batch_idx]
            priv_b = privileged_torch[batch_idx]
            t_act_b = teacher_action[batch_idx]

            # Forward (返回 action, z)
            pred_action, pred_z = student(p_b, h_b)

            # Multi-task Loss (Action MSE + Latent MSE)
            loss_action = mse_loss(pred_action, t_act_b)
            loss_z = mse_loss(pred_z, priv_b)
            loss = loss_action + 5.0 * loss_z

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            action_loss_sum += loss_action.item()
            z_loss_sum += loss_z.item()

        if it % 50 == 0 or it == n_iter - 1:
            n_batches = num_envs / mini_batch_size
            print(f"  iter {it:4d}/{n_iter}: action_loss={action_loss_sum/n_batches:.6f}, "
                  f"z_loss={z_loss_sum/n_batches:.6f}")

    elapsed = time.time() - t0
    print(f"\n✅ 蒸馏完成: {elapsed:.1f}s")

    if not smoke_test:
        save_dir = os.path.join(PROJ_ROOT, log_dir, run_name, "student")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "model_final.pt")
        # 解析 teacher iter (model_{iter}.pt)
        m = re.search(r"model_(\d+)\.pt", teacher_ckpt)
        teacher_iter = int(m.group(1)) if m else None
        torch.save({
            "model_state_dict": student.state_dict(),  # 标准 key 名
            "student_state_dict": student.state_dict(),  # 兼容旧导出脚本
            "iter": n_iter,
            "teacher_ckpt": os.path.relpath(teacher_ckpt, PROJ_ROOT),  # 源 teacher .pt
            "teacher_run": run_name,           # codename, 与目录一致
            "teacher_iter": teacher_iter,      # teacher 迭代数
            "distill_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, save_path)
        print(f"   Checkpoint: {save_path}")
        print(f"   导出: rl/.venv/bin/python rl/export/export_policy.py --ckpt {save_path} --mode student")


def main():
    parser = argparse.ArgumentParser(description="KUAFU Student 蒸馏")
    parser.add_argument("--teacher_ckpt", required=True, help="Teacher checkpoint 路径 (.pt)")
    parser.add_argument("--run_name", type=str, required=True,
                        help="训练代号(须与 teacher 一致),产物存至 rl/checkpoints/<run_name>/student/")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.teacher_ckpt):
        print(f"❌ Teacher checkpoint 不存在: {args.teacher_ckpt}")
        sys.exit(1)

    # codename 一致性校验: 从 teacher_ckpt 路径解析 codename, 与 --run_name 比对
    parts = os.path.abspath(args.teacher_ckpt).replace("\\", "/").split("/")
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        inferred = parts[idx + 1] if idx + 1 < len(parts) else None
        if inferred and inferred != args.run_name:
            print(f"❌ --run_name({args.run_name}) 与 teacher 路径中的代号({inferred})不一致")
            print(f"   teacher_ckpt: {args.teacher_ckpt}")
            sys.exit(1)

    distill(
        teacher_ckpt=args.teacher_ckpt,
        num_envs=args.num_envs,
        iterations=args.iterations,
        smoke_test=args.smoke_test,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
