# -*- coding: utf-8 -*-
"""
KUAFU Student 蒸馏 — design.md §2.6 阶段 2

Teacher (特权信息) → Student (仅本体感受 + RMA latent):
  1. policy 对齐: DAgger (student 在环境中执行, teacher 给参考动作)
  2. adapter z 监督: MSE(student_z, teacher_z) [阶段 2 后期加入]

运行:
  rl/.venv/bin/python rl/train/distill.py --teacher_ckpt rl/checkpoints/teacher_*/model_500.pt

产出:
  rl/checkpoints/student_{timestamp}/model_final.pt
"""
import os
import sys
import argparse
import time

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

import jax
import jax.numpy as jp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def distill(
    teacher_ckpt: str,
    num_envs: int = 1024,
    iterations: int = 500,
    log_dir: str = "rl/checkpoints",
    smoke_test: bool = False,
):
    """Student DAgger 蒸馏.

    design.md §2.5:
      Student = trunk(proprio 140) + adapter(history→z 5) + policy_head(trunk+z→action 6)
      Teacher 已训练好, 在特权信息下给出参考动作; student 拟合 teacher 动作。

    本轮实现 DAgger:
      1. Student 在环境中执行动作 → 采集 (obs, proprio, history) 轨迹
      2. Teacher 用特权 obs 推理 → 参考动作
      3. Student 监督拟合 teacher 动作 (MSE loss)
    """
    from rl.env.kuafu_mjx_env import KuafuMjxEnv, OBS_DIM, PRIVILEGED_DIM, ACTION_DIM
    from rl.train.networks import StudentPolicy, count_parameters

    print("=" * 60)
    print("KUAFU Student 蒸馏 (design.md §2.6 阶段 2)")
    print("=" * 60)

    # ---- 加载 Teacher checkpoint ----
    print(f"  加载 Teacher: {teacher_ckpt}")
    teacher_ckpt_data = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
    teacher_model_state = teacher_ckpt_data.get("model_state_dict", {})
    teacher_obs_norm_state = teacher_ckpt_data.get("obs_norm_state_dict", {})
    print(f"  Teacher keys: {list(teacher_ckpt_data.keys())}")

    # 构造 Teacher actor (RSL-RL ActorCritic 结构) 用于推理参考动作
    class TeacherActor(torch.nn.Module):
        def __init__(self, obs_dim, action_dim, hidden=(256, 256, 256)):
            super().__init__()
            layers = []
            in_d = obs_dim
            for h in hidden:
                layers.append(nn.Linear(in_d, h))
                layers.append(nn.ELU())
                in_d = h
            self.actor = nn.Sequential(*layers)
            self.actor_mean = nn.Linear(in_d, action_dim)
            self.obs_mean = nn.Parameter(torch.zeros(obs_dim))
            self.obs_std = nn.Parameter(torch.ones(obs_dim))

        def forward(self, obs):
            obs_n = (obs - self.obs_mean) / (self.obs_std + 1e-8)
            return torch.tanh(self.actor_mean(self.actor(obs_n)))

    # Teacher obs = proprio(140) + privileged(9) = 149
    teacher = TeacherActor(OBS_DIM + PRIVILEGED_DIM, ACTION_DIM).cuda()
    renamed = {}
    for k, v in teacher_model_state.items():
        if k.startswith("actor."):
            renamed[k[len("actor."):]] = v
        elif k.startswith("actor_mean."):
            renamed[k] = v
    teacher.load_state_dict(renamed, strict=False)
    if teacher_obs_norm_state and "obs_rms.mean" in teacher_obs_norm_state:
        teacher.obs_mean.data = teacher_obs_norm_state["obs_rms.mean"].cuda()
        teacher.obs_std.data = torch.sqrt(teacher_obs_norm_state["obs_rms.var"]).cuda()
    teacher.eval()
    print("  Teacher 推理模型就绪")

    # ---- 创建 Student 网络 ----
    student = StudentPolicy(
        proprio_dim=OBS_DIM,
        history_obs_dim=35,
        history_len=50,
        action_dim=ACTION_DIM,
        latent_dim=5,
        hidden_dims=(256, 256, 256),
    ).cuda()
    print(f"  Student 参数: {count_parameters(student):,}")

    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    # ---- 环境 (teacher 模式, 获取特权 obs 供 teacher 推理) ----
    env = KuafuMjxEnv(teacher=True, num_envs=num_envs)
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))

    state = reset_fn(jax.random.split(jax.random.PRNGKey(42), num_envs))

    n_iter = 5 if smoke_test else iterations
    batch_size = min(num_envs, 256)
    history_buffer = np.zeros((num_envs, 50, 35), dtype=np.float32)  # 滑动历史窗口

    print(f"\n开始 DAgger 蒸馏: {n_iter} iterations")
    t0 = time.time()

    for it in range(n_iter):
        # 1. 从环境采集 batch_size 条数据
        collected = 0
        proprio_list, history_list, teacher_full_obs_list = [], [], []

        while collected < batch_size:
            obs = state.obs
            proprio_jax = obs["state"]       # (num_envs, 140)
            priv_jax = obs["privileged_state"]  # (num_envs, 9)

            # 构建 teacher 输入: proprio + privileged
            teacher_input = jp.concatenate([proprio_jax, priv_jax], axis=-1)  # (num_envs, 149)

            # student 推理动作 (用于环境执行, DAgger)
            proprio_np = np.array(proprio_jax)
            history_np = history_buffer.copy()
            with torch.no_grad():
                student_action = student(
                    torch.from_numpy(proprio_np).float().cuda(),
                    torch.from_numpy(history_np).float().cuda(),
                )
            action_np = student_action.cpu().numpy()

            # 环境步进
            jax_action = jp.array(action_np)
            state = step_fn(state, jax_action)

            # auto-reset done 环境 (与 train.py 一致)
            done_jax = state.done
            if bool(jax.device_get(done_jax.any())):
                reset_state = reset_fn(jax.random.split(jax.random.PRNGKey(it * 1000 + collected), num_envs))
                done_mask = done_jax.astype(jax.numpy.bool_)
                state = jax.tree_util.tree_map(
                    lambda cur, new: jax.numpy.where(
                        done_mask.reshape((-1,) + (1,) * (cur.ndim - 1)), new, cur),
                    state, reset_state)

            # 更新历史缓冲 (用最新 proprio 的前 35 维 base obs)
            # 简化: 从 140 维 obs 取最后 35 维作为当前步 base_obs
            current_base = proprio_np[:, -35:]  # (num_envs, 35)
            history_buffer = np.roll(history_buffer, -1, axis=1)
            history_buffer[:, -1, :] = current_base

            # 采集
            proprio_list.append(proprio_np)
            history_list.append(history_np)
            teacher_full_obs_list.append(np.array(teacher_input))
            collected += num_envs

        # 2. Teacher 给参考动作
        proprio_batch = np.concatenate(proprio_list, axis=0)[:batch_size]
        history_batch = np.concatenate(history_list, axis=0)[:batch_size]
        teacher_input_batch = np.concatenate(teacher_full_obs_list, axis=0)[:batch_size]

        with torch.no_grad():
            teacher_action = teacher(
                torch.from_numpy(teacher_input_batch).float().cuda())

        # 3. Student 训练 (DAgger: 拟合 teacher 动作)
        student_action = student(
            torch.from_numpy(proprio_batch).float().cuda(),
            torch.from_numpy(history_batch).float().cuda(),
        )
        loss = mse_loss(student_action, teacher_action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 50 == 0 or it == n_iter - 1:
            print(f"  iter {it:4d}/{n_iter}: action_loss={loss.item():.6f}")

    elapsed = time.time() - t0
    print(f"\n✅ 蒸馏完成: {elapsed:.1f}s")

    if not smoke_test:
        save_dir = os.path.join(PROJ_ROOT, log_dir, f"student_{int(time.time())}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "model_final.pt")
        torch.save({
            "model_state_dict": student.state_dict(),  # 标准 key 名
            "student_state_dict": student.state_dict(),  # 兼容旧导出脚本
            "iter": n_iter,
        }, save_path)
        print(f"   Checkpoint: {save_path}")
        print(f"   导出: rl/.venv/bin/python rl/export/export_policy.py --ckpt {save_path} --mode student")


def main():
    parser = argparse.ArgumentParser(description="KUAFU Student 蒸馏")
    parser.add_argument("--teacher_ckpt", required=True, help="Teacher checkpoint 路径")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.teacher_ckpt):
        print(f"❌ Teacher checkpoint 不存在: {args.teacher_ckpt}")
        sys.exit(1)

    distill(
        teacher_ckpt=args.teacher_ckpt,
        num_envs=args.num_envs,
        iterations=args.iterations,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
