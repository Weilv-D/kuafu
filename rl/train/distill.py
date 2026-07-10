# -*- coding: utf-8 -*-
"""
KUAFU Student 蒸馏 — design.md §2.6 阶段 2

Teacher (特权信息) → Student (仅本体感受 + RMA latent):
  1. policy 对齐: 规范 DAgger (student 在环境中执行, teacher 给参考动作, 跨 iter 聚合数据集)
  2. adapter z 监督: MSE(student_z, teacher_z)

与早期在线 DAgger 的区别:
  - 早期仅用"当前 iter 的 env 一步采样"训练, 既偏离 Ross et al. (2011) 的 DAgger
    (应聚合 (s, a*) 数据集), 也令显存峰值随 num_envs 线性增长。
  - 现改为有上限回放缓冲 + DataLoader 分片采样: 跨 iter 聚合样本提升样本效率与抗遗忘,
    缓冲可驻 CPU、按 mini-batch 上 GPU, 显存峰值与 num_envs 解耦。

运行:
  rl/.venv/bin/python rl/train/distill.py \
    --run_name garlic --teacher_ckpt rl/checkpoints/garlic/teacher/model_3999.pt

产出:
  rl/checkpoints/<run_name>/student/model_final.pt
  rl/checkpoints/<run_name>/student/events.out.tfevents.*   (TensorBoard)
  rl/checkpoints/<run_name>/student/replay_<iter>.pt         (定期样本回放, 离线分析)
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
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from rl.train.seed_utils import seed_all, capture_provenance
from rl.train import dlpack_utils as dlu
from rl.train.train_config import DISTILL


class DAggerReplayBuffer:
    """有上限 ring 回放缓冲, 跨 iter 聚合 (s, a*, z*).

    缓冲驻 buffer_device (默认 cpu) 以控显存; 训练时按 mini-batch 上 GPU。
    capacity 之外的旧样本按 ring 覆盖, 保证近期交互比例 (mixed DAgger 思路)。
    """

    def __init__(self, capacity, proprio_dim, history_shape, action_dim, z_dim, device="cpu"):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.cursor = 0
        self.size = 0
        c = self.capacity
        self.proprio = torch.zeros(c, proprio_dim, device=self.device)
        self.history = torch.zeros(c, *history_shape, device=self.device)
        self.action = torch.zeros(c, action_dim, device=self.device)
        self.z = torch.zeros(c, z_dim, device=self.device)

    def add(self, proprio_b, history_b, action_b, z_b):
        n = int(proprio_b.shape[0])
        if n == 0:
            return
        idx = torch.arange(self.cursor, self.cursor + n, dtype=torch.long) % self.capacity
        self.proprio[idx] = proprio_b.to(self.device)
        self.history[idx] = history_b.to(self.device)
        self.action[idx] = action_b.to(self.device)
        self.z[idx] = z_b.to(self.device)
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def dataset(self):
        n = self.size
        return TensorDataset(self.proprio[:n], self.history[:n], self.action[:n], self.z[:n])


def distill(
    teacher_ckpt: str,
    num_envs: int = 1024,
    iterations: int = 1000,
    log_dir: str = "rl/checkpoints",
    smoke_test: bool = False,
    run_name: str = None,
    seed: int = 42,
    device: str = "cuda",
    buffer_device: str = DISTILL["buffer_device"],
    buffer_capacity: int = DISTILL["buffer_capacity"],
    train_batches: int = DISTILL["train_batches"],
    mini_batch_size: int = DISTILL["mini_batch_size"],
    max_grad_norm: float = DISTILL["max_grad_norm"],
    z_loss_weight: float = DISTILL["z_loss_weight"],
    log_tb: bool = True,
    save_replay_every: int = 200,
):
    """Student 规范 DAgger 蒸馏.

    Student = trunk(proprio + z) + adapter(history→z) + policy_head(trunk+z→action)
    Teacher 已训练好 (actor 吃 proprio+z=149), student 用历史推测特权 z, 拟合 teacher 动作。
    """
    from rl.env.kuafu_mjx_env import (
        KuafuMjxEnv, OBS_DIM, OBS_DIM_BASE, RMA_STATIC_DIM, ACTION_DIM, ACTOR_OBS_DIM)
    from rl.train.networks import StudentPolicy, count_parameters
    from rl.train.teacher_model import TeacherInferenceModel

    seed_all(seed)
    device = dlu.resolve_device(device)
    buffer_device = dlu.resolve_device(buffer_device) if buffer_device != "cpu" else "cpu"

    print("=" * 60)
    print("KUAFU Student 蒸馏 (design.md §2.6 阶段 2, 规范 DAgger)")
    print("=" * 60)
    print(f"  seed={seed}  device={device}  buffer_device={buffer_device}")
    print(f"  buffer_capacity={buffer_capacity}  train_batches/iter={train_batches}  "
          f"mini_batch={mini_batch_size}  max_grad_norm={max_grad_norm}  z_w={z_loss_weight}")

    # ---- DLPack 零拷贝契约守卫 (启动期一次) ----
    dlu.verify_dlpack_zero_copy(device)

    # ---- 加载 Teacher (actor 吃 proprio+z = 149, 部署时 z 由 adapter 预测) ----
    print(f"  加载 Teacher: {teacher_ckpt}")
    try:
        teacher = TeacherInferenceModel.from_checkpoint(
            teacher_ckpt, obs_dim=ACTOR_OBS_DIM).to(device)
    except (FileNotFoundError, RuntimeError, AssertionError) as e:
        print(f"❌ Teacher checkpoint 加载失败: {e}")
        sys.exit(1)
    teacher.eval()
    print(f"  Teacher 推理模型就绪 (actor {ACTOR_OBS_DIM}维 [proprio140+z9] + obs_normalizer)")

    # ---- 创建 Student 网络 (动态匹配 Teacher 隐藏层维度) ----
    teacher_hidden = [layer.out_features for layer in teacher.actor[:-1] if isinstance(layer, nn.Linear)]
    student = StudentPolicy(
        proprio_dim=OBS_DIM,
        history_obs_dim=OBS_DIM_BASE,
        history_len=50,
        action_dim=ACTION_DIM,
        latent_dim=RMA_STATIC_DIM,
        hidden_dims=tuple(teacher_hidden),
    ).to(device)

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

    rng = jax.random.PRNGKey(seed)
    state = reset_fn(jax.random.split(rng, num_envs))
    rng = jax.random.fold_in(rng, 0xABCD)  # 与初始 reset 解耦, 保证后续 reset 用不同种子

    n_iter = 5 if smoke_test else iterations
    # 滚动历史缓冲在推理设备 (GPU) 上, 供 student 实时推断; 训练样本另存 CPU 回放缓冲
    roll_history = torch.zeros(num_envs, 50, OBS_DIM_BASE, device=device)
    replay = DAggerReplayBuffer(
        capacity=buffer_capacity,
        proprio_dim=OBS_DIM,
        history_shape=(50, OBS_DIM_BASE),
        action_dim=ACTION_DIM,
        z_dim=RMA_STATIC_DIM,
        device=buffer_device,
    )

    writer = SummaryWriter(log_dir=os.path.join(PROJ_ROOT, log_dir, run_name, "student")) \
        if log_tb else None

    print(f"\n开始 DAgger 蒸馏: {n_iter} iterations")
    t0 = time.time()

    for it in range(n_iter):
        # 1. 采集当前环境状态 (JAX GPU → PyTorch GPU 零拷贝)
        obs = state.obs
        actor_obs_jax = obs["state"]             # (num_envs, 149) = proprio140 + static_z9

        # student 吃原始 proprio(140), adapter 从历史预测 z(9); teacher 吃完整 149
        proprio_torch = dlu.to_torch(actor_obs_jax[:, :OBS_DIM])    # (num_envs, 140)
        static_z_torch = dlu.to_torch(actor_obs_jax[:, OBS_DIM:])  # (num_envs, 9) 真值 z

        # 时序对齐: 推理动作前先更新滚动历史 (取 proprio 末 35 维 = 最近 base obs)
        current_base = proprio_torch[:, -OBS_DIM_BASE:]
        roll_history = torch.roll(roll_history, shifts=-1, dims=1)
        roll_history[:, -1, :] = current_base

        # student 推理动作 (在 eval 模式下只返回 action)
        student.eval()
        with torch.no_grad():
            student_action = student(proprio_torch, roll_history)

        # 快照本次执行所用的历史 (独立于后续 done-reset 的就地清零), 保证回放 (s,a*) 配对正确
        hist_sample = roll_history.detach().clone()

        # 环境物理步进
        jax_action = dlu.to_jax(student_action)
        state = step_fn(state, jax_action)

        # 检查 done 环境并进行 Selective Auto-Reset (与 train.py 一致)
        done_jax = state.done
        done_any = bool(jax.device_get(done_jax.any()))
        if done_any:
            rng, reset_rng = jax.random.split(rng)
            reset_state = reset_fn(jax.random.split(reset_rng, num_envs))
            done_mask = done_jax.astype(jp.bool_)
            state = jax.tree_util.tree_map(
                lambda cur, new: jp.where(
                    done_mask.reshape((-1,) + (1,) * (cur.ndim - 1)), new, cur),
                state, reset_state)

            # 同步重置 done 环境的滚动历史 (避免跨 episode 污染)
            done_mask_torch = dlu.to_torch(done_jax).bool()
            roll_history[done_mask_torch] = 0.0

        # 2. Teacher 给参考动作 (Teacher actor 吃 proprio+z = 149)
        with torch.no_grad():
            teacher_action = teacher(dlu.to_torch(actor_obs_jax))

        # 3. 聚合本 iter 样本到回放缓冲 (跨 iter 累积)
        replay.add(proprio_torch.detach(), hist_sample,
                   teacher_action.detach(), static_z_torch.detach())

        # 4. 从回放缓冲分片采样训练 (DataLoader + 每 iter 固定种子保证可复现)
        student.train()
        action_loss_sum, z_loss_sum, grad_norm_sum = 0.0, 0.0, 0.0
        gen = torch.Generator().manual_seed(seed + it)
        loader = DataLoader(replay.dataset(), batch_size=mini_batch_size,
                            shuffle=True, generator=gen, drop_last=False)
        n_batches = 0
        for p_b, h_b, t_act_b, z_b in loader:
            if n_batches >= train_batches:
                break
            p_b = p_b.to(device)
            h_b = h_b.to(device)
            t_act_b = t_act_b.to(device)
            z_b = z_b.to(device)

            pred_action, pred_z = student(p_b, h_b)

            # Multi-task Loss (Action MSE + Latent MSE)
            # z 监督静态环境外因 (9 维), 瞬态 active_push 不进 latent
            loss_action = mse_loss(pred_action, t_act_b)
            loss_z = mse_loss(pred_z, z_b)
            loss = loss_action + z_loss_weight * loss_z

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            optimizer.step()

            action_loss_sum += loss_action.item()
            z_loss_sum += loss_z.item()
            grad_norm_sum += grad_norm.item()
            n_batches += 1

        if writer is not None:
            writer.add_scalar("train/action_loss", action_loss_sum / max(1, n_batches), it)
            writer.add_scalar("train/z_loss", z_loss_sum / max(1, n_batches), it)
            writer.add_scalar("train/grad_norm", grad_norm_sum / max(1, n_batches), it)
            writer.add_scalar("train/buffer_size", replay.size, it)
            # 蒸馏质量: student 与 teacher 动作差的分布 (周期性)
            if it % 50 == 0:
                with torch.no_grad():
                    diff = (pred_action.detach() - t_act_b).abs().mean(dim=0)
                for a in range(diff.shape[0]):
                    writer.add_scalar(f"action_diff/act{a}", diff[a].item(), it)

        if it % 50 == 0 or it == n_iter - 1:
            print(f"  iter {it:4d}/{n_iter}: action_loss={action_loss_sum/max(1,n_batches):.6f}, "
                  f"z_loss={z_loss_sum/max(1,n_batches):.6f}, "
                  f"grad_norm={grad_norm_sum/max(1,n_batches):.4f}, "
                  f"buffer={replay.size}")

        # 定期保存样本回放供离线分析
        if save_replay_every and it > 0 and it % save_replay_every == 0 and not smoke_test:
            _save_replay(replay, run_name, log_dir, it, num_samples=64)

    elapsed = time.time() - t0
    print(f"\n✅ 蒸馏完成: {elapsed:.1f}s")

    if writer is not None:
        writer.flush()
        writer.close()

    if not smoke_test:
        save_dir = os.path.join(PROJ_ROOT, log_dir, run_name, "student")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "model_final.pt")
        # 解析 teacher iter (model_{iter}.pt)
        m = re.search(r"model_(\d+)\.pt", teacher_ckpt)
        teacher_iter = int(m.group(1)) if m else None
        torch.save({
            "model_state_dict": student.state_dict(),
            "student_state_dict": student.state_dict(),
            "iter": n_iter,
            "seed": seed,
            "provenance": capture_provenance(),
            "buffer_capacity": buffer_capacity,
            "max_grad_norm": max_grad_norm,
            "z_loss_weight": z_loss_weight,
            "teacher_ckpt": os.path.relpath(teacher_ckpt, PROJ_ROOT),
            "teacher_run": run_name,
            "teacher_iter": teacher_iter,
            "distill_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, save_path)
        print(f"   Checkpoint: {save_path}")
        print(f"   导出: rl/.venv/bin/python rl/export/export_policy.py --ckpt {save_path} --mode student")


def _save_replay(replay, run_name, log_dir, it, num_samples=64):
    save_dir = os.path.join(PROJ_ROOT, log_dir, run_name, "student")
    os.makedirs(save_dir, exist_ok=True)
    n = min(num_samples, replay.size)
    idx = torch.randperm(replay.size)[:n]
    torch.save({
        "iter": it,
        "proprio": replay.proprio[idx].cpu(),
        "history": replay.history[idx].cpu(),
        "teacher_action": replay.action[idx].cpu(),
        "static_z": replay.z[idx].cpu(),
    }, os.path.join(save_dir, f"replay_{it}.pt"))


def main():
    parser = argparse.ArgumentParser(description="KUAFU Student 蒸馏")
    parser.add_argument("--teacher_ckpt", required=True, help="Teacher checkpoint 路径 (.pt)")
    parser.add_argument("--run_name", type=str, required=True,
                        help="训练代号(须与 teacher 一致),产物存至 rl/checkpoints/<run_name>/student/")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (torch/numpy/random/JAX 同源)")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备 (cuda/cpu)")
    parser.add_argument("--buffer_device", type=str, default=DISTILL["buffer_device"], help="回放缓冲设备 (默认 cpu 控显存)")
    parser.add_argument("--buffer_capacity", type=int, default=DISTILL["buffer_capacity"], help="回放缓冲上限")
    parser.add_argument("--train_batches", type=int, default=DISTILL["train_batches"], help="每 iter 训练 batch 数")
    parser.add_argument("--mini_batch_size", type=int, default=DISTILL["mini_batch_size"])
    parser.add_argument("--max_grad_norm", type=float, default=DISTILL["max_grad_norm"], help="梯度裁剪阈值")
    parser.add_argument("--z_loss_weight", type=float, default=DISTILL["z_loss_weight"], help="latent MSE 损失权重")
    parser.add_argument("--no_tb", action="store_true", help="关闭 TensorBoard 日志")
    parser.add_argument("--save_replay_every", type=int, default=200, help="样本回放保存间隔(0=关)")
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
        seed=args.seed,
        device=args.device,
        buffer_device=args.buffer_device,
        buffer_capacity=args.buffer_capacity,
        train_batches=args.train_batches,
        mini_batch_size=args.mini_batch_size,
        max_grad_norm=args.max_grad_norm,
        z_loss_weight=args.z_loss_weight,
        log_tb=not args.no_tb,
        save_replay_every=args.save_replay_every,
    )


if __name__ == "__main__":
    main()
