# -*- coding: utf-8 -*-
"""
KUAFU Teacher PPO 训练入口 — design.md §2.6 阶段 1

MJX 环境 (JAX/GPU) → DirectVecEnv (DLPack 零拷贝) → RSL-RL 2.x PPO (PyTorch/GPU)
Teacher: critic 含特权信息 (friction/mass/COM/inertia), actor 仅本体感受。

运行:
  rl/.venv/bin/python rl/train/train.py --run_name garlic --num_envs 1024 --iterations 1000

产出:
  rl/checkpoints/<run_name>/run.json                 训练元数据
  rl/checkpoints/<run_name>/teacher/model_{iter}.pt  Teacher checkpoint
  rl/checkpoints/<run_name>/teacher/events.out.tfevents.*  TensorBoard
  rl/checkpoints/<run_name>/teacher/git/kuafu.diff   代码快照
"""
import os
import sys
import argparse
import time
import json
import glob

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import torch
from rl.train.seed_utils import seed_all, capture_provenance
from rl.train import dlpack_utils as dlu


def make_train_cfg() -> dict:
    """RSL-RL 2.x OnPolicyRunner 配置 — 全部源自 train_config (单一真相源)."""
    from rl.train.train_config import ALGORITHM, POLICY, RUN
    return {
        "algorithm": dict(ALGORITHM),
        "policy": dict(POLICY),
        "num_steps_per_env": RUN["num_steps_per_env"],
        "save_interval": RUN["save_interval"],
        "empirical_normalization": RUN["empirical_normalization"],
    }


class Curriculum:
    """全局课程: 按成功率滑动窗口连续提升 difficulty (DR/扰动强度).

    设计参考 terrain.py CurriculumController 与 ETH legged_gym: 训练初期即设难度下限,
    注入 DR + 随机推力(见 kuafu_mjx_env push), 避免策略过拟合标称参数、永久卡在
    difficulty=0 (原 per-episode ±0.05 在训练初期频繁跌倒时永不上升的缺陷)。
    """

    def __init__(self, start: float = 0.1, max_d: float = 1.0, step: float = 0.05,
                 window: int = 200, threshold: float = 0.8):
        self.d = start
        self.max_d = max_d
        self.step = step
        self.window = window
        self.threshold = threshold
        self._buf = []

    def update(self, successes):
        """successes: 本批 done 环境的存活成功标志列表 (bool)."""
        self._buf.extend(bool(s) for s in successes)
        if len(self._buf) > self.window:
            del self._buf[: len(self._buf) - self.window]
        if len(self._buf) >= self.window:
            rate = sum(self._buf) / len(self._buf)
            if rate >= self.threshold and self.d < self.max_d:
                self.d = min(self.max_d, self.d + self.step)


def main():
    parser = argparse.ArgumentParser(description="KUAFU Teacher PPO Training")
    from rl.train.train_config import RUN
    parser.add_argument("--num_envs", type=int, default=RUN["num_envs"], help="并行环境数")
    parser.add_argument("--iterations", type=int, default=RUN["iterations"], help="训练迭代数")
    parser.add_argument("--seed", type=int, default=RUN["seed"], help="随机种子")
    parser.add_argument("--run_name", type=str, required=True,
                        help="训练代号(如 garlic),产物存至 rl/checkpoints/<run_name>/teacher/")
    parser.add_argument("--log_dir", type=str, default="rl/checkpoints", help="checkpoint 根目录")
    parser.add_argument("--smoke_test", action="store_true", help="烟测模式 (5 iteration)")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练(传 .pt 路径,如 rl/checkpoints/garlic/teacher/model_3999.pt)")
    args = parser.parse_args()

    # 统一播种所有 RNG (torch/numpy/random 与 JAX 显式 key 同源)
    seed_all(args.seed)

    print("=" * 60)
    print("KUAFU Teacher PPO Training (design.md §2.6 阶段 1)")
    print("=" * 60)
    print(f"  并行环境: {args.num_envs}")
    print(f"  JAX 设备: {jax.devices()}")

    # ---- 创建环境 ----
    # 当前: 平地训练 (difficulty=0)。课程地形 (terrain.py CurriculumController)
    #       待平地 reward 收敛后在此处接入: 按 episode 成功率更新 difficulty。
    from rl.env.kuafu_mjx_env import (
        KuafuMjxEnv, OBS_DIM, PRIVILEGED_DIM, RMA_STATIC_DIM, TRANSIENT_DIM,
        ACTOR_OBS_DIM, CRITIC_PRIV_DIM, CRITIC_OBS_DIM)

    env = KuafuMjxEnv(teacher=True, num_envs=args.num_envs)

    # 解析统一计算设备 (无 GPU 时回退 CPU 并告警)
    device = dlu.resolve_device("cuda")
    # DLPack 零拷贝契约守卫 (启动期一次)
    dlu.verify_dlpack_zero_copy(device)

    # ---- 直接适配 rsl_rl 2.x (绕过 playground brax wrapper 的 info 结构限制) ----
    class DirectVecEnv:
        """JAX vmap 环境到 rsl_rl 2.x VecEnv 的直接适配器.

        绕过 playground 的 BraxAutoResetWrapper (其 auto-reset 会修改 info 结构导致
        scan pytree 不匹配), 直接用 JAX vmap + jax.lax.cond 做 auto-reset,
        通过 DLPack 与 PyTorch 零拷贝交换 GPU 张量。
        """
        def __init__(self, env, num_envs, seed, device="cuda"):
            self.env = env
            self.num_envs = num_envs
            self.num_actions = env.action_size
            self.num_obs = ACTOR_OBS_DIM                             # actor = proprio(140) + z(9)
            self.num_privileged_obs = CRITIC_PRIV_DIM if env._teacher else None  # critic 额外瞬态(3)
            self.device = device
            self.cfg = {"env_name": "kuafu", "num_envs": num_envs}
            self.max_episode_length = env._episode_length
            self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)

            # 课程: 全局成功率滑动窗口驱动 difficulty (避免 per-episode ±0.05 卡在 0,
            # 且训练初期即注入 DR + 随机推力, 防过拟合标称参数, ETH legged_gym 实践)
            self._curriculum = Curriculum(start=0.1, max_d=1.0, step=0.05,
                                          window=200, threshold=0.8)
            self._difficulty = jax.numpy.float32(self._curriculum.d)  # 标量全局难度

            self._reset_vmapped = jax.jit(jax.vmap(env.reset, in_axes=(0, 0)))
            self._step_vmapped = jax.jit(jax.vmap(env.step))

            self._rng = jax.random.PRNGKey(seed)
            diff_vec = jax.numpy.full(num_envs, self._difficulty)
            self._state = self._reset_vmapped(jax.random.split(self._rng, num_envs), diff_vec)

        def _to_torch(self, x):
            """JAX DeviceArray → torch.Tensor (DLPack 零拷贝契约)."""
            return dlu.to_torch(x, device=self.device)

        def _to_jax(self, t):
            """torch.Tensor → JAX DeviceArray (DLPack 零拷贝契约)."""
            return dlu.to_jax(t, device=None)

        def get_observations(self):
            obs = self._state.obs
            state_obs = self._to_torch(obs["state"]) if isinstance(obs, dict) else self._to_torch(obs)
            extras = {"observations": {}}
            if isinstance(obs, dict) and "privileged_state" in obs:
                # critic 吃 actor obs (149) + 瞬态特权 (3) = 152
                priv_obs = self._to_torch(obs["privileged_state"])
                extras["observations"]["critic"] = torch.cat([state_obs, priv_obs], dim=-1)
            return state_obs, extras

        def reset(self):
            """VecEnv 接口要求: 重置所有环境."""
            self._rng, reset_rng = jax.random.split(self._rng)
            # 使用当前全局 difficulty (标量广播为 per-env 向量)
            diff_vec = jax.numpy.full(self.num_envs, self._difficulty)
            self._state = self._reset_vmapped(jax.random.split(reset_rng, self.num_envs), diff_vec)
            self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            return self.get_observations()

        def step(self, action):
            jax_action = self._to_jax(action)
            self._state = self._step_vmapped(self._state, jax_action)

            # 在 auto-reset 前读取 done/reward/metrics (reset 后会清零)
            done_jax = self._state.done
            reward_jax = self._state.reward
            metrics_jax = self._state.metrics
            fallen_jax = metrics_jax.get("fallen", jax.numpy.zeros_like(done_jax))

            # auto-reset done 环境 (保持 JAX array 在 GPU 上)
            done_any = jax.device_get(done_jax.any())
            if done_any:
                self._rng, reset_rng = jax.random.split(self._rng)
                
                # 全局课程: 统计 done 环境是否"存活成功"(timeout 且未倒下), 更新成功率
                # 滑动窗口 → 连续提升 difficulty (DR + 扰动随难度缩放, 见 kuafu_mjx_env)
                cur_env_state = self._state.info["env_state"]
                survived = (cur_env_state.step_count >= self.max_episode_length) & (cur_env_state.fall_count == 0)
                survived_done = jax.device_get(survived & done_jax)
                self._curriculum.update([bool(x) for x in survived_done])
                self._difficulty = jax.numpy.float32(self._curriculum.d)

                reset_state = self._reset_vmapped(
                    jax.random.split(reset_rng, self.num_envs),
                    jax.numpy.full(self.num_envs, self._difficulty))
                done_mask = done_jax.astype(jax.numpy.bool_)
                self._state = jax.tree_util.tree_map(
                    lambda cur, new: jax.numpy.where(
                        done_mask.reshape((-1,) + (1,) * (cur.ndim - 1)), new, cur),
                    self._state, reset_state)

            # done 帧返回 reset 后的初始观测 (PPO 新 episode 首步用初始 obs)
            state_obs, extras = self.get_observations()
            reward = self._to_torch(reward_jax)
            done = self._to_torch(done_jax)

            self.episode_length_buf += 1

            # 收集 episode 级指标到 info["log"] (RSL-RL 自动写入 TensorBoard)
            # 仅在有环境 done 时才填充, 其余步留空 {} — RSL-RL 收集器遇空 dict 自动跳过,
            # 避免中途帧的 0.0 被计入均值导致指标被稀释趋零
            log = {}
            if done_any:
                done_mask = (done > 0)
                n_done = done_mask.sum().clamp(min=1)
                # episode_length 在清零前读取 (上面 +1 后, done 帧的值即该 episode 总长)
                log["episode_length"] = (self.episode_length_buf * done_mask).sum().item() / n_done.item()
                # 记录全局难度进展均值
                log["difficulty"] = self._to_torch(self._difficulty).mean().item()
                for key in ["orientation", "lin_vel_tracking"]:
                    if key in metrics_jax:
                        val = self._to_torch(metrics_jax[key])
                        log[key] = (val * done_mask).sum().item() / n_done.item()

            self.episode_length_buf = torch.where(
                done > 0, torch.zeros_like(self.episode_length_buf), self.episode_length_buf)

            # time_outs: 仅 timeout(非倒下) 时为 True, 用于 value bootstrap
            # 倒下 (fallen) 的 episode 不做 bootstrap (终止态 value=0)
            fallen = self._to_torch(fallen_jax)
            time_outs = (done > 0) & (fallen < 0.5)  # done 但未倒下 = 超时
            info = {"time_outs": time_outs.float(),
                    "observations": extras.get("observations", {}), "log": log}
            return state_obs, reward, done, info

        @property
        def unwrapped(self):
            return self.env

        @property
        def step_dt(self):
            return self.env.dt

    torch_env = DirectVecEnv(env, args.num_envs, args.seed, device=device)
    print(f"  obs={torch_env.num_obs}, privileged={torch_env.num_privileged_obs}, "
          f"action={torch_env.num_actions}")

    # ---- 维度一致性守卫 (防止规格再次漂移) ----
    assert torch_env.num_privileged_obs == CRITIC_PRIV_DIM, \
        f"critic 额外特权维度错: {torch_env.num_privileged_obs} != {CRITIC_PRIV_DIM}"
    assert RMA_STATIC_DIM + TRANSIENT_DIM == PRIVILEGED_DIM, \
        f"特权拆分错: {RMA_STATIC_DIM}+{TRANSIENT_DIM} != {PRIVILEGED_DIM}"
    assert torch_env.num_obs == ACTOR_OBS_DIM, \
        f"actor obs 维度错: {torch_env.num_obs} != {ACTOR_OBS_DIM}"
    assert ACTOR_OBS_DIM + CRITIC_PRIV_DIM == CRITIC_OBS_DIM, \
        f"critic 总维度错: {ACTOR_OBS_DIM}+{CRITIC_PRIV_DIM} != {CRITIC_OBS_DIM}"

    # ---- 训练配置 ----
    train_cfg = make_train_cfg()

    # ---- 日志目录: rl/checkpoints/<run_name>/teacher/ ----
    run_root = os.path.join(PROJ_ROOT, args.log_dir, args.run_name)
    log_dir = os.path.join(run_root, "teacher")

    # 防覆盖校验: 目录已存在且含 .pt, 且非续训 -> 报错
    existing = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if existing and not args.resume and not args.smoke_test:
        print(f"❌ 目录已含 checkpoint: {log_dir}")
        print(f"   续训请加 --resume <latest.pt>, 或换 --run_name")
        sys.exit(1)

    os.makedirs(log_dir, exist_ok=True)

    # ---- 写训练元数据 run.json ----
    run_meta = {
        "run_name": args.run_name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "seed": args.seed,
        "resume_from": args.resume,
        "algorithm": "PPO",
        "policy": "ActorCritic [512,512,512] elu",
        "device": device,
        "provenance": capture_provenance(),
    }
    meta_path = os.path.join(run_root, "run.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    # ---- RSL-RL Runner ----
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(torch_env, train_cfg, log_dir=log_dir, device=device)
    print(f"  日志: {log_dir}")

    # ---- 载入 Checkpoint 恢复训练 ----
    if args.resume:
        print(f"  载入 Checkpoint 恢复训练: {args.resume}")
        runner.load(args.resume)

    # ---- 训练 ----
    start_iter = runner.current_learning_iteration
    n_iter = 5 if args.smoke_test else args.iterations
    run_iter = max(0, n_iter - start_iter)

    if args.smoke_test:
        print("🔥 烟测: 5 iteration")
        run_iter = 5

    total_steps = args.num_envs * train_cfg["num_steps_per_env"] * run_iter
    print(f"开始训练: 需进行 {run_iter} 轮迭代 (已完成 {start_iter} 轮, 目标 {n_iter} 轮) × {args.num_envs} envs × {train_cfg['num_steps_per_env']} steps = {total_steps:,} steps")
    t0 = time.time()
    runner.learn(num_learning_iterations=run_iter, init_at_random_ep_len=True)
    elapsed = time.time() - t0

    print(f"\n✅ 训练完成: {elapsed:.1f}s, {total_steps:,} steps, {total_steps/elapsed:,.0f} steps/s")
    if not args.smoke_test:
        final = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
        print(f"   Checkpoint: {final}")
        print(f"   导出: rl/.venv/bin/python rl/export/export_policy.py --ckpt {final}")


if __name__ == "__main__":
    main()
