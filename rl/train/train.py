# -*- coding: utf-8 -*-
"""
KUAFU Teacher PPO 训练入口 — design.md §2.6 阶段 1

MJX 环境 (JAX/GPU) → DirectVecEnv (DLPack 零拷贝) → RSL-RL 2.x PPO (PyTorch/GPU)
Teacher: critic 含特权信息 (friction/mass/COM/inertia), actor 仅本体感受。

运行:
  rl/.venv/bin/python rl/train/train.py --num_envs 1024 --iterations 1000

产出:
  rl/checkpoints/teacher_{timestamp}/model_{iter}.pt
"""
import os
import sys
import argparse
import time

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax


def make_train_cfg() -> dict:
    """RSL-RL 2.x OnPolicyRunner 配置格式."""
    return {
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.005,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 512, 512],
            "critic_hidden_dims": [512, 512, 512],
            "activation": "elu",
        },
        "num_steps_per_env": 24,
        "save_interval": 50,
        "empirical_normalization": True,
    }


def main():
    parser = argparse.ArgumentParser(description="KUAFU Teacher PPO Training")
    parser.add_argument("--num_envs", type=int, default=1024, help="并行环境数")
    parser.add_argument("--iterations", type=int, default=3000, help="训练迭代数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log_dir", type=str, default="rl/checkpoints", help="checkpoint 目录")
    parser.add_argument("--smoke_test", action="store_true", help="烟测模式 (5 iteration)")
    args = parser.parse_args()

    print("=" * 60)
    print("KUAFU Teacher PPO Training (design.md §2.6 阶段 1)")
    print("=" * 60)
    print(f"  并行环境: {args.num_envs}")
    print(f"  JAX 设备: {jax.devices()}")

    # ---- 创建环境 ----
    # 当前: 平地训练 (difficulty=0)。课程地形 (terrain.py CurriculumController)
    #       待平地 reward 收敛后在此处接入: 按 episode 成功率更新 difficulty。
    from rl.env.kuafu_mjx_env import KuafuMjxEnv, OBS_DIM, PRIVILEGED_DIM
    import torch

    env = KuafuMjxEnv(teacher=True, num_envs=args.num_envs)

    # ---- 直接适配 rsl_rl 2.x (绕过 playground brax wrapper 的 info 结构限制) ----
    class DirectVecEnv:
        """JAX vmap 环境到 rsl_rl 2.x VecEnv 的直接适配器.

        绕过 playground 的 BraxAutoResetWrapper (其 auto-reset 会修改 info 结构导致
        scan pytree 不匹配), 直接用 JAX vmap + jax.lax.cond 做 auto-reset,
        通过 DLPack 与 PyTorch 零拷贝交换 GPU 张量。
        """
        def __init__(self, env, num_envs, seed, device="cuda:0"):
            self.env = env
            self.num_envs = num_envs
            self.num_actions = env.action_size
            self.num_obs = OBS_DIM                                    # actor 只吃 proprio (无特权泄漏)
            self.num_privileged_obs = (OBS_DIM + PRIVILEGED_DIM) if env._teacher else None
            self.device = device
            self.cfg = {"env_name": "kuafu", "num_envs": num_envs}
            self.max_episode_length = env._episode_length
            self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)

            # difficulty 课程数组管理 (每一个 env 独立阶梯)
            self._difficulty = jax.numpy.zeros(num_envs, dtype=jax.numpy.float32)

            self._reset_vmapped = jax.jit(jax.vmap(env.reset, in_axes=(0, 0)))
            self._step_vmapped = jax.jit(jax.vmap(env.step))

            self._rng = jax.random.PRNGKey(seed)
            self._state = self._reset_vmapped(jax.random.split(self._rng, num_envs), self._difficulty)

        def _to_torch(self, x):
            """JAX DeviceArray → torch.Tensor (DLPack 零拷贝)."""
            return torch.utils.dlpack.from_dlpack(x)

        def _to_jax(self, t):
            """torch.Tensor → JAX DeviceArray (DLPack 零拷贝)."""
            return jax.dlpack.from_dlpack(t.contiguous())

        def get_observations(self):
            obs = self._state.obs
            state_obs = self._to_torch(obs["state"]) if isinstance(obs, dict) else self._to_torch(obs)
            extras = {"observations": {}}
            if isinstance(obs, dict) and "privileged_state" in obs:
                # critic 吃 proprio (140) + privileged (12) = 152
                priv_obs = self._to_torch(obs["privileged_state"])
                extras["observations"]["critic"] = torch.cat([state_obs, priv_obs], dim=-1)
            return state_obs, extras

        def reset(self):
            """VecEnv 接口要求: 重置所有环境."""
            self._rng, reset_rng = jax.random.split(self._rng)
            # 使用当前的 self._difficulty 以便保存课程阶梯状态
            self._state = self._reset_vmapped(jax.random.split(reset_rng, self.num_envs), self._difficulty)
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
                
                # JAX 离散阶梯课程更新逻辑:
                # 判定成功: 存活至最大步数且没有倒下 (timeout 且 fall_count 为 0)
                cur_env_state = self._state.info["env_state"]
                survived = (cur_env_state.step_count >= self.max_episode_length) & (cur_env_state.fall_count == 0)
                old_diff = cur_env_state.difficulty
                new_diff = jax.numpy.where(
                    survived,
                    jax.numpy.minimum(old_diff + 0.05, 1.0),
                    jax.numpy.maximum(old_diff - 0.05, 0.0)
                )
                done_mask_jax = done_jax.astype(jax.numpy.bool_)
                self._difficulty = jax.numpy.where(done_mask_jax, new_diff, self._difficulty)
                
                reset_state = self._reset_vmapped(jax.random.split(reset_rng, self.num_envs), self._difficulty)
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

    torch_env = DirectVecEnv(env, args.num_envs, args.seed)
    print(f"  obs={torch_env.num_obs}, privileged={torch_env.num_privileged_obs}, "
          f"action={torch_env.num_actions}")

    # ---- 训练配置 ----
    train_cfg = make_train_cfg()

    # ---- 日志目录 ----
    log_dir = os.path.join(PROJ_ROOT, args.log_dir, f"teacher_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)

    # ---- RSL-RL Runner ----
    from rsl_rl.runners import OnPolicyRunner
    device = "cuda"
    runner = OnPolicyRunner(torch_env, train_cfg, log_dir=log_dir, device=device)
    print(f"  日志: {log_dir}")

    # ---- 训练 ----
    n_iter = 5 if args.smoke_test else args.iterations
    if args.smoke_test:
        print("🔥 烟测: 5 iteration")

    total_steps = args.num_envs * 24 * n_iter
    print(f"开始训练: {n_iter} iters × {args.num_envs} envs × 24 steps = {total_steps:,} steps")
    t0 = time.time()
    runner.learn(num_learning_iterations=n_iter, init_at_random_ep_len=True)
    elapsed = time.time() - t0

    print(f"\n✅ 训练完成: {elapsed:.1f}s, {total_steps:,} steps, {total_steps/elapsed:,.0f} steps/s")
    if not args.smoke_test:
        final = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
        print(f"   Checkpoint: {final}")
        print(f"   导出: rl/.venv/bin/python rl/export/export_policy.py --ckpt {final}")


if __name__ == "__main__":
    main()
