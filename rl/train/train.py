# -*- coding: utf-8 -*-
"""
KUAFU Teacher PPO 训练入口 — design.md §2.6 阶段 1

MJX 环境 (JAX/GPU) → RSLRLBraxWrapper (DLPack 零拷贝) → RSL-RL 2.x PPO (PyTorch/GPU)
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

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

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
            "actor_hidden_dims": [256, 256, 256],
            "critic_hidden_dims": [256, 256, 256],
            "activation": "elu",
        },
        "num_steps_per_env": 24,
        "save_interval": 50,
        "empirical_normalization": True,
    }


def main():
    parser = argparse.ArgumentParser(description="KUAFU Teacher PPO Training")
    parser.add_argument("--num_envs", type=int, default=1024, help="并行环境数")
    parser.add_argument("--iterations", type=int, default=1000, help="训练迭代数")
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
            self.num_obs = OBS_DIM
            self.num_privileged_obs = PRIVILEGED_DIM if env._teacher else None
            self.device = device
            self.cfg = {"env_name": "kuafu", "num_envs": num_envs}
            self.max_episode_length = env._episode_length
            self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)

            self._reset_vmapped = jax.jit(jax.vmap(env.reset))
            self._step_vmapped = jax.jit(jax.vmap(env.step))

            self._rng = jax.random.PRNGKey(seed)
            self._state = self._reset_vmapped(jax.random.split(self._rng, num_envs))

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
                extras["observations"]["critic"] = self._to_torch(obs["privileged_state"])
            return state_obs, extras

        def step(self, action):
            jax_action = self._to_jax(action)
            self._state = self._step_vmapped(self._state, jax_action)

            # auto-reset done 环境
            done_np = jax.device_get(self._state.done)
            if done_np.any():
                self._rng, reset_rng = jax.random.split(self._rng)
                reset_keys = jax.random.split(reset_rng, self.num_envs)
                reset_state = self._reset_vmapped(reset_keys)
                done_mask = done_np
                self._state = jax.tree_util.tree_map(
                    lambda cur, new: jax.numpy.where(
                        done_mask.reshape((-1,) + (1,) * (cur.ndim - 1)), new, cur),
                    self._state, reset_state)

            obs = self._state.obs
            state_obs = self._to_torch(obs["state"]) if isinstance(obs, dict) else self._to_torch(obs)
            reward = self._to_torch(self._state.reward)
            done = self._to_torch(self._state.done)

            self.episode_length_buf += 1
            self.episode_length_buf = torch.where(
                done > 0, torch.zeros_like(self.episode_length_buf), self.episode_length_buf)

            info = {"time_outs": done.float(), "observations": {}, "log": {}}
            if isinstance(obs, dict) and "privileged_state" in obs:
                info["observations"]["critic"] = self._to_torch(obs["privileged_state"])
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
