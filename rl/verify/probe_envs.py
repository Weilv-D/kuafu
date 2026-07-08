# -*- coding: utf-8 -*-
"""
KUAFU 显存测算 — RTX 4070 8GB 最大并行环境数搜索

二分搜索 KUAFU MJX 模型在不 OOM 的前提下能承载的最大 envs 数。
设 JAX 显存预分配 80%, 保证训练时有余量给 PyTorch policy 网络。

运行:
  rl/.venv/bin/python rl/verify/probe_envs.py
"""
import os
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"

import jax


def probe_envs(num_envs: int) -> bool:
    """测试给定 envs 数能否 reset + step 1 次不 OOM."""
    try:
        from rl.env.kuafu_mjx_env import KuafuMjxEnv
        env = KuafuMjxEnv(teacher=True, num_envs=num_envs)

        rng = jax.random.PRNGKey(42)
        keys = jax.random.split(rng, num_envs)

        reset_fn = jax.jit(jax.vmap(env.reset))
        step_fn = jax.jit(jax.vmap(env.step))

        state = reset_fn(keys)
        action = jax.random.uniform(rng, (num_envs, env.action_size), minval=-0.1, maxval=0.1)
        state = step_fn(state, action)
        # 强制同步检查显存
        jax.block_until_ready(state.reward)
        return True
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "OUT_OF_MEMORY" in str(e) or "oom" in str(e).lower():
            return False
        raise


def main():
    print("=" * 60)
    print("KUAFU 显存测算 (RTX 4070 8GB)")
    print("=" * 60)
    print(f"  JAX 设备: {jax.devices()}")
    print(f"  MEM_FRACTION: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'default')}")
    print()

    # 逐级测试 (避免二分搜索的重复 JIT 编译开销)
    test_sizes = [128, 256, 512, 1024, 2048, 4096]
    best = 128

    for n in test_sizes:
        print(f"  测试 {n:5d} envs...", end=" ", flush=True)
        ok = probe_envs(n)
        if ok:
            print("✓")
            best = n
        else:
            print("✗ OOM")
            break

    # 留 20% 安全余量 (给 PyTorch policy + rollout buffer)
    recommended = int(best * 0.8)

    print(f"\n{'='*60}")
    print(f"最大可承载: {best} envs")
    print(f"推荐 (×0.8 安全余量): {recommended} envs")
    print(f"{'='*60}")
    print(f"\n使用方法:")
    print(f"  rl/.venv/bin/python rl/train/train.py --num_envs {recommended}")


if __name__ == "__main__":
    main()
