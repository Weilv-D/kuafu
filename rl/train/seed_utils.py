# -*- coding: utf-8 -*-
"""
KUAFU 训练可复现性与版本溯源工具.

设计依据:
  - PyTorch 官方复现文档: torch.manual_seed / torch.cuda.manual_seed_all 播种所有设备 RNG;
    numpy.random.seed 与 python random.seed 同步全局 RNG (https://docs.pytorch.org/docs/stable/notes/randomness.html).
  - JAX 采用显式 PRNG key (无全局状态), 根种子由同一整数派生, 与 torch/numpy 同源.
  - 跨框架/跨版本完全确定性无法保证 (PyTorch 与 JAX 官方均明示), 故以
    "全 RNG 播种 + 版本/环境溯源 (provenance)" 作为可复现的务实基线, 而非盲目 strict 确定性.
"""
import random
import subprocess
import sys

import numpy as np
import torch


def seed_all(seed: int):
    """播种所有随机源 (torch / numpy / python / cuda / JAX), 全部与同一整数同源.

    Args:
        seed: 统一整数种子.

    Returns:
        jax.random.PRNGKey(seed); 若 jax 不可用或 seed 为 None 则返回 None.
        调用方应使用返回 key 驱动 JAX 侧的随机流, 避免"漏播种"导致不可复现.
    """
    if seed is None:
        return None
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    try:
        import jax
        return jax.random.PRNGKey(seed)
    except Exception:
        return None


def capture_provenance() -> dict:
    """采集运行环境溯源信息, 写入 checkpoint 元数据以便复现与归因.

    包含框架版本、CUDA 版本、git 快照 (短 hash + 是否有未提交改动)。
    """
    prov = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
    }
    try:
        import jax

        prov["jax"] = jax.__version__
    except Exception:
        prov["jax"] = None

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ) != 0
        prov["git_commit"] = commit + ("-dirty" if dirty else "")
    except Exception:
        prov["git_commit"] = None
    return prov
