# -*- coding: utf-8 -*-
"""
KUAFU JAX↔PyTorch DLPack 零拷贝桥接.

两端使用统一设备标识 (cuda:N) 确保 DLPack 零拷贝契约成立.
非连续 torch 张量在导入 JAX 前先 .contiguous().
"""
import warnings

import jax
import torch


def resolve_device(prefer: str = "cuda") -> str:
    """解析可用计算设备, 返回统一标识 (cuda:N / cpu).

    返回带显式 device index 的 cuda:N, 保证 torch 与 JAX DLPack 设备标识一致,
    避免 torch 默认 device="cuda"(无 index)被 DLPack 解释为 255 导致零拷贝失败.
    """
    if prefer.startswith("cuda"):
        try:
            jax_gpu_devices = jax.devices("gpu")
        except RuntimeError:
            jax_gpu_devices = []
        if torch.cuda.is_available() and jax_gpu_devices:
            dev_id = jax_gpu_devices[0].id
            return f"cuda:{dev_id}"
        warnings.warn(
            "CUDA 不可用 (torch.cuda.is_available 或 jax gpu 设备为空), "
            "回退至 CPU。MJX 可在 CPU 运行但速度显著下降。",
            stacklevel=2,
        )
        return "cpu"
    return prefer


def to_torch(x, device=None):
    """JAX Array → torch.Tensor (DLPack 零拷贝)."""
    try:
        return torch.utils.dlpack.from_dlpack(x, copy=False, device=device)
    except Exception as e:
        raise RuntimeError(
            f"JAX→torch DLPack 失败: {e}. "
        ) from e


def to_jax(t, device=None):
    """torch.Tensor → JAX Array (DLPack 零拷贝).

    非连续张量先 .contiguous().
    """
    if not t.is_contiguous():
        t = t.contiguous()
    try:
        return jax.dlpack.from_dlpack(t, copy=False, device=device)
    except Exception as e:
        raise RuntimeError(
            f"torch→JAX DLPack 失败: {e}. "
        ) from e


def verify_dlpack_zero_copy(device: str = "cuda") -> None:
    """启动期一次性验证 DLPack 契约 (设备一致 / dtype 一致 / 数值相等)."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    x_t = torch.arange(12, dtype=torch.float32, device=dev).reshape(3, 4)
    x_j = to_jax(x_t)
    # 设备一致性 (JAX 设备以 platform + id 表示, 如 cpu:0 / gpu:0)
    if dev.type == "cuda":
        assert x_j.device.platform == "gpu" and x_j.device.id == (dev.index or 0), \
            f"DLPack 设备不一致: torch={dev}, jax={x_j.device}"
    else:
        assert x_j.device.platform == "cpu", \
            f"DLPack 设备不一致: torch={dev}, jax={x_j.device}"
    # dtype 一致性
    assert x_j.dtype == jax.numpy.float32, f"DLPack dtype 不一致: {x_j.dtype}"
    # 零拷贝契约: from_dlpack(copy=False) 在需要拷贝时会抛错, 通过即证明无拷贝
    y_t = to_torch(x_j)
    # 数值相等 (往返一致性)
    assert torch.allclose(y_t, x_t), "DLPack 往返数值不一致"
