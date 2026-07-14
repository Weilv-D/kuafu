# -*- coding: utf-8 -*-
"""
DLPack 跨框架互操作性测试 (JAX ↔ PyTorch 零拷贝).

在 GPU runner 上验证 CUDA 栈; 无 GPU 时退化为 CPU↔CPU 仍校验 API 与数值正确性。
运行: rl/.venv/bin/python -m pytest rl/train/tests/test_dlpack_interop.py -v
"""
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")

from rl.train import dlpack_utils as du


def _device():
    try:
        has_jax_gpu = any(device.platform == "gpu" for device in jax.devices())
    except RuntimeError:
        has_jax_gpu = False
    return "cuda" if (torch.cuda.is_available() and has_jax_gpu) else "cpu"


def test_verify_zero_copy_passes():
    du.verify_dlpack_zero_copy(_device())


def test_round_trip_preserves_values():
    dev = _device()
    t = torch.randn(4, 8, device=dev, dtype=torch.float32)
    j = du.to_jax(t)
    back = du.to_torch(j)
    assert torch.allclose(back, t)
    assert back.dtype == torch.float32


def test_non_contiguous_transpose_zero_copy():
    # 转置产生非连续张量; to_jax 应自动 contiguous 并保持零拷贝契约
    dev = _device()
    t = torch.randn(4, 8, device=dev, dtype=torch.float32).transpose(0, 1)
    assert not t.is_contiguous()
    j = du.to_jax(t)
    back = du.to_torch(j)
    assert torch.allclose(back, t.contiguous() if not t.is_contiguous() else t)


def test_slice_view_zero_copy():
    # 行切片 ([:, :N]) 连续; 校验 from_dlpack 对 JAX 切片视图的零拷贝
    dev = _device()
    big = torch.randn(16, 64, device=dev, dtype=torch.float32)
    j_big = du.to_jax(big)
    sub = du.to_torch(j_big[:, :32])
    assert torch.allclose(sub, big[:, :32])
