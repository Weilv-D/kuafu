# -*- coding: utf-8 -*-
"""
KUAFU JAX↔PyTorch DLPack 零拷贝桥接.

设计依据:
  - JAX / PyTorch 的 from_dlpack(copy=False) 在需要跨设备拷贝或内存不共享时会显式抛错;
    用它把"零拷贝"从隐含假设变为可失败断言, 在 CUDA13+JAX0.10+PyTorch2.12 目标栈上
    能第一时间暴露设备不一致、非连续、版本不兼容等问题.
  - DLPack 协议本身通过 stream 句柄完成跨框架计算顺序同步, 因此不必强制 block_until_ready
    (仅在极少数自定义 buffer 场景才需显式同步); 本模块保留可选同步钩子以备调试.
  - 非连续 torch 张量在导入 JAX 前先 .contiguous(), 避免静默拷贝; 连续张量该操作为 no-op.

所有 GPU 张量经此模块交换, 保证 device 一致性与零拷贝契约集中可控.
"""
import warnings

import jax
import torch


def resolve_device(prefer: str = "cuda") -> str:
    """解析可用计算设备, 无 GPU 时回退 CPU 并告警.

    用于统一 JAX 默认设备与 torch 张量设备, 避免 train.py 硬编码 cuda:0 与
    JAX 默认设备在多卡环境下错位.
    """
    if prefer.startswith("cuda"):
        if torch.cuda.is_available() and len(jax.devices("gpu")) > 0:
            return "cuda"
        warnings.warn(
            "CUDA 不可用 (torch.cuda.is_available 或 jax gpu 设备为空), "
            "回退至 CPU。MJX 可在 CPU 运行但速度显著下降。",
            stacklevel=2,
        )
        return "cpu"
    return prefer


def to_torch(x, device=None):
    """JAX Array → torch.Tensor (DLPack 零拷贝契约).

    copy=False: 若发生跨设备拷贝则抛错, 而非静默复制。
    """
    try:
        return torch.utils.dlpack.from_dlpack(x, copy=False, device=device)
    except Exception as e:  # 包装为可操作的错误信息
        raise RuntimeError(
            f"JAX→torch DLPack 零拷贝失败: {e}. "
            f"请确认 JAX 与 torch 位于同一物理设备且 dtype 受支持。"
        ) from e


def to_jax(t, device=None):
    """torch.Tensor → JAX Array (DLPack 零拷贝契约).

    非连续张量先 .contiguous() (no-op 若已连续), 避免 from_dlpack 静默拷贝。
    copy=False: 若发生跨设备拷贝则抛错。
    """
    if not t.is_contiguous():
        t = t.contiguous()
    try:
        return jax.dlpack.from_dlpack(t, copy=False, device=device)
    except Exception as e:
        raise RuntimeError(
            f"torch→JAX DLPack 零拷贝失败: {e}. "
            f"请确认 torch 与 JAX 位于同一物理设备且 dtype 受支持。"
        ) from e


def verify_dlpack_zero_copy(device: str = "cuda") -> None:
    """启动期一次性验证 DLPack 零拷贝契约 (设备一致 / dtype 一致 / 无拷贝 / 数值相等).

    在 CUDA13+JAX0.10+PyTorch2.12 目标栈上作为守卫运行; 失败即抛出明确错误。
    CPU 环境下退化为 CPU↔CPU 往返验证 (仍校验 API 与数值正确性)。
    """
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
