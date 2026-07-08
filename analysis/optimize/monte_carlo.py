# -*- coding: utf-8 -*-
"""KUAFU 蒙特卡洛参数采样 + Pareto 前沿提取

在物理合理的 (d, a, b) 参数空间内均匀采样 30 万组配置,
逐组计算 5 个指标, 过滤不可达/不合格解, 用非支配排序提取 Pareto 前沿.
最后验证选定解 (52, 93, 149) 在前沿中的位置.

输出:
  - output/pareto_samples.npz : 全部可行样本 (供绘图)
  - 控制台报告: 采样数/可行数/前沿规模/选定解位置
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import kuafu as kc
from optimize.analyze_params import (
    evaluate, SELECTED, D_RANGE, A_RANGE, B_RANGE,
)
from kuafu import OUTPUT

N_SAMPLES = 100_000          # 蒙特卡洛采样数 (10万足够稳定, 平衡精度与耗时)
SEED = 20260706              # 可复现种子


def sample_params(rng_state):
    """均匀采样一组 (d, a, b)."""
    d = rng_state.uniform(*D_RANGE)
    a = rng_state.uniform(*A_RANGE)
    b = rng_state.uniform(*B_RANGE)
    return d, a, b


def pareto_front(objectives):
    """非支配排序提取 Pareto 前沿 (全部最小化).

    objectives: (N, M) 数组, M 个目标均最小化.
    Returns: 布尔掩码 (N,), True 表示在前沿上.
    """
    N = len(objectives)
    is_pareto = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_pareto[i]:
            continue
        # 被 j 支配: j 在所有目标 <= i 且至少一个 < i
        dominated = np.all(objectives <= objectives[i], axis=1) & \
                    np.any(objectives < objectives[i], axis=1)
        dominated[i] = False            # 不算自己
        is_pareto[dominated] = False
    return is_pareto


def run():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    samples = []     # 可行样本 (含 d,a,b + 5 指标)
    n_infeasible = 0

    print(f"蒙特卡洛采样 {N_SAMPLES:,} 组配置 ...")
    for i in range(N_SAMPLES):
        d, a, b = sample_params(rng)
        r = evaluate(d, a, b)
        if r is None:
            n_infeasible += 1
            continue
        samples.append(r)
        if (i + 1) % 10_000 == 0:
            print(f"  进度 {i+1:,}/{N_SAMPLES:,}, "
                  f"可行 {len(samples):,}, 耗时 {time.time()-t0:.1f}s", flush=True)

    n_feasible = len(samples)
    print(f"\n采样完成: {N_SAMPLES:,} → 可行 {n_feasible:,} "
          f"({100*n_feasible/N_SAMPLES:.1f}%), 不可达 {n_infeasible:,}")
    print(f"耗时 {time.time()-t0:.1f}s")

    if n_feasible == 0:
        print("无可行样本, 退出.")
        return None

    # 组装目标矩阵
    keys = ["d", "a", "b", "D0_min", "D0_max",
            "tau_peak", "tau_dwell", "gamma_min", "cond_max", "stroke", "ab_ratio"]
    data = {k: np.array([s[k] for s in samples]) for k in keys}
    obj = np.array([[s["f1"], s["f2"], s["f3"], s["f4"], s["f5"]] for s in samples])

    print("\n非支配排序提取 Pareto 前沿 ...")
    pf_mask = pareto_front(obj)
    n_pf = int(pf_mask.sum())
    print(f"  Pareto 前沿规模: {n_pf} / {n_feasible} "
          f"({100*n_pf/n_feasible:.2f}%)")

    # 选定解位置核查
    sel = evaluate(**SELECTED)
    print(f"\n=== 选定解 ({SELECTED['d']}, {SELECTED['a']}, {SELECTED['b']}) ===")
    print(f"  tau_peak={sel['tau_peak']:.3f}, tau_dwell={sel['tau_dwell']:.3f}, "
          f"gamma_min={sel['gamma_min']:.1f}, cond_max={sel['cond_max']:.3f}, "
          f"stroke={sel['stroke']:.1f}")

    # 找选定解在样本中的最近邻 (判断是否在前沿附近)
    sel_obj = np.array([sel["f1"], sel["f2"], sel["f3"], sel["f4"], sel["f5"]])
    # 归一化目标用于距离
    obj_min, obj_max = obj.min(0), obj.max(0)
    span = np.where(obj_max - obj_min > 1e-9, obj_max - obj_min, 1.0)
    obj_n = (obj - obj_min) / span
    sel_n = (sel_obj - obj_min) / span
    dist = np.linalg.norm(obj_n - sel_n, axis=1)
    nearest = int(np.argmin(dist))
    on_front = bool(pf_mask[nearest])
    print(f"  最近样本 idx={nearest}, 归一化距离={dist[nearest]:.4f}, "
          f"在前沿={on_front}")
    print(f"  最近样本参数: d={data['d'][nearest]:.1f}, "
          f"a={data['a'][nearest]:.1f}, b={data['b'][nearest]:.1f}")

    # 存档
    out = os.path.join(OUTPUT, "pareto_samples.npz")
    save = {k: v for k, v in data.items()}
    save["obj"] = obj
    save["pareto_mask"] = pf_mask
    save["selected_obj"] = sel_obj
    np.savez(out, **save)
    print(f"\n样本存档: {out}")

    return dict(samples=samples, data=data, obj=obj, pf_mask=pf_mask,
                sel=sel, nearest=nearest, on_front=on_front)


if __name__ == "__main__":
    run()
