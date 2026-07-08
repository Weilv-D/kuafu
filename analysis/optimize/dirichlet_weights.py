# -*- coding: utf-8 -*-
"""KUAFU Dirichlet 权重稳健性扫掠

对 Pareto 前沿上的候选解, 用 N_W 组 Dirichlet 分布权重向量加权打分,
统计选定解 (52,93,149) 在前沿候选中的胜出率.

权重先验 (反映工程优先级): 扭矩与行程是主指标, 传动角/条件数为次要约束.
用非对称 α 向量体现: 扭矩/行程维度 α 高(权重集中), 传动角/条件数 α 低.
Dirichlet(α) 采样仍覆盖各种偏好组合, 但更频繁落到"重视扭矩/行程"的区域.

输出:
  - 控制台: 选定解胜出率, 前沿排名分布
  - output/dirichlet_results.npz : 权重向量 + 胜出解 idx
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import kuafu as kc
from kuafu import OUTPUT
from optimize.analyze_params import evaluate, SELECTED

N_WEIGHTS = 200         # Dirichlet 权重向量数
# 工程先验 α (5维对应 f1..f5): τ_peak, τ_dwell 高; 1/γ, κ 低; -stroke 高
#   α 越大 → 该维度权重期望越大(越受重视)
ALPHA_VEC = np.array([3.0, 2.0, 0.8, 0.8, 3.0])
SEED = 20260706
METRIC_NAMES = ["τ_peak", "τ_dwell", "1/γ_min", "κ_max", "-stroke"]


def normalize_objectives(obj, ref=None):
    """归一化目标矩阵到 [0,1], 用 ref 的 min/max (供选定点用同一映射)."""
    if ref is None:
        ref = obj
    omin = ref.min(0); omax = ref.max(0)
    span = np.where(omax - omin > 1e-12, omax - omin, 1.0)
    return (obj - omin) / span, (omin, omax, span)


def run():
    # 载入蒙特卡洛结果
    mc_path = os.path.join(OUTPUT, "pareto_samples.npz")
    if not os.path.exists(mc_path):
        print(f"找不到 {mc_path}, 请先运行 monte_carlo.py")
        return None
    data = np.load(mc_path, allow_pickle=False)
    obj = data["obj"]                 # (N_feasible, 5) 全部可行样本目标
    pf_mask = data["pareto_mask"]     # (N_feasible,) 前沿掩码
    sel_obj = data["selected_obj"]    # (5,) 选定解目标
    d_arr = data["d"]; a_arr = data["a"]; b_arr = data["b"]

    n_feasible = len(obj)
    n_pf = int(pf_mask.sum())
    print(f"载入: {n_feasible} 可行样本, {n_pf} Pareto 前沿解")

    # 前沿候选 + 选定解, 一起归一化 (用前沿的 min/max 作参考)
    pf_obj = obj[pf_mask]
    pf_n, (omin, omax, span) = normalize_objectives(pf_obj)
    sel_n = (sel_obj - omin) / span

    # 选定解到前沿的归一化距离 → 判断是否可视作前沿候选
    dist_pf = np.linalg.norm(pf_n - sel_n, axis=1)
    nearest_pf = int(np.argmin(dist_pf))
    print(f"选定解最近前沿解: idx={nearest_pf}, 距离={dist_pf[nearest_pf]:.4f}")
    pf_d = d_arr[pf_mask][nearest_pf]; pf_a = a_arr[pf_mask][nearest_pf]; pf_b = b_arr[pf_mask][nearest_pf]
    print(f"  该前沿解参数: d={pf_d:.1f}, a={pf_a:.1f}, b={pf_b:.1f}")

    # 候选集 = 前沿解 + 选定解 (作为虚拟候选参与排名)
    cand_obj = np.vstack([pf_obj, sel_obj[np.newaxis, :]])
    cand_n, _ = normalize_objectives(cand_obj, ref=pf_obj)
    n_cand = len(cand_obj)
    sel_idx = n_cand - 1              # 选定解在候选集中的索引

    # Dirichlet 权重扫掠 (工程先验 α 向量)
    rng = np.random.default_rng(SEED)
    weights = rng.dirichlet(ALPHA_VEC, size=N_WEIGHTS)   # (N_W, 5)

    # 每组权重下, 各候选的加权得分 (越小越好), 选最小者为"胜出"
    # scores[i,j] = sum_k weights[i,k] * cand_n[j,k]
    scores = weights @ cand_n.T        # (N_W, n_cand)
    winners = np.argmin(scores, axis=1)

    sel_wins = int(np.sum(winners == sel_idx))
    win_rate = sel_wins / N_WEIGHTS

    print(f"\n=== Dirichlet 权重稳健性 ({N_WEIGHTS} 组, α={ALPHA_VEC}) ===")
    print(f"  指标: {METRIC_NAMES}")
    print(f"  先验: 扭矩/行程主导, 传动角/条件数次要")
    print(f"候选集: {n_pf} 前沿解 + 1 选定解 = {n_cand}")
    print(f"选定解胜出: {sel_wins} / {N_WEIGHTS} = {100*win_rate:.1f}%")

    # 各前沿解的胜出分布 (前5)
    uniq, counts = np.unique(winners, return_counts=True)
    order = np.argsort(-counts)
    print(f"\n胜出次数前 5 候选:")
    for i in order[:5]:
        if i == sel_idx:
            tag = "← 选定解"
            params = f"d={SELECTED['d']}, a={SELECTED['a']}, b={SELECTED['b']}"
        else:
            tag = ""
            params = f"d={d_arr[pf_mask][i]:.1f}, a={a_arr[pf_mask][i]:.1f}, b={b_arr[pf_mask][i]:.1f}"
        print(f"  idx={i:4d}: {counts[i]:3d} 次 ({100*counts[i]/N_WEIGHTS:.1f}%) {params} {tag}")

    # 存档
    out = os.path.join(OUTPUT, "dirichlet_results.npz")
    np.savez(out, weights=weights, winners=winners,
             cand_obj=cand_obj, cand_norm=cand_n,
             sel_idx=sel_idx, win_rate=win_rate,
             pf_obj=pf_obj, sel_obj=sel_obj)
    print(f"\n存档: {out}")

    return dict(win_rate=win_rate, n_pf=n_pf, nearest_pf=nearest_pf,
                dist_pf=float(dist_pf[nearest_pf]))


if __name__ == "__main__":
    run()
