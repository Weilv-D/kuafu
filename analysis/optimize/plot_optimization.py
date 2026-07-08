# -*- coding: utf-8 -*-
"""KUAFU 蒙特卡洛优化结果可视化

产出两张图:
  (a) pareto_front.png  — 5 指标两两散点矩阵, 标注前沿 + 选定解
  (b) dirichlet_winrate.png — Dirichlet 权重扫掠的胜出分布
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import kuafu as kc
from kuafu import (OUTPUT, savefig, ax_clean, annotate_point, set_title_style,
                   final_style, C, ACCENT, HIGHLIGHT, NEUTRAL, SAFE)
from optimize.analyze_params import SELECTED, evaluate

LABELS = [
    r"$\tau_{peak}$ (Nm)",
    r"$\tau_{dwell}$ (Nm)",
    r"$1/\gamma_{min}$",
    r"$\kappa_{max}$",
    r"$-stroke$ (mm)",
]


def plot_pareto_front():
    """5 指标散点矩阵 + 前沿 + 选定解标注."""
    path = os.path.join(OUTPUT, "pareto_samples.npz")
    if not os.path.exists(path):
        print(f"跳过 pareto_front.png: 找不到 {path}")
        return
    data = np.load(path, allow_pickle=False)
    obj = data["obj"]; pf_mask = data["pareto_mask"]; sel_obj = data["selected_obj"]

    not_pf = ~pf_mask
    fig, axes = plt.subplots(5, 5, figsize=(13, 12))
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if i == j:
                # 对角: 指标分布直方图
                ax.hist(obj[:, i], bins=40, color=C["sky"], alpha=0.6, edgecolor="none")
                ax.axvline(sel_obj[i], color=HIGHLIGHT, lw=1.8, ls="--")
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(LABELS[i], fontsize=9)
                continue
            if j > i:
                ax.set_visible(False); continue
            # 下三角: 散点
            ax.scatter(obj[not_pf, j], obj[not_pf, i], s=2, c=NEUTRAL, alpha=0.18, edgecolors="none")
            ax.scatter(obj[pf_mask, j], obj[pf_mask, i], s=8, c=SAFE, alpha=0.7, edgecolors="none", label="Pareto 前沿")
            ax.scatter([sel_obj[j]], [sel_obj[i]], s=70, c=HIGHLIGHT, marker="*",
                       edgecolors="black", linewidths=0.6, zorder=5, label="选定解 (52,93,149)")
            ax_clean(ax)
            ax.tick_params(labelsize=7)
            if i == 4:
                ax.set_xlabel(LABELS[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(LABELS[i], fontsize=8)
    # 图例 (放在右上一个隐藏轴上)
    handles = [plt.Line2D([0],[0], marker="o", ls="", c=NEUTRAL, markersize=5, alpha=0.4, label="全部可行样本"),
               plt.Line2D([0],[0], marker="o", ls="", c=SAFE, markersize=6, label="Pareto 前沿"),
               plt.Line2D([0],[0], marker="*", ls="", c=HIGHLIGHT, markersize=11, label="选定解")]
    fig.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95)
    final_style(fig, "五杆机构参数蒙特卡洛优化 — Pareto 前沿 (10 万采样)")
    savefig(fig, "pareto_front.png")


def plot_dirichlet_winrate():
    """Dirichlet 胜出分布 + 选定解位置."""
    path = os.path.join(OUTPUT, "dirichlet_results.npz")
    if not os.path.exists(path):
        print(f"跳过 dirichlet_winrate.png: 找不到 {path}")
        return
    data = np.load(path, allow_pickle=False)
    winners = data["winners"]; cand_obj = data["cand_obj"]
    sel_idx = int(data["sel_idx"]); win_rate = float(data["win_rate"])

    uniq, counts = np.unique(winners, return_counts=True)
    order = np.argsort(-counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 左: 胜出次数条形图 (前 15)
    top_n = min(15, len(uniq))
    top_idx = uniq[order[:top_n]]
    top_cnt = counts[order[:top_n]]
    colors = [HIGHLIGHT if i == sel_idx else ACCENT for i in top_idx]
    bars = ax1.barh(range(top_n), top_cnt, color=colors, edgecolor="white", linewidth=0.5)
    labels = []
    for i in top_idx:
        if i == sel_idx:
            labels.append("选定解\n(52,93,149)")
        else:
            # 候选 idx 0..n_pf-1 是前沿解
            o = cand_obj[i]
            labels.append(f"τ={o[0]:.2f}\nγ={1/o[2]:.0f}°\nL={-o[4]:.0f}")
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel("Dirichlet 权重下胜出次数 (共 200 组)")
    set_title_style(ax1, f"选定解胜出率 {100*win_rate:.0f}%")
    ax_clean(ax1)

    # 右: 前沿候选在 τ_peak - stroke 平面, 点大小=胜出次数
    pf_obj = data["pf_obj"]; sel_obj = data["sel_obj"]
    # 胜出次数映射到候选集
    win_count = np.zeros(len(cand_obj), dtype=int)
    for u, c in zip(uniq, counts):
        win_count[u] = c
    sizes = 15 + win_count * 3
    ax2.scatter(pf_obj[:, 0], -pf_obj[:, 4], s=sizes[:len(pf_obj)],
                c=ACCENT, alpha=0.4, edgecolors="white", linewidths=0.4, label="前沿候选")
    ax2.scatter([sel_obj[0]], [-sel_obj[4]], s=180, c=HIGHLIGHT, marker="*",
                edgecolors="black", linewidths=0.6, zorder=5, label="选定解")
    ax2.set_xlabel(LABELS[0]); ax2.set_ylabel("stroke (mm)")
    set_title_style(ax2, "前沿候选 vs 选定解 (点大小 = 胜出次数)")
    ax_clean(ax2); ax2.legend(fontsize=8, loc="lower left")

    final_style(fig, "Dirichlet 权重稳健性扫掠 (200 组, α=1)")
    savefig(fig, "dirichlet_winrate.png")


if __name__ == "__main__":
    plot_pareto_front()
    plot_dirichlet_winrate()
