# -*- coding: utf-8 -*-
"""KUAFU 参数化机构指标计算 — 蒙特卡洛优化的基础模块

给定一组机构参数 (d, a, b), 在 D0 行程内计算 5 个归一化指标,
供 monte_carlo.py 做 Pareto 前沿提取.

指标定义 (均为"越小越好"的最小化目标, 便于 Pareto 排序):
  f1 = tau_peak       : 瞬态峰值扭矩 (D0 全程最大, Nm) — 主指标
  f2 = tau_dwell      : 驻留态扭矩 (D0_min 时, Nm) — 自锁质量
  f3 = 1/gamma_min    : 传动角下限的倒数 (1/度) — 力传递质量
  f4 = cond_max       : 雅可比条件数峰值 — 运动学良态
  f5 = -stroke        : 负的工作空间行程 (mm) — 爬阶能力

约束 (硬过滤, 不满足返回 None):
  - 可达性: D0_min..D0_max 全程可解
  - 对称非交叉构型: P1.x < Q.x < P2.x
  - gamma >= 30° 全程
  - 膝角 >= 12° (奇异余量, 打样容差)
"""
import numpy as np
import kuafu as kc

# ============================================================
# 参数范围 (物理约束)
# ============================================================
# 髋距 d: 舵机体宽 35mm + 法兰, d>=38 不干涉; 上限给余量
D_RANGE = (38.0, 70.0)
# 曲柄 (大腿) a / 连杆 (小腿) b
A_RANGE = (70.0, 120.0)
B_RANGE = (110.0, 180.0)

# 选定解 (文档标称值, 供优化验证)
SELECTED = dict(d=52.0, a=93.0, b=149.0)

# D0 扫描
D0_MIN = 58.0
D0_MAX = 207.0
D0_STEP = 5.0

# 约束阈值
GAMMA_MIN = 30.0      # 传动角下限 (度)
KNEE_MIN = 12.0       # 膝角下限 (度, 奇异余量)


def compute_D0_range(d, a, b):
    """给定 (d,a,b), 求可达的 D0 范围 [D0_min, D0_max].

    粗扫描(步长 5mm)定边界, 够蒙特卡洛用.
    D0_max: 连杆共线, 足端最靠近髋面. 找最大可达 D0.
    D0_min: 膝角约束 (>= KNEE_MIN).
    Returns: (D0_min, D0_max) 或 None (无解)
    """
    coarse = np.arange(D0_MIN, D0_MAX + 5.0, 5.0)
    D0_max = None
    for D0 in coarse:
        r = kc.kin_param(d, a, b, D0)
        if r is None:
            break
        D0_max = float(D0)
    if D0_max is None or D0_max <= D0_MIN:
        return None

    # D0_min: 膝角约束
    r0 = kc.kin_param(d, a, b, D0_MIN)
    if r0 is None:
        return None
    if r0["knee"] >= KNEE_MIN:
        return (D0_MIN, D0_max)
    for D0 in coarse:
        r = kc.kin_param(d, a, b, D0)
        if r is not None and r["knee"] >= KNEE_MIN:
            return (float(D0), D0_max)
    return None


def evaluate(d, a, b):
    """计算一组 (d,a,b) 的 5 个指标.

    性能优化: D0 用粗扫描(步长 10mm)算扭矩/传动角;
    雅可比条件数只在 D0_lo/mid/hi 三点采样.
    Returns: dict 含 f1..f5 及原始量, 或 None (不满足约束).
    """
    rng = compute_D0_range(d, a, b)
    if rng is None:
        return None
    D0_lo, D0_hi = rng
    if D0_hi - D0_lo < 30:           # 行程太短, 无工程价值
        return None

    D0s = np.arange(D0_lo, D0_hi + 0.5, 10.0)
    taus, gammas = [], []
    dwell_tau = None
    for D0 in D0s:
        r = kc.kin_param(d, a, b, D0)
        if r is None:
            return None
        tau = max(r["tau1"], r["tau2"])
        taus.append(tau)
        gammas.append(r["gamma"])
        if abs(D0 - D0_lo) < 5.0:
            dwell_tau = tau

    taus = np.array(taus); gammas = np.array(gammas)

    # 硬约束: gamma >= 30 全程
    if gammas.min() < GAMMA_MIN:
        return None

    # 雅可比条件数: 仅在 lo/mid/hi 三点采样, 取最大
    cond_max = 0.0
    for D0 in [D0_lo, 0.5*(D0_lo+D0_hi), D0_hi]:
        J, _ = kc.jacobian_param(d, a, b, D0)
        if J is None:
            return None
        sv = np.linalg.svd(J, compute_uv=False)
        c = sv[0] / sv[1]
        if c > cond_max:
            cond_max = float(c)

    tau_peak = float(taus.max())
    tau_dwell = float(dwell_tau if dwell_tau is not None else taus[0])
    gamma_min = float(gammas.min())
    stroke = float(D0_hi - D0_lo)

    return dict(
        d=d, a=a, b=b,
        D0_min=float(D0_lo), D0_max=float(D0_hi),
        tau_peak=tau_peak, tau_dwell=tau_dwell,
        gamma_min=gamma_min, cond_max=cond_max, stroke=stroke,
        ab_ratio=a / b,
        # Pareto 目标向量 (均最小化)
        f1=tau_peak, f2=tau_dwell,
        f3=1.0 / gamma_min, f4=cond_max, f5=-stroke,
    )


def evaluate_selected():
    """评估选定解 (52, 93, 149), 供报告."""
    return evaluate(**SELECTED)


if __name__ == "__main__":
    print("=== 选定解 (d=52, a=93, b=149) 指标 ===")
    r = evaluate_selected()
    if r is None:
        print("  不满足约束!")
    else:
        for k in ["D0_min", "D0_max", "tau_peak", "tau_dwell",
                  "gamma_min", "cond_max", "stroke", "ab_ratio"]:
            print(f"  {k:12s} = {r[k]:.3f}")
