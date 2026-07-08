#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KUAFU — 一键运行全部分析与测试."""
import sys, os, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(ROOT, "plot")
TEST_DIR = os.path.join(ROOT, "test")
OPT_DIR = os.path.join(ROOT, "optimize")

# 机构正向分析绘图 (快)
plot_scripts = ["plot_dynamics.py", "plot_force_ellipsoid.py",
                "plot_workspace.py", "plot_lqr.py"]
# 蒙特卡洛优化流水线 (采样 + 前沿 + Dirichlet + 绘图, 较慢)
opt_modules = ["optimize.monte_carlo", "optimize.dirichlet_weights",
               "optimize.plot_optimization"]

print("=" * 60)
print("KUAFU 分析 — 批量运行")
print("=" * 60)

print("\n[1/3] 机构正向分析绘图 ...")
for s in plot_scripts:
    path = os.path.join(PLOT_DIR, s)
    print(f"\n--- {s} ---")
    subprocess.run([sys.executable, path], cwd=ROOT)

print("\n" + "=" * 60)
print("[2/3] 蒙特卡洛参数优化 (采样 → Pareto → Dirichlet → 绘图) ...")
print("=" * 60)
for m in opt_modules:
    print(f"\n--- {m} ---")
    subprocess.run([sys.executable, "-m", m], cwd=ROOT)

print("\n" + "=" * 60)
print("[3/3] 运行单元测试 ...")
print("=" * 60)
subprocess.run([sys.executable, "-m", "pytest", TEST_DIR, "-v"], cwd=ROOT)

print("\n完成.")
