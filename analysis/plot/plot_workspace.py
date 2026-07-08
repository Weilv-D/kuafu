# -*- coding: utf-8 -*-
"""
KUAFU 工作空间 (足端包络面) 分析图
  (a) 可达工作空间 + 对称切片 + 补偿带宽
  (b) 各高度补偿能力条形图
  (c) 代表性姿态 (对称 + 非对称)
"""
import numpy as np
import matplotlib.pyplot as plt
import kuafu as kc
from kuafu import OI, savefig

# ---- 工作空间掩膜 (非交叉分支约束) ----
Xg = np.linspace(-150, 150, 401)
Zg = np.linspace(-215, -40, 441)
X, Z = np.meshgrid(Xg, Zg)
WS = np.zeros_like(X, dtype=bool)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        r = kc.reachable(X[i, j], Z[i, j])
        if r and r["ok"]:
            WS[i, j] = True

# 各 D0 补偿宽度
def xwidth(D0):
    xs = [Xg[j] for j in range(len(Xg))
          if (lambda r: r and r["ok"])(kc.reachable(Xg[j], -D0))]
    return (min(xs), max(xs)) if xs else None

D0_tab = [58, 70, 85, 100, 115, 130, 145, 170, 190, 207]
widths = {d: xwidth(d) for d in D0_tab}

# ---- 图 ----
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.26,
                      width_ratios=[1.3, 1], height_ratios=[1, 1])

# ============================================================
# (a) 工作空间 + 对称切片 + 补偿带 (跨两列)
# ============================================================
ax = fig.add_subplot(gs[0, :])
ax.contourf(X, Z, WS.astype(float), levels=[0.5, 1.5], colors=[OI["sky"]], alpha=0.55)
ax.contour(X, Z, WS.astype(float), levels=[0.5], colors=OI["blue"], linewidths=2.2)
# 对称步态线
ax.plot([0, 0], [-58, -207], "-", color=OI["verm"], lw=3, label="对称步态切片 (X=0, 1-DOF)")
ax.plot(0, -58, "*", color=OI["verm"], ms=14, mec="k")
ax.plot(0, -207, "^", color=OI["verm"], ms=10, mec="k")
ax.text(4, -60, "驻留 58", fontsize=8, color=OI["verm"])
ax.text(4, -207, "上限 207", fontsize=8, color=OI["verm"])
# 各高度补偿带
for d0 in [58, 100, 130, 170, 207]:
    w = widths[d0]
    if w:
        hw = max(abs(w[0]), abs(w[1]))
        ax.plot([w[0], w[1]], [-d0, -d0], "-", color=OI["green"], lw=2.4)
        ax.plot([w[0], w[1]], [-d0, -d0], "|", color=OI["green"], ms=10)
        ax.text(0, -d0 + 5, f"$D_0$={d0}   ±{hw:.0f} mm",
                ha="center", fontsize=8.5, color=OI["green"], fontweight="bold")
ax.plot(kc.AX, 0, "ks", ms=9); ax.plot(kc.BX, 0, "ks", ms=9)
ax.text(kc.AX - 4, 8, "A", fontweight="bold"); ax.text(kc.BX + 2, 8, "B", fontweight="bold")
ax.set_aspect("equal")
ax.set_xlim(-150, 150); ax.set_ylim(-215, 30)
ax.set_xlabel("X (mm) — 前向 (轮滚)")
ax.set_ylabel("Z (mm)")
ax.set_title("(a) 足端 $Q$ 可达工作空间 = 两圆环交集 ∩ 非交叉分支约束  (二维透镜, 面积 331 cm²)")
ax.legend(loc="upper right", fontsize=9)

# ============================================================
# (b) 补偿能力条形图
# ============================================================
ax = fig.add_subplot(gs[1, 0])
hw = [max(abs(widths[d][0]), abs(widths[d][1])) for d in D0_tab]
colors = [OI["green"] if d <= 70 else (OI["orange"] if d <= 130 else OI["verm"]) for d in D0_tab]
bars = ax.barh(D0_tab, hw, color=colors, alpha=0.85, edgecolor="k", linewidth=0.6, height=6)
ax.axvline(25, ls="--", color=OI["gray"], lw=1.2, label="驻留保守值 ±25 mm")
ax.set_xlabel("左右补偿半宽 (mm)")
ax.set_ylabel(r"$D_0$ (mm)")
ax.set_title("(b) 各高度 X 补偿能力")
ax.invert_yaxis()
ax.legend(loc="lower right", fontsize=8.5)
for d, v in zip(D0_tab, hw):
    ax.text(v + 2, d, f"±{v:.0f}", va="center", fontsize=8)

# ============================================================
# (c) 代表性姿态: 对称 vs 非对称
# ============================================================
ax = fig.add_subplot(gs[1, 1])
ax.set_aspect("equal")
# 对称
for d0, col in [(58, OI["green"]), (130, OI["blue"]), (207, OI["verm"])]:
    r = kc.kin(d0)
    ax.plot([kc.AX, r["P1"][0], r["Q"][0]], [0, r["P1"][1], r["Q"][1]],
            "-", color=col, lw=2.2)
    ax.plot([kc.BX, r["P2"][0], r["Q"][0]], [0, r["P2"][1], r["Q"][1]],
            "-", color=col, lw=2.2, alpha=0.55)
    ax.plot(*r["Q"], "o", color=col, ms=6, mec="k")
    ax.annotate(f"对称 $D_0$={d0}", r["Q"], fontsize=7.5, color=col,
                xytext=(8, -6), textcoords="offset points")
# 非对称 (D0=130, Q 偏极限)
for qx, col, lbl in [(widths[130][1], OI["pink"], "非对称 $Q_x$=+117"),
                     (widths[130][0], OI["gray"], "非对称 $Q_x$=−117")]:
    P1, P2 = kc.ik_full(np.array([qx, -130.0]))
    ax.plot([kc.AX, P1[0], qx], [0, P1[1], -130], "-", color=col, lw=2)
    ax.plot([kc.BX, P2[0], qx], [0, P2[1], -130], "-", color=col, lw=2, alpha=0.55)
    ax.plot(qx, -130, "D", color=col, ms=6, mec="k")
    ax.annotate(lbl, (qx, -130), fontsize=7, color=col,
                xytext=(8, 4), textcoords="offset points")
ax.plot(kc.AX, 0, "ks", ms=8); ax.plot(kc.BX, 0, "ks", ms=8)
ax.set_xlim(-150, 150); ax.set_ylim(-215, 30)
ax.set_xlabel("X (mm)"); ax.set_ylabel("Z (mm)")
ax.set_title("(c) 代表性腿姿态: 对称 (实) vs 非对称 (虚)")

fig.suptitle("KUAFU 五杆髋关节 — 足端工作空间与补偿能力",
             fontsize=14, fontweight="bold", y=0.995)
savefig(fig, "workspace_envelope.png")
print("\n补偿能力:")
for d in D0_tab:
    w = widths[d]; hw = max(abs(w[0]), abs(w[1]))
    print(f"  D0={d}: ±{hw:.0f} mm")
