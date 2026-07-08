# -*- coding: utf-8 -*-
"""
KUAFU 力雅可比分析 — 扭矩椭球 / 力椭球 / 奇异值谱
  关键关系: tau = J^T F   (末端力 F[N] -> 关节扭矩 tau[N·m], J 单位 mm/rad)
    - 扭矩椭球: 末端 L2 单位力 (|F|=1N) 各方向 -> tau 轨迹
    - 力椭球:   给定 |tau|_2 <= tau_lim, 末端可承受力 F 的椭球
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import kuafu as kc
from kuafu import OI, savefig, TAU_CONT, TAU_STALL, F_DES, F_GRAV

D0s_show = [58, 100, 150, 207]
col_map = {58: OI["green"], 100: OI["blue"], 150: OI["verm"], 207: OI["gray"]}

# 全程扫描数据
D0s = np.arange(58, 208, 1)
sv1 = []; sv2 = []; Fbear_worst = []
for d in D0s:
    J, _ = kc.jacobian(d)
    if J is None: sv1.append(np.nan); sv2.append(np.nan); Fbear_worst.append(np.nan); continue
    sv = np.linalg.svd(J, compute_uv=False)
    sv1.append(sv[0]); sv2.append(sv[1])
    Fbear_worst.append(TAU_CONT/kc.MM/sv[0])   # L2: |F|_max = tau_lim/MM/sigma_max
sv1 = np.array(sv1); sv2 = np.array(sv2); Fbear_worst = np.array(Fbear_worst)

# 静力法实际扭矩 (设计载)
taumax = np.array([max(kc.kin(d)["tau1"], kc.kin(d)["tau2"]) for d in D0s])

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

# ============================================================
# (a) 扭矩椭球: 末端单位力 F (|F|=1N, L2) 各方向 -> tau = J^T F
#     这是真正的椭圆 (J^T 把单位圆映成椭圆), 半轴 = sigma_i
# ============================================================
ax = fig.add_subplot(gs[0, 0])
ax.set_aspect("equal")
th = np.linspace(0, 2*np.pi, 200)
unit_F = np.array([np.cos(th), np.sin(th)])      # |F|=1 单位圆
for d0 in D0s_show:
    J, r = kc.jacobian(d0)
    col = col_map[d0]
    tau = J.T @ unit_F * kc.MM                   # tau[Nm] = J[mm]*F[N]*MM
    ax.plot(tau[0], tau[1], color=col, lw=2.2, label=f"$D_0$={d0}")
    # 主轴方向 (J 的左奇异向量 U) + 半长 sigma_i/1000
ax.add_patch(Circle((0, 0), TAU_CONT, fill=False, ls="--", color=OI["green"], lw=1.5,
                    label=r"连续安全 $\tau_c$=1 N·m"))
ax.add_patch(Circle((0, 0), TAU_STALL, fill=False, ls="--", color="k", lw=1.5,
                    label=r"堵转 $\tau_s$=2.94 N·m"))
ax.plot(0, 0, "k+", ms=11, mew=2)
ax.set_xlabel(r"$\tau_1$ (N·m)"); ax.set_ylabel(r"$\tau_2$ (N·m)")
ax.set_title("(a) 扭矩椭球: 末端单位力 $|F|$=1 N 各方向 $\\rightarrow\\ \\tau=J^TF$")
ax.set_xlim(-0.16, 0.16); ax.set_ylim(-0.16, 0.16)
ax.legend(loc="upper left", fontsize=8, ncol=2)
# 标注: 所有椭圆 << 1Nm 圆 => 任意方向 1N 外载远低于连续安全

# ============================================================
# (b) 力椭球: 给定 |tau|_2 <= tau_lim, 末端可承受力 F 的椭球
#     F = J^{-T} tau, |tau|<=tau_lim => 椭球, 主轴 = tau_lim/sigma_i
# ============================================================
ax = fig.add_subplot(gs[0, 1])
ax.set_aspect("equal")
th = np.linspace(0, 2*np.pi, 120)
unit = np.array([np.cos(th), np.sin(th)])
for d0 in D0s_show:
    J, _ = kc.jacobian(d0)
    col = col_map[d0]
    U, S, Vt = np.linalg.svd(J)
    # 力椭球: ||tau||_2 <= tau_c, 主轴方向=U列, 半长=tau_c/MM/sigma (N)
    F_ell = U @ np.diag([TAU_CONT/kc.MM/S[0], TAU_CONT/kc.MM/S[1]]) @ unit
    ax.plot(F_ell[0], F_ell[1], color=col, lw=2.2,
            label=f"$D_0$={d0}  (σ={S[0]:.0f}/{S[1]:.0f})")
ax.plot(0, 0, "k+", ms=11, mew=2)
ax.plot([0, 0], [-F_DES, 0], "-", color=OI["orange"], lw=2.5, alpha=0.8)
ax.plot(0, -F_DES, "v", color=OI["orange"], ms=9, mec="k")
ax.text(3, -F_DES, f"设计载 F={F_DES:.0f} N\n(竖直)", fontsize=8, color=OI["orange"])
ax.set_xlabel(r"$F_x$ (N, 水平)"); ax.set_ylabel(r"$F_z$ (N, 竖直)")
ax.set_title(r"(b) 力椭球: $\|\tau\|_2 \leq \tau_c$=1 N·m 时各方向可承受外载")
ax.set_xlim(-55, 55); ax.set_ylim(-55, 20)
ax.legend(loc="upper right", fontsize=8)
ax.axhline(0, color="#aaa", lw=0.5); ax.axvline(0, color="#aaa", lw=0.5)

# ============================================================
# (c) 奇异值谱 + 条件数 vs D0
# ============================================================
ax = fig.add_subplot(gs[1, 0])
ax.fill_between(D0s, sv2, sv1, color=OI["sky"], alpha=0.25, label="奇异值带")
ax.plot(D0s, sv1, color=OI["verm"], lw=2.2, label=r"$\sigma_{\max}$")
ax.plot(D0s, sv2, color=OI["blue"], lw=2.2, label=r"$\sigma_{\min}$")
ax.axvspan(58, 110, color=OI["sky"], alpha=0.12)
ax.set_xlabel(r"$D_0$ (mm)")
ax.set_ylabel(r"雅可比奇异值 $\sigma$ (mm/rad)")
ax.set_title("(c) 雅可比奇异值谱")
ax.set_xlim(58, 207); ax.legend(loc="upper left", fontsize=8.5)
ax2 = ax.twinx()
kappa = sv1/sv2
ax2.plot(D0s, kappa, "--", color=OI["pink"], lw=1.6, label=r"$\kappa=\sigma_{\max}/\sigma_{\min}$")
ax2.set_ylabel(r"条件数 $\kappa$", color=OI["pink"])
ax2.set_ylim(0.9, 2.0); ax2.legend(loc="lower right", fontsize=8.5); ax2.grid(False)

# ============================================================
# (d) 承载能力 vs 设计载
# ============================================================
ax = fig.add_subplot(gs[1, 1])
ax.plot(D0s, taumax, color=OI["verm"], lw=2.4,
        label=r"设计载 $F$=30 N 竖直 $\rightarrow \tau_{\max}$")
ax.axhline(TAU_CONT, ls="--", color=OI["green"], lw=1.4, label=r"$\tau_c$=1 N·m")
ax.axhline(TAU_STALL, ls="--", color="k", lw=1.4, label=r"$\tau_s$=2.94 N·m")
ax.fill_between(D0s, 0, TAU_CONT, color=OI["green"], alpha=0.10)
ax.fill_between(D0s, TAU_CONT, TAU_STALL, color=OI["orange"], alpha=0.14)
ax.set_xlabel(r"$D_0$ (mm)"); ax.set_ylabel(r"关节扭矩 (N·m)", color=OI["verm"])
ax.set_title("(d) 扭矩需求 vs 连续安全 / 堵转极限")
ax.set_xlim(58, 207); ax.set_ylim(0, 3.1)
ax.legend(loc="upper left", fontsize=8)
ax2 = ax.twinx()
ax2.plot(D0s, Fbear_worst, color=OI["green"], lw=2.0,
         label=r"最差方向可承受载 ($|\tau|_2\leq\tau_c$)")
ax2.axhline(F_DES, ls=":", color=OI["orange"], lw=1.4, label="设计载 30 N")
ax2.set_ylabel("可承受外载 (N)", color=OI["green"])
ax2.legend(loc="upper right", fontsize=8); ax2.grid(False)
ax2.annotate("安全区\n(可全悬空)", (58, Fbear_worst[0]),
             fontsize=8, color=OI["green"], xytext=(66, 60),
             arrowprops=dict(arrowstyle="->", color=OI["green"]))

fig.suptitle("KUAFU 静力学 — 力雅可比 $\\tau=J^TF$ 与承载能力分析",
             fontsize=14, fontweight="bold", y=0.995)
savefig(fig, "force_ellipsoid.png")
print("\n承载能力 (|tau|_2 <= 1 Nm, 最差方向):")
for d0 in D0s_show:
    J, _ = kc.jacobian(d0); sv = np.linalg.svd(J, compute_uv=False)
    print(f"  D0={d0}: sigma_max={sv[0]:.1f} -> F_max={TAU_CONT/kc.MM/sv[0]:.1f} N "
          f"(设计载{F_DES}N {'安全' if TAU_CONT/kc.MM/sv[0]>F_DES else '需脉冲'})")
