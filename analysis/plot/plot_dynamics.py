# -*- coding: utf-8 -*-
"""KUAFU 动力学 4 联图 — 主分析图"""
import numpy as np
import matplotlib.pyplot as plt
import kuafu as kc
from kuafu import OI, savefig, TAU_CONT, TAU_STALL, F_DES, F_GRAV, W_SERVO

D0s = np.arange(58, 208, 1)
res = [(d, kc.kin(d)) for d in D0s]
res = [(d, r) for d, r in res if r is not None]
D0v   = np.array([d for d, _ in res])
taumax= np.array([max(r["tau1"], r["tau2"]) for _, r in res])
gamma = np.array([r["gamma"] for _, r in res])
knee  = np.array([r["knee"]  for _, r in res])

# 奇异值/条件数
manip = []; cond = []
for d in D0v:
    J, _ = kc.jacobian(d)
    if J is None: manip.append(np.nan); cond.append(np.nan); continue
    sv = np.linalg.svd(J, compute_uv=False)
    manip.append(np.sqrt(sv[0]*sv[1])); cond.append(sv[0]/sv[1])
manip = np.array(manip); cond = np.array(cond)

# ---- 爬阶轨迹动力学 ----
t_t  = np.linspace(0, 1, 400)
D0_t = 104 - 46*np.cos(2*np.pi*t_t)
def _al(D0, key):
    r = kc.kin(D0); return r[key] if r else np.nan   # al1/al2 已是弧度
th1  = np.array([_al(d, "al1") for d in D0_t])
th2  = np.array([_al(d, "al2") for d in D0_t])
d1 = np.gradient(th1, t_t); d2 = np.gradient(th2, t_t)
dd1 = np.gradient(d1, t_t); dd2 = np.gradient(d2, t_t)
tau_dyn = []
for i in range(len(t_t)):
    M = kc.mass_matrix(th1[i], th2[i]); Gv = kc.gravity_vec(th1[i], th2[i])
    if M is None or Gv is None: tau_dyn.append([np.nan, np.nan]); continue
    qdd = np.array([dd1[i], dd2[i]])
    h = 1e-3
    Mp = kc.mass_matrix(th1[i]+h*d1[i], th2[i]+h*d2[i])
    dMdt = (Mp - M)/h
    Cqd = dMdt @ np.array([d1[i], d2[i]])
    tau_dyn.append(M @ qdd + Cqd + Gv)
tau_dyn = np.array(tau_dyn)

# ============================================================
fig, ax = plt.subplots(2, 2, figsize=(13.5, 10))

# ---- (a) 扭矩 + 传动角 ----
a = ax[0, 0]
a.plot(D0v, taumax, color=OI["verm"], lw=2.4, label=r"$\tau_{\max}$  (F = 30 N/腿)")
a.axhline(TAU_CONT, ls="--", color=OI["green"], lw=1.4, label=r"连续安全 $\tau_c$ = 1.0 N·m")
a.axhline(TAU_STALL, ls="--", color="k", lw=1.4, label=r"堵转 $\tau_s$ = 2.94 N·m")
a.fill_between(D0v, 0, TAU_CONT, color=OI["green"], alpha=0.10)
a.fill_between(D0v, TAU_CONT, TAU_STALL, color=OI["orange"], alpha=0.14)
a.axvspan(58, 110, color=OI["sky"], alpha=0.10)
a.text(84, 2.78, "补偿带", ha="center", fontsize=9, color="#555")
a.annotate(r"峰值 $\tau$=1.92 N·m", (145, 1.92), xytext=(150, 2.35),
           fontsize=8.5, color=OI["verm"],
           arrowprops=dict(arrowstyle="->", color=OI["verm"], lw=1))
a.set_xlabel(r"下垂量 $D_0$ (mm)")
a.set_ylabel(r"瞬态关节扭矩 $\tau$ (N·m)")
a.set_title("(a) 瞬态关节扭矩与传动角")
a.set_xlim(58, 207); a.set_ylim(0, 3.1)
a.legend(loc="upper left", fontsize=8)
a2 = a.twinx()
a2.plot(D0v, gamma, color=OI["blue"], lw=1.6, alpha=0.85, label=r"传动角 $\gamma$")
a2.plot(D0v, knee,  color=OI["pink"], lw=1.2, alpha=0.7,  label=r"膝角 $\kappa$")
a2.axhline(30, ls=":", color=OI["blue"], lw=1)
a2.set_ylabel(r"角度 (°)", color=OI["blue"])
a2.set_ylim(0, 130)
a2.legend(loc="lower right", fontsize=8)
a2.grid(False)

# ---- (b) 腿姿态包络 ----
a = ax[0, 1]
a.set_aspect("equal")
poses = [(58, OI["green"]), (100, OI["blue"]), (150, OI["verm"]), (207, OI["gray"])]
for d0, col in poses:
    r = kc.kin(d0)
    a.plot([kc.AX, r["P1"][0], r["Q"][0]], [0, r["P1"][1], r["Q"][1]],
           "-", color=col, lw=2.2, alpha=0.9)
    a.plot([kc.BX, r["P2"][0], r["Q"][0]], [0, r["P2"][1], r["Q"][1]],
           "-", color=col, lw=2.2, alpha=0.55)
    a.plot(*r["Q"], "o", color=col, ms=6, mec="k", mew=0.6)
    a.annotate(f"$D_0$={d0}", r["Q"], fontsize=8, color=col,
               xytext=(8, -4), textcoords="offset points")
a.plot([kc.AX, kc.BX], [0, 0], "ks", ms=9)
a.text(kc.AX-4, 8, "A", fontweight="bold"); a.text(kc.BX+2, 8, "B", fontweight="bold")
a.plot(0, -58, "*", color=OI["green"], ms=14, mew=0.5, mec="k")
a.set_xlim(-170, 170); a.set_ylim(-225, 45)
a.set_xlabel("X (mm)"); a.set_ylabel("Z (mm)")
a.set_title("(b) 对称步态腿姿态 (X=0 切片)")

# ---- (c) 可操作性 / 条件数 ----
a = ax[1, 0]
a.plot(D0v, manip, color=OI["green"], lw=2.4, label=r"可操作性 $w=\sqrt{\det(JJ^T)}$")
a.axvspan(58, 110, color=OI["sky"], alpha=0.10)
a.set_xlabel(r"$D_0$ (mm)")
a.set_ylabel(r"$w$ (mm²/rad)", color=OI["green"])
a.set_title("(c) 可操作性与条件数")
a.set_xlim(58, 207); a.legend(loc="upper left", fontsize=8.5)
a2 = a.twinx()
a2.plot(D0v, cond, color=OI["pink"], lw=1.8, label=r"条件数 $\kappa(J)=\sigma_1/\sigma_2$")
a2.axhline(1.0, ls=":", color=OI["pink"], lw=1)
a2.set_ylabel(r"$\kappa$", color=OI["pink"])
a2.set_ylim(0.9, 2.0); a2.legend(loc="lower right", fontsize=8.5); a2.grid(False)
a.text(84, max(manip)*0.93, "补偿带\n(各向同性)", ha="center", fontsize=8, color="#555")

# ---- (d) 爬阶轨迹动态扭矩 ----
a = ax[1, 1]
l1, = a.plot(t_t, D0_t, "-", color="k", lw=1.8, alpha=0.8)
a.set_xlabel("时间 (s)"); a.set_ylabel(r"$D_0$ (mm)", color="k")
a.set_title("(d) 爬阶轨迹动态扭矩 ($D_0$: 58→150→58, 1 s)")
a2 = a.twinx()
l2, = a2.plot(t_t, tau_dyn[:, 0], "-", color=OI["verm"], lw=1.7, label=r"$\tau_1$ 动态")
l3, = a2.plot(t_t, tau_dyn[:, 1], "-", color=OI["blue"],  lw=1.7, label=r"$\tau_2$ 动态")
a2.axhline(0, color="#888", lw=0.5)
a2.set_ylabel(r"动态扭矩增量 (N·m)", color=OI["verm"])
a.legend([l1, l2, l3], [r"$D_0(t)$", r"$\tau_1(t)$", r"$\tau_2(t)$"],
         loc="upper left", fontsize=8.5)
a2.grid(False)
a.annotate(r"峰值 0.31 N·m", (0.5, tau_dyn[200, 0]),
           xytext=(0.62, 0.42), fontsize=8.5, color=OI["verm"],
           arrowprops=dict(arrowstyle="->", color=OI["verm"], lw=1))

fig.suptitle("KUAFU 五杆髋关节 — 运动学与动力学", fontsize=14, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
savefig(fig, "dynamics.png")
print(f"  爬阶动态峰值: tau1={np.nanmax(np.abs(tau_dyn[:,0])):.3f} "
      f"tau2={np.nanmax(np.abs(tau_dyn[:,1])):.3f} Nm")
