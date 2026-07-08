# -*- coding: utf-8 -*-
"""
KUAFU 整机倒立摆 — LQR 控制器设计与闭环仿真验证
  (a) cart-pole 模型示意 + 参数
  (b) 闭环极点位置 (s 平面)
  (c) 不同初始倾角的 LQR 闭环响应
  (d) 非线性模型相图 + 闭环轨迹
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import kuafu as kc
from kuafu import OI, savefig, G, MM, R_WHEEL, M_TOT
from kuafu_physics import MC, MP, LP, LQR_K, OMEGA_N

# ---- cart-pole 参数 (从物理真源 kuafu_physics 导入) ----
mc = MC                            # cart = 轮 (DDSM315 ×2)
mp = MP                            # pendulum = 机身 + 腿 + 电子件
lp = LP                            # 摆长 = pendulum 质心相对轮轴
Lp = lp
Ip = mp*lp*lp/3
R = R_WHEEL*MM
# 惯性矩阵元
M11 = mc + mp
M12 = mp*lp
M22 = mp*lp*lp + Ip
detM = M11*M22 - M12*M12

# 线性化状态空间: [x, theta, xdot, thetadot], 输入 u=地面力 F
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, -M12*mp*G*lp/detM, 0, 0],
    [0,  M11*mp*G*lp/detM, 0, 0]])
B = np.array([[0], [0], [M22/detM], [-M12/detM]])

# LQR
Ql = np.diag([10.0, 50.0, 1.0, 1.0])
Rl = np.array([[0.5]])
P = solve_continuous_are(A, B, Ql, Rl)
K = np.linalg.solve(Rl, B.T@P)
eig = np.linalg.eigvals(A - B@K)
wn = np.sqrt(M11*mp*G*lp/detM)

# ---- 非线性 cart-pole 动力学 (用于仿真验证) ----
def cartpole_dyn(t, s):
    x, th, xd, thd = s
    st, ct = np.sin(th), np.cos(th)
    # 非线性惯量矩阵
    M = np.array([[M11, M12*ct], [M12*ct, M22]])
    cterm = np.array([-M12*st*thd*thd, -mp*G*lp*st])
    rhs = np.array([0.0, 0.0])
    # 控制力 F = -K s
    F = float((-K @ np.array(s))[0])
    rhs[0] = F
    acc = np.linalg.solve(M, rhs - cterm)
    return [xd, thd, acc[0], acc[1]]

# ---- 闭环仿真: 不同初始倾角 ----
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

# ============================================================
# (a) 模型示意
# ============================================================
ax = fig.add_subplot(gs[0, 0])
ax.set_aspect("equal"); ax.axis("off")
# 地面
ax.plot([-1.2, 1.2], [0, 0], "k-", lw=2)
# cart
ax.add_patch(plt.Rectangle((-0.18, 0.0), 0.36, 0.12, fc=OI["blue"], ec="k"))
# 轮
for wx in [-0.13, 0.13]:
    ax.add_patch(plt.Circle((wx, 0.05), 0.05, fc="white", ec="k", lw=1.5))
# pendulum (倾角 theta)
th0 = np.radians(15)
px = 0.18*np.sin(th0); pz = 0.12 + Lp*np.cos(th0)*3
ax.plot([0, px], [0.12, pz], "-", color=OI["verm"], lw=4)
ax.add_patch(plt.Circle((px, pz), 0.04, fc=OI["verm"], ec="k"))
# 标注
ax.annotate("", xy=(0.32, 0.06), xytext=(0.55, 0.06),
            arrowprops=dict(arrowstyle="->", color=OI["green"], lw=2))
ax.text(0.43, 0.08, "F", color=OI["green"], fontsize=12, fontweight="bold")
ax.text(0.02, pz + 0.05, r"$\theta$", fontsize=13)
ax.text(-1.15, 0.35, f"mc = {mc} kg\nmp = {mp} kg\nLp = {Lp*1e3:.1f} mm\n"
        f"R = {R*1e3:.1f} mm\nIp = {Ip*1e6:.1f} g·m²",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc"))
ax.text(-1.15, -0.25, "State: $[x,\\, \\theta,\\, \\dot{x},\\, \\dot{\\theta}]$\n"
        "Input: ground force $F = \\tau_{wheel}/R$",
        fontsize=9, family="monospace")
ax.set_xlim(-1.2, 1.0); ax.set_ylim(-0.35, 0.7)
ax.set_title("(a) cart-pole model (wheeled inverted pendulum)")

# ============================================================
# (b) 闭环极点 (s 平面)
# ============================================================
ax = fig.add_subplot(gs[0, 1])
ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
ax.fill_between([-30, 0], -15, 15, color=OI["green"], alpha=0.08)
ax.text(-14, 12, "Stable\n(LHP)", fontsize=9, color=OI["green"], ha="center")
for e in eig:
    ax.plot(e.real, e.imag, "x", color=OI["verm"], ms=12, mew=2.5)
    ax.annotate(f"{e.real:.1f}{'+' if e.imag >= 0 else '-'}{abs(e.imag):.1f}j",
                (e.real, e.imag), fontsize=8, xytext=(6, 6),
                textcoords="offset points", color=OI["verm"])
ax.set_xlabel("实部 Re(s)"); ax.set_ylabel("虚部 Im(s)")
ax.set_title(f"(b) 闭环极点 (不稳定 $\\omega_n$={wn:.1f} rad/s → 闭环稳定)")
ax.set_xlim(-26, 3); ax.set_ylim(-12, 12)
ax.set_aspect("equal")

# ============================================================
# (c) LQR 闭环响应 (theta, x) — 非线性仿真
# ============================================================
ax = fig.add_subplot(gs[1, 0])
ax2 = ax.twinx()
tspan = (0, 3); teval = np.linspace(0, 3, 300)
for th0_deg, col in [(5, OI["blue"]), (10, OI["green"]), (20, OI["orange"]), (30, OI["verm"])]:
    s0 = [0, np.radians(th0_deg), 0, 0]
    sol = solve_ivp(cartpole_dyn, tspan, s0, t_eval=teval, rtol=1e-7, atol=1e-9)
    ax.plot(sol.t, np.degrees(sol.y[1]), "-", color=col, lw=2,
            label=rf"$\theta_0$={th0_deg}°")
    ax2.plot(sol.t, sol.y[0]*1000, "--", color=col, lw=1.3, alpha=0.7)
ax.axhline(0, color="#888", lw=0.5)
ax.set_xlabel("时间 (s)")
ax.set_ylabel(r"$\theta$ (°)", color=OI["verm"])
ax2.set_ylabel("x 位移 (mm)", color=OI["blue"])
ax.set_title("(c) LQR 闭环响应 (非线性仿真): 实线 θ, 虚线 x")
ax.set_xlim(0, 3); ax.legend(loc="upper right", fontsize=8.5)
ax2.grid(False)

# ============================================================
# (d) 相图 (theta, thetadot) + 闭环轨迹
# ============================================================
ax = fig.add_subplot(gs[1, 1])
# 开环相图箭头
TH, THD = np.meshgrid(np.linspace(-0.6, 0.6, 13), np.linspace(-8, 8, 13))
dTH = THD
# 开环 thetadot: 非线性 ddot theta (F=0)
dTHD = np.zeros_like(TH)
for i in range(TH.shape[0]):
    for j in range(TH.shape[1]):
        th = TH[i, j]
        M = np.array([[M11, M12*np.cos(th)], [M12*np.cos(th), M22]])
        rhs = np.array([0.0, 0.0]) - np.array([0, -mp*G*lp*np.sin(th)])
        # 这里 rhs 已含重力项; F=0
        acc = np.linalg.solve(M, np.array([0, mp*G*lp*np.sin(th)]))
        dTHD[i, j] = acc[1]
ax.streamplot(TH, THD, dTH, dTHD, color="#cccccc", density=1.2, linewidth=0.7)
# 闭环轨迹
for th0_deg, col in [(5, OI["blue"]), (15, OI["orange"]), (25, OI["verm"])]:
    s0 = [0, np.radians(th0_deg), 0, 0]
    sol = solve_ivp(cartpole_dyn, (0, 4), s0, t_eval=np.linspace(0, 4, 400),
                    rtol=1e-7, atol=1e-9)
    ax.plot(sol.y[1], sol.y[3], "-", color=col, lw=2, label=rf"$\theta_0$={th0_deg}°")
    ax.plot(sol.y[1][0], sol.y[3][0], "o", color=col, ms=6)
ax.plot(0, 0, "k*", ms=14, mec="k", label="平衡点")
ax.set_xlabel(r"$\theta$ (rad)"); ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
ax.set_title("(d) 相图: 开环流场 (灰) + LQR 闭环轨迹 (彩)")
ax.set_xlim(-0.6, 0.6); ax.set_ylim(-8, 8)
ax.legend(loc="upper right", fontsize=8.5)

fig.suptitle("KUAFU 整机平衡 — 轮式倒立摆 LQR 控制器设计与验证",
             fontsize=14, fontweight="bold", y=0.995)
savefig(fig, "lqr_balance.png")
print(f"\n闭环极点: {np.array2string(np.sort_complex(eig), precision=2)}")
print(f"LQR 增益 K = {np.array2string(K.flatten(), precision=2)}")
print("\n闭环响应恢复时间 (theta 衰减到 1°):")
for th0_deg in [5, 10, 20, 30]:
    s0 = [0, np.radians(th0_deg), 0, 0]
    sol = solve_ivp(cartpole_dyn, (0, 5), s0, t_eval=np.linspace(0, 5, 500),
                    rtol=1e-7, atol=1e-9)
    th = np.degrees(sol.y[1])
    idx = np.where(np.abs(th) < 1.0)[0]
    t_settle = sol.t[idx[0]] if len(idx) else 5.0
    print(f"  θ0={th0_deg:>2}°: 恢复时间 ~{t_settle:.2f} s, 最大位移 {sol.y[0].max()*1000:.0f} mm")
