# -*- coding: utf-8 -*-
"""
KUAFU 机构解算 — 运动学 / 静力学 / 动力学 / 力椭球几何
  纯数值模块, 仅依赖 numpy. 坐标系: 原点在两髋点中点, +X 前向, +Z 向上.
  机构: 对称并联五杆髋关节, 两髋点 A/B, 曲柄 a(大腿), 连杆 b(小腿), 输出点 Q.

物理常量真源在项目顶层 kuafu_physics.py（analysis 与 rl 共用，保证零漂移）。
"""
import os, sys
import numpy as np

# 把项目根（本文件的上上两级目录）加入 path，使 kuafu_physics 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================
# 物理常量（真源在 kuafu_physics.py，此处 re-export 保持向后兼容）
# ============================================================
from kuafu_physics import (  # noqa: F401
    G, MM,
    AX, BX, A, B, A_LEN, B_LEN, XQ, R_WHEEL, D0_MIN, D0_MAX,
    TAU_WHEEL_RATED, TAU_WHEEL_STALL, RPM_WHEEL_RATED, RPM_WHEEL_NOLOAD,
    TAU_STALL, TAU_CONT, W_SERVO, SERVO_KP, SERVO_KV,
    M_TOT, F_GRAV, F_DES, M_CRANK, M_LINK, M_WHEEL,
    MC, MP, LP, R, LQR_K, OMEGA_N,
)

# ============================================================
# 运动学
# ============================================================
def solve_chain(P0, a, b, Qt, branch):
    d = Qt - P0; L = np.hypot(*d)
    if L > a + b - 1e-6 or L < abs(a - b) + 1e-6: return None
    c = (a*a + L*L - b*b) / (2*a*L)
    if abs(c) > 1: return None
    ang = np.arccos(c); base = np.arctan2(d[1], d[0])
    al = base - ang if branch else base + ang
    return P0 + a*np.array([np.cos(al), np.sin(al)]), al

def forward_kin(t1, t2):
    P1 = A + A_LEN*np.array([np.cos(t1), np.sin(t1)])
    P2 = B + A_LEN*np.array([np.cos(t2), np.sin(t2)])
    d = P2 - P1; L = np.hypot(*d)
    if L > 2*B_LEN - 1e-6 or L < 1e-6: return None
    mid = 0.5*(P1 + P2); perp = np.array([-d[1], d[0]])/L
    h = np.sqrt(max(B_LEN*B_LEN - (L/2)**2, 0))
    Qa = mid + h*perp; Qb = mid - h*perp
    return Qa if Qa[1] < Qb[1] else Qb

# MM = mm -> m 换算; 扭矩 Nmm -> Nm = ÷1000 = ×MM
#   注意: 长度在内部为 mm, 扭矩为 N·m, 两者量纲不同, 谨防混用.
def kin(D0, F=None):
    if F is None:
        F = F_DES          # 默认设计载 30N(含 3× 冲击); 静载分析请传 F_GRAV
    Q = np.array([XQ, -D0])
    r1 = solve_chain(A, A_LEN, B_LEN, Q, 1)
    r2 = solve_chain(B, A_LEN, B_LEN, Q, 0)
    if r1 is None or r2 is None: return None
    P1, al1 = r1; P2, al2 = r2
    u1 = Q - P1; u1 /= np.linalg.norm(u1)
    u2 = Q - P2; u2 /= np.linalg.norm(u2)
    Mmat = np.array([[u1[0], u2[0]], [u1[1], u2[1]]])
    try:
        T1, T2 = np.linalg.solve(Mmat, np.array([0.0, -F]))
    except np.linalg.LinAlgError:
        return None
    arm1 = abs((P1[0]-AX)*u1[1] - P1[1]*u1[0])
    arm2 = abs((P2[0]-BX)*u2[1] - P2[1]*u2[0])
    tau1 = abs(T1)*arm1*MM
    tau2 = abs(T2)*arm2*MM
    def knee_ang(P, P0, Q_):
        v1 = P0 - P; v2 = Q_ - P
        cosk = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cosk, -1, 1)))
    k1 = knee_ang(P1, A, Q); k2 = knee_ang(P2, B, Q)
    g1 = abs(np.degrees(np.arctan2(Q[0]-P1[0], -(Q[1]-P1[1]))))
    g2 = abs(np.degrees(np.arctan2(Q[0]-P2[0], -(Q[1]-P2[1]))))
    alpha1 = np.degrees(np.arctan2(P1[1], P1[0]-AX))
    return dict(P1=P1, P2=P2, Q=Q, al1=al1, al2=al2,
                tau1=tau1, tau2=tau2, T1=T1, T2=T2, u1=u1, u2=u2,
                gamma=0.5*(g1+g2), knee=0.5*(k1+k2), alpha1=alpha1,
                arm1=arm1, arm2=arm2)

def jacobian(D0, h=5e-4):
    r = kin(D0)
    if r is None: return None, r
    t1, t2 = r["al1"], r["al2"]
    J = np.zeros((2, 2))
    Qp = forward_kin(t1+h, t2); Qm = forward_kin(t1-h, t2)
    if Qp is None or Qm is None: return None, r
    J[:, 0] = (Qp - Qm)/(2*h)
    Qp = forward_kin(t1, t2+h); Qm = forward_kin(t1, t2-h)
    if Qp is None or Qm is None: return None, r
    J[:, 1] = (Qp - Qm)/(2*h)
    return J, r

def reachable(Qx, Qy):
    r1 = solve_chain(A, A_LEN, B_LEN, np.array([Qx, Qy]), 1)
    r2 = solve_chain(B, A_LEN, B_LEN, np.array([Qx, Qy]), 0)
    if r1 is None or r2 is None: return None
    P1, _ = r1; P2, _ = r2
    ok = (P1[0] < Qx) and (P2[0] > Qx)
    return dict(P1=P1, P2=P2, ok=ok)

def ik_full(Q):
    r1 = solve_chain(A, A_LEN, B_LEN, Q, 1)
    r2 = solve_chain(B, A_LEN, B_LEN, Q, 0)
    if r1 is None or r2 is None: return None
    return r1[0], r2[0]

# ============================================================
# 动力学 (拉格朗日, 集中质量模型) — 质量常量见 kuafu_physics
# ============================================================

def leg_points(t1, t2):
    P1 = A + A_LEN*np.array([np.cos(t1), np.sin(t1)])
    P2 = B + A_LEN*np.array([np.cos(t2), np.sin(t2)])
    Q  = forward_kin(t1, t2)
    if Q is None: return None
    return dict(P1=P1, P2=P2, Q=Q, cm1=0.5*(A+P1), cm2=0.5*(P1+Q))

def kinetic_energy(t1, t2, td1, td2, h=1e-4):
    st = leg_points(t1, t2)
    if st is None: return None
    v_P1 = A_LEN*np.array([-np.sin(t1), np.cos(t1)])*td1
    Qp = forward_kin(t1+h, t2); Qm = forward_kin(t1-h, t2)
    if Qp is None or Qm is None: return None
    dQ1 = (Qp - Qm)/(2*h)
    Qp = forward_kin(t1, t2+h); Qm = forward_kin(t1, t2-h)
    if Qp is None or Qm is None: return None
    dQ2 = (Qp - Qm)/(2*h)
    v_Q = dQ1*td1 + dQ2*td2
    v_cm1 = 0.5*v_P1
    v_cm2 = v_P1 + 0.5*(v_Q - v_P1)
    f = 1e-6
    return 0.5*(M_CRANK*np.dot(v_cm1, v_cm1)
              + M_LINK *np.dot(v_cm2, v_cm2)
              + M_WHEEL*np.dot(v_Q,   v_Q  ))*f

def mass_matrix(t1, t2, e=1.0):
    K00 = kinetic_energy(t1, t2, 0, 0)
    K10 = kinetic_energy(t1, t2, e, 0); K01 = kinetic_energy(t1, t2, 0, e)
    K11 = kinetic_energy(t1, t2, e, e); Km11 = kinetic_energy(t1, t2, -e, -e)
    K1m1 = kinetic_energy(t1, t2, e, -e); Km1 = kinetic_energy(t1, t2, -e, e)
    if any(x is None for x in [K00, K10, K01, K11, Km11, K1m1, Km1]): return None
    M11 = 2*(K10 - K00)/(e*e); M22 = 2*(K01 - K00)/(e*e)
    M12 = ((K11 + Km11) - (K1m1 + Km1))/(4*e*e)
    return np.array([[M11, M12], [M12, M22]])

def gravity_vec(t1, t2, h=1e-3):
    def PE(t1, t2):
        st = leg_points(t1, t2)
        if st is None: return None
        return G*(M_CRANK*st["cm1"][1] + M_LINK*st["cm2"][1] + M_WHEEL*st["Q"][1])*MM
    P0 = PE(t1, t2); Pp1 = PE(t1+h, t2); Pm1 = PE(t1-h, t2)
    Pp2 = PE(t1, t2+h); Pm2 = PE(t1, t2-h)
    if any(x is None for x in [P0, Pp1, Pm1, Pp2, Pm2]): return None
    return np.array([(Pp1 - Pm1)/(2*h), (Pp2 - Pm2)/(2*h)])

# ============================================================
# 力椭球 / 扭矩椭球 几何
#   核心关系 tau = J^T F  (J 单位 mm/rad, tau 单位 Nmm = J[mm]*F[N])
# ============================================================
def torque_ellipse(J, F=1.0):
    """末端力 |F|₂ = F 各方向→关节扭矩轨迹. 返回 (tau1, tau2) 椭圆."""
    th = np.linspace(0, 2*np.pi, 200)
    unit_F = np.array([np.cos(th), np.sin(th)]) * F
    tau = J.T @ unit_F * MM   # Nmm -> Nm
    return tau[0], tau[1]

def force_ellipse(J, tau_lim=1.0):
    """给定 |τ|₂ ≤ tau_lim, 末端可承受力 F 的椭球.
    Returns (axes, angles_deg, F_boundary_points)."""
    U, S, Vt = np.linalg.svd(J)
    th = np.linspace(0, 2*np.pi, 200)
    unit = np.array([np.cos(th), np.sin(th)])
    F_ell = U @ np.diag([tau_lim/MM/S[0], tau_lim/MM/S[1]]) @ unit
    axes = tau_lim/MM/S  # 半轴长 (N)
    angles = np.degrees(np.arctan2(U[1], U[0]))  # 主轴方向
    return axes, angles, F_ell

# ============================================================
# 参数化机构解算 (供蒙特卡洛优化使用)
#   与上面的全局常量版本并存: 全局常量函数读 AX/BX/A_LEN/B_LEN,
#   参数化版本接收显式 d/a/b. 现有测试与 plot 脚本完全不受影响.
# ============================================================
def _hip_points(d):
    """由髋距 d 返回两髋点 A, B (对称于 Y 轴)."""
    h = d / 2.0
    return np.array([-h, 0.0]), np.array([h, 0.0])

def kin_param(d, a, b, D0, F=None):
    """参数化静力学解算. 逻辑同 kin(), 但接收显式 d/a/b/D0.

    Args:
        d: 髋距 (mm); a: 曲柄长 (mm); b: 连杆长 (mm)
        D0: 足端下垂量 (mm); F: 足端设计载荷 (N), 默认 F_DES
    Returns:
        与 kin() 相同结构的 dict, 或 None (不可达/奇异)
    """
    if F is None:
        F = F_DES
    A_, B_ = _hip_points(d)
    Q = np.array([XQ, -D0])
    r1 = solve_chain(A_, a, b, Q, 1)
    r2 = solve_chain(B_, a, b, Q, 0)
    if r1 is None or r2 is None:
        return None
    P1, al1 = r1; P2, al2 = r2
    # 对称非交叉构型检查: P1 落 Q 左, P2 落 Q 右
    if not (P1[0] < Q[0] and P2[0] > Q[0]):
        return None
    u1 = Q - P1; u1 /= np.linalg.norm(u1)
    u2 = Q - P2; u2 /= np.linalg.norm(u2)
    Mmat = np.array([[u1[0], u2[0]], [u1[1], u2[1]]])
    try:
        T1, T2 = np.linalg.solve(Mmat, np.array([0.0, -F]))
    except np.linalg.LinAlgError:
        return None
    arm1 = abs((P1[0]-A_[0])*u1[1] - P1[1]*u1[0])
    arm2 = abs((P2[0]-B_[0])*u2[1] - P2[1]*u2[0])
    tau1 = abs(T1)*arm1*MM
    tau2 = abs(T2)*arm2*MM
    def knee_ang(P, P0, Q_):
        v1 = P0 - P; v2 = Q_ - P
        cosk = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cosk, -1, 1)))
    k1 = knee_ang(P1, A_, Q); k2 = knee_ang(P2, B_, Q)
    g1 = abs(np.degrees(np.arctan2(Q[0]-P1[0], -(Q[1]-P1[1]))))
    g2 = abs(np.degrees(np.arctan2(Q[0]-P2[0], -(Q[1]-P2[1]))))
    return dict(P1=P1, P2=P2, Q=Q, al1=al1, al2=al2,
                tau1=tau1, tau2=tau2, T1=T1, T2=T2, u1=u1, u2=u2,
                gamma=0.5*(g1+g2), knee=0.5*(k1+k2),
                arm1=arm1, arm2=arm2, A=A_, B=B_)

def forward_kin_param(d, a, b, t1, t2):
    """参数化正向运动学. 返回 Q 或 None."""
    A_, B_ = _hip_points(d)
    P1 = A_ + a*np.array([np.cos(t1), np.sin(t1)])
    P2 = B_ + a*np.array([np.cos(t2), np.sin(t2)])
    dd = P2 - P1; L = np.hypot(*dd)
    if L > 2*b - 1e-6 or L < 1e-6:
        return None
    mid = 0.5*(P1 + P2); perp = np.array([-dd[1], dd[0]])/L
    h = np.sqrt(max(b*b - (L/2)**2, 0))
    Qa = mid + h*perp; Qb = mid - h*perp
    return Qa if Qa[1] < Qb[1] else Qb

def jacobian_param(d, a, b, D0, h=5e-4):
    """参数化雅可比 (数值差分). 返回 (J, kin_result) 或 (None, kin_result)."""
    r = kin_param(d, a, b, D0)
    if r is None:
        return None, r
    t1, t2 = r["al1"], r["al2"]
    J = np.zeros((2, 2))
    Qp = forward_kin_param(d, a, b, t1+h, t2); Qm = forward_kin_param(d, a, b, t1-h, t2)
    if Qp is None or Qm is None:
        return None, r
    J[:, 0] = (Qp - Qm)/(2*h)
    Qp = forward_kin_param(d, a, b, t1, t2+h); Qm = forward_kin_param(d, a, b, t1, t2-h)
    if Qp is None or Qm is None:
        return None, r
    J[:, 1] = (Qp - Qm)/(2*h)
    return J, r

