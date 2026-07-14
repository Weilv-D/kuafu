# -*- coding: utf-8 -*-
"""
KUAFU 物理常量 — 项目级单值真源

所有数值取自《docs/KUAFU.md》单一真源。rl/（仿真、训练、部署）统一引用本文件，
保证机构参数、执行器规格、动力学量在仿真、训练、部署各环节零漂移。
机构参数的运动学/静力学/动力学结论与复核方法见《docs/KUAFU.md》§六。

单位约定：长度 mm（除非注明 m）、质量 kg、力 N、力矩 N·m、角度 度、时间 s。
"""
import hashlib
import os

import numpy as np

# ============================================================
# 通用物理常量
# ============================================================
G = 9.81                # 重力加速度 m/s²
MM = 1e-3               # mm → m 换算系数

# ============================================================
# 机构参数（KUAFU §五 结构设计 / 附录 A）
#   对称并联五杆髋关节，运动限定在 X-Z 平面，转轴 Y。
# ============================================================
AX, BX = -26.0, 26.0    # 两髋点 X 坐标，髋距 d = BX-AX = 52 mm
A = np.array([AX, 0.0]) # 左髋点（舵机轴心）
B = np.array([BX, 0.0]) # 右髋点（舵机轴心）
A_LEN = 93.0            # 曲柄（大腿）长度 mm，左右对称
B_LEN = 149.0           # 连杆（小腿）长度 mm，左右对称
XQ = 0.0                # 输出点 Q 的 X 偏移（对称步态为 0，常量勿改）
R_WHEEL = 39.08         # 轮半径 mm（Ø78.16）
WHEEL_WIDTH_MM = 34.8    # 轮胎轴向宽度 mm

# D₀ 行程（足端下垂量）
D0_MIN = 58.0           # 驻留态（最低姿态，舵机零力矩自锁）
D0_MAX = 207.0          # 最大伸展（爬阶上限，抬轮 149 mm）
D0_DWELL = 58.0         # 驻留态别名

# 对称步态曲柄半行程 rad (D0 58→207mm 对应单侧驱动曲柄偏转 ±1.52rad,
# 由纯运动学求解器 + MuJoCo 实测标定, 见 rl/kuafu.xml equality polycoef)
HIP_STROKE = 1.52
QX_RESIDUAL_SCALE = 20.0
D0_RESIDUAL_SCALE = 30.0
IK_GENERATOR_VERSION = "fivebar-grid-v2-relative-fk"

# ============================================================
# 执行器规格（KUAFU §3.1 硬件）
# ============================================================
# 轮毂电机 DDSM315 ×2（力矩控制，准直驱）
TAU_WHEEL_RATED = 0.55  # 单轮额定扭矩 N·m
TAU_WHEEL_STALL = 1.1   # 单轮堵转扭矩 N·m（额定 2×）
RPM_WHEEL_RATED = 200   # 额定转速 rpm（轮缘 0.82 m/s）
RPM_WHEEL_NOLOAD = 315  # 空载转速 rpm（轮缘 1.29 m/s）

# 髋关节舵机 ST3215 C018 ×4（位置控制）
TAU_STALL = 2.94        # 堵转扭矩 N·m @12V（1:345 金属齿）
TAU_CONT = 1.0          # 连续安全扭矩 N·m
W_SERVO = 4.7           # 空载转速 rad/s（45 rpm，0.222 s/60°）
SERVO_KP = 80.0         # 位置环 P 增益初值 N·m/rad（标定范围 80–120，首件实测）
SERVO_KV = 2.0          # 位置环 D 增益初值（标定范围 1–3）

# ============================================================
# 结构件材料（CAD 实测体积，9000he / HP MJF PA12）
# ============================================================
RHO_STRUCT = 1.01       # 成型件密度 g/cm³（HP MJF PA12, ASTM D792）
# CAD 实测体积 cm³：叉口与小腿一体成型
V_THIGH = 35.54         # 大腿（曲柄）
V_SHANK_A = 36.23       # 小腿 a 变体（连杆 + 轮端叉口一体）
V_SHANK_B = 33.34       # 小腿 b 变体（连杆 + 轮端叉口一体）
V_DECK_TOP = 177.49     # 上甲板
V_DECK_BOT = 366.43     # 下甲板（含 Y=±98 侧壁 + 电池凹槽）

# ============================================================
# 载荷与整机质量（KUAFU §5.4 质量分布，按 9000he 实测重算）
# ============================================================
M_TOT = 2.679           # 整机总质量 kg（含充电宝）
F_GRAV = M_TOT * G / 2.0   # 单腿静载 N（整机均分两腿）≈ 13.14
F_DES = 30.0            # 设计载荷 N/腿（含 3× 冲击系数）

# 腿部集中质量模型（KUAFU §6.3 拉格朗日动力学）
M_CRANK = 0.036         # 曲柄（大腿）等效质量 kg（CAD 35.54cm³×1.01，质心 a/2）
M_LINK = 0.035          # 连杆（小腿）等效质量 kg（CAD 均值 34.8cm³×1.01，质心中点）
M_WHEEL = 0.349         # 轮端集中质量 kg（DDSM315 集中在 Q）

# ============================================================
# 整机倒立摆 / LQR 物理基线（KUAFU §6.4）
#   驻留态腿被动自锁，整机降为 2 阶轮式倒立摆。
# ============================================================
MC = 0.698              # cart-pole cart 质量 kg（轮 DDSM315 ×2）
MP = 1.981              # cart-pole pendulum 质量 kg（机身 + 舵机 + 电子件 + 腿 + 甲板）
LP = 0.0560             # pendulum 摆长 m（pendulum 质心相对轮轴: Z_p=95.1mm - 轮轴39.08mm = 56.0mm）
R = R_WHEEL * MM        # 轮半径 m（cart-pole 输入 F·R = τ_wheel）

OMEGA_N = 17.17         # 不稳定自然频率 rad/s（周期 0.37 s，摔倒 ~65 ms）

# ============================================================
# 基层控制增益 — yaw 跟踪 + roll 调平 (STM32 250Hz 兜底层, sim 同源)
#   基层 = LQR pitch + yaw heading/rate 跟踪 + roll 左右 D0 差 PD
#   RL 残差叠在基层之上; RL 失效 → pitch/yaw/roll 三轴仍有兜底
#   增益为保守初值, 首件实测 + PPO 后标定
# ============================================================
YAW_KP = 0.30           # heading error -> yaw torque (N m/rad)
YAW_KD = 0.05           # yaw-rate error -> yaw torque (N m s/rad)

# roll: ΔD0 = -Kp_r·roll - Kd_r·ωx，单位 mm
# 保守初值, 首件台架标定: 几何 ~196mm 轮距, 3° roll 需 ~10mm 差, Kp ≈ 190 mm/rad
ROLL_KP = 190.0
ROLL_KD = 5.0

# D0 高速门控阈值 (arbiter 遥控安全用)
D0_GATE_V_THRESH = 0.3   # |v| > 此值时限制 D0_max
D0_GATE_W_THRESH = 0.6   # |ω| > 此值时限制 D0_max
D0_GATE_MAX_HIGH = 120.0 # 高速时 D0 上限 mm (防抬 COM topple, 临界裕度 0.19kg)

# ============================================================
# 域随机化范围（design.md §2.4，供 RL env 注入）
# ============================================================
DR_MASS = (0.85, 1.15)        # 整机质量 ×[0.85, 1.15]
DR_COM = (-0.020, 0.020)      # COM 偏移 m（X/Y/Z 各 ±20 mm）
DR_INERTIA = (0.5, 2.0)       # 转动惯量 ×[0.5, 2.0]
DR_FRICTION = (0.3, 1.2)      # 轮-地摩擦系数 μ
DR_WHEEL_R = (-0.001, 0.001)  # 轮半径偏移 m（±1 mm）
DR_TORQUE_CONST = (0.9, 1.1)  # DDSM 力矩常数 ×[0.9, 1.1]
DR_SERVO_PD = (0.7, 1.3)      # ST3215 PD 增益 ×[0.7, 1.3]
DR_DEADBAND = (0.0, 0.035)    # 舵机死区 rad（[0, 2°]）
DR_DELAY_ACT = (0.0, 0.030)      # 执行器延迟 s（[0, 30]ms）
DR_DELAY_SENSE = (0.0, 0.020) # 传感器延迟 s（[0, 20]ms）

# ============================================================
# 控制频率（单一真源：物理 500Hz / 基层 250Hz / RL 残差 50Hz）
#   仿真按真实多速率运行，禁止各层各自硬编码控制率。
# ============================================================
PHYS_DT = 1.0 / 500.0   # 物理积分周期 s（MuJoCo timestep）
BASE_DT = 1.0 / 250.0   # 基层 LQR/LQI 控制周期 s（STM32 主控制环）
RL_DT = 1.0 / 50.0      # RL 残差控制周期 s（Pi5 policy 频率）
PHYS_SUBSTEPS_PER_BASE = round(BASE_DT / PHYS_DT)   # 5
BASE_STEPS_PER_RL = round(RL_DT / BASE_DT)          # 5

# ============================================================
# LQR / LQI 合成（单一真源：禁止手填 K，参数变化必须重新生成）
#   倒立摆线性化模型：[x, θ, ẋ, θ̇]，输入 F(N) 地面水平力。
#   轮扭矩 ↔ F：F = (τ_L + τ_R) / R_wheel，τ_pitch=(τ_L+τ_R)/2（见 contract.py）。
#   反馈律：F = -K·e，e = [x-x_ref, θ, ẋ-ẋ_ref, θ̇]。
#   离散化：零阶保持（ZOH），dt = BASE_DT（250Hz 基层）。
# ============================================================
try:
    from scipy.linalg import expm, solve_discrete_are
except Exception:  # pragma: no cover
    expm = None
    solve_discrete_are = None


def cartpole_continuous(mc: float = MC, mp: float = MP, lp: float = LP,
                        ip: float = None, g: float = G):
    """线性化轮式倒立摆连续模型 A, B（状态 [x,θ,ẋ,θ̇]，输入 F）。

    约定：+x 前，+θ 机身前倾；+F ⇒ +ẋ（前进）。
    返回 (A, B)，A:4×4, B:4×1。
    """
    if ip is None:
        ip = (1.0 / 3.0) * mp * lp ** 2   # 匀质杆绕质心惯量（首件实测后覆盖）
    m = mp
    m_plus = mc + mp
    den = (ip + m * lp ** 2) - (m * lp) ** 2 / m_plus
    a = m * g * lp / den                 # θ 项系数（不稳定 +）
    c = m * lp / m_plus                  # ẋ 经 θ̈ 影响 ẍ 的系数
    f_th = -c * a                        # ẍ 中 θ 系数
    f_f = 1.0 / m_plus + c * (c / den)   # ẍ 中 F 系数
    f_thdd = -c / den                    # θ̈ 中 F 系数（负）
    A = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, f_th, 0.0, 0.0],
        [0.0, a, 0.0, 0.0],
    ], float)
    B = np.array([[0.0], [0.0], [f_f], [f_thdd]], float)
    return A, B


def discretize_zoh(A, B, dt: float):
    """零阶保持离散化：返回 (Ad, Bd)。"""
    if expm is None:
        raise RuntimeError("scipy 不可用，无法离散化 LQR")
    n, m = B.shape
    Z = np.zeros((n + m, n + m))
    Z[:n, :n] = A
    Z[:n, n:] = B
    Md = expm(Z * dt)
    return Md[:n, :n], Md[:n, n:]


def dlqr(Ad, Bd, Q, R):
    """离散代数 Riccati 求解，返回 (K, P, closed_loop_poles)。反馈 F=-K·e。"""
    if solve_discrete_are is None:
        raise RuntimeError("scipy 不可用，无法求解 DLQR")
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
    poles = np.linalg.eigvals(Ad - Bd @ K)
    return K, P, poles


# Controller design weights are physical-model parameters, not copied gains.  The
# position/integral weights were selected by the native B0 hold/command-direction
# gate; gains below are regenerated from these values at import/generation time.
LQR_Q_DIAG = (1000.0, 50.0, 1.0, 1.0)
LQI_QI = 1000.0
LQR_R = 0.5


def synth_lqr_k(dt: float = BASE_DT,
                q_diag=LQR_Q_DIAG, r_val=LQR_R):
    """从 canonical 参数合成离散 LQR 增益 K（1×4）。F = -K·e。"""
    A, B = cartpole_continuous()
    Ad, Bd = discretize_zoh(A, B, dt)
    Q = np.diag(q_diag)
    R = np.array([[r_val]])
    K, _P, poles = dlqr(Ad, Bd, Q, R)
    return K.ravel()


# 规范增益（@250Hz 基层），由 canonical 参数生成，禁止手填。
LQR_K_DT4 = synth_lqr_k(BASE_DT)
_LQR_A, _LQR_B = cartpole_continuous()
_LQR_Ad, _LQR_Bd = discretize_zoh(_LQR_A, _LQR_B, BASE_DT)
_LQR_POLES = np.linalg.eigvals(_LQR_Ad - _LQR_Bd @ LQR_K_DT4.reshape(1, 4))
LQR_POLES_DT4 = _LQR_POLES
LQR_MAX_POLE_DT4 = float(np.max(np.abs(_LQR_POLES)))
LQR_OMEGA_N_DT4 = float(-np.log(LQR_MAX_POLE_DT4) / BASE_DT)  # 等效自然频率 rad/s

# ------------------------------------------------------------
# LQI 轨迹跟踪（位置/速度参考）：状态增广 ∫(x-x_ref) dt
# ------------------------------------------------------------
def synth_lqi_k(dt: float = BASE_DT,
                q_diag=LQR_Q_DIAG, qi=LQI_QI, r_val=LQR_R):
    """增广状态 [x,θ,ẋ,θ̇, ∫(x-x_ref)] 的离散 LQR，返回 (K, Ki)。F = -K·e - Ki·∫e_x。"""
    A, B = cartpole_continuous()
    n = A.shape[0]
    # 增广：x_aug' = A_aug x_aug + B_aug F；新增维度 = 位置误差积分
    Aa = np.zeros((n + 1, n + 1))
    Aa[:n, :n] = A
    Aa[:n, n] = B.ravel() * 0.0  # x 不受 F 直接影响（纯积分器由下式）
    Aa[n, 0] = 1.0               # ẋ_integral = x
    Ba = np.vstack([B, np.zeros((1, 1))])
    # ZOH 离散化（增广连续）
    Z = np.zeros((n + 1 + 1, n + 1 + 1))
    Z[:n + 1, :n + 1] = Aa
    Z[:n + 1, n + 1:] = Ba
    Md = expm(Z * dt)
    Ada = Md[:n + 1, :n + 1]
    Bda = Md[:n + 1, n + 1:]
    Qa = np.diag(list(q_diag) + [qi])
    R = np.array([[r_val]])
    Pa = solve_discrete_are(Ada, Bda, Qa, R)
    Ka = np.linalg.inv(Bda.T @ Pa @ Bda + R) @ (Bda.T @ Pa @ Ada)
    Ka = Ka.ravel()
    return Ka[:n], Ka[n]


LQI_K_DT4, LQI_KI_DT4 = synth_lqi_k(BASE_DT)


# ------------------------------------------------------------
# 参考轨迹生成器（jerk-limited）：命令归零后冻结参考 ⇒ 原地位置保持
# ------------------------------------------------------------
class ReferenceManager:
    """速度/偏航/D0 的 jerk-limited 参考管理。命令归零后参考冻结，实现原地位置保持。"""

    def __init__(self, dt: float = RL_DT,
                 max_accel=(2.0, 4.0, 200.0), max_jerk=(8.0, 16.0, 800.0)):
        self.dt = dt
        self.max_accel = np.array(max_accel, float)   # v, w, d0
        self.max_jerk = np.array(max_jerk, float)
        self.ref = np.zeros(3)        # x_ref(pos), v_ref, w_ref -> 这里存 v,w,d0 参考
        self.ref_rate = np.zeros(3)

    def reset(self, cmd=np.zeros(3)):
        self.ref = np.array(cmd, float).copy()
        self.ref_rate = np.zeros(3)

    def update(self, cmd):
        cmd = np.array(cmd, float)
        for i in range(3):
            # 目标加速度限幅到 max_accel；加速度变化率限幅到 max_jerk
            a_target = (cmd[i] - self.ref[i]) / max(self.dt, 1e-6)
            a_target = np.clip(a_target, -self.max_accel[i], self.max_accel[i])
            da = np.clip(a_target - self.ref_rate[i],
                         -self.max_jerk[i] * self.dt, self.max_jerk[i] * self.dt)
            self.ref_rate[i] += da
            self.ref[i] += self.ref_rate[i] * self.dt
        return self.ref.copy()


# ============================================================
# 几何：对称并联五杆 IK/FK（单一真源）
#   髋面坐标系原点 = 两髋点 A-B 中点；+X 前，+Z 上，Q 在下方。
#   每腿独立 5 杆：髋 A(-AX,0)→曲柄→P1，髋 B(BX,0)→曲柄→P2，P1/P2→Q(0,-D0)。
#   方向护栏（由 verify_physics_source 强制）：∂qA/∂D0 < 0，∂qB/∂D0 > 0。
#   注：真实 IK 在 D0_MAX 时髋角可能超 ±2.0rad（见 fivebar_required_joint_range），
#   属机构/限位不一致，留待 P3 几何与关节限位对拍后修正；此处先固化真源与方向护栏。
# ============================================================
def _circle_intersect(c0, r0, c1, r1, side="upper"):
    """两圆交点。side: 'upper'(最大 z) / 'left'(最小 x) / 'right'(最大 x)。"""
    c0 = np.asarray(c0, float); c1 = np.asarray(c1, float)
    d = c1 - c0
    dist = np.linalg.norm(d)
    if dist > r0 + r1 or dist < abs(r0 - r1) or dist < 1e-9:
        return None
    a = (r0 * r0 - r1 * r1 + dist * dist) / (2 * dist)
    h2 = max(r0 * r0 - a * a, 0.0)
    h = np.sqrt(h2)
    p = c0 + d * (a / dist)
    perp = np.array([-d[1], d[0]]) / dist
    s1 = p + perp * h
    s2 = p - perp * h
    if side == "upper":
        return s1 if s1[1] >= s2[1] else s2
    if side == "left":
        return s1 if s1[0] <= s2[0] else s2
    if side == "right":
        return s1 if s1[0] >= s2[0] else s2
    if side == "lower":
        return s1 if s1[1] <= s2[1] else s2
    raise ValueError(side)


def fivebar_ik_point(qx_mm: float, d0_mm: float, side_a="left", side_b="right",
                     qzero_a=0.0, qzero_b=0.0):
    """给定输出点 ``(Qx, -D0)`` (mm) 解五杆原始髋角 (qA, qB) rad。

    分支选择：A 链取最左交点（膝外摆/肘上），B 链取最右交点，对应实际机构外摆构型。
    返回 (qA, qB, P1, P2)。qA/qB 为相对髋点的原始几何角；部署层用 qzero 标定关节零位。
    """
    qx = float(qx_mm)
    d0 = float(d0_mm)
    Q = np.array([qx, -d0])
    A = np.array([AX, 0.0])
    B = np.array([BX, 0.0])
    P1 = _circle_intersect(A, A_LEN, Q, B_LEN, side=side_a)
    P2 = _circle_intersect(B, A_LEN, Q, B_LEN, side=side_b)
    if P1 is None or P2 is None:
        return None
    qA = np.arctan2(P1[1] - A[1], P1[0] - A[0]) + qzero_a
    qB = np.arctan2(P2[1] - B[1], P2[0] - B[0]) + qzero_b
    return qA, qB, P1, P2


def fivebar_ik(d0_mm: float, side_a="left", side_b="right",
               qzero_a=0.0, qzero_b=0.0):
    """给定对称输出点 ``(0, -D0)`` (mm) 解五杆原始髋角。"""
    return fivebar_ik_point(0.0, d0_mm, side_a, side_b, qzero_a, qzero_b)


def fivebar_fk(qA: float, qB: float, qzero_a=0.0, qzero_b=0.0):
    """FK：给定髋角反解 Q 位置 (x,z) mm。

    Q 是两连杆圆（以 P1、P2 为圆心、B_LEN 为半径）的下交点（朝 -z），
    即真实足端。用于闭环校验与部署自检。
    """
    qA = float(qA) - qzero_a
    qB = float(qB) - qzero_b
    A = np.array([AX, 0.0]); B = np.array([BX, 0.0])
    P1 = A + A_LEN * np.array([np.cos(qA), np.sin(qA)])
    P2 = B + A_LEN * np.array([np.cos(qB), np.sin(qB)])
    Q = _circle_intersect(P1, B_LEN, P2, B_LEN, side="lower")
    if Q is None:
        return (P1 + P2) / 2.0
    return Q


def fivebar_fk_relative(qA_cmd: float, qB_cmd: float):
    """FK for dwell-relative actuator commands used by XML and firmware.

    ``fivebar_fk`` accepts absolute geometric crank angles.  The deployed
    command frame is relative to the D0=58 mm dwell pose, so convert through
    the canonical dwell angles before evaluating the closed-chain intersection.
    """
    dwell = fivebar_ik(D0_MIN)
    if dwell is None:
        raise RuntimeError("dwell five-bar configuration is unreachable")
    return fivebar_fk(dwell[0] - float(qA_cmd), dwell[1] - float(qB_cmd))


def fivebar_dq_dd0(d0_mm: float, eps=0.5):
    """数值 ∂(qA,qB)/∂D0，用于方向护栏测试。"""
    r0 = fivebar_ik(d0_mm - eps)
    r1 = fivebar_ik(d0_mm + eps)
    if r0 is None or r1 is None:
        return None
    return np.array([(r1[0] - r0[0]) / (2 * eps), (r1[1] - r0[1]) / (2 * eps)])


def fivebar_required_joint_range(d0_min=D0_MIN, d0_max=D0_MAX, n=64):
    """Return dwell-relative actuator command ranges over the D0 workspace."""
    qAs, qBs = [], []
    for d in np.linspace(d0_min, d0_max, n):
        qA, qB = fivebar_ik_cmd(d)
        qAs.append(qA); qBs.append(qB)
    qAs = np.array(qAs); qBs = np.array(qBs)
    return {
        "qA": (float(qAs.min()), float(qAs.max())),
        "qB": (float(qBs.min()), float(qBs.max())),
        "joint_limit": 3.3,
        "qA_over": bool(np.max(np.abs(qAs)) > 3.3),
        "qB_over": bool(np.max(np.abs(qBs)) > 3.3),
    }


def _continuous_angle_delta(angle: float, reference: float) -> float:
    """Return ``angle-reference`` on the branch continuous at ``reference``."""
    return float(np.arctan2(np.sin(angle - reference), np.cos(angle - reference)))


def fivebar_ik_cmd_xy(qx_mm: float, d0_mm: float, d0_ref: float = D0_MIN):
    """部署用关节命令: ``(Qx, D0)`` (mm) → dwell-relative ``(qA, qB)``.

    MuJoCo 的髋关节零位是驻留构型，而几何圆交点返回的是全局角。圆交点在
    ``pi`` 处会换写法而不是换构型，因此必须相对驻留角做环绕差分；直接相减会
    在约 80mm 处生成近 ``2*pi`` 的伪跳变。
    """
    d0 = float(np.clip(d0_mm, D0_MIN, D0_MAX))
    r = fivebar_ik_point(qx_mm, d0)
    r0 = fivebar_ik_point(0.0, d0_ref)
    if r is None or r0 is None:
        raise ValueError(f"five-bar target unreachable: qx={qx_mm}mm d0={d0}mm")
    # XML hinge +Y rotation decreases the X-Z geometric angle.  Dwell-relative
    # actuator commands therefore negate the continuous geometric displacement.
    return (-_continuous_angle_delta(r[0], r0[0]),
            -_continuous_angle_delta(r[1], r0[1]))


def fivebar_ik_cmd(d0_mm: float, d0_ref: float = D0_MIN):
    """部署用关节角命令: 给定对称 D0(mm) → (qA_cmd, qB_cmd) rad。

    以 d0_ref(驻留态, 默认 D0_MIN) 为关节零位, 并施加审计约定的符号:
      ∂qA/∂D0 < 0, ∂qB/∂D0 > 0 (与 XML 关节正方向 + 真源几何一致)。
    D0 越界自动夹取; 几何不可达(两圆不交)时退回最近可达(零位)。
    """
    return fivebar_ik_cmd_xy(0.0, d0_mm, d0_ref)


def fivebar_ik_table(n: int = 256):
    """D0→(qA_cmd,qB_cmd) 标定查找表 (JAX 安全, 供 MJX step 内插值)。

    返回 {"d0": ndarray(n), "qA": ndarray(n), "qB": ndarray(n)}。
    qA/qB 已施加驻留零位 + 部署符号 (见 fivebar_ik_cmd)。
    """
    d0_grid = np.linspace(D0_MIN, D0_MAX, n)
    q = np.asarray([fivebar_ik_cmd(d) for d in d0_grid])
    return {"d0": d0_grid, "qA": q[:, 0], "qB": q[:, 1]}


def fivebar_ik_grid(n_d0: int = 256, qx_limit_mm: float = 20.0, n_qx: int = 33):
    """Build the deployment IK grid for ``(Qx,D0)`` residual projection.

    The runtime only interpolates this numeric grid inside JAX.  Keeping the
    geometry solve here prevents Python/NumPy execution in a jitted environment
    step while preserving all six action dimensions.
    """
    d0 = np.linspace(D0_MIN, D0_MAX, n_d0)
    qx = np.linspace(-qx_limit_mm, qx_limit_mm, n_qx)
    qA = np.empty((n_qx, n_d0), dtype=np.float64)
    qB = np.empty((n_qx, n_d0), dtype=np.float64)
    for i, x in enumerate(qx):
        for j, depth in enumerate(d0):
            qA[i, j], qB[i, j] = fivebar_ik_cmd_xy(x, depth)
    return {"qx": qx, "d0": d0, "qA": qA, "qB": qB}


# ============================================================
# 执行器模型（单一真源，供仿真/部署共用）
# ============================================================
# DDSM315 轮毂电机：力矩 ≈ Kt·I；堵转 τ_stall @ I_stall。
DDSM_KT = TAU_WHEEL_STALL / 25.0   # N·m/A（假设 25A 堵转电流，标定后覆盖）
DDSM_I_STALL = 25.0                # A（标定后覆盖）
DDSM_BACK_EMF = (RPM_WHEEL_NOLOAD * 2 * np.pi / 60) / DDSM_I_STALL  # 近似

# ST3215 舵机：位置环 kp/kv，速率/加速度限制（标定后覆盖）
SERVO_MAX_SPEED = W_SERVO          # rad/s（空载）
SERVO_MAX_ACCEL = 40.0             # rad/s²（标定后覆盖）


def ddsm_torque_from_current(i_a: float) -> float:
    return DDSM_KT * i_a


def ddsm_current_from_torque(tau: float) -> float:
    return tau / DDSM_KT


# ============================================================
# 代码生成（单一真源 → XML / 固件头 / 协议 manifest）
#   消除 XML、Python、固件三处重复常量。
# ============================================================
def codegen_firmware_header() -> str:
    """生成 STM32 可用的 LQR/几何常量 C 头片段。"""
    geo = fivebar_required_joint_range()
    lines = [
        "/* AUTO-GENERATED by kuafu_physics.codegen_firmware_header — 禁止手改 */",
        f"#define KUAFU_MODEL_SCHEMA_VERSION \"{__import__('rl.env.contract', fromlist=['SCHEMA_VERSION']).SCHEMA_VERSION}\"",
        f"#define KUAFU_MODEL_HASH \"{model_hash()}\"",
        f"#define BASE_DT {BASE_DT:.6f}f",
        f"#define PHYS_DT {PHYS_DT:.6f}f",
        f"#define RL_DT {RL_DT:.6f}f",
        f"#define KUAFU_LQR_K0 {LQR_K_DT4[0]:.8f}f",
        f"#define KUAFU_LQR_K1 {LQR_K_DT4[1]:.8f}f",
        f"#define KUAFU_LQR_K2 {LQR_K_DT4[2]:.8f}f",
        f"#define KUAFU_LQR_K3 {LQR_K_DT4[3]:.8f}f",
        f"#define KUAFU_LQI_KI {LQI_KI_DT4:.8f}f",
        f"#define R_WHEEL_M {R_WHEEL*MM:.6f}f",
        f"#define D0_MIN_MM {D0_MIN:.1f}f",
        f"#define D0_MAX_MM {D0_MAX:.1f}f",
        f"#define AX_MM {AX:.1f}f",
        f"#define BX_MM {BX:.1f}f",
        f"#define A_LEN_MM {A_LEN:.1f}f",
        f"#define B_LEN_MM {B_LEN:.1f}f",
        f"#define WHEEL_WIDTH_MM {WHEEL_WIDTH_MM:.1f}f",
        f"#define QX_RESIDUAL_SCALE_MM {QX_RESIDUAL_SCALE:.1f}f",
        f"#define D0_RESIDUAL_SCALE_MM {D0_RESIDUAL_SCALE:.1f}f",
        f"#define KUAFU_ROLL_KP {ROLL_KP:.6f}f",
        f"#define KUAFU_ROLL_KD {ROLL_KD:.6f}f",
        f"#define KUAFU_SERVO_MAX_SPEED {SERVO_MAX_SPEED:.6f}f",
        f"#define KUAFU_OMEGA_NOLOAD {RPM_WHEEL_NOLOAD * 2 * np.pi / 60.0:.6f}f",
        f"#define D0_GATE_V_THRESH {D0_GATE_V_THRESH:.6f}f",
        f"#define D0_GATE_W_THRESH {D0_GATE_W_THRESH:.6f}f",
        f"#define D0_GATE_MAX_HIGH {D0_GATE_MAX_HIGH:.1f}f",
        f"#define DDSM_MAX_TORQUE_NM {TAU_WHEEL_STALL:.6f}f",
        f"#define TAU_WHEEL_RATED {TAU_WHEEL_RATED:.6f}f",
        f"#define KUAFU_YAW_KP {YAW_KP:.6f}f",
        f"#define KUAFU_YAW_KD {YAW_KD:.6f}f",
        f"/* joint limit feasibility: qA_over={geo['qA_over']} qB_over={geo['qB_over']} */",
        f"#define WHEEL_SPEED_SCALE {__import__('rl.env.contract', fromlist=['ProtocolFrameSpec']).ProtocolFrameSpec.WHEEL_SPEED_SCALE}",
    ]
    return "\n".join(lines)


def codegen_xml_constants() -> str:
    """生成可粘贴进 kuafu.xml 的注释块（常量来源声明）。"""
    return (
        "<!-- AUTO-GENERATED by kuafu_physics.codegen_xml_constants -->\n"
        f"<!-- schema={__import__('rl.env.contract', fromlist=['SCHEMA_VERSION']).SCHEMA_VERSION} "
        f"BASE_DT={BASE_DT} PHYS_DT={PHYS_DT} RL_DT={RL_DT} -->\n"
        f"<!-- LQR_K_DT4={np.array2string(LQR_K_DT4, precision=4)} R_wheel={R_WHEEL*MM} -->\n"
        f"<!-- AX={AX} BX={BX} A_LEN={A_LEN} B_LEN={B_LEN} D0=[{D0_MIN},{D0_MAX}] -->"
    )


def model_hash() -> str:
    """Hash every source value that can change a trained/deployed behavior."""
    from rl.env.contract import ProtocolFrameSpec, SCHEMA_VERSION

    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl", "kuafu.xml")
    with open(xml_path, "rb") as source:
        xml_hash = hashlib.sha256(source.read()).hexdigest()
    values = [
        G, MC, MP, LP, AX, BX, A_LEN, B_LEN, XQ, R_WHEEL, WHEEL_WIDTH_MM,
        D0_MIN, D0_MAX, TAU_WHEEL_RATED, TAU_WHEEL_STALL, RPM_WHEEL_RATED,
        RPM_WHEEL_NOLOAD, TAU_CONT, TAU_STALL, SERVO_KP, SERVO_KV,
        SERVO_MAX_SPEED, SERVO_MAX_ACCEL, QX_RESIDUAL_SCALE, D0_RESIDUAL_SCALE,
        D0_GATE_V_THRESH, D0_GATE_W_THRESH, D0_GATE_MAX_HIGH, IK_GENERATOR_VERSION,
        DR_MASS, DR_COM, DR_INERTIA, DR_FRICTION, DR_WHEEL_R, DR_TORQUE_CONST,
        DR_SERVO_PD, DR_DEADBAND, DR_DELAY_ACT, DR_DELAY_SENSE,
        YAW_KP, YAW_KD, ROLL_KP, ROLL_KD, PHYS_DT, BASE_DT, RL_DT,
        *LQR_Q_DIAG, LQI_QI, LQR_R, *LQR_K_DT4.tolist(), LQI_KI_DT4,
        ProtocolFrameSpec.version, ProtocolFrameSpec.WHEEL_SPEED_SCALE,
        SCHEMA_VERSION, xml_hash,
    ]
    payload = "|".join(map(str, values))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
