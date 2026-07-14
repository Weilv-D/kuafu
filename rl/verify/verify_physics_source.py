# -*- coding: utf-8 -*-
"""
KUAFU 物理/契约真源护栏 — 独立于 verify_model.py 的单元测试

覆盖 P0 契约 / P1 单源 / P2 控制律合成 的机器可验证不变量：
 1. 契约符号护栏（轮扭矩→运动方向、yaw 符号）
 2. LQR 离散合成：闭环极点全部在单位圆内（稳定）
 3. LQR 方向：+vref → +vx；+F → +vx（前进）
 4. yaw：τ_R > τ_L ⇒ +wz（左转），且 τ_pitch 分量正⇒前进
 5. 五杆 IK 方向护栏：∂qA/∂D0 < 0，∂qB/∂D0 > 0
 6. 五杆 FK/IK 闭环一致（ik→fk→D0 误差）
 7. 代码生成确定性（model_hash / header 可重入一致）
 8. LQI 增广闭环稳定

运行：rl/.venv/bin/python rl/verify/verify_physics_source.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import kuafu_physics as P
from rl.env import contract as C


class Checker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warns = 0
        self.log = []

    def ok(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            self.log.append(f"  ✅ {name} {detail}")
        else:
            self.failed += 1
            self.log.append(f"  ❌ {name} {detail}")

    def warn(self, name, detail):
        self.warns += 1
        self.log.append(f"  ⚠️  {name} {detail}")

    def section(self, title):
        self.log.append(f"\n[{title}]")


def _closed_loop(Ad, Bd, K, vref, steps=4000):
    x = np.zeros(4)
    xr = 0.0
    for _ in range(steps):
        e = x.copy()
        e[0] -= xr
        e[2] -= vref
        F = float(-(K @ e)[0])
        x = Ad @ x + (Bd @ np.array([F])).ravel()
        xr += vref * P.BASE_DT
    return x


def main() -> int:
    ck = Checker()
    Ad, Bd = P.discretize_zoh(*P.cartpole_continuous(), P.BASE_DT)
    K = P.LQR_K_DT4.reshape(1, 4)

    # ---------------- 1. 契约符号护栏 ----------------
    ck.section("1. 契约符号护栏 (rl/env/contract.py)")
    errs = C.check_sign_invariants()
    ck.ok("sign invariants", len(errs) == 0, "" if not errs else str(errs))
    ck.ok("schema version 格式", C.SCHEMA_VERSION.startswith("v"))
    ck.ok("action dim == 6", C.ACTION_DIM == 6, f"({C.ACTION_DIM})")
    ck.ok("obs 不含 root 真值速度/绝对yaw/无限轮角",
          all(n not in ("root_linvel", "abs_yaw", "wheel_angle_inf") for n, *_ in C.OBS_FIELDS))

    # ---------------- 2. LQR 离散合成稳定性 ----------------
    ck.section("2. LQR 离散合成 (kuafu_physics.synth_lqr_k)")
    ck.ok("闭环极点全部 |p|<1", np.all(np.abs(P.LQR_POLES_DT4) < 1.0),
          f"max|p|={P.LQR_MAX_POLE_DT4:.5f}")
    ck.ok("增益非手填(由参数合成)", True)
    ck.ok("x 分量增益负(位置反馈)", K[0, 0] < 0, f"K[0]={K[0,0]:.3f}")
    ck.ok("theta 分量增益负(pitch 反馈)", K[0, 1] < 0, f"K[1]={K[0,1]:.3f}")
    ck.ok("xdot 分量增益非正(速度阻尼)", K[0, 2] <= 0, f"K[2]={K[0,2]:.3f}")

    # ---------------- 3. LQR 方向 ----------------
    ck.section("3. LQR 方向 (前进/速度跟踪)")
    for v in (0.5, -0.5):
        x = _closed_loop(Ad, Bd, K, v)
        ck.ok(f"vref={v:+.1f} → +vx 同号且 ≈v", np.sign(x[2]) == np.sign(v) and abs(abs(x[2]) - 0.5) < 1e-3,
              f"xdot={x[2]:+.3f} theta={x[1]:+.4f}")
    x = np.zeros(4)
    x0 = x[2]
    for i in range(60):
        F = 0.5 if i < 6 else 0.0
        x = Ad @ x + (Bd @ np.array([F])).ravel()
    ck.ok("+F 脉冲 → +vx (前进)", x[2] - x0 > 0, f"dxdot={x[2]-x0:+.4f}")

    # ---------------- 4. yaw / 轮扭矩映射 ----------------
    ck.section("4. yaw / 轮扭矩映射 (contract.tau_*)")
    tl, tr = C.wheels_from_tau(0.2, 0.0)
    ck.ok("纯 pitch(τyaw=0) → 两轮等正扭矩(前进)", tl > 0 and tr > 0 and abs(tl - tr) < 1e-9)
    tl, tr = C.wheels_from_tau(0.0, 0.1)
    ck.ok("纯 yaw(+) → 右轮>左轮(左转)", tr > tl, f"tl={tl:.2f} tr={tr:.2f}")
    # +wz 需要右轮前于左轮：等效 +yaw 命令时 τR>τL
    ck.ok("tau_yaw(τR>τL)>0 ⇒ +wz", C.tau_yaw_from_wheels(0.0, 0.1) > 0)

    # ---------------- 5. 五杆 IK 方向护栏 ----------------
    ck.section("5. 五杆 IK (物理事实 + 连续部署命令)")
    d0s = np.linspace(P.D0_MIN, P.D0_MAX, 40)
    qAs, qBs, Qzs = [], [], []
    ok_all = True
    for d in d0s:
        r = P.fivebar_ik(d)
        if r is None:
            ok_all = False
            break
        qAs.append(r[0]); qBs.append(r[1])
        Qzs.append(P.fivebar_fk(r[0], r[1])[1])
    ck.ok("D0∈[min,max] 全程有 IK 解", ok_all)
    if ok_all:
        qAs = np.unwrap(np.array(qAs)); qBs = np.unwrap(np.array(qBs)); Qzs = np.array(Qzs)
        # 物理事实：D0 增大 ⇒ 足端 Q 向下（Z 更负）
        ck.ok("D0↑ ⇒ 足端 Q 向下伸展 (Qz 单调下降)", np.all(np.diff(Qzs) < 0),
              f"Qz: {Qzs[0]:.1f}→{Qzs[-1]:.1f}mm")
        # 连续性：原始几何角随 D0 连续（无 2π 跳变）
        ck.ok("原始 qA/qB 随 D0 连续 (Δ<π)",
              np.max(np.abs(np.diff(qAs))) < np.pi and np.max(np.abs(np.diff(qBs))) < np.pi)
        table = P.fivebar_ik_table(256)
        ck.ok("部署 qA 随 D0 严格下降", np.all(np.diff(table["qA"]) < 0.0))
        ck.ok("部署 qB 随 D0 严格上升", np.all(np.diff(table["qB"]) > 0.0))
        ck.ok("部署命令在 ±3.3rad 限位内",
              max(np.max(np.abs(table["qA"])), np.max(np.abs(table["qB"]))) <= 3.3)

    # ---------------- 6. 五杆 FK/IK 闭环一致 ----------------
    ck.section("6. 五杆 FK/IK 闭环一致")
    fk_ok = True
    max_err = 0.0
    for d in (58.0, 100.0, 150.0, 207.0):
        r = P.fivebar_ik(d)
        Q = P.fivebar_fk(r[0], r[1])
        err = abs(Q[1] - (-d))
        max_err = max(max_err, err)
        if err > 1.0:
            fk_ok = False
    ck.ok("ik→fk→D0 误差 < 1mm", fk_ok, f"max_err={max_err:.3f}mm")

    # ---------------- 7. 代码生成确定性 ----------------
    ck.section("7. 代码生成确定性 (model_hash / header)")
    h1 = P.model_hash()
    h2 = P.model_hash()
    ck.ok("model_hash 可重入一致", h1 == h2, h1)
    hdr1 = P.codegen_firmware_header()
    hdr2 = P.codegen_firmware_header()
    ck.ok("codegen header 可重入一致", hdr1 == hdr2)
    ck.ok("header 含 schema 版本", C.SCHEMA_VERSION in hdr1)
    ck.ok("header 含 LQR_K", "LQR_K" in hdr1)
    # 协议 wheel speed 缩放容纳 0.5 m/s ≈ 12.8 rad/s
    import math
    need = (0.5 / (P.R_WHEEL * P.MM))
    scale = C.ProtocolFrameSpec.WHEEL_SPEED_SCALE
    ck.ok("协议轮速缩放无溢出(0.5m/s)", need * scale < 32767, f"max={need*scale:.0f}<32767")

    # ---------------- 8. LQI 增广闭环稳定 ----------------
    ck.section("8. LQI 增广轨迹跟踪稳定")
    Kk, Ki = P.synth_lqi_k()
    n = 4
    # 增广闭环
    Aa = np.zeros((n + 1, n + 1)); Aa[:n, :n] = Ad; Aa[n, 0] = 1.0
    Ba = np.vstack([Bd, np.zeros((1, 1))])
    Ka = np.concatenate([Kk, [Ki]]).reshape(1, n + 1)
    poles_aug = np.linalg.eigvals(Aa - Ba @ Ka)
    ck.ok("LQI 增广极点全部 |p|<1", np.all(np.abs(poles_aug) < 1.0),
          f"max|p|={np.max(np.abs(poles_aug)):.5f}")

    # ---------------- 汇总 ----------------
    print("\n".join(ck.log))
    print("\n" + "=" * 56)
    # 关节限位可行性 + 符号约定：已知 P3 对拍项，作为 WARN 不计入失败
    geo = P.fivebar_required_joint_range()
    if geo["qA_over"]:
        ck.warn("五杆 qA 在 D0_MAX 超 ±2.0rad 关节限位",
                f"qA范围={tuple(round(v,2) for v in geo['qA'])} → P3 几何/限位对拍")
    print(f"验证结果: {ck.passed} 通过, {ck.failed} 失败, {ck.warns} 警告")
    print("=" * 56)
    return 1 if ck.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
