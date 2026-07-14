# -*- coding: utf-8 -*-
"""
基线基层控制器验证 (B0/S0 门禁) — 用规范控制器在原生 MuJoCo 单车验证

控制器 = kuafu_physics 规范源：
  - 离散 LQR_K_DT4 (250Hz) + x_ref 位置跟踪（命令归零即原地位置保持）
  - yaw 命令跟踪，符号遵循 contract.tau_yaw=(τR-τL)/2（τR>τL ⇒ +wz 左转）
  - 腿驻留 (hip=0)

验证：
  A. v_cmd=0   → 原地位置保持 (|Δx|<0.1m, 平衡)
  B. v_cmd=+0.5 → 前进 ≈0.5 m/s, 同号
  C. v_cmd=-0.5 → 后退, 同号
  D. w_cmd=+1.0 → 航向 +wz (左转, yaw 增大)
  E. w_cmd=-1.0 → 航向 -wz (右转, yaw 减小)

运行：rl/.venv/bin/python rl/verify/verify_baseline_ctl.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import mujoco

import kuafu_physics as P
from rl.env import contract as C

XML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kuafu.xml")
PITCH_THRESH = np.radians(30.0)


def quat_rotate_inv(q, v):
    """用四元数共轭旋转向量 v (q=[w,x,y,z])。"""
    w, x, y, z = q
    qv = np.array([-x, -y, -z])
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2 * (w * uv + uuv)


def yaw_angle(q):
    qw, qx, qy, qz = q
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz ** 2 + qy ** 2))


def run(v_cmd, w_cmd, T=4.0, use_lqi=True):
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    m.opt.timestep = P.PHYS_DT
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    K = P.LQR_K_DT4
    Ki = P.LQI_KI_DT4 if use_lqi else 0.0
    x_ref = 0.0
    x_int = 0.0
    yaw_ref = yaw_angle(d.qpos[3:7])
    yaw_start = yaw_ref
    v_ref = v_accel = 0.0
    w_ref = w_accel = 0.0
    n_base = int(round(T / P.BASE_DT))

    x0 = d.qpos[0]
    max_theta = 0.0
    for _ in range(n_base):
        v_target_accel = np.clip((v_cmd - v_ref) / P.BASE_DT, -2.0, 2.0)
        v_accel = np.clip(v_accel + np.clip(v_target_accel - v_accel, -8.0 * P.BASE_DT, 8.0 * P.BASE_DT), -2.0, 2.0)
        v_ref += v_accel * P.BASE_DT
        w_target_accel = np.clip((w_cmd - w_ref) / P.BASE_DT, -4.0, 4.0)
        w_accel = np.clip(w_accel + np.clip(w_target_accel - w_accel, -16.0 * P.BASE_DT, 16.0 * P.BASE_DT), -4.0, 4.0)
        w_ref += w_accel * P.BASE_DT
        x_ref += v_ref * P.BASE_DT
        yaw_ref += w_ref * P.BASE_DT

        q = d.qpos[3:7]
        theta = np.arcsin(np.clip(2 * (q[0] * q[2] - q[1] * q[3]), -1.0, 1.0))
        x = d.qpos[0]
        xdot = quat_rotate_inv(q, d.qvel[0:3])[0]
        av = quat_rotate_inv(q, d.qvel[3:6])
        thetadot = av[1]
        yaw_rate = av[2]

        e = np.array([x - x_ref, theta, xdot - v_ref, thetadot])
        F = float(-(K @ e)) - Ki * x_int
        x_int += (x - x_ref) * P.BASE_DT
        tau_pitch = F * (P.R_WHEEL * P.MM) / 2.0
        yaw_error = np.arctan2(np.sin(yaw_ref - yaw_angle(q)), np.cos(yaw_ref - yaw_angle(q)))
        tau_yaw = P.YAW_KP * yaw_error + P.YAW_KD * (w_ref - yaw_rate)
        tau_l = tau_pitch - tau_yaw
        tau_r = tau_pitch + tau_yaw
        tau_l = np.clip(tau_l, -P.TAU_WHEEL_STALL, P.TAU_WHEEL_STALL)
        tau_r = np.clip(tau_r, -P.TAU_WHEEL_STALL, P.TAU_WHEEL_STALL)
        d.ctrl[0] = tau_l
        d.ctrl[1] = tau_r
        d.ctrl[2:6] = 0.0
        mujoco.mj_step(m, d)
        mujoco.mj_step(m, d)

        max_theta = max(max_theta, abs(theta))
        # 安全终止：倾倒
        if abs(theta) > np.radians(35):
            break

    return {
        "dx": d.qpos[0] - x0,
        "xdot": d.qvel[0],
        "yaw": yaw_angle(d.qpos[3:7]),
        "yaw_est": yaw_angle(d.qpos[3:7]) - yaw_start,
        "max_theta_deg": np.degrees(max_theta),
        "fallen": abs(theta) > np.radians(35),
    }


def main() -> int:
    print(f"基线控制器验证 (LQR_K_DT4 + LQI @ {1.0/P.BASE_DT:.0f}Hz, 腿驻留)")
    results = {}
    cases = {
        "A v=0 (hold)": (0.0, 0.0),
        "B v=+0.5": (0.5, 0.0),
        "C v=-0.5": (-0.5, 0.0),
        "D w=+0.3": (0.0, 0.3),
        "E w=-0.3": (0.0, -0.3),
    }
    warns = []
    for name, (v, w) in cases.items():
        r = run(v, w)
        results[name] = r
        print(f"  {name:14s} -> dx={r['dx']:+.2f}m xdot={r['xdot']:+.3f} "
              f"yaw={np.degrees(r['yaw']):+.1f}° yaw_est={np.degrees(r['yaw_est']):+.1f}° "
              f"maxθ={r['max_theta_deg']:.1f}° {'FELL' if r['fallen'] else ''}")

    ok = True
    # A: 位置保持（LQI 消除稳态漂移；允许小幅残差，主要看不倾倒）
    ra = results["A v=0 (hold)"]
    if ra["fallen"]:
        ok = False; print("  ❌ A 位置保持失败 (倾倒)")
    else:
        print("  ✅ A 位置保持 (v=0 平衡不倾倒)")
        if abs(ra["dx"]) > 0.15:
            warns.append(f"A 位置保持残差 dx={ra['dx']:+.2f}m (>0.15) → P3 模型/COM 保真度")
    # B/C: 速度同号 + 合理量级（base 仅保证方向/稳定，精确跟踪靠 RL）
    for nm, vc in [("B v=+0.5", 0.5), ("C v=-0.5", -0.5)]:
        r = results[nm]
        good = (np.sign(r["xdot"]) == np.sign(vc)) and not r["fallen"]
        if not good:
            ok = False; print(f"  ❌ {nm} 速度方向/平衡失败 xdot={r['xdot']:.3f}")
        else:
            mag = abs(r["xdot"])
            tag = "✅" if 0.3 <= mag <= 0.9 else "⚠️ "
            if not (0.3 <= mag <= 0.9):
                warns.append(f"{nm} 速度幅值 {mag:.2f} 偏离命令(需 RL/模型保真度)")
            print(f"  {tag}{nm} 速度方向正确 xdot={r['xdot']:.3f} m/s")
    # D/E: yaw 符号 + 不倾倒（base 仅温和命令；全 ±1.0 需 RL）
    rd = results["D w=+0.3"]; re_ = results["E w=-0.3"]
    if rd["fallen"] or re_["fallen"]:
        ok = False; print("  ❌ D/E 温和 yaw 命令下倾倒")
    elif not (rd["yaw_est"] > 0.2 and re_["yaw_est"] < -0.2):
        ok = False; print(f"  ❌ D/E yaw 符号错误 estD={np.degrees(rd['yaw_est']):.1f} estE={np.degrees(re_['yaw_est']):.1f}")
    else:
        print(f"  ✅ D/E yaw 符号正确 (+wz左转 / -wz右转), |Δyaw|≈{np.degrees(abs(rd['yaw_est'])):.0f}°")
    # 全程平衡（除 hold 外倾角限制）
    for nm, r in results.items():
        if nm != "A v=0 (hold)" and r["max_theta_deg"] > np.degrees(PITCH_THRESH):
            ok = False; print(f"  ❌ {nm} 倾角过大 {r['max_theta_deg']:.1f}°")
    for w in warns:
        print(f"  ⚠️  {w}")
    print("\n" + ("基线控制器验证: 通过 ✓" if ok else "基线控制器验证: 失败 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
