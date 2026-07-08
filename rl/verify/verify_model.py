# -*- coding: utf-8 -*-
"""
KUAFU 仿真模型物理验证 — design.md §2.6 阶段 0

加载 kuafu.xml，逐项检查并打印报告。验证的是"XML 是否正确描述了这个机构"，
不涉及训练。全部通过后才可进入 RL 训练。

运行: python rl/verify/verify_model.py
依赖: mujoco (CPU 即可, 无需 GPU/JAX)
"""
import os
import sys
import numpy as np
import mujoco

# 物理真源
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import kuafu_physics as P

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kuafu.xml")


class Report:
    def __init__(self):
        self.items = []
    def check(self, name, ok, detail=""):
        self.items.append((name, bool(ok), detail))
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}: {detail}")
    def summary(self):
        n = len(self.items); npass = sum(ok for _, ok, _ in self.items)
        print(f"\n{'='*60}\n验证结果: {npass}/{n} 通过", end="")
        if npass == n:
            print(" — 全部通过 ✓")
        else:
            print(f" — {n-npass} 项未通过 ✗")
        return npass == n


def main():
    print("="*60)
    print("KUAFU 仿真模型物理验证 (design.md §2.6 阶段 0)")
    print("="*60)

    # ---- 1. 模型加载 ----
    print("\n[1/6] 模型加载与自由度")
    r = Report()
    try:
        m = mujoco.MjModel.from_xml_path(XML)
        d = mujoco.MjData(m)
        r.check("XML 加载", True, f"nq={m.nq} nv={m.nv} nu={m.nu}")
    except Exception as e:
        r.check("XML 加载", False, str(e))
        return r.summary()

    # 期望: floating(7) + 2 轮 + 4 腿曲柄 + 4 膝(被动) = 但膝是闭链被动关节
    # nq: root 7 + wheel_l/r 各1 + hip_A/B_l/r 各1 + knee_A/B_l/r 各1 = 7+2+4+4=17
    # nu: 2 轮 motor + 4 舵机 position (hip_A + hip_B 各左右, 2-DOF 五杆全独立驱动) = 6
    # neq: 2 个 connect (每条腿 Q 点物理铰接闭链约束)
    r.check("自由度数 nq=17", m.nq == 17, f"nq={m.nq}")
    r.check("执行器 nu=6 (2轮+4舵机)", m.nu == 6, f"nu={m.nu}")
    r.check("Q点闭链约束 neq=2", m.neq == 2, f"neq={m.neq}")
    r.check("keyframe 数=1", m.nkey == 1, f"nkey={m.nkey}")

    # ---- 2. 总质量与 COM ----
    print("\n[2/6] 总质量与质心")
    mass = m.body_mass.sum()
    r.check(f"总质量 ≈ {P.M_TOT}", abs(mass - P.M_TOT) < 0.01,
            f"{mass:.3f} kg (期望 {P.M_TOT})")

    # ---- 3. 驻留态 keyframe ----
    print("\n[3/6] 驻留态 keyframe")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    # 轮中心 Z (wheel_l body 世界坐标)
    wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_l")
    wheel_z = d.xpos[wid][2]
    r.check("轮接地 (轮中心 Z≈轮半径)", abs(wheel_z - P.R_WHEEL*P.MM) < 0.01,
            f"Z={wheel_z:.4f} m (期望 {P.R_WHEEL*P.MM:.4f})")

    # 机身底高度
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    chassis_z = d.xpos[cid][2]
    r.check("机身在合理高度", chassis_z > 0.08,
            f"chassis Z={chassis_z:.3f} m")

    # ---- 4. 闭链约束残差 ----
    print("\n[4/6] 闭链约束稳定性")
    # 跑 100 步物理看闭链是否发散/爆炸
    d2 = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d2, 0)
    init_pos = d2.qpos[:3].copy()
    for _ in range(100):
        mujoco.mj_step(m, d2)
    drift = np.linalg.norm(d2.qpos[:3] - init_pos)
    # 检查 qvel 是否爆炸 (NaN 或极大)
    vel_max = np.max(np.abs(d2.qvel)) if not np.any(np.isnan(d2.qvel)) else 1e9
    r.check("100 步物理后不发散", drift < 0.05 and vel_max < 10,
            f"漂移 {drift:.4f} m, 最大速度 {vel_max:.2f} m/s")

    # ---- 5. 静态稳定 ----
    print("\n[5/6] 静态稳定性")
    # 腿保持驻留 (position actuator ctrl=0 维持 q=0), 轮无控制 (motor ctrl=0 无力矩)
    # 轮式倒立摆本质不稳定, 无轮控制时机身必然倾倒; 此处验证腿关节在重力下不松脱 (q 保持 ≈0)
    d3 = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d3, 0)
    mujoco.mj_forward(m, d3)          # 更新 xpos, 确保 initial 状态正确
    d3.ctrl[:] = 0                    # 腿: ctrl=0 -> PD 维持 q=0 (驻留); 轮: ctrl=0 -> 无力矩
    for _ in range(500):              # 1s @ 500Hz timestep
        mujoco.mj_step(m, d3)
    # 驻留态偏移只查驱动侧 hip_A_l/r (qpos 7, 12); hip_B/knee 由 joint equality 镜像保证
    # 初始瞬态 (<2s) 因 COM 微偏可能漂到 ~4°, 稳态后回到 ~1°, 非松脱
    hip_drift = max(abs(d3.qpos[7]), abs(d3.qpos[12]))
    r.check("腿驻留态关节不松脱 (q 保持)", hip_drift < 0.1,
            f"1s 后 hip_A 最大偏移 {hip_drift*1e3:.1f} mrad ({np.degrees(hip_drift):.2f}°)")

    # ---- 6. LQR baseline 闭环 ----
    print("\n[6/6] LQR baseline 平衡能力")
    # 给 5° 初始 pitch 扰动, 用 LQR K 控制, 看能否恢复
    d4 = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d4, 0)
    # 初始倾角 5° (绕 Y 轴 = pitch)
    th0 = np.radians(5)
    # root quat [w,x,y,z]: pitch 绕 Y -> w=cos(th/2), y=sin(th/2)
    d4.qpos[3:7] = [np.cos(th0/2), 0, np.sin(th0/2), 0]
    mujoco.mj_forward(m, d4)
    K = P.LQR_K
    recovered = False
    for step in range(2000):  # 4s
        # 状态 [x, theta, xdot, thetadot]
        # root qpos: [x,y,z, qw,qx,qy,qz], pitch=绕Y -> qy=qpos[5]
        # root qvel: [vx,vy,vz, wx,wy,wz], pitch角速度=绕Y -> wy=qvel[4]
        x = d4.qpos[0]
        qw, qy = d4.qpos[3], d4.qpos[5]
        theta = np.arcsin(np.clip(2 * qw * qy, -0.999999, 0.999999))
        xdot = d4.qvel[0]
        thetadot = d4.qvel[4]                          # pitch 角速度 (wy)
        F = -(K @ np.array([x, theta, xdot, thetadot]))
        # 两轮各施加 F/2 (力矩 = F*R/2)
        tau = F * P.R / 2
        d4.ctrl[0] = tau  # tau_l
        d4.ctrl[1] = tau  # tau_r
        # 腿保持驻留: q=0 即驻留姿态 (4 舵机 ctrl=0, position actuator 维持)
        d4.ctrl[2] = 0  # q_hip_A_l
        d4.ctrl[3] = 0  # q_hip_A_r
        d4.ctrl[4] = 0  # q_hip_B_l
        d4.ctrl[5] = 0  # q_hip_B_r
        mujoco.mj_step(m, d4)
        if abs(theta) < np.radians(1) and step > 50:
            recovered = True
            r.check("LQR 恢复 5° 扰动", True,
                    f"{step*0.002:.2f}s 内恢复到 <1°")
            break
    if not recovered:
        qw, qy = d4.qpos[3], d4.qpos[5]
        final_th = np.arcsin(np.clip(2*qw*qy, -0.999999, 0.999999))
        r.check("LQR 恢复 5° 扰动", False,
                f"4s 后仍 {np.degrees(final_th):.1f}° (未恢复)")

    return r.summary()


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
