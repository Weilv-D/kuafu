# -*- coding: utf-8 -*-
"""
KUAFU 策略评估 — headless deterministic / DR / 命令扫描

在原生 MuJoCo (CPU 单环境, 与训练同 .xml) 中加载 policy, 关探索噪声, 记录全套
稳定性指标。obs/action 逻辑与训练环境 (kuafu_mjx_env.py) 及 playback.py 逐维一致。

三种模式:
  --mode deterministic  关噪声/DR, 固定命令 (v=0, ω=0, d0=dwell), 长 episode
  --mode dr             关噪声, 开 DR (初始姿态扰动), 多 episode 取统计
  --mode cmd_sweep      关噪声/DR, 扫描 v∈[-0.5,0.5], 每 cmd 一个 episode

运行:
  rl/.venv/bin/python rl/verify/eval_policy.py \
      --ckpt rl/checkpoints/garlic/teacher/model_3999.pt
  rl/.venv/bin/python rl/verify/eval_policy.py --ckpt ... --mode dr --episodes 20
  rl/.venv/bin/python rl/verify/eval_policy.py --ckpt ... --mode cmd_sweep
"""
import os
import sys
import argparse

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

import numpy as np
import mujoco

import kuafu_physics as P
from rl.env.kuafu_env import OBS_DIM_BASE, OBS_DIM, ACTION_DIM  # 37 / 148 / 6

CTRL_DT = 0.02    # 50 Hz
N_SUBSTEPS = 10   # 500 Hz 物理
PITCH_THRESH = np.radians(30)   # 与训练 _is_fallen 一致
ROLL_THRESH = np.radians(30)
OMEGA_NOLOAD = P.RPM_WHEEL_NOLOAD * 2 * np.pi / 60  # DDSM315 空载角速度 rad/s

# RMA: 训练时 teacher actor 以静态特权 latent z(9) 为条件 (Kumar 2021)。
# 仿真评估在标称平地下进行, z 取标称值:
# [friction, mass_scale, com_bias(x,y,z), inertia_scale, torque_scale, deadband, delay_steps]
NOMINAL_STATIC = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)


def _limit_wheel_torque(tau, omega):
    """DDSM back-EMF 限幅: τ_avail = τ_stall × (1 - |ω|/ω_noload). 与训练环境一致."""
    tau_avail = P.TAU_WHEEL_STALL * (1.0 - np.abs(omega) / OMEGA_NOLOAD)
    tau_avail = np.clip(tau_avail, 0.0, P.TAU_WHEEL_STALL)
    return np.clip(tau, -tau_avail, tau_avail)


# ---- obs / action: 与 playback.py / 训练 _base_observation 逐维对齐 ----
def rotate_vector_by_quaternion_conj(q, v):
    w, x, y, z = q[0], q[1], q[2], q[3]
    q_xyz = np.array([-x, -y, -z])
    uv = np.cross(q_xyz, v)
    uuv = np.cross(q_xyz, uv)
    return v + 2 * (w * uv + uuv)


def _build_obs(data, last_action, command, step, model=None):
    """37 维 base obs (含接触标志). command=[v_cmd, w_cmd, d0_cmd] 可注入.

    model: MjModel (原生 MuJoCo), 用于查 wheel geom id 算接触标志。
    若 None 则接触标志取 1.0 (假设接地, 平地评估安全默认)。
    """
    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qx * qz), -0.999999, 0.999999))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz**2 + qy**2))
    attitude = np.array([roll, pitch, yaw])
    q = np.array([qw, qx, qy, qz])
    ang_vel = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
    wheel_state = np.array([data.qpos[9], data.qpos[14], data.qvel[8], data.qvel[13]])
    hip_state = np.array([
        data.qpos[7], data.qpos[12], data.qpos[10], data.qpos[15],
        data.qvel[6], data.qvel[11], data.qvel[9], data.qvel[14]])
    wheel_torque = np.array([data.actuator_force[0], data.actuator_force[1]])
    hip_torque = np.array([
        data.actuator_force[2], data.actuator_force[3],
        data.actuator_force[4], data.actuator_force[5]])
    phase = step / 1000.0  # 对应 EPISODE_LENGTH = 1000
    phase_clock = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])

    # 接触标志 (2): 左右轮是否接地
    if model is not None:
        wl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wheel_l_geom")
        wr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wheel_r_geom")
        cl = 0.0
        cr = 0.0
        for i in range(data.ncon):
            g1, g2 = data.contact[i].geom1, data.contact[i].geom2
            if g1 == wl or g2 == wl:
                cl = 1.0
            if g1 == wr or g2 == wr:
                cr = 1.0
        contact = np.array([cl, cr], dtype=np.float32)
    else:
        contact = np.array([1.0, 1.0], dtype=np.float32)  # 平地默认接地

    return np.concatenate([attitude, ang_vel, wheel_state, hip_state,
                           wheel_torque, hip_torque, last_action, command,
                           phase_clock, contact])


def _apply_action(data, action, command=None):
    """基层(pitch LQR + yaw 差速 + roll 调平 PD) + RL 残差 → ctrl.

    与训练 kuafu_mjx_env.step 控制律一致:
    - 轮: τ_L = τ_pitch + τ_diff + Δτ_L,  τ_R = τ_pitch - τ_diff + Δτ_R
    - 腿: 基层 D0 对称 + roll PD 左右差 + RL 位置残差
    command: [v_cmd, w_cmd, d0_cmd], 用于 yaw 跟踪 w_cmd + D0 目标。None→[0,0,dwell]。
    """
    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    q = np.array([qw, qx, qy, qz])
    lin_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[0:3])
    xdot = lin_vel_local[0]
    ang_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
    thetadot = ang_vel_local[1]
    yaw_rate = ang_vel_local[2]
    omega_x = ang_vel_local[0]
    pitch = np.arcsin(np.clip(2 * (qw * qy - qx * qz), -0.999999, 0.999999))
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

    # 基层 pitch LQR
    F = -(P.LQR_K @ np.array([0.0, pitch, xdot, thetadot]))
    tau_pitch = F * P.R / 2.0

    # 基层 yaw 条件阻尼 (大 ωz 关闭防 pitch 耦合; 命令跟踪交 RL)
    if abs(yaw_rate) < P.YAW_DAMP_THRESH:
        tau_diff = np.clip(-P.YAW_KD * yaw_rate, -P.TAU_WHEEL_RATED, P.TAU_WHEEL_RATED)
    else:
        tau_diff = 0.0

    # DDSM back-EMF 限幅 (与训练 _limit_wheel_torque 一致)
    data.ctrl[0] = _limit_wheel_torque(
        tau_pitch + tau_diff + action[0] * P.TAU_WHEEL_RATED, data.qvel[8])
    data.ctrl[1] = _limit_wheel_torque(
        tau_pitch - tau_diff + action[1] * P.TAU_WHEEL_RATED, data.qvel[13])

    # 基层 D0 对称 + roll PD 左右差 + RL 残差
    d0_cmd = command[2] if command is not None else P.D0_MIN
    d0_norm = (d0_cmd - P.D0_MIN) / (P.D0_MAX - P.D0_MIN)
    hip_A_base = -d0_norm * P.HIP_STROKE
    hip_B_base = d0_norm * P.HIP_STROKE
    d_d0_mm = -(P.ROLL_KP * roll + P.ROLL_KD * omega_x)
    d_d0_rad = d_d0_mm / (P.D0_MAX - P.D0_MIN) * 2 * P.HIP_STROKE
    data.ctrl[2] = hip_A_base + d_d0_rad / 2.0 + action[2] * P.HIP_STROKE
    data.ctrl[3] = hip_A_base - d_d0_rad / 2.0 + action[3] * P.HIP_STROKE
    data.ctrl[4] = hip_B_base + d_d0_rad / 2.0 + action[4] * P.HIP_STROKE
    data.ctrl[5] = hip_B_base - d_d0_rad / 2.0 + action[5] * P.HIP_STROKE


def _get_pitch_roll(data):
    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    pitch = np.arcsin(np.clip(2 * (qw * qy - qx * qz), -0.999999, 0.999999))
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    return pitch, roll


def _is_fallen(data):
    pitch, roll = _get_pitch_roll(data)
    return abs(pitch) > PITCH_THRESH or abs(roll) > ROLL_THRESH


def _site_gap(data, model, suffix):
    sa = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"Q_A_{suffix}")]
    sb = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"Q_B_{suffix}")]
    return np.linalg.norm(sa - sb)


def run_episode(model, data, teacher, command, max_steps, rng=None, dr=False,
                latency=0):
    """跑一个 episode, 返回指标 dict. fallen 则提前终止.

    latency: 观测/动作延迟步数 (默认 0). 复现训练侧 latency randomization
    (kuafu_mjx_env 的 obs_delay_buffer / action_buffer), 用于检验部署鲁棒性。
    与训练一致, 同一 latency 同时作用于 obs 与 action。
    """
    import torch
    mujoco.mj_resetDataKeyframe(model, data, 0)
    if dr and rng is not None:
        # 初始 pitch/roll 扰动 ±2°
        th = rng.uniform(-np.radians(2), np.radians(2))
        ph = rng.uniform(-np.radians(2), np.radians(2))
        qp = np.array([np.cos(th/2), 0, np.sin(th/2), 0])
        qr = np.array([np.cos(ph/2), np.sin(ph/2), 0, 0])
        w1, x1, y1, z1 = qr; w2, x2, y2, z2 = qp
        data.qpos[3:7] = [w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                          w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2]
    mujoco.mj_forward(model, data)

    obs_history = np.zeros((4, OBS_DIM_BASE), dtype=np.float32)
    last_action = np.zeros(ACTION_DIM, dtype=np.float32)

    # 延迟缓冲 (复现训练侧 latency): 存历史 obs(148) 与 action(6)
    cap = max(latency, 0) + 1
    obs_delay_buf = [obs_history.flatten().astype(np.float32).copy() for _ in range(cap)]
    act_delay_buf = [last_action.copy() for _ in range(cap)]

    pitches, rolls = [], []
    ang_vels, lin_vels_local = [], []
    actions_sq = []
    gaps_l, gaps_r = [], []

    for step in range(max_steps):
        if _is_fallen(data):
            break
        # 推理: 用延迟后的 obs_history。latency=0 为当前; latency=k 取 k 步前 = buf[-(k+1)]
        # (缓冲末尾为当前帧, 与训练 _delayed_obs 语义一致)
        # teacher actor 条件于静态特权 z(9), 标称评估下补 nominal z → 157 维
        inf_obs148 = obs_delay_buf[-(latency + 1)] if latency > 0 else obs_history.flatten()
        inf_obs = np.concatenate([inf_obs148, NOMINAL_STATIC])
        with torch.no_grad():
            action = teacher(torch.from_numpy(inf_obs).float().unsqueeze(0)).numpy()[0]
        # 执行延迟后的动作 (latency=0 时为当前 action; latency=k 取 k 步前)
        applied = act_delay_buf[-(latency + 1)] if latency > 0 else action
        action_delta = action - last_action
        _apply_action(data, applied, command)
        last_action = action  # 注意: obs 里的 last_action 用 policy 原始输出 (与 env 一致)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(model, data)

        # 推理后才更新 history (与训练 step() 顺序一致: step 后 append base_obs)
        base_obs = _build_obs(data, last_action, command, step, model)
        obs_history = np.roll(obs_history, -1, axis=0)
        obs_history[-1] = base_obs

        # 推入延迟缓冲
        obs_delay_buf.append(obs_history.flatten().astype(np.float32).copy())
        if len(obs_delay_buf) > cap:
            obs_delay_buf.pop(0)
        act_delay_buf.append(action.copy())
        if len(act_delay_buf) > cap:
            act_delay_buf.pop(0)

        pitch, roll = _get_pitch_roll(data)
        pitches.append(pitch); rolls.append(roll)
        q = np.array([data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]])
        ang_vels.append(rotate_vector_by_quaternion_conj(q, data.qvel[3:6]))
        lin_vels_local.append(rotate_vector_by_quaternion_conj(q, data.qvel[0:3])[0])
        actions_sq.append(np.sum(action_delta**2))
        gaps_l.append(_site_gap(data, model, "l"))
        gaps_r.append(_site_gap(data, model, "r"))

    pitches = np.array(pitches); rolls = np.array(rolls)
    ang_vels = np.array(ang_vels); lin_vels_local = np.array(lin_vels_local)
    n = len(pitches)
    return {
        "stable_steps": n,
        "stable_seconds": n * CTRL_DT,
        "fallen": n < max_steps,
        "pitch_rms_deg": np.degrees(np.sqrt(np.mean(pitches**2))) if n else 0,
        "roll_rms_deg": np.degrees(np.sqrt(np.mean(rolls**2))) if n else 0,
        "ang_vel_xy_var": np.var(ang_vels[:, :2]) if n else 0,
        "ang_vel_xy_rms": np.sqrt(np.mean(np.sum(ang_vels[:, :2]**2, axis=1))) if n else 0,
        "ang_vel_z_rms": np.sqrt(np.mean(ang_vels[:, 2]**2)) if n else 0,
        "lin_vel_track_err": np.mean(np.abs(lin_vels_local - command[0])) if n else 0,
        "action_smoothness": np.mean(actions_sq) if n else 0,
        "loop_gap_mm": max(max(gaps_l) if gaps_l else [0], max(gaps_r) if gaps_r else [0]) * 1000,
    }


def print_report(title, results):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")
    if not results:
        print("  (无结果)"); return
    keys = ["stable_steps", "stable_seconds", "fallen", "pitch_rms_deg", "roll_rms_deg",
            "ang_vel_xy_var", "ang_vel_xy_rms", "ang_vel_z_rms",
            "lin_vel_track_err", "action_smoothness", "loop_gap_mm"]
    units = {"stable_steps": "步", "stable_seconds": "s", "fallen": "", "pitch_rms_deg": "°",
             "roll_rms_deg": "°", "ang_vel_xy_var": "rad²/s²", "ang_vel_xy_rms": "rad/s",
             "ang_vel_z_rms": "rad/s", "lin_vel_track_err": "m/s", "action_smoothness": "",
             "loop_gap_mm": "mm"}
    print(f"  {'指标':<20} {'mean':>12} {'min':>12} {'max':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    for k in keys:
        vals = np.array([r[k] for r in results])
        print(f"  {k:<20} {vals.mean():>12.4f} {vals.min():>12.4f} {vals.max():>12.4f}{units[k]}")


def main():
    parser = argparse.ArgumentParser(description="KUAFU 策略评估")
    parser.add_argument("--ckpt", required=True, help="Teacher checkpoint 路径")
    parser.add_argument("--mode", choices=["deterministic", "dr", "cmd_sweep"],
                        default="deterministic")
    parser.add_argument("--episodes", type=int, default=10, help="DR 模式 episode 数")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="单 episode 最大步数 (默认 10000=200s)")
    parser.add_argument("--latency", type=int, default=0,
                        help="观测/动作延迟步数 (复现训练 latency DR, 默认 0)")
    args = parser.parse_args()

    from rl.train.teacher_model import TeacherInferenceModel
    import torch

    xml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kuafu.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    teacher = TeacherInferenceModel.from_checkpoint(args.ckpt)
    print(f"Teacher 加载: {args.ckpt}")
    print(f"模式: {args.mode}, max_steps={args.max_steps} ({args.max_steps*CTRL_DT:.0f}s)")

    DWELL = P.D0_MIN

    if args.mode == "deterministic":
        command = np.array([0.0, 0.0, DWELL])
        r = run_episode(model, data, teacher, command, args.max_steps, latency=args.latency)
        print_report("deterministic (v=0, ω=0, d0=dwell)", [r])

    elif args.mode == "dr":
        rng = np.random.default_rng(42)
        command = np.array([0.0, 0.0, DWELL])
        results = []
        for ep in range(args.episodes):
            r = run_episode(model, data, teacher, command, args.max_steps, rng=rng, dr=True,
                            latency=args.latency)
            results.append(r)
            status = "倒下" if r["fallen"] else "稳定"
            print(f"  ep {ep+1}/{args.episodes}: {r['stable_steps']}步 ({r['stable_seconds']:.1f}s) [{status}]")
        print_report(f"DR × {args.episodes} episodes", results)

    elif args.mode == "cmd_sweep":
        velocities = np.linspace(-0.5, 0.5, 11)
        results = []
        print(f"\n命令扫描: v ∈ [-0.5, 0.5] m/s")
        print(f"  {'v_cmd':>8} {'稳定步数':>10} {'状态':>6} {'跟踪误差':>10}")
        for v in velocities:
            command = np.array([v, 0.0, DWELL])
            r = run_episode(model, data, teacher, command, args.max_steps, latency=args.latency)
            results.append(r)
            status = "倒下" if r["fallen"] else "稳定"
            print(f"  {v:>8.2f} {r['stable_steps']:>10} {status:>6} {r['lin_vel_track_err']:>10.4f}")
        print_report("cmd_sweep", results)


if __name__ == "__main__":
    main()
