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
from rl.env.kuafu_env import OBS_DIM_BASE, OBS_DIM, ACTION_DIM  # 35 / 140 / 6

CTRL_DT = P.RL_DT
PITCH_THRESH = np.radians(30)   # 与训练 _is_fallen 一致
ROLL_THRESH = np.radians(30)
OMEGA_NOLOAD = P.RPM_WHEEL_NOLOAD * 2 * np.pi / 60  # DDSM315 空载角速度 rad/s


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


def _build_obs(data, applied_action, command, step, model=None):
    """35 维固定尺度 Actor frame，与 MJX ``_base_observation`` 同序。

    applied_action: 实际施加（延迟后）的动作，与训练 env applied_action=delayed_action 一致。
    """
    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    q = np.array([qw, qx, qy, qz])
    ang_vel = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
    gravity = rotate_vector_by_quaternion_conj(q, np.array([0.0, 0.0, -1.0]))
    q_l = P.fivebar_fk_relative(data.qpos[7], data.qpos[10])
    q_r = P.fivebar_fk_relative(data.qpos[12], data.qpos[15])
    d0 = (-q_l[1] - q_r[1]) * 0.5
    wheel_speed = np.array([data.qvel[8], data.qvel[13]])
    hip_pos = np.array([data.qpos[7], data.qpos[10], data.qpos[12], data.qpos[15]])
    hip_vel = np.array([data.qvel[6], data.qvel[9], data.qvel[11], data.qvel[14]])
    est_vx = float(np.mean(wheel_speed) * P.R)
    d0_cmd = command[2]
    if abs(command[0]) > P.D0_GATE_V_THRESH or abs(command[1]) > P.D0_GATE_W_THRESH:
        d0_cmd = min(d0_cmd, P.D0_GATE_MAX_HIGH)
    return np.concatenate([
        np.array([command[0] / 0.5, command[1], (d0_cmd - 132.5) / 74.5]),
        gravity,
        ang_vel / 10.0,
        np.array([est_vx / 0.5, ang_vel[2], (d0 - 132.5) / 74.5, roll]),
        wheel_speed / 33.0,
        hip_pos / 3.3,
        hip_vel / P.SERVO_MAX_SPEED,
        applied_action,
        np.zeros(6),
    ])


class NativeBaseline:
    """Native-MuJoCo counterpart of the 500/250/50 Hz MJX baseline."""

    def __init__(self, data, torque_scale=1.0, deadband=0.0):
        self.x_ref = float(data.qpos[0])
        self.x_int = 0.0
        self.v_ref = self.v_accel = 0.0
        self.w_ref = self.w_accel = 0.0
        self.yaw_ref = self._yaw(data)
        self.torque_scale = torque_scale
        self.deadband = deadband

    @staticmethod
    def _yaw(data):
        qw, qx, qy, qz = data.qpos[3:7]
        return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz * qz + qy * qy))

    @staticmethod
    def _jerk(target, value, accel, max_accel, max_jerk):
        target_accel = np.clip((target - value) / P.BASE_DT, -max_accel, max_accel)
        accel = np.clip(accel + np.clip(target_accel - accel, -max_jerk * P.BASE_DT, max_jerk * P.BASE_DT),
                        -max_accel, max_accel)
        return value + accel * P.BASE_DT, accel

    def step(self, model, data, action, command):
        for _ in range(P.BASE_STEPS_PER_RL):
            q = data.qpos[3:7]
            body_velocity = rotate_vector_by_quaternion_conj(q, data.qvel[:3])
            body_rate = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
            qw, qx, qy, qz = q
            pitch = np.arcsin(np.clip(2 * (qw * qy - qx * qz), -1.0, 1.0))
            self.v_ref, self.v_accel = self._jerk(command[0], self.v_ref, self.v_accel, 2.0, 8.0)
            self.w_ref, self.w_accel = self._jerk(command[1], self.w_ref, self.w_accel, 4.0, 16.0)
            self.x_ref += self.v_ref * P.BASE_DT
            self.yaw_ref += self.w_ref * P.BASE_DT
            self.x_int = np.clip(self.x_int + (data.qpos[0] - self.x_ref) * P.BASE_DT, -0.25, 0.25)
            state = np.array([data.qpos[0] - self.x_ref, pitch, body_velocity[0] - self.v_ref, body_rate[1]])
            force = -(P.LQR_K_DT4 @ state) - P.LQI_KI_DT4 * self.x_int
            tau_pitch = force * P.R / 2.0 * self.torque_scale
            yaw_error = np.arctan2(np.sin(self.yaw_ref - self._yaw(data)), np.cos(self.yaw_ref - self._yaw(data)))
            tau_yaw = (P.YAW_KP * yaw_error + P.YAW_KD * (self.w_ref - body_rate[2])) * self.torque_scale
            data.ctrl[0] = _limit_wheel_torque(
                tau_pitch - tau_yaw + (action[0] - action[1]) * P.TAU_WHEEL_RATED * self.torque_scale,
                data.qvel[8])
            data.ctrl[1] = _limit_wheel_torque(
                tau_pitch + tau_yaw + (action[0] + action[1]) * P.TAU_WHEEL_RATED * self.torque_scale,
                data.qvel[13])
            roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
            roll_d0 = -(P.ROLL_KP * roll + P.ROLL_KD * body_rate[0])
            d0_l = np.clip(command[2] + roll_d0 / 2.0 + action[3] * 30.0, P.D0_MIN, P.D0_MAX)
            d0_r = np.clip(command[2] - roll_d0 / 2.0 + action[5] * 30.0, P.D0_MIN, P.D0_MAX)
            qA_l, qB_l = P.fivebar_ik_cmd_xy(action[2] * 20.0, d0_l)
            qA_r, qB_r = P.fivebar_ik_cmd_xy(action[4] * 20.0, d0_r)
            goals = np.array([qA_l, qA_r, qB_l, qB_r])
            actual = data.qpos[[7, 12, 10, 15]]
            goals = np.where(np.abs(goals - actual) < self.deadband, actual, goals)
            data.ctrl[2:6] = goals
            mujoco.mj_step(model, data)
            mujoco.mj_step(model, data)


def _get_pitch_roll(data):
    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    pitch = np.arcsin(np.clip(2 * (qw * qy - qx * qz), -0.999999, 0.999999))
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    return pitch, roll


def _is_fallen(data):
    q = np.asarray(data.qpos[3:7])
    gravity = rotate_vector_by_quaternion_conj(q, np.asarray([0.0, 0.0, -1.0]))
    return gravity[2] > -np.cos(PITCH_THRESH)


def _site_gap(data, model, suffix):
    sa = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"Q_A_{suffix}")]
    sb = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"Q_B_{suffix}")]
    return np.linalg.norm(sa - sb)


def run_episode(model, data, teacher, command, max_steps, rng=None, dr=False,
                latency=0, sense_latency=None):
    """跑一个 episode, 返回指标 dict. fallen 则提前终止.

    latency: action delay steps; sense_latency independently controls observation
    delay. Both buffers match the training-side action/observation delay semantics.
    """
    import torch
    nominal = None
    torque_scale = 1.0
    deadband = 0.0
    if dr and rng is not None:
        chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        wheel_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                     for name in ("wheel_l_geom", "wheel_r_geom")]
        nominal = {
            "body_mass": model.body_mass.copy(),
            "body_ipos": model.body_ipos.copy(),
            "body_inertia": model.body_inertia.copy(),
            "geom_friction": model.geom_friction.copy(),
            "geom_size": model.geom_size.copy(),
            "actuator_gainprm": model.actuator_gainprm.copy(),
            "actuator_biasprm": model.actuator_biasprm.copy(),
        }
        model.body_mass[:] = nominal["body_mass"] * rng.uniform(*P.DR_MASS)
        model.body_inertia[:] = nominal["body_inertia"] * rng.uniform(*P.DR_INERTIA)
        model.body_ipos[chassis] = nominal["body_ipos"][chassis] + rng.uniform(P.DR_COM[0], P.DR_COM[1], 3)
        model.geom_friction[:, 0] = nominal["geom_friction"][:, 0] * rng.uniform(*P.DR_FRICTION)
        wheel_delta = rng.uniform(*P.DR_WHEEL_R)
        for wheel_id in wheel_ids:
            model.geom_size[wheel_id, 0] = nominal["geom_size"][wheel_id, 0] + wheel_delta
        pd_scale = rng.uniform(*P.DR_SERVO_PD)
        for actuator in range(2, 6):
            model.actuator_gainprm[actuator, 0] = nominal["actuator_gainprm"][actuator, 0] * pd_scale
            model.actuator_biasprm[actuator, 1:] = nominal["actuator_biasprm"][actuator, 1:] * pd_scale
        torque_scale = rng.uniform(*P.DR_TORQUE_CONST)
        deadband = rng.uniform(*P.DR_DEADBAND)
        mujoco.mj_setConst(model, data)
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
    baseline = NativeBaseline(data, torque_scale=torque_scale, deadband=deadband)

    obs_history = np.zeros((4, OBS_DIM_BASE), dtype=np.float32)
    last_action = np.zeros(ACTION_DIM, dtype=np.float32)

    # 延迟缓冲 (复现训练侧 latency): 存历史 obs(140) 与 action(6)
    if sense_latency is None:
        sense_latency = latency
    cap = max(latency, sense_latency, 0) + 1
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
        inf_obs = obs_delay_buf[-(sense_latency + 1)] if sense_latency > 0 else obs_history.flatten()
        with torch.no_grad():
            action = teacher(torch.from_numpy(inf_obs).float().unsqueeze(0)).numpy()[0]
        # 执行延迟后的动作 (latency=0 时为当前 action; latency=k 取 k 步前)
        applied = act_delay_buf[-(latency + 1)] if latency > 0 else action
        action_delta = action - last_action
        baseline.step(model, data, applied, command)
        last_action = action  # raw policy output, for action_smoothness metric

        # 推理后才更新 history (与训练 step() 顺序一致: step 后 append base_obs)
        base_obs = _build_obs(data, applied, command, step, model)
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
    result = {
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
    if nominal is not None:
        model.body_mass[:] = nominal["body_mass"]
        model.body_ipos[:] = nominal["body_ipos"]
        model.body_inertia[:] = nominal["body_inertia"]
        model.geom_friction[:] = nominal["geom_friction"]
        model.geom_size[:] = nominal["geom_size"]
        model.actuator_gainprm[:] = nominal["actuator_gainprm"]
        model.actuator_biasprm[:] = nominal["actuator_biasprm"]
        mujoco.mj_setConst(model, data)
    return result


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
                        help="动作延迟步数 (默认 0)")
    parser.add_argument("--sense_latency", type=int, default=None,
                        help="独立观测延迟步数；省略时沿用 --latency")
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
        r = run_episode(model, data, teacher, command, args.max_steps, latency=args.latency,
                         sense_latency=args.sense_latency)
        print_report("deterministic (v=0, ω=0, d0=dwell)", [r])

    elif args.mode == "dr":
        rng = np.random.default_rng(42)
        command = np.array([0.0, 0.0, DWELL])
        results = []
        for ep in range(args.episodes):
            r = run_episode(model, data, teacher, command, args.max_steps, rng=rng, dr=True,
                            latency=args.latency, sense_latency=args.sense_latency)
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
            r = run_episode(model, data, teacher, command, args.max_steps, latency=args.latency,
                            sense_latency=args.sense_latency)
            results.append(r)
            status = "倒下" if r["fallen"] else "稳定"
            print(f"  {v:>8.2f} {r['stable_steps']:>10} {status:>6} {r['lin_vel_track_err']:>10.4f}")
        print_report("cmd_sweep", results)


if __name__ == "__main__":
    main()
