# -*- coding: utf-8 -*-
"""
KUAFU 策略回放 — design.md §2.6 阶段 3

在原生 MuJoCo (CPU 单环境) 中加载训练好的 policy, 可视化确认行为合理。
50Hz 控制频率, 500Hz 物理子步 (10:1), 与训练环境时序一致。

运行:
  rl/.venv/bin/python rl/verify/playback.py --ckpt rl/checkpoints/teacher_*/model_500.pt
  rl/.venv/bin/python rl/verify/playback.py --ckpt policy.pt --student
"""
import os
import sys
import argparse

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

import numpy as np
import mujoco
import mujoco.viewer

import kuafu_physics as P

# 观测/动作维度 (与训练环境单一真源一致)
from rl.env.kuafu_env import OBS_DIM_BASE, OBS_DIM, ACTION_DIM  # 31 / 124 / 6

# 控制频率 (与训练环境一致)
CTRL_DT = 0.02   # 50 Hz
N_SUBSTEPS = 10  # 500 Hz 物理


def rotate_vector_by_quaternion_conj(q, v):
    """Rotate vector v by conjugate of quaternion q (numpy version)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    q_xyz = np.array([-x, -y, -z])
    uv = np.cross(q_xyz, v)
    uuv = np.cross(q_xyz, uv)
    return v + 2 * (w * uv + uuv)


def _build_obs(data, obs_history, last_action):
    """构造 31 维 base obs (从 MuJoCo data 提取; 2-DOF 五杆全观测 4 舵机)."""
    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qx * qz), -0.999999, 0.999999))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz**2 + qy**2))
    attitude = np.array([roll, pitch, yaw])

    # 角速度转为本体坐标系
    q = np.array([qw, qx, qy, qz])
    ang_vel = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])

    wheel_state = np.array([data.qpos[9], data.qpos[14], data.qvel[8], data.qvel[13]])
    # 4 舵机 hip_A_l/r (qpos 7,12; qvel 6,11) + hip_B_l/r (qpos 10,15; qvel 9,14)
    hip_state = np.array([
        data.qpos[7], data.qpos[12], data.qpos[10], data.qpos[15],
        data.qvel[6], data.qvel[11], data.qvel[9], data.qvel[14]])
    wheel_torque = np.array([data.actuator_force[0], data.actuator_force[1]])
    # 4 舵机力矩: tau_l,tau_r,q_hip_A_l,q_hip_A_r,q_hip_B_l,q_hip_B_r
    hip_torque = np.array([
        data.actuator_force[2], data.actuator_force[3],
        data.actuator_force[4], data.actuator_force[5]])
    command = np.array([0.0, 0.0, P.D0_MIN])  # 静止 + 驻留
    phase_clock = np.array([0.0, 1.0])
    return np.concatenate([attitude, ang_vel, wheel_state, hip_state,
                           wheel_torque, hip_torque, last_action, command, phase_clock])


def _apply_action(data, action):
    """LQR 底层 + RL 残差叠加 → ctrl (2-DOF: action[2:6] 驱动 4 舵机)."""
    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    q = np.array([qw, qx, qy, qz])

    # LQR 状态量转投影至本体坐标系
    lin_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[0:3])
    xdot = lin_vel_local[0]

    ang_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
    thetadot = ang_vel_local[1]

    pitch = np.arcsin(np.clip(2 * (qw * qy - qx * qz), -0.999999, 0.999999))

    # LQR: x 项置 0 (与环境一致)
    F = -(P.LQR_K @ np.array([0.0, pitch, xdot, thetadot]))
    tau_lqr = F * P.R / 2.0
    data.ctrl[0] = np.clip(tau_lqr + action[0] * P.TAU_WHEEL_RATED, -1.1, 1.1)
    data.ctrl[1] = np.clip(tau_lqr + action[1] * P.TAU_WHEEL_RATED, -1.1, 1.1)
    # 腿: action[2:6] × HIP_STROKE → q_hip_A_l/r, q_hip_B_l/r (4 舵机独立)
    data.ctrl[2:6] = action[2:6] * P.HIP_STROKE


def playback_teacher(ckpt_path: str, xml_path: str, duration: float = 10.0):
    """加载 Teacher policy, 50Hz 控制 + 500Hz 物理回放."""
    import torch
    from rl.train.teacher_model import TeacherInferenceModel

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    teacher = TeacherInferenceModel.from_checkpoint(ckpt_path)
    teacher.eval()
    print(f"Teacher policy 加载成功")

    obs_history = np.zeros((4, OBS_DIM_BASE), dtype=np.float32)
    last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    n_ctrl_steps = int(duration / CTRL_DT)

    print(f"启动回放: {duration:.0f}s ({n_ctrl_steps} ctrl steps × {N_SUBSTEPS} substeps)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(n_ctrl_steps):
            if not viewer.is_running():
                break

            # 50Hz: 构造 obs → policy 推理
            base_obs = _build_obs(data, obs_history, last_action)
            obs_history = np.roll(obs_history, -1, axis=0)
            obs_history[-1] = base_obs
            obs_flat = obs_history.flatten()

            with torch.no_grad():
                action = teacher(torch.from_numpy(obs_flat).float()).numpy()
            last_action = action

            # 应用动作
            _apply_action(data, action)

            # 500Hz: 10 子步物理
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(model, data)
            viewer.sync()


def playback_student(ckpt_path: str, xml_path: str, duration: float = 10.0):
    """加载 Student policy, 50Hz 控制 + 500Hz 物理回放."""
    import torch
    from rl.train.networks import StudentPolicy

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("student_state_dict", ckpt.get("model_state_dict", ckpt))

    # 动态推断 Student trunk 的维度
    hidden_dims = []
    i = 0
    while f"trunk.{i*2}.weight" in state:
        hidden_dims.append(state[f"trunk.{i*2}.weight"].shape[0])
        i += 1

    student = StudentPolicy(
        proprio_dim=OBS_DIM, history_obs_dim=OBS_DIM_BASE, history_len=50,
        hidden_dims=tuple(hidden_dims),
    )
    student.load_state_dict(state, strict=False)
    student.eval()

    obs_history = np.zeros((4, OBS_DIM_BASE), dtype=np.float32)
    rma_history = np.zeros((1, 50, OBS_DIM_BASE), dtype=np.float32)  # RMA adapter 50 步历史
    last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    n_ctrl_steps = int(duration / CTRL_DT)

    print(f"启动 Student 回放: {duration:.0f}s")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(n_ctrl_steps):
            if not viewer.is_running():
                break

            # 50Hz: 构造 obs
            base_obs = _build_obs(data, obs_history, last_action)
            obs_history = np.roll(obs_history, -1, axis=0)
            obs_history[-1] = base_obs
            obs_flat = obs_history.flatten()

            # 更新 RMA 50 步历史
            rma_history = np.roll(rma_history, -1, axis=1)
            rma_history[0, -1, :] = base_obs

            # Student 推理
            with torch.no_grad():
                action = student(
                    torch.from_numpy(obs_flat).float().unsqueeze(0),
                    torch.from_numpy(rma_history).float(),
                ).numpy()[0]
            last_action = action

            _apply_action(data, action)

            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(model, data)
            viewer.sync()


def main():
    parser = argparse.ArgumentParser(description="KUAFU 策略回放")
    parser.add_argument("--ckpt", required=True, help="Checkpoint 路径")
    parser.add_argument("--student", action="store_true", help="Student 模式 (否则 Teacher)")
    parser.add_argument("--duration", type=float, default=10.0, help="回放时长 (s)")
    args = parser.parse_args()

    xml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kuafu.xml")

    if args.student:
        playback_student(args.ckpt, xml, args.duration)
    else:
        playback_teacher(args.ckpt, xml, args.duration)


if __name__ == "__main__":
    main()
