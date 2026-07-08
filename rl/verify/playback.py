# -*- coding: utf-8 -*-
"""
KUAFU 策略回放 — design.md §2.6 阶段 3

在原生 MuJoCo (CPU 单环境) 中加载训练好的 policy, 可视化确认行为合理。
MJX → 原生 MuJoCo 通过 mjx.get_data 转换。

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


def playback_teacher(ckpt_path: str, xml_path: str, duration: float = 10.0):
    """加载 RSL-RL Teacher checkpoint, 在原生 MuJoCo 中回放 policy 行为."""
    import torch
    from rl.train.teacher_model import TeacherInferenceModel

    # 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # 加载 teacher policy (精确匹配 checkpoint 结构)
    teacher = TeacherInferenceModel.from_checkpoint(ckpt_path)
    teacher.eval()
    print(f"Teacher policy 加载成功")

    print(f"启动回放: {duration:.0f}s")
    print("  (关闭窗口退出)")

    # 历史缓冲 (4 步 × 35 维)
    obs_history = np.zeros((4, 35), dtype=np.float32)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(int(duration / model.opt.timestep)):
            if not viewer.is_running():
                break

            # 构造 35 维 base obs (简化版, 实际从 sensor 读)
            qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
            roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
            pitch = np.arctan2(2 * (qw * qy - qx * qz), 1 - 2 * (qy**2 + qz**2))
            yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz**2 + qy**2))
            attitude = np.array([roll, pitch, yaw])
            ang_vel = data.qvel[3:6]
            wheel_state = np.array([data.qpos[9], data.qpos[14], data.qvel[8], data.qvel[13]])
            hip_state = np.array([data.qpos[7], data.qpos[10], data.qpos[12], data.qpos[15],
                                  data.qvel[6], data.qvel[9], data.qvel[11], data.qvel[14]])
            wheel_torque = np.array([data.actuator_force[0], data.actuator_force[1]])
            hip_torque = np.array([data.actuator_force[2], data.actuator_force[3],
                                   data.actuator_force[4], data.actuator_force[5]])
            last_action = np.zeros(6)  # 简化
            command = np.array([0.0, 0.0, 58.0])  # 静止 + 驻留
            phase_clock = np.array([0.0, 1.0])
            base_obs = np.concatenate([attitude, ang_vel, wheel_state, hip_state,
                                       wheel_torque, hip_torque, last_action, command, phase_clock])

            # 更新历史
            obs_history = np.roll(obs_history, -1, axis=0)
            obs_history[-1] = base_obs
            obs_flat = obs_history.flatten()  # (140,)

            # Teacher 推理
            with torch.no_grad():
                action = teacher(torch.from_numpy(obs_flat).float()).numpy()

            # LQR 底层 + RL 残差
            import kuafu_physics as P
            x = data.qpos[0]
            xdot = data.qvel[0]
            thetadot = data.qvel[4]
            F = -(P.LQR_K @ np.array([x, pitch, xdot, thetadot]))
            tau_lqr = F * P.R / 2.0
            data.ctrl[0] = np.clip(tau_lqr + action[0] * P.TAU_WHEEL_RATED, -1.1, 1.1)
            data.ctrl[1] = np.clip(tau_lqr + action[1] * P.TAU_WHEEL_RATED, -1.1, 1.1)
            data.ctrl[2:6] = action[2:6]

            mujoco.mj_step(model, data)
            viewer.sync()


def playback_student(ckpt_path: str, xml_path: str, duration: float = 10.0):
    """加载 Student policy (trunk + adapter), 在原生 MuJoCo 中回放."""
    from rl.train.networks import StudentPolicy

    import torch

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # 加载 student
    student = StudentPolicy(proprio_dim=140, history_obs_dim=35, history_len=50)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    student.load_state_dict(ckpt.get("student_state_dict", ckpt))
    student.eval()

    # 历史缓冲
    history = np.zeros((1, 50, 35), dtype=np.float32)
    proprio = np.zeros((1, 140), dtype=np.float32)

    print(f"启动 Student 回放: {duration:.0f}s")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(int(duration / model.opt.timestep)):
            if not viewer.is_running():
                break

            # 构造 obs (简化版, 实际从 sensor 读)
            proprio_tensor = torch.from_numpy(proprio)
            history_tensor = torch.from_numpy(history)
            with torch.no_grad():
                action = student(proprio_tensor, history_tensor)
            action_np = action.numpy()[0]

            # 应用动作 (LQR 底层 + RL 残差)
            import kuafu_physics as P
            qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
            theta = np.arctan2(2 * (qw * qy - qx * qz), 1 - 2 * (qy**2 + qz**2))
            F = -(P.LQR_K @ np.array([data.qpos[0], theta, data.qvel[0], data.qvel[4]]))
            tau_lqr = F * P.R / 2.0
            data.ctrl[0] = np.clip(tau_lqr + action_np[0] * P.TAU_WHEEL_RATED, -1.1, 1.1)
            data.ctrl[1] = np.clip(tau_lqr + action_np[1] * P.TAU_WHEEL_RATED, -1.1, 1.1)
            data.ctrl[2:6] = action_np[2:6]

            # 10 子步物理 (500Hz)
            for _ in range(10):
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
