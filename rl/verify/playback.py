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
from rl.env.kuafu_env import OBS_DIM_BASE, OBS_DIM, ACTION_DIM  # 35 / 140 / 6

# 从 eval_policy.py 导入统一的物理工具函数和 obs 构造器
from rl.verify.eval_policy import (
    _build_obs,
    _apply_action,
    rotate_vector_by_quaternion_conj,
    _limit_wheel_torque
)

# 控制频率 (与训练环境一致)
CTRL_DT = 0.02   # 50 Hz
N_SUBSTEPS = 10  # 500 Hz 物理
OMEGA_NOLOAD = P.RPM_WHEEL_NOLOAD * 2 * np.pi / 60  # DDSM315 空载角速度 rad/s


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

            # 50Hz: 推理 (用当前 history, 第一步全 0 与训练 reset 一致)
            obs_flat = obs_history.flatten()
            with torch.no_grad():
                action = teacher(torch.from_numpy(obs_flat).float()).numpy()

            # 应用动作
            _apply_action(data, action)
            last_action = action

            # 500Hz: 10 子步物理
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(model, data)

            # 推理后才更新 history (与训练 step() 顺序一致)
            command = np.array([0.0, 0.0, P.D0_MIN])
            base_obs = _build_obs(data, last_action, command, step)
            obs_history = np.roll(obs_history, -1, axis=0)
            obs_history[-1] = base_obs
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
        for step in range(n_ctrl_steps):
            if not viewer.is_running():
                break

            # 50Hz: 推理 (用当前 history, 第一步全 0 与训练 reset 一致)
            obs_flat = obs_history.flatten()
            with torch.no_grad():
                action = student(
                    torch.from_numpy(obs_flat).float().unsqueeze(0),
                    torch.from_numpy(rma_history).float(),
                ).numpy()[0]

            _apply_action(data, action)
            last_action = action

            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(model, data)

            # 推理后才更新 history (与训练 step() 顺序一致)
            command = np.array([0.0, 0.0, P.D0_MIN])
            base_obs = _build_obs(data, last_action, command, step)
            obs_history = np.roll(obs_history, -1, axis=0)
            obs_history[-1] = base_obs
            rma_history = np.roll(rma_history, -1, axis=1)
            rma_history[0, -1, :] = base_obs
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
