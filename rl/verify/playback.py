# -*- coding: utf-8 -*-
"""
KUAFU 策略回放 — design.md §2.6 阶段 3

在原生 MuJoCo (CPU 单环境) 中加载训练好的 policy, 可视化确认行为合理。
50Hz 控制频率, 500Hz 物理子步 (10:1), 与训练环境时序一致。

运行:
  rl/.venv/bin/python rl/verify/playback.py --ckpt rl/checkpoints/garlic/teacher/model_3999.pt
  rl/.venv/bin/python rl/verify/playback.py --ckpt rl/checkpoints/garlic/student/model_final.pt --student
"""
import os
import sys
import time
import argparse
import faulthandler

faulthandler.enable()  # 崩溃时打印 Python/C 栈, 便于定位原生堆损坏

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

import numpy as np
import mujoco
import mujoco.viewer

import kuafu_physics as P

# 观测/动作维度 (与训练环境单一真源一致)
from rl.env.kuafu_env import OBS_DIM_BASE, OBS_DIM, ACTION_DIM  # 37 / 148 / 6

# 从 eval_policy.py 导入统一的物理工具函数和 obs 构造器
from rl.verify.eval_policy import (
    _build_obs,
    _apply_action,
    rotate_vector_by_quaternion_conj,
    _limit_wheel_torque,
    NOMINAL_STATIC,
)

# 控制频率 (与训练环境一致)
CTRL_DT = 0.02   # 50 Hz
N_SUBSTEPS = 10  # 500 Hz 物理
OMEGA_NOLOAD = P.RPM_WHEEL_NOLOAD * 2 * np.pi / 60  # DDSM315 空载角速度 rad/s


def playback_teacher(ckpt_path: str, xml_path: str, duration: float = 10.0, latency: int = 0):
    """加载 Teacher policy, 50Hz 控制 + 500Hz 物理回放.

    latency: 观测/动作延迟步数, 复现训练侧 latency DR (默认 0).
    """
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
    cap = max(latency, 0) + 1
    obs_delay_buf = [obs_history.flatten().astype(np.float32).copy() for _ in range(cap)]
    act_delay_buf = [last_action.copy() for _ in range(cap)]
    n_ctrl_steps = int(duration / CTRL_DT)

    print(f"启动回放: {duration:.0f}s ({n_ctrl_steps} ctrl steps × {N_SUBSTEPS} substeps)"
          + (f", latency={latency}" if latency > 0 else ""))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(n_ctrl_steps):
            if not viewer.is_running():
                break

            # 50Hz: 推理 (用延迟后的 obs, 第一步全 0 与训练 reset 一致)
            # latency=k 取 k 步前 = buf[-(k+1)], 与训练 _delayed_obs 语义一致
            # teacher actor 条件于静态特权 z(9), 标称评估下补 nominal z → 157 维
            inf_obs148 = obs_delay_buf[-(latency + 1)] if latency > 0 else obs_history.flatten()
            inf_obs = np.concatenate([inf_obs148, NOMINAL_STATIC])
            with torch.no_grad():
                action = teacher(torch.from_numpy(inf_obs).float().unsqueeze(0)).numpy()[0]
            applied = act_delay_buf[-(latency + 1)] if latency > 0 else action

            # 应用动作 + 500Hz 物理子步: 加 viewer.lock 防止渲染线程并发读/重分配 data 导致堆损坏
            command = np.array([0.0, 0.0, P.D0_MIN])
            with viewer.lock():
                _apply_action(data, applied, command)
                last_action = action
                for _ in range(N_SUBSTEPS):
                    mujoco.mj_step(model, data)

            # 推理后才更新 history (与训练 step() 顺序一致)
            base_obs = _build_obs(data, last_action, command, step, model)
            obs_history = np.roll(obs_history, -1, axis=0)
            obs_history[-1] = base_obs
            obs_delay_buf.append(obs_history.flatten().astype(np.float32).copy())
            if len(obs_delay_buf) > cap:
                obs_delay_buf.pop(0)
            act_delay_buf.append(action.copy())
            if len(act_delay_buf) > cap:
                act_delay_buf.pop(0)
            viewer.sync()

        # 回放结束后保留窗口供肉眼检查, 用户关窗才退出 (避免 sim.exit() 死锁导致卡死)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
        os._exit(0)  # 在 with 块内强制退出, 跳过 __exit__ 的 sim.exit() 死锁


def playback_student(ckpt_path: str, xml_path: str, duration: float = 10.0, latency: int = 0):
    """加载 Student policy, 50Hz 控制 + 500Hz 物理回放.

    latency: 观测/动作延迟步数, 复现训练侧 latency DR (默认 0).
    """
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
    print(f"Student hidden_dims={hidden_dims}")

    obs_history = np.zeros((4, OBS_DIM_BASE), dtype=np.float32)
    rma_history = np.zeros((1, 50, OBS_DIM_BASE), dtype=np.float32)  # RMA adapter 50 步历史
    last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    cap = max(latency, 0) + 1
    obs_delay_buf = [obs_history.flatten().astype(np.float32).copy() for _ in range(cap)]
    rma_delay_buf = [rma_history.copy() for _ in range(cap)]
    act_delay_buf = [last_action.copy() for _ in range(cap)]
    n_ctrl_steps = int(duration / CTRL_DT)

    print(f"启动 Student 回放: {duration:.0f}s"
          + (f", latency={latency}" if latency > 0 else ""))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(n_ctrl_steps):
            if not viewer.is_running():
                break

            # 50Hz: 推理 (用延迟后的 proprio + RMA history)
            # latency=k 取 k 步前 = buf[-(k+1)], 与训练 _delayed_obs 语义一致
            inf_obs = obs_delay_buf[-(latency + 1)] if latency > 0 else obs_history.flatten()
            inf_rma = rma_delay_buf[-(latency + 1)] if latency > 0 else rma_history
            with torch.no_grad():
                action = student(
                    torch.from_numpy(inf_obs).float().unsqueeze(0),
                    torch.from_numpy(inf_rma).float(),
                ).numpy()[0]
            applied = act_delay_buf[-(latency + 1)] if latency > 0 else action

            # 加 viewer.lock 防止渲染线程并发读/重分配 data 导致堆损坏
            command = np.array([0.0, 0.0, P.D0_MIN])
            with viewer.lock():
                _apply_action(data, applied, command)
                last_action = action
                for _ in range(N_SUBSTEPS):
                    mujoco.mj_step(model, data)

            # 推理后才更新 history (与训练 step() 顺序一致)
            base_obs = _build_obs(data, last_action, command, step, model)
            obs_history = np.roll(obs_history, -1, axis=0)
            obs_history[-1] = base_obs
            rma_history = np.roll(rma_history, -1, axis=1)
            rma_history[0, -1, :] = base_obs
            obs_delay_buf.append(obs_history.flatten().astype(np.float32).copy())
            if len(obs_delay_buf) > cap:
                obs_delay_buf.pop(0)
            rma_delay_buf.append(rma_history.copy())
            if len(rma_delay_buf) > cap:
                rma_delay_buf.pop(0)
            act_delay_buf.append(action.copy())
            if len(act_delay_buf) > cap:
                act_delay_buf.pop(0)
            viewer.sync()

        # 回放结束后保留窗口供肉眼检查, 用户关窗才退出 (避免 sim.exit() 死锁导致卡死)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
        os._exit(0)  # 在 with 块内强制退出, 跳过 __exit__ 的 sim.exit() 死锁


def main():
    parser = argparse.ArgumentParser(description="KUAFU 策略回放")
    parser.add_argument("--ckpt", required=True, help="Checkpoint 路径")
    parser.add_argument("--student", action="store_true", help="Student 模式 (否则 Teacher)")
    parser.add_argument("--duration", type=float, default=10.0, help="回放时长 (s)")
    parser.add_argument("--latency", type=int, default=0,
                        help="观测/动作延迟步数 (复现训练 latency DR, 默认 0)")
    args = parser.parse_args()

    xml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kuafu.xml")

    if args.student:
        playback_student(args.ckpt, xml, args.duration, latency=args.latency)
    else:
        playback_teacher(args.ckpt, xml, args.duration, latency=args.latency)


if __name__ == "__main__":
    main()
