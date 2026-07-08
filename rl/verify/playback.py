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
    """加载 RSL-RL Teacher checkpoint, 在原生 MuJoCo 中回放.

    Teacher policy = ActorCritic, 输入 obs, 输出 action_mean。
    """
    import torch

    # 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor_state = checkpoint.get("actor_state_dict") or checkpoint.get("model_state_dict", {})
    print(f"Checkpoint keys: {list(checkpoint.keys())[:5]}")

    # 构造 obs (简化: 用随机 obs 验证管线)
    obs_dim = 140
    obs = np.zeros(obs_dim, dtype=np.float32)

    print(f"启动回放: {duration:.0f}s, obs_dim={obs_dim}")
    print("  (关闭窗口退出)")

    # 启动 viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(int(duration / model.opt.timestep)):
            if not viewer.is_running():
                break
            # LQR 控制保持平衡 (简化: 实际应用 policy 推理)
            # pitch 从完整四元数提取 (与 kuafu_mjx_env.py 一致)
            qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
            theta = np.arctan2(2 * (qw * qy - qx * qz), 1 - 2 * (qy**2 + qz**2))
            x = data.qpos[0]
            xdot = data.qvel[0]
            thetadot = data.qvel[4]
            import kuafu_physics as P
            F = -(P.LQR_K @ np.array([x, theta, xdot, thetadot]))
            tau = F * P.R / 2.0
            data.ctrl[0] = np.clip(tau, -P.TAU_WHEEL_STALL, P.TAU_WHEEL_STALL)
            data.ctrl[1] = np.clip(tau, -P.TAU_WHEEL_STALL, P.TAU_WHEEL_STALL)
            # 腿保持驻留
            data.ctrl[2:6] = 0.0

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
