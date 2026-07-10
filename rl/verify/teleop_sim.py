# -*- coding: utf-8 -*-
"""
KUAFU 仿真遥控 - 手柄/键盘实时操控虚拟机器人

在原生 MuJoCo viewer 中加载 policy, 用手柄(或键盘)实时下发 [v, ω, d0] 命令,
经 CommandArbiter 仲裁后注入策略 obs。obs/action 逻辑复用 eval_policy.py(零拷贝)。

这是上真机前的零风险验证: 插上手柄就能在 MuJoCo 里遥控虚拟机器人, 验证:
  1. 键盘 WASD 能前后/转向
  2. 松手后机器人保持平衡不摔
  3. 急停键立即停住且保持平衡
  4. d0 调节能看到腿伸缩
  5. 手柄抢占/交还自主(若接了自主 stub)切换平滑

运行:
  rl/.venv/bin/python rl/verify/teleop_sim.py --ckpt rl/checkpoints/garlic/teacher/model_3999.pt
  rl/.venv/bin/python rl/verify/teleop_sim.py --ckpt ... --device keyboard   # 无手柄时
  rl/.venv/bin/python rl/verify/teleop_sim.py --ckpt ... --student            # Student 策略
"""
import os
import sys
import argparse

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

import numpy as np
import mujoco
import torch

import kuafu_physics as P
from rl.env.kuafu_env import OBS_DIM_BASE, OBS_DIM, ACTION_DIM
# 复用 eval_policy 的 obs 构造 + 动作施加(纯函数, command 注入版, 零改动)
from rl.verify.eval_policy import _build_obs, _apply_action
from rl.verify.eval_policy import CTRL_DT, N_SUBSTEPS

from rl.teleop.command import ArbiterConfig, Mode
from rl.teleop.arbiter import CommandArbiter


def _build_sources(device: str):
    """按 --device 参数构造命令源列表。

    返回 (sources, hint): sources 供 Arbiter 使用; hint 用于打印操作说明。
    """
    from rl.teleop.autonomous_source import AutonomousSource

    if device in ("gamepad", "auto"):
        try:
            from rl.teleop.gamepad_source import GamepadSource
            gp = GamepadSource()
            print("检测到手柄, 使用 gamepad 操控")
            hint = ("手柄: 左摇杆Y=前后 右摇杆X=转向 LT/RT=蹲起 "
                    "A=急停 B=急停\n"
                    "  无手柄可加 --device keyboard")
            return [gp, AutonomousSource()], hint
        except RuntimeError as e:
            if device == "gamepad":
                raise
            print(f"({e}; 改用键盘)")

    from rl.teleop.keyboard_source import KeyboardSource
    kb = KeyboardSource()
    print("使用 keyboard 操控")
    hint = ("键盘: W/S=前后 A/D=转向 Q/E=蹲起 空格=急停 R=解锁\n"
            "  先按 R 解锁(启动即急停态)")
    return [kb, AutonomousSource()], hint


def _load_policy(ckpt: str, student: bool):
    """加载 Teacher(.pt) 或 Student(.pt)。复用 playback.py 的加载逻辑。"""
    if student:
        from rl.train.networks import StudentPolicy
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = ck.get("student_state_dict", ck.get("model_state_dict", ck))
        hidden_dims = []
        i = 0
        while f"trunk.{i*2}.weight" in state:
            hidden_dims.append(state[f"trunk.{i*2}.weight"].shape[0])
            i += 1
        net = StudentPolicy(
            proprio_dim=OBS_DIM, history_obs_dim=OBS_DIM_BASE,
            history_len=50, hidden_dims=tuple(hidden_dims))
        net.load_state_dict(state, strict=False)
        net.eval()
        return net, "student"
    else:
        from rl.train.teacher_model import TeacherInferenceModel
        return TeacherInferenceModel.from_checkpoint(ckpt), "teacher"


def run_teleop(model, data, policy, ptype, arbiter, viewer, duration, hint):
    """50Hz 控制循环 + 实时命令注入。

    obs/action 复用 eval_policy: _build_obs(data, history, last_action, command)
    """
    obs_history = np.zeros((4, OBS_DIM_BASE), dtype=np.float32)
    last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    rma_history = None
    if ptype == "student":
        rma_history = np.zeros((1, 50, OBS_DIM_BASE), dtype=np.float32)

    n_steps = int(duration / CTRL_DT) if duration > 0 else 10**9
    print(hint)
    print(f"控制频率 {1/CTRL_DT:.0f}Hz, 物理子步 {N_SUBSTEPS} ({N_SUBSTEPS/CTRL_DT:.0f}Hz)")
    print("ESC 或关窗口退出\n" + "-" * 48)

    for step in range(n_steps):
        if not viewer.is_running():
            break

        # --- 仲裁出命令 ---
        cmd = arbiter.poll()
        command = np.array(cmd.as_array(), dtype=np.float32)

        # --- 构造 obs(注入实时 command) ---
        obs_flat = obs_history.flatten()
        with torch.no_grad():
            if ptype == "teacher":
                action = policy(torch.from_numpy(obs_flat).float().unsqueeze(0)).numpy()[0]
            else:
                action = policy(
                    torch.from_numpy(obs_flat).float().unsqueeze(0),
                    torch.from_numpy(rma_history).float(),
                ).numpy()[0]
        last_action = action

        # --- 施加动作(LQR + 残差) ---
        _apply_action(data, action)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(model, data)

        # --- 更新 obs history(训练 step 顺序: step 后 append) ---
        base_obs = _build_obs(data, last_action, command, step)
        obs_history = np.roll(obs_history, -1, axis=0)
        obs_history[-1] = base_obs
        if ptype == "student":
            rma_history = np.roll(rma_history, -1, axis=1)
            rma_history[0, -1, :] = base_obs

        # --- HUD: 每 20 步(0.4s)打印状态 ---
        if step % 20 == 0:
            mode_str = cmd.mode.value.upper()
            lock = " [LOCKED]" if arbiter.estop_locked else ""
            print(f"\r{mode_str:<11}{lock} v={cmd.v:+.2f} ω={cmd.omega:+.2f} "
                  f"d0={cmd.d0:6.1f}mm  step={step:<6d}", end="", flush=True)

        viewer.sync()
    print()


def main():
    parser = argparse.ArgumentParser(description="KUAFU 仿真遥控")
    parser.add_argument("--ckpt", required=True, help="Checkpoint 路径 (.pt)")
    parser.add_argument("--student", action="store_true", help="Student 模式 (否则 Teacher)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="运行时长 s (0=不限, 关窗口退出)")
    parser.add_argument("--device", choices=["gamepad", "keyboard", "auto"],
                        default="auto", help="输入设备 (auto=有手柄用手柄否则键盘)")
    args = parser.parse_args()

    xml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kuafu.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    policy, ptype = _load_policy(args.ckpt, args.student)
    print(f"策略加载: {args.ckpt} ({ptype})")

    sources, hint = _build_sources(args.device)
    arbiter = CommandArbiter(sources, ArbiterConfig())

    with mujoco.viewer.launch_passive(model, data) as viewer:
        run_teleop(model, data, policy, ptype, arbiter, viewer, args.duration, hint)


if __name__ == "__main__":
    main()
