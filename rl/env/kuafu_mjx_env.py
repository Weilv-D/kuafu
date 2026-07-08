# -*- coding: utf-8 -*-
"""
KUAFU 残差 RL 环境 — MuJoCo MJX 实现

继承 MuJoCo Playground MjxEnv, JAX 全函数化, 向量化运行在 GPU 上。
驻留态腿被动自锁, 整机降为轮式倒立摆; RL 输出残差叠加在 LQR 底层之上。

design.md 对应章节:
  §2.1 观测空间 / §2.2 动作空间 / §2.3 Reward / §2.4 域随机化
  §2.5 Teacher-Student + RMA / §3.x MJCF 建模

通过 train.py 的 DirectVecEnv 适配器桥接到 PyTorch/RSL-RL 2.x:
  DLPack 零拷贝 (JAX DeviceArray ↔ torch.Tensor), JAX cuda13 与 torch cu130 共享 runtime。
  (绕过 playground 的 BraxAutoResetWrapper, 避免 info 结构不兼容)

依赖: mujoco-mjx, jax, mujoco_playground (MjxEnv 基类)
"""
import os
import sys
from typing import Any, Dict, Optional

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx
from flax import struct

from mujoco_playground._src.mjx_env import MjxEnv, State  # noqa: E402

# 物理真源
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import kuafu_physics as P

XML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kuafu.xml")

# ============================================================
# 常量 (从 kuafu_physics 导入, 避免魔数)
# ============================================================
CTRL_DT = 0.02          # 50 Hz 控制频率
SIM_DT = 0.002          # 500 Hz 物理子步
N_SUBSTEPS = int(CTRL_DT / SIM_DT)  # = 10
EPISODE_LENGTH = 1000   # 50 Hz × 20s

# 关节索引 (与 kuafu.xml body 树顺序一致, 见 verify_model.py qpos 核对)
# qpos: root(7) + hip_A_l(1) + knee_A_l(1) + wheel_l(1) + hip_B_l(1) + knee_B_l(1)
#         + hip_A_r(1) + knee_A_r(1) + wheel_r(1) + hip_B_r(1) + knee_B_r(1) = 17
QPOS_HIP_A_L = 7
QPOS_KNEE_A_L = 8
QPOS_WHEEL_L = 9
QPOS_HIP_B_L = 10
QPOS_KNEE_B_L = 11
QPOS_HIP_A_R = 12
QPOS_KNEE_A_R = 13
QPOS_WHEEL_R = 14
QPOS_HIP_B_R = 15
QPOS_KNEE_B_R = 16

# qvel: root_lin(3) + root_ang(3) + joints(10) = 16
# root free joint: qpos 7 (x,y,z,qw,qx,qy,qz) 但 qvel 只 6 (vx,vy,vz,wx,wy,wz)
# 所以 joint 的 qvel idx = qpos idx - 1
QVEL_X = 0
QVEL_PITCH_ANG = 4   # wy (角速度 pitch 分量)
QVEL_HIP_A_L = 6     # qpos 7 - 1
QVEL_KNEE_A_L = 7
QVEL_WHEEL_L = 8     # qpos 9 - 1
QVEL_HIP_B_L = 9     # qpos 10 - 1
QVEL_KNEE_B_L = 10
QVEL_HIP_A_R = 11    # qpos 12 - 1
QVEL_KNEE_A_R = 12
QVEL_WHEEL_R = 13    # qpos 14 - 1
QVEL_HIP_B_R = 14    # qpos 15 - 1
QVEL_KNEE_B_R = 15

# 执行器索引 (actuator 顺序: tau_l, tau_r, q_hip_A_l, q_hip_A_r)
# 对称步态: 仅 hip_A 有舵机, hip_B/knee 由 joint equality 镜像驱动 (见 kuafu.xml equality)
ACT_TAU_L = 0
ACT_TAU_R = 1
ACT_HIP_A_L = 2
ACT_HIP_A_R = 3

# 五杆对称耦合多项式: knee_A = poly(hip_A), 从 MuJoCo 实测 Q_A==Q_B 标定 (deg=4, max_err 0.004rad)
# hip_B = -hip_A, knee_B = -knee_A (镜像)。D0 58→207mm 全程对应 hip_A ∈ [0, -1.52] rad
HIP_STROKE = 1.52    # 曲柄半行程 rad (对称步态 D0 58→207mm)
_KNEE_POLY = jp.array([0.00153, -0.54342, 0.35176, -0.17995, -0.09033])  # [c0,c1,c2,c3,c4]

# 观测维度
# 训练用 4 步堆叠 → 108 维 proprio (RSL-RL ActorCritic 直接消费)
# RMA adapter 需 50 步历史 → 蒸馏时从 obs 序列提取, 见 distill.py
OBS_DIM_BASE = 27     # 9 组本体感受 (对称步态只观测驱动侧 hip_A_l/r)
HISTORY_STEPS = 4
OBS_DIM = OBS_DIM_BASE * HISTORY_STEPS  # 108
PRIVILEGED_DIM = 9    # friction(1)+mass_scale(1)+com_bias(3)+inertia_scale(1)+torque_scale(1)+delay(2)
ACTION_DIM = 4        # [dtau_L, dtau_R, hip_A_l_goal, hip_A_r_goal]
DELAY_BUFFER_LEN = 3   # 最大延迟缓冲 (3步×20ms=60ms, 覆盖 DR_DELAY_ACT=30ms + DR_DELAY_SENSE=20ms)

# 物理常量 (JAX 数组)
LQR_K = jp.array(P.LQR_K)          # [-4.47, -61.18, -5.82, -4.02]
WHEEL_R = P.R                       # 0.03908 m
TAU_WHEEL_RATED = P.TAU_WHEEL_RATED  # 0.55 Nm
TAU_WHEEL_STALL = P.TAU_WHEEL_STALL  # 1.1 Nm
TAU_CONT = P.TAU_CONT               # 1.0 Nm (腿连续安全)
G = P.G                             # 9.81

# DDSM back-EMF: 额定转速下力矩线性衰减
OMEGA_NOLOAD = P.RPM_WHEEL_NOLOAD * 2 * jp.pi / 60  # 315 rpm → rad/s

# 命令范围
V_CMD_RANGE = (-0.5, 0.5)     # m/s (轮缘额定 0.82 m/s, 留余量)
W_CMD_RANGE = (-1.0, 1.0)     # rad/s
D0_CMD_RANGE = (P.D0_MIN, P.D0_MAX)  # (58, 207) mm

# 终止阈值
PITCH_THRESH = jp.radians(45)  # 倒下判定 (硬阈值)
ROLL_THRESH = jp.radians(45)
# 软终止缓冲: 跌倒后不立即终止, 宽限 FALL_GRACE_STEPS 步 (×20ms) 让策略有机会恢复,
# 避免"一碰就死"局部最优 (T1/legged_gym alive 奖励 + 软终止的工程共识)
FALL_GRACE_STEPS = 10  # 10 步 × 20ms = 200ms 宽限

# orientation reward 参数 (exp 包装, 输入为重力向量水平分量平方和)
ORIENT_ALPHA = 8.0  # exp(-alpha * (gx²+gy²)), 直立时 gx²+gy²≈0 → reward≈1

# 域随机化范围 (从 kuafu_physics)
DR = {
    "mass": P.DR_MASS,
    "com": P.DR_COM,
    "inertia": P.DR_INERTIA,
    "friction": P.DR_FRICTION,
    "torque_const": P.DR_TORQUE_CONST,
}


def rotate_vector_by_quaternion_conj(q: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate vector v by conjugate of quaternion q.

    q: (qw, qx, qy, qz), v: (3,)
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    q_xyz = jp.stack([-x, -y, -z])
    uv = jp.cross(q_xyz, v)
    uuv = jp.cross(q_xyz, uv)
    return v + 2 * (w * uv + uuv)


@struct.dataclass
class EnvState:
    """环境内部状态 (不随 obs 暴露给 policy)."""
    rng: jax.Array
    # 命令
    v_cmd: jax.Array
    w_cmd: jax.Array
    d0_cmd: jax.Array
    # 上一步动作 (action_rate reward)
    prev_action: jax.Array
    prev_prev_action: jax.Array
    # 历史观测缓冲 (HISTORY_STEPS × OBS_DIM_BASE)
    obs_history: jax.Array
    # 步数
    step_count: jax.Array
    # 域随机化参数 (per-env)
    friction: jax.Array
    mass_scale: jax.Array
    com_bias: jax.Array
    inertia_scale: jax.Array
    torque_scale: jax.Array
    # deadband (舵机死区, rad)
    deadband: jax.Array
    # delay (执行器延迟, 单位: 控制步数; 1步=20ms)
    delay_steps: jax.Array
    # 动作延迟缓冲 (DELAY_BUFFER_LEN × ACTION_DIM)
    action_buffer: jax.Array
    # 软终止: 连续倒下步数计数 (≥FALL_GRACE_STEPS 才真正终止)
    fall_count: jax.Array


class KuafuMjxEnv(MjxEnv):
    """KUAFU 残差 RL 环境 (MJX GPU 向量化).

    teacher=True 时 obs 返回 dict {"state": ..., "privileged_state": ...},
    供 RSLRLBraxWrapper 自动拆分为 actor/critic 输入。
    teacher=False 时 obs 返回扁平数组 (student / 部署模式)。
    """

    def __init__(
        self,
        teacher: bool = True,
        num_envs: int = 1024,
        episode_length: int = EPISODE_LENGTH,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        from ml_collections import config_dict

        config = config_dict.ConfigDict()
        config.ctrl_dt = CTRL_DT
        config.sim_dt = SIM_DT
        super().__init__(config, config_overrides)

        self._teacher = teacher
        self._num_envs = num_envs
        self._episode_length = episode_length

        # 加载模型
        self._mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        self._mjx_model = mjx.put_model(self._mj_model)

        # keyframe 0 (dwell) 的 qpos
        self._keyframe_qpos = self._mj_model.qpos0.copy()

    # ---- MjxEnv 抽象属性 ----
    @property
    def xml_path(self) -> str:
        return XML_PATH

    @property
    def action_size(self) -> int:
        return ACTION_DIM

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def num_envs(self) -> int:
        return self._num_envs

    # ---- 域随机化 ----
    def _randomize_model(self, model: mjx.Model, rng: jax.Array) -> mjx.Model:
        """对 mjx.Model 注入域随机化 (per-env).

        在 reset 时调用, 对 mass/friction/inertia/COM/wheel_radius/servo_pd 注入随机扰动。
        """
        # mass ±15%
        mass_scale = jax.random.uniform(
            rng, minval=DR["mass"][0], maxval=DR["mass"][1])
        model = model.replace(body_mass=model.body_mass * mass_scale)

        # friction [0.3, 1.2]
        rng, friction_rng = jax.random.split(rng)
        friction = jax.random.uniform(
            friction_rng, minval=DR["friction"][0], maxval=DR["friction"][1])
        geom_friction = model.geom_friction.at[:, 0].set(
            model.geom_friction[:, 0] * friction)
        model = model.replace(geom_friction=geom_friction)

        # inertia ×[0.5, 2.0]
        rng, inertia_rng = jax.random.split(rng)
        inertia_scale = jax.random.uniform(
            inertia_rng, minval=DR["inertia"][0], maxval=DR["inertia"][1])
        diaginertia = model.body_inertia * inertia_scale
        model = model.replace(body_inertia=diaginertia)

        # COM 偏移 ±20mm (注入到 chassis body_ipos)
        rng, com_rng = jax.random.split(rng)
        com_bias = jax.random.uniform(
            com_rng, (3,), minval=P.DR_COM[0], maxval=P.DR_COM[1])
        # chassis 是 body 1, 修改其 inertial pos
        new_ipos = model.body_ipos.at[1].set(model.body_ipos[1] + com_bias)
        model = model.replace(body_ipos=new_ipos)

        # wheel_radius ±1mm (修改轮 geom size[0])
        rng, wr_rng = jax.random.split(rng)
        wheel_r_delta = jax.random.uniform(
            wr_rng, minval=P.DR_WHEEL_R[0], maxval=P.DR_WHEEL_R[1])
        # 轮 geom 是 wheel_l(idx 4) 和 wheel_r(idx 9), size[0]=半径
        geom_size = model.geom_size
        geom_size = geom_size.at[4, 0].set(geom_size[4, 0] + wheel_r_delta)
        geom_size = geom_size.at[9, 0].set(geom_size[9, 0] + wheel_r_delta)
        model = model.replace(geom_size=geom_size)

        # servo_pd ±30% (修改 position actuator 的 gainprm[0]=kp, biasprm[1]=-kp*d, biasprm[2]=-kv)
        rng, pd_rng = jax.random.split(rng)
        pd_scale = jax.random.uniform(
            pd_rng, minval=P.DR_SERVO_PD[0], maxval=P.DR_SERVO_PD[1])
        # position actuator idx: 2,3 (q_hip_A_l, q_hip_A_r; hip_B 由 equality 镜像, 无 actuator)
        # gainprm[0] = kp, biasprm[2] = -kv (MuJoCo position actuator: gain=fixed kp, bias=affine -kp*q0 -kv*vel)
        gainprm = model.actuator_gainprm
        biasprm = model.actuator_biasprm
        for i in [ACT_HIP_A_L, ACT_HIP_A_R]:
            gainprm = gainprm.at[i, 0].set(gainprm[i, 0] * pd_scale)
            biasprm = biasprm.at[i, 0].set(biasprm[i, 0] * pd_scale)  # -kp*q0
            biasprm = biasprm.at[i, 2].set(biasprm[i, 2] * pd_scale)  # -kv
        model = model.replace(actuator_gainprm=gainprm, actuator_biasprm=biasprm)

        return model, friction, mass_scale, inertia_scale, com_bias

    # ---- 命令采样 ----
    def _sample_command(self, rng: jax.Array):
        """随机采样速度/角速度/D0 命令."""
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        v_cmd = jax.random.uniform(k1, minval=V_CMD_RANGE[0], maxval=V_CMD_RANGE[1])
        w_cmd = jax.random.uniform(k2, minval=W_CMD_RANGE[0], maxval=W_CMD_RANGE[1])
        d0_cmd = jax.random.uniform(k3, minval=D0_CMD_RANGE[0], maxval=D0_CMD_RANGE[1])
        # 10% 概率零命令 (静止平衡)
        rng, k_zero = jax.random.split(rng)
        is_zero = jax.random.bernoulli(k_zero, p=0.1)
        v_cmd = jp.where(is_zero, 0.0, v_cmd)
        w_cmd = jp.where(is_zero, 0.0, w_cmd)
        return rng, v_cmd, w_cmd, d0_cmd

    # ---- LQR 底层 ----
    def _lqr_balance(self, data: mjx.Data) -> jax.Array:
        """LQR 轮式倒立摆平衡: 输出每轮力矩.

        使用本体感受投影：
        - xdot: 本体坐标系前向速度 (vx_local)
        - theta: 本体坐标系 Pitch 角 (解耦后的 arcsin)
        - thetadot: 本体坐标系 Pitch 角速度 (wy_local)
        """
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        q = jp.stack([qw, qx, qy, qz])

        # 本体系线速度
        lin_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[0:3])
        xdot = lin_vel_local[0]

        # 本体系角速度
        ang_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
        thetadot = ang_vel_local[1]  # wy

        # 本体系 Pitch (解耦)
        theta = jp.arcsin(jp.clip(2 * (qw * qy - qx * qz), -0.999999, 0.999999))

        state = jp.stack([0.0, theta, xdot, thetadot])
        F = -(LQR_K @ state)
        tau = F * WHEEL_R / 2.0
        return tau

    # ---- DDSM back-EMF 限制 ----
    def _limit_wheel_torque(self, tau: jax.Array, omega: jax.Array) -> jax.Array:
        """DDSM 力矩-转速限制: τ_avail = τ_stall × (1 - |ω|/ω_noload)."""
        tau_avail = TAU_WHEEL_STALL * (1.0 - jp.abs(omega) / OMEGA_NOLOAD)
        tau_avail = jp.clip(tau_avail, 0.0, TAU_WHEEL_STALL)
        return jp.clip(tau, -tau_avail, tau_avail)

    # ---- 观测 ----
    def _base_observation(self, data: mjx.Data, env_state: EnvState) -> jax.Array:
        """27 维本体感受观测 (对称步态: 只观测驱动侧 hip_A_l/r)."""
        # 机身姿态 (3): roll/pitch/yaw from quaternion (Pitch 采用 arcsin)
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        roll = jp.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = jp.arcsin(jp.clip(2 * (qw * qy - qx * qz), -0.999999, 0.999999))
        yaw = jp.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz**2 + qy**2))
        attitude = jp.stack([roll, pitch, yaw])

        # 角速度 (3): 本体系下的 wx/wy/wz
        q = jp.stack([qw, qx, qy, qz])
        ang_vel = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])

        # 轮状态 (4): 左右轮位置+速度
        wheel_pos = jp.stack([data.qpos[QPOS_WHEEL_L], data.qpos[QPOS_WHEEL_R]])
        wheel_vel = jp.stack([data.qvel[QVEL_WHEEL_L], data.qvel[QVEL_WHEEL_R]])
        wheel_state = jp.concatenate([wheel_pos, wheel_vel])

        # 髋关节状态 (4): 驱动侧 2 舵机位置+速度 (hip_B 由对称耦合确定, 不冗余观测)
        hip_pos = jp.stack([data.qpos[QPOS_HIP_A_L], data.qpos[QPOS_HIP_A_R]])
        hip_vel = jp.stack([data.qvel[QVEL_HIP_A_L], data.qvel[QVEL_HIP_A_R]])
        hip_state = jp.concatenate([hip_pos, hip_vel])

        # 轮力矩观测 (2): 执行器力 (actuator_force)
        wheel_torque = jp.array([
            data.actuator_force[ACT_TAU_L], data.actuator_force[ACT_TAU_R]
        ])

        # 腿力矩观测 (2): 驱动侧舵机电流→力矩代理
        hip_torque = jp.array([
            data.actuator_force[ACT_HIP_A_L], data.actuator_force[ACT_HIP_A_R],
        ])

        # 上一步动作 (4)
        last_action = env_state.prev_action

        # 命令 (3): v_cmd, w_cmd, d0_cmd
        command = jp.stack([env_state.v_cmd, env_state.w_cmd, env_state.d0_cmd])

        # 相位时钟 (2): sin/cos (当前步数 / 周期)
        phase = env_state.step_count.astype(jp.float32) / EPISODE_LENGTH
        phase_clock = jp.stack([jp.sin(2 * jp.pi * phase), jp.cos(2 * jp.pi * phase)])

        return jp.concatenate([
            attitude,        # 3
            ang_vel,         # 3
            wheel_state,     # 4
            hip_state,       # 4
            wheel_torque,    # 2
            hip_torque,      # 2
            last_action,     # 4
            command,         # 3
            phase_clock,     # 2
        ])                    # = 27

    def _privileged_observation(self, env_state: EnvState) -> jax.Array:
        """9 维特权观测 (teacher only)."""
        return jp.concatenate([
            env_state.friction[jp.newaxis],      # 1
            env_state.mass_scale[jp.newaxis],     # 1
            env_state.com_bias,                   # 3
            env_state.inertia_scale[jp.newaxis],  # 1
            env_state.torque_scale[jp.newaxis],   # 1
            env_state.deadband[jp.newaxis],       # 1 死区真值
            env_state.delay_steps.astype(jp.float32)[jp.newaxis],  # 1 延迟步数真值
        ])

    def _observation(self, data: mjx.Data, env_state: EnvState):
        """构建观测: teacher 返回 dict, student 返回扁平数组.

        obs_history 已在 step() 中更新 (含最新帧), 这里只 reshape。
        """
        flat_obs = env_state.obs_history.reshape(-1)    # (140,)

        if self._teacher:
            priv = self._privileged_observation(env_state)  # (9,)
            return {"state": flat_obs, "privileged_state": priv}
        return flat_obs

    # ---- Reward ----
    def _compute_reward(self, data: mjx.Data, env_state: EnvState, action: jax.Array) -> jax.Array:
        """task + style + safety reward (各项未乘 dt, step 中统一乘 CTRL_DT).

        设计参照 MuJoCo Playground (Go1/T1 轮式人形/Berkeley Humanoid) 源码:
        - tracking 用 exp(-error²/σ) 核, σ=0.25
        - orientation 用重力向量水平分量 (无欧拉角万向锁歧义), exp 包装保 [0,1] 正奖励
        - alive 存活奖励对抗过早终止 (配合软终止缓冲)
        """
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        q = jp.stack([qw, qx, qy, qz])

        # --- task (正向) ---
        # orientation: 重力向量在本体系的投影, 水平分量平方和 (gx²+gy²)
        # 直立时重力沿 -z → gx²+gy²≈0 → exp≈1; 倾倒时水平分量增大 → exp→0
        # 无欧拉角分解, 物理连续无歧义 (MIT Cheetah / Unitree 工业共识)
        grav_local = rotate_vector_by_quaternion_conj(q, jp.array([0.0, 0.0, -1.0]))
        orientation = jp.exp(-ORIENT_ALPHA * jp.sum(grav_local[:2] ** 2))

        # lin_vel_tracking: 跟踪 v_cmd (本体系前向速度 xdot)
        lin_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[0:3])
        xdot = lin_vel_local[0]
        lin_vel_error = (xdot - env_state.v_cmd) ** 2
        lin_vel_tracking = jp.exp(-lin_vel_error / 0.25)

        # ang_vel_tracking: 跟踪 w_cmd (yaw 角速度, 本体系 wz)
        ang_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
        yaw_rate = ang_vel_local[2]  # wz
        ang_vel_error = (yaw_rate - env_state.w_cmd) ** 2
        ang_vel_tracking = jp.exp(-ang_vel_error / 0.25)

        # default_pose: 跟踪 d0_cmd (关节空间正则)。对称步态下驱动曲柄 hip_A 单调编码 D0
        # d0_cmd=dwell 时 hip_target=0, 即惩罚偏离默认站立姿态 (joint pose regularization)
        # (hip_A ∈ [0, -1.52] ↔ D0 ∈ [58, 207]mm, 标定见 _KNEE_POLY)
        d0_norm = (env_state.d0_cmd - P.D0_MIN) / (P.D0_MAX - P.D0_MIN)
        hip_target = -d0_norm * HIP_STROKE          # 目标 hip_A 角 (伸展为负)
        hip_actual = (data.qpos[QPOS_HIP_A_L] + data.qpos[QPOS_HIP_A_R]) / 2.0
        default_pose = jp.exp(-((hip_actual - hip_target) ** 2) / 0.1)

        # alive: 存活奖励, 对抗训练初期"一碰就死"的局部最优 (T1 轮式用 0.25, KUAFU 有
        # 坚实 orientation+tracking 故降至 0.1; 配合 FALL_GRACE_STEPS 软终止)
        alive = 1.0

        # --- style (负向惩罚) ---
        # action_rate: -‖a_t - a_{t-1}‖² (一阶, Go1/T1 不用二阶 jerk)
        action_rate = -jp.sum((action - env_state.prev_action) ** 2)

        # energy: -Σ|ω·τ| 全驱动关节 (轮 + 髋, Go1 _cost_energy 形式)
        energy = -(
            jp.abs(data.qvel[QVEL_WHEEL_L] * data.actuator_force[ACT_TAU_L])
            + jp.abs(data.qvel[QVEL_WHEEL_R] * data.actuator_force[ACT_TAU_R])
            + jp.abs(data.qvel[QVEL_HIP_A_L] * data.actuator_force[ACT_HIP_A_L])
            + jp.abs(data.qvel[QVEL_HIP_A_R] * data.actuator_force[ACT_HIP_A_R]))

        # torque_limit: 超连续安全扭矩惩罚 (仅驱动侧 2 舵机)
        tau_excess = jp.maximum(jp.abs(data.actuator_force[ACT_HIP_A_L]) - TAU_CONT, 0) \
            + jp.maximum(jp.abs(data.actuator_force[ACT_HIP_A_R]) - TAU_CONT, 0)
        torque_limit = -tau_excess

        total = (
            1.0 * lin_vel_tracking
            + 0.5 * ang_vel_tracking
            + 1.0 * orientation
            + 0.3 * default_pose
            + 0.1 * alive
            + 0.01 * action_rate
            + 0.001 * energy
            + 0.5 * torque_limit
        )
        return total

    # ---- 终止 ----
    def _is_fallen(self, data: mjx.Data) -> jax.Array:
        """硬倒下判定 (pitch/roll 超阈值). 用于 fall_count 累加 + alive 门控."""
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        # 重力向量判定 (与 orientation reward 一致, 无欧拉角歧义)
        q = jp.stack([qw, qx, qy, qz])
        grav_local = rotate_vector_by_quaternion_conj(q, jp.array([0.0, 0.0, -1.0]))
        # 直立时 grav_local≈(0,0,-1), z 分量 < cos(45°)≈0.707 即倾倒超 45°
        fallen = grav_local[2] > -jp.cos(PITCH_THRESH)
        return fallen

    def _is_done(self, data: mjx.Data, env_state: EnvState) -> jax.Array:
        """软终止: 连续倒下 ≥ FALL_GRACE_STEPS 或超时.

        跌倒后宽限 10 步 (200ms) 让策略有机会恢复, 避免训练初期"一碰就死"。
        (与 alive=0.1 奖励配合, 鼓励长时平衡)
        """
        fallen = self._is_fallen(data)
        # fall_count: 倒下时累加, 恢复时清零
        new_fall_count = jp.where(fallen, env_state.fall_count + 1, 0)
        hard_fall = new_fall_count >= FALL_GRACE_STEPS
        timeout = env_state.step_count >= self._episode_length
        return hard_fall | timeout

    # ============================================================
    # MjxEnv 接口实现
    # ============================================================
    def reset(self, rng: jax.Array) -> State:
        """重置环境到驻留态 + 域随机化 + 命令采样."""
        rng, cmd_rng, dr_rng, torque_rng, db_rng, delay_rng = jax.random.split(rng, 6)

        # 域随机化 (mass/friction/inertia/COM/wheel_r/servo_pd 注入物理)
        model, friction, mass_scale, inertia_scale, com_bias = self._randomize_model(
            self._mjx_model, dr_rng)
        torque_scale = jax.random.uniform(
            torque_rng, minval=DR["torque_const"][0], maxval=DR["torque_const"][1])

        # deadband [0, 2°] (舵机齿轮间隙)
        deadband = jax.random.uniform(
            db_rng, minval=P.DR_DEADBAND[0], maxval=P.DR_DEADBAND[1])

        # delay [0, 30ms] → 步数 (1步=20ms, 最大 1-2 步)
        delay_s = jax.random.uniform(
            delay_rng, minval=P.DR_DELAY_ACT[0], maxval=P.DR_DELAY_ACT[1])
        delay_steps = jp.round(delay_s / CTRL_DT).astype(jp.int32)

        # 命令
        cmd_rng, v_cmd, w_cmd, d0_cmd = self._sample_command(cmd_rng)

        # 初始 data (keyframe dwell + 小扰动)
        data = mjx.make_data(model)
        qpos0 = jp.asarray(self._keyframe_qpos)
        rng, noise_rng = jax.random.split(rng)
        # 关节角小噪声 (±0.05 rad ≈ ±3°)
        joint_noise = jax.random.uniform(
            noise_rng, (model.nq - 7,), minval=-0.05, maxval=0.05)
        qpos0 = qpos0.at[7:].set(qpos0[7:] + joint_noise)
        data = data.replace(qpos=qpos0)
        data = mjx.forward(model, data)

        # 初始状态
        env_state = EnvState(
            rng=rng,
            v_cmd=v_cmd,
            w_cmd=w_cmd,
            d0_cmd=d0_cmd,
            prev_action=jp.zeros(ACTION_DIM),
            prev_prev_action=jp.zeros(ACTION_DIM),
            obs_history=jp.zeros((HISTORY_STEPS, OBS_DIM_BASE)),  # 初始全 0, 首帧 step 后填充

            step_count=jp.int32(0),
            friction=friction,
            mass_scale=mass_scale,
            com_bias=com_bias,
            inertia_scale=inertia_scale,
            torque_scale=torque_scale,
            deadband=deadband,
            delay_steps=delay_steps,
            action_buffer=jp.zeros((DELAY_BUFFER_LEN, ACTION_DIM)),
            fall_count=jp.int32(0),
        )

        obs = self._observation(data, env_state)
        reward = jp.float32(0.0)
        done = jp.bool_(False)
        metrics = {
            "upright": jp.float32(0.0),
            "lin_vel_tracking": jp.float32(0.0),
            "fallen": jp.float32(0.0),
        }
        info = {"env_state": env_state, "model": model}
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """执行一步控制: LQR + RL 残差叠加 → 10 子步物理 → reward/obs/done."""
        data = state.data
        env_state = state.info["env_state"]
        model = state.info["model"]

        # ---- 执行器延迟: 从 action_buffer 取延迟前的动作 ----
        # action_buffer shape: (DELAY_BUFFER_LEN, ACTION_DIM), [0]=最旧(3步前), [1]=2步前, [2]=1步前
        # delay_steps=0 → 用当前 action; delay_steps=1 → 用 1步前; delay_steps=2 → 用 2步前
        delayed_action = jp.where(
            env_state.delay_steps >= 2,
            env_state.action_buffer[1],  # 2 步前
            jp.where(
                env_state.delay_steps >= 1,
                env_state.action_buffer[2],  # 1 步前
                action,                      # 当前
            )
        )
        # 更新 action_buffer: 推入当前 action, 丢弃最旧
        new_action_buffer = jp.concatenate([
            env_state.action_buffer[1:],
            action[jp.newaxis, :],
        ], axis=0)

        # ---- LQR 底层 (torque_scale 模拟电机常数偏差, 对 LQR+RL 统一生效) ----
        tau_lqr = self._lqr_balance(data) * env_state.torque_scale

        # ---- RL 残差叠加 (用延迟后的动作) ----
        tau_wheel_l = tau_lqr + delayed_action[0] * TAU_WHEEL_RATED * env_state.torque_scale
        tau_wheel_r = tau_lqr + delayed_action[1] * TAU_WHEEL_RATED * env_state.torque_scale
        tau_wheel_l = self._limit_wheel_torque(
            jp.clip(tau_wheel_l, -TAU_WHEEL_STALL, TAU_WHEEL_STALL),
            data.qvel[QVEL_WHEEL_L])
        tau_wheel_r = self._limit_wheel_torque(
            jp.clip(tau_wheel_r, -TAU_WHEEL_STALL, TAU_WHEEL_STALL),
            data.qvel[QVEL_WHEEL_R])

        # 腿: 位置目标 (q=0 = 驻留态, delayed_action±1 → ±HIP_STROKE rad)
        # 对称步态: action[2]=hip_A_l 目标, action[3]=hip_A_r 目标 (hip_B 由 equality 镜像)
        hip_goal = delayed_action[2:4] * HIP_STROKE

        # 舵机死区: |hip_goal| < deadband 时输出 0 (齿轮间隙)
        hip_goal = jp.where(jp.abs(hip_goal) < env_state.deadband, 0.0, hip_goal)

        # 组装 ctrl (用随机化后的 model, 非 self._mjx_model; nu=4: tau_l, tau_r, q_hip_A_l, q_hip_A_r)
        ctrl = jp.zeros(model.nu)
        ctrl = ctrl.at[ACT_TAU_L].set(tau_wheel_l)
        ctrl = ctrl.at[ACT_TAU_R].set(tau_wheel_r)
        ctrl = ctrl.at[ACT_HIP_A_L].set(hip_goal[0])
        ctrl = ctrl.at[ACT_HIP_A_R].set(hip_goal[1])
        data = data.replace(ctrl=ctrl)

        # ---- 物理步进 (10 子步) ----
        for _ in range(N_SUBSTEPS):
            data = mjx.step(model, data)
            # 强制对称耦合 follower (权威运动学, 防止约束求解器漂移到错误分支)
            # hip_B = -hip_A, knee_A = poly(hip_A), knee_B = -knee_A
            hA_l = data.qpos[QPOS_HIP_A_L]; hA_r = data.qpos[QPOS_HIP_A_R]
            dhA_l = data.qvel[QVEL_HIP_A_L]; dhA_r = data.qvel[QVEL_HIP_A_R]
            kA_l = jp.polyval(_KNEE_POLY[::-1], hA_l)   # jp.polyval 高阶在前
            kA_r = jp.polyval(_KNEE_POLY[::-1], hA_r)
            # knee 对 hip_A 的导数 (链法则用于 qvel)
            dkA_l = jp.polyval((_KNEE_POLY[1:]*jp.arange(1,5))[::-1], hA_l)
            dkA_r = jp.polyval((_KNEE_POLY[1:]*jp.arange(1,5))[::-1], hA_r)
            # 链式 .at[].set() 更新多索引 (单 qpos/qvel 参数内串接)
            qpos = (data.qpos
                    .at[QPOS_HIP_B_L].set(-hA_l)
                    .at[QPOS_HIP_B_R].set(-hA_r)
                    .at[QPOS_KNEE_A_L].set(kA_l)
                    .at[QPOS_KNEE_A_R].set(kA_r)
                    .at[QPOS_KNEE_B_L].set(-kA_l)
                    .at[QPOS_KNEE_B_R].set(-kA_r))
            qvel = (data.qvel
                    .at[QVEL_HIP_B_L].set(-dhA_l)
                    .at[QVEL_HIP_B_R].set(-dhA_r)
                    .at[QVEL_KNEE_A_L].set(dkA_l*dhA_l)
                    .at[QVEL_KNEE_A_R].set(dkA_r*dhA_r)
                    .at[QVEL_KNEE_B_L].set(-dkA_l*dhA_l)
                    .at[QVEL_KNEE_B_R].set(-dkA_r*dhA_r))
            data = data.replace(qpos=qpos, qvel=qvel)

        # ---- 更新状态 ----
        rng, resample_rng = jax.random.split(env_state.rng)
        # 每 100 步重采样命令 (2s @ 50Hz)
        need_resample = (env_state.step_count % 100 == 0) & (env_state.step_count > 0)
        rng, k1, k2, k3 = jax.random.split(resample_rng, 4)
        new_v = jax.random.uniform(k1, minval=V_CMD_RANGE[0], maxval=V_CMD_RANGE[1])
        new_w = jax.random.uniform(k2, minval=W_CMD_RANGE[0], maxval=W_CMD_RANGE[1])
        new_d0 = jax.random.uniform(k3, minval=D0_CMD_RANGE[0], maxval=D0_CMD_RANGE[1])
        v_cmd = jp.where(need_resample, new_v, env_state.v_cmd)
        w_cmd = jp.where(need_resample, new_w, env_state.w_cmd)
        d0_cmd = jp.where(need_resample, new_d0, env_state.d0_cmd)

        # ---- reward / done / obs (用旧 env_state, 因 reward 需要 prev_action) ----
        raw_reward = self._compute_reward(data, env_state, action)
        fallen = self._is_fallen(data)

        # fall_count 软终止计数 (倒下累加, 恢复清零)
        new_fall_count = jp.where(fallen, env_state.fall_count + 1, 0)
        done = (new_fall_count >= FALL_GRACE_STEPS) | (
            env_state.step_count >= self._episode_length)

        # reward × CTRL_DT (Go1/T1 标准: 每项 ×scale 后乘 dt, 保持 PPO value 尺度一致)
        # 软终止期间不额外惩罚 (靠 episode 结束截断后续 reward + alive 门控)
        reward = raw_reward * CTRL_DT

        # 更新 env_state (reward 算完后再更新 prev_action)
        env_state = env_state.replace(
            rng=rng,
            v_cmd=v_cmd,
            w_cmd=w_cmd,
            d0_cmd=d0_cmd,
            prev_prev_action=env_state.prev_action,
            prev_action=action,
            step_count=env_state.step_count + 1,
            action_buffer=new_action_buffer,
            fall_count=new_fall_count,
        )

        # 更新 obs history (append 当前帧, 丢弃最老帧)
        base_obs = self._base_observation(data, env_state)
        new_history = jp.concatenate([
            env_state.obs_history[1:],
            base_obs[jp.newaxis, :],
        ], axis=0)
        env_state = env_state.replace(obs_history=new_history)

        # lin_vel_tracking / orientation 实际值 (供 metric 记录)
        q = jp.stack([data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]])
        lin_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[0:3])
        xdot = lin_vel_local[0]
        lin_vel_track_val = jp.exp(-((xdot - v_cmd) ** 2) / 0.25)
        grav_local = rotate_vector_by_quaternion_conj(q, jp.array([0.0, 0.0, -1.0]))
        orient_val = jp.exp(-ORIENT_ALPHA * jp.sum(grav_local[:2] ** 2))

        obs = self._observation(data, env_state)
        # terminated_by_fall: done 且因连续倒下触发 (供 train.py time_outs 区分 timeout vs fall)
        terminated_by_fall = (new_fall_count >= FALL_GRACE_STEPS).astype(jp.float32)
        metrics = {
            "orientation": orient_val,
            "lin_vel_tracking": lin_vel_track_val,
            "fallen": terminated_by_fall,  # done 帧的终止原因 (1=摔倒终止, 0=超时)
        }
        info = {"env_state": env_state, "model": model}
        return state.replace(
            data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)


def make_env(teacher: bool = True, num_envs: int = 1024, **kwargs) -> KuafuMjxEnv:
    """工厂函数: 创建 KUAFU MJX 环境."""
    return KuafuMjxEnv(teacher=teacher, num_envs=num_envs, **kwargs)
