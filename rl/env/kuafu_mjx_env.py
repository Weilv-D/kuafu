# -*- coding: utf-8 -*-
"""
KUAFU 残差 RL 环境 — MuJoCo MJX 实现

继承 MuJoCo Playground MjxEnv, JAX 全函数化, 向量化运行在 GPU 上。
驻留态腿被动自锁, 整机降为轮式倒立摆; RL 输出残差叠加在 LQR 底层之上。

Architecture is defined by rl.env.contract and docs/architecture/system.md.

通过 train.py 的 DirectVecEnv 适配器桥接到 PyTorch/RSL-RL 2.x:
  DLPack 零拷贝 (JAX DeviceArray ↔ torch.Tensor), JAX cuda13 与 torch cu130 共享 runtime。
  (绕过 playground 的 BraxAutoResetWrapper, 避免 info 结构不兼容)

依赖: mujoco-mjx, jax, mujoco_playground (MjxEnv 基类)
"""
import os
import sys
import json
from typing import Any, Dict, Optional, Tuple

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
IK_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fivebar_ik_table.json")

# ============================================================
# 常量 (从 kuafu_physics 导入, 避免魔数)
# ============================================================
CTRL_DT = P.RL_DT       # 50 Hz 控制频率
SIM_DT = P.PHYS_DT      # 500 Hz 物理子步
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

# 执行器索引 (actuator 顺序: tau_l, tau_r, q_hip_A_l, q_hip_A_r, q_hip_B_l, q_hip_B_r)
# 2-DOF 五杆: 4 个舵机 (hip_A + hip_B 各左右) 全部独立位置控制, 对齐真机
ACT_TAU_L = 0
ACT_TAU_R = 1
ACT_HIP_A_L = 2
ACT_HIP_A_R = 3
ACT_HIP_B_L = 4
ACT_HIP_B_R = 5

# RL 腿残差的工作空间投影: action∈[-1,1] → 有界 Qx/D0 位移 (mm)。
# 两个分量分别进入二维五杆 IK，禁止再相加后伪装成 D0 残差。
D0_RESIDUAL_SCALE = P.D0_RESIDUAL_SCALE
QX_RESIDUAL_SCALE = P.QX_RESIDUAL_SCALE

# 观测维度
# 35维实机可观测帧，四步因果历史。
OBS_DIM_BASE = 35    # 仅实机可观测本体感受
HISTORY_STEPS = 4
OBS_DIM = OBS_DIM_BASE * HISTORY_STEPS  # 140 本体感受 (4步因果历史)
PRIVILEGED_DIM = 12    # teacher critic 特权 = 静态外因(9) + 瞬态扰动(3)
# Critic-only 特权：静态环境外因 9 维与瞬态推力 3 维。
RMA_STATIC_DIM = 9    # friction(1)+mass_scale(1)+com_bias(3)+inertia_scale(1)+torque_scale(1)+deadband(1)+delay_steps(1)
TRANSIENT_DIM = 3     # active_push(3) 瞬态外力
ACTION_DIM = 6        # [dtau_common, dtau_yaw, dQx_L, dD0_L, dQx_R, dD0_R]
DELAY_BUFFER_LEN = 3   # 最大延迟缓冲 (3步×20ms=60ms, 覆盖 DR_DELAY_ACT=30ms + DR_DELAY_SENSE=20ms)
DIFF_COMMAND = 0
DIFF_DR = 1
DIFF_TERRAIN = 2
DIFF_PUSH = 3
DIFF_D0 = 4
DIFFICULTY_DIM = 5

# Actor 只消费本体感受；Critic 追加仿真特权。
PROPRIO_DIM = OBS_DIM
Z_DIM = 0
ACTOR_OBS_DIM = PROPRIO_DIM
CRITIC_PRIV_DIM = PRIVILEGED_DIM
CRITIC_OBS_DIM = ACTOR_OBS_DIM + CRITIC_PRIV_DIM

# ---- 规范基层控制常量（与 kuafu_physics 单源一致）----
# 多速率：物理 500Hz / 基层 250Hz / RL 残差 50Hz
BASE_DT = P.BASE_DT                 # 4ms 基层控制周期
PHYS_DT = P.PHYS_DT                 # 2ms 物理周期
RL_DT = P.RL_DT                     # 20ms RL 周期
BASE_STEPS_PER_RL = P.BASE_STEPS_PER_RL      # 5
PHYS_SUBSTEPS_PER_BASE = P.PHYS_SUBSTEPS_PER_BASE  # 2
# 规范合成增益（禁止手填）：离散 LQR @250Hz + LQI 积分增益（消除位置稳态漂移）
LQR_K_DT4 = P.LQR_K_DT4
LQI_KI = float(P.LQI_KI_DT4)
WHEEL_R = P.R_WHEEL * P.MM          # 轮半径 m
# yaw 符号：τ_yaw=(τR-τL)/2 ⇒ 右轮>左轮 ⇒ +wz（左转），见 rl/env/contract.py

# 物理常量 (JAX 数组)
WHEEL_R = P.R                       # 0.03908 m
TAU_WHEEL_RATED = P.TAU_WHEEL_RATED  # 0.55 Nm
TAU_WHEEL_STALL = P.TAU_WHEEL_STALL  # 1.1 Nm
TAU_CONT = P.TAU_CONT               # 1.0 Nm (腿连续安全)
G = P.G                             # 9.81

# P4 动作/工作空间惩罚 (防止对抗策略靠静止/饱和/顶限位通过, audit P0/P4)
HIP_RANGE = 3.3                     # 髋关节硬限位 (rad, 已放宽以达 D0_MAX)
WORKSPACE_SAFE = 2.8                # 工作空间安全边界 (rad): 超出即惩罚, 留余量给限位
RESID_W = 0.02                      # 残差幅度惩罚权重 (鼓励贴近基层控制器)
SAT_W = 0.1                         # tanh 饱和惩罚权重 (|a|>0.9 时)
WORKSPACE_W = 0.05                  # 工作空间越界惩罚权重

# DDSM back-EMF: 额定转速下力矩线性衰减
OMEGA_NOLOAD = P.RPM_WHEEL_NOLOAD * 2 * jp.pi / 60  # 315 rpm → rad/s

# 基层 heading/roll 增益（与 kuafu_physics/STM32 同源）
YAW_KD = P.YAW_KD
YAW_KP = P.YAW_KP
ROLL_KP = P.ROLL_KP     # roll 调平比例 (mm/rad)
ROLL_KD = P.ROLL_KD     # roll 阻尼

# 命令范围
V_CMD_RANGE = (-0.5, 0.5)     # m/s (轮缘额定 0.82 m/s, 留余量)
W_CMD_RANGE = (-1.0, 1.0)     # rad/s
D0_CMD_RANGE = (P.D0_MIN, P.D0_MAX)  # (58, 207) mm

# 终止阈值 (与评估/回放统一为 30°, 见 eval_policy.py PITCH_THRESH;
# 物理可恢复俯仰 ~25° (KUAFU.md), 留 5° 余量, 消除 25°~45° 不可恢复却未终止的死区)
PITCH_THRESH = jp.radians(30)  # 倒下判定 (硬阈值)
ROLL_THRESH = jp.radians(30)
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
    # 基层位置跟踪参考（LQR/LQI）：x_ref 积分自 v_cmd；命令归零即冻结→原地位置保持
    x_ref: jax.Array
    x_int: jax.Array
    yaw_ref: jax.Array
    v_ref: jax.Array
    v_accel: jax.Array
    w_ref: jax.Array
    w_accel: jax.Array
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
    # independent sensor/compute delay used by the Actor observation
    sense_delay_steps: jax.Array
    # 动作延迟缓冲 (DELAY_BUFFER_LEN × ACTION_DIM)
    action_buffer: jax.Array
    # 观测延迟缓冲 (DELAY_BUFFER_LEN × OBS_DIM) — 传感器/计算延迟 DR (latency randomization)
    obs_delay_buffer: jax.Array
    # 软终止: 连续倒下步数计数 (≥FALL_GRACE_STEPS 才真正终止)
    fall_count: jax.Array
    # 课程系统参数 (per-env)
    difficulty: jax.Array
    # 外部推力扰动向量 (3,)
    push_force: jax.Array
    # 实际施加的推力向量 (3,)
    active_push: jax.Array
    # Episode aggregates used by curriculum gates; terminal-frame metrics cannot
    # distinguish a policy that tracked for two seconds from one that got lucky.
    track_v_abs_sum: jax.Array
    track_w_abs_sum: jax.Array
    track_d0_abs_sum: jax.Array
    track_count: jax.Array
    nonzero_command_count: jax.Array


class KuafuMjxEnv(MjxEnv):
    """KUAFU 残差 RL 环境 (MJX GPU 向量化).

    teacher=True 时 obs 返回 dict {"state": ..., "privileged_state": ...},
    供 RSLRLBraxWrapper 自动拆分为 actor/critic 输入。
    teacher=False 时 obs 返回阻马/动作历史 (student / 部署模式)。
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

        # 五杆二维 IK 查表 (JAX 安全)。校准 artifact 是部署契约的一部分，
        # 不在环境构造时重新生成未校验的几何表。
        with open(IK_TABLE_PATH, encoding="utf-8") as source:
            _ik_grid = json.load(source)
        if _ik_grid.get("model_hash") != P.model_hash():
            raise RuntimeError("five-bar calibration table model hash mismatch")
        self._ik_qx = jp.asarray(_ik_grid["qx"])
        self._ik_d0 = jp.asarray(_ik_grid["d0"])
        self._ik_qA = jp.asarray(_ik_grid["qA_grid"])
        self._ik_qB = jp.asarray(_ik_grid["qB_grid"])
        dwell = P.fivebar_ik(P.D0_MIN)
        self._dwell_qA = jp.asarray(dwell[0])
        self._dwell_qB = jp.asarray(dwell[1])

        # 加载模型
        self._mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        self._mjx_model = mjx.put_model(self._mj_model)

        # 获取机身 body ID 供 xfrc_applied 注入
        self._chassis_body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self._wheel_l_geom_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, "wheel_l_geom")
        self._wheel_r_geom_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, "wheel_r_geom")

        # 地形几何 ID (M4 台阶/斜坡, 由 _apply_terrain 按 difficulty 缩放)
        # 注: MJX 不支持 heightfield×cylinder 碰撞, 故地形用倾斜平面(斜坡) +
        # 静态 step box(台阶) 实现, 二者均兼容圆柱轮 (cylinder-plane/box 碰撞已支持)
        self._floor_geom_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._step_geom_ids = [
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"step{i}") for i in range(4)
        ]

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
    def _randomize_model(self, model: mjx.Model, rng: jax.Array, difficulty: jax.Array) -> Tuple[mjx.Model, jax.Array, jax.Array, jax.Array, jax.Array]:
        """对 mjx.Model 注入域随机化 (per-env), 随机化范围随 difficulty 缩放.

        在 reset 时调用, 对 mass/friction/inertia/COM/wheel_radius/servo_pd 注入随机扰动。
        """
        dr_difficulty = difficulty[DIFF_DR]
        # mass ±15% (与独立 DR 难度缩放)
        mass_scale_raw = jax.random.uniform(
            rng, minval=DR["mass"][0], maxval=DR["mass"][1])
        mass_scale = 1.0 + dr_difficulty * (mass_scale_raw - 1.0)
        model = model.replace(body_mass=model.body_mass * mass_scale)

        # friction [0.3, 1.2] (与 difficulty 缩放)
        rng, friction_rng = jax.random.split(rng)
        friction_raw = jax.random.uniform(
            friction_rng, minval=DR["friction"][0], maxval=DR["friction"][1])
        friction_multiplier = 1.0 + dr_difficulty * (friction_raw - 1.0)
        geom_friction = model.geom_friction.at[:, 0].set(
            model.geom_friction[:, 0] * friction_multiplier)
        model = model.replace(geom_friction=geom_friction)

        # inertia ×[0.5, 2.0] (与 difficulty 缩放)
        rng, inertia_rng = jax.random.split(rng)
        inertia_scale_raw = jax.random.uniform(
            inertia_rng, minval=DR["inertia"][0], maxval=DR["inertia"][1])
        inertia_scale = 1.0 + dr_difficulty * (inertia_scale_raw - 1.0)
        diaginertia = model.body_inertia * inertia_scale
        model = model.replace(body_inertia=diaginertia)

        # COM 偏移 ±20mm (注入到 chassis body_ipos, 与 difficulty 缩放)
        rng, com_rng = jax.random.split(rng)
        com_bias_raw = jax.random.uniform(
            com_rng, (3,), minval=P.DR_COM[0], maxval=P.DR_COM[1])
        com_bias = com_bias_raw * dr_difficulty
        # chassis, 修改其 inertial pos
        new_ipos = model.body_ipos.at[self._chassis_body_id].set(model.body_ipos[self._chassis_body_id] + com_bias)
        model = model.replace(body_ipos=new_ipos)

        # wheel_radius ±1mm (修改轮 geom size[0], 与 difficulty 缩放)
        rng, wr_rng = jax.random.split(rng)
        wheel_r_delta_raw = jax.random.uniform(
            wr_rng, minval=P.DR_WHEEL_R[0], maxval=P.DR_WHEEL_R[1])
        wheel_r_delta = wheel_r_delta_raw * dr_difficulty
        # 轮 geom
        geom_size = model.geom_size
        geom_size = geom_size.at[self._wheel_l_geom_id, 0].set(geom_size[self._wheel_l_geom_id, 0] + wheel_r_delta)
        geom_size = geom_size.at[self._wheel_r_geom_id, 0].set(geom_size[self._wheel_r_geom_id, 0] + wheel_r_delta)
        model = model.replace(geom_size=geom_size)

        # servo_pd ±30% (修改 position actuator 的 gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kv, 与 difficulty 缩放)
        rng, pd_rng = jax.random.split(rng)
        pd_scale_raw = jax.random.uniform(
            pd_rng, minval=P.DR_SERVO_PD[0], maxval=P.DR_SERVO_PD[1])
        pd_scale = 1.0 + dr_difficulty * (pd_scale_raw - 1.0)
        # position actuator idx: 2,3,4,5 (q_hip_A_l/r, q_hip_B_l/r; 4 舵机全部独立驱动)
        # MuJoCo position actuator: biasprm = [0, -kp, -kv] (affine -kp*q0 -kv*vel);
        # 故 kp 缩放必须作用在 biasprm[1] (而非 [0] 常量项), 否则平衡位被偏移到 pd_scale×cmd。
        gainprm = model.actuator_gainprm
        biasprm = model.actuator_biasprm
        for i in [ACT_HIP_A_L, ACT_HIP_A_R, ACT_HIP_B_L, ACT_HIP_B_R]:
            gainprm = gainprm.at[i, 0].set(gainprm[i, 0] * pd_scale)  # kp
            biasprm = biasprm.at[i, 1].set(biasprm[i, 1] * pd_scale)  # -kp*q0
            biasprm = biasprm.at[i, 2].set(biasprm[i, 2] * pd_scale)  # -kv
        model = model.replace(actuator_gainprm=gainprm, actuator_biasprm=biasprm)

        return model, friction_multiplier, mass_scale, inertia_scale, com_bias

    # ---- 地形 (M4 台阶/斜坡) ----
    def _apply_terrain(self, model: mjx.Model, difficulty: jax.Array, rng: jax.Array) -> mjx.Model:
        """按课程难度生成地形 (M4 台阶/斜坡), 兼容 MJX 圆柱轮碰撞.

        MJX 不支持 heightfield×cylinder, 故用:
          - 斜坡: 旋转 ground plane 法向 (绕 Y 轴 ≤10°), 随 difficulty 缩放
          - 台阶: 4 级静态 step box, 高度 = (i+1)×30mm × difficulty (M4 验收 30mm)
        difficulty=0 → 平面不倾斜 + 台阶高度≈0, 完全退回已验证的平地行为。
        出生在原点 (x=0, y=0), 台阶位于 x≥0.6m, 斜坡在原点 z=0 通过, 起步安全。
        """
        # 斜坡: 绕 Y 轴旋转 ground plane, 角度 = difficulty × 10°
        terrain_difficulty = difficulty[DIFF_TERRAIN]
        ang = terrain_difficulty * jp.radians(10.0)
        c, s = jp.cos(ang / 2.0), jp.sin(ang / 2.0)
        floor_quat = jp.array([c, 0.0, s, 0.0])  # (w, x, y, z), 绕 Y
        geom_quat = model.geom_quat.at[self._floor_geom_id].set(floor_quat)
        model = model.replace(geom_quat=geom_quat)

        # 台阶: 逐级高度 (i+1)×30mm × difficulty (最小 0.5mm 避免退化)
        for i, gid in enumerate(self._step_geom_ids):
            h = jp.maximum((i + 1) * 0.03 * terrain_difficulty, 1e-3)
            size = model.geom_size.at[gid, 2].set(h / 2.0)      # box 半高
            pos = model.geom_pos.at[gid, 2].set(h / 2.0)        # 底面贴地 (bottom z=0)
            model = model.replace(geom_size=size, geom_pos=pos)
        return model

    # ---- 命令采样 ----
    def _sample_command(self, rng: jax.Array, difficulty: jax.Array):
        """独立采样命令与 D0 工作空间难度。"""
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        v_limit = 0.05 + 0.45 * difficulty[DIFF_COMMAND]
        w_limit = 0.1 + 0.9 * difficulty[DIFF_COMMAND]
        v_cmd = jax.random.uniform(k1, minval=-v_limit, maxval=v_limit)
        w_cmd = jax.random.uniform(k2, minval=-w_limit, maxval=w_limit)
        d0_upper = P.D0_MIN + (P.D0_MAX - P.D0_MIN) * difficulty[DIFF_D0]
        d0_cmd = jax.random.uniform(k3, minval=D0_CMD_RANGE[0], maxval=d0_upper)
        # 10% 概率零命令 (静支平衡)
        rng, k_zero = jax.random.split(rng)
        is_zero = jax.random.bernoulli(k_zero, p=0.1)
        v_cmd = jp.where(is_zero, 0.0, v_cmd)
        w_cmd = jp.where(is_zero, 0.0, w_cmd)
        return rng, v_cmd, w_cmd, d0_cmd

    # ---- 规范基层控制器（pitch LQR/LQI + yaw 命令跟踪，三轴兜底）----
    def _wheel_torque(self, data: mjx.Data, x_ref: jax.Array, x_int: jax.Array,
                       yaw_ref: jax.Array, v_ref: jax.Array, w_ref: jax.Array,
                      delayed_action: jax.Array, torque_scale: jax.Array):
        """基层轮扭矩（每 base 步重算）。返回 (tau_wheel_l, tau_wheel_r)。

        规范控制律（kuafu_physics 单源）：
          pitch: 离散 LQR_K_DT4 状态 e=[x-x_ref, θ, ẋ-v_cmd, θ̇] → 地面力 F, 两轮等分
                 τ_pitch = F·R/2。叠加 LQI 积分项 -Ki·∫(x-x_ref) 消除位置稳态漂移。
          yaw:   命令跟踪 τ_yaw = YAW_KP·(w_cmd-ωz) - YAW_KD·ωz；
                 轮映射 τL=τ_pitch-τ_yaw, τR=τ_pitch+τ_yaw（=> τR>τL ⇒ +wz 左转）。
         RL 残差（合约动作 [dtau_common, dtau_yaw]·τ_rated）叠加在基层之上：
           τL += dtau_common - dtau_yaw,  τR += dtau_common + dtau_yaw
         """
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        q = jp.stack([qw, qx, qy, qz])

        lin_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[0:3])
        xdot = lin_vel_local[0]
        ang_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
        thetadot = ang_vel_local[1]  # wy = pitch rate
        yaw_rate = ang_vel_local[2]  # wz = yaw rate

        theta = jp.arcsin(jp.clip(2 * (qw * qy - qx * qz), -0.999999, 0.999999))
        state = jp.stack([data.qpos[0] - x_ref, theta, xdot - v_ref, thetadot])
        F = -(LQR_K_DT4 @ state) - LQI_KI * x_int
        tau_pitch = F * WHEEL_R / 2.0 * torque_scale

        yaw = jp.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz ** 2 + qy ** 2))
        yaw_error = jp.arctan2(jp.sin(yaw_ref - yaw), jp.cos(yaw_ref - yaw))
        tau_yaw = (YAW_KP * yaw_error + YAW_KD * (w_ref - yaw_rate)) * torque_scale
        # 合约残差: dtau_common (前向) / dtau_yaw (偏航), 经 wheels_from_tau 映射到轮
        dtau_common = delayed_action[0] * TAU_WHEEL_RATED * torque_scale
        dtau_yaw = delayed_action[1] * TAU_WHEEL_RATED * torque_scale
        tau_wheel_l = tau_pitch - tau_yaw + dtau_common - dtau_yaw
        tau_wheel_r = tau_pitch + tau_yaw + dtau_common + dtau_yaw
        tau_wheel_l = self._limit_wheel_torque(
            jp.clip(tau_wheel_l, -TAU_WHEEL_STALL, TAU_WHEEL_STALL),
            data.qvel[QVEL_WHEEL_L])
        tau_wheel_r = self._limit_wheel_torque(
            jp.clip(tau_wheel_r, -TAU_WHEEL_STALL, TAU_WHEEL_STALL),
            data.qvel[QVEL_WHEEL_R])
        return tau_wheel_l, tau_wheel_r

    def _leg_goals(self, data: mjx.Data, env_state: EnvState,
                    delayed_action: jax.Array, d0_cmd: jax.Array) -> jax.Array:
        """基层腿目标（每 RL 步一次）：D0 对称 + roll 调平 + RL 残差，经精确五杆 IK。

        返回 4 维 hip 目标 [A_l, A_r, B_l, B_r]（rad）。
        流程：d0_cmd ± roll 拆分 → 每侧 D0(mm) → fivebar_ik_cmd → (qA,qB)。
        RL 残差（合约 [dQx_L, dD0_L, dQx_R, dD0_R]）作为每侧目标点
        ``(Qx,D0)`` 的独立增量，经二维精确 IK。
        审计约定：∂qA/∂D0<0, ∂qB/∂D0>0（fivebar_ik_cmd 已施加）。
        """
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        roll = jp.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
        q = jp.stack([qw, qx, qy, qz])
        ang_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
        omega_x = ang_vel_local[0]
        # roll 调平: 正 roll → 左腿伸更多(+, 右腿 -, mm)
        d_d0_mm = -(ROLL_KP * roll + ROLL_KD * omega_x)
        # 每侧 D0: 命令 + roll 拆分 + 独立 D0 残差；Qx 独立投影，不能混入 D0。
        d0_left = d0_cmd + d_d0_mm / 2.0 + delayed_action[3] * D0_RESIDUAL_SCALE
        d0_right = d0_cmd - d_d0_mm / 2.0 + delayed_action[5] * D0_RESIDUAL_SCALE
        d0_left = jp.clip(d0_left, P.D0_MIN, P.D0_MAX)
        d0_right = jp.clip(d0_right, P.D0_MIN, P.D0_MAX)
        qx_left = jp.clip(delayed_action[2] * QX_RESIDUAL_SCALE, self._ik_qx[0], self._ik_qx[-1])
        qx_right = jp.clip(delayed_action[4] * QX_RESIDUAL_SCALE, self._ik_qx[0], self._ik_qx[-1])
        qA_l, qB_l = self._interpolate_ik(qx_left, d0_left)
        qA_r, qB_r = self._interpolate_ik(qx_right, d0_right)
        return jp.stack([qA_l, qA_r, qB_l, qB_r])

    def _interpolate_ik(self, qx_mm: jax.Array, d0_mm: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Bilinear interpolation of the source-generated five-bar IK grid."""
        qA_at_qx = jax.vmap(lambda row: jp.interp(d0_mm, self._ik_d0, row))(self._ik_qA)
        qB_at_qx = jax.vmap(lambda row: jp.interp(d0_mm, self._ik_d0, row))(self._ik_qB)
        return (jp.interp(qx_mm, self._ik_qx, qA_at_qx),
                jp.interp(qx_mm, self._ik_qx, qB_at_qx))

    def _fk_from_hips(self, qA: jax.Array, qB: jax.Array) -> jax.Array:
        """JAX-safe closed-chain FK for dwell-relative actuator angles (mm)."""
        angle_a = jp.asarray(P.AX) + P.A_LEN * jp.cos(self._dwell_qA - qA)
        z_a = P.A_LEN * jp.sin(self._dwell_qA - qA)
        angle_b = jp.asarray(P.BX) + P.A_LEN * jp.cos(self._dwell_qB - qB)
        z_b = P.A_LEN * jp.sin(self._dwell_qB - qB)
        midpoint = jp.stack([(angle_a + angle_b) / 2.0, (z_a + z_b) / 2.0])
        dx = angle_b - angle_a
        dz = z_b - z_a
        distance = jp.maximum(jp.sqrt(dx * dx + dz * dz), 1e-6)
        half = jp.sqrt(jp.maximum(P.B_LEN * P.B_LEN - (distance / 2.0) ** 2, 0.0))
        perp = jp.stack([-dz / distance, dx / distance])
        candidate_a = midpoint + perp * half
        candidate_b = midpoint - perp * half
        return jp.where(candidate_a[1] < candidate_b[1], candidate_a, candidate_b)

    def _d0_from_hips(self, qA: jax.Array, qB: jax.Array) -> jax.Array:
        """Return D0 from the full two-hip FK, preserving Qx residuals."""
        return -self._fk_from_hips(qA, qB)[1]

    # ---- DDSM back-EMF 限制 ----
    def _limit_wheel_torque(self, tau: jax.Array, omega: jax.Array) -> jax.Array:
        """DDSM 力矩-转速限制: τ_avail = τ_stall × (1 - |ω|/ω_noload)."""
        tau_avail = TAU_WHEEL_STALL * (1.0 - jp.abs(omega) / OMEGA_NOLOAD)
        tau_avail = jp.clip(tau_avail, 0.0, TAU_WHEEL_STALL)
        return jp.clip(tau, -tau_avail, tau_avail)

    # ---- 观测 ----
    def _wheel_contact(self, data: mjx.Data) -> jax.Array:
        """左右轮接触标志 (2 维): 1.0 = 接地, 0.0 = 离地。

        从 data.contact.geom 过滤含轮 geom 的接触。M2/M4 涌现关键:
        策略据此判断轮是否接地, 决定伸腿/调平时机。
        """
        # contact.geom: (nconmax, 2) — 仅前 ncon 个接触有效, 越界槽位为历史残留(未清零),
        # 若其 geom id 恰为轮 id 会误报接地。故先以 ncon 掩码过滤无效槽位。
        geom = data.contact.geom
        active = jp.arange(geom.shape[0]) < data.ncon
        l_mask = ((geom[:, 0] == self._wheel_l_geom_id) | (geom[:, 1] == self._wheel_l_geom_id)) & active
        r_mask = ((geom[:, 0] == self._wheel_r_geom_id) | (geom[:, 1] == self._wheel_r_geom_id)) & active
        contact_l = jp.any(l_mask).astype(jp.float32)
        contact_r = jp.any(r_mask).astype(jp.float32)
        return jp.stack([contact_l, contact_r])

    def _base_observation(self, data: mjx.Data, env_state: EnvState) -> jax.Array:
        """35 维、固定物理尺度的实机可观测 Actor 帧。"""
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        q = jp.stack([qw, qx, qy, qz])
        roll = jp.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        grav_local = rotate_vector_by_quaternion_conj(q, jp.array([0.0, 0.0, -1.0]))
        ang_vel = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
        lin_vel = rotate_vector_by_quaternion_conj(q, data.qvel[0:3])
        wheel_vel = jp.stack([data.qvel[QVEL_WHEEL_L], data.qvel[QVEL_WHEEL_R]])
        d0_est = (
            self._d0_from_hips(data.qpos[QPOS_HIP_A_L], data.qpos[QPOS_HIP_B_L])
            + self._d0_from_hips(data.qpos[QPOS_HIP_A_R], data.qpos[QPOS_HIP_B_R])
        ) / 2.0
        hip_pos = jp.stack([
            data.qpos[QPOS_HIP_A_L], data.qpos[QPOS_HIP_B_L],
            data.qpos[QPOS_HIP_A_R], data.qpos[QPOS_HIP_B_R]])
        hip_vel = jp.stack([
            data.qvel[QVEL_HIP_A_L], data.qvel[QVEL_HIP_B_L],
            data.qvel[QVEL_HIP_A_R], data.qvel[QVEL_HIP_B_R]])
        sensor_age_ms = jp.ones(6) * env_state.sense_delay_steps.astype(jp.float32) * CTRL_DT * 1000.0
        return jp.concatenate([
            jp.stack([env_state.v_cmd / 0.5, env_state.w_cmd, (env_state.d0_cmd - 132.5) / 74.5]),
            grav_local,
            ang_vel / 10.0,
            jp.stack([lin_vel[0] / 0.5, ang_vel[2], (d0_est - 132.5) / 74.5, roll]),
            wheel_vel / 33.0,
            hip_pos / HIP_RANGE,
            hip_vel / P.SERVO_MAX_SPEED,
            env_state.prev_action,
            sensor_age_ms / 100.0,
        ])

    def _static_privileged_observation(self, env_state: EnvState) -> jax.Array:
        """Critic-only 9 维静态环境外因，不含瞬态 active_push。"""
        return jp.concatenate([
            env_state.friction[jp.newaxis],      # 1
            env_state.mass_scale[jp.newaxis],     # 1
            env_state.com_bias,                   # 3
            env_state.inertia_scale[jp.newaxis],  # 1
            env_state.torque_scale[jp.newaxis],   # 1
            env_state.deadband[jp.newaxis],       # 1 死区真值
            env_state.delay_steps.astype(jp.float32)[jp.newaxis],  # 1 延迟步数真值
        ])

    def _delayed_obs(self, env_state: EnvState) -> jax.Array:
        """对喂给 policy 的本体感受观测施加观测延迟 (sensor/compute latency).

        缓冲填充顺序为 [丢弃最旧, 追加最新] (step 中 new_obs_delay_buffer),
        故 obs_delay_buffer[0]=2步前(最旧), [1]=1步前, [2]=当前。
        delay_steps>=2 取 2 步前(buf[0]), >=1 取 1 步前(buf[1]), 否则当前。
        reset 时缓冲为 0(与 obs_history 初始全 0 一致, 无回归)。
        """
        cur = env_state.obs_history.reshape(-1)
        buf = env_state.obs_delay_buffer
        return jp.where(
            env_state.sense_delay_steps >= 2,
            buf[0],
            jp.where(env_state.sense_delay_steps >= 1, buf[1], cur))

    def _observation(self, data: mjx.Data, env_state: EnvState):
        """构建观测: teacher 返回 dict, student 返回扁平数组.

        obs_history 已在 step() 中更新 (含最新帧); 此处对 policy 观测施加观测延迟。

        Actor always receives causal proprioception; the Critic receives privileged
        simulation values only during PPO training.
        """
        flat_obs = self._delayed_obs(env_state)    # (140,) 已含观测延迟

        if self._teacher:
            privileged = jp.concatenate([
                self._static_privileged_observation(env_state), env_state.active_push])
            return {"state": flat_obs, "privileged_state": privileged}
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

        # tilt_cost: 对倾角的线性惩罚, 与 orientation(exp) 互补。orientation 在接近直立时
        # 饱和、对大倾角恢复激励不足; tilt_cost 提供全程梯度, 加速被推后的快速扶正
        # (perturbation recovery, legged_gym/ETH 实践)。grav_local[:2] 模长≈sin(tilt)。
        tilt_cost = -jp.sqrt(grav_local[0] ** 2 + grav_local[1] ** 2)

        # lin_vel_tracking: 跟踪 v_cmd (本体系前向速度 xdot)
        lin_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[0:3])
        xdot = lin_vel_local[0]
        lin_vel_error = (xdot - env_state.v_cmd) ** 2
        lin_vel_tracking = jp.exp(-lin_vel_error / 0.15)

        # ang_vel_tracking: 跟踪 w_cmd (yaw 角速度, 本体系 wz)
        ang_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
        yaw_rate = ang_vel_local[2]  # wz
        ang_vel_error = (yaw_rate - env_state.w_cmd) ** 2
        ang_vel_tracking = jp.exp(-ang_vel_error / 0.15)

        # d0_avg_tracking: measured hips 经 JAX 查表反插得到 D0 (mm)。
        d0_l = self._d0_from_hips(data.qpos[QPOS_HIP_A_L], data.qpos[QPOS_HIP_B_L])
        d0_r = self._d0_from_hips(data.qpos[QPOS_HIP_A_R], data.qpos[QPOS_HIP_B_R])
        avg_d0 = (d0_l + d0_r) / 2.0
        d0_err_sq = (avg_d0 - env_state.d0_cmd) ** 2
        d0_avg_tracking = jp.exp(-d0_err_sq / 100.0)  # 100mm² ≈ (10mm)²

        # roll_leveling: 显式奖励机身水平 (roll→0)。机构支持左右 D0 不等调平,
        # 此项激活该能力 (原无, default_pose 反而抑制)。
        roll = jp.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
        roll_leveling = jp.exp(-roll ** 2 / 0.05)  # σ=0.05 rad²≈(13°)²

        # contact_asymmetry_penalty: 抑制长时间单轮卸载 (限制 M4 为短暂轻抬)
        contact = self._wheel_contact(data)
        contact_asym = -jp.abs(contact[0] - contact[1])

        # extension_cost: 仅惩罚超出 d0_cmd 的过度伸展 (不惩罚命令内伸展,
        # 避免与 d0_avg_tracking 梯度互抵); FK 自洽。
        extension_cost = -jp.maximum(avg_d0 - env_state.d0_cmd, 0.0) / 1000.0

        # alive: 存活奖励, 对抗训练初期"一碰就死"的局部最优 (T1 轮式用 0.25, KUAFU 有
        # 坚实 orientation+tracking 故降至 0.1; 配合 FALL_GRACE_STEPS 软终止)
        # 门控: 倒下期间不发 alive, 防止软终止窗口内"蹭存活奖励"的退化策略
        fallen_flag = self._is_fallen(data).astype(jp.float32)
        alive = 1.0 - fallen_flag

        # --- style (负向惩罚) ---
        # ang_vel_xy: 惩罚 roll/pitch 角速度 (ωx²+ωy²), 抑制高频抖动。
        # orientation 只约束姿态角, ang_vel_xy 约束角速度 — 两者互补防"姿态正但抖动"
        ang_vel_xy = -(ang_vel_local[0] ** 2 + ang_vel_local[1] ** 2)

        # action_rate: -‖a_t - a_{t-1}‖² (一阶, Go1/T1 不用二阶 jerk)
        action_rate = -jp.sum((action - env_state.prev_action) ** 2)

        # energy: 分执行器类型度量 (保护电机不过热 + 鼓励省电)
        # 轮 DDSM315 (准直驱 gear≈1): |τ·ω| = 机械功率 ≈ 电能消耗 ✓
        # 髋 ST3215 (1:345 高减速比): τ² = 铜损 I²R 代理 (保持力矩时 ω=0 但仍发热,
        #   |τ·ω| 无法惩罚静态大力矩; τ² 才能反映过热风险 ∝ 电流²)
        wheel_energy = (
            jp.abs(data.qvel[QVEL_WHEEL_L] * data.actuator_force[ACT_TAU_L])
            + jp.abs(data.qvel[QVEL_WHEEL_R] * data.actuator_force[ACT_TAU_R]))
        hip_energy = (
            data.actuator_force[ACT_HIP_A_L] ** 2
            + data.actuator_force[ACT_HIP_A_R] ** 2
            + data.actuator_force[ACT_HIP_B_L] ** 2
            + data.actuator_force[ACT_HIP_B_R] ** 2)
        energy = -(wheel_energy + hip_energy)

        # torque_limit: 超连续安全扭矩惩罚 (4 舵机)
        tau_excess = (
            jp.maximum(jp.abs(data.actuator_force[ACT_HIP_A_L]) - TAU_CONT, 0)
            + jp.maximum(jp.abs(data.actuator_force[ACT_HIP_A_R]) - TAU_CONT, 0)
            + jp.maximum(jp.abs(data.actuator_force[ACT_HIP_B_L]) - TAU_CONT, 0)
            + jp.maximum(jp.abs(data.actuator_force[ACT_HIP_B_R]) - TAU_CONT, 0))
        torque_limit = -tau_excess

        # ---- P4 动作/工作空间惩罚 (audit P0/P4: 对抗策略不得靠静止/饱和/顶限位通过) ----
        # residual_magnitude: 惩罚 RL 残差幅度, 鼓励贴近基层 LQR/LQI 控制器
        residual_mag = -jp.sum(action ** 2)

        # saturation: tanh 残差接近 ±1 饱和时惩罚 (残差已无梯度余量, 且逼近执行器极限)
        sat = -jp.sum(jp.maximum(jp.abs(action) - 0.9, 0.0) ** 2)

        # workspace: 髋关节逼近硬限位 (±3.3rad) 时惩罚, 保持机构在工作空间内
        hip_q = jp.stack([
            data.qpos[QPOS_HIP_A_L], data.qpos[QPOS_HIP_A_R],
            data.qpos[QPOS_HIP_B_L], data.qpos[QPOS_HIP_B_R]])
        workspace = -jp.sum(jp.maximum(jp.abs(hip_q) - WORKSPACE_SAFE, 0.0) ** 2)

        # termination_penalty: 倒下当步给负奖励, 配合 LQR 兜底, 引导尽快恢复/避免倒下
        fall_penalty = -1.0 * fallen_flag

        total = (
            1.5 * lin_vel_tracking
            + 1.0 * ang_vel_tracking
            + 1.0 * orientation
            + 0.5 * tilt_cost
            + 0.3 * d0_avg_tracking
            + 1.0 * roll_leveling
            + 0.5 * extension_cost
            + 0.3 * contact_asym
            + 0.1 * alive
            + 0.05 * ang_vel_xy
            + 0.01 * action_rate
            + 0.001 * energy
            + 0.5 * torque_limit
            + 1.0 * fall_penalty
            + RESID_W * residual_mag
            + SAT_W * sat
            + WORKSPACE_W * workspace
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
        (与 alive 门控配合: 倒下时 alive=0 + fall_penalty=-1, 鼓励长时平衡)
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
    def reset(self, rng: jax.Array, difficulty: jax.Array = jp.zeros(DIFFICULTY_DIM)) -> State:
        """重置环境到驻留态 + 域随机化 + 命令采样."""
        rng, cmd_rng, dr_rng, torque_rng, db_rng, delay_rng, sense_delay_rng = jax.random.split(rng, 7)

        # 域随机化 (mass/friction/inertia/COM/wheel_r/servo_pd 注入物理, 与 difficulty 缩放)
        model, friction, mass_scale, inertia_scale, com_bias = self._randomize_model(
            self._mjx_model, dr_rng, difficulty)
        # 地形 heightfield (difficulty=0 → 全平; 见 _apply_terrain)
        model = self._apply_terrain(model, difficulty, dr_rng)
        
        # DDSM 力矩常数 DR (与 difficulty 缩放)
        torque_scale_raw = jax.random.uniform(
            torque_rng, minval=DR["torque_const"][0], maxval=DR["torque_const"][1])
        torque_scale = 1.0 + difficulty[DIFF_DR] * (torque_scale_raw - 1.0)

        # deadband [0, 2°] (舵机齿轮间隙, 与 difficulty 缩放)
        deadband_raw = jax.random.uniform(
            db_rng, minval=P.DR_DEADBAND[0], maxval=P.DR_DEADBAND[1])
        deadband = deadband_raw * difficulty[DIFF_DR]

        # delay [0, 30ms] → 步数 (1步=20ms, 最大 1-2 步, 与 difficulty 缩放)
        delay_s_raw = jax.random.uniform(
            delay_rng, minval=P.DR_DELAY_ACT[0], maxval=P.DR_DELAY_ACT[1])
        delay_s = delay_s_raw * difficulty[DIFF_DR]
        delay_steps = jp.round(delay_s / CTRL_DT).astype(jp.int32)
        sense_delay_s_raw = jax.random.uniform(
            sense_delay_rng, minval=P.DR_DELAY_SENSE[0], maxval=P.DR_DELAY_SENSE[1])
        sense_delay_steps = jp.round(
            sense_delay_s_raw * difficulty[DIFF_DR] / CTRL_DT).astype(jp.int32)

        # 命令 (范围与 difficulty 缩放)
        cmd_rng, v_cmd, w_cmd, d0_cmd = self._sample_command(cmd_rng, difficulty)

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
            sense_delay_steps=sense_delay_steps,
            action_buffer=jp.zeros((DELAY_BUFFER_LEN, ACTION_DIM)),
            obs_delay_buffer=jp.zeros((DELAY_BUFFER_LEN, OBS_DIM)),
            fall_count=jp.int32(0),
            difficulty=difficulty,
            push_force=jp.zeros(3),
            active_push=jp.zeros(3),
            track_v_abs_sum=jp.float32(0.0),
            track_w_abs_sum=jp.float32(0.0),
            track_d0_abs_sum=jp.float32(0.0),
            track_count=jp.int32(0),
            nonzero_command_count=jp.int32(0),
            x_ref=jp.float32(0.0),
            x_int=jp.float32(0.0),
            yaw_ref=jp.float32(0.0),
            v_ref=jp.float32(0.0),
            v_accel=jp.float32(0.0),
            w_ref=jp.float32(0.0),
            w_accel=jp.float32(0.0),
        )

        obs = self._observation(data, env_state)
        reward = jp.float32(0.0)
        done = jp.bool_(False)
        # metrics key 必须与 step() 返回的完全一致 (DirectVecEnv step 用 tree_map 合并
        # done 帧的 reset_state, pytree 结构不同会抛 ValueError)
        metrics = {
            "orientation": jp.float32(0.0),
            "lin_vel_tracking": jp.float32(0.0),
            "lin_vel_err": jp.float32(0.0),
            "yaw_err": jp.float32(0.0),
            "d0_err": jp.float32(0.0),
            "fallen": jp.float32(0.0),
            "difficulty": jp.mean(difficulty),
            "v_cmd": jp.float32(0.0),
            "w_cmd": jp.float32(0.0),
            "resid_norm": jp.float32(0.0),
        }
        info = {"env_state": env_state, "model": model}
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """执行一步控制: LQR + RL 残差叠加 → 10 子步物理 → reward/obs/done."""
        action = jp.clip(action, -1.0, 1.0)
        data = state.data
        env_state = state.info["env_state"]
        model = state.info["model"]
        metric_v_cmd = env_state.v_cmd
        metric_w_cmd = env_state.w_cmd
        metric_d0_cmd = env_state.d0_cmd

        # ---- 执行器延迟: 从 action_buffer 取延迟前的动作 ----
        # action_buffer 在 step 末尾填充 [丢弃最旧, 追加最新], 故 step 开始时缓冲为
        # [a_{t-3}, a_{t-2}, a_{t-1}] (不含当前)。delay_steps=k 取 k 步前的动作:
        #   delay=1 → 1步前(buf[2]); delay=2 → 2步前(buf[1])。与 _delayed_obs 语义一致。
        delayed_action = jp.where(
            env_state.delay_steps >= 2,
            env_state.action_buffer[1],  # 2 步前 a_{t-2}
            jp.where(
                env_state.delay_steps >= 1,
                env_state.action_buffer[2],  # 1 步前 a_{t-1}
                action,                      # 当前
            )
        )
        # 更新 action_buffer: 推入当前 action, 丢弃最旧
        new_action_buffer = jp.concatenate([
            env_state.action_buffer[1:],
            action[jp.newaxis, :],
        ], axis=0)

        # ---- 瞬时推力扰动 (velocity kick) 注入（须在物理步进前设置 xfrc）----
        # 每 200 步 (4 秒) 重采样一个推力方向与强度 (最大 15N)
        rng, push_rng = jax.random.split(env_state.rng)
        is_push_resample = (env_state.step_count % 200 == 0)
        k_force, k_dir = jax.random.split(push_rng)
        push_mag = jax.random.uniform(k_force, minval=0.0, maxval=15.0)
        push_angle = jax.random.uniform(k_dir, minval=0.0, maxval=2.0 * jp.pi)
        new_push_force = jp.stack([
            push_mag * jp.cos(push_angle),
            push_mag * jp.sin(push_angle),
            jp.float32(0.0)
        ])
        push_force = jp.where(is_push_resample, new_push_force, env_state.push_force)
        # 仅在每个 4s 周期的前 5 步（100ms）施加推力，且在刚 reset 的前几步不施加
        is_push_active = ((env_state.step_count % 200) < 5) & (env_state.step_count > 5)
        active_push = jp.where(is_push_active, push_force * env_state.difficulty[DIFF_PUSH], 0.0)
        # 注入推力到 chassis 质心 (xfrc_applied)
        xfrc_applied = data.xfrc_applied
        xfrc_applied = xfrc_applied.at[self._chassis_body_id, :3].set(active_push)
        data = data.replace(xfrc_applied=xfrc_applied)

        # ---- 腿目标（每 RL 步一次）：D0 对称 + roll 调平 + RL 残差 ----
        hip_goal = self._leg_goals(data, env_state, delayed_action, env_state.d0_cmd)
        actual_hips = jp.stack([
            data.qpos[QPOS_HIP_A_L], data.qpos[QPOS_HIP_A_R],
            data.qpos[QPOS_HIP_B_L], data.qpos[QPOS_HIP_B_R]])
        hip_error = hip_goal - actual_hips
        hip_goal = jp.where(jp.abs(hip_error) < env_state.deadband, actual_hips, hip_goal)

        # ---- 多速率基层控制：5 base 步(4ms) × 2 物理子步(2ms) = 10 物理子步 ----
        # 物理 500Hz / 基层 250Hz / RL 残差 50Hz。RL 残差(delayed_action) 跨 5 base 步保持。
        # 2-DOF 五杆: 闭链靠 <connect site1/site2> 物理铰接维持 (硬 solver)。
        x_ref = env_state.x_ref
        x_int = env_state.x_int
        yaw_ref = env_state.yaw_ref
        v_ref = env_state.v_ref
        v_accel = env_state.v_accel
        w_ref = env_state.w_ref
        w_accel = env_state.w_accel
        for _ in range(BASE_STEPS_PER_RL):
            # Jerk-limited velocity/yaw references; as command reaches zero the
            # integrated position and heading references freeze for hold control.
            v_target_acc = jp.clip((env_state.v_cmd - v_ref) / BASE_DT, -2.0, 2.0)
            v_accel = jp.clip(v_accel + jp.clip(v_target_acc - v_accel, -8.0 * BASE_DT, 8.0 * BASE_DT), -2.0, 2.0)
            v_ref = v_ref + v_accel * BASE_DT
            w_target_acc = jp.clip((env_state.w_cmd - w_ref) / BASE_DT, -4.0, 4.0)
            w_accel = jp.clip(w_accel + jp.clip(w_target_acc - w_accel, -16.0 * BASE_DT, 16.0 * BASE_DT), -4.0, 4.0)
            w_ref = w_ref + w_accel * BASE_DT
            x_ref = x_ref + v_ref * BASE_DT
            yaw_ref = yaw_ref + w_ref * BASE_DT
            x_int = jp.clip(x_int + (data.qpos[0] - x_ref) * BASE_DT, -0.25, 0.25)
            tau_wheel_l, tau_wheel_r = self._wheel_torque(
                data, x_ref, x_int, yaw_ref, v_ref, w_ref,
                delayed_action, env_state.torque_scale)
            ctrl = jp.zeros(model.nu)
            ctrl = ctrl.at[ACT_TAU_L].set(tau_wheel_l)
            ctrl = ctrl.at[ACT_TAU_R].set(tau_wheel_r)
            ctrl = ctrl.at[ACT_HIP_A_L].set(hip_goal[0])
            ctrl = ctrl.at[ACT_HIP_A_R].set(hip_goal[1])
            ctrl = ctrl.at[ACT_HIP_B_L].set(hip_goal[2])
            ctrl = ctrl.at[ACT_HIP_B_R].set(hip_goal[3])
            data = data.replace(ctrl=ctrl)
            for _ in range(PHYS_SUBSTEPS_PER_BASE):
                data = mjx.step(model, data)

        # ---- 更新状态 ----
        rng, resample_rng = jax.random.split(rng)
        # 每 100 步重采样命令 (2s @ 50Hz, 范围与 difficulty 缩放)
        need_resample = (env_state.step_count % 100 == 0) & (env_state.step_count > 0)
        _, new_v, new_w, new_d0 = self._sample_command(resample_rng, env_state.difficulty)
        v_cmd = jp.where(need_resample, new_v, env_state.v_cmd)
        w_cmd = jp.where(need_resample, new_w, env_state.w_cmd)
        d0_cmd = jp.where(need_resample, new_d0, env_state.d0_cmd)

        # ---- reward / done / obs (用旧命令状态；重采样命令从下一控制步才生效) ----
        raw_reward = self._compute_reward(data, env_state, action)
        fallen = self._is_fallen(data)

        # fall_count 软终止计数 (倒下累加, 恢复清零)
        new_fall_count = jp.where(fallen, env_state.fall_count + 1, 0)
        done = (new_fall_count >= FALL_GRACE_STEPS) | (
            env_state.step_count >= self._episode_length)

        # reward × CTRL_DT (Go1/T1 标准: 每项 ×scale 后乘 dt, 保持 PPO value 尺度一致)
        # 软终止期间: alive=0 + fall_penalty=-1 双重抑制 (靠 episode 结束截断后续 reward)
        reward = raw_reward * CTRL_DT

        # Episode tracking aggregates use the command that controlled this step.
        q_metric = jp.stack([data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]])
        metric_lin_vel = rotate_vector_by_quaternion_conj(q_metric, data.qvel[0:3])[0]
        metric_ang_vel = rotate_vector_by_quaternion_conj(q_metric, data.qvel[3:6])
        metric_d0 = (
            self._d0_from_hips(data.qpos[QPOS_HIP_A_L], data.qpos[QPOS_HIP_B_L])
            + self._d0_from_hips(data.qpos[QPOS_HIP_A_R], data.qpos[QPOS_HIP_B_R])
        ) / 2.0
        nonzero_command = ((jp.abs(env_state.v_cmd) > 0.05)
                           | (jp.abs(env_state.w_cmd) > 0.10)
                           | (jp.abs(env_state.d0_cmd - P.D0_MIN) > 5.0))

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
            push_force=push_force,
            active_push=active_push,
            x_ref=x_ref,
            x_int=x_int,
            yaw_ref=yaw_ref,
            v_ref=v_ref,
            v_accel=v_accel,
            w_ref=w_ref,
            w_accel=w_accel,
            track_v_abs_sum=env_state.track_v_abs_sum + jp.abs(metric_lin_vel - env_state.v_cmd),
            track_w_abs_sum=env_state.track_w_abs_sum + jp.abs(metric_ang_vel[2] - env_state.w_cmd),
            track_d0_abs_sum=env_state.track_d0_abs_sum + jp.abs(metric_d0 - env_state.d0_cmd),
            track_count=env_state.track_count + 1,
            nonzero_command_count=env_state.nonzero_command_count + nonzero_command.astype(jp.int32),
        )

        # 更新 obs history (append 当前帧, 丢弃最老帧)
        base_obs = self._base_observation(data, env_state)
        new_history = jp.concatenate([
            env_state.obs_history[1:],
            base_obs[jp.newaxis, :],
        ], axis=0)
        env_state = env_state.replace(obs_history=new_history)

        # 更新观测延迟缓冲 (与动作延迟同构: 推入当前 obs, 丢弃最旧)
        new_obs_delay_buffer = jp.concatenate([
            env_state.obs_delay_buffer[1:],
            new_history.reshape(-1)[jp.newaxis, :],
        ], axis=0)
        env_state = env_state.replace(obs_delay_buffer=new_obs_delay_buffer)

        # lin_vel_tracking / orientation 实际值 (供 metric 记录)
        q = jp.stack([data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]])
        lin_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[0:3])
        ang_vel_local = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
        xdot = lin_vel_local[0]
        lin_vel_track_val = jp.exp(-((xdot - metric_v_cmd) ** 2) / 0.15)
        grav_local = rotate_vector_by_quaternion_conj(q, jp.array([0.0, 0.0, -1.0]))
        orient_val = jp.exp(-ORIENT_ALPHA * jp.sum(grav_local[:2] ** 2))

        obs = self._observation(data, env_state)
        # terminated_by_fall: done 且因连续倒下触发 (供 train.py time_outs 区分 timeout vs fall)
        terminated_by_fall = (new_fall_count >= FALL_GRACE_STEPS).astype(jp.float32)
        lin_vel_err = (xdot - metric_v_cmd) ** 2
        yaw_rate = ang_vel_local[2]
        yaw_err = (yaw_rate - metric_w_cmd) ** 2
        # d0_err 的单位为 mm²；课程阈值必须相应使用 5²，而不是米制 0.05²。
        avg_d0_m = (
            self._d0_from_hips(data.qpos[QPOS_HIP_A_L], data.qpos[QPOS_HIP_B_L])
            + self._d0_from_hips(data.qpos[QPOS_HIP_A_R], data.qpos[QPOS_HIP_B_R])
        ) / 2.0
        d0_err = (avg_d0_m - metric_d0_cmd) ** 2
        resid_norm = jp.sum(action ** 2)
        metrics = {
            "orientation": orient_val,
            "lin_vel_tracking": lin_vel_track_val,
            "lin_vel_err": lin_vel_err,
            "yaw_err": yaw_err,
            "d0_err": d0_err,
            "fallen": terminated_by_fall,  # done 帧的终止原因 (1=摔倒终止, 0=超时)
            "difficulty": jp.mean(env_state.difficulty),
            "v_cmd": metric_v_cmd,
            "w_cmd": metric_w_cmd,
            "resid_norm": resid_norm,
        }
        info = {"env_state": env_state, "model": model}
        return state.replace(
            data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)


def make_env(teacher: bool = True, num_envs: int = 1024, **kwargs) -> KuafuMjxEnv:
    """工厂函数: 创建 KUAFU MJX 环境."""
    return KuafuMjxEnv(teacher=teacher, num_envs=num_envs, **kwargs)
