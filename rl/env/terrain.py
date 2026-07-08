# -*- coding: utf-8 -*-
"""
KUAFU 课程地形系统 — design.md §2.6 / §3.4

[状态: 未接入训练] 当前 train.py 不引用本模块。平地平衡 reward 收敛后,
在 train.py 外层循环中调用 CurriculumController, 将 difficulty 传入环境。

注意: get_terrain_params 使用 Python if/elif (不可 JIT)。
接入 vmap 环境时需重写为 jax.lax.switch, 或从 host 侧每 episode 更新 difficulty。

按课程阶段 (difficulty 0.0→1.0) 程序化生成地形:
  0.0: plane (平地平衡)
  0.3: plane_tilt (坡度 ±10°)
  0.5: hfield (随机粗糙噪声)
  0.7: mesh_stair (30mm 台阶, M4 验收)
  1.0: perturbation (平地 + 随机推力扰动)

地形通过修改 MJCF floor geom 实现, 或在 step 中注入外部扰动力。
本轮提供地形参数生成 + 外部扰动采样, 供 KuafuMjxEnv 使用。
"""
import jax
import jax.numpy as jp
import numpy as np


def get_terrain_params(difficulty: float, rng: jax.Array) -> dict:
    """按课程难度生成地形参数.

    Args:
        difficulty: 0.0 (平地) → 1.0 (最难)
        rng: JAX 随机数种子

    Returns:
        dict: terrain_type, tilt_angle, roughness_amplitude, stair_height, push_force
    """
    rng, t_rng, tilt_rng, rough_rng, stair_rng, push_rng = jax.random.split(rng, 6)

    # 地形类型按 difficulty 分段
    if difficulty < 0.2:
        terrain_type = "plane"
    elif difficulty < 0.4:
        terrain_type = "plane_tilt"
    elif difficulty < 0.6:
        terrain_type = "hfield"
    elif difficulty < 0.8:
        terrain_type = "mesh_stair"
    else:
        terrain_type = "perturbation"

    # 坡度 (plane_tilt): difficulty 越高坡度越大, 最大 ±10°
    tilt_angle = jax.random.uniform(tilt_rng, minval=-1, maxval=1) * jp.radians(10) * difficulty

    # 粗糙度 (hfield): 噪声幅度 0-15mm
    roughness = jax.random.uniform(rough_rng, maxval=0.015) * difficulty

    # 台阶高度 (mesh_stair): 0-30mm
    stair_height = jax.random.uniform(stair_rng, maxval=0.030) * difficulty

    # 推力扰动 (perturbation): 随机脉冲力 0-2N
    push_force = jax.random.uniform(push_rng, minval=-2.0, maxval=2.0) * difficulty

    return {
        "terrain_type": terrain_type,
        "tilt_angle": tilt_angle,
        "roughness": roughness,
        "stair_height": stair_height,
        "push_force": push_force,
        "rng": rng,
    }


def sample_push_impulse(rng: jax.Array, difficulty: float, num_envs: int) -> jax.Array:
    """采样外部推力扰动 (向量化).

    design.md §2.4: 随机脉冲力 (2 N·s), 模拟侧推。
    每 episode 随机时刻施加一次, 持续 0.1s。

    Returns: (num_envs, 3) 外力向量 (X/Y/Z, N)
    """
    rng, k_force, k_dir, k_time = jax.random.split(rng, 4)
    magnitude = jax.random.uniform(k_force, (num_envs, 1), maxval=2.0) * difficulty
    direction = jax.random.normal(k_dir, (num_envs, 3))
    direction = direction / (jp.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8)
    # 只在水平面施加 (Z=0)
    direction = direction.at[:, 2].set(0.0)
    return magnitude * direction


CURRICULUM_STAGES = [
    {"name": "flat_balance",  "difficulty": 0.0, "threshold": 0.90},
    {"name": "slope",         "difficulty": 0.3, "threshold": 0.85},
    {"name": "rough",         "difficulty": 0.5, "threshold": 0.80},
    {"name": "stair_30mm",    "difficulty": 0.7, "threshold": 0.80},
    {"name": "perturbation",  "difficulty": 1.0, "threshold": 0.80},
]


class CurriculumController:
    """课程控制器: 按成功率自动解锁下一阶段.

    design.md §2.6: legged_gym 自动课程范式。
    """

    def __init__(self, stages=None):
        self.stages = stages or CURRICULUM_STAGES
        self.current_stage_idx = 0
        self.success_window = []  # 滑动窗口记录最近 episode 成功/失败
        self.window_size = 100

    def update(self, episode_success: bool) -> bool:
        """记录 episode 结果, 返回是否解锁下一阶段."""
        self.success_window.append(episode_success)
        if len(self.success_window) > self.window_size:
            self.success_window.pop(0)

        if len(self.success_window) < self.window_size:
            return False

        success_rate = sum(self.success_window) / len(self.success_window)
        threshold = self.stages[self.current_stage_idx]["threshold"]

        if success_rate >= threshold and self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.success_window.clear()
            print(f"[课程] 解锁阶段 {self.current_stage_idx}: "
                  f"{self.stages[self.current_stage_idx]['name']} "
                  f"(成功率 {success_rate:.1%} ≥ {threshold:.0%})")
            return True
        return False

    @property
    def difficulty(self) -> float:
        return self.stages[self.current_stage_idx]["difficulty"]

    @property
    def stage_name(self) -> str:
        return self.stages[self.current_stage_idx]["name"]
