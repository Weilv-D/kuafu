# -*- coding: utf-8 -*-
"""
KUAFU 课程/地形 设计参考 (design.md §2.6 / §3.4)

[状态: 参考规格, 未接入训练] 本模块是课程阶段与地形参数的*设计参考*,
当前 train.py 的 DirectVecEnv.step 已实现等价的内联课程逻辑 (per-episode
按 success 调整 difficulty), 不 import 本模块。KUAFU_MJX_ENV 也在 step 内
联实现了推力扰动 (xfrc_applied) 与 difficulty 缩放的域随机化, 同样不调用
本模块的 get_terrain_params / sample_push_impulse。

→ 因此: 本模块的函数/类为参考实现, 修改它们不会自动影响训练。
  若要将地形/推力真正接入训练, 应在 KuafuMjxEnv.step 内调用, 或把
  DirectVecEnv 的内联课程替换为下面的 CurriculumController。

课程阶段 (difficulty 0.0→1.0, 与 train.py 内联逻辑范围一致):
  0.0: plane (平地平衡)
  0.3: plane_tilt (坡度 ±10°)
  0.5: hfield (随机粗糙噪声)
  0.7: mesh_stair (30mm 台阶, M4 验收)
  1.0: perturbation (平地 + 随机推力扰动)
"""
import jax
import jax.numpy as jp
import numpy as np


def get_terrain_params(difficulty: float, rng: jax.Array) -> dict:
    """按课程难度生成地形参数 (JIT 兼容).

    使用 difficulty 连续缩放而非 if/elif 分支, 保证 jax.jit/vmap 安全。
    terrain_type 由 host 侧 CurriculumController 决定 (Python if), 仅传 difficulty 标量给 JAX。

    Args:
        difficulty: 0.0 (平地) → 1.0 (最难), 连续值
        rng: JAX 随机数种子

    Returns:
        dict: tilt_angle, roughness, stair_height, push_force (均为 JAX array)
    """
    rng, tilt_rng, rough_rng, stair_rng, push_rng = jax.random.split(rng, 5)

    # 所有参数由 difficulty 连续缩放 (无 if 分支, JIT 安全)
    tilt_angle = jax.random.uniform(tilt_rng, minval=-1, maxval=1) * jp.radians(10) * difficulty
    roughness = jax.random.uniform(rough_rng, maxval=0.015) * difficulty
    stair_height = jax.random.uniform(stair_rng, maxval=0.030) * difficulty
    push_force = jax.random.uniform(push_rng, minval=-2.0, maxval=2.0) * difficulty

    return {
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
