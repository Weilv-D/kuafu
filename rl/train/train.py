# -*- coding: utf-8 -*-
"""
KUAFU Teacher PPO 训练入口 — design.md §2.6 阶段 1

MJX 环境 (JAX/GPU) → DirectVecEnv (DLPack 零拷贝) → RSL-RL 2.x PPO (PyTorch/GPU)
Teacher: critic 含特权信息 (friction/mass/COM/inertia), actor 仅本体感受。

运行:
  rl/.venv/bin/python rl/train/train.py --run_name garlic --num_envs 1024 --iterations 1000

产出:
  rl/checkpoints/<run_name>/run.json                 训练元数据
  rl/checkpoints/<run_name>/teacher/model_{iter}.pt  Teacher checkpoint
  rl/checkpoints/<run_name>/teacher/events.out.tfevents.*  TensorBoard
  rl/checkpoints/<run_name>/teacher/git/kuafu.diff   代码快照
"""
import os
import sys
import argparse
import time
import json
import glob
import random
import tempfile
import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.join(
    os.path.expanduser("~"), ".cache", "kuafu_jax")
# CUDA graphs 加速采集 (jax 0.10 用 JAX_ENABLE_CUDA_GRAPHS 环境变量,
# 旧的 XLA 级 --xla_gpu_enable_cuda_graphs 标志在本机 jax 构建已被移除)
os.environ["JAX_ENABLE_CUDA_GRAPHS"] = "1"

import jax
import jax.numpy as jp
import torch
from rl.train.seed_utils import seed_all, capture_provenance
from rl.train import dlpack_utils as dlu
from rl.env.kuafu_mjx_env import (
    KuafuMjxEnv, PRIVILEGED_DIM, RMA_STATIC_DIM, TRANSIENT_DIM,
    ACTOR_OBS_DIM, CRITIC_PRIV_DIM, CRITIC_OBS_DIM, DIFFICULTY_DIM,
)
from rl.train.curriculum import Curriculum as KuafuCurriculum, AXES, DIFF_INDICES, AXIS_CONFIG


def make_train_cfg() -> dict:
    """RSL-RL 2.x OnPolicyRunner 配置 — 全部源自 train_config (单一真相源)."""
    from rl.train.train_config import ALGORITHM, POLICY, RUN
    return {
        "algorithm": dict(ALGORITHM),
        "policy": dict(POLICY),
        "num_steps_per_env": RUN["num_steps_per_env"],
        "save_interval": RUN["save_interval"],
        "empirical_normalization": RUN["empirical_normalization"],
    }


# 8-axis independent curriculum 现统一实现于 rl/train/curriculum.py (见该模块
# AXIS_CONFIG / DIFF_INDICES)。本文件直接 import KuafuCurriculum 使用, 不再
# 维护标量 d_max 课程。难度采样改为 band[0.8*level, level] 的 per-axis 采样,
# 每轴独立按存活率(地形/扰动轴)或 存活率+跟踪(命令/D0 轴)升级/降级。


class DirectVecEnv:
    """JAX vmap 环境到 rsl_rl 2.x VecEnv 的直接适配器.

    绕过 playground 的 BraxAutoResetWrapper (其 auto-reset 会修改 info 结构导致
    scan pytree 不匹配), 直接用 JAX vmap + jax.lax.cond 做 auto-reset,
    通过 DLPack 与 PyTorch 零拷贝交换 GPU 张量。
    """
    def __init__(self, env, num_envs, seed, device="cuda", jax_key=None):
        self.env = env
        self.num_envs = num_envs
        self.num_actions = env.action_size
        self.num_obs = ACTOR_OBS_DIM                             # actor = 35 x 4 causal frames
        # Critic input = actor 140 + 12 simulation-only values = 152.
        self.num_privileged_obs = CRITIC_OBS_DIM if env._teacher else None
        self.device = device
        self.cfg = {"env_name": "kuafu", "num_envs": num_envs}
        self.max_episode_length = env._episode_length
        self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)

        # 8-axis 独立课程: 每轴按 AXIS_CONFIG 的门控独立升级/降级, 难度采样为
        # band[0.8*level, level] 的 per-axis 采样 (level = 当前轴等级/4)。
        self._curriculum = KuafuCurriculum()
        self._difficulty = jp.array(self._curriculum.difficulty_vector())  # (8,) per-axis 上限
        # 每轴累积的 episode 缓冲 (done-env 计数触发, min_episodes=256)
        self._ep_buf = {ax: [] for ax in AXES}
        # 诊断日志 (per-axis 跟踪分量)。预填 nan 使其从首步即在 log dict 中出现
        # (RSL-RL 只记录首个 log dict 中出现的键)。
        self._diag = {
            "diag_lin_vel_err_mean": float("nan"),
            "diag_yaw_err_mean": float("nan"),
            "diag_d0_err_mean": float("nan"),
        }
        for _ax in AXES:
            self._diag["diag_track_" + _ax + "_pass"] = float("nan")

        # 探索护栏 (防熵坍塌): 课程升级时重注噪声 std, noise_std 跌破地板时抬高 entropy_coef.
        # 引用由 train.py 在 runner 建好后注入 (self._actor_critic / self._alg); 未注入则护栏静默关闭.
        self._actor_critic = None
        self._alg = None
        self.noise_std_floor = 0.03     # 低于此值 -> 抬高 entropy_coef
        self.noise_std_recover = 0.06   # 高于此值 -> 恢复基线 entropy_coef
        self.entropy_coef_base = 0.01   # 与 train_config 默认一致
        self.entropy_coef_boost = 0.04  # 熵地板触发时的临时系数
        self.std_bump_on_upgrade = 0.15  # 课程升级时把 std 抬回此值 (仅抬高, 不降低) 重开探索

        self._reset_vmapped = jax.jit(
            jax.vmap(env.reset, in_axes=(0, 0)), donate_argnums=(0, 1))
        self._step_vmapped = jax.jit(jax.vmap(env.step), donate_argnums=(0,))

        self._rng = jax_key if jax_key is not None else jax.random.PRNGKey(seed)
        self._rng, diff_rng = jax.random.split(self._rng)
        diff_vec = self._sample_difficulty(diff_rng)
        self._state = self._reset_vmapped(jax.random.split(self._rng, num_envs), diff_vec)

    def _to_torch(self, x):
        """JAX DeviceArray → torch.Tensor (DLPack 零拷贝契约)."""
        return dlu.to_torch(x, device=self.device)

    def _to_jax(self, t):
        """torch.Tensor → JAX DeviceArray (DLPack 零拷贝契约)."""
        return dlu.to_jax(t, device=None)

    def get_observations(self):
        obs = self._state.obs
        state_obs = self._to_torch(obs["state"]) if isinstance(obs, dict) else self._to_torch(obs)
        extras = {"observations": {}}
        if isinstance(obs, dict) and "privileged_state" in obs:
            # critic receives actor observation plus all simulation-only values.
            priv_obs = self._to_torch(obs["privileged_state"])
            extras["observations"]["critic"] = torch.cat([state_obs, priv_obs], dim=-1)
        return state_obs, extras

    def reset(self):
        """VecEnv 接口要求: 重置所有环境."""
        self._rng, reset_rng, diff_rng = jax.random.split(self._rng, 3)
        # per-axis band[0.8*level, level] 采样: 每轴在当前等级附近采样, 既有 per-env
        # 多样性又能诚实评估当前等级 (避免永远卡在等级 0 测不到下一等级)。
        diff_vec = self._sample_difficulty(diff_rng)
        self._state = self._reset_vmapped(jax.random.split(reset_rng, self.num_envs), diff_vec)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        return self.get_observations()

    def _sample_difficulty(self, rng):
        """Per-axis band 采样: diff[a] ~ Uniform(0.8*upper_a, upper_a), 等级 0 时退化为 0。"""
        upper = self._difficulty  # (8,)
        lo = 0.8 * upper
        return jax.random.uniform(rng, (self.num_envs, DIFFICULTY_DIM), minval=lo, maxval=upper)

    @staticmethod
    def _pad_to_common_contacts(state_a, state_b):
        """Pad 两状态 Data 的 contact(ncon)/efc(nefc) 数组到两者 max, 使 where 合法。

        MJX 按实际接触数定长这些数组; 重置批次与步进批次的 max 接触数常不同
        (尤其粗糙地形), 直接 tree_map(where) 会因 pytree 结构不一致崩溃。pad 后
        ncon/nefc 标量叶保留每 env 实际值, 零填充槽位被 MJX 忽略。
        """
        impl_a = state_a.data._impl
        impl_b = state_b.data._impl
        ncon_a = int(impl_a.contact.dist.shape[1])
        ncon_b = int(impl_b.contact.dist.shape[1])
        max_ncon = max(ncon_a, ncon_b)
        nefc_a = int(impl_a.efc_J.shape[1])
        nefc_b = int(impl_b.efc_J.shape[1])
        max_nefc = max(nefc_a, nefc_b)

        def _pad_one(state, cur_ncon, cur_nefc):
            def _fn(x):
                if isinstance(x, jax.Array) and x.ndim >= 2:
                    if x.shape[1] == cur_ncon:
                        return jax.numpy.pad(x, [(0, 0), (0, max_ncon - cur_ncon)] + [(0, 0)] * (x.ndim - 2))
                    if x.shape[1] == cur_nefc:
                        return jax.numpy.pad(x, [(0, 0), (0, max_nefc - cur_nefc)] + [(0, 0)] * (x.ndim - 2))
                return x
            return jax.tree_util.tree_map(_fn, state)
        return _pad_one(state_a, ncon_a, nefc_a), _pad_one(state_b, ncon_b, nefc_b)

    def step(self, action):
        jax_action = self._to_jax(action)
        self._state = self._step_vmapped(self._state, jax_action)

        # 在 auto-reset 前读取 done/reward/metrics (reset 后会清零)
        done_jax = self._state.done
        reward_jax = self._state.reward
        metrics_jax = self._state.metrics
        fallen_jax = metrics_jax.get("fallen", jax.numpy.zeros_like(done_jax))

        # auto-reset done environments using JAX-side selective reset.
        # Avoids host sync for done_any when possible; only sync for curriculum stats.
        done_jax_bool = done_jax.astype(jax.numpy.bool_)
        done_any = bool(jax.device_get(done_jax.any()))
        if done_any:
            self._rng, reset_rng, diff_rng = jax.random.split(self._rng, 3)

            # 8-axis 独立课程: 每轴累积 done-env episode, 达到 min_episodes 触发升级/降级评估
            cur_env_state = self._state.info["env_state"]
            difficulty_np = jax.device_get(cur_env_state.difficulty)        # (N, 8)
            done_np = jax.device_get(done_jax)                              # (N,)
            survival_np = jax.device_get(cur_env_state.step_count)          # (N,)
            fallen_np = jax.device_get(fallen_jax)                          # (N,)
            count_np = np.maximum(jax.device_get(cur_env_state.track_count), 1)
            lin_vel_err_np = jax.device_get(cur_env_state.track_v_abs_sum) / count_np
            yaw_err_np = jax.device_get(cur_env_state.track_w_abs_sum) / count_np
            d0_err_np = jax.device_get(cur_env_state.track_d0_abs_sum) / count_np
            nonzero_np = jax.device_get(cur_env_state.nonzero_command_count) > 0
            # 诊断: 全局 err 均值 (跨所有 done env), 用于校准 track_err 门槛
            diag = {
                "diag_lin_vel_err_mean": float(lin_vel_err_np.mean()) if lin_vel_err_np.size else float("nan"),
                "diag_yaw_err_mean": float(yaw_err_np.mean()) if yaw_err_np.size else float("nan"),
                "diag_d0_err_mean": float(d0_err_np.mean()) if d0_err_np.size else float("nan"),
            }
            for axis_name in AXES:
                aidx = DIFF_INDICES[axis_name]
                cfg = AXIS_CONFIG[axis_name]
                # 该轴等级附近的 env 才计入 (band 采样下 difficulty 已在等级附近)
                rel = done_np & (difficulty_np[:, aidx] >= 0.5 * float(self._difficulty[aidx]))
                if not rel.any():
                    continue
                idx = np.where(rel)[0]
                survived = (fallen_np[idx] < 0.5)
                if cfg.track_metric == "linvel_yaw":
                    track_pass = ((lin_vel_err_np[idx] <= cfg.track_err["lin_vel"])
                                  & (yaw_err_np[idx] <= cfg.track_err["yaw"])
                                  & nonzero_np[idx])
                elif cfg.track_metric == "d0":
                    track_pass = (d0_err_np[idx] <= cfg.track_err["d0"])
                else:
                    track_pass = np.ones_like(survived, dtype=bool)
                diag["diag_track_" + axis_name + "_pass"] = float(track_pass.mean())
                episodes = [{"survived": bool(s), "track_pass": bool(t)}
                            for s, t in zip(survived, track_pass)]
                self._ep_buf[axis_name].extend(episodes)
                if len(self._ep_buf[axis_name]) >= self._curriculum.min_episodes:
                    changed = self._curriculum.update_axis(axis_name, self._ep_buf[axis_name])
                    self._ep_buf[axis_name] = []
                    if changed == "up":
                        self._reopen_exploration()
            self._diag = diag
            self._adjust_entropy_floor()
            self._difficulty = jp.array(self._curriculum.difficulty_vector())

            # Selective reset: compute reset for ALL envs (JAX-fast on GPU) then select
            diff_vec = self._sample_difficulty(diff_rng)
            reset_state = self._reset_vmapped(
                jax.random.split(reset_rng, self.num_envs), diff_vec)
            # MJX 把 contact(ncon)/efc(nefc) 数组按实际接触数定长, 故 reset 批次与
            # 步进批次的 max ncon/nefc 可能不同 (粗糙地形接触数远大于平地), 直接
            # where 会因 pytree 结构不一致崩溃。先 pad 到两者 max, 再 where (ncon
            # 标量叶保留每 env 实际值, 零填充槽位被 MJX 忽略)。
            self._state, reset_state = self._pad_to_common_contacts(self._state, reset_state)
            self._state = jax.tree_util.tree_map(
                lambda cur, new: jax.numpy.where(
                    done_jax_bool.reshape((-1,) + (1,) * (cur.ndim - 1)), new, cur),
                self._state, reset_state)

        # done 帧返回 reset 后的初始观测 (PPO 新 episode 首步用初始 obs)
        state_obs, extras = self.get_observations()
        reward = self._to_torch(reward_jax)
        done = self._to_torch(done_jax)

        self.episode_length_buf += 1

        # 收集 episode 级指标到 info["log"] (RSL-RL 自动写入 TensorBoard)
        # episode 级指标仅在有环境 done 时才填充, 其余步留空 {} - RSL-RL 收集器遇空 dict 自动跳过,
        # 避免中途帧的 0.0 被计入均值导致指标被稀释趋零
        log = {}
        if done_any:
            done_mask = (done > 0)
            n_done = done_mask.sum().clamp(min=1)
            # episode_length 在清零前读取 (上面 +1 后, done 帧的值即该 episode 总长)
            log["episode_length"] = (self.episode_length_buf * done_mask).sum().item() / n_done.item()
            for key in ["orientation", "lin_vel_tracking"]:
                if key in metrics_jax:
                    val = self._to_torch(metrics_jax[key])
                    log[key] = (val * done_mask).sum().item() / n_done.item()
            # 探索护栏可观测量 (entropy_coef 实时值 + 当前 noise_std)
            log["entropy_coef"] = self._alg.entropy_coef if self._alg is not None else float("nan")
            if self._actor_critic is not None and hasattr(self._actor_critic, "std"):
                log["noise_std_guard"] = float(self._actor_critic.std.data.mean().item())
        # 课程等级 + 诊断 + difficulty 每步记录 (RSL-RL 只记录首个 log dict 中出现的
        # 键, 故这些键必须每步都存在, 否则 done 帧首次出现前不会被写入 TB)。
        for axis_name in AXES:
            log["curriculum_" + axis_name + "_level"] = float(self._curriculum.axes[axis_name].level)
            log["difficulty_" + axis_name] = float(self._difficulty[DIFF_INDICES[axis_name]])
        if getattr(self, "_diag", None) is not None:
            for k, v in self._diag.items():
                log[k] = v
        log["difficulty_mean"] = self._to_torch(
            self._state.info["env_state"].difficulty).mean().item()

        self.episode_length_buf = torch.where(
            done > 0, torch.zeros_like(self.episode_length_buf), self.episode_length_buf)

        # time_outs: 仅 timeout(非倒下) 时为 True, 用于 value bootstrap
        # 倒下 (fallen) 的 episode 不做 bootstrap (终止态 value=0)
        fallen = self._to_torch(fallen_jax)
        time_outs = (done > 0) & (fallen < 0.5)  # done 但未倒下 = 超时
        info = {"time_outs": time_outs.float(),
                "observations": extras.get("observations", {}), "log": log}
        return state_obs, reward, done, info

    @staticmethod
    def _host_tree(tree):
        return jax.tree_util.tree_map(
            lambda value: np.asarray(jax.device_get(value)) if hasattr(value, "shape") else value,
            tree)

    @staticmethod
    def _device_tree(tree):
        return jax.tree_util.tree_map(
            lambda value: jax.numpy.asarray(value) if isinstance(value, np.ndarray) else value,
            tree)

    def checkpoint_state(self):
        """Capture vectorized MJX state, delays, integrators, and RNGs for exact resume."""
        return {
            "rng": np.asarray(jax.device_get(self._rng)),
            "state": self._host_tree(self._state),
            "episode_length": self.episode_length_buf.detach().cpu(),
        }

    def restore_checkpoint_state(self, snapshot):
        self._rng = jax.numpy.asarray(snapshot["rng"])
        self._state = self._device_tree(snapshot["state"])
        self.episode_length_buf = snapshot["episode_length"].to(self.device)

    def _reopen_exploration(self):
        """课程升级时把策略噪声 std 抬回下限, 重开探索.

        仅抬高 (clamp_min), 绝不降低: 课程升级发生在已 mastery (fall_rate≤0.5) 时,
        彼时 std 已偏低, 抬回可让策略重新采样恢复动作以扛住更强扰动; 若 std 本就偏高则不动.
        假定 noise_std_type="scalar" (ActorCritic.std 为可学习参数).
        """
        if self._actor_critic is None or not hasattr(self._actor_critic, "std"):
            return
        with torch.no_grad():
            self._actor_critic.std.data.clamp_(min=self.std_bump_on_upgrade)

    def _adjust_entropy_floor(self):
        """熵地板 (AE-PPO target-entropy 简化版): noise_std 跌破下限则抬高 entropy_coef,
        回升过恢复阈值则回基线, 滞回避免抖动. 防长程课程推到强扰动难度时策略熵坍塌、丧失可塑性.
        """
        if self._alg is None or self._actor_critic is None or not hasattr(self._actor_critic, "std"):
            return
        noise_std = float(self._actor_critic.std.data.mean().item())
        if noise_std < self.noise_std_floor:
            self._alg.entropy_coef = self.entropy_coef_boost
        elif noise_std > self.noise_std_recover:
            self._alg.entropy_coef = self.entropy_coef_base

    @property
    def unwrapped(self):
        return self.env

    @property
    def step_dt(self):
        return self.env.dt


class _CurriculumPersistMixin:
    """让 checkpoint 同时保存/恢复 8-axis 课程的每轴等级与 episode 缓冲, 避免
    --resume 把课程重置回全 0。

    rsl_rl 的 save/load 不存 Curriculum 状态; 本 mixin 在 checkpoint 内联持久化
    curriculum.state_dict (每轴 level/streak/fail_streak) + episode_buffers +
    kuafu_curriculum_schema="v2"。legacy 5-axis 旧 checkpoint 因 schema 不匹配
    拒绝 resume (8 轴难度维度不兼容)。
    """
    def save(self, path, infos=None):
        # RSL-RL saves a complete policy/optimizer state.  Route that write through
        # a sibling temporary file, then add KUAFU's curriculum/RNG/schema state and
        # atomically replace the destination so interrupted saves never yield a
        # half-checkpoint.
        directory = os.path.dirname(path)
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=directory)
        os.close(fd)
        try:
            super().save(temporary, infos)
            checkpoint = torch.load(temporary, map_location="cpu", weights_only=False)
            cur = self.env._curriculum
            checkpoint["kuafu_state"] = {
                "schema_version": __import__("rl.env.contract", fromlist=["SCHEMA_VERSION"]).SCHEMA_VERSION,
                "model_hash": __import__("kuafu_physics").model_hash(),
                "curriculum": {
                    "schema": "v2",
                    "state": cur.state_dict(),
                },
                "episode_buffers": {ax: list(self.env._ep_buf[ax]) for ax in AXES},
                "entropy_coef": float(self.alg.entropy_coef),
                "learning_rate": float(self.alg.learning_rate),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "jax_rng": jax.device_get(self.env._rng),
                "environment": self.env.checkpoint_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, path, load_optimizer=True, ignore_hash=False):
        # Load the full checkpoint on CPU first.  load_state_dict transfers
        # model/optimizer tensors to the runner device automatically, and CPU
        # RNG state must be restored on CPU regardless of the runner device.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("kuafu_state")
        if state is None:
            raise RuntimeError("checkpoint lacks KUAFU resume state; legacy checkpoints cannot resume")
        from rl.env.contract import SCHEMA_VERSION
        import kuafu_physics as P
        hash_ok = state["schema_version"] == SCHEMA_VERSION and (
            ignore_hash or state["model_hash"] == P.model_hash())
        if not hash_ok:
            raise RuntimeError("checkpoint schema/model hash does not match this run")
        # Move model state dict to runner device before loading
        model_state = {k: v.to(self.device) for k, v in checkpoint["model_state_dict"].items()}
        self.alg.policy.load_state_dict(model_state)
        if self.alg.rnd and checkpoint.get("rnd_state_dict") is not None:
            self.alg.rnd.load_state_dict(checkpoint["rnd_state_dict"])
        if load_optimizer:
            opt_state = checkpoint["optimizer_state_dict"]
            self.alg.optimizer.load_state_dict(opt_state)
            self.alg.optimizer.zero_grad()
        self.current_learning_iteration = checkpoint["iter"]
        cur_state = state["curriculum"]
        if cur_state.get("schema") != "v2":
            raise RuntimeError(
                "legacy 5-axis curriculum checkpoint (schema=%r) cannot resume; "
                "start a new run_name (8-axis curriculum is incompatible)."
                % cur_state.get("schema"))
        cur = self.env._curriculum
        cur.load_state_dict(cur_state["state"])
        self.env._ep_buf = {ax: list(state.get("episode_buffers", {}).get(ax, [])) for ax in AXES}
        self.env._difficulty = jp.array(cur.difficulty_vector())
        self.alg.entropy_coef = float(state["entropy_coef"])
        self.alg.learning_rate = float(state["learning_rate"])
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate
        # CPU RNG stays on CPU (must not use map_location=device)
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if state.get("environment") is None:
            raise RuntimeError("checkpoint lacks exact vectorized environment state")
        self.env.restore_checkpoint_state(state["environment"])
        # Only restore CUDA RNG if CUDA is available
        if state.get("torch_cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
        return checkpoint["infos"]


def main():
    parser = argparse.ArgumentParser(description="KUAFU Teacher PPO Training")
    from rl.train.train_config import RUN
    parser.add_argument("--num_envs", type=int, default=RUN["num_envs"], help="并行环境数")
    parser.add_argument("--iterations", type=int, default=RUN["iterations"], help="训练迭代数")
    parser.add_argument("--seed", type=int, default=RUN["seed"], help="随机种子")
    parser.add_argument("--num_steps_per_env", type=int, default=RUN["num_steps_per_env"],
                        help="每次 rollout 步数 (默认 72)")
    parser.add_argument("--run_name", type=str, required=True,
                        help="训练代号(如 garlic),产物存至 rl/checkpoints/<run_name>/teacher/")
    parser.add_argument("--log_dir", type=str, default="rl/checkpoints", help="checkpoint 根目录")
    parser.add_argument("--smoke_test", action="store_true", help="烟测模式 (5 iteration)")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练(传 .pt 路径,如 rl/checkpoints/garlic/teacher/model_3999.pt)")
    parser.add_argument("--resume_ignore_hash", action="store_true",
                        help="恢复时跳过 model_hash 校验(仅校验 schema_version)。"
                             "用于仅在地形/资源等良性资产变更后继续旧 checkpoint 的训练/诊断。")
    args = parser.parse_args()

    # 统一播种所有 RNG (torch/numpy/random 与 JAX 显式 key 同源)
    jax_key = seed_all(args.seed)

    print("=" * 60)
    print("KUAFU Teacher PPO Training (design.md §2.6 阶段 1)")
    print("=" * 60)
    print(f"  并行环境: {args.num_envs}")
    print(f"  JAX 设备: {jax.devices()}")

    # ---- 创建环境 ----
    # 课程: 8-axis 独立课程 (rl/train/curriculum.py), per-axis band[0.8*level, level]
    # 采样; 地形 (斜坡/台阶/粗糙 hfield) 由 KuafuMjxEnv._apply_terrain 按 difficulty 生成。

    env = KuafuMjxEnv(teacher=True, num_envs=args.num_envs)

    # 解析统一计算设备 (无 GPU 时回退 CPU 并告警)
    device = dlu.resolve_device("cuda")
    # DLPack 零拷贝契约守卫 (启动期一次)
    dlu.verify_dlpack_zero_copy(device)

    # ---- 直接适配 rsl_rl 2.x (绕过 playground brax wrapper 的 info 结构限制) ----
    torch_env = DirectVecEnv(env, args.num_envs, args.seed, device=device, jax_key=jax_key)
    print(f"  obs={torch_env.num_obs}, privileged={torch_env.num_privileged_obs}, "
          f"action={torch_env.num_actions}")

    # ---- 维度一致性守卫 (防止规格再次漂移) ----
    assert torch_env.num_privileged_obs == CRITIC_OBS_DIM, \
        f"critic 总维度错: {torch_env.num_privileged_obs} != {CRITIC_OBS_DIM}"
    assert RMA_STATIC_DIM + TRANSIENT_DIM == PRIVILEGED_DIM, \
        f"特权拆分错: {RMA_STATIC_DIM}+{TRANSIENT_DIM} != {PRIVILEGED_DIM}"
    assert torch_env.num_obs == ACTOR_OBS_DIM, \
        f"actor obs 维度错: {torch_env.num_obs} != {ACTOR_OBS_DIM}"
    assert ACTOR_OBS_DIM + CRITIC_PRIV_DIM == CRITIC_OBS_DIM, \
        f"critic 总维度错: {ACTOR_OBS_DIM}+{CRITIC_PRIV_DIM} != {CRITIC_OBS_DIM}"

    # ---- 训练配置 ----
    train_cfg = make_train_cfg()
    train_cfg["num_steps_per_env"] = args.num_steps_per_env

    # ---- 日志目录: rl/checkpoints/<run_name>/teacher/ ----
    run_root = os.path.join(PROJ_ROOT, args.log_dir, args.run_name)
    log_dir = os.path.join(run_root, "teacher")

    if args.smoke_test:
        # Smoke runs are disposable and never share a formal run directory.
        run_root = os.path.join(PROJ_ROOT, args.log_dir, "_smoke", f"{args.run_name}-{int(time.time())}")
        log_dir = os.path.join(run_root, "teacher")
    if args.resume and os.path.abspath(os.path.dirname(args.resume)) == os.path.abspath(log_dir):
        raise SystemExit("resume output must use a new --run_name; source checkpoints are immutable")

    # 防覆盖校验: 目录已存在且含 .pt, 且非续训 -> 报错
    existing = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if existing and not args.resume:
        print(f"❌ 目录已含 checkpoint: {log_dir}")
        print(f"   续训请加 --resume <latest.pt>, 或换 --run_name")
        sys.exit(1)

    os.makedirs(log_dir, exist_ok=True)

    # ---- 写训练元数据 run.json ----
    from rl.env.contract import SCHEMA_VERSION
    import kuafu_physics as P
    run_meta = {
        "run_name": args.run_name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "seed": args.seed,
        "resume_from": args.resume,
        "algorithm": "PPO",
        "policy": "TanhActorCritic [512,512,512] elu",
        "device": device,
        "schema_version": SCHEMA_VERSION,
        "model_hash": P.model_hash(),
        "provenance": capture_provenance(),
    }
    meta_path = os.path.join(run_root, "run.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    # ---- RSL-RL Runner ----
    from rsl_rl.runners import OnPolicyRunner
    # 注入 TanhActorCritic 到 runner 命名空间: runner 用 eval(policy_cfg["class_name"])
    # 解析策略类, 故需让 "TanhActorCritic" 可解析 (audit P0: 训练=无界 vs 部署=tanh)。
    from rl.train.tanh_actor_critic import TanhActorCritic
    import rsl_rl.runners.on_policy_runner as _runner_mod
    _runner_mod.TanhActorCritic = TanhActorCritic
    train_cfg["policy"]["class_name"] = "TanhActorCritic"

    class _StepRunner(_CurriculumPersistMixin, OnPolicyRunner):
        pass
    runner = _StepRunner(torch_env, train_cfg, log_dir=log_dir, device=device)
    print(f"  日志: {log_dir}")

    # ---- 载入 Checkpoint 恢复训练 ----
    if args.resume:
        print(f"  载入 Checkpoint 恢复训练: {args.resume}")
        runner.load(args.resume, ignore_hash=args.resume_ignore_hash)

    # 注入探索护栏引用 (Curriculum 升级 bump std / noise_std 地板抬 entropy_coef).
    # DirectVecEnv 内部据此在难度跳变与熵跌破地板时介入; 不注入则护栏静默关闭.
    torch_env._actor_critic = runner.alg.policy
    torch_env._alg = runner.alg

    # ---- 训练 ----
    start_iter = runner.current_learning_iteration
    n_iter = 5 if args.smoke_test else args.iterations
    run_iter = max(0, n_iter - start_iter)

    if args.smoke_test:
        print("🔥 烟测: 5 iteration")
        run_iter = 5

    total_steps = args.num_envs * train_cfg["num_steps_per_env"] * run_iter
    print(f"开始训练: 需进行 {run_iter} 轮迭代 (已完成 {start_iter} 轮, 目标 {n_iter} 轮) × {args.num_envs} envs × {train_cfg['num_steps_per_env']} steps = {total_steps:,} steps")
    t0 = time.time()
    runner.learn(num_learning_iterations=run_iter, init_at_random_ep_len=not bool(args.resume))
    elapsed = time.time() - t0

    # Fail-fast: check for NaN in model parameters
    for name, param in runner.alg.policy.named_parameters():
        if not torch.isfinite(param).all():
            print(f"FATAL: NaN detected in parameter {name}")
            sys.exit(1)

    print(f"\n✅ 训练完成: {elapsed:.1f}s, {total_steps:,} steps, {total_steps/elapsed:,.0f} steps/s")
    if not args.smoke_test:
        final = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
        print(f"   Checkpoint: {final}")
        print(f"   导出: rl/.venv/bin/python rl/export/export_policy.py --ckpt {final}")


if __name__ == "__main__":
    main()
