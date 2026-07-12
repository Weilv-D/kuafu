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

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.join(
    os.path.expanduser("~"), ".cache", "kuafu_jax")
# CUDA graphs 加速采集 (jax 0.10 用 JAX_ENABLE_CUDA_GRAPHS 环境变量,
# 旧的 XLA 级 --xla_gpu_enable_cuda_graphs 标志在本机 jax 构建已被移除)
os.environ["JAX_ENABLE_CUDA_GRAPHS"] = "1"

import jax
import torch
from rl.train.seed_utils import seed_all, capture_provenance
from rl.train import dlpack_utils as dlu
from rl.env.kuafu_mjx_env import (
    KuafuMjxEnv, OBS_DIM, PRIVILEGED_DIM, RMA_STATIC_DIM, TRANSIENT_DIM,
    ACTOR_OBS_DIM, CRITIC_PRIV_DIM, CRITIC_OBS_DIM,
)


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


class Curriculum:
    """全局课程: 高难度环境存活指标双向调节 difficulty (DR/扰动强度).

    仅统计 per-env 采样中 difficulty > d_max×0.7 的高难度环境 (避免低难度虚高),
    以滑动窗口统计其平均存活步数与摔倒率:

      - 升级: avg_survival ≥ 800 且 fall_rate ≤ 0.5 → d_max += step (直到 1.0)
      - 降级: avg_survival ≤ 600 或 fall_rate ≥ 0.65 → d_max -= step (下限 0.1)

    设计参考 ETH legged_gym: 以"能否稳定存活 + 不倒"驱动难度渐进, 而非要求活满
    固定时长; 训练初期即注入 DR + 随机推力, 防过拟合标称参数。双向调节让策略
    退化 (如熵坍缩后) 时回退到可驾驭难度恢复, 避免永久卡死。
    """

    def __init__(self, start: float = 0.1, max_d: float = 1.0, step: float = 0.05,
                 window: int = 200, min_d: float = 0.1,
                 upgrade_avg_survival: float = 800.0, upgrade_fall_rate: float = 0.5,
                 fallback_avg_survival: float = 600.0, fallback_fall_rate: float = 0.65):
        self.d = start
        self.max_d = max_d
        self.min_d = min_d
        self.step = step
        self.window = window
        self.upgrade_avg_survival = upgrade_avg_survival
        self.upgrade_fall_rate = upgrade_fall_rate
        self.fallback_avg_survival = fallback_avg_survival
        self.fallback_fall_rate = fallback_fall_rate
        self._surv_buf = []   # 高难度 done env 存活步数
        self._fall_buf = []    # 高难度 done env 是否摔倒 (1/0)
        self._last_avg_survival = float("nan")
        self._last_fall_rate = float("nan")

    def update(self, survival_steps, fell):
        """survival_steps / fell: 本批高难度 done 环境的存活步数与摔倒标志 (numpy 数组).

        返回 "up" / "down" / None, 供调用方在难度跳变时重注探索噪声.
        """
        for s, f in zip(survival_steps, fell):
            self._surv_buf.append(float(s))
            self._fall_buf.append(float(f))
        if len(self._surv_buf) > self.window:
            del self._surv_buf[: len(self._surv_buf) - self.window]
        # 窗口未满时仍记录真实(部分)统计, 避免 resume 首轮 NaN 污染 TB 曲线;
        # 但 d_max 的升降仅在窗口填满后决策, 防止部分样本误判。
        if self._surv_buf:
            self._last_avg_survival = sum(self._surv_buf) / len(self._surv_buf)
            self._last_fall_rate = sum(self._fall_buf) / len(self._fall_buf)
        else:
            self._last_avg_survival = float("nan")
            self._last_fall_rate = float("nan")
        if len(self._surv_buf) < self.window:
            return None
        avg_survival = self._last_avg_survival
        fall_rate = self._last_fall_rate
        if avg_survival >= self.upgrade_avg_survival and fall_rate <= self.upgrade_fall_rate:
            if self.d < self.max_d:
                self.d = min(self.max_d, self.d + self.step)
                return "up"
            return None
        elif avg_survival <= self.fallback_avg_survival or fall_rate >= self.fallback_fall_rate:
            if self.d > self.min_d:
                self.d = max(self.min_d, self.d - self.step)
                return "down"
            return None
        return None


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
        self.num_obs = ACTOR_OBS_DIM                             # actor = proprio(148) + z(9)
        # critic 输入 = actor(157) + 瞬态特权(3) = 160 (RSL-RL 实际从 extras 取此维,
        # 但此处显式声明以免未来版本误读 env.num_privileged_obs 而建错 critic)。
        self.num_privileged_obs = CRITIC_OBS_DIM if env._teacher else None
        self.device = device
        self.cfg = {"env_name": "kuafu", "num_envs": num_envs}
        self.max_episode_length = env._episode_length
        self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)

        # 课程: d_max 由高难度环境平均存活步数 + 摔倒率双向调节, per-env 采样 Uniform(0, d_max)
        # 训练初期即注入 DR + 随机推力, 防过拟合标称参数, ETH legged_gym 实践
        self._curriculum = Curriculum(start=0.1, max_d=1.0, step=0.05, window=200,
                                      upgrade_avg_survival=800.0, upgrade_fall_rate=0.5,
                                      fallback_avg_survival=600.0, fallback_fall_rate=0.65)
        self._difficulty = jax.numpy.float32(self._curriculum.d)  # d_max (课程上界)

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
        diff_vec = jax.random.uniform(diff_rng, (num_envs,),
                                      minval=0.0, maxval=self._difficulty)
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
            # critic 吃 actor obs (157) + 瞬态特权 (3) = 160
            priv_obs = self._to_torch(obs["privileged_state"])
            extras["observations"]["critic"] = torch.cat([state_obs, priv_obs], dim=-1)
        return state_obs, extras

    def reset(self):
        """VecEnv 接口要求: 重置所有环境."""
        self._rng, reset_rng, diff_rng = jax.random.split(self._rng, 3)
        # per-env 独立采样难度 Uniform(0, d_max), 策略同时面对简单与困难场景
        diff_vec = jax.random.uniform(diff_rng, (self.num_envs,),
                                      minval=0.0, maxval=self._difficulty)
        self._state = self._reset_vmapped(jax.random.split(reset_rng, self.num_envs), diff_vec)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        return self.get_observations()

    def step(self, action):
        jax_action = self._to_jax(action)
        self._state = self._step_vmapped(self._state, jax_action)

        # 在 auto-reset 前读取 done/reward/metrics (reset 后会清零)
        done_jax = self._state.done
        reward_jax = self._state.reward
        metrics_jax = self._state.metrics
        fallen_jax = metrics_jax.get("fallen", jax.numpy.zeros_like(done_jax))

        # auto-reset done 环境 (保持 JAX array 在 GPU 上)
        done_any = jax.device_get(done_jax.any())
        if done_any:
            self._rng, reset_rng, diff_rng = jax.random.split(self._rng, 3)

            # 课程: 仅统计 done 且高难度 (difficulty > d_max×0.7) 环境的存活步数与摔倒率
            # 避免低难度虚高 + 非 done 稀释; 升级条件 avg_survival>800 且 fall_rate<0.5
            cur_env_state = self._state.info["env_state"]
            high_diff = cur_env_state.difficulty > (self._difficulty * 0.7)
            relevant = done_jax & high_diff
            relevant_np = jax.device_get(relevant)
            if relevant_np.any():
                survival_np = jax.device_get(cur_env_state.step_count)
                fallen_np = jax.device_get(fallen_jax)
                changed = self._curriculum.update(survival_np[relevant_np], fallen_np[relevant_np])
                if changed == "up":
                    self._reopen_exploration()
            self._adjust_entropy_floor()
            self._difficulty = jax.numpy.float32(self._curriculum.d)

            # per-env 独立采样难度 Uniform(0, d_max)
            diff_vec = jax.random.uniform(diff_rng, (self.num_envs,),
                                          minval=0.0, maxval=self._difficulty)
            reset_state = self._reset_vmapped(
                jax.random.split(reset_rng, self.num_envs), diff_vec)
            done_mask = done_jax.astype(jax.numpy.bool_)
            self._state = jax.tree_util.tree_map(
                lambda cur, new: jax.numpy.where(
                    done_mask.reshape((-1,) + (1,) * (cur.ndim - 1)), new, cur),
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
            # 课程高难度窗口指标: 平均存活步数 + 摔倒率 (驱动 d_max 升降)
            log["curriculum_avg_survival"] = self._curriculum._last_avg_survival
            log["curriculum_fall_rate"] = self._curriculum._last_fall_rate
            # 探索护栏可观测量 (entropy_coef 实时值 + 当前 noise_std)
            log["entropy_coef"] = self._alg.entropy_coef if self._alg is not None else float("nan")
            if self._actor_critic is not None and hasattr(self._actor_critic, "std"):
                log["noise_std_guard"] = float(self._actor_critic.std.data.mean().item())
        # difficulty 每步记录 (d_max 课程标量 + per-env 实际均值)
        log["difficulty"] = self._to_torch(self._difficulty).mean().item()
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
    """让 checkpoint 同时保存/恢复 Curriculum 的 d_max, 避免 --resume 把课程重置回 0.1。

    rsl_rl 的 save/load 不存 Curriculum 状态; 本 mixin 在同名 sidecar
    curriculum_{it}.pt 中额外持久化 d_max, resume 时回写 env._curriculum.d。
    """
    def save(self, path, infos=None):
        super().save(path, infos)
        try:
            cur = self.env._curriculum
            sidecar = path.replace("model_", "curriculum_")
            torch.save({
                "d": float(cur.d),
                "difficulty": float(self.env._difficulty),
                # 持久化滑动窗口缓冲与最近统计, 避免 resume 后课程冷启动 / TB NaN
                "surv_buf": list(cur._surv_buf),
                "fall_buf": list(cur._fall_buf),
                "last_avg_survival": float(cur._last_avg_survival),
                "last_fall_rate": float(cur._last_fall_rate),
            }, sidecar)
        except Exception:
            pass

    def load(self, path, load_optimizer=True):
        infos = super().load(path, load_optimizer)
        sidecar = path.replace("model_", "curriculum_")
        if os.path.exists(sidecar):
            try:
                sd = torch.load(sidecar, weights_only=False)
                self.env._curriculum.d = float(sd["d"])
                self.env._difficulty = jax.numpy.float32(sd["d"])
                cur = self.env._curriculum
                cur._surv_buf = list(sd.get("surv_buf", []))
                cur._fall_buf = list(sd.get("fall_buf", []))
                cur._last_avg_survival = float(sd.get("last_avg_survival", float("nan")))
                cur._last_fall_rate = float(sd.get("last_fall_rate", float("nan")))
            except Exception:
                pass
        return infos


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
    args = parser.parse_args()

    # 统一播种所有 RNG (torch/numpy/random 与 JAX 显式 key 同源)
    jax_key = seed_all(args.seed)

    print("=" * 60)
    print("KUAFU Teacher PPO Training (design.md §2.6 阶段 1)")
    print("=" * 60)
    print(f"  并行环境: {args.num_envs}")
    print(f"  JAX 设备: {jax.devices()}")

    # ---- 创建环境 ----
    # 课程: 双向滑动窗口调节 d_max (Curriculum 类, 初始 0.1 即注入 DR+扰动),
    # per-env 采样 Uniform(0, d_max); 地形(斜坡+台阶)由 KuafuMjxEnv._apply_terrain 按 difficulty 生成。

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

    # 防覆盖校验: 目录已存在且含 .pt, 且非续训 -> 报错
    existing = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if existing and not args.resume and not args.smoke_test:
        print(f"❌ 目录已含 checkpoint: {log_dir}")
        print(f"   续训请加 --resume <latest.pt>, 或换 --run_name")
        sys.exit(1)

    os.makedirs(log_dir, exist_ok=True)

    # ---- 写训练元数据 run.json ----
    run_meta = {
        "run_name": args.run_name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "seed": args.seed,
        "resume_from": args.resume,
        "algorithm": "PPO",
        "policy": "ActorCritic [512,512,512] elu",
        "device": device,
        "provenance": capture_provenance(),
    }
    meta_path = os.path.join(run_root, "run.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    # ---- RSL-RL Runner ----
    from rsl_rl.runners import OnPolicyRunner
    class _StepRunner(_CurriculumPersistMixin, OnPolicyRunner):
        pass
    runner = _StepRunner(torch_env, train_cfg, log_dir=log_dir, device=device)
    print(f"  日志: {log_dir}")

    # ---- 载入 Checkpoint 恢复训练 ----
    if args.resume:
        print(f"  载入 Checkpoint 恢复训练: {args.resume}")
        runner.load(args.resume)

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
    runner.learn(num_learning_iterations=run_iter, init_at_random_ep_len=True)
    elapsed = time.time() - t0

    print(f"\n✅ 训练完成: {elapsed:.1f}s, {total_steps:,} steps, {total_steps/elapsed:,.0f} steps/s")
    if not args.smoke_test:
        final = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
        print(f"   Checkpoint: {final}")
        print(f"   导出: rl/.venv/bin/python rl/export/export_policy.py --ckpt {final}")


if __name__ == "__main__":
    main()
