# -*- coding: utf-8 -*-
"""jax 侧采集 rollout: 把整段 PPO 采集塞进单个 lax.scan。

仅复刻 RSL-RL 采集语义 (归一顺序 / auto-reset / 课程难度采样) 与 actor/critic 前向,
不改动任何物理/求解器/精度。所有权重从 teacher .pt 载入, 与 torch 数值对齐 (S0 gate)。
所有 scan 输出均为静态形状 [T, N, ...]; 任何依赖运行数据长度的筛选在 Python 端做
(见 runner_scan.py 的课程提取), 绝不在 jit 内做 arr[mask]。
"""
import os
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

import jax
import jax.numpy as jnp
from jax import nn as jnn
from jax import tree_util

import numpy as np
import torch

from rl.train import dlpack_utils as dlu


def _t2j(t):
    return jnp.asarray(t.detach().cpu().numpy())


def load_ckpt_weights(ckpt_path):
    import torch

    sd = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    m = sd["model_state_dict"]
    _map = {"0.weight": "w0", "0.bias": "b0", "2.weight": "w2", "2.bias": "b2",
            "4.weight": "w4", "4.bias": "b4", "6.weight": "w6", "6.bias": "b6"}
    actor = {v: _t2j(m[f"actor.{k}"]) for k, v in _map.items()}
    critic = {v: _t2j(m[f"critic.{k}"]) for k, v in _map.items()}
    std = _t2j(m["std"])
    obs_norm = {
        "mean": _t2j(sd["obs_norm_state_dict"]["_mean"]),
        "var": _t2j(sd["obs_norm_state_dict"]["_var"]),
        "count": _t2j(sd["obs_norm_state_dict"]["count"]),
    }
    priv_norm = {
        "mean": _t2j(sd["privileged_obs_norm_state_dict"]["_mean"]),
        "var": _t2j(sd["privileged_obs_norm_state_dict"]["_var"]),
        "count": _t2j(sd["privileged_obs_norm_state_dict"]["count"]),
    }
    return {"actor": actor, "critic": critic, "std": std, "obs_norm": obs_norm, "priv_norm": priv_norm}


def torch_norm_state_to_jax(norm_module):
    sd = norm_module.state_dict()
    return {
        "mean": _t2j(sd["_mean"]),
        "var": _t2j(sd["_var"]),
        "count": _t2j(sd["count"]),
    }


def jax_norm_state_to_torch(state):
    import torch

    return {
        "_mean": torch.from_numpy(jnp.asarray(state["mean"])).to(torch.float32),
        "_var": torch.from_numpy(jnp.asarray(state["var"])).to(torch.float32),
        "_std": torch.from_numpy(jnp.asarray(jnp.sqrt(state["var"]))).to(torch.float32),
        "count": torch.tensor(int(state["count"].item()), dtype=torch.long),
    }


def mlp_forward(p, x):
    x = jnn.elu(x @ p["w0"].T + p["b0"])
    x = jnn.elu(x @ p["w2"].T + p["b2"])
    x = jnn.elu(x @ p["w4"].T + p["b4"])
    x = x @ p["w6"].T + p["b6"]
    return x


def norm_forward(state, x):
    std = jnp.sqrt(state["var"])
    return (x - state["mean"]) / (std + 1e-2)


def _welford_update(state, x):
    mean = state["mean"]
    var = state["var"]
    count = state["count"]

    count_x = x.shape[0]
    count2 = count + count_x
    rate = count_x / count2

    var_x = jnp.var(x, axis=0, keepdims=True)
    mean_x = jnp.mean(x, axis=0, keepdims=True)

    delta = mean_x - mean
    mean2 = mean + rate * delta
    var2 = var + rate * (var_x - var + delta * (mean_x - mean2))
    return {"mean": mean2, "var": var2, "std": jnp.sqrt(var2), "count": count2}


def norm_update(state, x, until=1.0e8):
    # until=None 表示始终更新 (与 torch EmpiricalNormalization 默认行为一致);
    # 否则到达 count>=until 后冻结. until 为静态标量, 用 Python if 分流,
    # 避免 state["count"](traced) 与 None 比较导致 jax 报错.
    if until is None:
        return _welford_update(state, x)
    return jax.lax.cond(state["count"] >= until, lambda: state, lambda: _welford_update(state, x))


def _auto_reset(state, reset_state, done):
    done_mask = done.astype(jnp.bool_)
    return tree_util.tree_map(
        lambda cur, new: jnp.where(
            done_mask.reshape((-1,) + (1,) * (cur.ndim - 1)), new, cur),
        state, reset_state)


def make_body(step_fn, reset_fn, until, n_envs):
    """body 把权重 / std / d_max 放在 carry 里 (常量但每轮可变), 这样 jit 编译一次后
    跨迭代复用, 不会因权重视图变化或课程 d_max 调整而反复重编译。"""
    def body(carry, noise_t):
        S, rng, on, pn, t, actor_params, critic_params, std, d_max = carry
        raw_state = S.obs["state"]
        raw_priv = S.obs["privileged_state"]
        critic_in = jnp.concatenate([raw_state, raw_priv], axis=-1)
        # 更新顺序对齐 torch EmpiricalNormalization: 先 update(obs_t) 再用含 obs_t 的新
        # stats 做 normalize (torch forward 先 update 后返回归一结果). 否则 jax 比 torch
        # 滞后一步, 产生可累积的 1-step 残差. 首步 (t==0) 同样归一, 不能跳过, 否则每轮
        # rollout 第 0 步动作全错, 导致 episode 在 rollout 边界崩溃.
        on2 = norm_update(on, raw_state, until)
        pn2 = norm_update(pn, critic_in, until)
        obs_norm = norm_forward(on2, raw_state)
        priv_norm = norm_forward(pn2, critic_in)

        mean = mlp_forward(actor_params, obs_norm)
        action = mean + std * noise_t

        S_pre = step_fn(S, action)
        reward = S_pre.reward
        done = S_pre.done
        fallen = S_pre.metrics["fallen"]
        step_count = S_pre.info["env_state"].step_count.astype(jnp.float32)
        difficulty = S_pre.info["env_state"].difficulty
        orientation = S_pre.metrics["orientation"]
        lin_vel_tracking = S_pre.metrics["lin_vel_tracking"]

        rng, rrng = jax.random.split(rng)
        diff_vec = jax.random.uniform(rrng, (n_envs,), minval=0.0, maxval=d_max)
        S_reset = reset_fn(jax.random.split(rrng, n_envs), diff_vec)
        S_next = _auto_reset(S_pre, S_reset, done)

        value = mlp_forward(critic_params, priv_norm).reshape(-1)
        log_std = jnp.log(std)
        log_prob = (-0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
        time_out = done & (fallen < 0.5)

        out = {
            "obs_norm": obs_norm,
            "priv_norm": priv_norm,
            "actions": action,
            "rewards": reward,
            "dones": done.astype(jnp.float32),
            "time_outs": time_out.astype(jnp.float32),
            "values": value,
            "log_prob": log_prob,
            "mean": mean,
            "sigma": std * jnp.ones_like(mean),
            "fallen": fallen,
            "step_count": step_count,
            "difficulty": difficulty,
            "orientation": orientation,
            "lin_vel_tracking": lin_vel_tracking,
        }
        return (S_next, rng, on2, pn2, t + 1, actor_params, critic_params, std, d_max), out

    return body


# 跨迭代复用同一份 jit 编译产物: 键为 (env id, n_envs, until, num_steps)
_SCAN_FN_CACHE = {}


def _get_scan_fn(env, n_envs, until, num_steps):
    key = (id(env), n_envs, until, num_steps)
    fn = _SCAN_FN_CACHE.get(key)
    if fn is None:
        step_fn = jax.vmap(env.step)
        reset_fn = jax.vmap(env.reset, in_axes=(0, 0))
        body = make_body(step_fn, reset_fn, until, n_envs)

        def scan_fn(carry, xs):
            return jax.lax.scan(body, carry, xs, length=num_steps)

        fn = jax.jit(scan_fn)
        _SCAN_FN_CACHE[key] = fn
    return fn


def collect_rollout(env, num_steps, init_state, init_obs_norm, init_priv_norm,
                    d_max, rng, weights, until=1.0e8, noise=None):
    n_envs = int(init_state.obs["state"].shape[0])

    if noise is None:
        noise = jax.random.normal(rng, (num_steps, n_envs, env.action_size))

    init_carry = (init_state, rng, init_obs_norm, init_priv_norm, jnp.int32(0),
                  weights["actor"], weights["critic"], weights["std"], d_max)

    scan_fn = _get_scan_fn(env, n_envs, until, num_steps)
    (final_carry, out) = scan_fn(init_carry, noise)

    final_obs_norm = final_carry[2]
    final_priv_norm = final_carry[3]
    final_state = final_carry[0]
    final_rng = final_carry[1]
    last_critic_in = jnp.concatenate(
        [final_state.obs["state"], final_state.obs["privileged_state"]], axis=-1
    )
    last_critic_obs = norm_forward(final_priv_norm, last_critic_in)

    return out, final_obs_norm, final_priv_norm, last_critic_obs, final_state, final_rng


def weights_from_torch_policy(policy):
    """从在线 torch ActorCritic 提取权重, 映射到 jax 的 {w0,b0,...} 约定.

    与 load_ckpt_weights 完全相同的键映射 (actor/critic 的 Linear 在 nn.Sequential
    中位于索引 0,2,4,6)。每轮迭代重新提取, 保证 jax actor 与刚更新过的 torch 策略一致。
    """
    sd = policy.state_dict()
    out = {"actor": {}, "critic": {}}
    # 键名沿用 mlp_forward 约定: w0/w2/w4/w6 与 b0/b2/b4/b6 对应 nn.Sequential 的
    # Linear 索引 0/2/4/6
    idx_map = [(0, "w0", "b0"), (2, "w2", "b2"), (4, "w4", "b4"), (6, "w6", "b6")]
    for branch in ("actor", "critic"):
        for torch_i, wkey, bkey in idx_map:
            w = sd[f"{branch}.{torch_i}.weight"].detach().cpu().numpy()
            b = sd[f"{branch}.{torch_i}.bias"].detach().cpu().numpy()
            out[branch][wkey] = jnp.asarray(np.ascontiguousarray(w))
            out[branch][bkey] = jnp.asarray(np.ascontiguousarray(b))
    out["std"] = jnp.asarray(np.ascontiguousarray(sd["std"].detach().cpu().numpy()))
    return out


def normalizer_state_from_torch(norm):
    """torch EmpiricalNormalization -> jax 归一化状态 dict."""
    return {
        "mean": jnp.asarray(np.ascontiguousarray(norm._mean.detach().cpu().numpy())),
        "var": jnp.asarray(np.ascontiguousarray(norm._var.detach().cpu().numpy())),
        "std": jnp.asarray(np.ascontiguousarray(norm._std.detach().cpu().numpy())),
        "count": jnp.int64(int(norm.count.item())),
    }


def normalizer_state_to_torch(state, norm):
    """把 jax 归一化状态写回 torch EmpiricalNormalization (逐位同步)."""
    dev = norm._mean.device
    norm._mean.copy_(torch.as_tensor(np.asarray(state["mean"]), device=dev))
    norm._var.copy_(torch.as_tensor(np.asarray(state["var"]), device=dev))
    norm._std.copy_(torch.as_tensor(np.asarray(state["std"]), device=dev))
    norm.count.copy_(torch.tensor(int(state["count"]), dtype=torch.long, device=dev))


def collect_rollout_from_policy(env, num_steps, init_state, obs_norm_state,
                                priv_norm_state, d_max, rng, policy, until=1.0e8):
    """从在线 torch 策略提取权重并跑一次性 scan 采集, 返回与 collect_rollout 相同元组。"""
    W = weights_from_torch_policy(policy)
    return collect_rollout(env, num_steps, init_state, obs_norm_state,
                           priv_norm_state, d_max, rng, W, until=until)


def to_torch_trajectory(out, device="cuda:0"):
    traj = {}
    for k, v in out.items():
        traj[k] = dlu.to_torch(v, device=device)
    return traj
