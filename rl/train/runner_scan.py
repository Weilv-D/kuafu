# -*- coding: utf-8 -*-
"""KuafuOnPolicyRunner: 用一次性 jax lax.scan 替换 RSL-RL 逐步采集, 复用 PPO 的
storage / compute_returns / update 不变, 保证训练循环语义完全不变。

设计 (与 design.md / 用户方案 A 一致):
  - 单 rollout 内固定 d_max (避免 scan 内动态变长), rollout 结束后用整段轨迹一次性
    更新 Curriculum (Option (a): scan 只吐静态形状全量张量 -> Python 端 mask+gather+reduce
    -> 原样调 Curriculum.update, 更新规则逐位相同)。
  - jax actor + jax 归一器在 scan 内完成; 整段轨迹一次 DLPack 回 torch, 逐条喂入
    RolloutStorage (复用 process_env_step), 再把 jax 终态归一化统计回写 torch 归一器。
  - 任何依赖运行数据长度的筛选都在 Python 端, jit 内绝不 arr[mask] (动态-Shape 禁令)。
"""
import time
import os
from collections import deque

import numpy as np
import jax
import torch

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from rl.train import jax_rollout as jr


class KuafuOnPolicyRunner(OnPolicyRunner):
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # ---- writer 初始化 (与父类一致, 仅 tensorboard / 无 log) ----
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                self.writer = None

        # init_at_random_ep_len 在 scan 模式下无法零成本实现 (reset 已在 scan 内完成),
        # 此处忽略; 对训练质量无本质影响, 仅少一点初始探索随机性。
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.train_mode()

        # ---- book keeping ----
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # scan 状态: 从 DirectVecEnv 的当前 jax 状态接力; rng 独立推进
        carry_state = self.env._state
        rng = jax.random.PRNGKey(int(torch.randint(0, 2**31 - 1, ()).item()))

        for it in range(start_iter, tot_iter):
            start = time.time()

            with torch.inference_mode():
                # ===== 一次性 scan 采集 (env + jax actor + jax 归一器) =====
                W = jr.weights_from_torch_policy(self.alg.policy)
                on = jr.normalizer_state_from_torch(self.obs_normalizer)
                pn = jr.normalizer_state_from_torch(self.privileged_obs_normalizer)
                d_max = float(self.env._difficulty)
                (out, on_final, pn_final, last_critic_obs, carry_state, rng) = (
                    jr.collect_rollout_from_policy(
                        self.env.env, self.num_steps_per_env, carry_state, on, pn,
                        d_max, rng, self.alg.policy, until=None))
                # 保持 DirectVecEnv 的 jax 状态与 scan 终态一致 (环境若被外部访问如 eval)
                self.env._state = carry_state

                traj = jr.to_torch_trajectory(out, device=self.device)
                last_critic_obs_t = torch.as_tensor(np.asarray(last_critic_obs), device=self.device)

                # ===== 喂入 RolloutStorage (复用 process_env_step) =====
                self._fill_storage(traj)

                # ===== 课程: Option (a) 整段轨迹一次性更新 (更新规则不变) =====
                self._scan_curriculum(traj)

                # ===== 回写 torch 归一器终态 (逐位同步) =====
                jr.normalizer_state_to_torch(on_final, self.obs_normalizer)
                jr.normalizer_state_to_torch(pn_final, self.privileged_obs_normalizer)

                # ===== 价值 bootstrap =====
                if self.training_type == "rl":
                    self.alg.compute_returns(last_critic_obs_t)

            stop = time.time()
            collection_time = stop - start
            start = stop

            # ===== PPO 更新 (完全复用) =====
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # ===== 重建日志所需的 episode 统计 =====
            ep_infos = self._build_ep_infos(traj)

            if self.log_dir is not None and not self.disable_logs and self.writer is not None:
                # 重建 rewbuffer / lenbuffer 供 self.log 使用
                self._accumulate_buffers(traj, cur_reward_sum, cur_episode_length,
                                         rewbuffer, lenbuffer)
                self.log(locals())

            # 保存
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        # 训练结束保存最终模型
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ------------------------------------------------------------------ #
    def _fill_storage(self, traj):
        T = traj["obs_norm"].shape[0]
        for t in range(T):
            tr = self.alg.transition
            tr.observations = traj["obs_norm"][t]
            tr.privileged_observations = traj["priv_norm"][t]
            tr.actions = traj["actions"][t]
            tr.values = traj["values"][t].view(-1, 1)
            tr.actions_log_prob = traj["log_prob"][t].view(-1, 1)
            tr.action_mean = traj["mean"][t]
            tr.action_sigma = traj["sigma"][t]
            rewards = traj["rewards"][t]
            dones = traj["dones"][t]
            infos = {"time_outs": traj["time_outs"][t], "observations": {}}
            self.alg.process_env_step(rewards, dones, infos)

    def _scan_curriculum(self, traj):
        dones = np.asarray(traj["dones"].detach().cpu().numpy())
        difficulty = np.asarray(traj["difficulty"].detach().cpu().numpy())
        fallen = np.asarray(traj["fallen"].detach().cpu().numpy())
        step_count = np.asarray(traj["step_count"].detach().cpu().numpy())
        d_max = float(self.env._difficulty)
        high = difficulty > (d_max * 0.7)
        relevant = (dones > 0.5) & high
        if relevant.any():
            # 步优先 (t-major) 顺序, 与逐步 Curriculum.update 的追加顺序一致
            surv = step_count[relevant]
            fell = fallen[relevant]
            self.env._curriculum.update(surv, fell)
        self.env._difficulty = jax.numpy.float32(self.env._curriculum.d)

    def _build_ep_infos(self, traj):
        dones = np.asarray(traj["dones"].detach().cpu().numpy()) > 0.5
        if not dones.any():
            return []
        orientation = np.asarray(traj["orientation"].detach().cpu().numpy())[dones].mean()
        lin_vel = np.asarray(traj["lin_vel_tracking"].detach().cpu().numpy())[dones].mean()
        step_count = np.asarray(traj["step_count"].detach().cpu().numpy())[dones].mean()
        fallen = np.asarray(traj["fallen"].detach().cpu().numpy())[dones].mean()
        c = self.env._curriculum
        log = {
            "episode_length": float(step_count),
            "orientation": float(orientation),
            "lin_vel_tracking": float(lin_vel),
            "fallen_rate": float(fallen),
            "curriculum_avg_survival": (float(c._last_avg_survival)
                                        if c._last_avg_survival == c._last_avg_survival else float("nan")),
            "curriculum_fall_rate": (float(c._last_fall_rate)
                                     if c._last_fall_rate == c._last_fall_rate else float("nan")),
            "difficulty": float(self.env._difficulty),
            "difficulty_mean": float(np.asarray(traj["difficulty"].detach().cpu().numpy()).mean()),
        }
        return [log]

    def _accumulate_buffers(self, traj, cur_reward_sum, cur_episode_length,
                            rewbuffer, lenbuffer):
        T = traj["rewards"].shape[0]
        for t in range(T):
            cur_reward_sum += traj["rewards"][t]
            cur_episode_length += 1
            dones_t = traj["dones"][t] > 0.5
            new_ids = dones_t.nonzero(as_tuple=False)
            if new_ids.numel() > 0:
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0
