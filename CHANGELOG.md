# CHANGELOG

## [Unreleased] JaxRollout：一次性 scan 采集

采集管线性能优化，不改变训练语义。

### 新增

- `rl/train/jax_rollout.py`：把整段 PPO rollout 塞进单个 `jax.lax.scan`。
  - jax actor/critic `mlp_forward` 复刻 RSL-RL `[512,512,512] elu`；`EmpiricalNormalization` 用 Welford 递推在 scan 内增量更新。
  - 静态形状全量张量输出 `[T, N, ...]`（obs_norm / priv_norm / actions / rewards / dones / time_outs / values / log_prob / mean / sigma / fallen / step_count / difficulty / orientation / lin_vel_tracking）。
  - 归一顺序（t=0 原始、t>0 归一）与基线 `obs_normalizer(obs)` 调用时序逐位一致；探索噪声、auto-reset、timeout bootstrap 一致。
- `rl/train/runner_scan.py`：`KuafuOnPolicyRunner`，复用 PPO 的 `storage` / `compute_returns` / `update`。
  - 单 DLPack 零拷贝回 torch 后逐条喂入 `PPO.process_env_step`（含 timeout 的 value bootstrap）；jax 终态归一化统计回写 torch 归一器。
  - 课程采用 Option (a)：整段轨迹一次性 mask+gather+reduce 后调 `Curriculum.update`（窗口 200，Welford 顺序无关）。
- `rl/verify/s0_parity.py`：S0 回归护栏（jax actor/critic 前向 vs torch `ActorCritic`，f64 结构性 <1e-5、f32 <2e-3、std 逐位相等）。
- `rl/train/train.py`：`--jax_scan_rollout` 开关（切换 `KuafuOnPolicyRunner`），checkpoint 格式 / 权重 / 归一器 / 课程与逐步采集完全兼容。

### 验证

- 物理验证 `verify_model.py`：13/13。
- S0 权重对齐：`s0_parity.py` PASS（f64 结构性 max diff 1.78e-15）。
- A/B 对拍（`--resume model_600`，step 0）：scan 与逐步采集 returns / episode 长度逐位一致；采集 ~1.95× 提速（RTX 4070，3072 envs × 72 步）。

### 已知近似（对训练无实质影响）

课程调整在 scan 模式每迭代至多一次（基线逐步追加、每步至多一次）；窗口统计量因 Welford 顺序无关而完全相同，`d_max` 单迭代一次调整足够。
