# CHANGELOG

## [Unreleased] RL 训练管线

### 训练

- Teacher PPO（RSL-RL 2.x + DLPack 零拷贝）逐步采集；actor 以本体感受 148(含接触标志) + 静态 latent z 9 = 157 维为条件，critic 额外吃瞬态推力 3 维共 160 维。
- 双向滑动窗口课程学习：`d_max` 随高难度环境平均存活步数与摔倒率升降（升 ≥800 且 ≤0.3，降 ≤600 或 ≥0.5），per-env 采样 `Uniform(0, d_max)`；`d_max` 与窗口统计量经 `curriculum_{it}.pt` 跨 resume 持久化。
- 默认 72 步/rollout；`--num_envs` 在 RTX 4070 8GB 下单卡可跑至 4096（峰值显存约 4.6GB）。

### 护栏与导出

- S0 对齐护栏（`rl/verify/s0_parity.py`）：jax actor/critic 前向与 torch `ActorCritic` 数值对齐（f64 结构性 <1e-5、f32 <2e-3、std 逐位相等），覆盖权重映射与 `mlp_forward` 的改动。
- ONNX 导出（`rl/export/export_policy.py`）：teacher（含 normalizer）/ student 两种模式。

### 显存与编译优化

- `reset`/`step` 闭包对 `state`/`rng`/`difficulty` 启用 `donate_argnums` 原地回收 JAX 缓冲；不捐经 DLPack 借入的 `action` 张量（避免 torch 缓冲被 XLA 误回收）。
- 关闭 XLA 预分配（`XLA_PYTHON_CLIENT_PREALLOCATE=false`）+ 编译缓存（`JAX_COMPILATION_CACHE_DIR`）跨 run 复用 XLA 计算图。
