# AGENTS.md — KUAFU 强化学习开发速查

面向 Coding Agent 的 RL 模块操作约定。物理与代码修改的单一真相源见 `kuafu_physics.py`；
完整设计与决策见 `docs/KUAFU.md` 与 `docs/plans/`。

## 环境

- venv: `rl/.venv`（Python 3.12，JAX 0.10 cuda13 + MuJoCo MJX + RSL-RL 2.x + torch 2.12 cu130）
- 安装: `rl/.venv/bin/pip install -r rl/requirements.txt`
- 运行任何 rl 脚本前激活 venv: `rl/.venv/bin/python <script>`

## 物理常量真源

`kuafu_physics.py`（顶层）是机构/物理参数单一真相源，被 `rl/` 仿真、训练、`export` 与部署共用。
**改物理参数只改此处**，不要散落到 `.xml` 或 env 里。

## 训练

```bash
# Teacher PPO（逐步采集，默认）
rl/.venv/bin/python rl/train/train.py --run_name garlic --num_envs 3072
# 一次性 scan 采集 (JaxRollout, 采集 ~1.95× 提速, 语义逐位一致)
rl/.venv/bin/python rl/train/train.py --run_name garlic --num_envs 3072 --jax_scan_rollout
# 续训 (两种采集模式 checkpoint 格式与权重/归一器/课程完全兼容)
rl/.venv/bin/python rl/train/train.py --run_name garlic --num_envs 3072 \
  --jax_scan_rollout --resume rl/checkpoints/garlic/teacher/model_3999.pt
# 默认 72 步/rollout; --num_steps_per_env 可调; 显存受限降 --num_envs (1024≈2.0GB, 3072≈4.3GB)
```

- 维度约定: actor 157 = proprio 148 + z 9; critic 160 = actor 157 + 瞬态推力 3。
- 课程: 双向滑动窗口调 `d_max`（初始 0.1，上限 1.0），per-env 采样 `Uniform(0, d_max)`；
  `Curriculum` 在 `train.py`，`DirectVecEnv._curriculum` 持有。

## 验证（改代码前后必跑）

```bash
# 1. 物理验证 (CPU, 13/13) — 任何物理/机构改动
rl/.venv/bin/python rl/verify/verify_model.py
# 2. S0 权重对齐护栏 (jax actor/critic vs torch ActorCritic) — 权重映射/mlp_forward 改动
rl/.venv/bin/python rl/verify/s0_parity.py
# 3. 策略回放 (原生 MuJoCo CPU, 可视化行为) — 训练产出行为合理性
rl/.venv/bin/python rl/verify/playback.py --ckpt rl/checkpoints/garlic/teacher/model_3999.pt
```

## 导出与部署

```bash
rl/.venv/bin/python rl/export/export_policy.py --ckpt <model>.pt --mode teacher   # 含 normalizer
rl/.venv/bin/python rl/export/export_policy.py --ckpt <student>.pt --mode student
```

## JaxRollout 采集（jax_rollout.py + runner_scan.py）

- 单 rollout 用 `jax.lax.scan` 整段塞 GPU；jax actor + jax `EmpiricalNormalization` 在 scan 内完成。
- scan 输出静态形状 `[T, N, ...]`；单 DLPack 回 torch → `PPO.process_env_step` → `compute_returns` → `update`。
- 归一顺序 / auto-reset / timeout bootstrap / 探索噪声 与逐步采集逐位一致（见 `rl/README.md` JaxRollout 节）。
- 改 `mlp_forward`、`weights_from_torch_policy`、`norm_update` 后必须过 `s0_parity.py`。
