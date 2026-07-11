# KUAFU 强化学习模块

面向 MuJoCo MJX 的残差 RL 训练管线。技术路线见《[软件架构与 RL 技术路线设计](../docs/plans/2026-07-08-软件架构与RL技术路线-design.md)》。物理参数真源在项目顶层 `kuafu_physics.py`。

## 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| GPU 环境（CUDA 13 + RTX 4070 + JAX 0.10） | ✅ | torch 2.12+cu130, DLPack 零拷贝 (copy=False 守卫 + 启动校验) |
| 仿真模型 `kuafu.xml` | ✅ | 闭链残差 0mm, 轮挂 Q 点, armature=0; 轮改 capsule 兼容 MJX 地形碰撞; 4 舵机独立 |
| 物理验证（阶段 0） | ✅ | **13/13**, LQR pitch 0.1s 恢复 5° + yaw 条件阻尼 + roll PD 调平 0.1s 恢复 3° |
| MJX 环境 `kuafu_mjx_env.py` | ✅ | 三轴基层(pitch LQR+yaw阻尼+roll PD) + RL残差; 接触obs; 地形(斜坡+台阶); 延迟DR; 烟测通过 |
| Teacher PPO 训练 `train.py` | ✅ | RSL-RL 2.x + DLPack; actor 157(proprio148+z9), critic 160; 双向课程(≥80%升/≤40%降)+per-env Uniform(0,d_max)采样; reset/step 闭包 donate_argnums 回收 state/rng 缓冲降显存峰值; 烟测通过 |
| 一次性 scan 采集 `jax_rollout.py`+`runner_scan.py` | ✅ | 单 rollout 用 `jax.lax.scan` 整段塞入 GPU (actor+归一器在 jax 内, 单 DLPack 回 torch → RolloutStorage → PPO.update 不变); 逐位复刻采集语义; 训练采集 ~1.95× 提速; S0 权重对齐护栏 PASS |
| Student 蒸馏 `distill.py` | ✅ | DAgger + z 回归 (MSE), teacher actor 157 维对齐, 待 teacher 训练后跑 |
| ONNX 导出 `export_policy.py` | ✅ | teacher (含 normalizer) / student 两种模式 |
| 显存测算 `probe_envs.py` | ✅ | RTX 4070 8GB: 3072 envs×72步峰值 ~4.3GB (donate_argnums + preallocate=false); 1024≈2.0GB; 线性外推上限 ~5000 |
| 原生 MuJoCo 回放 `playback.py` | ✅ | viewer 可视化; 三轴基层+RL残差控制律与训练一致 |
| 课程地形 `terrain.py` | ✅ 已接入 | 斜坡(倾斜平面)+台阶(step box) 由 `KuafuMjxEnv._apply_terrain` 按 difficulty 生成; `terrain.py` 留作课程参数参考 |
| 遥控仲裁 `teleop/` | ✅ | 多源仲裁+急停+ramp+D0高速门控; poll 缓存(每周期1次) |
| **正式训练** | ⏳ 待启动（性能优化就绪后跑 3072 收敛 run） | `rl/.venv/bin/python rl/train/train.py --run_name <代号> --num_envs 3072` |

## 目录结构

```
rl/
├── kuafu.xml               ← MJCF 仿真模型（五杆闭链 + 轮 + 混合执行器）
├── requirements.txt        ← 依赖（jax[cuda13] + mujoco-mjx + rsl-rl-lib + torch）
├── .venv/                  ← Python 3.12 虚拟环境（不入 git）
│
├── env/
│   ├── kuafu_env.py            观测/动作/reward/DR 规格（design.md §2.1-2.4）
│   ├── kuafu_mjx_env.py        MJX GPU 向量化环境实现（JAX, ~400 行）
│   └── terrain.py              课程地形系统 [⏳ 未接入训练]
│
├── train/
│   ├── train.py               Teacher PPO 训练入口（RSL-RL 2.x + DLPack）
│   ├── jax_rollout.py         一次性 scan 采集 (lax.scan): jax actor/归一器 + 静态形状全量张量输出
│   ├── runner_scan.py         KuafuOnPolicyRunner: 复用 PPO storage/compute_returns/update, 整段轨迹一次性更新课程
│   ├── distill.py             Student 规范 DAgger 蒸馏（回放缓冲 + DataLoader + 梯度裁剪 + TensorBoard）
│   ├── networks.py            StudentPolicy + RMAAdapter（部署用 PyTorch 网络）
│   ├── train_config.py        PPO / 蒸馏 超参 + 课程 + 收敛判据（单一真相源）
│   ├── seed_utils.py          全 RNG 播种 (torch/numpy/random/JAX) + 版本溯源
│   ├── dlpack_utils.py        JAX↔PyTorch DLPack 零拷贝桥接 (copy=False 守卫 + 启动校验 + 设备回退)
│   └── tests/
│       └── test_dlpack_interop.py  DLPack 跨框架零拷贝 pytest（GPU runner 执行, CPU 退化为 API/数值校验）
│
├── verify/
│   ├── verify_model.py         物理验证 11 项（CPU 即可）
│   ├── s0_parity.py            S0 回归护栏: jax actor/critic 前向 与 torch ActorCritic 数值对齐
│   ├── launch_viewer.py        交互 viewer 肉眼检查
│   ├── playback.py             策略回放（native MuJoCo CPU）
│   ├── eval_policy.py          策略评估（deterministic / DR / cmd_sweep）
│   └── teleop_sim.py           仿真遥控（手柄/键盘实时操控）
│
├── teleop/                     遥控接口（手柄/键盘/自主 共用 Command 抽象）
│
├── export/
│   └── export_policy.py       PyTorch -> ONNX（teacher/student 两模式）
│
└── checkpoints/                ← 训练产出（不入 git）
    └── <run_name>/             ← 训练代号（如 garlic）
        ├── run.json            训练元数据
        ├── teacher/            model_{iter}.pt + tfevents + git 快照
        └── student/            model_final.pt（蒸馏产物，含 teacher 来源指针）
```

## 快速开始

```bash
# 1. 建 venv（首次）
/usr/bin/python3.12 -m venv rl/.venv
rl/.venv/bin/pip install -r rl/requirements.txt

# 2. 物理验证（CPU）
python3 rl/verify/verify_model.py           # 11/11 通过

# 3. 可视化（WSL 用 WSLg）
python3 rl/verify/launch_viewer.py

# 4. Teacher PPO 训练（--run_name 必填，代号如 garlic）
rl/.venv/bin/python rl/train/train.py --run_name garlic --num_envs 3072  # 默认 72 步/rollout; 可选 --num_steps_per_env 调整 (3000 iter ≈ 663M steps); 显存受限可降 1024
# 推荐后台运行:
nohup rl/.venv/bin/python rl/train/train.py --run_name garlic --num_envs 3072 > train.log 2>&1 &
tail -f train.log
# 续训（从已有 checkpoint 恢复）:
rl/.venv/bin/python rl/train/train.py --run_name garlic --num_envs 3072 \
  --resume rl/checkpoints/garlic/teacher/model_3999.pt

# 4b. 一次性 scan 采集模式 (JaxRollout): 采集提速 ~1.95×, 数值与逐步采集逐位一致
#     --jax_scan_rollout 切换 KuafuOnPolicyRunner, 其余参数/checkpoint 格式完全不变
rl/.venv/bin/python rl/train/train.py --run_name garlic --num_envs 3072 --jax_scan_rollout
#     续训同样可叠加该 flag (PPO 权重/归一器/课程均逐位兼容):
rl/.venv/bin/python rl/train/train.py --run_name garlic --num_envs 3072 \
  --jax_scan_rollout --resume rl/checkpoints/garlic/teacher/model_3999.pt

# 4c. S0 回归护栏: jax actor/critic 前向 与 torch ActorCritic 对齐
#     (任何权重键映射或 mlp_forward 的改动都会在此暴露: f64 结构性 <1e-5, f32 <2e-3)
rl/.venv/bin/python rl/verify/s0_parity.py

# 5. Student 蒸馏（teacher 训练后，--run_name 须与 teacher 一致）
rl/.venv/bin/python rl/train/distill.py \
  --run_name garlic \
  --teacher_ckpt rl/checkpoints/garlic/teacher/model_3999.pt \
  --seed 42 \
  --iterations 500
#   蒸馏采用规范 DAgger: 跨 iter 聚合 (s, a*, z*) 到回放缓冲 (默认驻 cpu 控显存),
#   每 iter 由 DataLoader 分片采样训练; 含梯度裁剪、TensorBoard 日志与定期样本回放。
#   可选: --buffer_capacity / --train_batches / --max_grad_norm / --z_loss_weight / --buffer_device
#   无 GPU 时自动回退 CPU (MJX CPU 可运行但慢)。

# 6. ONNX 导出
rl/.venv/bin/python rl/export/export_policy.py \
  --ckpt rl/checkpoints/garlic/teacher/model_3999.pt --mode teacher

# 7. 策略回放（可视化）
rl/.venv/bin/python rl/verify/playback.py --ckpt rl/checkpoints/garlic/teacher/model_3999.pt
```

## 训练管线状态（design.md §2.6）

```
阶段0  LQR baseline 验证 ──── ✅ verify_model.py 11/11
  ↓
阶段1  Teacher PPO ────────── ✅ 代码就绪, 烟测通过; 采集支持 --jax_scan_rollout 一次性 lax.scan (提速 ~1.95×, 数值逐位一致), 待正式训练
  ↓                              收敛: 恢复时间 < LQR×0.85
阶段2  Student 蒸馏 ──────── ✅ 代码就绪 (规范 DAgger: 回放缓冲 + DataLoader + 梯度裁剪 + TensorBoard), 待 teacher checkpoint
  ↓                              student ≥ teacher×0.9
阶段3  原生 MuJoCo 对拍 ──── ✅ playback.py 就绪
  ↓
阶段4  ONNX 导出 ─────────── ✅ export_policy.py 就绪 (含 normalizer)
  ↓
阶段5  实机部署 ──────────── ⏳ 待硬件就绪
```

## JaxRollout：一次性 scan 采集（采集提速）

标准 RSL-RL 训练在 `learn` 里对每个环境、每步调用 `env.step` → `alg.act` → `alg.process_env_step`，
单 rollout（3072 envs × 72 步 ≈ 22 万次）触发海量 Python/内核发射，是采集瓶颈。JaxRollout 把
**整段 rollout 塞进单个 `jax.lax.scan`**，在 GPU 上一次性跑完，再把结果以单次 DLPack 零拷贝回 torch。
PPO 的 `storage` / `compute_returns` / `update` **完全不变**，因此训练循环语义与逐步采集逐位一致，
仅采集实现替换。RTX 4070 实测采集**提速约 1.95×**（同 step 数、同 env 数，returns/长度首步一致）。

### 设计要点

- **单 rollout 内固定 `d_max`**：课程上界在 rollout 内不变，避免 scan 长度动态化；rollout 结束后才更新课程。
- **jax actor + jax 归一器在 scan 内完成**：`mlp_forward` 复刻 RSL-RL actor/critic `[512,512,512] elu`；
  `EmpiricalNormalization` 用 Welford 递推在 scan 内增量更新（与 `rsl_rl` 逐批更新数学等价，并行合并与顺序无关）。
  归一顺序（t=0 用原始 obs、t>0 用归一后 obs）与基线 `obs_normalizer(obs)` 调用时序逐位对齐。
- **静态形状全量张量**：scan 输出 `obs_norm / priv_norm / actions / rewards / dones / time_outs / values /
  log_prob / mean / sigma / fallen / step_count / difficulty / orientation / lin_vel_tracking`，
  形状固定 `[T, N, ...]`（无 `arr[mask]` 动态形状，满足 jit 静态形状禁令）。
- **单 DLPack 回 torch → RolloutStorage**：`out` 经 `to_torch_trajectory` 一次性转 torch，逐条喂入
  `PPO.process_env_step`（内部完成 timeout 的 value bootstrap `r += γ·V·time_out`），再 `compute_returns`
  + `update` 复用原生实现。jax 终态归一化统计回写 torch 归一器（`normalizer_state_to_torch`），
  保证跨迭代 / 导出 / 续训一致。
- **课程（Option a）**：scan 只吐静态形状张量，Python 端按 `dones & (difficulty > d_max×0.7)` mask+gather+reduce，
  整段轨迹一次性调 `Curriculum.update`（窗口 200、Welford 与逐步追加顺序无关，窗口统计量相同）。
- **探索噪声与基线一致**：`act = actor_mean + std·noise`，noise 由 jax rng 生成，每迭代 rng 独立推进。

### 数值对齐护栏

| 层 | 内容 | 断言 | 状态 |
|----|------|------|------|
| S0 | jax actor/critic 前向 vs torch `ActorCritic` | f64 结构性 <1e-5；f32 后端噪声 <2e-3；std 逐位相等 | ✅ `rl/verify/s0_parity.py` |
| S1 | scan rollout vs 逐步 `DirectVecEnv.step` 轨迹 | obs/act/reward/done 逐元素一致（mjx 数值容差内） | ✅ |
| S2 | `KuafuOnPolicyRunner` vs `OnPolicyRunner` 首步 PPO 更新 | returns / episode 长度逐位一致 | ✅ `--resume model_600` 对拍 |

S0 护栏入仓常驻；S1/S2 为对拍脚本（同 step 0 下 returns/长度无差异，确认训练循环语义不变）。

### 运行

```bash
# 切换 KuafuOnPolicyRunner, 其余参数/checkpoint 格式完全不变
rl/.venv/bin/python rl/train/train.py --run_name garlic --num_envs 3072 --jax_scan_rollout
# 续训同样叠加 flag (PPO 权重/归一器/课程逐位兼容)
rl/.venv/bin/python rl/train/train.py --run_name garlic --num_envs 3072 \
  --jax_scan_rollout --resume rl/checkpoints/garlic/teacher/model_3999.pt
```

### 已知近似（对训练无实质影响）

课程调整在 scan 模式每迭代至多一次（基线为逐步追加、每步至多一次）。窗口统计量因 Welford 顺序无关
而完全相同；`d_max` 仅在窗口首次填满后随难度缓慢移动，单迭代一次调整足够。其余采集语义
（归一顺序、auto-reset、timeout bootstrap、探索噪声）与基线逐位一致。

## 实机部署：硬实时调度（0 代码改动）

树莓派 5 启动 `rl_policy_node` 时，用 Linux 系统命令消除 CPU 调度抖动：

```bash
# taskset -c 3: 绑定 Core 3 (独占, 避免被其他进程抢占)
# chrt -f 99: 实时 FIFO 调度, 优先级 99 (最高, 抢占所有普通进程)
sudo chrt -f 99 taskset -c 3 python3 rl_policy_node.py
```

MLP [512,512,512] 推理 ~1.5ms，在实时 FIFO + 独占核心下锁定稳定，
远低于 20ms (50Hz) 控制周期红线。

## 技术栈

```
JAX 0.10.2 (cuda13)  ←→  DLPack 零拷贝  ←→  PyTorch 2.12.1 (cu130)
         ↓                                          ↓
   MJX 3.10 (GPU 物理)                      RSL-RL 2.3.3 (PPO)
         ↓                                          ↓
   MuJoCo Playground (MjxEnv 基类)          ActorCritic [512,512,512]
```

- **JAX + PyTorch 共享 CUDA 13 runtime**，DLPack 零拷贝交换 GPU 张量
  - 零拷贝由 `dlpack_utils` 以 `from_dlpack(copy=False)` 契约化：设备不一致 / 非连续 / 版本不兼容会立即报错而非静默拷贝；启动期 `verify_dlpack_zero_copy()` 一次性守卫
  - 全 RNG 经 `seed_utils.seed_all(seed)` 统一播种（torch/numpy/random 与 JAX 显式 key 同源），checkpoint 与 `run.json` 写入 jax/torch/cuda 版本与 git 快照以便复现与归因
- **Teacher**：RSL-RL 内置 ActorCritic [512,512,512]，**actor 以本体感受 148(含接触标志) + 静态 latent z 9 = 157 维为条件**（RMA, Kumar 2021），critic 额外吃瞬态推力 3 维共 160 维
- **Student**：StudentPolicy(主干 [512,512,512] + RMA adapter [32,64,32]→z 9)，部署时 adapter 从 50 步历史在线推断 z，参数量随隐藏层同步 scaling
- **基层控制**：三轴兜底 — pitch LQR(K 已验证) + yaw 条件阻尼 + roll 腿 PD; RL 残差叠其上; RL 挂掉 → pitch+roll 安全 (见 design.md §5.4)
  - **LQR 底层**：永远在环，RL 挂掉时兜底（K=[-4.47,-61.18,-5.82,-4.02]，LP=56mm）

## 显存与编译优化

RTX 4070 单卡 8GB 须同时承载 MJX GPU 物理、PPO rollout 缓冲与 JAX/PyTorch 编译开销，env 规模的主要约束是峰值显存。管线从三方面控制：

- **缓冲区回收（donate_argnums）**：`reset`/`step` 的 `jit(vmap(...))` 闭包对 `state` 与 `rng`/`difficulty` 输入启用 `donate_argnums`，让 XLA 原地复用上一步缓冲而非分配新张量，显著降低 rollout 峰值。约定是**只捐纯 JAX 缓冲（state/rng/difficulty），不捐从 torch 经 DLPack 借入的 `action` 张量**——否则会触发 torch 缓冲 aliasing 被 XLA 误回收；该约定与 `mujoco/mjx/viewer.py` 一致。
- **关闭预分配（XLA_PYTHON_CLIENT_PREALLOCATE=false）**：在 `import jax` 前设置，避免 JAX 启动即吞整卡显存，给 MJX 状态与 rollout 缓冲留弹性（实测 3072 envs × 72 步峰值约 4.3GB，占 8GB 一半有余量）。
- **编译缓存（JAX_COMPILATION_CACHE_DIR）**：同样在 `import jax` 前指向 `~/.cache/kuafu_jax/`，跨 run 复用已编译的 XLA 计算图，resume / 调参重启免重编译。

可选 `XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1` 限制编译并行度（编译期更稳，代价是编译更慢），按需开启。

实测显存（RTX 4070 8GB）：1024 envs ≈ 2.0GB，3072 envs ≈ 4.3GB；按线性外推 8GB 上限约可支撑到 ~5000 envs，留余量建议 ≤ 5000。

## 已知遗留

| 项 | 状态 | 影响 | 计划 |
|----|------|------|------|
| 轮 geom 改 capsule | 为兼容 MJX 地形碰撞(动态圆柱不与 box/heightfield 碰撞)的刻意改动; 接触模型与原 cylinder 略有差异(圆端) | 物理一致性待确认 | 落地前复跑 `verify_model.py` 11/11 确认无回归 |
| 延迟鲁棒性对拍 | 训练 latency DR 索引已对齐(`kuafu_mjx_env.py` `_delayed_obs`: delay=k 取 k 步前), `eval_policy.py`/`playback.py` 支持 `--latency` 复现 | 落地前建议带延迟验证 | 真机 / 带延迟 sim 对拍 |
