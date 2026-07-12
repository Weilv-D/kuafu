# KUAFU 强化学习模块

面向 MuJoCo MJX 的残差 RL 训练管线。技术路线见《[软件架构与 RL 技术路线设计](../docs/plans/2026-07-08-软件架构与RL技术路线-design.md)》。物理参数真源在项目顶层 `kuafu_physics.py`。

## 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| GPU 环境（CUDA 13 + RTX 4070 + JAX 0.10） | ✅ | torch 2.12+cu130, DLPack 零拷贝 (copy=False 守卫 + 启动校验) |
| 仿真模型 `kuafu.xml` | ✅ | 闭链残差 0mm, 轮挂 Q 点, armature=0; 轮改 capsule 兼容 MJX 地形碰撞; 4 舵机独立 |
| 物理验证（阶段 0） | ✅ | **13/13**, LQR pitch 0.1s 恢复 5° + yaw 条件阻尼 + roll PD 调平 0.1s 恢复 3° |
| MJX 环境 `kuafu_mjx_env.py` | ✅ | 三轴基层(pitch LQR+yaw阻尼+roll PD) + RL残差; 接触obs; 地形(斜坡+台阶); 延迟DR; 烟测通过 |
| Teacher PPO 训练 `train.py` | ✅ | RSL-RL 2.x + DLPack; actor 157(proprio148+z9), critic 160; 双向滑动窗口课程(d_max 随高难度平均存活≥800且摔倒率≤0.3升、≤600或≥0.5降)+per-env Uniform(0,d_max)采样; reset/step 闭包 donate_argnums 回收 state/rng 缓冲降显存峰值; 烟测通过 |
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
│   └── terrain.py              课程地形系统（_apply_terrain 按 difficulty 生成斜坡+台阶）
│
├── train/
│   ├── train.py               Teacher PPO 训练入口（RSL-RL 2.x + DLPack）
│   ├── jax_rollout.py         jax actor/critic MLP 前向与权重映射（S0 对齐护栏用）
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

# 4b. S0 回归护栏: jax actor/critic 前向 与 torch ActorCritic 对齐
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
阶段1  Teacher PPO ────────── ✅ 代码就绪, 烟测通过, 待正式训练
  ↓                              收敛: 恢复时间 < LQR×0.85
阶段2  Student 蒸馏 ──────── ✅ 代码就绪 (规范 DAgger: 回放缓冲 + DataLoader + 梯度裁剪 + TensorBoard), 待 teacher checkpoint
  ↓                              student ≥ teacher×0.9
阶段3  原生 MuJoCo 对拍 ──── ✅ playback.py 就绪
  ↓
阶段4  ONNX 导出 ─────────── ✅ export_policy.py 就绪 (含 normalizer)
  ↓
阶段5  实机部署 ──────────── ⏳ 待硬件就绪
```

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
