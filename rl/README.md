# KUAFU 强化学习模块

面向 MuJoCo MJX 的残差 RL 训练管线。技术路线见《[软件架构与 RL 技术路线设计](../docs/plans/2026-07-08-软件架构与RL技术路线-design.md)》。物理参数真源在项目顶层 `kuafu_physics.py`。

## 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| GPU 环境（CUDA 13 + RTX 4070 + JAX 0.10） | ✅ | torch 2.12+cu130, DLPack 零拷贝 |
| 仿真模型 `kuafu.xml` | ✅ | 闭链残差 0mm, 轮挂 Q 点, armature=0, 碰撞清理 |
| 物理验证（阶段 0） | ✅ | 11/11, LQR 0.1s 恢复 5° |
| MJX 环境 `kuafu_mjx_env.py` | ✅ | JAX 向量化 reset/step/reward/obs/DR, 烟测通过 |
| Teacher PPO 训练 `train.py` | ✅ | RSL-RL 2.x + DLPack 桥接, 烟测 5 iters 通过 |
| Student 蒸馏 `distill.py` | ✅ | DAgger + teacher 推理, 待 teacher 训练后跑 |
| ONNX 导出 `export_policy.py` | ✅ | teacher (含 normalizer) / student 两种模式 |
| 显存测算 `probe_envs.py` | ✅ | RTX 4070: 2048 envs 可行, 推荐 1024 |
| 原生 MuJoCo 回放 `playback.py` | ✅ | viewer 可视化 |
| **课程地形 `terrain.py`** | ⏳ 未接入 | 平地 reward 收敛后在 train.py 外层接入 |
| **正式训练** | ⏳ 待启动 | `rl/.venv/bin/python rl/train/train.py --num_envs 1024` |

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
│   ├── distill.py             Student DAgger 蒸馏
│   ├── networks.py            StudentPolicy + RMAAdapter（部署用 PyTorch 网络）
│   └── train_config.py        PPO 超参 + 课程 + 收敛判据
│
├── verify/
│   ├── verify_model.py         物理验证 11 项（CPU 即可）
│   ├── launch_viewer.py        交互 viewer 肉眼检查
│   ├── probe_envs.py           显存测算（RTX 4070 最大 envs 搜索）
│   └── playback.py             策略回放（native MuJoCo CPU）
│
├── export/
│   └── export_policy.py       PyTorch → ONNX（teacher/student 两模式）
│
└── checkpoints/              ← 训练产出（不入 git）
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

# 4. 显存测算
rl/.venv/bin/python rl/verify/probe_envs.py  # 输出推荐 envs 数

# 5. Teacher PPO 训练
rl/.venv/bin/python rl/train/train.py --num_envs 1024 --iterations 1000

# 6. Student 蒸馏（teacher 训练后）
rl/.venv/bin/python rl/train/distill.py \
  --teacher_ckpt rl/checkpoints/teacher_*/model_1000.pt

# 7. ONNX 导出
rl/.venv/bin/python rl/export/export_policy.py \
  --ckpt rl/checkpoints/teacher_*/model_1000.pt --mode teacher

# 8. 策略回放（可视化）
rl/.venv/bin/python rl/verify/playback.py --ckpt policy.onnx
```

## 训练管线状态（design.md §2.6）

```
阶段0  LQR baseline 验证 ──── ✅ verify_model.py 11/11
  ↓
阶段1  Teacher PPO ────────── ✅ 代码就绪, 烟测通过, 待正式训练
  ↓                              收敛: 恢复时间 < LQR×0.85
阶段2  Student 蒸馏 ──────── ✅ 代码就绪 (DAgger), 待 teacher checkpoint
  ↓                              student ≥ teacher×0.9
阶段3  原生 MuJoCo 对拍 ──── ✅ playback.py 就绪
  ↓
阶段4  ONNX 导出 ─────────── ✅ export_policy.py 就绪 (含 normalizer)
  ↓
阶段5  实机部署 ──────────── ⏳ 待硬件就绪
```

## 技术栈

```
JAX 0.10.2 (cuda13)  ←→  DLPack 零拷贝  ←→  PyTorch 2.12.1 (cu130)
         ↓                                          ↓
   MJX 3.10 (GPU 物理)                      RSL-RL 2.3.3 (PPO)
         ↓                                          ↓
   MuJoCo Playground (MjxEnv 基类)          ActorCritic [256,256,256]
```

- **JAX + PyTorch 共享 CUDA 13 runtime**，DLPack 零拷贝交换 GPU 张量
- **Teacher**：RSL-RL 内置 ActorCritic，critic 含特权信息（friction/mass/COM/inertia），actor 仅本体感受 140 维
- **Student**：StudentPolicy(trunk + RMA adapter)，186k 参数 < 200k（Pi5 ONNX <1ms）
- **LQR 底层**：永远在环，RL 挂掉时兜底（K=[-4.47,-61.18,-5.82,-4.02]，LP=56mm）

## 已知遗留

| 项 | 状态 | 影响 | 计划 |
|----|------|------|------|
| `terrain.py` 未接入 `train.py` | 代码就绪但未引用 | 不影响平地训练 | 平地 reward 收敛后，在 train.py 外层循环接入 CurriculumController |
| history_len 4 vs 50 | 环境用 4 步堆叠(140 维), RMA 需 50 步 | 不影响训练（RSL-RL 吃 140 维），蒸馏时已处理 | distill.py 从 50 步序列取 base_obs(35) 喂给 adapter |
