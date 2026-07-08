# KUAFU 强化学习模块

面向 MuJoCo MJX 的残差 RL 训练管线。技术路线与决策见《[软件架构与 RL 技术路线设计](../docs/plans/2026-07-08-软件架构与RL技术路线-design.md)》（ADR-1 MuJoCo MJX、ADR-2 残差 RL→Teacher-Student+RMA、ADR-3 混合动作空间）。物理参数真源在项目顶层 `kuafu_physics.py`。

## 当前状态

| 组件 | 状态 |
|------|------|
| GPU 环境（CUDA 13 + RTX 4070 + JAX） | ✅ 就绪 |
| 仿真模型 `kuafu.xml` | ✅ 闭链残差 0 mm |
| 物理验证（阶段 0） | ✅ 11/11 通过，LQR 0.1s 恢复 5° |
| RL 环境（obs/act/reward/DR） | ✅ 规格就绪 |
| 训练配置（PPO + 课程） | ✅ 配置就绪 |
| ONNX 导出 | ✅ 骨架就绪（待 policy.pt） |
| **训练** | ⏳ 待启动（下一步） |

## 目录结构

```
rl/
├── kuafu.xml            ← MJCF 仿真模型（五杆闭链 + 轮 + 混合执行器）
├── verify/              ← 物理验证（design.md §2.6 阶段 0）
│   ├── verify_model.py      加载模型 → 11 项物理检查
│   └── launch_viewer.py     交互 viewer 肉眼检查姿态
├── env/                 ← RL 环境规格（design.md §2.1-2.4）
│   └── kuafu_env.py         观测 35×4 / 动作 6 / reward / 域随机化
├── train/               ← 训练配置（design.md §2.6）
│   └── train_config.py      PPO 超参 + 课程 + 收敛判据
├── export/              ← 部署导出（design.md §六）
│   └── export_policy.py     PyTorch → ONNX + 维度/NaN 校验
└── README.md
```

## 快速验证

```bash
# 激活 venv（首次: python3.12 -m venv .venv && .venv/bin/pip install -r rl/requirements.txt）
source .venv/bin/activate

# 1. 物理验证（CPU 即可，无需 GPU）
python rl/verify/verify_model.py        # 11/11 通过

# 2. 可视化（需图形环境，WSL 用 WSLg）
python rl/verify/launch_viewer.py

# 3. 确认 GPU 可见
python -c "import jax; print(jax.devices())"   # [CudaDevice(id=0)]

# 4. 查看环境/训练规格
python rl/env/kuafu_env.py
python rl/train/train_config.py
```

## 训练管线（下一步，物理验证通过后）

```
阶段0  LQR baseline 验证 ──── ✅ 已通过（verify_model.py [6/6]）
  ↓
阶段1  残差 RL（M5）────────── ⏳ PPO + MJX 4096 envs + 域随机化 + curriculum
  ↓                              收敛: 恢复时间 < LQR×0.85
阶段2  Student 蒸馏 ──────── ⏳ adapter z 监督 + policy DAgger/KL
  ↓                              student ≥ teacher×0.9
阶段3  原生 MuJoCo 对拍 ──── ⏳ CPU 单环境可视化
  ↓
阶段4  ONNX 导出 ─────────── ⏳ python rl/export/export_policy.py
  ↓
阶段5  实机部署（M5）────── 待硬件就绪
```

## 仿真模型要点

`kuafu.xml` 是仓库首个 MJCF，手写 + IK 精确几何。关键：

- **闭链五杆**：每腿两根串联肘链，Q 点 `<equality><connect>` 焊死。驻留态闭链残差 0 mm。
- **混合执行器**：轮 `<motor>`（力矩 ±1.1 Nm），腿 `<position>`（kp=80 kv=2 forcerange±1 Nm）。
- **质量**：总 2.205 kg，电子件用 `<inertial>` 质量点压低 COM 到 71.1 mm（KUAFU §5.4）。
- **同一 XML 兼容** MJX（GPU 训练）与原生 MuJoCo（CPU 可视化/对拍）。

## 与文档的对应

| 模块 | 文档出处 |
|------|---------|
| 仿真建模 | design.md §三 |
| 观测/动作空间 | design.md §2.1-2.2 |
| Reward | design.md §2.3 |
| 域随机化 | design.md §2.4 |
| Teacher-Student+RMA | design.md §2.5 |
| 训练管线 | design.md §2.6 |
| 部署链路 | design.md §六 |
| 物理基线 | KUAFU.md §6.4 |

## 依赖

见 `rl/requirements.txt`。验证仅需 `mujoco`（CPU），训练需 `jax[cuda]`/`mujoco-mjx`/`flax`/`rsl-rl`。
