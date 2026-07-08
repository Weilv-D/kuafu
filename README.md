# 夸父 KUAFU — 桌面双轮腿机器人

面向强化学习的桌面级双轮腿验证平台，"大脑 + 小脑"分层架构，整合轮式平衡与腿式姿态变换。

## 项目结构

```
KuaFu/
├── README.md
├── kuafu_physics.py                ← 物理常量真源（analysis + rl 共用）
├── mechanism.html                  ← 3D 交互可视化
├── docs/                           ← 设计文档
│   ├── KUAFU.md                    ← 单一真源（总览/硬件/结构/分析/装机/符号）
│   └── plans/                      ← 设计过程文档
├── analysis/                       ← 机构分析代码（运动学/静力学/动力学）
│   ├── kuafu/                      ← 共享包 (机构解算 + 统一样式)
│   ├── plot/  optimize/  test/
│   └── run_all.py
└── rl/                             ← 强化学习模块（仿真 + 训练管线）
    ├── kuafu.xml                   ← MJCF 仿真模型（五杆闭链 + 混合执行器）
    ├── verify/                     ← 物理验证 + 可视化 viewer
    ├── env/                        ← RL 环境（obs/act/reward/域随机化）
    ├── train/                      ← PPO 训练配置 + 课程
    └── export/                     ← ONNX 导出（PyTorch → 部署）
```

## 快速开始

```bash
cd analysis
pip install -e .          # 安装 kuafu 包 (一次性) + 依赖
python run_all.py        # 生成图表 + 运行测试
```

## 关键参数

整机 2.21 kg（含充电宝），包络 240 × 231 × 147 mm（驻留态）。
机构参数（d/a/b/D₀）经蒙特卡洛 10 万采样优化验证，详见 `docs/KUAFU.md`。

---

v2.4 / 2026-07-08
