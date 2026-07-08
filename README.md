# 夸父 KUAFU — 桌面双轮腿机器人

面向强化学习的桌面级双轮腿验证平台，"大脑 + 小脑"分层架构，整合轮式平衡与腿式姿态变换。

## 项目结构

```
KuaFu/
├── README.md
├── docs/                           ← 设计文档
│   └── KUAFU.md                    ← 单一真源（总览/硬件/结构/分析/装机/符号）
├── analysis/                       ← 分析代码
│   ├── kuafu/                      ← 共享包 (机构解算 + 统一样式)
│   │   ├── mechanism.py            ←   运动学/静力学/动力学/力椭球 (纯 numpy)
│   │   └── styling.py              ←   色盲安全色板 + 出版级排版
│   ├── plot/                       ← 绘图脚本
│   │   ├── plot_dynamics.py
│   │   ├── plot_force_ellipsoid.py
│   │   ├── plot_workspace.py
│   │   └── plot_lqr.py
│   ├── test/                       ← 单元测试
│   │   └── test_kinematics.py
│   ├── output/                     ← 生成图表
│   ├── run_all.py                  ← 一键运行
│   ├── pyproject.toml              ← 包定义 (pip install -e .)
│   └── requirements.txt
├── 3Dmodel/                        ← CAD 模型
└── mechanism.html                  ← 3D 交互可视化
```

## 快速开始

```bash
cd analysis
pip install -e .          # 安装 kuafu 包 (一次性) + 依赖
python run_all.py        # 生成图表 + 运行测试
```

## 关键参数

整机 2.21 kg（含充电宝），包络 240 × 231 × 147 mm（驻留态），预算 ≤1500 元。
机构参数（d/a/b/D₀）经蒙特卡洛 10 万采样优化验证，详见 `docs/KUAFU.md`。

---

v2.1 / 2026-07-08
