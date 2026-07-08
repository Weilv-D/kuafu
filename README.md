# 夸父 KUAFU — 桌面双轮腿机器人

面向强化学习的桌面级双轮腿验证平台，"大脑 + 小脑"分层架构，整合轮式平衡与腿式姿态变换。

## 项目结构

```
KuaFu/
├── README.md
├── docs/                           ← 设计文档
│   ├── KUAFU项目信息书.md
│   ├── KUAFU结构设计.md
│   ├── KUAFU运动学与动力学分析.md
│   └── SYMBOLS.md
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

| 符号 | 值 | 含义 |
|------|-----|------|
| d | 52 mm | 髋距 |
| a | 93 mm | 曲柄（大腿） |
| b | 149 mm | 连杆（小腿） |
| D₀ | 58–207 mm | 足端下垂量 |
| Y | 98 mm | 半轮距 |
| r | 39.08 mm | 轮半径 |

详见 `docs/SYMBOLS.md`。

---

v2.0 / 2026-06-25
