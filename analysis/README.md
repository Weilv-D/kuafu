# KUAFU 运动学与动力学分析

本目录用数值方法验证《[运动学与动力学分析](../docs/KUAFU运动学与动力学分析.md)》中的全部结论。
**目的**：证明机构参数 d=52, a=93, b=149, D₀∈[58,207] 在运动学、静力学、动力学、整机平衡四个层面自洽且裕度充足，支撑《项目信息书》的验收指标。符号定义见《[SYMBOLS](../docs/SYMBOLS.md)》。

## 分析逻辑：四章 → 四图 → 各自证明什么

分析沿"几何可行 → 受力可行 → 运动可行 → 整机可行"四层递进，每层一张图、一个结论：

```
第一章 运动学      plot_workspace.py       →  workspace_envelope.png
   机构能否到位?     足端可达包络 + 非交叉分支约束 + 各高度补偿带宽
   结论: 透镜包络 331 cm²; 驻留补偿 ±45mm, 中段 ±118mm > 需求 ±25mm (4× 富余)
        ↓
第二章 静力学      plot_force_ellipsoid.py → force_ellipsoid.png
   到位后舵机扛得住吗?  τ=JᵀF 力雅可比 → 扭矩/力椭球 + 奇异值谱 + 承载极限
   结论: 设计载下 τ 0.51–1.92 Nm < 堵转 2.94; 驻留态全悬空仍可承 41.8N > 设计载 30N
        ↓
第三章 动力学      plot_dynamics.py        →  dynamics.png
   动起来惯性够吗?    拉格朗日 M/C/G + 爬阶轨迹反解动态扭矩
   结论: 动态增量 ~0.3 Nm ≪ 静载, 惯性可忽略; 爬阶 1s/级角速度可行
        ↓
第四章 整机平衡    plot_lqr.py             →  lqr_balance.png
   整机不倒吗?       驻留态腿自锁 → cart-pole 降阶 → LQR 闭环仿真
   结论: 闭环极点全左半平面稳定; 准静态扭矩裕度极大; 瓶颈在控制带宽非扭矩
```

**贯穿全文的安全准则**：所有瞬态动作（D₀>70）遵循"脉冲→回承重面锁止"，舵机长期可靠靠驻留态零力矩自锁（G≈0.05 Nm）+ 承重面绕过舵机轴的双重保障，而非持续大扭矩。

## 目录结构

```
analysis/
├── kuafu/                  共享包: 机构解算 + 统一样式 (职责分离)
│   ├── mechanism.py        运动学/静力学/动力学/力椭球几何 (纯 numpy, 无 matplotlib)
│   └── styling.py          调色板 + rcParams + 绘图工具 (matplotlib)
├── plot/                  四个绘图脚本, 各对应文档一章
│   ├── plot_workspace.py       §1.2 工作空间
│   ├── plot_force_ellipsoid.py §2.3 力椭球与承载能力
│   ├── plot_dynamics.py        §2.1/§3.2 静力扭矩 + 爬阶动力学
│   └── plot_lqr.py             §4    整机倒立摆 LQR
├── test/
│   └── test_kinematics.py 单元测试 (16 项), 锁定核心数值结论
├── output/                生成的四张图 (run_all 自动写入, 已纳入版本控制)
├── run_all.py             一键复现: 生成四图 + 跑测试
├── pyproject.toml         包定义 (kuafu 为可导入包)
└── requirements.txt
```

## 复现

```bash
pip install -e .          # 让 kuafu 成为可导入包 (一次性)
python run_all.py          # 生成 output/*.png + 跑全部测试
```

单独运行某张图：`python plot/plot_dynamics.py`（脚本直接 `import kuafu`，无需 sys.path hack）。

## 科学性如何被验证

数值结论不止画在图上，更由单元测试锁定。测试与文档结论的对应：

| 测试类 | 锁定的结论 | 文档出处 |
|--------|-----------|---------|
| `TestKinematics` | 驻留 τ=0.51, 峰值 τ=1.92, 上限 τ=1.44 Nm; γ≥31° 全程 | §2.1 表 |
| `TestKinematics` | 静载(9.96N)下 τ < 连续安全值 65% | §2.2 |
| `TestJacobian` | 全程 κ<2.0 远离奇异 | §1.4 |
| `TestForceEllipse` | 力椭球边界 ‖JᵀF‖₂≡τ_lim 恒定; 扭矩椭球半轴=σᵢ/1000 | §2.3 |
| `TestWorkspace` | 驻留中心可达, ±50mm 外不可达 | §1.2 |

测试通过即代表核心数值结论未被回归破坏。
