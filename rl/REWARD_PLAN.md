# KUAFU Reward 迭代计划

> 单一真源：reward 设计、版本基线、eval 标准、迭代决策记录。
> v1 训练已完成（2026-07-09），本文档驱动 v2+ 迭代。

## 核心原则

1. **演进，不革命** — 当前 9 项 reward 已对齐 Go1/T1/Berkeley Humanoid 主流实践，
   v1 已收敛（reward -0.69→31.3, episode 65→888 步）。绝不推倒重设计。
2. **先 eval，再改** — 代码审查发现的"缺 ang_vel_xy"是假设，必须 deterministic eval
   验证失败模式后才能动手。拿假设赌 5h GPU 是最大浪费。
3. **一次只改一项** — A/B 对照，建立 reward 台账，每项作用可归因。
4. **改 reward 项 → 从头 retrain，不 fine-tune** — 加/删 reward term 改变优化地貌，
   fine-tune 的 stale value + 低 exploration(0.04) 无法逃出旧盆地。

## v1 基线（已存档）

| 项目 | 值 |
|---|---|
| checkpoint | `checkpoints/teacher_1783531465/model_2999.pt` |
| 配置 | 1024 envs × 3000 iter × 24 steps = 74M steps |
| 时长 | 5h29m |
| 最终 reward | 31.29 |
| 最终 episode length | 827 步 (上限 1000 = 20s) |
| reward 组成 | 9 项 (见 `kuafu_mjx_env.py:_compute_reward`) |

### v1 reward 组成

```
task (正向):
  1.0 × lin_vel_tracking    exp(-(xdot-v_cmd)²/0.25)
  0.5 × ang_vel_tracking    exp(-(wz-w_cmd)²/0.25)
  1.0 × orientation         exp(-α·(gx²+gy²)), α=8.0   [重力向量]
  0.3 × default_pose        exp(-(hip_actual-hip_target)²/0.1)

maintain:
  0.1 × alive               常数 1.0
  0.01 × action_rate        -‖a_t - a_{t-1}‖²

penalty:
  0.001 × energy            -(|ω轮·τ轮| + τ髋²)        [分执行器类型]
  0.5 × torque_limit        -max(|τ髋|-τ_cont, 0)
```
step 中: `reward = total × CTRL_DT`

### v1 已知缺口（待 eval 验证）

| 候选项 | 形式 | 假设的失败模式 | 优先级 |
|---|---|---|---|
| **ang_vel_xy** | `-(ωx²+ωy²)` 权重 ~0.05 | 姿态正但高频抖动，角速度方差大 | **P0** |
| lin_vel_z | `-vz²` | 竖直方向跳动 | P2 |
| joint_acc | `-‖q̈‖²` | 关节加速度大（action_rate 部分覆盖） | P3 |

## Eval 标准（deterministic）

eval 脚本: `verify/eval_policy.py`（headless，无 viewer）

### 模式

| 模式 | 探索噪声 | DR | 命令 | episode | 目的 |
|---|---|---|---|---|---|
| `deterministic` | 关 | 关 | 固定 (v=0,ω=0,d0=dwell) | 10000 步 (200s) | 真实稳定能力上限 |
| `dr` | 关 | 开 (nominal) | 固定 | 10000 步 | 鲁棒性 |
| `cmd_sweep` | 关 | 关 | 扫描 v∈[-0.5,0.5] | 每 cmd 2000 步 | 命令跟踪能力 |

### 记录指标

| 指标 | 计算 | 健康标准 |
|---|---|---|
| 稳定时长 | 倒下前步数 (pitch/roll>阈值) | deterministic ≥ 数千步 |
| pitch/roll 均方根 | `√(mean(θ²))` | < 2° |
| **角速度方差** | `var(ωx), var(ωy)` | **P0 判据：< 0.01 rad²/s²** |
| lin_vel 跟踪误差 | `mean(|xdot-v_cmd|)` | < 0.05 m/s |
| 动作平滑度 | `mean(‖Δa‖²)` | < 0.01 |

### v1 → v2 判定流程

```
跑 v1 eval (deterministic)
  ├─ 稳定 ≥ 5000 步 + 角速度方差 < 0.01
  │   → v1 已足够, 进入 distill/部署阶段, ang_vel_xy 列入 v2 polish
  ├─ 稳定 ≥ 5000 步 + 角速度方差 ≥ 0.01
  │   → 加 ang_vel_xy (P0), 从头 retrain v2
  └─ 稳定 < 5000 步
      → 分析倒下原因 (漂移? 抖动? 命令跟踪差?)
      → 针对性加项, 从头 retrain v2
```

## v1 Eval 结果（2026-07-09）— 重大发现

### Train/eval gap（非 reward 问题，是架构问题）

| 环境 | deterministic 稳定步数 | pitch RMS |
|---|---|---|
| MJX（训练环境，eval_mjx.py） | **300 步满稳定** | 2.81° |
| 原生 MuJoCo 无 follower（eval_policy.py） | **13-43 步即倒** | 13-14° |
| 原生 MuJoCo 有 follower（eval_policy_follower.py） | **2000 步满稳定** | 3.55° |

**根因：训练 step() 每个 physics substep 做 follower forcing（强制 hip_B/knee 按五杆
运动学耦合），策略依赖这个约束。但 eval（playback.py）和真机部署路径没有 follower
forcing，原生 MuJoCo 的 `<equality><joint>` 约束求解器在闭环机构上会漂移。**

→ 这不是 reward 设计缺陷，加 ang_vel_xy 解决不了。
→ 必须先解决 train/deploy 路径一致性，reward 迭代才有意义。

### 真机控制逻辑确认（design.md ADR-3 + KUAFU.md §5.3.1）

**真机五杆控制**：每条腿 2 个 ST3215 舵机（hip_A + hip_B），各自独立位置控制
（PD 内环，1:345 减速比刚性）。五杆闭链靠**两个舵机各自的位置环物理维持**，
不靠软件约束。design.md 原设计 action 是 **6 维**（2 轮 + 4 舵机）。

**当前仿真实现（b099aa4 对称切片）**：action **4 维**，hip_B **无执行器**，
靠 `<joint equality>` + **follower forcing**（每子步强制覆盖 qpos/qvel）维持。

**矛盾本质**：仿真的 hip_B 是"无执行器的幽灵关节"，被 follower forcing 强制拖动。
真机 hip_B 是真实舵机，必须收到位置指令。策略学到依赖完美刚性约束，
而非学会协调两个舵机。→ follower forcing 是"作弊"，sim-to-real 不可弥合。

### 五杆闭链维持方案验证（原生 MuJoCo 实验）

| 方案 | 闭链 gap | eval 稳定性 | 真机可行性 |
|---|---|---|---|
| joint equality + follower forcing（当前） | 0mm (强制) | ✅ 2000步 | ❌ 真机无 follower |
| site connect 默认 solver | 11.6mm | ❌ 13步倒 | - |
| site connect 极硬 solver | 0.53mm | 未测 | - |
| **双舵机 PD + site connect 硬约束** | **静态0.56mm/动态1.8mm** | **待测** | **✅ = 真机** |

**结论**：双 position actuator（hip_A + hip_B 各自 PD 环）+ 硬 site connect
能稳定维持五杆（物理真实：两舵机 + 刚性连杆），且映射真机控制逻辑。

### 治本修复方案：双舵机 + site connect（方案 1 修正版）

**XML 改动**：
1. hip_B 加 `<position>` actuator（同 hip_A 的 kp=80/kv=2/forcerange±2.94）
2. 6 个 `<joint equality>` 删除
3. 加 2 个 `<connect site1="Q_A_l" site2="Q_B_l"/>` + 右腿（硬 solver 参数）
4. knee 保持被动铰链（无执行器，靠 connect 几何约束跟随）

**env step 改动**：
1. action 仍 4 维（2 轮 + 2 hip_A 位置目标）
2. `ctrl[hip_B] = -ctrl[hip_A]`（软镜像，代码赋值）
3. **删除 follower forcing**（line 655-669）
4. nu 从 4→6（加 hip_B_l/r position actuator）

**真机部署映射**：
- Pi5 下发 4 维 action → STM32 内部 `hip_B_goal = -hip_A_goal`
- STM32 分别给 4 个舵机（hip_A_l/r + hip_B_l/r）位置指令
- 五杆靠 4 个真实 ST3215 位置环 + 物理刚性连杆维持 = 仿真完全一致

## v2 计划（待修复方向决策后更新）

- [ ] 修复 train/eval gap（方向 A/B/C/D 待定）
- [ ] 修复后重跑 v1 eval 验证
- [ ] eval 可信后，再评估是否需要加 ang_vel_xy 等 reward 项
- [ ] **决策已定（D 思路延伸）**：先确定真机控制逻辑 → 已确认真机双舵机各自位置控制

## Reward 台账

| 版本 | 改动 | 基线 | 效果 |
|---|---|---|---|
| v1 | 初始 9 项 (baseline) | episode 827, reward 31.3 | MJX eval 稳定, MuJoCo eval 倒 (gap) |
