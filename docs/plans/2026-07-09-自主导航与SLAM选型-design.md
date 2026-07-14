# 夸父 KUAFU — 自主导航与 SLAM 选型设计

> **历史规划文档**：本文保留自主导航选型记录；当前部署、协议和训练契约以 `../contracts/interface.md`、`../operations/deployment.md` 和 `../validation/acceptance.md` 为准。

> **文档版本**: v1.0 / 2026-07-09
> **关联**: 《[KUAFU](../KUAFU.md)》、[《软件架构与 RL 技术路线》](./2026-07-08-软件架构与RL技术路线-design.md)。硬件规格、分层架构、RL 策略、通信链路以这两份为准；本文聚焦自主侧的定位 / SLAM / 规划选型，以及与已建成的遥控接口（`rl/teleop/`）如何衔接。
> **核实日期**: 2026-07。

本文为遥控接口（手柄/键盘，已实现于 `rl/teleop/`）补齐**自主侧**的选型：机器人如何知道自己在哪（定位）、如何建图（SLAM）、如何规划到目标（全局/局部规划）。遥控与自主共用同一套 `Command` 抽象，经 `CommandArbiter` 仲裁（手柄抢占式，见 ADR-7 集成）。所有决策有调研依据，关键决策记录(ADR)见 §零。

---

## 零、决策记录 (ADR)

> 本节是所有技术决策的记录。每条决策记录约束、选项、依据，后续文档与代码以此为准。与《软件架构》ADR-1~6 同一编号序列，本文从 ADR-7 起。

### ADR-7 自主命令接入：经 `AutonomousSource` → `CommandArbiter`（与遥控共用接口）

- **约束**：遥控（手柄/键盘）已实现于 `rl/teleop/`，仲裁器 `CommandArbiter` 按"手柄抢占式"语义工作（手柄一动即夺权，松手超 `handoff_time` 交还自主，见 `rl/teleop/arbiter.py`）。自主规划器必须零侵入接入，不改仲裁器、不改策略。
- **选定**：自主规划器输出 `[v, ω, d0]`，经 `AutonomousSource`（`rl/teleop/autonomous_source.py`，当前为 stub）包装成 `Command(mode=AUTONOMOUS)`，注入仲裁器。仲裁器已实现抢占/交还/ramp/限幅/急停，自主源只产命令，**不碰平衡**。
- **依据**：策略(policy)本就是 command-following（obs 第 30-32 维 = `[v_cmd, ω_cmd, d0_cmd]`，见 `rl/env/kuafu_env.py` OBS_SPEC）。遥控与自主的差别只在"谁来产这个 3 维命令"。统一到 `CommandSource` Protocol 后，新增任意命令源只写一个类。
- **结论**：自主规划器是又一个 `CommandSource`。半自动 = 手柄抢占式仲裁（已实现），自主规划器零改动接入。

### ADR-8 定位方案：轮式里程计常驻 + 视觉增量（分层定位）

- **约束**：Pi5（aarch64，8GB，无独立 GPU）算力有限；倒立摆运动学限制了累积里程（不能急转、速度保守）；BMI088 IMU + 轮编码器（电机自带）常驻可用，D435i 建图阶段可用。
- **选定分层定位**：
  - **L1 航点回放**：轮编码器（左右轮速）+ BMI088（Mahony/互补滤波）做航位推算(dead reckoning)，常驻运行。
  - **L2/L3**：在 L1 基础上叠加 RTAB-Map 视觉定位（见 ADR-9），里程计作为 RTAB-Map 的外部 odom 源，视觉提供回环修正与全局一致性。
- **依据**：室内良好标定下，轮编码器+IMU 漂移约行驶距离的 **0.7%–2.7%**（[Imperial College JFR2010](https://www.doc.ic.ac.uk/~ajd/Publications/civera_etal_jfr2010.pdf) ~0.7%；[UGA Thesis](https://openscholar.uga.edu/record/5593/files/IqbalJawadMS.pdf) 1.3%±0.87%~2.7%）。桌面小场景单次行程通常 < 10 m，对应 0.1–0.3 m 漂移，航点回放可接受。IMU 显著降低角度漂移（差速驱动航向误差是位置误差主导源，[ScienceDirect 2024 AGV 定位](https://www.sciencedirect.com/science/article/pii/S240584402410998X)）。纯 dead reckoning 不适合闭环建图（漂移不可逆），建图阶段仍需视觉回环。
- **否决"纯视觉独立定位"**：D435i 在快速运动下视觉里程计退化（运动模糊→tracking loss），倒立摆虽运动保守，但定位不应单点依赖视觉。里程计常驻、视觉增量的分层更鲁棒。
- **结论**：L1 用轮式里程计+IMU（零硬件成本，常驻）；L2+ 叠加 RTAB-Map 视觉。

### ADR-9 SLAM 栈：RTAB-Map（RGB-D，Jazzy 官方）

- **约束**：无 LiDAR（仅有 D435i RGB-D）；ROS2 Jazzy；Pi5 aarch64 CPU（无 GPU）；需回环检测做全局一致性。
- **选定 RTAB-Map**：
  - **官方支持 ROS2 Jazzy**（v0.23.7，`sudo apt install ros-jazzy-rtabmap-ros`，[Jazzy 文档](https://docs.ros.org/en/jazzy/p/rtabmap/)）。
  - **原生消费 D435i RGB-D**：直接消费 `aligned_depth + color + camera_info`，提供 `rgbd` 配置（[introlab/rtabmap_ros](https://github.com/introlab/rtabmap_ros)）。
  - **内置回环检测 + 视觉里程计**，可接受外部 odom（轮式里程计）作为先验，视觉做修正。
  - **Pi5 CPU 预期**：建图节点 `DetectionRate` 默认 1 Hz（不追求相机帧率），CPU 用 ORB/FAST 特征；参考 Pi3 ~4-5 Hz、Pi4 "reasonably well"（[官方论坛](http://official-rtab-map-forum.206.s1.nabble.com/RGB-D-SLAM-example-on-ROS-and-Raspberry-Pi-3-td1250.html)）。Pi5 预期 **1–5 Hz 建图 + 视觉里程计**，桌面小场景够用，需调参。
- **否决 ORB-SLAM3**：核心仓库 `UZ-SLAMLab/ORB_SLAM3` 仅 ROS1，ROS2 仅有第三方社区封装且停留在 **Humble（非 Jazzy）**（[suchetanrs/ORB-SLAM3-ROS2-Docker](https://github.com/suchetanrs/ORB-SLAM3-ROS2-Docker)）；官方仓库近两年无显著更新，社区封装碎片化、IMU 集成困难（[#10](https://github.com/suchetanrs/ORB-SLAM3-ROS2-Docker/issues/10)）；快速运动 tracking loss（[Rover-SLAM arXiv 2405.03413](https://arxiv.org/html/2405.03413v3)）。无官方 Jazzy 支持是硬伤。
- **否决 Cartographer**：核心输入是 **2D LiDAR**（论文标题即 *Real-Time Loop Closure in 2D LIDAR SLAM*，[Google Research](https://research.google.com/pubs/archive/45466.pdf)），不原生消费 RGB-D；3D 模式需 3D LiDAR（[Intermodalics](https://www.intermodalics.ai/expertise/visual-slam-and-cartographer)）。KUAFU 无 LiDAR。
- **否决 SLAM Toolbox**：订阅 `sensor_msgs/LaserScan`，不消费 depth/PointCloud（[Jazzy 文档](https://docs.ros.org/en/jazzy/p/slam_toolbox/)）。depth 转 LaserScan 是退化用法（丢失高度信息，社区报告 map 被截断，[Robotics SE](https://robotics.stackexchange.com/questions/116936)）。
- **可选 VINS-Fusion（VIO）**：有 ROS2 社区封装（[yangfuyuan/vins_fusion_ros2](https://github.com/yangfuyuan/vins_fusion_ros2)），作为 RTAB-Map 外部里程计增强，非主建图栈。Pi5 开销需实测。
- **结论**：RTAB-Map 主栈（RGB-D + 回环），轮式里程计作外部 odom 源。

### ADR-10 导航栈：Nav2，全局 NavFn + 局部 MPPI

- **约束**：ROS2 Jazzy 标准导航栈；倒立摆**不能瞬停/急转**（加速度、急转受限，否则摔），速度命令必须平滑且留平衡余量；差速驱动运动学。
- **选定 Nav2**：ROS2 官方导航栈，Jazzy 完整支持（[Jazzy 迁移文档](https://docs.nav2.org/migration/Jazzy.html)）。原生支持差速驱动（differential drive）。
- **全局规划器**：**NavFn**（默认，基于 Dijkstra/A* 势场法，[算法选择指南](https://docs.nav2.org/setup_guides/algorithm/select_algorithm.html)）。桌面小场景够用；复杂场景可换 SmacPlanner2D/Theta\*。
- **局部规划器：MPPI（首选）/ TEB（备选）**：
  - **MPPI**（`nav2_mppi_controller`）支持**显式** `ax_max/ax_min`、`aw_max/aw_z_max`（[配置文档](https://docs.nav2.org/configuration/packages/configuring-mppic.html)），内置 `DifferentialDriveMotionModel`（[README](https://api.nav2.org/nav2-rolling/html/md_nav2_mppi_controller_README.html)）。采样式 MPC，整段时域轨迹整体优化，避免瞬时跳变。Nav2 官方定位为 TEB 继任者，Jazzy 已用 Eigen 重写提速 40-45%。对低加速度高惯量车辆有专门改进（[Open Navigation: MPPI for High-Inertia Vehicles](https://opennav.org/news/mppi-low-acceleration/)），与倒立摆"低加速度、不能急停"高度契合。
  - **TEB**（备选）：显式 `acc_lim_x/acc_lim_theta`、`min_turning_radius`，时间最优轨迹，参数成熟（[Fixstars 详解](https://blog.us.fixstars.com/explanation-of-teb_local_planner-algorithm-and-parameters/)）。MPPI 若调参困难回退 TEB。
- **否决 DWB（默认 DWA）**：基于速度窗口采样，易在障碍前产生急停/急转（[DWB 文档](https://docs.nav2.org/configuration/packages/configuring-dwb-controller.html)），对倒立摆不友好。
- **否决 RPP（Regulated Pure Pursuit）**：曲率跟随，弯道处曲率突变对平衡不友好，适合结构化道路非平衡机器人。
- **关键补强：`cmd_vel` → STM32 间插速度平滑层**（**必加，不可省略**）：
  - Nav2 不建模倒立摆的俯仰动力学/倾覆稳定性，它只把机器人当受运动学约束的差速底盘。
  - 平衡余量必须通过：**保守的 `ax_max/aw_max`** + **S-curve / jerk-limit 速度平滑器**（纯梯形仍有加速度阶跃，[MDPI Sensors 2024 jerk-limited 规划](https://www.mdpi.com/1424-8220/24/16/5332)）+ **底层 LQR/平衡控制器接管**实现。
  - 已实现的 `CommandArbiter` 的 ramp/限幅（`rl/teleop/arbiter.py`）正是这层平滑——自主命令同样经它，**自主规划器的输出天然被 ramp/限幅保护**。这是 ADR-7 集成的附带收益：自主和遥控共用同一安全层。
- **结论**：Nav2 + NavFn 全局 + MPPI 局部；`cmd_vel` 经 `AutonomousSource` → `CommandArbiter`（ramp/限幅）→ policy，平衡由 LQR 兜底，导航层不碰电机。

---

## 一、需求与约束

### 1.1 场景

| 场景 | 命令来源 | 自主等级 | 本文覆盖 |
|---|---|---|---|
| 桌面演示 / 手动遥控 | 手柄/键盘 | L0 遥控 | ✅ 已实现（`rl/teleop/`） |
| 航点回放（预设路径点跟踪） | 规划器 | L1 | 本文选型 |
| 目标点导航 + 避障 | Nav2 局部规划 + D435i | L2 | 本文选型 |
| 全自主建图漫游 | RTAB-Map + Nav2 | L3 | 本文选型 |

### 1.2 硬件约束

| 项 | 规格 | 对导航的影响 |
|---|---|---|
| 主控 | 树莓派5（8GB，aarch64，无独立 GPU） | SLAM 必须 CPU 可跑；算力预算紧 |
| 感知 | D435i（RGB-D，USB3）、BMI088（IMU） | RGB-D 建图可用，无 LiDAR |
| 编码器 | 轮电机自带 + 舵机回读 | 轮式里程计可行 |
| 通信 | 自定义 UART Pi↔STM32（ADR-4，非 micro-ROS） | 命令经 `CommandArbiter` 下发，非 ROS topic 直驱电机 |
| 机构 | 双轮腿倒立摆，整机 2.68kg | **不能瞬停/急转**，加速度/jerk 受限 |

### 1.3 倒立摆运动学约束（选型核心驱动因素）

这是与普通差速底盘导航的根本差异，贯穿 ADR-8/9/10：

1. **速度/加速度/jerk 上限**：最大线/角速度取平衡控制器稳定裕度的子集（约 60-70%）；加速度需 S-curve 或 jerk-limit 平滑（[INRIA 速度剖面](https://inria.hal.science/hal-00732930/PDF/Path_and_speed_planning_for_smooth_autonomous_navigation.pdf)、[MDPI jerk-limited](https://www.mdpi.com/1424-8220/24/16/5332)）。
2. **禁止原地急转 / 禁止急停**：差速原地旋转对倒立摆危险，TEB 设非零 `min_turning_radius` 或严格限 `aw_max`。
3. **分层控制**：Nav2（运动学规划）→ 平滑层（`CommandArbiter`）→ STM32（LQR 平衡 + 俯仰控制），导航层不直接驱动电机（[自平衡机器人导航研究](https://irispublishers.com/ojrat/pdf/OJRAT.MS.ID.000554.pdf) 的级联 PID 范式）。
4. **给底层平衡留余量**：导航层的速度包络是平衡稳定域的子集。

---

## 二、定位方案选型

### 2.1 分层定位（ADR-8）

```
┌─────────────────────────────────────────────────────┐
│ L1 常驻: 轮式里程计 + IMU (dead reckoning)            │
│   左右轮速 → 差速运动学 → (x, y, θ)                  │
│   BMI088 (Mahony/互补滤波) → 修正 θ                   │
│   漂移 ~0.7-2.7% 行程, 桌面 <10m 可接受              │
└───────────────────────┬─────────────────────────────┘
                        │ odom (常驻, 低延迟)
┌───────────────────────▼─────────────────────────────┐
│ L2/L3 增量: RTAB-Map 视觉定位                         │
│   D435i RGB-D → 视觉里程计 + 回环检测                 │
│   外部 odom = L1 轮式里程计 (先验)                    │
│   视觉提供全局一致性, 修正 dead reckoning 累积漂移    │
└─────────────────────────────────────────────────────┘
```

### 2.2 轮式里程计建模（L1 核心）

差速驱动运动学：
```
v = (v_l + v_r) / 2          # 线速度 (左右轮速均值)
ω = (v_r - v_l) / W          # 角速度 (W = 轮距)
ẋ = v·cos(θ);  ẏ = v·sin(θ);  θ̇ = ω
```
- 轮速来源：DDSM315 编码器（转速 rpm，`kuafu_physics.py` `RPM_WHEEL_*`）。
- 航向修正：BMI088 陀螺仪 Z 轴（Mahony 滤波，STM32 侧已 1kHz 跑，见 ADR-5）。
- 漂移来源：轮径误差、打滑、编码器量化；IMU 主要压制角度漂移（位置漂移靠视觉回环修正）。

---

## 三、SLAM 栈选型（ADR-9）

### 3.1 候选对比

| SLAM 栈 | 输入 | ROS2 Jazzy | aarch64/Pi5 CPU | 回环检测 | 适用? |
|---|---|---|---|---|---|
| **RTAB-Map** | RGB-D ✅ | ✅ 官方 v0.23.7 | ⚠️ 1-5Hz 调参 | ✅ | **✅ 选定** |
| ORB-SLAM3 | RGB-D（社区封装） | ❌ 仅 Humble 第三方 | ⚠️ | ✅ | ❌ 维护风险 |
| VINS-Fusion | 相机+IMU | ⚠️ 社区封装 | ⚠️ | ❌（VIO 非 SLAM） | ⚠️ 可选增强 |
| Cartographer | **2D LiDAR** ❌ | — | — | ✅ | ❌ 无 LiDAR |
| SLAM Toolbox | **2D LiDAR** ❌ | — | — | ✅ | ❌ 无 LiDAR |

### 3.2 选定 RTAB-Map 的部署形态

- 建图模式：`rgbd` 配置，消费 D435i 的 `aligned_depth_to_color` + `color` + `camera_info`。
- 外部里程计：喂入 L1 轮式里程计（`/odom`），RTAB-Map 的视觉里程计作为 `odom` 的修正/验证。
- `DetectionRate` 默认 1 Hz（建图节点不追求相机帧率），CPU 用 ORB/FAST 特征，避免 SURF/SIFT（GPU 依赖）。
- 定位模式（L2 目标点导航）：加载建好的地图，纯定位（localization），频率可高于建图。

---

## 四、规划层选型（ADR-10）

### 4.1 全局规划

| 规划器 | 算法 | 适用 | 选定? |
|---|---|---|---|
| **NavFn** | Dijkstra/A* 势场 | 通用 2D 占据栅格 | **✅ 默认** |
| SmacPlanner2D | Smac(A*) | 复杂场景/更优路径 | 备选 |
| Theta\* | 视线优化 | 路径更平滑 | 备选 |

桌面小场景 NavFn 足够。路径需后处理平滑（B-spline/Bezier，[PMC 路径平滑综述](https://pmc.ncbi.nlm.nih.gov/articles/PMC6165411/)），减少被迫急减速。

### 4.2 局部规划（倒立摆约束核心）

| 控制器 | 加速度约束 | 平滑性 | 倒立摆适配 | 选定? |
|---|---|---|---|---|
| **MPPI** | ✅ 显式 `ax_max/aw_max` | ✅ 时域整体最优 | ✅ 最契合 | **✅ 首选** |
| TEB | ✅ 显式 `acc_lim_*` | ✅ 时间最优 | ✅ | 备选 |
| RPP | ⚠️ 有限 | ⚠️ 弯道曲率突变 | ❌ | 否决 |
| DWB | ⚠️ 间接 | ❌ 易急停急转 | ❌ | 否决 |

**关键**：无论选 MPPI 还是 TEB，`cmd_vel` 都**不能**直接下发电机。必须经 `CommandArbiter` 的 ramp/限幅层（ADR-7 集成已提供），再由 LQR 平衡控制器落地。Nav2 不建模俯仰动力学。

---

## 五、与现有架构集成

### 5.1 命令流拓扑（遥控 + 自主统一）

```
┌───────────┐  ┌───────────────────────┐  ┌──────────┐
│ 手柄/键盘  │  │ Nav2 (全局 NavFn +    │  │ 急停按钮  │
│GamepadSrc │  │  局部 MPPI)           │  │(任意源)  │
│KeyboardSrc│  │  → /cmd_vel [v,ω]     │  │          │
└─────┬─────┘  └──────────┬────────────┘  └────┬─────┘
      │ MANUAL            │ AUTONOMOUS          │ ESTOP
      │                   ▼                     │
      │        ┌────────────────────┐           │
      │        │ AutonomousSource   │           │
      │        │ (rl/teleop/, stub) │           │
      │        │ /cmd_vel → Command │           │
      │        └─────────┬──────────┘           │
      │                  │                      │
      └────────┬─────────┴──────────────────────┘
               ▼
      ┌────────────────────────────┐
      │   CommandArbiter            │  ← 已实现, rl/teleop/arbiter.py
      │  优先级: ESTOP > 遥控 > 自主 │
      │  手柄一动即抢占自主(半自动)   │
      │  ramp/限幅/超时/急停         │  ← 自主命令同样经此, 天然受平滑保护
      └──────────────┬─────────────┘
                     │ Command [v,ω,d0]
                     ▼
      ┌──────────────────────────┐
      │  Policy (零改动)          │  obs[30:33] = [v,ω,d0]
      │  (ONNX, Pi5, 50Hz)        │
      └──────────────┬───────────┘
                     │ action (残差)
                     ▼
      ┌──────────────────────────┐
      │  STM32: LQR + 平衡 + 电机环 │  ← 平衡兜底, 导航层不碰
      └──────────────────────────┘
```

### 5.2 关键设计原则

1. **规划器只产 `[v, ω, d0]`，不碰底层**：平衡由 policy + LQR 保证，规划器不关心俯仰/电机。
2. **手柄永远能抢占**：自主运行中，人握手柄一动，`CommandArbiter` 立即切 MANUAL（ADR-7），自主挂起；松手超 `handoff_time`（默认 1.5s）才交还自主。
3. **自主命令天然受 ramp/限幅保护**：Nav2 输出经 `AutonomousSource` → `CommandArbiter`，仲裁器的 ramp/限幅（ADR-10 关键补强）自动生效，无需为自主另写平滑层。
4. **d0（姿态）策略**：自主导航时 d0 通常保持驻留态（`D0_MIN`，腿自锁、整机降为纯轮式倒立摆，最稳）；需要越障时再由上层抬升。规划器可选输出 d0，默认驻留。

### 5.3 接入清单（未来实现）

| 组件 | 当前状态 | 实现位置 |
|---|---|---|
| `AutonomousSource` | stub（`poll()` 返回 None） | `rl/teleop/autonomous_source.py`：订阅 `/cmd_vel`，包装成 `Command(mode=AUTONOMOUS)` |
| Nav2 配置 | 未实现 | Pi5：`nav2_bringup` + MPPI 参数（保守 `ax_max/aw_max`） |
| RTAB-Map 建图 | 未实现 | Pi5：`rtabmap_slam`（rgbd 配置，外部 odom = 轮式里程计） |
| 轮式里程计节点 | 未实现 | STM32 上报轮速 → Pi5 `odom_publisher`（差速运动学 + IMU Mahony） |
| 速度平滑 | **已实现**（`CommandArbiter` ramp/限幅） | 自主命令复用，无需另写 |

---

## 六、里程碑（自主路线）

| 级别 | 目标 | 需要 | 验收标准 |
|---|---|---|---|
| **L1** | 航点回放 | 轮式里程计 + 纯跟踪(pure pursuit) | 给定路径点序列，跟踪到位姿误差 < 0.3m，全程不摔 |
| **L2** | 目标点导航 + 避障 | L1 + Nav2(MPPI) + D435i 局部避障 | 人给目标点，自主规划到达，遇障碍绕行，不摔 |
| **L3** | 全自主建图 | L2 + RTAB-Map 建图 + Nav2 全局规划 | 未知环境建图后自主导航，回环修正漂移 |

每级都是遥控（已实现）的增量：L1 起自主源接入 `CommandArbiter`，手柄随时可抢占修正。

---

## 附录 A：调研索引

| 主题 | 关键来源 |
|---|---|
| RTAB-Map (Jazzy + RGB-D) | [rtabmap Jazzy 文档](https://docs.ros.org/en/jazzy/p/rtabmap/)、[introlab/rtabmap_ros](https://github.com/introlab/rtabmap_ros)、[Pi3 性能论坛](http://official-rtab-map-forum.206.s1.nabble.com/RGB-D-SLAM-example-on-ROS-and-Raspberry-Pi-3-td1250.html)、[特征选择 #358](https://github.com/introlab/rtabmap_ros/issues/358) |
| ORB-SLAM3 否决 | [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)、[ROS2 社区封装(Humble)](https://github.com/suchetanrs/ORB-SLAM3-ROS2-Docker)、[#10 IMU 集成问题](https://github.com/suchetanrs/ORB-SLAM3-ROS2-Docker/issues/10)、[Rover-SLAM tracking loss](https://arxiv.org/html/2405.03413v3) |
| Cartographer/SLAM Toolbox 否决(需 LiDAR) | [Cartographer 论文](https://research.google.com/pubs/archive/45466.pdf)、[Intermodalics 非视觉 SLAM](https://www.intermodalics.ai/expertise/visual-slam-and-cartographer)、[SLAM Toolbox Jazzy](https://docs.ros.org/en/jazzy/p/slam_toolbox/)、[depth 转 LaserScan 退化](https://robotics.stackexchange.com/questions/116936) |
| VINS-Fusion (VIO 增强) | [yangfuyuan/vins_fusion_ros2](https://github.com/yangfuyuan/vins_fusion_ros2)、[UTMSYS Wiki](https://wiki.utmsys.org/en/Algorithm_Development/SLAM/VINS-Fusion/) |
| 轮式里程计漂移 | [Imperial College JFR2010 ~0.7%](https://www.doc.ic.ac.uk/~ajd/Publications/civera_etal_jfr2010.pdf)、[UGA Thesis 1.3-2.7%](https://openscholar.uga.edu/record/5593/files/IqbalJawadMS.pdf)、[ScienceDirect 2024 IMU 融合](https://www.sciencedirect.com/science/article/pii/S240584402410998X) |
| Nav2 (Jazzy 标准) | [Nav2 Jazzy 迁移](https://docs.nav2.org/migration/Jazzy.html)、[算法选择指南](https://docs.nav2.org/setup_guides/algorithm/select_algorithm.html) |
| MPPI 局部规划器 | [配置文档](https://docs.nav2.org/configuration/packages/configuring-mppic.html)、[README(运动模型)](https://api.nav2.org/nav2-rolling/html/md_nav2_mppi_controller_README.html)、[高惯量车辆改进](https://opennav.org/news/mppi-low-acceleration/)、[差速调参 #5375](https://github.com/ros-navigation/navigation2/issues/5375) |
| TEB / DWB / RPP | [TEB 详解](https://blog.us.fixstars.com/explanation-of-teb_local_planner-algorithm-and-parameters/)、[DWB 配置](https://docs.nav2.org/configuration/packages/configuring-dwb-controller.html) |
| 倒立摆导航特殊处理 | [自平衡导航级联 PID](https://irispublishers.com/ojrat/pdf/OJRAT.MS.ID.000554.pdf)、[轨迹规划+动态平衡 ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S0094114X25002368)、[INRIA 速度剖面](https://inria.hal.science/hal-00732930/PDF/Path_and_speed_planning_for_smooth_autonomous_navigation.pdf)、[MDPI jerk-limited 规划](https://www.mdpi.com/1424-8220/24/16/5332)、[路径平滑综述 PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6165411/) |
| 开源案例 | [Autonomous-Navigation-of-Self-Balancing-Segway](https://github.com/Jash-2000/Autonomous-Navigation-of-Self-Balancing-Segway) |
