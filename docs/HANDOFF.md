# KUAFU 项目交接文档（给新 Agent）

> 最后更新：2026-07-27。本文档面向**完全不了解本项目的工程师/Agent**，读完即可开始开发。

---

## 1. 这是什么项目

**KUAFU（夸父）** 是一台**双轮腿自平衡机器人**（wheel-legged balancer）：两条五连杆腿（4 个 ST3215 总线舵机驱动）+ 两个轮毂电机，靠 LQR 控制倒立摆平衡。

**计算架构（两板）：**
- **STM32F407ZG**（本仓库主体）：实时控制——IMU 融合、LQR 平衡、舵机/轮电机驱动、安全状态机。**250Hz 控制环，1kHz 姿态融合。**
- **树莓派 5**（`pi5_runtime/`）：上层决策、RL 策略推理、运动指令。通过 USART6 @921600 下发。

**重要：本固件能在没有树莓派的情况下独立完成站立平衡**（自平衡是 baseline 能力，Pi 只负责 ACTIVE 模式的运动指令）。无 Pi 时进 STAND 模式原地平衡。

---

## 2. 硬件清单

| 部件 | 型号 | 接口 | 关键参数 |
|------|------|------|----------|
| MCU | STM32F407ZG | — | 168MHz, 192KB RAM |
| IMU | BMI088 | I2C1, DRDY@PB1/EXTI1 1kHz | 6 轴，无磁 |
| 轮电机 ×2 | DDSM315 | USART2 RS485 @115200 | 镜像安装（左+向，右-向） |
| 腿舵机 ×4 | ST3215 | USART3 @1Mbaud，经 Waveshare 适配板 | ID 1-4，半双lex→全双工 |
| Pi 5 | — | USART6 @921600 DMA | — |

**轮电机方向约定**（`pin_config.h`）：`WHEEL_DIR_L=+1`，`WHEEL_DIR_R=-1`（右轮镜像安装，下发时取反）。

---

## 3. 仓库结构

```
kuafu/
├── stm32_firmware/          ← 本项目主体（Keil 工程）
│   ├── Core/
│   │   ├── Inc/             头文件、引脚/参数配置
│   │   │   ├── pin_config.h        ★ 所有硬件/安全参数
│   │   │   ├── kuafu_generated.h   ★ LQR 增益、物理常量（自动生成，勿手改）
│   │   │   ├── lqr_controller.h    LQR 状态机
│   │   │   └── ...
│   │   └── Src/
│   │       ├── main.c              主循环、外设初始化、调度
│   │       ├── lqr_controller.c    LQR/LQI 控制器（★ 经常调）
│   │       ├── mahony.c            姿态融合（Mahony，Kp=10）
│   │       ├── safety_state.c      安全状态机（INIT/STAND/ACTIVE/CLIMB/FAULT）
│   │       ├── startup_manager.c   上电启动序列（IMU发现→陀螺标定→执行器发现→READY）
│   │       ├── bmi088.c/ddsm315.c/st3215.c/pi_link.c  设备驱动
│   │       └── ...
│   ├── MDK-ARM/             Keil 工程文件 + 产物
│   │   ├── stm32_firmware.uvprojx
│   │   ├── stm32_firmware/stm32_firmware.hex   ← 烧录用
│   │   ├── stm32_firmware/stm32_firmware.map   ← 查符号地址用
│   │   └── debug_tools/    ★ SWD 调试脚本（pyocd）
│   └── tests/               主机单元测试（CMake/MSVC）
├── pi5_runtime/             树莓派侧代码
├── rl/                      RL 训练 + MuJoCo 仿真（kuafu.xml）
├── sim_stand_test.py        CPU MuJoCo 站立测试 harness
├── kuafu_physics.py         物理模型（生成 LQR 增益的源）
├── docs/                    本文档所在目录
└── README.md
```

**改 LQR 增益的正确流程**：改 `kuafu_physics.py` → 跑 `generate_artifacts.py` 重新生成 `kuafu_generated.h` → 其中的 `MODEL_HASH` 变化会让 Pi 链路握手重新校验。**不要手改 `kuafu_generated.h`**。

---

## 4. 开发、编译、烧录、测试命令

> **环境**：Windows + Git Bash。所有命令在仓库根 `C:\Users\Deng2\Desktop\temp\kuafu` 下执行。

### 4.1 主机单元测试（不依赖硬件，先跑这个）

```bash
cd stm32_firmware/tests
cmake -B build -G "Visual Studio 17 2022" -A x64     # 首次
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

**必须 100% 通过才能动硬件。** 测试覆盖：LQR 包络/积分钳/重锚、yaw 钳位、safety 状态机、ddsm315/st3215 协议、startup、device_health 等。

> ⚠️ 中文 Windows：`CMakeLists.txt` 里 MSVC 必须带 `/utf-8`（已配），否则 UTF-8 注释触发 C4819→C2220。

### 4.2 Keil 编译（产出 .hex）

```bash
cd stm32_firmware/MDK-ARM
"/c/Keil_v5/UV4/UV4.exe" -j0 -r stm32_firmware.uvprojx -o build.log
tail -3 build.log
```

期望结尾：`0 Error(s), 0 Warning(s)`。有 warning 不算通过——本工程保持 0 warning。

### 4.3 烧录（pyocd + CMSIS-DAP）

**前提**：板子上电、SWD 探针接好（SWDIO/SWCLK/GND/NRST）。

```bash
cd stm32_firmware/MDK-ARM
python -m pyocd flash --target stm32f407zgtx \
    --connect under-reset --erase=chip --frequency 1000000 \
    stm32_firmware/stm32_firmware.hex
```

若报 `Unexpected ACK '5'`：通常是板子没电、SWD 线松、或前次会话卡住探针。降速到 `--frequency 500000` 或 100000 试；再不行拔插探针 USB + 板子断电重上电。

### 4.4 一键流程（推荐）

改完代码 → 测试 → 编译 → 烧录：

```bash
cd stm32_firmware/tests && cmake --build build --config Release && \
(cd build && ctest -C Release) && \
cd /c/Users/Deng2/Desktop/temp/kuafu/stm32_firmware/MDK-ARM && \
"/c/Keil_v5/UV4/UV4.exe" -j0 -r stm32_firmware.uvprojx -o build.log && \
tail -3 build.log && \
python -m pyocd flash --target stm32f407zgtx --connect under-reset \
    --erase=chip --frequency 1000000 stm32_firmware/stm32_firmware.hex
```

---

## 5. SWD 实时调试（本项目最强大的工具）

板子**上电运行中**就能通过 SWD 读 RAM 里的任意全局变量，**不影响运行**。这是诊断运行时问题的主要手段。

### 5.1 现成脚本（`stm32_firmware/MDK-ARM/debug_tools/`）

| 脚本 | 用途 |
|------|------|
| `kuafu_balance_trace.py [秒]` | 20Hz 抓俯仰/yaw/轮速/力矩/x_error/x_int（默认 20s） |
| `kuafu_boot_trace.py [秒]` | 复位后 20Hz 抓启动序列（phase/mode/fault 跳变） |
| `kuafu_watch.py` | 5Hz 看门狗式实时显示关键状态 |
| `kuafu_fault_dump.py` | 读 HardFault dump（`0x2001FFF0`，magic `0xDEADF417`） |
| `kuafu_freeze_catch.py` | 抓挂起现场 |
| `read_health.py` / `read_imu_state.py` | 读执行器健康/IMU 状态 |

### 5.2 ⚠️ 符号地址会随构建变化

**每次改了全局变量（加/删/改结构体大小），`.map` 里的地址会移位**，调试脚本里的硬编码地址就读到垃圾。**改完全局后必须重新提取地址：**

```bash
python - <<'EOF'
import re
syms = ["g_mahony","g_lqr","g_ddsm_left","g_ddsm_right","g_startup_manager",
        "g_safety_state","g_body_gyro","g_ctrl_tau_l","g_ctrl_tau_r","g_servos",
        "uwTick","g_system_ticks"]
with open("stm32_firmware/MDK-ARM/stm32_firmware/stm32_firmware.map",encoding="utf-8",errors="ignore") as f:
    txt=f.read()
for s in syms:
    for line in txt.splitlines():
        if re.search(r"\b"+s+r"\b",line) and "0x200" in line:
            print(line[:110]); break
EOF
```

然后更新脚本里的 `ADDR` 字典（**本次基线的地址见下表，仅对当前 commit 有效**）：

| 符号 | 地址 | 含义 |
|------|------|------|
| `g_mahony` | `0x20000340` | pitch@+40, yaw@+44 |
| `g_lqr` | `0x20000370` | x_est@+20, x_ref@+24, x_int@+28 |
| `g_ddsm_left/right` | `0x200003A8 / 0x200003D0` | velocity_rads@+8, health@+20 |
| `g_servos[4]` | `0x200003F8` | 每元素 52B，health@+32 |
| `g_startup_manager` | `0x200004C8` | phase（0=INIT..4=READY,5=FAILED） |
| `g_safety_state` | `0x2000073C` | mode@+0, fault_mask@+12 |
| `uwTick` | `0x20000080` | HAL 毫秒计数器 |
| `g_system_ticks` | `0x20000008` | DRDY 1kHz 计数 |
| `g_body_pitch / g_body_pitch_rate` | `0x20000010 / 0x20000014` | DRDY 发布的最新姿态 |
| `g_pitch_rate_filt` | `0x2000000C` | 俯仰率一阶低通状态 |
| `g_ctrl_tau_l / g_ctrl_tau_r` | `0x20000018 / 0x2000001C` | 缓存的轮力矩命令 |
| `g_body_gyro` | `0x20000088` | float[3] |
| `g_actuator_configured` | `0x20000005` | 执行器配置完成标志 |
| `g_pi_cmd_heartbeat` | `0x2000085C` | mode@+0, vx@+4, wz@+8, d0@+12, last_hb@+16 |
| `g_loop_heartbeat_ms` | `0x20000020` | 主循环心跳（stall 取证） |

### 5.3 复位并抓启动 trace 的标准姿势

```bash
cd stm32_firmware/MDK-ARM/debug_tools
python -c "from pyocd.core.helpers import ConnectHelper as C; s=C.session_with_chosen_probe(options={'connect_mode':'attach','frequency':1000000}); t=s.target; t.reset(); t.resume(); s.close()"
python kuafu_balance_trace.py 60
```

---

## 6. 安全状态机（必须理解）

启动序列：`WAIT_POWER → IMU_DISCOVERY → GYRO_CALIBRATION → ACTUATOR_DISCOVERY → READY`

运行模式（`safety_state.h`）：`INIT → STAND → ACTIVE/CLIMB → FAULT`

**关键门控：**
- **轮子授权** `wheel_authorized = (phase==READY) && (mode != FAULT)`。未授权时 DDSM 力矩强制为 0。
- **陀螺标定静止门**：仅当三轴角速度都 < 0.08 rad/s 才累加样本（移动样本**跳过但不重置**），需 2000 个有效样本。**上电时机器人必须在地面静止 ~2s。**
- **FAULT 是锁存的**，唯一恢复方式是**整机断电重启**（启动失败 STARTUP_FAILED 同理锁存）。
- 新鲜度故障（IMU/轮/舵机反馈超龄）经 8 拍（32ms）去抖后锁存。当前阈值：轮 250ms、舵机 500ms、IMU 见 `pin_config.h`。

---

## 7. 当前状态（截至 2026-07-27）

### 已修复并验证 ✅
1. USART3 1Mbaud 中断风暴挂死 → RX 改 DMA 循环。
2. HAL 在 ORE 时关 DMA 不重臂 → 主循环自动重臂（USART3+USART6）。
3. 断链每拍 `lqr_reset` 废位置环 → 断链只锚定一次。
4. 架空轮速积分成 x_error 致落地冲撞 → 架空重锚（持续 250ms 高速+水平才触发）。
5. yaw 固定参考 + Mahony 无磁偏漂移致原地转 → 无转向指令时 yaw_ref 跟随实测。
6. 启动失败（标定无静止门 + I2C 长超时）→ 静止门 + 2ms 超时。
7. 舵机误锁（50Hz 同步写与 5ms 轮询相位锁死 → S1 吃所有延迟回复）→ 回复超时 3→6ms、新鲜度 250→500ms、查询周期 5→6ms 解锁相。
8. LQR 控制截止期丢失 → `lqr_deadline_pending` 锁存。
9. 健壮性：栈 1KB→4KB、IWDG 8.2s→1s、Mahony eInt 限幅、fusion_dt 钳位、SysTick 最高优先级。
10. RX DMA 中止竞态永久失聪（ORE 后 HAL 异步关 DMA，主循环重臂被迟到的中止完成擦掉，EIE=0 再无回调）→ 重臂前同步 `HAL_UART_AbortReceive` + 清 ORE + DMA EN=0 失聪看门狗强制重臂（USART3+USART6）。
11. 噪声字节误杀整条 DMA（HAL 对 NE/FE 也走中止路径，单次噪声≈1ms 盲区毁掉全部在途帧，舵机轮询失败率 ~25% → 新鲜度 FAULT）→ USART3 ISR 预吞 NE/FE（读 SR+DR 清标志），parser 自行按 0xFFFF+校验重同步；ORE 仍走中止-重臂。修复后站立 57s+ 无故障，舵机错误率 ~25%→<4%。
12. **DDSM315 模式帧误诊（之前 commit 把电机留在速度环）**：原 commit 把 mode 写到 `byte[2]` 并补 CRC（DDSM210 风格），但 DDSM315 协议 3 要求 mode 值在 `byte[9]` 且**无 CRC**（官方 `ddsm_ctrl.cpp` 例程同此布局）。SWD 实测两轮 `mode=2`（速度环），LQR 的"力矩"被当成速度设定值下发，是当年"爆冲/平衡不了"的根因之一。改回协议 3 格式（`byte[9]=mode` 无 CRC），发现序列改为 **enable → mode → torque**，并在使能后对每台轮补发一次电流环模式帧；SWD 实测两轮 `mode=1`（电流环）确认切换生效。注意：DDSM315 wiki 自身前后矛盾（散文说 byte[9] 无 CRC，串口示例却写 byte[2]+CRC），以行为实测为准。
13. **无回复帧误判健康超时**：`mode`/`enable`（0xA0）帧电机不回复，总线每发一次就记一次 timeout；多个 12ms 无回复窗口叠加越过 32ms 新鲜度门槛，锁存**虚假 WHEEL_R FAULT 把轮力矩清零**——这正是"平衡不了"的症状。总线新增 `expect_reply` 标志，无回复命令超时不惩罚健康。修复后 SWD 实测 `safety=STAND`、`fault=0x0`，两轮均在线、超时计数低。
14. **舵机门控放宽**：`SAFETY_SERVO_MAX_AGE_MS` 500→1000ms、`ST_REPLY_TIMEOUT_MS` 6→10ms。舵机靠内部位置环自保持，遥测丢失≠失控；放宽后既保留健康监控又避免地面振动引发的 query-timeout 突刺锁存 `FAULT_SERVO`（S3 最严重）。

### 未解决 / 待验证 ⚠️
- **冷启动地面站立**：修复 #10/#11/#13/#14 后 SWD 实测无 FAULT、两轮在线、电流环生效；需连跑 3-5 次确认复现性（接 Pi 做真实平衡）。
- **抖动极限环**：俯仰 ±2.3°、~1.9Hz，空中地面都有。原假设是速度环滞后所致——现已确认是**电流环**，若极限环仍在则是 LQR 增益问题（俯仰率滤波现 35Hz、K3、力矩死带可调），需接 Pi 实地验证。
- **LQR 增益是否需重调**：先前增益是在"误诊速度环"状态下整定的；当前已确认真电流环，被控对象与文档/StackForce 参考架构一致，理论上应更稳。若平衡行为变化，从保守增益起步重整定。
- **架空→落地过渡**：缓存的饱和 `x_int` + 轮速突变会触发爆冲。**这是测试场景的副产物，不是产品需求**（产品是"上电就站在地上"）。
- **SWD 探针偶发掉线**（`Unexpected ACK '0'`）：长 trace 会中途断；降速（200kHz）更稳。
- Git 仓库：`main` 跟踪 `origin/main`（github.com/Weilv-D/kuafu）。

### 已知约束
- 无磁力计 → Mahony yaw 长期漂移，无法做绝对航向保持，只能做角速度阻尼。
- CPU MuJoCo harness 的轮接触解算有伪影，速度环仿真不可信（pitch 环可参考）。
- 地面测试是破坏性的（撞、浪涌欠压重启、机械冲击），**不能高频裸地迭代**。

---

## 8. 开发方法论（血泪经验）

### 有效 ✅
- **SWD 实时读 RAM** —— 最大功臣。舵机总线静默、DMA 被关、S1 超时聚集 全靠它发现。
- **boot/balance trace** —— 逐拍捕捉瞬态，统计均值/标准差/频率。
- **RAM-top 前沿取证** —— IWDG 复位原因、stall dump、HardFault dump，跨复位存活。
- **主机单元测试** —— 防回归（但测不了动力学）。

### 失败 ❌
- **裸地裸测 = 破坏性**，不能高频迭代。
- **空中测不到地面动力学**（接触/摩擦/差速），"空中调好"对地面无保证。
- **靠用户肉眼描述猜** —— "疯狂后退/逆时针旋转"里，物理旋转 vs 姿态解算漂移分不清，反复猜错。

### 给后续 Agent 的建议
1. **固化"上电在地面上"为标准流程**，别为"架空→放下"伪需求牺牲迭代。
2. **地面测试时 SWD 线拖着跑**，失败瞬间也有数据。
3. **优先复现 `ground6` 的 90s 干净站立**（连跑 3-5 次），稳定后再调抖动。
4. **改全局变量后立刻重提取 .map 地址并更新调试脚本**，否则读垃圾数据误导。

---

## 9. 联系点

- 用户 Deng2 是机器人拥有者，负责物理操作（上电、放置、观察）。
- 开发模式：用户描述现象 → 你用 SWD 抓数据定位根因 → 改代码 → 测试/编译/烧录 → 用户实地验证。**别只凭描述猜方向。**
- 物理/机械问题（电机力矩常数不一致、线缆拖拽、轮打滑）固件无法根治，需用户排查机械侧。

---

**快速入门**：读完本文档 → 跑 §4.1 主机测试确认环境 → 跑 §4.2 编译确认工程可用 → `git status`（若有）看未提交改动 → 读 `AGENTS.md` 和 `docs/KUAFU.md` 补充背景 → 按用户当前需求开工。
