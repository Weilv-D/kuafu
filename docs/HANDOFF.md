# KUAFU 项目交接文档（给新 Agent）

> 最后更新：2026-09-20。本文档面向**完全不了解本项目的工程师/Agent**，读完即可开始开发。

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

> **环境**：Windows + Git Bash。所有命令在仓库根 `C:\WORKSPACE\temp\kuafu` 下执行。

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
cd /c/WORKSPACE/temp/kuafu/stm32_firmware/MDK-ARM && \
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

然后更新脚本里的 `ADDR` 字典（**本次基线的地址见下表，仅对当前 commit 有效**；2026-09-20 逻辑审查后已对照新构建复核，下表全部有效）：

| 符号 | 地址 | 含义 |
|------|------|------|
| `g_wheel_mode_sent` | `0x20000000` | 电流环模式补发掩码（bit0=左,bit1=右） |
| `g_wheel_output_gate` | `0x20000001` | ★ 轮输出最终授权（250Hz 判决，持久到下个期限） |
| `g_actuator_configured` | `0x20000008` | 执行器配置完成标志 |
| `g_system_ticks` | `0x2000000C` | DRDY 1kHz 计数 |
| `g_pitch_rate_filt` | `0x20000010` | 俯仰率一阶低通状态 |
| `g_body_pitch / g_body_pitch_rate` | `0x20000014 / 0x20000018` | DRDY 发布的最新姿态 |
| `g_ctrl_tau_l / g_ctrl_tau_r` | `0x2000001C / 0x20000020` | 缓存的轮力矩命令 |
| `g_loop_heartbeat_ms` | `0x20000034` | 主循环心跳（stall 取证，EXTI1 观察） |
| `g_st3215_ring` | `0x20000324` | servo RX 环：laps@+8, produced@+12, consumed@+16, overrun@+20 |
| `g_imu` | `0x2000033C` | accel@+4, gyro@+16, temp@+28 |
| `g_mahony` | `0x200003CC` | pitch@+40, yaw@+44 |
| `g_ddsm_left/right` | `0x20000404 / 0x2000042C` | velocity_rads@+8, health@+20 |
| `g_servos[4]` | `0x20000454` | 每元素 52B，health@+32 |
| `g_startup_manager` | `0x20000524` | phase（0=INIT..4=READY,5=FAILED） |
| `g_control_section` | `0x20000704` | ★ 控制节状态：lqr@+20（x_est@+40,x_ref@+44,x_int@+48）、supervisor.phase@+76（AC6 短枚举） |
| `g_pi_transport` | `0x20000858` | Pi RX 环：read_idx@+6, laps@+8, produced@+12, consumed@+16, overrun@+20 |
| `g_safety_state` | `0x20000888` | mode@+0, fault_mask@+12 |
| `g_pi_cmd_heartbeat` | `0x200009A8` | mode@+0, vx@+4, wz@+8, d0@+12, last_hb@+16 |
| `g_body_gyro` | `0x200000A8` | float[3] |
| `uwTick` | `0x200000A0` | HAL 毫秒计数器 |

### 5.3 复位并抓启动 trace 的标准姿势

```bash
cd stm32_firmware/MDK-ARM/debug_tools
python -c "from pyocd.core.helpers import ConnectHelper as C; s=C.session_with_chosen_probe(options={'connect_mode':'attach','frequency':1000000}); t=s.target; t.reset(); t.resume(); s.close()"
python kuafu_balance_trace.py 60
```

---

## 6. 安全状态机（必须理解）

启动序列：`WAIT_POWER → IMU_DISCOVERY → ACTUATOR_DISCOVERY → READY`（旧的陀螺标定阶段已退役，枚举数值 2 保留空位以稳定 trace 编码——陀螺零偏标定**不阻断启动**，在后台累积，见 §7.1）

运行模式（`safety_state.h`）：`INIT → STAND → ACTIVE/CLIMB → FAULT`

**关键门控：**
- **轮子授权** `wheel_authorized = (phase==READY) && (mode 为运行态 STAND/ACTIVE/CLIMB) && (mode != FAULT)`。INIT 期间轮子只收只读查询帧；未授权时 DDSM 力矩强制为 0。最终输出还叠加监督器判决与电流环补发完成（`g_wheel_output_gate`，见 §7.5）。**派发策略**：门开启→LQR 力矩帧；门关闭且轮仍使能（任何模式，含 FAULT 与监督器降级）→显式零力矩帧（电流环会锁存最后设定值，必须主动清零）；门关闭且轮未使能→只读查询帧。三类帧都有状态回复，反馈新鲜度在所有分支下等价。
- **陀螺标定静止门**：仅当三轴角速度都 < 0.08 rad/s 才累加样本（移动样本**跳过但不重置**），窗口累计 1000 个有效样本（~1s）并通过方差门。**不阻断启动**——上电无需保持静止。
- **FAULT 是锁存的**，唯一恢复方式是**整机断电重启**（启动失败 STARTUP_FAILED 同理锁存）。例外：**轮/舵机新鲜度这类瞬态故障可自动恢复**（见 §7.2）。
- 新鲜度故障（IMU/轮/舵机反馈超龄）经 8 拍（32ms）去抖后锁存。当前阈值：轮 250ms、舵机 1000ms、IMU 20ms（`pin_config.h`）。去抖另受 100ms 迁移宽限窗抑制，但**只有设备驱动的迁移**（INIT→STAND、FAULT→STAND 恢复）发放宽限窗并清零计数器——命令驱动的迁移（模式请求、断链降级）不清零，Pi 抖动模式请求无法抑制故障（2026-09-16 修复）。

---

## 7. 当前状态（截至 2026-09-20）

### 7.0 逻辑审查记录（最新在前）

#### 2026-09-20 第二次审查（独立第四轮）

在 09-20 三轮基线之上，从源码独立重推了一遍架构与全部模块（先审代码、后读历史记录核对不变量），修复 1 项缺陷（中高）与 1 处过时注释，并把 3 项设计决策正式写入文档；完整清单与证据见 `docs/validation/stm32-firmware-2026-09-20-2.md`。对新接手者行为相关的：

1. **门关闭时不再残留电流环力矩（中高严重度）**：派发层此前只在 FAULT 发零力矩帧，其它门关闭状态发只读查询——而 DDSM315 电流环会**保持最后一个力矩设定值**。监督器把机器人降级 HOLDING 时模式仍是 STAND/ACTIVE（典型诱因：舵机总线拥塞使 100ms 腿保持窗过期），轮子仍使能、平衡已停摆，电机会带着关闭前的 LQR 力矩主动驱动机器人，直到 45° 倾角故障兜底（机器人实际已倒）。现在只要**门关闭且轮仍使能**就发显式零力矩帧（力矩帧与查询帧同样有回复，反馈新鲜度不受影响）；查询帧仅用于轮未使能的场景（INIT/发现期/禁用排空窗）。
2. **过时注释修正**：`SERVO_FAIL_LIMIT` 注释声称的"连续失败→致命 FAULT 锁定"路径早已不存在，实际语义是轮询的 offline_after（连续 3 次失败置离线），故障锁存归安全层新鲜度路径。
3. **三项设计决策文档化（不改码）**：IMU 通路失效期间遥测完全静默（FAULT 帧也发不出，只能 SWD 取证）；DDSM 反馈模式字节仅作诊断、不做运行时校验（单电机 spontaneous 掉电无现场证据，按证据驱动方法论不新增故障源）；LQR 位置估计在门关闭期间冻结（elapsed-dt 钳制最多补积一步 20ms，100ms 降级最多引入 ~4cm x_est 偏差——不值得为二阶误差重构现场验证过的控制器）。

主机测试全过（含零力矩派发新回归）；Keil 0 错误 0 警告；SWD 符号地址对照新 map 全部不变。

#### 2026-09-20 审查（09-19/20，三轮）

在 09-16 基线之上对固件全模块做了两轮审查（第二轮含对第一轮修复的独立复核：回声滤波字节级追踪、FAULT 下总线队列优先级、UB 审计、NVIC 优先级审计）与一轮遗留收尾，共修复 10 项缺陷（1 高、5 中、4 低）并全部验证；完整清单与证据见 `docs/validation/stm32-firmware-2026-09-20.md`。对新接手者行为相关的五条：

1. **瞬态故障恢复路径已修复（高严重度）**：舵机使能序列器此前不分模式地重跑——FAULT 期间它一边与 50Hz 一次性禁用互相打架（~80ms 使能/禁用抖动），一边把 `servos_enabled` 重新置 1；瞬态故障（轮/舵机新鲜度，可自动恢复）恢复后，监督器在这段"竞态 0→1"上清除重发阶段，**轮门在物理失力的腿上重新打开**且永远不会有物理重使能。现在序列器带 `mode != FAULT` 门控：FAULT 期间禁用独占总线、软件标志保持低（正是监督器重发阶段要求的真实"先降后升"），恢复后才物理重发。新增控制节回归测试钉死该契约。
2. **总线超时/完成事件自愈**：`ddsm_bus_step` 的超时判决于总线中断掩码下重验 phase 与 deadline，补上并发契约唯一未覆盖的主循环侧相位迁移（过期判决可击杀 ISR 刚启动的新事务）；ST3215 写帧（sync-write/torque）自带 20ms 截止期，丢失 TX 完成中断时经显式 `AbortTransmit` 复位 HAL 发送状态并重开总线——一次丢帧不再拖垮整个舵机子系统。
3. **舵机新鲜度要求真实首帧**：删除引导期"乐观 online"标志——此前从未回帧的舵机在上电后 1s 内显得 fresh，启动时序（~800ms）允许轮门在腿部反馈从未见过的情况下短暂打开（~200ms）。
4. **校准镜像的闩锁已修正（仅 bench 构建）**：两个校准构建此前用**瞬态**故障位（`FAULT_SERVO` / `FAULT_WHEEL_*`）做"禁止动作"闩锁——瞬态故障在设备轮询恢复新鲜后数秒内自动 FAULT→STAND，重新武装轮授权/舵机使能路径，与镜像的"全程无力矩"契约相悖。现附加 `FAULT_INIT`（严重位，锁存到断电），瞬态位保留用于诊断遥测；生产镜像不受影响（两宏默认 0）。
5. **调度器结构加固**：1kHz 融合块的深层 `continue` 改为显式 `if (g_imu.initialized)` 守卫（此前会静默跳过其后所有调度阶段，属重排陷阱）；`pi_link` 三处临界区统一为可嵌套 PRIMASK 规范；`bmi088_read_temp` 补句柄防护；`queued_mode` 不再把使能帧的 CRC 记为控制模式；退役的 `STARTUP_GYRO_CALIBRATION` 枚举成员已删除（数值空位保留以稳定 trace 编码）。

Keil 0 错误 0 警告；SWD 符号地址对照新 map 全部不变。保留的设计决策（见验证记录）：轮新鲜度 32ms 去抖窗口内轮门按设计保持开启；`leg_posture_safe` 为俯仰角代理而非腿部几何直测。

#### 2026-09-16 审查

在 09-13 基线之上二次全库审查（固件 + Pi5 运行时），修复 11 项缺陷并全部验证；完整清单与证据见 `docs/validation/stm32-firmware-2026-09-16.md`。对新接手者行为相关的五条：

1. **宽限窗抑制孔洞已封堵**：此前**所有**模式迁移都会重置四个新鲜度陈旧计数器并重开 100ms 宽限窗——Pi 以 >10Hz 抖动 `mode_request` 即可永久抑制轮/舵机失联故障。现在只有设备驱动的迁移（INIT→STAND、FAULT→STAND 恢复）享有宽限与清零；命令驱动迁移（STAND↔ACTIVE/CLIMB、断链降级）只记时间。
2. **50Hz 舵机写改为忙重试**：总线忙不再丢弃整个 20ms 腿部写周期（此前连续 5 次丢失会让 `LEG_HOLD_MAX_AGE_MS` 过期、监督器降级、轮输出门关闭 ~100ms——自平衡机器人即跌倒风险；单舵机劣化时总线忙占比可达 ~50%，该场景会反复出现）。`servo_intent_allowed` 语义随之改为纯模式裁决（与轮意图对称），总线仲裁归队列层。
3. **Pi 端 D0 动态门控镜像**：`teleop_single` 此前在腿抬高 + 快推杆（|v|>0.3 或 |w|>0.6 且 d0>120mm）时每拍触发 `ValueError`，**心跳整段静默**，200ms 看门狗恰在激烈机动中降级机器人。现已在编码前镜像 120mm 门控，且 except 路径回退为安全心跳（STAND/零指令），任何坏值都不再断流。
4. **协议解码器支持对端重启重同步**：STM32 复位后 TX 序列从 0 重来，Pi 侧单调序列门会把所有 CRC 合法帧黑洞数分钟。现在连续 8 帧被序列门拒绝即判定对端重启并重同步（孤立重复帧仍被丢弃）。
5. **遥测过期不再静默**：`serial_node` 在遥测过期时改为显式发 STAND/零指令心跳（保持链路契约），恢复后重置观测历史；ESTOP 退出路径补 `flush()`；武装状态下的阻塞 BLE 重连改为先强制安全态。

另：link_probe 的 `FAULT_BITS` 表整体错位 1 位（如 SERVO 故障被标成 WHEEL_L_LOST）已对齐固件定义，并补 FAULT 帧计数/打印、`--mode` 校验、HEALTH 长度守卫与 deadline 钳位；诊断遥测负温度 uint8 铸型饱和化；死字段 `gyro_calibrated`（三处输入结构体，从未被消费）移除。Keil 0 错误 0 警告；SWD 符号表复核后地址不变。

#### 2026-09-13 审查

全库深度审查（架构 + 逐模块细节）修复了 6 项缺陷，全部有回归测试覆盖并通过红绿验证；完整清单见 `docs/validation/stm32-firmware-2026-09-13.md`。对新接手者行为相关的三条：

1. **上电死锁已修复**：监督器 ops-less 使能重发阶段此前在"干净上电"（从未经过故障周期）时永远无法清除 `enable_reissue_needed`——因为调度器把 `startup_ready` 与 `servo_enable_verified` 耦合为同一变量，二者同拍翻 1，而早期返回路径从不记录"verified 观察到低"。轮输出门在冷启动时永不打开。修复：每步锁存低电平观察；撤销后的重挂（drop-requirement）仅在从 READY 被降级时重新武装。
2. **INIT 模式不再物理使能轮子**：`wheel_authorized` 额外要求运行模式，对齐模式表（INIT = 只读查询）。此前 startup READY 后、安全模式仍为 INIT 的 ~20ms 窗口里轮子已使能。
3. **LQR 估计授权叠加 imu_fresh**：DRDY 中断线死亡时 `g_imu_control_valid` 冻结为 1，旧代码在 ~52ms 去抖故障锁存前一直按冻结姿态出力矩；现在 20ms 新鲜度界即撤销。

另：st3215 UART 错误处理从 ISR 延迟到主循环（消除对 target/phase/解析器/健康计数的并发修改）；RAM 顶部 64B 取证区从链接器布局排除（IRAM2 上界 0x2001FFBF）；移除 `firmware_runtime.clear_motion` 死代码。

### 7.1 运行模式与启动

固件可**独立自平衡**：上电后无需树莓派即进入 `STAND` 原地平衡。树莓派只负责 `ACTIVE` 模式下的运动指令，不参与平衡环本身。

启动状态机为 `WAIT_POWER → IMU_DISCOVERY → ACTUATOR_DISCOVERY → READY`。各阶段在条件未满足时**持续重试**，不会因上电瞬间的扰动而锁死。**陀螺零偏标定不再阻断启动**：IMU 初始化后即进入执行器发现，平衡环随即启动；零偏在背景中持续累积，机器人稳定后自动修正。姿态由 Mahony 的加速度计参考融合得出，因此零偏未标定时也能安全平衡（放地上即可自平衡，无需预先保持静止）。

### 7.2 安全故障模型

安全状态机：`INIT → STAND → ACTIVE/CLIMB → FAULT`。

- **硬故障（永久锁存，需复位）**：TILT、俯仰率超限、过温、IMU 失联、急停、初始化失败、内部错误。
- **新鲜度故障（自动清除）**：轮电机 / 舵机的遥测暂时丢失。这类故障在设备恢复回复后**自动解除**，`FAULT` 期间仍持续轮询，恢复即回到 `STAND`。这避免了一次总线瞬断就永久禁用平衡。

### 7.3 轮电机驱动（DDSM315，电流环）

轮电机运行在**电流（力矩）环**（`mode=0x01`），与 `kuafu_physics.py` 的力矩模型以及 LQR 设计一致。

- 模式帧采用协议 3 格式：`byte[9]` 写入模式值且**无 CRC**（官方例程同此；DDSM315 wiki 自身前后矛盾，格式以 SWD 实测为准）。
- 发现序列为 enable → mode → torque，使能后再对每台轮补发一次电流环模式帧。
- `mode` / `enable`（0xA0）帧电机不回复，总线以 `expect_reply` 标志区分，其超时**不计入**健康统计，避免多个无回复窗口叠加误锁轮故障。

### 7.4 舵机门控（ST3215）

门控阈值：`SAFETY_SERVO_MAX_AGE_MS = 1000 ms`、`ST_REPLY_TIMEOUT_MS = 10 ms`。舵机靠内部位置环自保持，遥测丢失不等于失控，放宽门限可避免地面振动引起的误锁。

### 7.5 控制与总线实现要点（改动时勿破坏）

- **LQR 断链只锚定一次**，不每拍重置位置环；架空时重锚，落地不冲撞。
- **无转向指令时 yaw 参考跟随实测**，避免 Mahony 无磁偏漂移导致的原地旋转。
- **USART 噪声字节（NE/FE）由 ISR 预吞**（读 SR+DR 清标志），否则单次噪声会误杀整条 DMA；ORE 仍走中止-重臂路径，并有看门狗强制重臂防止永久失聪。
- **250Hz 控制节**在墙上时钟期限执行（与 DRDY 无关），一次期限跑完一条线性序列：安全状态机 → 监督器判决 → `g_wheel_output_gate` 门控 → LQR → trace，随后才是总线派发。门控判决持久到下个期限，LQR 与派发同守一门；DRDY 停摆只冻结遥测，不冻结故障检测与取证。
- **50Hz 舵机期限是"忙重试"而非"忙丢弃"**：sync-write/FAULT 禁使能帧被总线拒绝时保留 pending，下个调度遍重试（与轮派发的 `next_wheel_tx_ms` 同型）。这是 `LEG_HOLD_MAX_AGE_MS`（100ms）腿保持窗口的可靠性前提——连续 5 个丢失周期即关闭轮输出门。
- **模式化指令契约**：`vx/wz` 仅在 ACTIVE 生效（STAND 是位置保持，CLIMB 只驱动腿高 `D0`）；residual 还要求链路新鲜。固件不信任发送方把非本模式字段清零。
- **RX 环 overrun 计数**：servo/Pi 两环以"圈数×容量+位置"重构生产者（TC 中断为圈数真源，圈数先于 NDTR 读取），消费滞后一整圈时精确计数丢弃字节；重臂路径重置消费端。SWD 可读 `g_st3215_ring`/`g_pi_transport` 的 overrun 字段。

### 7.6 已知约束与待验证

- **无 Pi 自平衡已确认闭环活跃**（SWD：`STAND`、`fault=0`、LQR 出扭矩、两轮电流环、电机反馈力矩与指令一致）。实地站立姿态待确认：当前稳态俯仰约 +0.08 rad（轻微前倾），疑为腿零位 / 俯仰参考偏差，非失控。
- **抖动极限环**：俯仰 ±2.3°、~1.9 Hz，空中地面均存在。已确认是电流环行为；若仍存在则属 LQR 增益问题（俯仰率滤波 35 Hz、K3、力矩死带可调），需接 Pi 实地验证。LQR 增益先前在误诊的速度环状态下整定，现被控对象已一致，可观察平衡行为是否改善。
- **架空→落地过渡**：缓存的饱和 `x_int` 加轮速突变会触发爆冲，属测试场景副产物，非产品需求（产品是"上电就站在地上"）。
- **左轮 RS485 通信最易掉线**（timeout 计数约为右轮 2.4×）；虽不再致命，仍建议检查左轮线缆。
- **SWD 探针偶发掉线**（`Unexpected ACK '0'`）：长 trace 会中途断，降速至 200 kHz 更稳。
- BMI088 加计配置 `ACC_CONF=0xAC` 已按 Bosch 官方 BMI08x_SensorAPI 位域核实：高半字节带宽 NORMAL、低半字节 ODR 1600Hz，均合法；1600Hz 采样覆盖 1kHz 轮询，每通道新鲜度时间戳语义成立（来源见 `bmi088.c` 注释）。
- **IMU 通路失效期间遥测完全静默**：遥测槽在 `g_imu.initialized` + DRDY tick 守卫内，I2C 失效或 DRDY 停摆时 Pi 收不到任何帧（含 FAULT/健康帧）；此时 250Hz 控制节仍按墙钟检测并锁存故障，但只能靠 SWD 取证。
- **DDSM 反馈模式字节仅作诊断**：固件不运行时校验反馈帧里的模式字节（若电机自发掉电回复默认速度环，恢复路径是倾角故障而非模式校验）；`ddsm_bus_mode_feedback()` 供 SWD/诊断读取。
- **LQR 位置估计在门关闭期间冻结**：`last_lqr_ms` 只在实际运行时推进，重开后 elapsed-dt 钳制（≤20ms）最多补积一步；100ms 级降级引入的 x_est 偏差上限 ~4cm，对 K0 项 ≈0.04Nm，在可接受范围内。
- 无磁力计 → Mahony yaw 长期漂移，只能做角速度阻尼，无法绝对航向保持。
- CPU MuJoCo harness 的轮接触解算有伪影，速度环仿真不可信（pitch 环可参考）。
- 地面测试是破坏性的（撞、浪涌欠压重启、机械冲击），不能高频裸地迭代。
- 仓库 `main` 跟踪 `origin/main`（`github.com/Weilv-D/kuafu`）。

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
