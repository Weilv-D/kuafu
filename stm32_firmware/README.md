# 夸父 KuaFu — STM32F407ZGT6 固件说明书 (Cerebellum Firmware)

本文件夹包含了夸父（KuaFu）桌面双轮足平衡机器人的 **STM32F407ZGT6 主控固件代码**。固件基于 STM32 HAL 库编写，负责处理机器人底层的硬实时任务：IMU 姿态解算（1 kHz）、双轮 LQR 自平衡算法（250 Hz）、髋关节舵机同步控制（50 Hz）以及与上层树莓派 5 的桥接通信。

---

## 目录结构

*   `Config/`
    *   [pin_config.h](Config/pin_config.h): 统一的引脚映射、LQR 控制增益参数、机器人并联机构的物理参数和安全阈值定义。
*   `Comm/`
    *   [crc8.h](Comm/crc8.h) / [crc8.c](Comm/crc8.c): 用于 DDSM315 电机校验的查表法 CRC-8/MAXIM 计算实现。
    *   [pi_link.h](Comm/pi_link.h) / [pi_link.c](Comm/pi_link.c): 树莓派 5 自定义桥接串口协议（波特率 921600），包含遥控/动作帧解析与传感器遥测上传。
*   `Drivers/`
    *   [bmi088.h](Drivers/bmi088.h) / [bmi088.c](Drivers/bmi088.c): BMI088 六轴 IMU 的 I2C 驱动，支持陀螺仪 DRDY（INT3）中断同步。
    *   [ddsm315.h](Drivers/ddsm315.h) / [ddsm315.c](Drivers/ddsm315.c): DDSM315 轮毂电机 RS485 半双工驱动（使能、模式设置、转矩/速度控制与反馈解析）。
    *   [st3215.h](Drivers/st3215.h) / [st3215.c](Drivers/st3215.c): ST3215 菊花链总线舵机 1 Mbps 串口驱动，支持多舵机同步位置下发 `SyncWrite`。
*   `Control/`
    *   [mahony.h](Control/mahony.h) / [mahony.c](Control/mahony.c): Mahony 互补滤波姿态估计算法。
    *   [kinematics.h](Control/kinematics.h) / [kinematics.c](Control/kinematics.c): 并联五杆机构逆运动学（IK）解算器，将腿长映射为舵机目标角度。
    *   [lqr_controller.h](Control/lqr_controller.h) / [lqr_controller.c](Control/lqr_controller.c): 双轮倒立摆 LQR 平衡控制器，并融合上层输出的残差扭矩。
*   `Core/`
    *   [safety_state.h](Core/safety_state.h) / [safety_state.c](Core/safety_state.c): 状态机（INIT、STAND、ACTIVE、CLIMB、FAULT）与多重安全保护逻辑。
    *   [main.c](Core/main.c): 主程序入口。配置系统时钟（168 MHz），实现 1 kHz 中断同步计数器及主循环 Slot 分时调度器。

---

## 实时时序设计

为防止低速半双工通信（如 115200 bps 的 RS485）阻塞高频姿态解算，系统采用了**前后台分时异步时序**：

1.  **1 kHz 传感器脉冲 (前台中断)**：
    *   IMU 陀螺仪的 Data Ready (INT3 引脚) 连接至 STM32 的 **PB1**，触发 EXTI1 中断。
    *   中断服务函数仅累加全局滴答计数器 `g_system_ticks`，将所有阻塞操作移出中断。
2.  **主循环分时调度器 (后台 ticks 触发)**：
    *   主循环检测到系统滴答计数器更新时，首先高频执行 I2C 读取 BMI088 与 Mahony 姿态滤波。
    *   随后将 4 ms（250 Hz）的电机控制周期划分为 **4 个 1 ms Slot（时间槽）**：
        *   **Slot 0** (0-1ms): 下发左轮扭矩指令 ──► 阻塞式读取左轮 DDSM 反馈数据（限时 1.5ms）。
        *   **Slot 1** (1-2ms): 下发右轮扭矩指令 ──► 阻塞式读取右轮 DDSM 反馈数据（限时 1.5ms）。
        *   **Slot 2** (2-3ms): 通过 USART6 将当前 IMU 姿态与关节状态打包上传至树莓派。
        *   **Slot 3** (3-4ms): 计算 LQR 自平衡输出，叠加树莓派下发的残差，并进行系统级安全诊断。
3.  **50 Hz 髋关节舵机环**：
    *   在大循环中每 20ms 定时执行。通过 1 Mbps 串口 `SyncWrite` 四腿髋关节角度，并轮询单台舵机状态。

---

## 全系统针脚连接方案

> 主控芯片：STM32F407ZGT6（实物丝印省略 "P"，如 PB8 标为 B8）

### 1. 串口资源分配总览

| 串口外设 | 功能用途 | STM32 引脚 (丝印) | 波特率 |
| :--- | :--- | :--- | :--- |
| **USART1** | CH340 串口下载 / PC 调试打印 | PA9 (TXD) / PA10 (RXD) | 115200 bps |
| **USART2** | RS485 模块 → DDSM315 电机总线 | A2 (TX) / A3 (RX) | 115200 bps |
| **USART3** | 舵机控制板 → ST3215 总线 | B10 (TX) / B11 (RX) | 1000000 bps |
| **USART6** | 树莓派 5 主控桥接串口 | C6 (TX) / C7 (RX) | 921600 bps |
| **I2C1** | BMI088 IMU 姿态传感器 | B8 (SCL) / B9 (SDA) | 400 kHz |

---

### 2. 全系统针脚连接表

#### A. CH340 串口下载器 (程序烧录 + 调试)

| CH340 模块引脚 | 信号方向 | STM32 引脚 | 实物丝印 | 说明 |
| :--- | :---: | :--- | :---: | :--- |
| 3.3V | ──► | 3.3V 供电 | **3V3** | 仅调测供电；若已接电池则**不接此线** |
| TXD | ──► | PA10 (USART1_RX) | **RXD** | 电脑发送 → 单片机接收 |
| RXD | ◄── | PA9 (USART1_TX) | **TXD** | 单片机发送 → 电脑接收 |
| GND | ─── | GND | **GND** | 必须共地 |
| rst | ✗ | 悬空不接 | — | 悬空 |

**⚠️ 串口下载步骤：**
1. 将板子上 BOOT0 拨码/跳线拨到 **1 (接 3.3V)**；
2. 在 FlyMcu 选择对应 COM 口并载入 `.hex` 固件；
3. 手动按板载 **RST 复位键** 即可自动开始烧录；
4. 烧录完成后将 BOOT0 拨回 **0 (接 GND)**，按复位键即可正常运行。

#### B. 树莓派 5 ↔ STM32 (主控桥接通信)

| 树莓派 5 专用 UART 口 (JST-SH 1.0mm) | 信号方向 | STM32 引脚 | 实物丝印 | 说明 |
| :--- | :---: | :--- | :---: | :--- |
| Pin 1 (TX) | ──► | PC7 (USART6_RX) | **C7** | 树莓派发送 → STM32 接收 |
| Pin 2 (GND) | ─── | GND | **GND** | 信号共地 |
| Pin 3 (RX) | ◄── | PC6 (USART6_TX) | **C6** | STM32 发送 → 树莓派接收 |

- 电平：3.3V TTL 直连；
- 树莓派端串口节点：`/dev/ttyAMA0`（需通过 `raspi-config` 关闭控制台占用）。

#### C. 舵机控制板 ↔ STM32 (ST3215 髋关节舵机)

| 舵机控制板引脚 | 信号方向 | STM32 引脚 | 实物丝印 | 说明 |
| :--- | :---: | :--- | :---: | :--- |
| RXD | ──► | PB10 (USART3_TX) | **B10** | STM32 发送 → 控制板接收 |
| TXD | ◄── | PB11 (USART3_RX) | **B11** | 控制板发送 → STM32 接收 |
| GND | ─── | GND | **GND** | 信号共地 |

- 通信：1 Mbps，半双工 TTL 模式；
- 控制板逻辑端采用自供电，**切勿接入 STM32 的 3.3V**。动力电源单独接 12V Buck 降压输出。

#### D. RS485 模块 ↔ STM32 (DDSM315 电机)

| RS485 模块引脚 | 信号方向 | STM32 引脚 | 实物丝印 | 说明 |
| :--- | :---: | :--- | :---: | :--- |
| RX | ──► | PA2 (USART2_TX) | **A2** | STM32 发送 → 模块接收 |
| TX | ◄── | PA3 (USART2_RX) | **A3** | 模块发送 → STM32 接收 |
| V (VCC) | ──► | 3.3V | **3V3** | 模块逻辑供电 |
| GND | ─── | GND | **GND** | 信号共地 |

- RS485 转换模块必须为**带自动收发流控**的版本（无需方向控制引脚）；
- 485 差分总线侧（A、B 线）并联两台 DDSM315 电机（左轮 ID=1，右轮 ID=2）。

#### E. BMI088 IMU ↔ STM32 (I2C 模式)

| BMI088 模块引脚 | 信号方向 | STM32 引脚 | 实物丝印 | 说明 |
| :--- | :---: | :--- | :---: | :--- |
| VCC | ──► | 3.3V | **3V3** | IMU 供电 |
| GND | ─── | GND | **GND** | 信号共地 |
| SCL | ──► | PB8 (I2C1_SCL) | **B8** | I2C 时钟线 |
| SDA | ◄──► | PB9 (I2C1_SDA) | **B9** | I2C 数据线 |
| INT3 | ◄── | PB1 (EXTI1) | **B1** | 陀螺仪就绪中断，输入给单片机计数 |

---

## 硬件布线规范

1.  **强弱电隔离**：电池动力线（18.5V）、舵机供电线（12V）需与 I2C、UART 信号线物理隔离至少 2 cm，严禁平行走线，防大电流电磁干扰导致通信丢包。
2.  **IMU 排线线长**：I2C 信号线极易受分布电容影响，连接线必须控制在 **15 cm 以内**（推荐小于 10 cm）。SCL/SDA 必须有 4.7kΩ 左右的上拉电阻（通常模块板上已自带）。
3.  **单点共地 (Star Grounding)**：STM32、树莓派、RS485模块、舵机控制板的地线均需通过粗线汇集到电池主分电板的 GND 焊盘上，防止地回流影响控制精度。
4.  **禁止双路供电**：CH340 的 3.3V 仅在无电池时作调测调试用。系统接入动力电池时，**严禁**连接 CH340 的 3.3V 针脚。
