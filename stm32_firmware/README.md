# 夸父 KuaFu — STM32F407ZGT6 固件说明书

本目录为夸父平衡机器人底层的 STM32F407ZGT6 主控固件，基于 STM32 HAL 库实现硬实时控制，涵盖 1 kHz IMU 姿态解算、250 Hz 双轮 LQR 自平衡、50 Hz 髋关节舵机同步控制，以及与上层树莓派 5 的串口桥接通信。

---

## 核心板硬件规格

主控板型号为 LXB407ZG-P1 核心板，外设引脚与功能定义如下：

*   **USB 接口**：板载 Type-C 接口直接连接 STM32 的 PA11 和 PA12 引脚，不具备硬件 USB 转串口电路。烧录与调试时，必须将外部 CH340 串口模块连接至标有 TXD 和 RXD 的 USART1 排针。
*   **按键配置**：
    *   **SW2**：复位按键，低电平复位。
    *   **SW3**：BOOT0 控制按键，按下时引脚输入高电平，松开时默认下拉至低电平。
    *   **SW1**：用户按键，连接至 PA15 引脚，低电平有效。
*   **LED 配置**：
    *   **LED1**：电源指示灯，上电常亮。
    *   **LED2**：用户指示灯，连接至 PC13 引脚，输入低电平点亮。

---

## 目录结构

*   `Core/`
    *   `Inc/`: 存放所有头文件，包含引脚定义、控制算法、通信协议以及状态机的声明：
        *   [pin_config.h](Core/Inc/pin_config.h): 硬件引脚映射、LQR 控制增益、并联机构几何参数与安全阈值。
        *   [crc8.h](Core/Inc/crc8.h): DDSM 电机通信校验的 CRC-8/MAXIM 声明。
        *   [pi_link.h](Core/Inc/pi_link.h): 树莓派桥接串口协议声明。
        *   [bmi088.h](Core/Inc/bmi088.h): BMI088 六轴惯性传感器驱动声明。
        *   [ddsm315.h](Core/Inc/ddsm315.h): DDSM315 电机控制驱动声明。
        *   [st3215.h](Core/Inc/st3215.h): ST3215 菊花链总线舵机驱动声明。
        *   [mahony.h](Core/Inc/mahony.h): Mahony 滤波姿态解算声明。
        *   [kinematics.h](Core/Inc/kinematics.h): 五杆并联逆运动学解算声明。
        *   [lqr_controller.h](Core/Inc/lqr_controller.h): 平衡控制器声明。
        *   [safety_state.h](Core/Inc/safety_state.h): 安全保护状态机声明。
    *   `Src/`: 存放所有源文件，包含上述各个模块的业务逻辑实现：
        *   [main.c](Core/Src/main.c): 主程序入口，分时任务调度及中断计数。
        *   [safety_state.c](Core/Src/safety_state.c): 系统运行状态机与安全阀限监测保护。
        *   [crc8.c](Core/Src/crc8.c) / [pi_link.c](Core/Src/pi_link.c) / [bmi088.c](Core/Src/bmi088.c) / [ddsm315.c](Core/Src/ddsm315.c) / [st3215.c](Core/Src/st3215.c): 底层设备驱动与协议实现。
        *   [mahony.c](Core/Src/mahony.c) / [kinematics.c](Core/Src/kinematics.c) / [lqr_controller.c](Core/Src/lqr_controller.c): 控制算法与几何学逆运动学解算。
        *   [system_stm32f4xx.c](Core/Src/system_stm32f4xx.c) / [stm32f4xx_hal_msp.c](Core/Src/stm32f4xx_hal_msp.c) / [stm32f4xx_it.c](Core/Src/stm32f4xx_it.c): 系统底层、中断处理及硬件接口驱动。
*   `Drivers/`
    *   `CMSIS/` / `STM32F4xx_HAL_Driver/`: ST 官方 HAL 标准库底层驱动。
*   `MDK-ARM/`
    *   [stm32_firmware.uvprojx](MDK-ARM/stm32_firmware.uvprojx): Keil MDK5 核心工程项目文件。

---

## 实时时序设计

系统采用前后台异步分时时序，避免 RS485 半双工总线通信阻塞高频姿态解算：

1.  **1 kHz 传感器脉冲**：
    *   陀螺仪数据就绪引脚连接至 PB1，触发 EXTI1 中断。
    *   中断服务函数仅累加滴答计数器 `g_system_ticks` 即可退出，不执行任何阻塞操作。
2.  **主循环任务调度**：
    *   后台主循环检测到滴答更新后，立即执行 I2C 寄存器读取和姿态解算。
    *   将 4 ms 的控制周期划分为 4 个分时时间槽：
        *   **Slot 0**：发送左轮控制指令，并阻塞等待接收反馈数据，超时上限 1.5 ms。
        *   **Slot 1**：发送右轮控制指令，并阻塞等待接收反馈数据，超时上限 1.5 ms。
        *   **Slot 2**：通过 USART6 接口向树莓派发送姿态与关节状态数据包。
        *   **Slot 3**：运行 LQR 平衡控制计算、叠加控制残差并执行系统级安全诊断。
3.  **50 Hz 髋关节舵机控制**：
    *   主循环每 20 ms 触发一次，通过 1 Mbps 串口广播下发同步控制角度，并轮询读取单台舵机状态。

---

## 引脚连接与串口分配

### 1. 串口外设总览

| 串口外设 | 物理用途 | STM32 硬件引脚 | 默认波特率 |
| :--- | :--- | :--- | :--- |
| **USART1** | CH340 烧录与调试接口 | PA9 (TXD) / PA10 (RXD) | 115200 bps |
| **USART2** | RS485 模块与 DDSM 电机总线 | PA2 / PA3 | 115200 bps |
| **USART3** | 舵机控制板与 ST3215 舵机总线 | PB10 / PB11 | 1000000 bps |
| **USART6** | 树莓派 5 桥接通信接口 | PC6 / PC7 | 921600 bps |
| **I2C1** | BMI088 IMU 传感器总线 | PB8 / PB9 | 400 kHz |

### 2. 详细接口连线

#### A. CH340 串口下载器

| 模块侧引脚 | 信号方向 | MCU 侧引脚 | 实物丝印 | 备注 |
| :--- | :---: | :--- | :---: | :--- |
| 3.3V | ──► | 3.3V 供电 | 3V3 | 调测供电，接入动力电池时切勿连接 |
| TXD | ──► | PA10 | RXD | 下载器发送端连接 MCU 接收端 |
| RXD | ◄── | PA9 | TXD | 下载器接收端连接 MCU 发送端 |
| GND | ─── | GND | GND | 信号共地 |
| rst | ✗ | 悬空 | — | 悬空不接 |

**串口烧录步骤：**
1. 按住核心板上的 BOOT0 按键（SW3），按下并释放 RST 按键（SW2），最后松开 BOOT0 按键进入 Bootloader 模式。
2. 在烧录软件中选择 COM 端口并载入固件。
3. 启动下载直至完成。
4. 按下 RST 按键（SW2）运行程序。

#### B. 树莓派 5 通信接口

| 树莓派侧引脚 | 信号方向 | MCU 侧引脚 | 实物丝印 | 备注 |
| :--- | :---: | :--- | :---: | :--- |
| Pin 1 (TX) | ──► | PC7 | C7 | 接收来自树莓派的数据 |
| Pin 2 (GND) | ─── | GND | GND | 信号共地 |
| Pin 3 (RX) | ◄── | PC6 | C6 | 向树莓派发送遥测数据 |

- 电平标准：3.3V TTL 电平直连。
- 树莓派串口节点：`/dev/ttyAMA0`，需配置系统关闭串口控制台占用。

#### C. ST3215 舵机总线

| 舵机控制板引脚 | 信号方向 | MCU 侧引脚 | 实物丝印 | 备注 |
| :--- | :---: | :--- | :---: | :--- |
| RXD | ──► | PB10 | B10 | 信号输出端 |
| TXD | ◄── | PB11 | B11 | 信号输入端 |
| GND | ─── | GND | GND | 信号共地 |

- 通信模式：1 Mbps 半双工 TTL 串联。
- 控制板逻辑端使用板载自供电，切勿连接 MCU 的 3.3V 供电线。舵机 12V 动力电源单独由降压模块提供。

#### D. RS485 电机总线

| 转换模块引脚 | 信号方向 | MCU 侧引脚 | 实物丝印 | 备注 |
| :--- | :---: | :--- | :---: | :--- |
| RX | ──► | PA2 | A2 | 信号输出端 |
| TX | ◄── | PA3 | A3 | 信号输入端 |
| VCC | ──► | 3.3V | 3V3 | 模块逻辑供电 |
| GND | ─── | GND | GND | 信号共地 |

- 硬件要求：RS485 转换模块需具备自动收发流控功能。
- 总线并联配置：左轮电机 ID 为 1，右轮电机 ID 为 2。

#### E. BMI088 传感器

| 传感器引脚 | 信号方向 | MCU 侧引脚 | 实物丝印 | 备注 |
| :--- | :---: | :--- | :---: | :--- |
| VCC | ──► | 3.3V | 3V3 | 传感器供电 |
| GND | ─── | GND | GND | 信号共地 |
| SCL | ──► | PB8 | B8 | I2C 时钟线 |
| SDA | ◄──► | PB9 | B9 | I2C 数据线 |
| INT3 | ◄── | PB1 | B1 | 陀螺仪就绪中断输出 |

---

## 硬件布线规范

1.  **强弱电隔离**：动力供电主线需与 I2C、UART 信号线物理隔离至少 2 cm，禁止平行走线，防大电流电磁干扰。
2.  **IMU 排线限制**：I2C 信号排线长度需控制在 15 cm 以内。SCL 和 SDA 信号线必须拉高至 3.3V（通常 BMI088 模块已集成 4.7kΩ 上拉电阻）。
3.  **单点共地**：MCU、树莓派、RS485模块、舵机控制板的地线需通过粗导线并联汇集至分电板的主 GND 焊盘。
4.  **防电源冲突**：外部 CH340 模块的 3.3V 供电线仅用于无主电池时的调试。接入电池时严禁连接该供电线。

---

## 协议与文档参考来源

1.  **DDSM315 电机协议**：
    *   微雪电子 DDSM315 硬件产品 Wiki：[https://www.waveshare.net/wiki/DDSM315](https://www.waveshare.net/wiki/DDSM315)
    *   微雪官方驱动开源仓库：[https://github.com/waveshareteam/ddsm_example](https://github.com/waveshareteam/ddsm_example)
2.  **ST3215 舵机协议**：
    *   微雪电子 ST3215 总线舵机 Wiki：[https://www.waveshare.net/wiki/ST3215_Servo](https://www.waveshare.net/wiki/ST3215_Servo)
    *   飞特总线舵机 ESP32 控制例程仓库：[https://github.com/waveshareteam/ServoDriverST](https://github.com/waveshareteam/ServoDriverST)
3.  **BMI088 传感器规格**：
    *   Bosch Sensortec BMI088 数据手册：[https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi088-ds001.pdf](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi088-ds001.pdf)
4.  **主控通信协议**：
    *   参考本仓库设计文档：[docs/plans/2026-07-08-软件架构与RL技术路线-design.md](../docs/plans/2026-07-08-%E8%BD%AF%E4%BB%B6%E6%9E%B6%E6%9E%84%E4%B8%8ERL%E6%8A%80%E6%9C%AF%E8%B7%AF%E7%BA%BF-design.md)
