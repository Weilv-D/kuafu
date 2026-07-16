# Debug Tools

通过 DAPLink (CMSIS-DAP) + pyOCD 经 SWD 非侵入式读取 STM32 运行时内存，
用于在不改动固件、不占用串口的前提下验证传感器与控制状态。

## 依赖

- Python 3
- pyOCD: `pip install pyocd`
- STM32F407 设备包（首次使用）: `pyocd pack install stm32f407zgtx`
- Keil 生成的 `stm32_firmware.axf`（用于结构体偏移校验，见下）

## 工具

### read_imu_state.py — IMU/姿态实时监控

读取并解码关键全局变量（地址取自 `../stm32_firmware/stm32_firmware.map`，
结构体偏移取自 `fromelf --fieldoffsets`），连续采样输出：

- `g_system_ticks` — PB1(陀螺仪 DRDY) 中断计数，验证调度心跳
- `g_imu` — 加速度(m/s²) / 陀螺仪(rad/s) / 温度(°C)
- `g_safety_state` — 陀螺仪零偏、校准完成标志、状态机/故障码
- `g_mahony` — roll / pitch / yaw 姿态角
- `g_body_gyro` — 去偏后的机体角速度

用法：

```bash
# 默认 5 次，间隔 0.5s
python read_imu_state.py

# 采集 10 次，间隔 0.3s
python read_imu_state.py 10 0.3
```

判定参考（板子静止水平放置）：
- 加速度 z ≈ +9.8 m/s²，|a| ≈ 9.8
- 陀螺仪 ≈ 0 (±0.01 rad/s)，零偏收敛后 calib=1
- roll/pitch ≈ 0°，yaw 缓慢漂移（无磁力计属正常）
- 未接舵机时 mode=FAULT/SERVO 属预期，不影响 IMU 验证

## 连接

DAPLink → STM32 SWD 五线：`SWCLK→CLK`、`SWDIO→DIO`、`3V3`、`GND`、`RST`。
本工程的 DAPLink 唯一 ID 为 `LU_2022_8888`（脚本内已写死，更换调试器时改 `PROBE`）。
