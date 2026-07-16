# Debug Tools

通过 DAPLink (CMSIS-DAP) + pyOCD 经 SWD 非侵入式读取 STM32 运行时内存，
用于在不改动固件、不占用串口的前提下验证传感器、标定舵机与监控控制状态。

## 依赖

- Python 3
- pyOCD: `pip install pyocd`
- STM32F407 设备包（首次使用）: `pyocd pack install stm32f407zgtx`

## 连接

DAPLink → STM32 SWD 五线：`SWCLK→CLK`、`SWDIO→DIO`、`3V3`、`GND`、`RST`。
本工程的 DAPLink 唯一 ID 为 `LU_2022_8888`（脚本内已写死，更换调试器时改 `PROBE`）。

目标芯片全速运行（阻塞式 I2C + 看门狗）会使 SWD 初始握手不稳定，所有脚本均用
`connect-under-reset`（连接时拉 NRST）+ 重试保证可靠附着，连上后恢复核心运行。

## read_imu_state.py — IMU/姿态实时监控

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

## calib_servo_zero.py — 舵机机械零位（dwell）标定

手动找零：固件在 FAULT/SERVO 下会关闭 4 个 ST3215 的扭矩（自由状态），
你用手把每条腿摆到 dwell 机械中位，脚本经 SWD 读回各舵机的 present
position raw tick，即为 `pin_config.h` 里的 `SERVO_CENTER[i]`。

用法：

```bash
# 1. 接好 4 个 ST3215（ID 1/2/3/4，USART3 PB10 总线）并供电
# 2. 上电 STM32（connect-under-reset 会自动复位一次让 is_online 重置）
python calib_servo_zero.py
#    实时显示 4 个舵机的 tick，手摆正后 Ctrl+C 捕获
#    脚本会打印可直接粘贴进 pin_config.h 的 SERVO_CENTER_INIT
# 3. 改 pin_config.h，重编译重烧
```

约定（固件 `st3215.c`）：
`position_rad = (raw_tick - 2048) * (2π/4096)`，故
`raw_tick = position_rad / (2π/4096) + 2048`。

注意：
- 同时也需实测 `SERVO_DIR`（右侧 RF/RB 默认 -1），见 pin_config.h 注释。
- 舵机离线（is_online=OFF）时 tick 无意义，先确保 4 个都 online 再摆位。
- 这是 `docs/hardware/calibration.md` Bring-Up 第 3 步，须在低扭矩下进行。
