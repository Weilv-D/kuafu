# STM32 Firmware Acceptance Record — 2026-07-16

## Scope

This record covers STM32 firmware, powered peripheral communication, startup,
and actuator authorization safety. RL training and ground locomotion are outside
this acceptance record.

## Hardware

- MCU: STM32F407ZG
- Debug probe: DAPLink `LU_2022_8888`
- IMU: BMI088 on I2C1 with PB1 gyro DRDY
- Wheels: two DDSM315 motors on USART2 RS485, left ID 1 and right ID 2
- Servos: four ST3215 on USART3 through Waveshare Bus Servo Adapter A, IDs 1–4
- Battery voltage input: not connected

## Firmware Behavior

- Startup completes device initialization and reaches `READY`.
- DDSM315 requests alternate every 4 ms across IDs 1 and 2.
- ST3215 startup uses broadcast torque-disable and individual enable.
- Wheel enable requires startup readiness, no fault, compatible Pi model hash,
  fresh heartbeat, and explicit mode request.
- Without Pi authorization, firmware sends read-only wheel queries and both
  wheels remain disabled.
- Sustained temperature above 65°C for 100 ms latches over-temperature fault;
  isolated samples do not.

## Verification Results

| Check | Result |
|---|---|
| Host C tests | 100% passed |
| Keil target build | 0 errors, 0 warnings |
| Final HEX | 69,523 bytes |
| HEX SHA-256 | `6717BF9C00236FD55F9F9FC8A9E95AA744F0627A8A45CABCD9C6123D0DB87829` |
| pyOCD flash | 32,768 bytes erased; 25,600 bytes programmed |
| Startup | `READY` at 1.739 s |
| Runtime mode | Safe `STAND` |
| Fault mask | `0x00000000` |
| BMI088 | Online, 0–1 ms, 33.1–33.4°C |
| Left DDSM315 | ID 1, online, 0–12 ms |
| Right DDSM315 | ID 2, online, 0–12 ms |
| ST3215 IDs 1–4 | Online, 0–20 ms, 41–43°C |
| Uncommanded wheel motion | None with Pi authorization absent |

## Calibration Values

```text
SERVO_CENTER_INIT = {275, 1097, 2809, 1023}
SERVO_DIR_INIT    = {+1, -1, +1, -1}
```

Increasing `D0` maps to joint signs `[A_l<0,A_r<0,B_l>0,B_r>0]` and raw tick
changes `[decrease,increase,increase,decrease]`.

## Acceptance

STM32 firmware and electronics bring-up are accepted. The flashed image is the
normal runtime image with DDSM ID calibration disabled and servo zero calibration
disabled. Mechanical motion gates remain governed by
`docs/hardware/calibration.md`.
