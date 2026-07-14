# KUAFU STM32 Firmware

The STM32F407 owns baseline stabilization and actuator safety. The Pi5 supplies high-level commands and bounded residual actions; it does not send direct servo angles.

## Control

- 250 Hz discrete LQR/LQI on position, pitch, velocity, and pitch rate.
- Jerk-limited velocity/yaw references integrate to position/heading hold references.
- Wheel decomposition is common torque plus yaw torque, with `tau_R > tau_L` defined as positive yaw.
- 50 Hz leg control maps independent Qx/D0 residuals through five-bar IK.
- Action freshness clears only learned residuals. Heartbeat freshness additionally commands zero velocity/yaw while preserving baseline hold. Hard actuator shutdown is reserved for true faults.

`Core/Inc/kuafu_generated.h` is generated from `../kuafu_physics.py`. Verify it before firmware builds:

```bash
rl/.venv/bin/python rl/verify/generate_artifacts.py
```

## UART

Frames are `A5 | version | type | length | seq:u16 | timestamp:u32 | payload | crc8 | 5A`. The streaming parser retains partial DMA-IDLE fragments, rejects CRC failures and replayed sequences, and separates heartbeat from action freshness. The normative payload and scale definitions are in `../docs/contracts/interface.md`.

## Build And Bring-Up

The checked-in target project is `MDK-ARM/stm32_firmware.uvprojx`. Host syntax checks can validate controller, kinematics, and protocol C files, but a Keil/ARM target build and HIL loopback remain required before flashing.

Follow `../docs/hardware/calibration.md` and `../docs/validation/acceptance.md`; default servo centers, mirror signs, IMU axes, battery value, and motor direction are not hardware calibration results.
