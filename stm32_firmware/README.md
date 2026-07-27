# KUAFU STM32 Firmware

The STM32F407ZG firmware owns actuator safety, device discovery, state estimation,
baseline balance control, leg kinematics, and the real-time link to the Raspberry
Pi 5. The firmware reached electronics bring-up acceptance on 2026-07-16 with the
BMI088, two DDSM315 wheel motors, and four ST3215 servos powered together.

## Runtime

- BMI088 sampling and Mahony attitude estimation run from the 1 kHz gyro DRDY
  timebase.
- LQR/LQI command calculation runs at 250 Hz.
- The shared DDSM315 bus runs one request/response transaction every 4 ms and
  alternates left ID 1 and right ID 2. Each wheel is therefore serviced at
  125 Hz. Adapter echo and misaligned bytes are handled by a sliding CRC window.
- ST3215 feedback is polled at 50 Hz. Startup broadcasts torque-disable to all
  servos, completes discovery, then enables IDs 1–4 individually.
- The Pi bridge uses USART6 at 921600 baud with circular DMA reception.

## Actuator Safety

Wheel power is a separately authorized domain. Startup and `INIT` use read-only
DDSM feedback queries; they do not send a zero-current motion command and do not
enable either wheel. Wheel enable requires both of the following:

1. startup phase `READY`; and
2. no latched safety fault.

Self-balancing is a baseline capability that works standalone: `INIT -> STAND`
does not require a Pi link, so after startup the robot enables its wheels and
runs the LQR zero-velocity hold with no Raspberry Pi attached. The Pi link only
gates higher-level behavior:

- a compatible Pi `HELLO` model hash plus a fresh heartbeat plus an explicit
  mode request are required to enter `ACTIVE` (the only mode with a live learned
  residual, i.e. the only mode in which the robot walks);
- a stale action removes residual commands, while a stale heartbeat drops
  `ACTIVE`/`CLIMB` back to `STAND` and re-anchors the local hold reference.

`STAND` and `CLIMB` both run the LQR/LQI baseline as a zero-velocity hold (the
Pi commands zero velocity/yaw); `CLIMB` is otherwise a reserved placeholder that
shares `STAND`'s servo and wheel code path and is not trained in the RL policy.
See `docs/architecture/system.md` for the full mode table and the CLIMB caveat.

In `FAULT` the firmware keeps streaming explicit zero-torque frames to both
wheels (a read-only query would leave the last torque latched in the motor) and
queues motor disable. The safety state machine runs on the millisecond control
deadline, not on the gyro DRDY timebase, so a dead BMI088 still latches
`FAULT_IMU` and zeroes the wheels instead of freezing with torque applied.
Gyro bias calibration only accumulates samples while all gyro axes read below
0.08 rad/s; a moving robot never completes calibration. Temperature must remain
above 65°C continuously for 100 ms before the over-temperature fault is
latched, which rejects isolated telemetry spikes without weakening sustained
over-temperature protection. The IWDG window is ~1 s.

## Calibrated Hardware Values

- DDSM315 IDs: left 1, right 2.
- ST3215 IDs and firmware order: `[1,2,3,4] = [A_l,A_r,B_l,B_r]`.
- Servo dwell centers: `{275,1097,2809,1023}`.
- Servo directions: `{+1,-1,+1,-1}`.
- Increasing leg extension produces raw tick changes
  `[decrease,increase,increase,decrease]`.
- Battery voltage sensing is not populated. `battery_mv=0` means unavailable and
  is never an undervoltage fault input.

## Build And Test

Run the host test suite and target build from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File stm32_firmware\tests\run_host_tests.ps1
powershell -ExecutionPolicy Bypass -File stm32_firmware\tools\build_keil.ps1
```

The target project is `MDK-ARM/stm32_firmware.uvprojx`. The accepted build has
zero compiler errors and zero warnings. Flash and inspect it with the tools under
`MDK-ARM/debug_tools`. The final evidence is recorded in
`../docs/validation/stm32-firmware-2026-07-16.md`.

The electronics gate does not replace mechanical motion acceptance. Wheel
direction, yaw sign, tethered balance, and the five-bar range sweep remain ordered
tests in `../docs/hardware/calibration.md` and `../docs/validation/acceptance.md`.
