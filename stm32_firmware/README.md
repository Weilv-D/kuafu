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

Wheel power is a separately authorized domain. Startup, `INIT`, `FAULT`, and a
missing or incompatible Pi link use read-only DDSM feedback queries; they do not
send a zero-current motion command and do not enable either wheel. Wheel enable
requires all of the following:

1. startup phase `READY`;
2. no latched safety fault;
3. a compatible Pi `HELLO` model hash;
4. a fresh heartbeat; and
5. an explicit `STAND`, `ACTIVE`, or `CLIMB` mode request.

`ACTIVE` is the only mode in which the learned residual is live, so it is the only
mode in which the robot walks. `STAND` and `CLIMB` both run the LQR/LQI baseline as
a zero-velocity hold (the Pi commands zero velocity/yaw); `CLIMB` is otherwise a
reserved placeholder that shares `STAND`'s servo and wheel code path and is not
trained in the RL policy. See `docs/architecture/system.md` for the full mode table
and the CLIMB caveat.

Loss of authorization disables both wheels. A stale action removes residual
commands, while a stale heartbeat removes motion authorization. Temperature must
remain above 65°C continuously for 100 ms before the over-temperature fault is
latched, which rejects isolated telemetry spikes without weakening sustained
over-temperature protection.

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
