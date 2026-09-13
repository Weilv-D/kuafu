# KUAFU STM32 Firmware

The STM32F407ZG firmware owns actuator safety, device discovery, state estimation,
baseline balance control, leg kinematics, and the real-time link to the Raspberry
Pi 5. The firmware reached electronics bring-up acceptance on 2026-07-16 with the
BMI088, two DDSM315 wheel motors, and four ST3215 servos powered together.

## Runtime

- BMI088 accel/gyro are read with independent per-channel health. The legacy
  aggregate `health` only refreshes on a valid accel+gyro pair, so one dead
  channel can no longer be masked by the other. `mahony_update_validated()`
  rejects invalid gyro outright (control authority revoked immediately) and
  bounds accel-invalid gyro-only propagation to 100 ms.
- The 250 Hz control section runs on a wall-clock deadline (never on the gyro
  data-ready interrupt): safety state machine -> actuator-supervisor verdict ->
  wheel output gate -> LQR/LQI computation -> balance trace, in one linear
  sequence, with bus dispatch after it. A denial and every command derived
  from it are therefore ordered within a single control period, and the gate
  verdict is held in `g_wheel_output_gate` between deadlines. If the IMU dies,
  fault detection and tracing keep running on the wall clock.
- LQR/LQI command calculation integrates with the measured wall-clock delta
  (`lqr_update_elapsed_dt`), bounded to 20 ms; a skipped or late deadline is
  never integrated as a nominal 4 ms step.
- Base velocity/yaw references are an ACTIVE-mode contract
  (`firmware_runtime.velocity_command_active`): STAND is a position hold and
  CLIMB drives leg height only, regardless of what the heartbeat carries;
  the learned residual is additionally gated by link freshness.
- The shared DDSM115/315 bus keeps its proven 4 ms alternating schedule (each
  wheel 125 Hz). Commands beyond the in-flight transaction are queued (depth 4)
  with overflow accounting, and `ddsm_bus_get_last_tx()` exposes
  queued/TX-complete/feedback status without inventing ACKs for no-reply
  frames.
- ST3215 feedback is polled round-robin through a circular DMA ring whose
  producer is reconstructed as lap_count * size + write_index (the DMA
  transfer-complete interrupt is the lap truth source; the lap count is read
  before NDTR, an ordering under which every interrupt interleaving is exact
  or a bounded undercount and never an overcount). A consumer lag past one
  full ring is billed to `overrun_count`; speed decodes as sign-magnitude per
  the ST serial protocol. The Pi USART6 ring uses the same accounting in
  `pi_transport`, and both re-arm paths (post-error abort) reset their
  consumer state so pre-restart bytes are never re-parsed.
- An `ActuatorSupervisor` is the final output verdict: after any fault (or
  restart) recovery re-issues servo torque enable and requires a transmitted
  leg hold, fresh leg feedback, a safe posture, and verified enables before
  its `wheel_output_allowed` verdict opens. The persistent wheel gate combines
  that verdict with the runtime intent and completed current-loop
  re-assertion, and both the LQR computation and the bus dispatch obey the
  same gate, so torque can never leave the MCU while the legs underneath it
  are unverified. Serious faults (tilt, pitch rate, overtemp, IMU, emergency,
  init, internal) latch until reset.
- The Pi bridge uses USART6 at 921600 baud with circular DMA reception.

## Balance Trace

`balance_trace` records a 256-sample (1.024 s) window at every 250 Hz control
tick — including INIT/FAULT — with real firmware timestamps, mode/fault/startup/
actuator state, per-device freshness ages, and separate target/sent/feedback
wheel torques in body-frame Nm. After a fault it keeps a 64-sample tail, then
freezes. `MDK-ARM/debug_tools/dump_balance_trace.py` attaches over SWD without
resetting or halting the target, exports only genuinely recorded samples, and
writes a provenance sidecar (`*.meta.json`) plus build/model-hash metadata.

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

**RL policy contract:** the observation/action contract is now `v1.2.0`. Timing,
sign, and torque-gating changes make existing `v1.1.0` policy artifacts
incompatible; ACTIVE-mode deployment requires a retrained or explicitly
revalidated policy (see `policy.onnx.manifest.json`). Standalone STAND balance
does not depend on the policy.

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
