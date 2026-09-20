# KUAFU STM32 Firmware

The STM32F407ZG firmware owns actuator safety, device discovery, state estimation,
baseline balance control, leg kinematics, and the real-time link to the Raspberry
Pi 5. The firmware reached electronics bring-up acceptance on 2026-07-16 with the
BMI088, two DDSM315 wheel motors, and four ST3215 servos powered together.

## Runtime

- BMI088 accel/gyro are read with independent per-channel health. The legacy
  aggregate `health` only refreshes on a valid accel+gyro pair, so one dead
  channel can no longer be masked by the other. A fusion cycle reads both
  channels under one caller-supplied timestamp (`bmi088_read_pair`): the pair
  is defined by that shared instant rather than by two separately sampled
  tick values, so the aggregate refresh never depends on where the two I2C
  transactions fall relative to a millisecond boundary.
  `mahony_update_validated()` rejects invalid gyro outright (control authority
  revoked immediately) and bounds accel-invalid gyro-only propagation to
  100 ms.
- The 250 Hz control section runs on a wall-clock deadline (never on the gyro
  data-ready interrupt): safety state machine -> actuator-supervisor verdict ->
  wheel output gate -> LQR/LQI computation -> balance trace, in one linear
  sequence, with bus dispatch after it. A denial and every command derived
  from it are therefore ordered within a single control period, and the gate
  verdict is held in `g_wheel_output_gate` between deadlines. If the IMU dies,
  fault detection and tracing keep running on the wall clock.
- Telemetry splits by timebase. Fusion-state frames (IMU at 250 Hz, joints at
  250 Hz) ride the 1 kHz data-ready tick and the fusion gate — with the
  fusion path down their values are frozen and worthless. Diagnostic frames
  (diag at 250 Hz, health at 10 Hz, FAULT while latched) ride the SysTick
  wall clock instead: an IMU-path failure — I2C dead or the data-ready line
  itself stopped — leaves fault latching, per-device ages, error counters,
  and the FAULT broadcast to the Pi fully alive, so the failure that killed
  balance is also the one best reported over the link.
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
- Servo freshness requires a real first feedback frame: the poll round-robin
  restores online state on the first valid reply, and a servo that never
  replied cannot carry startup into STAND or the wheel gate open on legs
  whose feedback has never been seen.
- ST3215 feedback is polled round-robin through a circular DMA ring whose
  producer is reconstructed as lap_count * size + write_index (the DMA
  transfer-complete interrupt is the lap truth source; the lap count is read
  before NDTR, an ordering under which every interrupt interleaving is exact
  or a bounded undercount and never an overcount). A consumer lag past one
  full ring is billed to `overrun_count`; speed decodes as sign-magnitude per
  the ST serial protocol. Write frames (sync-write, torque) carry their own
  20 ms deadline, and BOTH write-frame failure paths — a lost TX-complete
  interrupt and a UART error recorded during the write — recover through an
  explicit transmit abort, so either costs one frame instead of the servo
  subsystem. The ring is initialised before the DMA is started, and the
  Pi USART6 ring uses the same accounting in `pi_transport`; both re-arm
  paths (post-error abort) rebase their consumer state at the current lap
  boundary — preserving the lap truth, since the synchronous abort
  completes the old stream — so pre-restart bytes are never re-parsed and
  no already-counted lap is ever erased. When a consumer falls a full ring
  behind, the overwritten prefix is billed to `overrun_count` and the
  still-valid remainder is parsed in the same poll: the parse span comes
  from the absolute produced/consumed cursors, never from a
  read-index-vs-write-index comparison, so an overrun can never leave the
  consumer stalled one ring behind the producer (billing every fresh byte as
  another overrun and parsing nothing), and a stale lap/NDTR sample — the
  bounded undercount the read ordering permits — can only defer parsing by
  one poll.
- The 50 Hz leg-write deadline is retry-on-busy, not drop-on-busy: a
  refused sync-write (or FAULT torque-disable) keeps its deadline pending
  and is retried on the next scheduler pass until a frame is actually
  queued. Five dropped periods would expire the 100 ms leg-hold window and
  close the wheel gate on a balancing robot, which a degraded servo bus
  made a recurring event. This is the same retry shape the 4 ms wheel
  dispatch has always had.
- An `ActuatorSupervisor` is the final output verdict: after any fault (or
  restart) recovery re-issues servo torque enable and requires a transmitted
  leg hold, fresh leg feedback, a safe posture, and verified enables before
  its `wheel_output_allowed` verdict opens. The servo enable sequencer in the
  scheduler is FAULT-gated, so the physical re-issue happens only after the
  safety machine has left FAULT — during FAULT the one-shot torque-disable
  owns the servo bus and the software `servos_enabled` flag stays low, which
  is the drop-and-return the supervisor's re-issue stage keys on. The
  persistent wheel gate combines that verdict with the runtime intent and
  completed current-loop re-assertion, and both the LQR computation and the
  bus dispatch obey the same gate, so torque can never leave the MCU while
  the legs underneath it are unverified. Serious faults (tilt, pitch rate,
  overtemp, IMU, emergency, init, internal) latch until reset; the transient
  device faults (wheel/servo freshness) auto-recover through that same
  re-issue stage.
- The Pi bridge uses USART6 at 921600 baud with circular DMA reception.
  The receive sequence gate is restart-tolerant in both directions: a
  validated HELLO starts a new session, and eight consecutive CRC-valid,
  non-HELLO frames rejected by the monotonic gate alone are judged a Pi
  restart (the firmware mirror of the Pi-side decoder resync), so a
  rebooted Pi whose single startup HELLO was corrupted on the wire is not
  locked out of `ACTIVE` until the STM32 is power-cycled. Payload-invalid
  frames never count toward the resync and isolated duplicates are still
  dropped. The decoder consumes a whole ring of bytes in bounded slices —
  each slice fills the holding buffer, drains every complete frame and keeps
  only the trailing fragment, and the next slice continues behind it — so a
  chunk larger than the buffer is decoded instead of dropped, and only a
  fragment that can never complete inside the buffer (impossible under the
  frame-length validation) is discarded. Frame encoders decode non-finite
  values as their safe sentinels (zero torque, dwell tick) rather than
  undefined float→int casts.

## Balance Trace

`balance_trace` records a 256-sample (1.024 s) window at every 250 Hz control
tick — including INIT/FAULT — with real firmware timestamps, mode/fault/startup/
actuator state, per-device freshness ages, and separate target/sent/feedback
wheel torques in body-frame Nm. After a fault it keeps a 64-sample tail, then
freezes so the post-fault window survives for the dump; the freeze releases
when a new fault arrives after a fault-free period (the newer evidence
supersedes the preserved window) or once the system has been fault-free for a
full tail, so an auto-recovered transient fault cannot blind the rest of the
session, while a latched serious fault keeps the ring frozen by design.
`MDK-ARM/debug_tools/dump_balance_trace.py` attaches over SWD without
resetting or halting the target, exports only genuinely recorded samples, and
writes a provenance sidecar (`*.meta.json`) plus build/model-hash metadata.

## Actuator Safety

Wheel power is a separately authorized domain. Startup and `INIT` use read-only
DDSM feedback queries; they do not send a zero-current motion command and do not
enable either wheel. Wheel enable requires all of the following:

1. startup phase `READY`;
2. an operational mode (`STAND`/`ACTIVE`/`CLIMB`) — `INIT` keeps the domain
   locked to read-only queries; and
3. no latched safety fault.

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

Whenever the wheel output gate closes on a wheel that is still enabled — in
`FAULT`, or in an operational mode demoted by the actuator supervisor — the
firmware streams explicit zero-torque frames to it (a read-only query would
leave the last torque latched in the motor's current loop, actively driving
an uncontrolled robot); `FAULT` additionally queues motor disable, and plain
read queries are used only while the wheels are disabled (INIT/discovery),
where a command frame would be pointless. Torque frames elicit the same
status reply as queries, so feedback freshness survives every branch. The
safety state machine runs on the millisecond control
deadline, not on the gyro DRDY timebase, so a dead BMI088 still latches
`FAULT_IMU` and zeroes the wheels instead of freezing with torque applied.
Gyro bias calibration only accumulates samples while all gyro axes read below
0.08 rad/s; a moving robot never completes calibration. Temperature must remain
above 65°C continuously for 100 ms before the over-temperature fault is
latched, which rejects isolated telemetry spikes without weakening sustained
over-temperature protection. The freshness-fault debounce (8 ticks / 32 ms)
is additionally suppressed for 100 ms after the two device-driven mode
transitions (INIT→STAND, FAULT→STAND recovery) only — command-driven
transitions grant no grace and reset no counters, so a Pi flapping mode
requests cannot hold the suppression window open. The IWDG window is ~1 s.

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
`MDK-ARM/debug_tools`. The most recent logic-review evidence is recorded in
`../docs/validation/` (latest record: `stm32-firmware-2026-09-21.md`; the
electronics bring-up acceptance remains `stm32-firmware-2026-07-16.md`).

The electronics gate does not replace mechanical motion acceptance. Wheel
direction, yaw sign, tethered balance, and the five-bar range sweep remain ordered
tests in `../docs/hardware/calibration.md` and `../docs/validation/acceptance.md`.
