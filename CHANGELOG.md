# Changelog

## Unreleased

### Firmware logic review, second pass (2026-09-20)

An independent fourth-pass review layered on the 09-20 three-pass baseline
(architecture and every module re-derived from source before the prior
records were consulted). Record: `docs/validation/stm32-firmware-2026-09-20-2.md`.

- Wheel dispatch no longer leaves torque latched in the DDSM315 current
  loop when the output gate closes on a still-enabled wheel: an explicit
  zero-torque frame is sent in any mode (previously only `FAULT`; other
  gate-closed states sent a read query, and the motor kept driving the
  pre-closure LQR torque — reachable via a supervisor HOLDING demotion
  while the mode stayed operational, e.g. the 100 ms leg-hold window
  expiring under servo-bus congestion — until the 45° TILT backstop).
  Torque frames elicit the same status reply as queries, so feedback
  freshness is unchanged in every branch. New regression in
  `test_ddsm315.c` pins the zero-frame encoding and the reply channel.
- `SERVO_FAIL_LIMIT` comment corrected: it is the round-robin poll's
  `offline_after` (3 consecutive read failures mark a servo offline), not a
  fatal-FAULT latch; fault latching belongs to the safety layer's freshness
  path.
- Documented as deliberate (no code change): telemetry is silent while the
  IMU path is down (slots live in the DRDY-gated fusion block — SWD-only
  forensics in that state); the DDSM feedback mode byte is diagnostic-only
  (no runtime re-assert verification); the LQR position estimate freezes
  during gate-closed periods (≤ one 20 ms catch-up step, ~4 cm worst-case
  `x_est` bias for a 100 ms demotion).
- Documentation: HANDOFF repository path updated after the workspace move;
  the dispatch policy is now stated wire-accurately in the safety-gating
  section; this changelog gained the previously missing 09-20 three-pass
  entry below.

### Firmware logic review (2026-09-20, three passes)

Full-codebase review layered on the 09-13/09-16 baselines: two review
passes (the second re-verifying the first's fixes via byte-level echo-filter
traces, FAULT bus-queue priority, a UB audit, and an NVIC priority audit)
plus a closeout pass. 10 defects fixed (1 high, 5 medium, 4 low). Record:
`docs/validation/stm32-firmware-2026-09-20.md`.

- (High) The servo enable sequencer was mode-blind: during `FAULT` it fought
  the 50 Hz one-shot torque-disable (~80 ms enable/disable churn) and re-set
  `servos_enabled`, so a transient-fault recovery could clear the
  supervisor's re-issue stage on the racing 0→1 and reopen the wheel gate on
  physically torque-free legs. The sequencer is now gated on
  `mode != FAULT`, giving the supervisor the genuine drop-and-return its
  re-issue stage requires.
- (Medium) The DDSM bus timeout verdict is re-validated under the IRQ mask
  against phase and deadline, closing the one main-context phase transition
  the concurrency contract left uncovered (a stale verdict could kill a
  transaction the completion ISR had just started).
- (Medium) ST3215 write frames (sync-write/torque) carry a 20 ms deadline
  and recover through an explicit `AbortTransmit`, so a lost TX-complete
  interrupt costs one frame instead of wedging the servo bus until a
  freshness fault.
- (Medium) Servo freshness requires a real first feedback frame: the
  boot-time optimistic online flag could make a never-replied servo look
  fresh for the first second, briefly opening the wheel gate on legs whose
  feedback had never been seen.
- (Medium, bench builds only) The two calibration images latch `FAULT_INIT`
  (serious) in addition to the transient diagnostic bits: a transient bit
  alone auto-recovers to STAND within seconds of the freshly polled devices
  reporting fresh, re-arming actuation on an image that must stay
  torque-free.
- (Medium) The 1 kHz tick block's deep `continue` became an explicit
  `if (g_imu.initialized)` guard — the `continue` silently skipped every
  scheduler stage below it (50 Hz legs, servo poll), coupling them to IMU
  init state.
- (Low) `pi_link` critical sections restore the saved PRIMASK instead of a
  bare `__enable_irq()`; `bmi088_read_temp` gained the handle guard its
  sibling readers have; `queued_mode` no longer records an enable frame's
  CRC as a control mode; the retired `STARTUP_GYRO_CALIBRATION` enum member
  was deleted (numeric gap kept for trace-encoding stability) and dead
  declarations clarified.

### Firmware & Pi5-runtime logic review (2026-09-16)

- The freshness-fault grace window (100 ms + stale-counter reset) is now
  granted only by device-driven mode transitions (INIT->STAND, FAULT->STAND
  recovery). Command-driven transitions (mode requests, link-loss demotion)
  previously reset the counters too, so a Pi flapping `mode_request` at
  >10 Hz could suppress wheel/servo loss faults indefinitely while the
  wheels stayed enabled.
- The 50 Hz leg-write deadline is no longer dropped when the ST3215 bus is
  busy: the pending flag survives until a frame is actually queued (or the
  mode has nothing to send), retried on the next scheduler pass — the same
  retry shape the wheel dispatch always had. Five bus-busy collisions used
  to expire the 100 ms leg-hold window and close the wheel gate (~100 ms of
  zero torque on a balancing robot); with one degraded servo this recurred
  regularly. `firmware_runtime.servo_intent_allowed` is now a pure
  mode-level verdict, symmetric with `wheel_intent_allowed`.
- `teleop_single` mirrors the firmware's dynamic D0 gate before encoding:
  raising the legs and pushing the stick fast used to fail validation every
  tick, silencing the heartbeat stream exactly during aggressive motion
  until the 200 ms watchdog demoted the robot. The TX error path now falls
  back to a safe STAND heartbeat instead of silence, a lost gamepad no
  longer replays its last stick values, a blocking BLE reconnect first
  forces disarm, and the exit ESTOP is flushed before close.
- The Pi-side `StreamDecoder` resynchronizes after a peer sequence restart
  (STM32 reboot restarts its TX counter at 0): 8 consecutive CRC-valid
  sequence-gated rejections clear the high-water mark. Previously every
  telemetry frame was blackholed until the rebooted counter climbed past
  the stored value. Isolated duplicates are still dropped.
- `serial_node` keeps the heartbeat contract with an explicit safe request
  (STAND, zero commands) when telemetry is stale instead of going silent,
  and resets the observation history when telemetry resumes.
- `link_probe` fault-bit table realigned with `safety_state.h` (it was
  shifted one bit — FAULT_SERVO printed as WHEEL_L_LOST); FAULT frames are
  now counted and printed, `--mode` is validated, HEALTH payloads are
  length-guarded, the loop re-clamps its deadline after a stall, and the
  exit ESTOP is flushed.
- Diag telemetry saturates the temperature into the uint8 field (a negative
  BMI088 reading was an undefined float-to-unsigned cast).
- Removed the dead `gyro_calibrated` input plumbing from
  `SafetyInputs_t`/`StartupInputs_t`/`ControlSectionInputs_t` (never
  consumed anywhere; gyro calibration is documented as a non-gate).

### Firmware logic review (2026-09-13)

- Fixed an initial-power-on deadlock in the actuator supervisor's ops-less
  enable-reissue stage: the scheduler couples `startup_ready` and
  `servo_enable_verified` to the same sequencer variable, so the verified flag
  is never observed low while `startup_ready` is high, and the unconditional
  clearing of `enable_verified_observed_low` on every STARTUP/HOLDING entry
  discarded the boot-time low observations. The wheel-output gate could never
  open from a clean boot (only a prior FAULT cycle unblocked it). Low-state
  observations are now latched on every step, and the drop-requirement is
  re-armed only when a revocation follows an authorized (READY) state;
  transient leg-readiness gaps no longer re-arm it.
- Wheel enable no longer occurs while the safety mode is still `INIT`; the
  wheel power domain stays locked to read-only queries until STAND, matching
  the documented mode table.
- ST3215 UART error handling is deferred from the error ISR to the main-loop
  bus step, eliminating an interrupt-context race on the transaction pointer,
  phase, parser, and health counters (the DDSM driver already used locking;
  this bus now uses deferral).
- The 64-byte RAM-top forensic dump area (`STALL_DUMP` + `FAULT_DUMP`) is now
  excluded from the linker layout (IRAM2 ends at `0x2001FFBF`), so future RAM
  growth can never collide with the reset-surviving dumps.
- LQR estimation authority now additionally requires IMU pair freshness, so a
  dead gyro data-ready line (which freezes `g_imu_control_valid` at its last
  value) revokes torque at the 20 ms freshness bound instead of the ~52 ms
  freshness-fault debounce.
- Removed the unconsumed `firmware_runtime.clear_motion` output (motion
  clearing is owned by the safety state machine) and duplicate dead
  trace-validity defines in `main.c`.
- Documentation aligned: schema references updated to `v1.2.0`, the startup
  chain no longer lists the unused GYRO_CALIBRATION phase, the wheel-domain
  gating text matches the standalone-STAND design, and stale DDSM timeout /
  debounce numbers corrected (12 ms / 8 ticks).

### Control and contract

- Versioned 140-dimensional Actor / 152-dimensional Critic contract at schema `v1.1.0`, with a tanh-squashed policy transform shared by PPO, ONNX export, and the Pi5 runtime.
- Source-generated discrete LQR/LQI firmware constants, dwell-relative five-bar IK generation, and versioned UART framing with CRC-8/MAXIM.
- High-speed D0 gate (120 mm cap when `|v| > 0.3 m/s` or `|w| > 0.6 rad/s`) and roll leveling (`ROLL_KP = 190 mm/rad`, `ROLL_KD = 5.0`).

### Firmware

- STM32F407ZG bare-metal runtime: 1 kHz IMU fusion, 250 Hz LQR/LQI baseline, five-bar workspace projection.
- Self-balancing in STAND mode works standalone (no Pi required): wheel torque authorization is gated on startup completion and absence of fault, independent of Pi link or heartbeat freshness. The Pi link gates ACTIVE motion commands only.
- LQR control runs every 250 Hz deadline regardless of DDSM bus state; the dispatch layer skips transmission when the bus is busy so the controller is never starved.
- DDSM115 speed feedback decoded in 0.1 RPM units matching the command encoding.
- DDSM transaction timeout set to 12 ms and freshness-fault debounce to 8 ticks to accommodate marginal RS485 links without spurious FAULT_WHEEL.
- Right DDSM315 motor is physically mirrored: `WHEEL_DIR_R = -1` and the LQR yaw-differential formula is adapted so forward torque + positive yaw produce a correct right turn.
- DDSM315 torque polarity is opposite to the cart-pole convention; LQR output is negated accordingly.
- Pitch-gated torque fade (20°→50° linear) prevents wheels spinning at full power when the robot is beyond recoverable tilt; anti-windup zeros the position integral at 50°.
- Wheel dispatch streams zero-torque frames whenever the output gate is
  closed on an enabled wheel (any mode), ensuring the current loop actually
  stops instead of holding its last setpoint; read queries are used only
  while the wheels are disabled.
- BMI088 initialization retries with I2C bus recovery (9 SCL clocks per AN3273) on failure; a soft reset no longer requires a power-cycle for the IMU to come online.
- USART ORE/NE/FE sub-error diagnostics surfaced in DeviceHealth for SWD readout; the 46-byte health telemetry payload is unchanged.
- Pi link auto-accepts the first well-formed heartbeat as proof of protocol compatibility when no explicit HELLO has been received.

### Teleop

- Two-state arm/disarm safety model: sources start DISARMED (firmware STAND, balance held, no motion commands) and require an explicit arm action before the wheels track commands.
- Stick input shaping (deadzone then square curve) and trigger deadzones; full STAND/ACTIVE/FAULT intent carried end to end over the wire.
- Gamepad hot-plug and haptic feedback on state transitions.
- ``--no-policy`` mode for baseline-only operation: skips ONNX loading, sends zero
  residual, requires no ``policy.onnx`` or manifest. Usable with or without teleop.

### Runtime and tests

- Pi5 ONNX Actor loop with manifest, schema, and calibration-table digest validation.
- Release-gate validation, atomic schema-aware checkpoints with CPU-first loading.
- Host test suites: firmware C tests (CMake), teleop Python tests, and contract tests.
- Documentation reorganized around architecture, contracts, operations, validation, and hardware calibration.

### Compatibility

- Schema `v1.0.0` checkpoints (157-dimensional RMA) are `legacy-v0` and cannot be resumed, exported, or deployed.
