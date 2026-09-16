# Changelog

## Unreleased

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
- FAULT state streams zero-torque frames to the wheels (not query frames), ensuring motors actually stop.
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
