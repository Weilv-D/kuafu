# Changelog

## Unreleased

### Control and contract

- Versioned 140-dimensional Actor / 152-dimensional Critic contract at schema `v1.1.0`, with a tanh-squashed policy transform shared by PPO, ONNX export, and the Pi5 runtime.
- Source-generated discrete LQR/LQI firmware constants, dwell-relative five-bar IK generation, and versioned UART framing with CRC-8/MAXIM.
- High-speed D0 gate (120 mm cap when `|v| > 0.3 m/s` or `|w| > 0.6 rad/s`) and roll leveling (`ROLL_KP = 190 mm/rad`, `ROLL_KD = 5.0`).

### Firmware

- STM32F407ZG bare-metal runtime: 1 kHz IMU fusion, 250 Hz LQR/LQI baseline, five-bar workspace projection.
- Self-balancing in STAND mode works standalone (no Pi required): wheel torque authorization is gated on startup completion and absence of fault, independent of Pi link or heartbeat freshness. The Pi link gates ACTIVE motion commands only.
- LQR control runs every 250 Hz deadline regardless of DDSM bus state; the dispatch layer skips transmission when the bus is busy so the controller is never starved.
- DDSM115 speed feedback decoded in 0.1 RPM units matching the command encoding.
- DDSM transaction timeout relaxed to 8 ms and freshness-fault debounce to 5 ticks to accommodate marginal RS485 links without spurious FAULT_WHEEL.
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
