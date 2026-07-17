# Changelog

## Unreleased

### Control and contract

- Versioned 140-dimensional Actor / 152-dimensional Critic contract at schema `v1.1.0`, with a tanh-squashed policy transform shared by PPO, ONNX export, and the Pi5 runtime.
- Source-generated discrete LQR/LQI firmware constants, dwell-relative five-bar IK generation, and versioned UART framing with CRC-8/MAXIM.
- High-speed D0 gate (120 mm cap when `|v| > 0.3 m/s` or `|w| > 0.6 rad/s`) and roll leveling (`ROLL_KP = 190 mm/rad`, `ROLL_KD = 5.0`).

### Firmware

- STM32F407ZG bare-metal runtime: 1 kHz IMU fusion, 250 Hz LQR/LQI baseline, five-bar workspace projection, and a separately-authorized wheel power domain gated by Pi model hash, heartbeat freshness, and explicit mode request.
- DDSM115 speed feedback decoded in 0.1 RPM units matching the command encoding.
- Freshness-fault debounce (3 ticks) with a 100 ms mode-transition grace window; hard faults (tilt, pitch rate, overtemperature) latch immediately.
- USART ORE/NE/FE sub-error diagnostics surfaced in the 46-byte health telemetry payload.

### Teleop

- Two-state arm/disarm safety model: sources start DISARMED (firmware STAND, balance held, no motion commands) and require an explicit arm action before the wheels track commands.
- Stick input shaping (deadzone then square curve) and trigger deadzones; full STAND/ACTIVE/FAULT intent carried end to end over the wire.
- Gamepad hot-plug and haptic feedback on state transitions.

### Runtime and tests

- Pi5 ONNX Actor loop with manifest, schema, and calibration-table digest validation.
- Release-gate validation, atomic schema-aware checkpoints with CPU-first loading.
- Host test suites: firmware C tests (CMake), teleop Python tests, and contract tests.
- Documentation reorganized around architecture, contracts, operations, validation, and hardware calibration.

### Compatibility

- Schema `v1.0.0` checkpoints (157-dimensional RMA) are `legacy-v0` and cannot be resumed, exported, or deployed.
