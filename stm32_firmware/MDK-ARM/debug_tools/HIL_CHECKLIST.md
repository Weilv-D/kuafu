# STM32 Firmware HIL Checklist

Record date, operator, firmware commit, HEX SHA-256, power state, and observed result for every gate. Stop immediately on unexpected motion, heat, noise, reset loops, or communication loss.

## Gate 1 — STM32 and IMU only

- Wheels: unpowered. Servo power: off.
- Flash under reset, then run `read_health.py` without `--under-reset`.
- Expect advancing ticks, calibrated gyro, fresh IMU, INIT followed by actuator-discovery failure because actuators are absent, and no motion.

## Gate 2 — Servo communication

- Wheels: unpowered. Robot mechanically supported. Servo adapter and servo power: on.
- Confirm IDs 1–4, calibrated centers `{275,1097,2809,1023}`, and expected raw extension signs `[decrease,increase,increase,decrease]`.
- Expect all four health ages to refresh. Torque direction must be checked at low torque before allowing position motion. Stop on any unexpected direction.

## Gate 3 — Unloaded wheels

- Robot lifted clear of the ground; wheel direction cannot propel the chassis. Servos may remain supported.
- Apply wheel power only after confirming INIT/FAULT commands are zero torque.
- Confirm both wheel health records refresh and body-frame positive command/feedback signs agree. Do not perform a ground balance test here.

## Gate 4 — Pi protocol loopback

- Robot supported; wheels unpowered unless Gate 3 already passed.
- Run `hil_protocol.py` without `--send` first. Then select one scenario at a time with `--send --port COMx`.
- Expect wrong hash/version/CRC/replay to be rejected, stale action to remove only residuals, stale heartbeat to enter hold, and emergency mode to latch FAULT.

## Gate 5 — Tethered ground test

- Requires explicit operator authorization, mechanical tether, clear area, emergency power cut, and successful Gates 1–4.
- Begin in STAND with zero velocity/yaw and no learned residual. Progress to ACTIVE only after sign checks.
- Hardware acceptance remains incomplete until all P9 items in `docs/validation/acceptance.md` pass.
