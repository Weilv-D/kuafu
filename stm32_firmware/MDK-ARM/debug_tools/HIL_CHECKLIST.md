# STM32 Firmware HIL Checklist

Stop immediately on unexpected motion, heat, noise, resets, or communication
loss. Keep the robot supported until the ordered motion gates explicitly allow
ground contact.

## Electronics Gate

- Flash with DAPLink connect-under-reset.
- Confirm startup phase 4 (`READY`), fault mask zero, and safe `STAND`.
- Confirm BMI088 initialized with age below 20 ms.
- Confirm wheel IDs 1 and 2 are online with age below 50 ms.
- Confirm servo IDs 1–4 are online with age below 250 ms.
- Confirm temperatures remain below 65°C.
- With no compatible Pi heartbeat, confirm both wheels remain disabled and still.

The 2026-07-16 accepted run met this gate: BMI088 0–1 ms, wheels 0–12 ms,
servos 0–20 ms, BMI088 33.1–33.4°C, servos 41–43°C.

## Servo Motion Gate

- Robot supported; wheels remain unauthorized.
- Start from centers `{275,1097,2809,1023}`.
- A small extension must produce joint signs `[-,-,+,+]` and raw tick changes
  `[decrease,increase,increase,decrease]`.
- Return to dwell and verify no binding, skew, or sustained current rise.

## Wheel Motion Gate

- Robot lifted clear of the ground with immediate power cutoff available.
- Establish a compatible Pi link and explicitly request a mode only for the
  supervised test.
- Verify one wheel at a time, then forward and yaw signs together.
- Remove the heartbeat and confirm wheel authorization is revoked.

## Ground Gate

- Requires passed electronics, servo motion, and wheel motion gates.
- Use a tether, clear area, emergency power cutoff, zero learned residual, and
  zero velocity/yaw command.
- Progress from dwell hold to low-speed tracking, disturbances, slopes, and
  steps in the order defined by `docs/validation/acceptance.md`.
