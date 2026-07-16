# Validation And Acceptance

Simulation completion never substitutes for hardware acceptance. STM32 hardware
gates are executed in order, and a failed direction, freshness, thermal, or
safety check returns the system to the preceding gate.

## STM32 Electronics Acceptance

| Gate | Pass condition | 2026-07-16 result |
|---|---|---|
| Build | Host tests pass; Keil has 0 errors and 0 warnings | Passed |
| Flash | DAPLink programs and resets STM32F407ZG | Passed |
| Startup | Phase `READY`, safe `STAND`, fault mask zero | Passed at 1.739 s |
| IMU | BMI088 initialized, age below 20 ms, valid temperature | Passed, 0–1 ms, 33.1–33.4°C |
| Wheels | IDs 1/2 online, each age below 50 ms | Passed, 0–12 ms |
| Servos | IDs 1–4 online, each age below 250 ms | Passed, 0–20 ms, 41–43°C |
| Authorization | No compatible Pi heartbeat means no wheel enable | Passed; wheels remained still |
| Battery input | Unavailable sentinel does not create a fault | Passed; `battery_mv=0` |

The detailed evidence is in `stm32-firmware-2026-07-16.md`.

## Mechanical Motion Acceptance

These gates remain supervised physical tests:

1. reduced-motion servo direction at dwell;
2. unloaded left/right wheel direction and yaw sign;
3. gradual five-bar workspace sweep;
4. tethered zero-command balance;
5. low-speed flat-ground tracking;
6. heartbeat and emergency-stop fault injection;
7. push, slope, and step tests.

The accepted electronics state is the prerequisite for gate 1. It is not a
claim that ground locomotion has passed.
