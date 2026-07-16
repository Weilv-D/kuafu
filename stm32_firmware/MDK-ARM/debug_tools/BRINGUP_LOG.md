# Bring-Up Log — 2026-07-16

Chronological record of findings and decisions from the STM32 bring-up session.
Debug method, SWD addresses, and reliability notes live in `README.md` (this
file is the narrative; that file is the reference). Read both before resuming.

## Bugs Found And Fixed (committed to main)

### BMI088 accelerometer never enabled (commit cae3bb3)
- SWD showed `g_imu.accel` all zeros while gyro was fine.
- Root cause: `bmi088.c` wrote `0x03` to `ACC_PWR_CTRL` (0x7D); datasheet
  requires `0x04` (bit2 = normal/active). The acc stayed inactive, so Mahony
  lost its gravity reference and could not correct pitch/roll.
- Fixed `0x03 → 0x04`. Verified: accel z ≈ +9.8 m/s² level, roll/pitch track tilt.

### USART3 half-duplex vs full-duplex adapter (commit 4cdb77b)
- SWD showed every servo hit `consecutive_failures=3` → `FAULT_SERVO`.
- Root cause: ST3215 servos sit behind a Waveshare Bus Servo Adapter (A) that
  converts the single-wire servo bus to a 2-wire UART, but firmware used
  `HAL_HalfDuplex_Init` on PB10 only (no RX).
- Fixed: `HAL_UART_Init`, enable PB10 (TX) + PB11 (RX). Driver already used the
  full-duplex HAL API. Wiring: jumper at A, same-name PB10→TXD / PB11→RXD.

## Servo Communication — Open Issue

### Confirmed by SWD isolation
- Single id1, 25 s: ~100% success, no crash.
- Single id2, 20 s: ~99% success, no crash.
- All 4 rotated at 50 Hz: **always crashes at ~9 s (~358th query)** — RX fills
  with NE (noise) + garbage, all servos go offline → `FAULT_SERVO`.

### Root cause (high confidence)
The `STATE_FAULT` branch disabled all 4 servo torques **every 50 Hz cycle**. In
full-duplex each TX leaves echo bytes on RX; 4 packets × 50 Hz floods the bus and
desyncs every subsequent `read_state` — a snowball. Single-servo tests never
enter FAULT (force-kept online), which is why they are stable.

Second factor: `System_Initial_Setup` did not refresh the IWDG, so slow UART
blocking during init exceeded the ~512 ms watchdog (PR=0, RLR=4095, LSI 32 kHz)
and reset the chip in a loop — which also made SWD unhalt-able.

### Fixes written (committed, unverified)
- IWDG refresh calls through `System_Initial_Setup`.
- FAULT torque-disable made one-shot (`static fault_torque_disabled`).
- `st3215.c` is at the committed baseline; the experimental header-sync /
  bus-quiet / RX-drain logic was reverted (added blocking, did not survive the
  watchdog loop). Re-add carefully after the two fixes above are confirmed.

Unverified because SWD is unusable with servos powered (noise) and the chip
resets with servos unpowered (adapter pulls PB10/PB11 low, stalling init).

## Servo Zero (dwell) — Not Yet Done

- Dwell = D0 = 58 mm (shortest virtual leg), per `kuafu_physics.py` and
  `kinematics.c`. The only correct zero: sim/firmware/physics define joint
  angles relative to it. At dwell qA=qB=0; extending makes qA<0, qB>0.
- `SERVO_CENTER_INIT` is still `{2048,2048,2048,2048}` — must measure per robot.
- `SERVO_DIR_INIT = {+1,-1,+1,-1}` also needs bench verification.
- ST3215 ship with ID=0; must re-address to 1/2/3/4.

## IMU — Open Issue

At session end `g_system_ticks` was stuck at 1 — PB1 DRDY fired once then
stopped, i.e. the BMI088 gyro stopped producing data-ready. Earlier the IMU was
healthy (accel z≈9.8, calib=1). Likely a loose IMU connection (PB1/VCC)
disturbed while wiring servos.

## Resume Checklist (in this order)

1. **Re-seat all wiring** — IMU (PB8/PB9/PB1/VCC/GND); confirm common GND
   between STM32 / servo supply / DAPLink (continuity beep). Most intermittent
   faults here trace to ground/reference issues.
2. Power only STM32+IMU+DAPLink (servos OFF). Run `read_imu_state.py`: ticks
   advancing, accel z≈9.8. Re-confirms the BMI088 fix and IMU wiring.
3. Flash the working-tree firmware (watchdog + one-shot FAULT) with servos OFF
   via `pyocd load --connect under-reset`.
4. Power servos ON. Verify by **behavior** (no SWD): servos hold steady without
   twitching. That is the real success criterion.
5. If stable, calibrate dwell zero with `calib_servo_zero.py`.
6. Only then re-attempt SWD-based measurement; if SWD still dies with servos on,
   accept behavioral verification + the CDC port.

## Key Lesson

The full-duplex adapter + 1 Mbps + 20 cm unterminated TX/RX pair is marginal.
If servo comms stay flaky after the watchdog/one-shot fixes, the next lever is
**electrical**: shorten/twist the TX/RX pair, add a common-ground bus bar, or
lower the baud rate (re-configure ST3215 via vendor tool + match
`huart3.Init.BaudRate`). Do not pile software workarounds on a noisy physical layer.
