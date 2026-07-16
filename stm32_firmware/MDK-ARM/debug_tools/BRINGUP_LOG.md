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

## Servo Bus State

The Waveshare adapter uses full-duplex USART3 with PB10→TXD, PB11→RXD, jumper A,
and a common ground. IDs 1/2/3/4 were all read successfully in firmware order
`[A_l,A_r,B_l,B_r]`. Two software faults found during isolation are fixed:

- `System_Initial_Setup` refreshes the IWDG during blocking peripheral setup.
- FAULT sends the four torque-disable packets once instead of flooding the
  echoed full-duplex receive path every 20 ms.

The formal firmware was rebuilt, flashed, and behaviorally verified with wheel
power disconnected. All four servos enabled without a visible jump and held the
measured dwell pose steadily. Powered-servo SWD remains electrically marginal:
even at 100 kHz it can lose ACK after successful reads. Behavioral verification
or the DAPLink CDC path is therefore preferred while the servo bus is powered.

## Servo Calibration State

Dwell is `(Qx,D0)=(0,58 mm)`, with `qA=qB=0`. Two consistent nine-sample median
captures produced:

```text
SERVO_CENTER_INIT = {275, 1097, 2809, 1023}  // [A_l,A_r,B_l,B_r]
```

The centers and dwell hold are verified. The intended mirror mapping is
`SERVO_DIR_INIT={+1,-1,+1,-1}`. For increasing D0, the shared joint signs must
be `[A_l<0,A_r<0,B_l>0,B_r>0]` and raw ticks must
`[decrease,increase,increase,decrease]`. This direction mapping still requires the reduced-torque
physical test in `docs/hardware/calibration.md`; center values must not be used
to compensate for a direction error.

## IMU State

After reseating the wiring, PB1 data-ready recovered. Formal-firmware reads
showed `g_system_ticks` advancing at approximately 1038 Hz and acceleration
magnitude near 9.96 m/s². The board was resting on its side, so gravity appeared
primarily on the sensor X axis. The earlier single-interrupt condition is no
longer present.

## Next Bench Gates

1. Keep wheel power off and verify the four servo directions with a small move
   from `D0=58 mm` to `D0=63 mm` at reduced torque, speed, and acceleration.
2. Confirm telemetry signs and raw tick changes against the calibration table.
3. Expand gradually to the five-bar range sweep, monitoring closure, current,
   temperature, and clearance.
4. Improve the servo/SWD electrical path before relying on long powered SWD
   sessions: shorten or twist TX/RX, use a common-ground bus bar, and consider a
   lower servo baud rate configured consistently in the servos and firmware.
