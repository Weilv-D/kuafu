# STM32 Bring-Up Record — 2026-07-16

## Accepted Configuration

The STM32F407ZG runs with the BMI088, four ST3215 servos, and two DDSM315 wheel
motors powered together. DAPLink probe `LU_2022_8888` provides SWD flashing and
read-only runtime inspection. The accepted firmware starts in 1.739 seconds,
reaches `READY`, enters safe `STAND`, and reports no fault. Wheel enable remains
revoked until a compatible Pi explicitly authorizes a mode with a fresh
heartbeat.

## Physical Interfaces

- BMI088: PB8 SCL, PB9 SDA, PB1 gyro DRDY.
- DDSM315 auto-direction RS485 module: PA2 TX to module RX, PA3 RX from module
  TX, common ground. The verified differential polarity is module A to motor B
  and module B to motor A.
- ST3215 Waveshare Bus Servo Adapter A: PB10 to TXD, PB11 to RXD, jumper A,
  common ground, 1 Mbps.
- DDSM315 IDs are left 1 and right 2. Each ID was assigned while only that motor
  was connected to the RS485 bus.
- ST3215 IDs are 1–4 in `[A_l,A_r,B_l,B_r]` order.

## Servo Geometry

The measured dwell pose is `(Qx,D0)=(0,58 mm)` with centers
`{275,1097,2809,1023}` and directions `{+1,-1,+1,-1}`. Increasing `D0` requires
joint signs `[A_l<0,A_r<0,B_l>0,B_r>0]` and raw tick changes
`[decrease,increase,increase,decrease]`. The centers define zero; they are never
used to compensate for an incorrect direction.

## DDSM315 Bus Behavior

The bus uses 115200 baud, 10-byte frames, CRC-8/MAXIM, and one request followed
by one response. A read-only `0x74` query is used whenever wheels are not
authorized. The driver continuously receives bytes, removes exact local echo,
and slides a 10-byte CRC window across shifted data. A malformed candidate does
not prematurely terminate the transaction.

The total bus request rate is 250 Hz. Transactions alternate left and right, so
each wheel is refreshed at 125 Hz. The transaction timeout is 4 ms, matching the
vendor example. Final feedback age remained 0–12 ms for both motors.

## Startup And Safety

Startup waits for power stabilization, initializes BMI088, collects 1000 gyro
samples, configures both wheels disabled, broadcasts ST3215 torque-disable, and
then begins regular feedback polling. Configuration commands are retried while
their bus is busy and the step advances only after a command is accepted.

The earlier uncontrolled wheel motion occurred when startup enabled the wheels
and entered balance control without explicit Pi authorization. The accepted
firmware separates wheel authorization from startup readiness. No Pi
authorization means no wheel enable and no wheel torque command.

Battery voltage sensing is not connected. Diagnostic value zero is the defined
unavailable sentinel and does not create a voltage fault.

## Final Evidence

- Host firmware test suite: 100% passed.
- Keil target build: 0 errors, 0 warnings.
- Final HEX size: 69,523 bytes.
- pyOCD flash: 32,768 bytes erased and 25,600 bytes programmed.
- Startup: `READY` at 1.739 s, safe `STAND`, fault mask `0x00000000`.
- BMI088: initialized, 0–1 ms age, 33.1–33.4°C.
- DDSM315: both online, 0–12 ms age, wheels stationary without Pi authorization.
- ST3215: all four online, 0–20 ms age, 41–43°C.

This completes STM32 firmware and electronics bring-up. Mechanical motion gates
remain ordered and supervised: servo direction motion, wheel body-frame signs,
tethered balance, and the five-bar range sweep.
