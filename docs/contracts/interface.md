# Interface Contract

`rl/env/contract.py` is the normative executable contract at schema `v1.1.0`. This document is the readable form.

## Coordinate And Sign Rules

| Quantity | Definition |
|---|---|
| chassis frame | `+X` forward, `+Y` left, `+Z` up |
| common torque | `(tau_L + tau_R) / 2` |
| yaw torque | `(tau_R - tau_L) / 2` |
| positive yaw | right wheel torque exceeds left wheel torque |
| five-bar output | `(Qx, -D0)` in the chassis X-Z plane |

All protocol values use SI units except `D0`, which is explicitly millimetres in the command and observation contract.

### Five-Bar Joint Signs And Servo Order

The A-chain pivot is at `x=-26 mm`, the B-chain pivot is at `x=+26 mm`, and
dwell is `(Qx,D0)=(0,58 mm)`. Hip angles are dwell-relative. Increasing `D0`
therefore requires `qA<0` and `qB>0` on both sides.

Firmware control and UART telemetry use hip wire order `[A_l,A_r,B_l,B_r]`.
The Pi runtime reorders it to Actor order `[A_l,B_l,A_r,B_r]`. ST3215 raw tick
direction is a hardware mapping beneath this contract; with the current
`SERVO_DIR_INIT={+1,-1,+1,-1}`, extension must produce raw tick changes
`[decrease,increase,increase,decrease]`. Firmware converts feedback back into
the shared dwell-relative joint signs before transmission.

## Command

| Field | Range | Note |
|---|---|---|
| `vx` | `[-0.5, 0.5]` m/s | forward speed command |
| `wz` | `[-1.0, 1.0]` rad/s | yaw-rate command |
| `d0` | `[58, 207]` mm | target foot drop; limited to 120 mm when `|v| > 0.3` or `|w| > 0.6` |

## Actor Observation

One frame has 35 normalized physical-scale values. Four causal frames are concatenated in chronological order for a 140-dimensional input. The first-frame history on reset is `[0, 0, 0, current]`.

| Field | Dimensions | Runtime source |
|---|---:|---|
| command `vx,wz,D0` | 3 | command arbiter (clamped/gated) |
| projected gravity | 3 | IMU attitude |
| body gyro | 3 | IMU gyro |
| estimated `vx,wz,D0,roll` | 4 | wheel-odometry/IMU/servo estimators |
| wheel speeds | 2 | DDSM feedback |
| hip position | 4 | ST3215 feedback |
| hip velocity | 4 | ST3215 feedback |
| previous applied action | 6 | delayed action actually sent to actuators |
| sensor age | 6 | IMU age (3) + joint age (3) |

Forward velocity and yaw rate are wheel-odometry estimates, not simulation root truth. The `prev_applied_action` field is the delayed action actually sent to actuators, not the raw policy output. The Actor does not receive absolute yaw, root-truth velocity, unlimited wheel angle, MuJoCo contact, or domain-randomization truth.

## Actor Action

```text
[dtau_common, dtau_yaw, dQx_L, dD0_L, dQx_R, dD0_R]
```

Each component is in `[-1, 1]`. The STM32 and MJX map Qx to ±20 mm and D0 to ±30 mm before workspace and rate limits. No action field directly names a servo angle.

## UART Frame

Every frame is big-endian:

```text
A5 | version:u8 | type:u8 | length:u8 | sequence:u16 | timestamp_ms:u32 |
payload | crc8-maxim:u8 | 5A
```

Version is `1`. A `0x03` HELLO payload contains the 16-character lowercase physical model hash and starts a new receive session. Partial DMA chunks are retained, CRC-invalid bytes are resynchronized, and replayed/out-of-order sequences are discarded. A heartbeat payload contains `mode:u8`, `vx:i16/1000`, `wz:i16/1000`, and `D0:i16` millimetres; an action payload contains six `i16/10000` normalized actions. STM32 accepts ACTIVE only after a valid HELLO and fresh heartbeat.

Telemetry uses two frame types: `0x81` IMU payload `[roll,pitch,yaw,gyro_x,gyro_y,gyro_z]`, all `i16/1000`; and `0x82` joints payload wire order `[wheel_L, wheel_R, A_l, A_r, B_l, B_r]`, each group containing `(pos,vel,tau/current)`. Velocity and angular-rate fields use `×1000` and wheel torque uses `×10000`. The Pi runtime reorders hips to Actor order `[A_l,B_l,A_r,B_r]`, derives the 35-value frame, and rejects telemetry older than 100 ms.
