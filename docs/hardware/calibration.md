# Hardware Calibration

Hardware calibration is a release prerequisite. Record the robot serial number,
firmware schema, physical-model hash, date, operator, raw measurements, and
measurement uncertainty. Wiring must first match `wiring.md`; firmware pin and
actuator settings live in `stm32_firmware/Core/Inc/pin_config.h`.

## Servo Coordinate Contract

Each side is a planar five-bar with an A-chain pivot at `x=-26 mm`, a B-chain
pivot at `x=+26 mm`, and output point `(Qx,-D0)`. Dwell is `Qx=0`, `D0=58 mm`.
Joint angles are relative to dwell, so all four reported joint positions are
zero at that pose. Increasing `D0` means extending the leg and must satisfy:

```text
qA < 0, dqA/dD0 < 0
qB > 0, dqB/dD0 > 0
```

The firmware servo-array and UART wire order is `[A_l,A_r,B_l,B_r]`, represented
by the legacy slot labels `[LF,RF,LB,RB]`. The Actor order is different:
`[A_l,B_l,A_r,B_r]`. Do not interchange these orders, and do not infer the
geometric chain from the `F/B` letters; `A/B` plus the pivot coordinate is the
authoritative naming.

For servo index `i`, command and feedback use inverse mappings:

```text
raw_tick = center[i] + dir[i] * q[i] * 4096/(2*pi)
q[i]     = dir[i] * (raw_tick - center[i]) * 2*pi/4096
```

Because `dir` is either `+1` or `-1`, the same sign correctly maps both command
and feedback. Clockwise/counter-clockwise wording is deliberately avoided: the
left and right servos are viewed from opposite mounting faces. Raw tick change
and five-bar motion are the unambiguous bench criteria.

## Calibrated Servo Mapping

The dwell centers were measured on 2026-07-16 at `Qx=0`, `D0=58 mm`, using two
consistent nine-sample median captures. The current mapping is:

| Index | Slot / joint | ID | Dwell center | `dir` | Expected raw tick change when `D0` increases |
|---:|---|---:|---:|---:|---|
| 0 | LF / `A_l` | 1 | 275 | `+1` | decreases |
| 1 | RF / `A_r` | 2 | 1097 | `-1` | increases |
| 2 | LB / `B_l` | 3 | 2809 | `+1` | increases |
| 3 | RB / `B_r` | 4 | 1023 | `-1` | decreases |

The centers and dwell hold are physically verified. The `dir` values express
the intended mirrored mounting and are not considered physically accepted until
the reduced-torque direction test below passes.

## Servo Direction Verification

1. Keep wheel-motor power off, support the chassis, and provide immediate servo
   power cutoff. Start at the measured dwell pose.
2. Use reduced torque, speed, and acceleration. Command a small workspace move
   such as `Qx=0`, `D0=63 mm`; do not test by sending arbitrary raw ticks.
3. Both output points must move away from the hip-pivot line without lateral
   skew or linkage binding. The expected raw tick pattern is
   `LF down, RF up, LB up, RB down`.
4. Telemetry in the shared joint frame must show `A_l<0`, `A_r<0`, `B_l>0`,
   and `B_r>0`. Returning to `D0=58 mm` must return all four joint angles to
   approximately zero and all raw ticks to their measured centers.
5. If one actuator moves the wrong way, stop power immediately. Correct only
   that actuator's `SERVO_DIR_INIT` entry; do not change its center to compensate
   for a sign error.
6. Repeat on both sides, then expand the sweep gradually toward `D0=207 mm` while
   checking closure gap, current, temperature, and mechanical clearance.

## Remaining Required Measurements

1. Confirm positive left/right wheel torque produces positive chassis X motion.
2. Confirm right torque greater than left produces positive yaw.
3. Determine BMI088 axis permutation and signs for projected gravity, pitch
   rate, roll rate, and yaw rate.
4. Measure loaded wheel radius and the current/torque/speed curve of each
   DDSM315.
5. Verify ST3215 direction, range, velocity, acceleration, current/torque
   relation, and thermal limit.
6. Sweep the five-bar workspace at low torque and verify `D0=58-207 mm`, Qx
   range, closure gap, and the joint-sign rules above on the physical assembly.
7. Measure mass, COM, pitch inertia, wheel-ground friction, battery sag, and
   telemetry age/jitter.

Physical geometry and controller parameters change only in `kuafu_physics.py`
and require regenerated artifacts. Servo centers and mirror signs are
robot-specific firmware calibration values. Domain-randomization ranges are
centered on measured distributions rather than guessed broad intervals.

## Bring-Up Sequence

1. Firmware protocol loopback with motors disabled.
2. IMU and encoder sign checks.
3. Servo dwell zero and reduced-torque direction verification.
4. Reduced-torque five-bar range sweep.
5. Tethered baseline hold at dwell.
6. Low-speed forward, reverse, and yaw tracking.
7. Residual action enable with watchdog fault injection.
8. Pushes, slopes, then single steps.
