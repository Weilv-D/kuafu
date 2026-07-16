# STM32 Hardware Calibration

## Servo Coordinate Contract

Each five-bar side has A and B chains. Dwell is `Qx=0`, `D0=58 mm`, where all
four shared-frame joint angles are zero. Increasing `D0` extends the leg and
requires:

```text
A_l < 0    A_r < 0    B_l > 0    B_r > 0
```

Firmware and UART order is `[A_l,A_r,B_l,B_r]`, also represented by legacy slot
names `[LF,RF,LB,RB]`. Do not use the Actor order or infer direction from the
letters `F/B`.

Command and feedback use the inverse mapping:

```text
raw_tick = center[i] + dir[i] * q[i] * 4096/(2*pi)
q[i]     = dir[i] * (raw_tick - center[i]) * 2*pi/4096
```

## Accepted Servo Zero And Direction Mapping

| Index | Joint | ID | Dwell center | Direction | Raw tick when `D0` increases |
|---:|---|---:|---:|---:|---|
| 0 | `A_l` | 1 | 275 | `+1` | decreases |
| 1 | `A_r` | 2 | 1097 | `-1` | increases |
| 2 | `B_l` | 3 | 2809 | `+1` | increases |
| 3 | `B_r` | 4 | 1023 | `-1` | decreases |

The centers were measured from two consistent nine-sample median captures and
the dwell pose held steadily. Direction is defined by the joint signs and tick
changes above. Clockwise/counter-clockwise descriptions are intentionally not
used because mirrored mounting faces reverse the observer's view.

## Wheel Calibration State

Left DDSM315 is ID 1 and right DDSM315 is ID 2. Both return valid 10-byte CRC
frames. `WHEEL_DIR_L` and `WHEEL_DIR_R` remain `+1`; body-frame forward and yaw
signs must be confirmed during the supervised unloaded-wheel motion gate before
ground contact.

## Ordered Motion Gates

1. Lift and secure the robot. Confirm both wheels remain disabled without Pi
   authorization.
2. At reduced servo speed and acceleration, command `Qx=0`, `D0=63 mm` and
   confirm joint signs `[-,-,+,+]` and raw ticks `[down,up,up,down]`.
3. Return to `D0=58 mm` and confirm all joint angles return near zero and raw
   positions return to their calibrated centers.
4. Apply a small positive command to one unloaded wheel at a time. Confirm the
   configured body-frame velocity sign; stop immediately on the wrong direction.
5. Confirm right torque greater than left produces the defined positive yaw.
6. Expand the five-bar sweep gradually toward `D0=207 mm`, monitoring linkage
   closure, current, temperature, and clearance.
7. Perform tethered zero-command balance before flat-ground tracking.

The electronics bring-up record does not claim these motion gates. A failed
direction, thermal, freshness, or mechanical-clearance check returns testing to
the preceding safe gate.

Battery-voltage calibration is not applicable because the sensing input is not
connected.
