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

Left DDSM315 is ID 1 and right DDSM315 is ID 2. Both returned valid 10-byte CRC
frames during electronics bring-up. Current configuration is `WHEEL_DIR_L=+1`
and `WHEEL_DIR_R=-1`; these are configuration assumptions, not proof of the
present hardware's body-frame forward and yaw signs. Confirm them during the
supervised unloaded-wheel motion gate before ground contact.

Check the complete chain rather than changing an isolated minus sign: physical
forward tilt, reported pitch and pitch rate, controller torque, raw wheel command,
and actual wheel motion. A host sign test proves the software transform only.
The torque/current conversion remains provisional until measured at the wheel
shaft; matching commanded and reported raw current alone does not calibrate Nm.

## Ordered Motion Gates

1. With actuator power isolated, lift and mechanically secure the robot, provide
   an accessible emergency power cut, and verify the supported test/inhibit state
   before restoring actuator power. Standalone firmware may enable wheels without
   Pi authorization; an absent Pi is not a safe inhibit.
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

## Evidence Required Before Balance Acceptance

Record firmware build/trace version, mechanical configuration, IMU mounting and
calibration, battery/supply conditions, wheel/servo IDs, and explicit pass/fail
results for each gate. Do not reuse a historical electronics pass as motion proof.

- With wheel power inhibited, tilt forward/backward and check pitch and pitch-rate
  signs against the body convention; check a stationary pose for offset/drift.
- Collect six stable accelerometer faces to estimate per-axis bias and scale.
  Neutral defaults do not constitute a completed sensor calibration.
- Use unloaded, reduced-output, one-wheel-at-a-time tests to establish raw command
  and encoder directions. Stop on unexpected direction or actuator state.
- Confirm current-loop mode using protocol feedback where available. Measure
  torque with an appropriate supported load/fixture; supply current is not a
  substitute for motor phase current or wheel-shaft torque. Do not run a sustained
  stall test merely to derive a conversion constant.
- For tethered contact testing, retain a valid trace across first failure,
  including real timestamps, mode/fault, actuator state, target/sent/feedback
  torques, IMU validity and feedback ages. An empty buffer or all-zero export is
  not proof of stable operation.
- Test fault/recovery only in a supported fixture. Software recovery must not be
  interpreted as proof that a fallen mechanism can safely re-erect itself.

Battery-voltage calibration is not applicable because the sensing input is not
connected.
