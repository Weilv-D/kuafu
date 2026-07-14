# KUAFU Physical Model

## Authority

`kuafu_physics.py` is the executable authority for physical constants, timing, controller synthesis, five-bar geometry, and the model hash. This document explains those values; it does not create a second set of tunable constants. `rl/verify/generate_artifacts.py` verifies the firmware header generated from that authority.

## Frames And Units

The chassis frame has `+X` forward, `+Y` left, and `+Z` upward. Angles are radians, lengths are metres at layer boundaries, and the five-bar helper API uses millimetres for `Qx` and `D0`. Wheel torque uses N m. The complete cross-layer definition is `docs/contracts/interface.md`.

## Mechanical Model

Each leg is a planar two-actuator five-bar in the X-Z plane. Hip pivots are at `(-26, 0)` and `(26, 0)` mm. Cranks are 93 mm and links are 149 mm. The output target is `(Qx, -D0)`, with D0 ranging from 58 mm at dwell to 207 mm at maximum extension.

The actuator command is measured relative to the dwell posture, not as an absolute geometric angle. The circle-intersection representation crosses the `pi` branch over the workspace; deployment commands use a wrapped difference relative to dwell. Therefore the required monotonic properties are:

```text
dqA / dD0 < 0
dqB / dD0 > 0
```

`rl/verify/calibrate_fivebar.py` regenerates a 256-point dwell-relative table and refuses an output that violates endpoint, continuity, FK/IK closure, limit, or sign guards. Servo electrical zero and mirror direction are separate physical calibration values.

## Timing

| Layer | Period | Rate |
|---|---:|---:|
| MuJoCo physics | 2 ms | 500 Hz |
| baseline control | 4 ms | 250 Hz |
| policy residual | 20 ms | 50 Hz |

`BASE_DT`, `PHYS_DT`, and `RL_DT` are defined in `kuafu_physics.py`. A controller parameter, mass, COM, inertia, wheel radius, or control period change requires regenerated gains and artifacts.

## Baseline Dynamics

The baseline controller uses the discrete ZOH model with state error:

```text
e = [x - x_ref, pitch, vx - v_ref, pitch_rate]
F = -K e - Ki integral(x - x_ref)
```

The reference generator limits acceleration and jerk. `x_ref` integrates `v_ref`; after a zero command reaches zero, it remains fixed so the base layer holds position. Heading uses the same rule for `yaw_ref` and `w_ref`. Wheel torque is decomposed as:

```text
tau_pitch = (tau_L + tau_R) / 2
```

Thus right torque greater than left torque is positive yaw, or a left turn.

## Model Limits

The DDSM wheel command is clipped by stall torque and back-EMF availability. Hip targets are projected through the five-bar workspace and actuator limits. The XML keeps capsule wheels because MJX does not implement cylinder-box collision. This is a deliberate collision approximation whose effective axial support is wider than the measured 34.8 mm tire; contact behaviour must be calibrated before P9.
