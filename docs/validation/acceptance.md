# Validation And Acceptance

## Software Gates

| Gate | Evidence | Pass condition |
|---|---|---|
| P0 contract | protocol/policy tests | bounded tanh action, matching dimensions, CRC/fragment/replay handling |
| P1 source | generated-artifact check | no generated header diff; schema/hash match |
| P2 control | source and native tests | stable discrete poles; positive force/velocity and yaw direction |
| P3 five-bar | table generator | endpoints 58/207 mm, FK/IK closure, signs, limits |
| P4 environment | JIT reset/step/auto-reset | no tracer error, no metric PyTree mismatch, nonzero command gate |
| P5 checkpoint | resume test | CPU-first load; model/optimizer/LR/entropy/curriculum/RNG/schema/hash restored |
| P6 export | Torch/ONNX parity | maximum action error below `1e-5` |
| P7 HIL | UART and watchdog test | version/hash rejection, partial frames, stale action/heartbeat fallbacks |

## Simulation Stages

Frozen scenarios are run by `rl/verify/scenario_runner.py`, which writes per-episode `episodes.jsonl` and a `release_gate`-compatible `summary.json`.

| Stage | Required result |
|---|---|
| S0 | 20 s survival at least 99%, tilt RMS at most 2 degrees, no saturation failure |
| S1 | every signed command bucket: velocity MAE 0.08-0.10 m/s, yaw MAE at most 0.15 rad/s |
| S2 | D0 MAE at most 5 mm, roll at most 2 degrees, transition settle at most 0.5 s |
| S3 | independent DR/noise/delay degradation at most 10% |
| S4 | each ±2 to ±10 degree slope direction succeeds at least 90% |
| S5 | frozen 30 mm step set Wilson 95% lower bound at least 80%, recover within 2 s |
| S6 | 0.5-2 N·s impulse pushes recover within 2 s at least 90%, exceed native baseline |
| S7 | mixed frozen holdout meets every applicable bucket minimum |

S6 uses impulse-based push sampling (0.5, 1.0, 1.5, 2.0 N·s in both directions) injected at fixed timing; the policy recovery rate must exceed the native LQR/LQI baseline. S3 samples actual friction coefficients in `[0.3, 1.2]`. Curriculum advancement uses independent 7-axis evaluation every 25 updates; two consecutive evaluations are required for advancement, at most one advancement per evaluation, and 20-30% learned scenarios remain active. Frozen holdout seeds never control curriculum advancement.

`rl/verify/release_gate.py` requires the complete S0, S1, S2, S3, S4, S5, S6, and S7 summary sections. Missing sections, non-finite metrics, missing slope/bucket entries, or a push result that does not exceed the native baseline fail closed.

## Hardware Gates

P9 requires the calibration record, table-top verification, tethered low-torque hold, flat tracking, push recovery, D0 movement, and step traversal in that order. No simulation result substitutes for a hardware gate. A failed sign, saturation, watchdog, or temperature check returns the system to the preceding gate.
