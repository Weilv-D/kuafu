# STM32 Firmware Software Repair Validation — 2026-09-11

This record covers the post-audit software repair sprint only. It supersedes
none of the 2026-07-16 electronics bring-up claims and, critically, it does
**not** claim any motion or balance acceptance: no hardware was flashed or
driven during this work. All results below are host simulations, host unit
tests, target compilation, and offline checks.

## Scope: defects addressed

0. **Deep-review catch (fixed after the first integration pass):** the
   supervisor module and its host tests were correct, but `main.c` still fed
   it stale bookkeeping — `servos_enabled` was never reset on FAULT entry, the
   supervisor was invoked with `ops=NULL` (its re-enable request had no
   executor), `fault_servo_disable_idx` exhausted after the first fault, and
   `g_wheel_mode_sent` suppressed current-loop re-assertion. On the real
   device a recovered transient fault could still leave limp legs under live
   wheel control. Fix: FAULT-entry edge resets the servo re-enable sequence,
   the disable one-shot, and the wheel-mode bookkeeping; the supervisor's
   "leg target transmitted" input now requires a real leg-position sync-write
   on the wire within the last 100 ms (`g_leg_hold_tx_ms`), not merely queued
   torque-enable frames.

1. Safety state machine dropped a NEW serious fault (e.g. tilt) that appeared in
   the same 250 Hz update where an old transient wheel/servo fault recovered,
   briefly returning STAND with a cleared mask. Faults are now merged before
   recovery is considered; serious faults stay latched.
2. After a transient FAULT the four ST3215 servos were torque-disabled once but
   never re-enabled on recovery, while wheels could resume — leaving the legs
   limp under live wheel control. The `actuator_supervisor` is now the final
   output verdict inside a single 250 Hz control section (safety -> verdict ->
   gate -> LQR -> trace -> dispatch): its `wheel_output_allowed` verdict, the
   runtime intent, and completed current-loop re-assertion are combined into
   `g_wheel_output_gate`, held between deadlines and obeyed by both the LQR
   computation and the bus dispatch. An integration review of the first pass
   found the verdict applied only to a per-iteration local copy that the next
   loop iteration overwrote (the denial survived ~1 ms, and roughly half the
   torque frames still reached the bus while the supervisor denied); the
   restructured control section removes the deadline/DRDY phase misalignment
   that made the override possible, and moving dispatch after the control
   section means no frame on the wire predates the verdict that authorises
   it.
3. A failed accelerometer read could be masked by a successful gyro read (one
   I2C aggregate health). BMI088 now tracks per-channel health/validity; fusion
   rejects invalid gyro immediately (control revoked) and bounds gyro-only
   propagation to 100 ms; NaN/Inf and norm/innovation guards added.
4. Startup no longer ignores IMU channel validity, reports a failure reason,
   and latches after a 15 s total budget instead of retrying invisibly.
5. Balance trace: fake row-indexed timestamps, wheel-intent-only recording,
   uint8 fault truncation, partial-ring export corruption, and an SWD dump that
   reset/resumed the target are all replaced (trace v2: real timestamps, real
   state enums, 32-bit fault mask, target/sent/feedback split, seqlock snapshot,
   attach-only dump with provenance sidecar).
6. DDSM115/315 bus: busy commands are queued (depth 4) with overflow counts;
   queued/TX-complete/feedback status is observable; UART error path no longer
   leaves the enable mask inconsistent. ST3215 speed decodes as sign-magnitude
   (bit15 direction) per the ST serial protocol — the old two's-complement cast
   misread slow reverse speeds by ~32768x.
7. Servo/Pi DMA reception reconstructs the producer as
   `lap_count * size + write_index`, with the DMA transfer-complete interrupt
   as the lap truth source (the lap count is read before NDTR — an ordering
   under which every interrupt interleaving is exact or a bounded undercount,
   never an overcount, so a sampling race cannot fabricate an overrun). A
   consumer lag past one full ring drops exactly the overwritten prefix and
   counts it; previously a main-loop stall beyond 640 µs overwrote the ring
   silently. Both post-error re-arm paths reset their consumer state so
   pre-restart bytes are never re-parsed, and the frame-level CRC plus the
   monotonic sequence check bound the worst case to one dropped frame.
8. Control/model consistency: LQI clamp unified at ±0.05 (conservative
   baseline), ACTIVE-mode final D0 gating order matched with MJX, torque
   conversion made an explicit single provisional source, and the controller
   integrates measured elapsed time. A new bounded timing verification models
   the real 8 ms per-wheel refresh with 4 ms phase offset and held feedback.

## Verification evidence (all on this workspace, no hardware)

- Host firmware tests: `firmware_host_tests` — all suites pass
  (CMake + MSVC, `run_host_tests.ps1`), including suites for the actuator
  supervisor, Mahony validation, the lap-based DMA ring (exact/undercount/
  never-overcount interleavings, capacity and loss accounting), Pi transport
  overrun/reset, the DDSM queued-control-frame path (an enable queued behind
  an in-flight torque keeps the active reply window open), the
  ACTIVE-only velocity-command contract, and trace.  The control section
  itself is now a host-tested module (`control_section`): its integration
  suite reproduces the scheduler interleaving and covers gate-open
  sequencing, sustained supervisor denial (zero torque at every deadline),
  STAND-vs-ACTIVE velocity semantics, the one-pulse FAULT entry edge,
  estimation-loss tracing, and the pitch-fade envelope.  Extracting it also
  surfaced and fixed an ops-less supervisor defect: with ops=NULL the
  re-issue flag could never clear, structurally disabling wheel output; the
  verified flag must now drop and return on a later step (physical enables
  do not survive a fault/restart), covered in both the supervisor and
  integration suites.
- Regression for defect 1 reproduced before the fix (standalone harness) and
  now covered by `test_safety_state.c` (transient-recovery merge case).
- Target build: Keil UV4 — 0 errors, 0 warnings; RAM RW+ZI 27.6 kB (trace
  ring 20.5 kB documented budget), artifacts and map symbols verified, SWD
  debug-script address tables re-extracted from the new map (including the
  new `g_wheel_output_gate`, `g_st3215_ring`, `g_pi_transport` symbols).
- BMI088 `ACC_CONF=0xAC` resolved against the authoritative bitfield order
  in Bosch's BMI08x_SensorAPI (`bmi08a.c`: bandwidth occupies the high
  nibble, ODR the low nibble): 0xAC is bandwidth NORMAL + ODR 1600 Hz, both
  valid codes, and the 1600 Hz sensor rate covers the 1 kHz poll, so the
  per-channel freshness stamp is honest. The earlier "verify on bench" note
  stemmed from a reversed-nibble reading of the register and is withdrawn.
- Python: `verify_physics_source` 28/28, `verify_model` 11/11,
  `verify_controller_golden` PASS, `verify_control_timing` 27 bounded cases
  PASS, trace offline decoder tests and accel six-face calibration tool tests
  PASS, train/teleop suites 65/65 (isolated venv, `rsl-rl-lib==2.3.3` per
  `rl/requirements.txt`).

## Contract and policy impact

`rl/env/contract.py` SCHEMA_VERSION is now `v1.2.0`: timing/sign/torque-gating
changes invalidate `v1.1.0` policies. `policy.onnx.manifest.json` carries the
invalidation metadata. ACTIVE deployment needs a retrained/revalidated policy.
Standalone STAND balance is unaffected.

## Explicitly NOT done / remaining hardware gates

The robot has still never demonstrated balance under this (or any) firmware.
Before any tethered-contact attempt, in a secured fixture with an emergency
power cut, per `docs/hardware/calibration.md`:

1. IMU mounting axes/zero and pitch/pitch-rate sign vs. body convention.
2. Six-face accelerometer bias/scale capture (tool: `stm32_firmware/tools/
   accel_calibrate.py`; neutral defaults are not a calibration).
3. Unloaded, reduced-output, one-wheel-at-a-time direction checks for
   `WHEEL_DIR_L/R` and yaw sign.
4. Current-loop mode confirmation and measured N·m/raw at the wheel shaft
   (`DDSM_TORQUE_TO_RAW` remains provisional).
5. A valid post-fault trace capture via the attach-only dump tool.

Wiring/calibration docs were updated to state that standalone STAND may enable
wheels without a Pi; never treat Pi absence as an inhibit.
