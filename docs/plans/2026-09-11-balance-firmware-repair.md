# Kuafu Balance Firmware Repair Implementation Plan

**Goal:** Repair confirmed firmware safety, actuator recovery, IMU validity and trace defects; align control/model assumptions and deliver independently testable software without claiming hardware balance acceptance.

**Architecture:** Keep standalone STAND balance and the protected 4 ms alternating DDSM bus schedule. Extract only the actuator lifecycle needed for host testing; separate safety, data validity, diagnostics and physical calibration. Preserve existing user edits; do not flash hardware, commit or push.

**Tech Stack:** STM32F407 C/HAL, Keil, CMake/MSVC host tests, Python/MuJoCo/MJX validation.

## Baseline

User changes copied to `C:/WORKSPACE/temp/kuafu_baseline_20260911_165235` (tracked patch, status, untracked files). Existing balance trace is not valid hardware evidence.

## 1. Safety and actuator recovery

- Add regression tests for new serious faults during transient fault recovery, combine current faults before considering recovery, keep serious faults latched.
- Extract actuator supervisor and test repeated faults/recovery, bus busy, TX failures, initial startup and reconnect.
- Revoke stale wheel commands/authorization, recover leg targets and holding before wheel output, reset mode/enable bookkeeping, distinguish queued/TX completed/feedback verified.
- Keep no-Pi STAND; expire motion commands/residuals on Pi loss.

Files: `stm32_firmware/Core/{Src,Inc}/safety_state.*`, `firmware_runtime.*`, `main.c`, new `actuator_supervisor.*`, matching host tests and build registration.

## 2. IMU validity and estimation

- Independent accel/gyro health, no invalid sample fusion/calibration, bounded gyro-only degradation and control revocation on invalid gyro/stale data.
- Finite/magnitude guards and bounded accelerometer correction; timestamp-aware filtering.
- Improve stationary bias collection without reinstating indefinite startup calibration gating.
- Neutral accel calibration configuration and offline calibration tooling; explicit initialization failure diagnostics and bounded startup retry.

Files: `bmi088.*`, `mahony.*`, `startup_manager.*`, `safety_state.*`, `main.c`, respective host tests.

## 3. Trace correctness

- Real timestamps, mode/fault/startup/actuator state, freshness, control and sent command distinctions.
- Sample outside wheel-intent gate; preserve a bounded post-fault window.
- Consistent body coordinates/Nm naming, version/build metadata, RAM budget.
- Correct empty/partial/full/wrapped export and concurrent snapshot validation without halting a balancing CPU.

Files: `balance_trace.*`, `dump_balance_trace.py`, `main.c`, C and offline Python tests.

## 4. Buses and protocols

- Verify ST3215 speed sign/magnitude and units against matching protocol/SDK before changing parsing.
- Correct DDSM command completion/error state and rearming mode handling without inventing ACKs.
- Detect DMA producer overrun for servo/Pi RX, count drops and resynchronize based on actual burst budgets.

Files: `ddsm315.*`, `st3215.*`, `pi_transport.*`, `main.c`, corresponding tests.

## 5. Timing and simulation consistency

- Keep protected bus rate; expose elapsed time/missed deadlines and avoid integrating a long stall as a single nominal dt.
- Model alternating 8 ms wheel updates, phase offset, delayed/held feedback and jitter; test physical-parameter uncertainty.
- Share LQI clamp (conservative 0.05 baseline), final D0 gating and explicitly provisional torque conversion.
- End-to-end controller/body/raw sign tests, regenerated headers from source only, honest policy contract invalidation where necessary.

Files: `firmware_runtime.*`, `lqr_controller.*`, `main.c`, `kuafu_physics.py`, `rl/env/kuafu_mjx_env.py`, generated header, `rl/verify`, host tests.

## 6. Verification and delivery

For each confirmed defect: failing test, minimal implementation, module regression, full suite. Configure/build host tests with Visual Studio 16 2019 in `C:/WORKSPACE/temp/kuafu_host_build`; run CTest. Run trace offline tests, physics/model/controller/baseline and timing verification. Address declared missing Python dependencies only in an isolated environment. Attempt actual Keil build and inspect RAM/link results, reporting toolchain blocks truthfully.

Update hardware calibration/wiring, firmware README and a new validation record, leaving historical validation records intact.

## Acceptance boundaries

Software acceptance requires passing reproducible regressions, honest build results and documented remaining items. IMU physical axes/zero, actual wheel direction, current-loop operation, measured Nm/raw and tethered contact balance remain hardware gates; do not guess calibration constants or claim real balance is verified.
