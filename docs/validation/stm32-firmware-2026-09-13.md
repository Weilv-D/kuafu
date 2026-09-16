# STM32 Firmware Logic Review — 2026-09-13

This record documents a full-codebase logic review of the embedded firmware
(`stm32_firmware/`), the defects found, the fixes applied, and the verification
evidence. It is a **software** record: no hardware was flashed or driven, and
it claims no motion or balance acceptance. It does not supersede the 2026-07-16
electronics bring-up or the 2026-09-11 software repair record; it continues
them.

## Review method

Two passes over the whole firmware tree (`Core/Inc`, `Core/Src`, host tests,
Keil project), plus the cross-layer documents:

1. **Architecture pass** — module partition, mode/state transitions, scheduler
   deadline structure, peripheral initialization order, the
   safety-state/supervisor/runtime/dispatch authorization chain, and each
   module's concurrency contract (main-loop vs ISR context).
2. **Detail pass** — boundary conditions, exception and re-arm paths, resource
   lifetime (queues, rings, parser state), interrupt timing, and cross-context
   state consistency.

Every fix was validated red/green: the new regression test fails against the
pre-fix source (verified with `git stash`) and passes against the fixed source.

## Defects found and fixed

Severity is ranked by worst-case consequence on the running robot.

### 1. (Critical) Cold-boot deadlock in the actuator supervisor's enable-reissue stage

**Files:** `Core/Src/actuator_supervisor.c`, `Core/Inc/actuator_supervisor.h`,
regression in `tests/test_control_section.c`.

`main.c` feeds the supervisor `startup_ready` and `servo_enable_verified` from
the same sequencer variable (`servos_enabled`): they flip 0→1 **together** at
the deadline the servo enable sequence completes. The supervisor's ops-less
re-issue stage requires `enable_verified_observed_low` to be set before
`enable_reissue_needed` may clear — but every entry to STARTUP/HOLDING
unconditionally cleared that latch, and the early-return branches (taken for
the whole pre-READY window) never set it. Result: on a clean power-on the flag
is latched clear, `verified` is high from the first deadline onward,
`enable_reissue_needed` can never clear, the supervisor never reaches READY,
`g_wheel_output_gate` never opens, and the robot cannot self-balance from
power-on. Only an incidental FAULT→recovery cycle (which resets the
sequencer bookkeeping) unblocked it — which is why host-only validation on
2026-09-11 did not surface it: the host test modeled `startup_ready=1` with
`servo_enable_verified` transitioning later, an interleaving the scheduler
never produces.

**Fix.** (a) `enable_verified_observed_low` is latched at the top of every
step whenever the verified input reads low, so boot-time observations on the
early-return paths are recorded. (b) The latch is cleared only when a
revocation **follows an authorized (READY) state** — the case where physical
enables may genuinely have been lost and a stale "verified" must not be
trusted. (c) The READY-phase legs-not-ready demotion no longer clears the
latch: a transient feedback/hold gap is not an enable-lifecycle restart, and
every persistent gap escalates to a latched FAULT in the safety layer, which
is the path that re-arms the requirement.

The stale-verified defense is preserved: a fault that interrupts an
authorized state still demands a fresh drop-and-return of the verified flag
before wheel output resumes (covered by the existing ops-less supervisor
tests).

### 2. (Major) Wheels physically enabled during safety-mode INIT

**Files:** `Core/Src/main.c`; contract text in `docs/architecture/system.md`,
`stm32_firmware/README.md`, `docs/HANDOFF.md`.

The documented mode table keeps INIT locked to read-only DDSM queries ("do not
enable either wheel"), but `wheel_authorized` only required startup READY and
mode != FAULT. In the ~20 ms window between startup READY and the safety
machine's INIT→STAND transition, the firmware enabled both wheel motors and
re-asserted current-loop mode while the safety mode was still INIT. No torque
was dispatched (the output gate stayed closed), but the state contract was
violated and the motors drew closed-loop current while "locked".

**Fix.** `wheel_authorized` now additionally requires an operational mode
(STAND/ACTIVE/CLIMB). Documents aligned. With fix 1 in place the supervisor
boot sequence still resolves (the low-verified observations made during the
INIT window satisfy the re-issue stage).

### 3. (Major) st3215 UART-error handling mutated bus state from interrupt context

**Files:** `Core/Src/st3215.c`, `Core/Inc/st3215.h`, regression in
`tests/test_st3215.c`.

`st3215_bus_on_uart_error` ran in the USART3 error ISR and immediately called
`finish_read_failure`, mutating `bus->target`, `bus->phase`, the frame parser,
and the target's `DeviceHealth_t` counters — all of which the main loop also
reads-modifies-writes while draining the DMA ring and queueing reads. The DDSM
driver disciplines these with a PRIMASK critical section; the st3215 driver had
no protection. An ORE arriving exactly while the main loop was between a
`bus->target != NULL` check and its use could dereference a nulled target, and
health counters could tear.

**Fix.** The ISR now only records `error_pending`; `st3215_bus_step` (main
loop) applies the failure. This mirrors the proven `g_uart3_rx_rearm`
deferral pattern for the same bus. Transaction cleanup latency is unchanged in
practice (the step runs every scheduler pass; the 10 ms reply deadline remains
the backstop).

### 4. (Major) RAM-top forensic dump area not excluded from the linker

**Files:** `MDK-ARM/stm32_firmware.uvprojx`, both `.sct` scatter files.

`STALL_DUMP` (0x2001FFC0, main.c) and `FAULT_DUMP` (0x2001FFF0,
stm32f4xx_it.c) live in the top 64 bytes of SRAM and rely on surviving an
IWDG reset. The Keil memory layout, however, extended IRAM2 to 0x2001FFFF with
no reservation: today ZI usage (~27 KB) places nothing there, but any future
RAM growth could have the linker allocate live data into the dump window —
corrupting either the forensics or the running program, silently.

**Fix.** IRAM2 now ends at 0x2001FFBF (size 0x3FC0) in the project dialog and
both scatter files. The post-fix map confirms `RW_IRAM2 … Max: 0x00003fc0`.

### 5. (Moderate) Stale-attitude torque window on gyro DRDY-line death

**Files:** `Core/Src/control_section.c`, regression in
`tests/test_control_section.c`.

`g_imu_control_valid` is refreshed only inside the DRDY-driven fusion block.
If the EXTI/data-ready path itself dies (wire break, gyro INT misconfig), the
flag freezes at its last value while the published attitude freezes with it;
the LQR continued computing torque on frozen attitude until the debounced
freshness fault latched (~52 ms at the 20 ms age threshold + 8-tick debounce).

**Fix.** The control section's estimation authority is now
`imu_control_valid && imu_fresh`; stale-authority is revoked at the 20 ms
freshness bound. The Mahony 100 ms gyro-only bound remains as defense in
depth, and the freshness fault path is unchanged.

### 6. (Minor) Dead safety logic and stale documents

- `firmware_runtime.clear_motion` was computed every step but consumed by
  nobody (motion clearing is owned by `safety_state`'s `clear_action` /
  `enter_hold`). Removed, with ownership documented in the header.
- `main.c` carried an unused duplicate block of the trace-validity defines.
- Documents aligned with the source: schema references updated to `v1.2.0`
  (`AGENTS.md`, `README.md`, `docs/contracts/interface.md`); the startup chain
  no longer lists the never-entered GYRO_CALIBRATION phase
  (`docs/architecture/system.md`, `docs/HANDOFF.md`); the wheel-domain gating
  text now matches the standalone-STAND design instead of a stale five-condition
  Pi-gated variant; the heartbeat-implies-compatible fallback is documented;
  HANDOFF gating facts corrected (servo freshness 1000 ms, gyro calib window
  1000 samples and non-blocking); CHANGELOG stale numbers corrected (DDSM
  timeout 12 ms, freshness debounce 8 ticks).

## Verification

- **Host tests:** `tests/CMakeLists.txt` suite (MSVC 2019, `/W4 /WX /utf-8`),
  rebuilt clean — all pass, zero warnings. Three new regressions added
  (boot-coupled enable flags; deferred st3215 error; stale-freshness torque
  revocation), each verified to fail against the corresponding pre-fix source.
- **Target build:** `MDK-ARM/stm32_firmware.uvprojx` via UV4 (ARMCC 5.06u7) —
  `0 Error(s), 0 Warning(s)`; map confirms the RAM-top reservation.
- **Red/green evidence:** pre-fix sources restored via `git stash` reproduce
  each regression failure (`test_control_section.c:235` for defect 1;
  `test_st3215.c:143-144` for defect 3; `test_control_section.c:255-256`,
  stale torque −0.079 N·m, for defect 5).

## Remaining hardware gates (unchanged)

IMU axis/zero physical verification, wheel direction, current-loop Nm/raw
calibration, tethered balance, and the five-bar range sweep remain ordered
tests in `docs/hardware/calibration.md` and `docs/validation/acceptance.md`.
The cold-boot deadlock fix (defect 1) in particular should be confirmed on
hardware with a plain power-on-to-STAND run before any balance-tuning work.
