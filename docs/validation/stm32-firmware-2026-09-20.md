# STM32 Firmware Logic Review — 2026-09-20

This record documents a full-codebase logic review of the STM32 firmware,
layered on the 2026-09-13 and 2026-09-16 records (both re-verified as
baseline). The work ran across 2026-09-19/20 in three passes — two review
passes, then a closeout pass that resolved the residual items the review had
catalogued — over the same tree:

1. **Architecture pass** — module partition (drivers / buses / health, safety
   machine, startup manager, supervisor, runtime deadlines, control
   section), the mode and phase state machines, the scheduler deadline
   structure (1 kHz fusion, 250 Hz control, 50 Hz legs, 4 ms wheel bus,
   6 ms servo poll), peripheral bring-up order (DMA clocks before UARTs,
   DRDY armed before discovery, `stm32f4xx_hal_msp.c` and
   `stm32f4xx_hal_conf.h` cross-checked against the `MX_*` initialisers),
   and every ISR/main-loop concurrency contract.
2. **Detail pass** — boundary conditions, exception and re-arm paths,
   queue/ring/parser lifetimes, interrupt timing, and cross-context state
   consistency, walked scenario by scenario: cold boot, serious-fault
   latch, transient-fault entry and recovery, link loss/recovery, bus
   degradation, DMA restart, IMU init failure, and both bench calibration
   builds. Subsequent verification passes re-derived the tricky paths from
   scratch — byte-level traces of the ST3215 echo filter against real
   query/reply sequences, DDSM queue priority under FAULT (disable frames
   always enter the queue ahead of the zero-torque telemetry frames because
   the sequencer block precedes the dispatch block in the loop), the
   wheel-dispatch alternation invariant, a full undefined-behaviour audit
   (signed shifts, float→int casts, array bounds, memmove vs memcpy), and
   an NVIC priority audit (SysTick/EXTI1 at 0, DMA and UART IRQs at 1) —
   and confirmed the first-pass fixes introduce no regressions.

It is a software record: no hardware was flashed or driven.

## Verification evidence

- Host suite: `firmware_host_tests` — all suites pass, including two new
  regressions: the control-section transient-fault recovery contract
  (`test_transient_fault_recovery_reruns_enable_sequencer`) and the servo
  write-phase self-heal (`test_st3215.c` lost-TX-complete case), plus the
  offline balance-trace pytest (5 cases).
- Target build: Keil MDK quality gate (`tools/build_keil.ps1`) — 0 errors,
  0 warnings; SWD symbol addresses unchanged against the 2026-09-16 table
  in `docs/HANDOFF.md` (re-checked against the fresh `.map`).

## Defects found and fixed

Severity is ranked by worst-case consequence on the running robot (or, for
the bench builds, on the operator's expectation of them).

### 1. (High) Servo enable sequencer was mode-blind: transient-fault recovery could reopen the wheel gate on physically disabled legs

**Files:** `Core/Src/main.c` (sequencer gate), regression in
`tests/test_control_section.c`.

The fault-entry edge resets `servos_enabled = 0` so the software stops
claiming "verified", and the 50 Hz FAULT branch physically disables servo
torque (one-shot). But the enable sequencer ran on `enable_actuators` alone,
which stays 1 while the startup manager sits in READY — including throughout
FAULT. Consequences, in order of severity:

- During FAULT the sequencer re-sent the four torque-enable frames at 5 ms
  spacing, racing the one-shot disable (one frame per 20 ms deadline): ~80 ms
  of enable/disable churn that directly contradicts the documented FAULT
  contract ("torque disabled (one-shot)").
- The sequencer re-set `servos_enabled = 1` within ~20 ms of fault entry,
  while the disable one-shot was still releasing the physical torque. On a
  transient fault (wheel/servo freshness — the class that auto-recovers),
  the supervisor then cleared its re-issue stage on that racing 0→1
  observation and returned to READY after recovery with **software-verified
  but physically torque-free legs**. The wheel gate reopened; nothing would
  ever re-issue the physical enables (the sequencer believed them done). A
  wheel-feedback glitch followed by recovery — a recurring bench event —
  would drop the robot.

Fix: the sequencer is now gated on `current_mode != STATE_FAULT`. During
FAULT the disable one-shot owns the servo bus and `servos_enabled` stays 0 —
exactly the genuine drop the supervisor's ops-less re-issue stage is designed
to require — and the physical re-issue runs only after the safety machine has
left FAULT. INIT remains allowed because at boot the sequencer must complete
before INIT can exit to STAND (`startup_ready == servos_enabled` feeds the
transition). This restores the behavior the fault-entry-edge comment already
documented ("the servo re-enable path re-runs on recovery") and aligns the
scheduler with the contract the host tests model.

### 2. (Medium) DDSM bus timeout verdict could kill the next transaction (ISR race)

**Files:** `Core/Src/ddsm315.c`.

`ddsm_bus_step` decided "deadline reached" in main-loop context and then
executed the finish without re-validation. Between the check and the finish,
the USART2 completion ISR may legally finish the timed-out transaction and
start the next queued frame (fresh target, fresh 12 ms deadline). The stale
verdict then penalised the **new** target's health, forced the bus IDLE while
the new frame was still mid-transmission, and dropped a queued frame when the
next submit hit a BUSY UART. This is the one main-context phase transition
the header's "IDLE exits only via start_pending" concurrency contract did not
cover. Fix: the phase and deadline are re-validated under the bus IRQ mask
before the finish executes; both re-checks together close the window (a
newly started transaction has a fresh deadline and fails the second check).

### 3. (Medium) Optimistic servo "online" flag opened a boot window with unverified legs

**Files:** `Core/Src/main.c`.

Servo health was initialised with `online = 1` ("optimistic discovery").
`device_health_is_fresh` then reported a servo that had **never replied** as
fresh for the first `SAFETY_SERVO_MAX_AGE_MS` (1 s) after boot. The boot
timeline (500 ms power wait + ~250 ms IMU init + discovery) lands at ~800 ms,
so INIT→STAND and the wheel gate could briefly open (~200 ms) with legs whose
feedback had never been seen, until the freshness debounce latched
FAULT_SERVO. Fix: the optimistic flag is removed; freshness requires a real
first valid frame (the round-robin poll restores online state on the first
reply, so healthy hardware is unaffected — first replies arrive during the
discovery phase's 24 ms poll cycles, well before READY).

### 4. (Medium, bench builds only) Calibration images latched a TRANSIENT fault, so their "never actuate" guarantee evaporated

**Files:** `Core/Src/main.c` (both `#if` blocks).

The two bench-only builds pinned their safety intent on a fault latch:
`DDSM_ID_CALIBRATION_TARGET` triggered `FAULT_WHEEL_LEFT | FAULT_WHEEL_RIGHT`
("all wheel actuation remains disabled") and `SERVO_ZERO_CALIBRATION_MODE`
triggered `FAULT_SERVO` ("a calibration image must never leave INIT"). Both
fault classes are **transient** by design: the safety machine clears them
once the device polls fresh again. Since the startup manager and the
discovery/poll machinery are mode-blind, a calibration image left powered
recovers FAULT→STAND seconds after boot — re-arming wheel authorization (ID
build) and the servo enable path (zeroing build) on images whose entire
contract is "torque-free bench measurement". Production images are unaffected
(both macros default to 0). Fix: both triggers now include `FAULT_INIT` (a
serious, latched-until-reset fault) while keeping the transient bits in the
mask for diagnostic telemetry; the feedback polls keep running and torque
stays physically disabled for the whole session.

### 5. (Medium) `continue` in the 1 kHz tick block silently skipped scheduler stages

**Files:** `Core/Src/main.c`.

The fusion block bailed out with `continue` while the BMI088 init sequence
was in progress. A `continue` at that nesting depth also skips every stage
after the tick block — the 50 Hz leg writer and the servo round-robin poll.
Today those stages are independently guarded (`actuator_configured`, startup
phase) so the skip is unreachable-in-effect, but the control flow couples
their execution to the IMU init state through silence: any future reordering
of the scheduler (or of the startup phases) would break legs/polling with no
compile-time or test-visible symptom. Fix: the `continue` is now an explicit
`if (g_imu.initialized) { ... }` guard around the fusion/telemetry body —
behaviour-identical in every reachable state, structurally safe against
reordering.

### 6. (Medium) A lost TX-complete interrupt could wedge the servo bus until a freshness fault

**Files:** `Core/Src/st3215.c`, `Core/Inc/st3215.h`, regression in
`tests/test_st3215.c`.

Write frames (sync-write, torque) elicit no reply, so their only completion
event is the USART3 TX-complete interrupt. The read phases had a reply
deadline; the write phases had nothing — a lost or wedged TX-complete would
hold `ST_BUS_TX_ONLY` until the 1 s servo freshness fault latched and the
IWDG bounded the underlying storm, containing the failure on the safe side
but taking the whole servo subsystem down for what is, in the common case,
one recoverable frame. A deadline alone is not a complete fix: the HAL
transmit state must be explicitly cleared, or every later transmit attempt
fails against a BUSY UART. Fix: write transactions carry a 20 ms deadline
(a 40-byte sync-write occupies the 1 Mbaud line for ~0.4 ms); on expiry the
bus aborts the transmit explicitly — a no-op when the transfer already
finished — and reopens. A truncated frame is dropped by the servo's own
checksum and a late TX-complete finds the bus idle and is ignored. The same
closeout pass stamps `last_tx.finished_ms` on the DDSM timeout path, the one
diagnostic field its failure branch left stale.

### 7. (Low) `pi_link` used bare `__enable_irq()` in its critical sections

**Files:** `Core/Src/pi_link.c`.

`pi_link_clear_action` / `pi_link_enter_hold` / `pi_link_transmit` masked
interrupts with `__disable_irq()` and unconditionally re-enabled with
`__enable_irq()`. These are main-loop-context today, but the DDSM driver's
documented rule exists precisely because a bare enable inside an ISR unmasked
interrupts mid-handler; keeping two idioms in one firmware is a trap for the
first future ISR-side caller. Fix: all three use the nestable save/restore
PRIMASK idiom (`link_lock_irqs`/`link_unlock_irqs`), matching `ddsm315.c`.

### 8. (Low) `bmi088_read_temp` lacked the handle guard its sibling readers have

**Files:** `Core/Src/bmi088.c`.

`bmi088_read_accel`/`bmi088_read_gyro` reject a NULL `hi2c`; `read_temp`
dereferenced it straight into `HAL_I2C_Mem_Read` (hard fault on a
half-initialised IMU struct). Unreachable through the current call graph (the
tick block only reads after `g_imu.initialized`, which implies `hi2c` set),
but the asymmetry was a latent trap for future callers. Fix: same guard.

### 9. (Low) `queued_mode` recorded the CRC of enable frames as a control mode

**Files:** `Core/Src/ddsm315.c`.

`ddsm_bus_submit` stored `packet[9]` into `queued_mode` for every `0xA0`
frame. For enable frames `packet[9]` is the CRC; only mode frames (whose
sub-command slot `packet[2]` is zero) carry the mode value there. The field
is diagnostic-only today, but a wrong-by-construction diagnostic is a trap
for the first SWD/telemetry consumer. Fix: the mode is recorded only for
genuine mode frames.

### 10. (Low, documentation-of-contract) Dead/ambiguous declarations clarified

**Files:** `Core/Inc/startup_manager.h`, `Core/Src/control_section.c`.

- `STARTUP_GYRO_CALIBRATION` (enum value 2) has been unreachable since gyro
  calibration stopped gating startup. The member is removed and the numeric
  gap documented as deliberate, so `ACTUATOR_DISCOVERY/READY/FAILED` retain
  the values the balance-trace `startup_state` encoding and the SWD dump
  tools were built against. (No behavioural change.)
- The supervisor input `tx_failed` is never wired in the scheduler
  integration. That is deliberate and now documented at the wiring site: the
  equivalent protection is structural (retry-on-busy leg writes,
  `leg_hold_tx_recent` expiry demoting the supervisor, freshness faults
  latching the safety machine), and feeding it from the busy-retry path would
  re-introduce drop-on-busy verdicts the retry design removed.

## Residual design decisions reviewed and kept

- **Wheel freshness rides a 32 ms debounce.** During the debounce window the
  gate stays open on stale wheel feedback by design (freshness is a
  safety-machine input, not a supervisor input); the balance trace marks the
  validity bits so the window is visible post hoc.
- **`leg_posture_safe` is pitch-derived**, not a direct leg-geometry check —
  the documented posture proxy for the supervisor's verdict.
- `firmware_runtime`'s `wheel_busy_cycles`/`servo_busy_cycles` counters have
  no consumer; they remain SWD-visible diagnostics.
