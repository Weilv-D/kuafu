# STM32 Firmware & Runtime Logic Review — 2026-09-16

This record documents a second full-codebase logic review, layered on the
2026-09-13 record. The 2026-09-13 fixes (boot-deadlock in the supervisor,
INIT-mode wheel lockout, estimation-freshness authority, deferred ST3215
error handling, RAM-top dump exclusion) were re-verified as a baseline and
are assumed throughout. This pass covered the same firmware tree plus the
Pi5 runtime (`pi5_runtime/`), which is the other half of the safety
contract. It is a software record: no hardware was flashed or driven.

## Review method

1. **Architecture pass** — module partition (drivers / buses / health,
   safety machine, startup manager, supervisor, runtime deadlines, control
   section), the mode and phase state machines, the scheduler deadline
   structure (1 kHz fusion, 250 Hz control, 50 Hz legs, 4 ms wheel bus,
   6 ms servo poll), peripheral bring-up order (DMA clocks before UARTs,
   DRDY armed before discovery), and every ISR/main-loop concurrency
   contract.
2. **Detail pass** — boundary conditions, exception and re-arm paths,
   queue/ring/parser lifetimes, interrupt timing, and cross-context state
   consistency, walked scenario by scenario (cold boot, fault entry,
   transient-fault recovery, link loss/recovery, bus degradation, STM32
   reboot, Pi-side stalls).
3. **Counterpart pass (Pi5)** — the Python runtime reviewed against the
   firmware's acceptance rules (heartbeat/action timing, D0 gate, mode
   codes) and the shared sequence-gate semantics.

Every code fix has a regression test or a build gate: the safety-machine
and decoder fixes were verified red/green against the pre-fix source; the
scheduler-only fixes are covered by the Keil zero-warning build plus the
reasoning recorded here (their logic lives in `main.c`, which the host
suite intentionally does not compile).

## Defects found and fixed

Severity is ranked by worst-case consequence on the running robot.

### 1. (High, firmware) Mode-transition grace window could be held open forever

**Files:** `Core/Src/safety_state.c`, regression in `tests/test_safety_state.c`.

Every mode transition — including the command-driven ones (STAND→ACTIVE,
ACTIVE→STAND/CLIMB, CLIMB→STAND/ACTIVE, link-loss demotion) — reset all
four freshness stale counters and re-opened the 100 ms freshness-fault
grace window. A Pi whose `mode_request` flapped at >10 Hz (buggy teleop
layer, button chatter) therefore kept the grace window permanently open:
wheel/servo freshness faults could never latch while the wheels stayed
enabled. IMU loss was already independently covered by the estimation
gate (`imu_control_valid && imu_fresh`), and servo loss by the supervisor
verdict, but wheel-loss had no second guard.

**Fix.** Grace plus counter reset is now granted only by the two
device-driven transitions (INIT→STAND, FAULT→STAND recovery) — the ones
where what the buses are asked to do actually changes. Command-driven
transitions only record the time. Red/green: the new flap test fails on
the pre-fix source (mode stays STAND, no `FAULT_WHEEL_LEFT`) and passes
after.

### 2. (High, firmware) A busy servo bus dropped whole 50 Hz leg-write deadlines

**Files:** `Core/Src/main.c`, `Core/Src/firmware_runtime.c`,
`Core/Inc/firmware_runtime.h`, `tests/test_firmware_runtime.c`.

The 50 Hz servo deadline block cleared `servo_deadline_pending`
unconditionally, then attempted one sync-write (or FAULT disable). If the
ST3215 bus was mid-transaction at that instant the write was refused and
the whole 20 ms period elapsed with no leg command. Five consecutive
misses expired `LEG_HOLD_MAX_AGE_MS` (100 ms), demoted the actuator
supervisor to HOLDING, and closed the wheel gate — i.e. ~100 ms of zero
torque on a balancing robot. With a healthy bus the collision probability
is ~7 % per deadline (negligible), but with one servo timing out ~10 ms
of every 24 ms poll cycle it rose to ~50 %, making the gate closure a
recurring event exactly while the bus was already degraded.

**Fix.** The deadline is now "served" only when a frame was actually
queued, the mode has nothing to transmit (INIT), or the request is
statically unsolvable (IK rejected already-clamped inputs — deterministic,
retry cannot help). A busy bus keeps the deadline pending and the write
is retried on the next scheduler pass — the same retry shape the wheel
dispatch layer always had (`next_wheel_tx_ms` only advances on success).
`firmware_runtime.servo_intent_allowed` became a pure mode-level verdict
(`operational`), symmetric with `wheel_intent_allowed`; sampling the bus
there would have re-introduced the drop-on-busy behavior the retry
removes.

### 3. (Critical, Pi5) Aggressive stick input could silence the heartbeat stream

**File:** `pi5_runtime/teleop_single.py`.

`teleop_single` clamped D0 only to [58, 207] mm while the firmware (and
`command_frames`) additionally rejects D0 > 120 mm whenever |v| > 0.3 m/s
or |ω| > 0.6 rad/s. With the legs raised, pushing the stick fast made
`command_frames` raise on every tick, and the except path sent nothing:
no heartbeat, no action, for as long as the input was held. The firmware
heartbeat watchdog (200 ms) then demoted the robot out of ACTIVE exactly
during aggressive motion — reproducible 100 % of the time.

**Fix.** The dynamic gate is mirrored before encoding (`d0_tx =
min(d0, D0_GATE_MAX_HIGH)` when over threshold — the same clamp
`serial_node.set_command` already applied), and the except path now falls
back to a valid safe heartbeat (STAND, zero commands) instead of silence,
so no bad value can ever stop the heartbeat. Gate-boundary consistency
with the firmware's quantized acceptance was checked in both directions.

### 4. (Major, Pi5) Telemetry sequence gate blackholed the stream after an STM32 reboot

**Files:** `pi5_runtime/protocol.py`, regression in
`rl/train/tests/test_protocol.py`.

The decoder's monotonic sequence gate never reset. On an STM32 reboot the
firmware TX sequence restarts at 0, and every CRC-valid telemetry frame
was then rejected until the new counter climbed past the decoder's stored
high-water mark — seconds to minutes at telemetry rates, with the
rebooted STM32 waiting for commands the Pi refused to send because its
own link check saw no telemetry.

**Fix.** The decoder counts consecutive CRC-valid frames rejected by the
sequence gate; after 8 it concludes the peer restarted, clears the
high-water mark and resynchronizes (the burst frame itself is accepted).
Genuine duplicates never reach the threshold because any accepted frame
clears the streak. Red/green verified against the pre-fix decoder.

### 5. (Major, Pi5) Stale telemetry silenced the policy node's TX entirely

**File:** `pi5_runtime/serial_node.py`.

When no fresh telemetry existed (>100 ms), `tick()` sent nothing at all,
so every transient stall (first ONNX inference, GC pause) made the Pi
itself violate the 50 Hz heartbeat contract and handed control to the
firmware watchdog.

**Fix.** The node now keeps the heartbeat contract with an explicit safe
request (STAND, zero commands, zero residual) whenever policy output is
impossible, and resets the observation history when telemetry resumes so
the causal window never fuses across the outage.

### 6. (Major, Pi5) Safety-critical exit frames were not flushed

**Files:** `pi5_runtime/teleop_single.py`, `pi5_runtime/link_probe.py`.

Both `finally` paths wrote the ESTOP frame and immediately closed the
port; `close()` gives no guarantee that OS-buffered bytes reach the wire,
so the designated explicit-ESTOP path could silently drop its final
frame. Both now `flush()` before closing (the HELLO path already did).

### 7. (Major, Pi5) Blocking BLE reconnect could run while ARMED

**File:** `pi5_runtime/teleop_single.py`.

The idle-recovery path ran `bt_reconnect()` + `joy.reconnect()` inline —
multi-second subprocess/evdev work — without regard to the armed state,
stopping heartbeats while the robot could still be tracking commands.

**Fix.** A reconnect attempt first forces the safe state (disarm, zero
commands, STAND request); the gamepad is idle by definition of the
trigger, so this loses nothing.

### 8. (Minor, Pi5) Link-probe fault table mislabeled every fault class

**File:** `pi5_runtime/link_probe.py`.

`FAULT_BITS` was shifted one bit relative to `safety_state.h` (which
reserves bit 1 for the unused `FAULT_HEARTBEAT`): e.g. firmware
`FAULT_SERVO` (0x10) printed as `WHEEL_L_LOST`. The diagnostic tool was
actively misleading during fault triage. The table now matches the
firmware bit-for-bit, `--mode` is validated (`choices=range(5)`), FAULT
frames are counted and printed (previously silently ignored), HEALTH
payloads are length-guarded, the ESTOP exit flushes, and the 50 Hz loop
re-clamps its deadline after a stall instead of free-running.

### 9. (Minor, firmware) Negative temperature was cast to uint8 unsaturated

**File:** `Core/Src/main.c`.

`(uint8_t)g_imu.temperature` is undefined behavior for the BMI088's
negative range (−40 °C on a cold bench would wrap, e.g. −5 → 251). The
diag field is now saturated to [0, 255].

### 10. (Minor, hygiene) Dead `gyro_calibrated` plumbing removed

**Files:** `Core/Inc/safety_state.h`, `Core/Inc/startup_manager.h`,
`Core/Inc/control_section.h`, `Core/Src/control_section.c`,
`Core/Src/main.c`, `tests/test_safety_state.c`.

`gyro_calibrated` was set by the scheduler and passed through three input
structs but never consumed anywhere — a vestige of the retired
calibration startup gate. It suggested a gating that does not exist; the
field is gone from all three structs (the deliberate "calibration is not
a gate" design is now stated in `startup_manager.h` instead).

### 11. (Minor, Pi5) Assorted teleop robustness

`pi5_runtime/teleop_single.py`: a lost gamepad no longer replays its last
stick values alongside the FAULT request (v/w zeroed on disconnect); the
STM32 status printer and the "still idle" reminder no longer share one
throttle timestamp (they suppressed each other); the documented default
arm/disarm buttons now match the code defaults (btn7/btn6) with the env
override called out; HEALTH frames are length-guarded like every other
frame type.

## Verified-not-broken (selected)

Re-checked in depth during this pass, no change required:

- The deferred ST3215 UART-error application (`error_pending`) cannot
  double-queue against a late TX-complete: HAL's `gState` serializes
  transmits, so a stale TX-complete finds the phase it expects.
- The DDSM enable/mode/torque dispatch order: the FIFO bus queue
  guarantees no torque frame precedes its enable on the wire; enable bits
  are set only after `HAL_UART_Transmit_IT` accepted the frame.
- INIT-phase freshness accounting: transient stale ticks accumulated
  during actuator discovery cannot latch in INIT (only serious faults
  escalate there), and the discovery completes well inside the debounce
  budget.
- The Pi-side D0 gate boundary under i16 quantization stays consistent in
  both directions (the unquantized clamp is always at least as strict as
  the firmware's post-quantization check).
- FAULT→STAND auto-recovery keeps the wheel gate closed through the
  supervisor's re-hold/re-enable sequence even though `wheel_authorized`
  returns immediately.

## Verification evidence

- Host firmware suite: `firmware host tests passed` (build
  `tests/build-host`, MSVC) — including the new grace-flap regression and
  the updated intent-semantics assertions.
- Red/green: `test_safety_state.c` flap test fails against the HEAD
  `safety_state.c` (mode stays 1, no `FAULT_WHEEL_LEFT`), passes fixed;
  `test_decoder_resyncs_after_peer_sequence_restart` fails against the
  HEAD `protocol.py` decoder, passes fixed.
- Target build: Keil MDK `stm32_firmware.uvprojx` — `0 Error(s),
  0 Warning(s)` (Code=31436, RO=524, RW=168, ZI=27448).
- Python suites: 73 passed (`rl/train/tests`, `rl/teleop/tests`,
  `tests/test_balance_trace_offline.py`, `tools/test_accel_calibrate.py`).
  `rl/train/tests/test_tanh_policy.py` cannot collect in this environment
  (`rsl_rl` not installed); untouched by this review.
- SWD symbol table in `docs/HANDOFF.md` re-extracted from the new map:
  all documented addresses unchanged (this round's edits touched only
  function bodies and stack-resident input structs).

## Flash gate

The `.hex` produced by this build is **not yet flashed or motion-tested**.
Before the next powered session: flash, run `kuafu_boot_trace.py` through
INIT→STAND, and confirm the standalone hold. The known-open items in
`docs/HANDOFF.md` §7.6 (steady-state pitch offset, limit-cycle jitter)
are unchanged by this round.
