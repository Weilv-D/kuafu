# STM32 Firmware Logic Review — 2026-09-21 (seventh pass)

This record documents an independent whole-repository review of the firmware
and the Pi5 runtime. Per the task framing the review started from the source
alone: the module partition, the four deadline layers, every state machine
(mode, startup phase, actuator supervisor, bus transactions), all driver
concurrency contracts, and the cross-layer protocol were re-derived before any
prior record was consulted. Earlier records were read only afterwards, to
cross-check conclusions; the two of their decisions that this pass re-examined
(CLIMB reachable only through ACTIVE, the heartbeat-fallback link
compatibility) were re-derived and kept.

It is a software record: no hardware was flashed or driven.

## Scope and method

1. **Architecture pass** — module partition (drivers / buses / health, safety
   machine, startup manager, actuator supervisor, runtime intents, control
   section, trace), the mode/phase/supervisor state machines and their
   interlocking gates, the four-layer deadline structure (1 kHz DRDY-gated
   fusion, 250 Hz wall-clock control, 50 Hz retry-on-busy legs, 4 ms wheel
   slots + 6 ms servo poll), peripheral bring-up order, and every
   ISR/main-loop concurrency contract.
2. **Detail pass** — boundary conditions (ring indices, NDTR reload samples,
   queue overflow, saturating counters, float→int casts, fusion-dt and
   elapsed-dt clamps), parser resynchronization, fault entry/recovery/re-issue,
   gate dispatch under every mode, both bench calibration builds, tick wrap,
   and the IWDG budget against every blocking path.
3. **Contract pass** — `pi_link`/`pi_transport` against the Pi-side
   `protocol.py`/`serial_node.py`/`teleop_single.py`/`runtime.py` (framing,
   CRC coverage, sequence gates, heartbeat/action validation ranges, health
   payload layout, joint wire order vs Actor order), plus the workspace
   solvability of every IK input the scheduler can produce.

## Defects found and fixed

Three findings, all closed with regressions. Severity is ranked by worst-case
consequence on the running robot.

### 1. (High) The IMU aggregate health depended on two independently sampled tick values — a phase-locked main loop could starve it into a latched `FAULT_IMU`

**Files:** `Core/Inc/bmi088.h`, `Core/Src/bmi088.c`, `Core/Src/main.c`,
regression in `tests/test_bmi088.c`.

`bmi_try_mark_pair_valid()` refreshes the aggregate `health` only when the
accel and gyro samples of the same fusion cycle both validate, which is the
correct semantics (one dead channel must never be masked by the other). Each
reader, however, stamped its own `HAL_GetTick()`: the pair matched only when
the two I2C transactions happened to fall inside the same millisecond. Every
other millisecond, the pair check failed and the aggregate health was not
refreshed even though both channels had just validated.

That alone would be a missed refresh. The failure mode is worse than a missed
refresh: if the main-loop period is an integer number of milliseconds, the loop
is phase-locked to the tick, and the accel/gyro pair lands on the same side of
the same millisecond boundary on every iteration — the aggregate health then
never refreshes again. `imu_fresh` goes false, the 8-tick/32 ms debounce
latches `FAULT_IMU` (a serious fault: power-cycle only), and the robot is
bricked while both sensor channels are demonstrably healthy. The trigger is not
an exotic environment either: it needs only a loop period that is a whole
number of milliseconds, which any deadline-bounded scheduler can settle into.

**Fix:** the pair is now identified by the timestamp the caller passes to both
channels of one fusion cycle. `bmi088_read_accel()`/`bmi088_read_gyro()` take
`now_ms`, and `bmi088_read_pair()` reads both channels under a single instant
captured before the I2C burst. The aggregate health therefore refreshes exactly
when both channels of that cycle validate — the "only a real pair counts"
semantics is preserved and the tick-boundary dependency is gone. `main.c`'s
fusion block calls the pair API.

**Regression:** the shared-stamp contract is tested by advancing the mocked
tick between the two channel reads (the interleaving that used to break the
pair) and asserting the aggregate health still refreshes; a failed gyro read
still leaves it unrefreshed; the per-channel independence assertions are kept.

### 2. (Medium) `BMI_INIT_GYRO_MAP` aborted the init round on a retryable failure, unlike every other init state

**Files:** `Core/Src/bmi088.c`, regression in `tests/test_bmi088.c`.

The final init write (gyro interrupt map) returned `-1` on a retryable I2C
failure instead of following the shared `bmi_init_write()` convention, under
which a retryable failure restarts the sequence with a fresh deadline (`0`)
and only the exhausted 5-attempt budget fails the round (`-1`). Consequence: a
single hiccup on the last write of an otherwise healthy init wasted the whole
round and forced the startup manager to re-issue the full attempt budget —
five extra IMU discovery rounds of startup latency for a transient NACK, and
an inconsistent failure contract inside one state machine.

**Fix:** the state returns the `bmi_init_write()` result directly, matching
every sibling state. A regression drives the twelve preceding operations,
fails exactly the last write, and asserts the round restarts within the same
attempt budget (`init_attempts == 1`, `initialized` reached).

### 3. (Medium) A Pi chunk larger than the decoder's holding buffer was dropped whole instead of parsed in slices

**Files:** `Core/Src/pi_link.c`, regression in `tests/test_pi_link.c`.

`rx_stream` is 152 bytes while `pi_transport` can deliver up to 256 bytes in
one call (a whole ring after a main-loop stall lapped the parser). The old
append path treated "chunk larger than the free space" as lost framing: it
cleared the stream and, when the chunk exceeded the buffer entirely, returned
without parsing — silently discarding up to ~6 fresh command frames that the
Pi had already sent and considered delivered.

**Fix:** `pi_link_parse_packet()` now feeds the stream in bounded slices. Each
slice fills the free space, the decoder drains every complete frame it
completes and retains only the trailing fragment, and the next slice continues
behind it. Progress is unconditional (the retained fragment is bounded by the
frame-length validation, so the free space never reaches zero), and the only
remaining drop — a fragment that can never complete inside the buffer — is
unreachable for well-formed frames. Overrun accounting in `pi_transport` is
unchanged.

**Regression:** ten heartbeats (190 bytes, larger than the buffer) fed as one
chunk all decode, the newest velocity command is applied, and the heartbeat
stays fresh.

## Reviewed and kept (bounded residuals and deliberate decisions)

Derived end-to-end this pass and either proven correct or accepted with a
bounded worst case; recorded so the next review does not re-litigate without
new evidence.

- **A transient-fault entry does not need an LQR re-anchor.** During `FAULT`
  the wheel output gate is closed, so `lqr_update_elapsed_dt()` never runs and
  `x_est` is frozen — there is no phantom odometry to re-anchor away, unlike
  the link-loss path where the controller keeps integrating. Re-anchoring at
  fault entry would be a no-op at best and a hidden reference jump at worst.
- **A CLIMB request from STAND is ignored.** Re-derived and kept: `CLIMB` is
  reachable only through the `ACTIVE` arming step (documented in the operating
  modes table), and the Pi never requests it from STAND. Not a state-consistency
  gap, because the firmware's telemetry reports the actual mode back and every
  Pi-side program already treats STAND as the safe fallback.
- **An INIT request from a walking mode is honoured as STAND** (new, this
  pass). `INIT` is a boot state with no inbound transition, so a sender asking
  for it while the robot moves is asking it to stop; honouring it as STAND
  keeps the mode consistent on both sides of the link. Documented in
  `docs/architecture/system.md`.
- **The heartbeat fallback may set `link_compatible`** — the deliberate
  "first fully validated heartbeat" rule recorded in the interface contract
  (the Pi does not repeat HELLO after a firmware reset). Not a defect.
- **`servos_enabled` clearing during FAULT** is the documented drop-and-return
  that the supervisor's re-issue stage keys on, not a lost enable.
- **The actuator supervisor's LATCHED state is terminal by design**, matching
  the serious-fault power-cycle latch.
- **`Device_Age_Ms` reporting `UINT16_MAX` for a device validated exactly at
  tick 0** is cosmetic: the same path reports age 0 as fresh, and no device
  validates at tick 0 in the boot sequence.
- **`bmi088_read_temp()` marking the aggregate health on failure** is the
  conservative reading of "no thermal data": reaching its 3-failure threshold
  requires the temp register specifically to fail while every accel/gyro pair
  succeeds, and the safe response to losing thermal telemetry on a balancing
  robot is to stop.
- Re-derived clean, among others: the boot→INIT→STAND arming chain and the
  supervisor's observed-low latch under every entry path; the FAULT-gated
  enable sequencer; the grace-window suppression (device-driven only); the
  masked DDSM timeout re-validation; the ST3215 echo filter; ring/transport lap
  accounting across re-arm in both directions; the wheel-dispatch alternation
  and its sent-torque bookkeeping; the zero-torque gate-closed policy
  including the one-frame arming edge; the 50 Hz retry-on-busy leg writer;
  LQR/LQI clamps, anti-windup, airborne leaky bucket and speed brake; Mahony
  per-channel validity and the gyro-only bound; the IK workspace proof for
  every reachable (qx, d0) input pair; the DDSM discovery queue burst (FIFO
  ordered, self-limiting, step-retried rather than frame-dropped); and the
  cross-layer field order `[wheel_L, wheel_R, A_l, A_r, B_l, B_r]` ↔ Actor
  order `[A_l, B_l, A_r, B_r]`.

## Verification evidence

- Host C tests: rebuilt from scratch with MSVC (`/W4 /WX /utf-8`, NMake
  Makefiles) in a clean build directory — all suites pass, including five new
  regressions (shared-stamp pair, failed-channel non-refresh, tick-straddling
  pair, GYRO_MAP retry contract, oversized-chunk decode) and the extended
  mode-request coverage in `test_safety_state.c`.
- Target build: not executed in this environment (no Keil UV4 and no ARM
  toolchain on the review host). The changes touch no linker script, no
  startup file, and add no globals: `bmi088_read_pair` is the only new
  function and the sequence gate state byte is unchanged, so the RW/ZI image
  and the SWD address contract in `docs/HANDOFF.md` §5.2 are expected to be
  identical. A target build remains required before flashing.
- Pi-side suites: unchanged this pass (no Python-side change); the prior
  67-pytest plus 5-case balance-trace evidence still applies.
