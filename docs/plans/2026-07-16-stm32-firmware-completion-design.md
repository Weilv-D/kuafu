# STM32 Firmware Completion Design

## Goal And Completion Boundary

Complete the STM32F407 firmware as the robot's safety-critical baseline control
and actuator owner. Completion means that production code contains no simulated
diagnostic values, every sensor and actuator has bounded-time health monitoring,
control and protocol behavior are covered by host tests, the Keil target builds
without warnings, and safe HIL gates can be executed deterministically.

Simulation and software tests cannot replace physical acceptance. Servo and
wheel direction, actuator range, thermal behavior, tethered hold, tracking,
push recovery, D0 movement, and step traversal remain evidence-based hardware
gates under `docs/validation/acceptance.md`.

## Chosen Architecture

Retain the bare-metal superloop and refactor it into deadline-bounded layers.
This preserves the validated HAL, LQR/LQI, kinematics, protocol, Keil project,
and robot-specific servo centers while removing unbounded peripheral work from
the control path. An RTOS rewrite would introduce scheduling and driver-reentry
risk during active hardware bring-up; isolated patches would leave the current
timing and verification gaps unresolved.

The runtime has four layers:

1. A 1 kHz IMU layer consumes data-ready events, updates timestamped sensor
   state, and advances attitude fusion without waiting on actuator buses.
2. A 250 Hz safety-control layer snapshots the newest valid inputs, runs local
   LQR/LQI, and applies deterministic degradation for stale or invalid data.
3. A 50 Hz leg/Pi layer projects Qx/D0 through five-bar IK, schedules servo
   targets and feedback, parses Pi traffic, and publishes telemetry.
4. A low-rate diagnostic layer reports temperatures, bus ages, error counters,
   reset cause, and safety mode.

## Startup And Safety Ownership

Startup proceeds through sensor self-test, stationary gyro calibration,
actuator discovery, and zero-output readiness. Wheel and servo torque are not
enabled merely because initialization began. The system enters STAND only when
required health predicates are satisfied. ACTIVE additionally requires a valid
model-hash HELLO and fresh heartbeat. Learned residuals are always subordinate
to the STM32 baseline.

FAULT is latched until reset. Tilt, rate, overtemperature, IMU loss, persistent
wheel-feedback loss, persistent servo loss, and emergency stop
have independent fault bits. Stale action clears residuals; stale heartbeat
commands zero velocity/yaw and returns ACTIVE/CLIMB to STAND while preserving a
bounded local hold. Initialization failure never transitions silently to STAND.

## Non-Blocking Drivers And Data Flow

USART2 uses a bounded transaction state machine for alternating DDSM315 command
and feedback. USART3 serializes ST3215 SyncWrite, torque control, and round-robin
feedback; its receiver performs header resynchronization and discards adapter
echo without blocking the scheduler. USART6 uses continuous DMA-backed stream
parsing so IDLE handling does not stop and restart reception.

Every source exposes a health record containing `last_valid_ms`, consecutive
timeouts, checksum/CRC failures, protocol failures, and online state. Control
reads coherent snapshots rather than mutable driver structures. Short gaps may
reuse bounded-age feedback; exceeding the configured age produces a documented
degradation or fault.

This robot has no battery-divider hardware and battery monitoring is explicitly
out of scope. The `battery_mv` diagnostic field transmits zero, defined as
unavailable/not fitted. New diagnostic fields are added only through an explicit
protocol/schema update synchronized with the executable contract and Pi runtime;
the v1.1.0 command semantics and hip wire order remain unchanged.

## Servo Coordinate Contract

Firmware and UART hip order is `[A_l,A_r,B_l,B_r]`; Actor order is
`[A_l,B_l,A_r,B_r]`. Dwell is `(Qx,D0)=(0,58 mm)`. Increasing D0 requires both
A joints to become negative and both B joints positive. With calibrated centers
`{275,1097,2809,1023}` and intended direction map `{+1,-1,+1,-1}`, raw ticks
must change `[decrease,increase,increase,decrease]` during extension. Physical
direction acceptance is performed at reduced torque and cannot be inferred
from clockwise/counter-clockwise wording.

## Verification Strategy

Host-native C tests use a small HAL/time substitute to cover CRC, streaming
protocol parsing, sequence wrap, fragments, replay rejection, command limits,
kinematics, servo mapping, safety transitions, and timeouts. Tests use a virtual
clock and deterministic bus input.

Target verification requires a zero-warning Keil build plus artifact, schema,
model-hash, and map consistency checks. HIL proceeds in increasing-risk stages:

1. STM32 + IMU + DAPLink: startup, data-ready rate, reset cause, watchdog, IMU
   loss.
2. Servos powered, wheels off: dwell, directions, feedback age, disconnect,
   thermal fault.
3. Wheels unloaded: left/right sign, zero command, feedback timeout, emergency
   stop.
4. Pi loopback: version/hash rejection, CRC, replay, fragments, heartbeat and
   action freshness.
5. Tethered low-torque hold followed by the remaining P9 sequence.

The repository documents pass evidence and leaves every unexecuted physical
gate explicitly open.
