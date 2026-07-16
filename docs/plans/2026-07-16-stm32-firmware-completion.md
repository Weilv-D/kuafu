# STM32 Firmware Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver deadline-bounded, testable STM32F407 production firmware with complete safety ownership, robust actuator/Pi communication, calibrated servo mapping, deterministic diagnostics, and staged HIL evidence.

**Architecture:** Retain the existing bare-metal scheduler, but separate pure control/safety logic from HAL transports. UART buses become bounded transaction state machines, control consumes coherent timestamped snapshots, startup gates actuator enable, and FAULT remains latched until reset. Existing protocol command semantics remain compatible; additive health telemetry carries extended diagnostics.

**Tech Stack:** C11-compatible production modules compiled by ARMCC 5.06, STM32F4 HAL, CMake + Visual Studio 2019 Build Tools for host-native C tests, Python 3 + pyOCD for DAPLink/HIL, pytest for Pi protocol tests when the project environment is available.

---

## Preconditions

- Preserve the current working-tree calibration and bring-up changes; do not reset or overwrite them.
- Keep wheel-motor power disconnected until the wheel-direction HIL gate.
- Keep `SERVO_ZERO_CALIBRATION_MODE=0` in production builds.
- Treat battery monitoring as not fitted. Send `battery_mv=0`; do not add an ADC or battery fault.
- Do not claim P9 hardware gates that were not physically executed.

### Task 1: Consolidate The Verified Bring-Up Baseline

**Files:**
- Modify: `stm32_firmware/Core/Inc/pin_config.h`
- Modify: `stm32_firmware/Core/Src/main.c`
- Modify: `stm32_firmware/Core/Src/safety_state.c`
- Modify: `stm32_firmware/MDK-ARM/debug_tools/*.py`
- Modify: `docs/hardware/calibration.md`
- Modify: `docs/contracts/interface.md`
- Modify: `stm32_firmware/MDK-ARM/debug_tools/BRINGUP_LOG.md`

**Step 1: Review the existing diff**

Run: `git diff --check && git diff -- stm32_firmware docs/hardware docs/contracts`

Expected: no whitespace errors; centers are `{275,1097,2809,1023}`; calibration mode is `0`; extension direction is consistently documented as raw ticks `[-,+,+,-]`.

**Step 2: Build the exact baseline**

Run: `C:\Keil_v5\UV4\UV4.exe -b stm32_firmware\MDK-ARM\stm32_firmware.uvprojx -j0 -o stm32_firmware\MDK-ARM\baseline_build.log`

Expected: `0 Error(s), 0 Warning(s)`.

**Step 3: Verify debug scripts without hardware mutation**

Run: `python -m py_compile stm32_firmware\MDK-ARM\debug_tools\read_imu_state.py stm32_firmware\MDK-ARM\debug_tools\calib_servo_zero.py`

Expected: exit 0; remove generated `__pycache__` before commit.

**Step 4: Commit the verified baseline**

```bash
git add docs/KUAFU.md docs/contracts/interface.md docs/hardware/calibration.md docs/hardware/wiring.md \
  stm32_firmware/Core stm32_firmware/MDK-ARM/debug_tools
git commit -m "feat(firmware): record calibrated servo bring-up baseline"
```

### Task 2: Add A Host-Native Firmware Test Harness

**Files:**
- Create: `stm32_firmware/tests/CMakeLists.txt`
- Create: `stm32_firmware/tests/hal_stubs/stm32f4xx_hal.h`
- Create: `stm32_firmware/tests/test_support.h`
- Create: `stm32_firmware/tests/test_support.c`
- Create: `stm32_firmware/tests/test_main.c`
- Create: `stm32_firmware/tests/run_host_tests.ps1`

**Step 1: Create a failing smoke test**

```c
static void test_crc8_maxim_known_vector(void) {
    const uint8_t bytes[] = {0x01, 0x02, 0x03, 0x04};
    TEST_EQ_U8(0xF4, crc8_calculate(bytes, sizeof(bytes)));
}
```

**Step 2: Configure the MSVC build**

`CMakeLists.txt` must compile production `crc8.c`, `kinematics.c`, and later pure modules directly, with `hal_stubs` before `Core/Inc` on the include path. Enable `/W4 /WX` and register the executable with CTest.

**Step 3: Run the failing configuration/build**

Run: `powershell -ExecutionPolicy Bypass -File stm32_firmware\tests\run_host_tests.ps1`

Expected: initial failure until the assertion helper and correct known vector are present.

**Step 4: Implement minimal assertion support and correct test vectors**

Provide `TEST_TRUE`, `TEST_EQ_INT`, `TEST_NEAR`, a failure counter, and deterministic fake `HAL_GetTick()` controlled by `test_set_time_ms()`.

**Step 5: Run host tests**

Expected: CMake configures with `Visual Studio 16 2019`, MSVC compiles with warnings-as-errors, and CTest reports 100% passed.

**Step 6: Commit**

```bash
git add stm32_firmware/tests
git commit -m "test(firmware): add host-native C harness"
```

### Task 3: Extract And Test Servo Mapping

**Files:**
- Create: `stm32_firmware/Core/Inc/servo_mapping.h`
- Create: `stm32_firmware/Core/Src/servo_mapping.c`
- Modify: `stm32_firmware/Core/Src/main.c`
- Modify: `stm32_firmware/MDK-ARM/stm32_firmware.uvprojx`
- Create: `stm32_firmware/tests/test_servo_mapping.c`

**Step 1: Write failing mapping tests**

```c
TEST_EQ_INT(275, servo_angle_to_tick(0.0f, 0));
TEST_EQ_INT(1097, servo_angle_to_tick(0.0f, 1));
TEST_TRUE(servo_angle_to_tick(-0.1f, 0) < 275);
TEST_TRUE(servo_angle_to_tick(-0.1f, 1) > 1097);
TEST_TRUE(servo_angle_to_tick(+0.1f, 2) > 2809);
TEST_TRUE(servo_angle_to_tick(+0.1f, 3) < 1023);
TEST_NEAR(-0.1f, servo_tick_to_angle(servo_angle_to_tick(-0.1f, 0), 0), 0.002f);
```

**Step 2: Run tests and confirm unresolved-symbol failure**

**Step 3: Implement the pure API**

```c
int16_t servo_angle_to_tick(float angle_rad, uint8_t index);
float servo_tick_to_angle(uint16_t raw_tick, uint8_t index);
uint8_t servo_tick_is_valid(int32_t raw_tick);
```

Clamp only at the final driver boundary; the mapping API must expose out-of-range results so safety logic can reject unreachable commands rather than silently saturating.

**Step 4: Replace the static mapping functions in `main.c`**

Feedback must use the raw position tick stored by the ST3215 driver, not a driver-relative angle around 2048.

**Step 5: Run host tests and Keil build**

Expected: all mapping tests pass; target build remains warning-free.

**Step 6: Commit**

```bash
git add stm32_firmware/Core stm32_firmware/tests stm32_firmware/MDK-ARM/stm32_firmware.uvprojx
git commit -m "refactor(firmware): isolate servo coordinate mapping"
```

### Task 4: Introduce Timestamped Device Health

**Files:**
- Create: `stm32_firmware/Core/Inc/device_health.h`
- Create: `stm32_firmware/Core/Src/device_health.c`
- Modify: `stm32_firmware/Core/Inc/ddsm315.h`
- Modify: `stm32_firmware/Core/Inc/st3215.h`
- Modify: `stm32_firmware/Core/Inc/bmi088.h`
- Create: `stm32_firmware/tests/test_device_health.c`
- Modify: `stm32_firmware/MDK-ARM/stm32_firmware.uvprojx`

**Step 1: Write failing age/failure tests**

Test unsigned tick wrap, success clearing consecutive failures, failure counters saturating, bounded reuse of last data, and stale transition at exact thresholds.

**Step 2: Define the shared record**

```c
typedef struct {
    uint32_t last_valid_ms;
    uint16_t timeout_count;
    uint16_t checksum_count;
    uint16_t protocol_count;
    uint8_t consecutive_failures;
    uint8_t online;
} DeviceHealth_t;
```

Expose pure functions `device_health_init`, `device_health_mark_valid`, `device_health_mark_failure`, and `device_health_is_fresh`.

**Step 3: Embed health records in IMU, DDSM, and ST3215 state**

Remove duplicated `last_update_ms`, `is_online`, and failure fields only after all call sites use the shared record.

**Step 4: Run host tests and Keil build**

**Step 5: Commit**

```bash
git add stm32_firmware/Core stm32_firmware/tests stm32_firmware/MDK-ARM/stm32_firmware.uvprojx
git commit -m "feat(firmware): add timestamped device health"
```

### Task 5: Redesign The Safety State Machine As Pure Logic

**Files:**
- Modify: `stm32_firmware/Core/Inc/safety_state.h`
- Modify: `stm32_firmware/Core/Src/safety_state.c`
- Modify: `stm32_firmware/Core/Inc/pin_config.h`
- Create: `stm32_firmware/tests/test_safety_state.c`

**Step 1: Write failing transition and fault tests**

Cover INIT→STAND readiness, STAND→ACTIVE compatibility/freshness, stale action fallback, stale heartbeat ACTIVE→STAND, emergency stop, tilt, excessive pitch rate, overtemperature, IMU stale, left/right wheel stale, servo stale, invalid mode, and fault latching across later healthy inputs.

**Step 2: Replace HAL-dependent inputs with a snapshot**

```c
typedef uint32_t FaultMask_t;
typedef struct {
    uint32_t now_ms;
    float pitch_rad;
    float pitch_rate_rads;
    float max_temp_c;
    uint8_t gyro_calibrated;
    uint8_t imu_fresh;
    uint8_t wheel_l_fresh;
    uint8_t wheel_r_fresh;
    uint8_t servos_fresh;
    uint8_t link_compatible;
    uint8_t heartbeat_fresh;
    uint8_t action_fresh;
    uint8_t requested_mode;
} SafetyInputs_t;
```

`safety_state_update(const SafetyInputs_t *)` must contain no HAL calls and return transition/fallback decisions explicitly.

**Step 3: Define independent internal fault bits**

Include tilt, pitch rate, overtemperature, emergency, IMU, wheel-left, wheel-right, servo, initialization, and internal-state faults. Battery faults are absent.

**Step 4: Preserve compatibility**

Map the 32-bit internal mask to the legacy low-byte diagnostic field and primary one-byte fault packet; reserve the full mask for additive health telemetry.

**Step 5: Run exhaustive host tests and target build**

**Step 6: Commit**

```bash
git add stm32_firmware/Core stm32_firmware/tests
git commit -m "feat(firmware): make safety transitions deterministic"
```

### Task 6: Add A Non-Blocking Startup Manager

**Files:**
- Create: `stm32_firmware/Core/Inc/startup_manager.h`
- Create: `stm32_firmware/Core/Src/startup_manager.c`
- Modify: `stm32_firmware/Core/Src/main.c`
- Create: `stm32_firmware/tests/test_startup_manager.c`
- Modify: `stm32_firmware/MDK-ARM/stm32_firmware.uvprojx`

**Step 1: Write failing phase tests**

Test `WAIT_POWER`, `IMU_DISCOVERY`, `GYRO_CALIBRATION`, `ACTUATOR_DISCOVERY`, `READY`, and `FAILED`, including retry deadlines and no actuator-enable output before READY.

**Step 2: Implement a pure phase machine**

Inputs are timestamps and readiness flags; outputs request one bounded hardware action at a time. No `while` retry loop or long `HAL_Delay` is permitted.

**Step 3: Integrate startup into the scheduler**

Replace `System_Initial_Setup()`. Motors receive disable/zero commands during discovery. Servo calibration builds remain torque-free. Production torque enable occurs only after readiness and immediately before entering STAND.

**Step 4: Verify watchdog behavior**

The watchdog is refreshed only after the scheduler completes required high-priority work, not inside infinite initialization loops.

**Step 5: Run tests/build and commit**

```bash
git add stm32_firmware/Core stm32_firmware/tests stm32_firmware/MDK-ARM/stm32_firmware.uvprojx
git commit -m "feat(firmware): gate actuator enable through startup state"
```

### Task 7: Convert DDSM315 To A Bounded Transaction State Machine

**Files:**
- Modify: `stm32_firmware/Core/Inc/ddsm315.h`
- Modify: `stm32_firmware/Core/Src/ddsm315.c`
- Modify: `stm32_firmware/Core/Src/main.c`
- Create: `stm32_firmware/tests/test_ddsm315.c`

**Step 1: Write failing pure frame tests**

Verify torque clamping and encoding, feedback CRC/ID rejection, signed speed/current, position scale, and error-code propagation.

**Step 2: Write failing transaction tests**

Use fake time and callbacks to test TX completion, echo discard, RX completion, 2 ms timeout, alternating left/right slots, and recovery after a valid frame.

**Step 3: Split codec from HAL transport**

Provide pure `ddsm_build_*` and `ddsm_parse_feedback`; wrap them in a `DDSM_Bus_t` state machine driven by `ddsm_bus_step(now_ms)` and UART completion callbacks.

**Step 4: Remove blocking UART calls from the 250 Hz path**

No `HAL_UART_Receive(..., timeout)` may remain in `main.c`.

**Step 5: Run host tests, Keil build, and unloaded-bus HIL smoke**

**Step 6: Commit**

```bash
git add stm32_firmware/Core stm32_firmware/tests
git commit -m "feat(firmware): bound DDSM bus transactions"
```

### Task 8: Convert ST3215 To A Resynchronizing Non-Blocking Bus

**Files:**
- Modify: `stm32_firmware/Core/Inc/st3215.h`
- Modify: `stm32_firmware/Core/Src/st3215.c`
- Modify: `stm32_firmware/Core/Src/main.c`
- Create: `stm32_firmware/tests/test_st3215.c`

**Step 1: Write failing codec/parser tests**

Cover SyncWrite layout/checksum, torque packet, valid feedback, wrong ID, bad checksum, noise before header, partial frame, echoed request, and two adjacent frames.

**Step 2: Store raw position explicitly**

Add `uint16_t position_tick`; derive shared-frame radians only through `servo_mapping`.

**Step 3: Implement one serialized bus owner**

The bus queues one SyncWrite/torque/read operation, receives bytes continuously,
resynchronizes on `FF FF`, ignores known TX echo, and closes read transactions at
their deadline without blocking.

**Step 4: Integrate round-robin feedback and one-shot FAULT disable**

All four servos must remain queryable; persistent failure updates shared health and safety. Do not permanently stop polling an offline servo, so recovery can be observed even though FAULT stays latched.

**Step 5: Run host tests, Keil build, and powered-servo communication smoke**

**Step 6: Commit**

```bash
git add stm32_firmware/Core stm32_firmware/tests
git commit -m "feat(firmware): harden ST3215 stream transport"
```

### Task 9: Make The Pi UART DMA Stream Continuous

**Files:**
- Create: `stm32_firmware/Core/Inc/pi_transport.h`
- Create: `stm32_firmware/Core/Src/pi_transport.c`
- Modify: `stm32_firmware/Core/Src/main.c`
- Modify: `stm32_firmware/Core/Src/pi_link.c`
- Create: `stm32_firmware/tests/test_pi_link.c`
- Modify: `stm32_firmware/MDK-ARM/stm32_firmware.uvprojx`

**Step 1: Port protocol cases into host C tests**

Cover HELLO hash match/mismatch, version rejection, sequence replay and wrap,
fragment retention, concatenated frames, CRC errors, garbage resync, heartbeat
limits, action limits, and separate freshness behavior.

**Step 2: Implement circular-DMA consumption**

DMA remains running. `pi_transport_poll()` compares the last consumed index with
current NDTR, feeds one or two contiguous spans to `pi_link_parse_packet`, and
handles wrap without stopping DMA.

**Step 3: Remove DMA stop/restart from `USART6_IRQHandler`**

IDLE may only trigger a prompt poll; it must not reset the receive buffer.

**Step 4: Run host tests and Keil build**

**Step 5: Commit**

```bash
git add stm32_firmware/Core stm32_firmware/tests stm32_firmware/MDK-ARM/stm32_firmware.uvprojx
git commit -m "feat(firmware): preserve continuous Pi UART DMA"
```

### Task 10: Integrate The Deadline-Bounded Scheduler

**Files:**
- Create: `stm32_firmware/Core/Inc/firmware_runtime.h`
- Create: `stm32_firmware/Core/Src/firmware_runtime.c`
- Modify: `stm32_firmware/Core/Src/main.c`
- Create: `stm32_firmware/tests/test_firmware_runtime.c`
- Modify: `stm32_firmware/MDK-ARM/stm32_firmware.uvprojx`

**Step 1: Write timing/degradation tests**

Simulate skipped ticks, tick wrap, stale snapshots, driver-busy conditions, and
FAULT. Assert one 250 Hz control update per deadline, no unbounded catch-up loop,
zero actuator intent in INIT/FAULT, and residual removal on stale action.

**Step 2: Move policy-independent scheduling into a testable runtime**

The runtime receives sensor/command snapshots and produces wheel/servo intents.
HAL callbacks and byte transports remain adapters in `main.c`.

**Step 3: Ensure coherent snapshots**

Use short interrupt masking or sequence counters around ISR-owned Pi state and
driver feedback. Do not hold interrupts across computation or transmission.

**Step 4: Remove remaining blocking operations from steady state**

Search command: `rg -n "HAL_UART_(Receive|Transmit)\(|HAL_Delay\(" stm32_firmware/Core/Src`

Expected: only explicitly bounded startup/debug exceptions remain; none occur in
the 1 kHz/250 Hz/50 Hz steady-state path.

**Step 5: Run all host tests and target build**

**Step 6: Commit**

```bash
git add stm32_firmware/Core stm32_firmware/tests stm32_firmware/MDK-ARM/stm32_firmware.uvprojx
git commit -m "refactor(firmware): integrate deadline-bounded runtime"
```

### Task 11: Complete Diagnostics And Pi Compatibility

**Files:**
- Modify: `stm32_firmware/Core/Inc/pi_link.h`
- Modify: `stm32_firmware/Core/Src/pi_link.c`
- Modify: `pi5_runtime/protocol.py`
- Modify: `pi5_runtime/serial_node.py`
- Modify: `rl/train/tests/test_protocol.py`
- Modify: `docs/contracts/interface.md`
- Modify: `rl/env/contract.py` only if the executable schema needs an additive diagnostic declaration

**Step 1: Write failing health-frame tests in C and Python**

Define additive telemetry type `0x84` containing a 32-bit fault mask, mode,
reset cause, IMU/wheel/servo ages, and saturated bus error counters. Existing
`0x83` remains four bytes with `battery_mv=0` meaning not fitted.

**Step 2: Implement health serialization and Python decoding**

All multibyte fields stay big-endian. Old decoders may ignore `0x84`; command
frame format and protocol version remain unchanged.

**Step 3: Replace the simulated battery value**

Delete `dummy_battery_mv`. Send zero and document its sentinel meaning.

**Step 4: Run C protocol tests and Python protocol tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 rl/.venv/bin/python -m pytest rl/train/tests/test_protocol.py -q`

Expected: pass in the required project environment. If that environment is
missing, record the exact dependency blocker and do not substitute system Python.

**Step 5: Commit**

```bash
git add stm32_firmware/Core pi5_runtime rl/train/tests/test_protocol.py docs/contracts/interface.md rl/env/contract.py
git commit -m "feat(protocol): add firmware health telemetry"
```

### Task 12: Add Reproducible Build And HIL Tools

**Files:**
- Create: `stm32_firmware/tools/build_keil.ps1`
- Create: `stm32_firmware/MDK-ARM/debug_tools/read_health.py`
- Create: `stm32_firmware/MDK-ARM/debug_tools/hil_protocol.py`
- Create: `stm32_firmware/MDK-ARM/debug_tools/HIL_CHECKLIST.md`
- Modify: `stm32_firmware/README.md`

**Step 1: Write the build wrapper**

The wrapper locates `UV4.exe`, removes only its own prior log, builds the named
target, fails on any warning/error, and prints artifact paths, sizes, and map
addresses.

**Step 2: Add read-only health inspection**

`read_health.py` resolves symbols from the map, uses 100 kHz under-reset only
when explicitly requested, and prints timestamps/faults without hardcoded RAM
addresses.

**Step 3: Add protocol HIL injection**

`hil_protocol.py` accepts a selected serial port and can send valid HELLO,
wrong hash/version, CRC corruption, fragments, replays, stale heartbeat/action,
and emergency-stop cases. It must default to dry-run frame printing unless
`--send` is supplied.

**Step 4: Document safe staged execution**

The checklist separates IMU-only, servo-only, unloaded-wheel, Pi-loopback, and
tethered gates with power state and expected fault behavior.

**Step 5: Run script syntax/help tests and commit**

```bash
git add stm32_firmware/tools stm32_firmware/MDK-ARM/debug_tools stm32_firmware/README.md
git commit -m "tools(firmware): add reproducible build and HIL checks"
```

### Task 13: Final Software Gates And Controlled Flash

**Files:**
- Modify: `docs/validation/acceptance.md`
- Modify: `stm32_firmware/MDK-ARM/debug_tools/BRINGUP_LOG.md`
- Create: `docs/validation/stm32-firmware-2026-07-16.md`

**Step 1: Run the complete host suite**

Run: `powershell -ExecutionPolicy Bypass -File stm32_firmware\tests\run_host_tests.ps1`

Expected: 100% pass, warnings-as-errors.

**Step 2: Run repository-required checks**

```bash
rl/.venv/bin/python rl/verify/verify_physics_source.py
rl/.venv/bin/python rl/verify/verify_controller_golden.py
rl/.venv/bin/python rl/verify/generate_artifacts.py --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 rl/.venv/bin/python -m pytest rl/train/tests -q
```

Expected: all pass in the required environment.

**Step 3: Build the target**

Run: `powershell -ExecutionPolicy Bypass -File stm32_firmware\tools\build_keil.ps1`

Expected: ARMCC `0 Error(s), 0 Warning(s)` and fresh HEX/AXF/map.

**Step 4: Flash with high-power actuators off**

Run: `python -m pyocd load --probe LU_2022_8888 --target stm32f407zgtx --connect under-reset stm32_firmware\MDK-ARM\stm32_firmware\stm32_firmware.hex`

Expected: erase/program success. Do not power wheels during this step.

**Step 5: Execute only the currently authorized HIL stage**

Begin with STM32+IMU. Proceed to servos and unloaded wheels only through explicit
operator power-state confirmations. Stop immediately on unexpected motion.

**Step 6: Record evidence without overstating acceptance**

The validation record lists every command, build hash, measured result, and
unexecuted P9 gate. “Firmware software complete” and “robot hardware accepted”
remain separate statuses.

**Step 7: Final commit**

```bash
git add docs/validation stm32_firmware/MDK-ARM/debug_tools/BRINGUP_LOG.md
git commit -m "docs: record STM32 firmware validation evidence"
```
