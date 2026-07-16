# STM32 Bring-Up Log (2026-07-16)

A chronological record of the hardware bring-up session: toolchain setup, bugs
found and fixed, and open issues. Read this before resuming servo/IMU debugging.

## Environment

- Target: STM32F407ZG, Keil MDK-ARM 5.36 (ARMCC 5.06) at `C:\Keil_v5`
- Debugger: DAPLink (jixin.pro CMSIS-DAP), USB VID_C251/PID_F001, ID `LU_2022_8888`
  - Enumerates as COM9 (CDC) + HID. Flash/debug go through the HID (SWD).
- Host tools: pyOCD 0.45 (`pip install pyocd`), pack `pyocd pack install stm32f407zgtx`
- SWD wiring: `SWCLK→CLK`, `SWDIO→DIO`, `3V3`, `GND`, `RST` (board header labeled
  `GND 3V3 RST DIO CLK`; the `RXD/TXD` next to it are UART, not SWD).

## Debug Method

Non-intrusive SWD memory reads via pyOCD (no firmware changes, no UART needed).
Symbol addresses come from `stm32_firmware.map`; struct field offsets from
`fromelf --fieldoffsets` (Keil uses `-fshort-enums`, so enums are 1 byte).

Key globals read this way:
- `g_system_ticks` (0x20000000) — PB1/IMU-DRDY interrupt count; the whole
  scheduler heartbeat. If it stops advancing, the main loop is not running.
- `g_imu` (0x20000218) — accel[3]@4, gyro[3]@0x10, temp@0x1c (BMI088_t, 0x20)
- `g_mahony` (0x20000238) — roll@0x24, pitch@0x28, yaw@0x2c (0x30)
- `g_safety_state` (0x200003d8) — mode@0, fault@1, err@8, gyro_offset@0xc,
  is_calib@0x18 (0x1c, short-enums)
- `g_servos[4]` (0x200002cc) — ST3215_State_t[4], stride 0x20; position_rad@4,
  is_online@0x1c, consecutive_failures@0x1d

Scripts in `MDK-ARM/debug_tools/`: `read_imu_state.py` (IMU monitor),
`calib_servo_zero.py` (servo dwell-zero capture).

## SWD Reliability Notes (important)

- While the target runs at full speed (blocking I2C + watchdog), the initial SWD
  handshake is flaky. pyOCD `cmd -v` retries and works; the Python API needs
  `connect_mode='under-reset'` (assert NRST during attach) + retries.
- **With the servo system powered, SWD becomes unusable** (`Unexpected ACK`,
  failed reads/writes). The 1 Mbps servo UART + adapter injects noise onto SWD.
  Workaround: power off servos to flash/debug, or debug via the bus CDC port.
- If a watchdog reset loop bricks SWD (can't halt because reset keeps firing),
  flash with `pyocd load --connect under-reset` — it attaches during NRST.

## Bugs Found And Fixed (committed)

### 1. BMI088 accelerometer never enabled (commit `cae3bb3`)
- **Symptom:** `g_imu.accel` read all zeros via SWD; gyro was fine.
- **Root cause:** `bmi088.c` wrote `0x03` to `ACC_PWR_CTRL` (0x7D). Per the BMI088
  datasheet, normal/active mode is `0x04` (bit2). `0x03` leaves the acc inactive,
  so data registers read zero; Mahony then loses its gravity reference and cannot
  correct pitch/roll — balance control would be impossible.
- **Fix:** `0x03 → 0x04`, with a comment citing BST-BMI088-DS001.
- **Verified:** after reflash, accel z ≈ +9.8 m/s² level, roll/pitch track tilt.

### 2. USART3 was half-duplex but servos use a full-duplex adapter (commit `4cdb77b`)
- **Symptom:** every servo query hit the 3-failure `FAULT_SERVO` limit; position
  stayed at zero init. Diagnosed via SWD: `consecutive_failures=3` on all 4 servos.
- **Root cause:** the ST3215 servos connect through a Waveshare Bus Servo Adapter
  (A), which converts the single-wire half-duplex servo bus into a 2-wire UART.
  Firmware used `HAL_HalfDuplex_Init` on PB10 only — no RX path.
- **Fix:** `HAL_HalfDuplex_Init → HAL_UART_Init`, enable both PB10 (TX) and PB11
  (RX) as AF push-pull. Driver (`st3215.c`) already uses the standard full-duplex
  HAL API, so no protocol change.
- **Wiring (Waveshare adapter):** jumper at position A; same-name pairing
  PB10(TX)→TXD, PB11(RX)→RXD (the adapter crosses internally), common GND.
- **Docs:** added `docs/hardware/wiring.md` (all peripherals) + `pin_config.h`
  comment updated.

## Servo Communication — Open Issue

After fix #2, single-servo queries are rock solid but the 4-servo rotation crashes.

### Confirmed facts (via SWD isolation tests)
- Single id1, 25 s: ~100% success (no crash).
- Single id2, 20 s: ~99% success (no crash).
- All 4 rotated at 50 Hz: **always crashes at ~9 s (~358th query)**. Symptoms:
  RX suddenly fills with `NE` (noise) + garbage bytes, all 4 servos hit
  `consecutive_failures=3`, go offline → `FAULT_SERVO`.

### Root cause analysis (high confidence)
The crash is triggered by the `STATE_FAULT` branch in `main.c`:
```c
else if (mode == STATE_FAULT) {
    for (i=0;i<4;i++) st3215_set_torque_enable(&huart3, ids[i], 0);  // every 50 Hz!
}
```
Once any servo transiently fails and enters FAULT, this fires 4 torque-disable
packets **every 50 Hz cycle**. In full-duplex, each TX produces RX echo/loopback
bytes that accumulate and desync subsequent `read_state` queries — a bus-flood
snowball. Single-servo tests never enter FAULT (force-kept online), so they
never hit this path. That is why single = stable, multi = crash.

A second contributing factor: `System_Initial_Setup` did not refresh the IWDG, so
when servo UART blocking made init slow, the ~512 ms watchdog (PR=0/RLR=4095 on
LSI 32 kHz) reset the chip in a loop — which also made SWD unhalt-able.

### Fixes written (in working tree, NOT yet verified — see "Status")
- `main.c`: IWDG refresh calls added through `System_Initial_Setup`.
- `main.c`: FAULT torque-disable made one-shot (`static fault_torque_disabled`).
- `st3215.c`: **reverted to the committed 4cdb77b baseline** — the experimental
  header-sync / bus-quiet / RX-drain logic was reverted because it added blocking
  and did not survive the watchdog loop. Re-add carefully after the watchdog +
  one-shot fixes are confirmed.

These are uncommitted and unverified because SWD is unusable with servos powered
and the chip resets in a loop with servos off (adapter pulls PB10/PB11 low when
unpowered, stalling `HAL_UART_Transmit` in init).

## Servo Zero (dwell) Calibration — Not Yet Done

- Dwell = D0 = 58 mm (shortest virtual leg), per `kuafu_physics.py`
  (`D0_DWELL = 58.0`) and `kinematics.c` (`KIN_MIN_LEG_D0`). This is the only
  correct zero because sim/firmware/physics all define joint angles relative to it.
- At dwell: qA=0, qB=0; extending the leg makes qA negative, qB positive
  (symmetric). Geometrically the two cranks are nearly folded together.
- `SERVO_CENTER_INIT` in `pin_config.h` is still the `{2048,2048,2048,2048}`
  placeholder — must be measured per robot with `calib_servo_zero.py`.
- `SERVO_DIR_INIT = {+1,-1,+1,-1}` (right side mirrored) also needs bench check.
- ST3215 ship with ID=0; must be re-addressed to 1/2/3/4 with the vendor tool.

## IMU — Open Issue

At session end `g_system_ticks` was stuck at 1 — the PB1 DRDY interrupt fired
once then stopped, meaning the BMI088 gyro stopped producing data-ready. Earlier
in the session IMU was healthy (accel z≈9.8, gyro≈0, calib=1). Likely a loose
IMU connection (PB1/VCC) disturbed while wiring servos. Re-seat IMU wiring first.

## Resume Checklist (do in this order)

1. **Re-seat all wiring** — IMU (PB8/PB9/PB1/VCC/GND), confirm common GND
   between STM32 / servo supply / DAPLink with a continuity beep. Most intermittent
   faults here trace to ground/reference issues.
2. Power only STM32+IMU+DAPLink (servos OFF). Verify `read_imu_state.py` shows
   ticks advancing, accel z≈9.8. This re-confirms fix #1 and IMU wiring.
3. Flash the working-tree firmware (watchdog + one-shot FAULT) with servos OFF
   via `pyocd load --connect under-reset`.
4. Power servos ON. Observe servo behavior directly (no SWD): do they hold
   steady without twitching? That is the real success criterion for comms.
5. If stable, calibrate dwell zero with `calib_servo_zero.py` (torque must be
   disabled during calibration — temporarily, see the calibration build note).
6. Only then re-attempt SWD-based multi-servo success-rate measurement; if SWD
   still dies with servos on, accept behavioral verification + the CDC port.

## Key Lesson

The full-duplex adapter + 1 Mbps + 20 cm unterminated TX/RX pair is marginal.
If servo comms remain flaky after the watchdog/one-shot fixes, the next lever is
**electrical**: shorten/twist the TX/RX pair, add a common-ground bus bar, or
lower the baud rate (requires re-configuring ST3215 baud via the vendor tool and
matching `huart3.Init.BaudRate`). Do not keep piling software workarounds on a
noisy physical layer.
