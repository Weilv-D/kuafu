# Debug Tools

Non-intrusive STM32 debugging via DAPLink (CMSIS-DAP) + pyOCD over SWD. No
firmware changes, no UART needed — these scripts read live target memory while
the firmware runs.

## Setup

- Python 3 + `pip install pyocd`
- STM32F407 pack (once): `pyocd pack install stm32f407zgtx`
- DAPLink probe ID is hardcoded as `LU_2022_8888` in each script (`PROBE`); change
  it if you swap debuggers.

## SWD Debug Method

Symbol addresses come from `../stm32_firmware/stm32_firmware.map` (regenerate
after each build); struct field offsets from
`fromelf --fieldoffsets ../stm32_firmware/stm32_firmware.axf` (Keil uses
`-fshort-enums`, so enums are 1 byte — do not assume 4).

Key globals and their addresses (built 2026-07-16; re-check the .map after rebuild):
- `g_system_ticks` (0x20000000) — PB1/IMU-DRDY interrupt count; the scheduler
  heartbeat. If it stops advancing, the main loop is not running.
- `g_imu` (0x20000218) — BMI088_t (0x20): accel[3]@0x04, gyro[3]@0x10, temp@0x1c
- `g_mahony` (0x20000238) — MahonyFilter_t (0x30): roll@0x24, pitch@0x28, yaw@0x2c
- `g_safety_state` (0x200003d8) — SafetyState_t (0x1c): mode@0, fault@1, err@8,
  gyro_offset[3]@0xc, is_calib@0x18
- `g_servos[4]` (0x200002cc) — ST3215_State_t[4], stride 0x20: position_rad@0x04,
  is_online@0x1c, consecutive_failures@0x1d

## SWD Reliability Notes

- The target runs at full speed (blocking I2C + watchdog), so the initial SWD
  handshake is flaky. The Python API needs `connect_mode='under-reset'` (assert
  NRST during attach) + retries; `pyocd cmd -v` retries automatically.
- **With the servo system powered, SWD often becomes unusable** (`Unexpected ACK`,
  failed reads/writes) — the 1 Mbps servo UART + adapter injects noise onto SWD.
  Prefer behavioral verification (servo motion) or the DAPLink CDC port when
  servos are on.
- If a watchdog reset loop bricks SWD (reset keeps firing before halt takes),
  flash with `pyocd load --connect under-reset`.

## Scripts

### read_imu_state.py — IMU / attitude monitor

Reads and decodes the globals above in one SWD session, printing accel, gyro,
gyro bias, safety mode, and Mahony roll/pitch/yaw over several samples.

```bash
python read_imu_state.py            # 5 samples, 0.5 s apart
python read_imu_state.py 10 0.3     # 10 samples, 0.3 s apart
```

Sanity (board level and still): accel z ≈ +9.8 m/s², |a| ≈ 9.8; gyro ≈ 0
(±0.005 rad/s) with `calib=1`; roll/pitch ≈ 0°. yaw drifts slowly (no
magnetometer). `mode=FAULT/SERVO` is expected when servos are absent.

### calib_servo_zero.py — servo dwell-zero capture

While the firmware holds servos torque-free (STATE_FAULT disables torque), this
reads each servo's present-position tick live. Pose each leg at dwell by hand,
then Ctrl+C to capture the four `SERVO_CENTER` values for `pin_config.h`.

```bash
# 1. Connect 4x ST3215 (IDs 1/2/3/4, USART3 PB10 bus), powered, common GND
# 2. Run; pose legs at dwell; Ctrl+C to capture
python calib_servo_zero.py
```

See `../docs/hardware/wiring.md` for the servo adapter wiring and
`../docs/hardware/calibration.md` for the dwell definition.

## Bring-Up Log

`BRINGUP_LOG.md` records the chronological findings and decisions from the
2026-07-16 bring-up session (bugs found, root-cause analysis, open issues,
resume checklist). Read it before resuming servo/IMU debugging.
