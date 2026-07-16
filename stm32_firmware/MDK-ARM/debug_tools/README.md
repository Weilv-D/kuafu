# Debug Tools

DAPLink (CMSIS-DAP) and pyOCD tools for reading live STM32 state. The IMU tool
is non-intrusive. Servo zero capture requires the dedicated torque-free firmware
mode described below.

## Setup

- Python 3 + `pip install pyocd`
- STM32F407 pack (once): `pyocd pack install stm32f407zgtx`
- DAPLink probe ID is hardcoded as `LU_2022_8888` in each script (`PROBE`); change
  it if you swap debuggers.

## SWD Debug Method

The scripts resolve symbol addresses from `../stm32_firmware/stm32_firmware.map`
at startup (regenerate it with each build); struct field offsets come from
`fromelf --fieldoffsets ../stm32_firmware/stm32_firmware.axf` (Keil uses
`-fshort-enums`, so enums are 1 byte — do not assume 4).

Key globals and their addresses (built 2026-07-16; re-check the .map after rebuild):
- `g_system_ticks` (0x20000004) — PB1/IMU-DRDY interrupt count; the scheduler
  heartbeat. If it stops advancing, the main loop is not running.
- `g_imu` (0x20000224) — BMI088_t (0x34): accel[3]@0x04, gyro[3]@0x10, temp@0x1c,
  health@0x20, init state@0x30
- `g_mahony` (0x20000258) — MahonyFilter_t (0x30): roll@0x24, pitch@0x28, yaw@0x2c
- `g_safety_state` (0x20000604) — SafetyState_t (0x1c): mode@0, timer@4,
  fault_mask@8, gyro_offset[3]@0xc, is_calib@0x18
- `g_servos[4]` (0x200002fc) — ST3215_State_t[4], stride 0x28: position_tick@0x02,
  position_rad@0x04, health.consecutive_failures@0x26, health.online@0x27

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

Zero capture must use a firmware image built with
`SERVO_ZERO_CALIBRATION_MODE=1`. That mode leaves wheel actuation inactive,
disables ST3215 torque during startup, suppresses all servo position commands,
and keeps feedback polling active. Do not run zero capture against ordinary
control firmware whose centers have not yet been measured.

Connect IDs 1/2/3/4, place both five-bars at `Qx=0`, `D0=58 mm`, and take a
short median capture:

```bash
python calib_servo_zero.py --capture
```

The tool waits for round-robin feedback to replace startup values, reads all
four servo structures in one SWD transfer at 100 kHz, and reports the median of
nine samples in firmware order `[LF,RF,LB,RB]=[A_l,A_r,B_l,B_r]`. Interactive
monitoring remains available with `python calib_servo_zero.py`; press Ctrl+C to
capture the displayed pose.

After writing the measured centers, set `SERVO_ZERO_CALIBRATION_MODE=0`, rebuild,
flash with servo power off, and verify dwell hold with wheel power disconnected.
Direction is a separate calibration: extension must make joint signs
`A_l<0, A_r<0, B_l>0, B_r>0`. See `../../../docs/hardware/wiring.md` and
`../../../docs/hardware/calibration.md` for the complete procedure.

## Bring-Up Log

`BRINGUP_LOG.md` records the 2026-07-16 findings, current verified hardware
state, and remaining bench gates. Read it before resuming servo/IMU work.
