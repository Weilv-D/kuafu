# STM32 Debug Tools

These tools use DAPLink and pyOCD to inspect the running STM32 without changing
target memory. Symbol addresses are resolved from the current Keil map file, so
the tools remain valid when a rebuild moves globals.

## Environment

- Probe ID: `LU_2022_8888`
- Target: `stm32f407zgtx`
- pyOCD package path on the bring-up workstation:
  `C:\Users\Deng2\AppData\Roaming\Python\Python313\site-packages`

Example:

```powershell
$env:PYTHONPATH='C:\Users\Deng2\AppData\Roaming\Python\Python313\site-packages'
python stm32_firmware\MDK-ARM\debug_tools\read_health.py --samples 10 --interval 0.05
```

Use `--connect under-reset` when flashing. Normal health reads attach without
reset and halt the core only long enough to take a coherent snapshot.

## Tools

- `read_health.py` reports startup phase, safety mode and faults, BMI088 state and
  temperature, UART/I2C state, wheel and servo freshness, error counters, and
  servo temperatures.
- `read_imu_state.py` provides a focused IMU and attitude view.
- `calib_servo_zero.py` captures median raw positions for all four servos. Use it
  only with `SERVO_ZERO_CALIBRATION_MODE=1`.
- `hil_protocol.py` exercises Pi protocol cases. Run its dry-run mode before any
  serial transmission.
- `HIL_CHECKLIST.md` defines the ordered hardware gates.

## Servo Zero Capture

The dwell pose is `Qx=0`, `D0=58 mm`. Firmware order is
`[LF,RF,LB,RB]=[A_l,A_r,B_l,B_r]`. The accepted centers are
`[275,1097,2809,1023]`. Direction is verified from joint signs and raw tick
changes, never from viewing-dependent clockwise/counter-clockwise wording.

## Accepted Health Snapshot

The 2026-07-16 final run reached startup phase 4 (`READY`) in 1.739 seconds,
entered safe `STAND`, and reported fault mask zero. BMI088 age was 0–1 ms, wheel
ages 0–12 ms, and servo ages 0–20 ms. Temperatures were 33.1–33.4°C for BMI088
and 41–43°C for the servos. With no compatible Pi heartbeat, wheel authorization
remained false and both wheels stayed still.

See `BRINGUP_LOG.md` for the complete bring-up record and final wiring.
