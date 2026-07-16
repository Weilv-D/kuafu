"""Read live IMU/control state from STM32 via pyOCD SWD (non-intrusive, single session).

Reads key globals by address (from stm32_firmware.map) in ONE SWD session and
decodes them, looping to show how they change over time. Used to verify IMU
data acquisition + Mahony fusion.

The target runs at full speed (blocking I2C + watchdog), which makes the initial
SWD handshake flaky. We therefore connect UNDER RESET (assert NRST, attach the
DAP, then resume the core) and retry a few times.

Struct layouts from `fromelf --fieldoffsets` (Keil uses -fshort-enums, 1-byte enums):
  BMI088_t        (0x20): hi2c@0  accel@4   gyro@0x10  temp@0x1c
  MahonyFilter_t  (0x30): q0@0..q3@0xc  Kp@0x10 Ki@0x14 eInt@0x18  roll@0x24 pitch@0x28 yaw@0x2c
  SafetyState_t   (0x1c): mode@0 fault@1 timer@4 err@8  offset@0xc  is_calib@0x18
"""
import re, struct, time, sys
from pathlib import Path

from pyocd.core.helpers import ConnectHelper

PROBE = "LU_2022_8888"
TARGET = "stm32f407zgtx"

MAP_PATH = Path(__file__).resolve().parents[1] / "stm32_firmware" / "stm32_firmware.map"

def symbol_address(name):
    text = MAP_PATH.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"^\s+{re.escape(name)}\s+(0x[0-9a-fA-F]+)\s+Data\b", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"cannot find {name} in {MAP_PATH}")
    return int(match.group(1), 16)

# Resolve addresses from the latest build instead of keeping stale literals.
A_SYS_TICKS = symbol_address("g_system_ticks")
A_BODY_GYRO = symbol_address("g_body_gyro")
A_IMU       = symbol_address("g_imu")
A_MAHONY    = symbol_address("g_mahony")
A_SAFETY    = symbol_address("g_safety_state")

def f(b, o): return struct.unpack_from("<f", b, o)[0]
def u(b, o): return struct.unpack_from("<I", b, o)[0]

MODE_NAMES = {0:"INIT",1:"STAND",2:"ACTIVE",3:"CLIMB",4:"FAULT"}
FAULT_NAMES = {0x00:"NONE",0x01:"TILT",0x02:"HEARTBEAT",0x04:"OVERTEMP",0x08:"EMERGENCY",0x10:"SERVO"}

def snapshot(session):
    def rd(addr, size):
        return bytes(session.target.read_memory_block8(addr, size))
    st  = rd(A_SYS_TICKS, 4)
    imu = rd(A_IMU, 0x20)
    mah = rd(A_MAHONY, 0x30)
    saf = rd(A_SAFETY, 0x1c)
    bg  = rd(A_BODY_GYRO, 12)

    ticks = u(st, 0)
    ax,ay,az = f(imu,4),f(imu,8),f(imu,12)
    gx,gy,gz = f(imu,0x10),f(imu,0x14),f(imu,0x18)
    temp = f(imu,0x1c)
    roll,pitch,yaw = f(mah,0x24),f(mah,0x28),f(mah,0x2c)
    mode = saf[0]
    fault = saf[1]
    err = saf[8]
    off0,off1,off2 = f(saf,0xc),f(saf,0x10),f(saf,0x14)
    calib = u(saf,0x18)
    bgx,bgy,bgz = f(bg,0),f(bg,4),f(bg,8)
    return dict(ticks=ticks, ax=ax,ay=ay,az=az, gx=gx,gy=gy,gz=gz, temp=temp,
                roll=roll,pitch=pitch,yaw=yaw, mode=mode, fault=fault, err=err,
                off=(off0,off1,off2), calib=calib, bg=(bgx,bgy,bgz))

def open_session(tries=6):
    """Connect-under-reset with retries; returns a running session or None."""
    last_err = None
    for attempt in range(tries):
        try:
            session = ConnectHelper.session_with_chosen_probe(
                blocking=False, unique_id=PROBE, target_override=TARGET, frequency=1_000_000)
            if session is None:
                raise RuntimeError("probe not found")
            # connect_mode='under-reset' asserts NRST during DAP attach so the
            # running target (busy with blocking I2C) can't miss the handshake.
            session.options.set('connect_mode', 'under-reset')
            session.open()
            # core was halted by connect-under-reset; let it run so ticks advance
            session.target.resume()
            return session
        except Exception as e:
            last_err = e
            try:
                if 'session' in dir() and session is not None:
                    session.close()
            except Exception:
                pass
            time.sleep(0.3)
    print(f"ERROR: could not connect after {tries} tries: {last_err}")
    return None

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    session = open_session()
    if session is None:
        sys.exit(1)
    try:
        print(f"connected: {session.board.name} / {session.target.part_number}")
        print(f"reading {n} snapshots @ {interval}s interval ...")
        # connect-under-reset reboots the chip; wait for gyro calibration
        # (1000 DRDY samples ≈ 1 s) so the first snapshot reflects steady state.
        print("waiting for gyro calibration (calib=1) ...", flush=True)
        waited = 0.0
        calibrated = False
        while waited < 5.0:
            try:
                d0 = snapshot(session)
            except Exception:
                d0 = None
            if d0 and d0['calib'] == 1:
                calibrated = True
                break
            time.sleep(0.3); waited += 0.3
        if calibrated:
            print(f"  (calib reached after ~{waited:.1f}s)\n")
        else:
            print(f"  (calib not reached after ~{waited:.1f}s; continuing)\n")
        prev_ticks = None
        for i in range(n):
            try:
                d = snapshot(session)
            except Exception as e:
                print(f"[{i}] read error: {e}")
                time.sleep(interval); continue
            d_rate = (d['ticks'] - prev_ticks)/interval if prev_ticks is not None else 0
            prev_ticks = d['ticks']
            acc_mag = (d['ax']**2+d['ay']**2+d['az']**2)**0.5
            fl = d['fault']
            fdesc = ",".join(v for k,v in FAULT_NAMES.items() if fl & k) or "NONE"
            print(f"[{i}] ticks={d['ticks']:>10} (~{d_rate:7.0f}/s)")
            print(f"    accel(m/s2)  x={d['ax']:+7.3f} y={d['ay']:+7.3f} z={d['az']:+7.3f}  |a|={acc_mag:6.2f}")
            print(f"    gyro (rad/s) x={d['gx']:+7.4f} y={d['gy']:+7.4f} z={d['gz']:+7.4f}")
            print(f"    gyro_bias    x={d['off'][0]:+.5f} y={d['off'][1]:+.5f} z={d['off'][2]:+.5f}  calib={d['calib']}")
            print(f"    body_gyro    x={d['bg'][0]:+.4f} y={d['bg'][1]:+.4f} z={d['bg'][2]:+.4f}")
            print(f"    attitude(deg) roll={d['roll']*57.2958:+7.2f} pitch={d['pitch']*57.2958:+7.2f} yaw={d['yaw']*57.2958:+7.2f}")
            print(f"    mode={MODE_NAMES.get(d['mode'],d['mode'])} fault={fdesc} err={d['err']:#x} temp={d['temp']:.1f}C")
            print()
            if i < n-1: time.sleep(interval)
    finally:
        session.close()

if __name__ == "__main__":
    main()
