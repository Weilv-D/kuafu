"""Trace balance-relevant state at 20 Hz over SWD (attach, no reset).

Adds yaw / yaw-rate / LQR position state vs the boot trace, to diagnose
spinning (differential torque) and drifting (position loop) behaviour.

Usage:  python kuafu_balance_trace.py [seconds]   (default 20)
"""

import struct
import sys
import time

from pyocd.core.helpers import ConnectHelper

ADDR = {
    "mahony": 0x200003d0,   # roll@+36 pitch@+40 yaw@+44
    "lqr": 0x2000071c,      # x_est@+20 x_ref@+24 x_int@+28 yaw_ref@+40
    "ddsm_l": 0x20000408,   # velocity_rads@+8
    "ddsm_r": 0x20000430,
    "startup": 0x20000528,
    "safety": 0x2000088c,   # mode@+0 fault_mask@+12
    "body_gyro": 0x200000a8,
    "tau_l": 0x2000001c,
    "tau_r": 0x20000020,
    "bpr": 0x20000018,
}

MODES = ["INIT", "STAND", "ACTIVE", "CLIMB", "FAULT"]
PHASES = ["WAIT_POWER", "IMU_DISC", "GYRO_CALIB", "ACT_DISC", "READY", "FAILED"]
FAULT_BITS = [(0, "TILT"), (2, "OVERTEMP"), (3, "EMERG"), (4, "SERVO"),
              (5, "IMU"), (6, "WHEEL_L"), (7, "WHEEL_R"), (8, "P_RATE"),
              (9, "INIT"), (10, "INTERNAL")]


def read_f(target, addr):
    return struct.unpack("<f", bytes(target.read_memory_block8(addr, 4)))[0]


def read_u32(target, addr):
    return struct.unpack("<I", bytes(target.read_memory_block8(addr, 4)))[0]


def fault_str(mask):
    if mask == 0:
        return "-"
    return "|".join(n for b, n in FAULT_BITS if mask & (1 << b)) or f"0x{mask:x}"


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    with ConnectHelper.session_with_chosen_probe(
            options={"connect_mode": "attach", "frequency": 1000000}) as session:
        target = session.target
        print("t_s    phase      mode   fault     pitch    bpr      yaw     gz      wl      wr      tau_l   tau_r   xerr    xint")
        start = time.monotonic()
        next_sample = start
        while time.monotonic() - start < duration:
            now = time.monotonic()
            if now < next_sample:
                time.sleep(next_sample - now)
                continue
            next_sample += 0.05
            t = now - start
            phase = read_u32(target, ADDR["startup"])
            mode = read_u32(target, ADDR["safety"])
            fault_mask = read_u32(target, ADDR["safety"] + 12)
            pitch = read_f(target, ADDR["mahony"] + 40)
            yaw = read_f(target, ADDR["mahony"] + 44)
            bpr = read_f(target, ADDR["bpr"])
            gz = read_f(target, ADDR["body_gyro"] + 8)
            wl = read_f(target, ADDR["ddsm_l"] + 8)
            wr = read_f(target, ADDR["ddsm_r"] + 8)
            tau_l = read_f(target, ADDR["tau_l"])
            tau_r = read_f(target, ADDR["tau_r"])
            xerr = read_f(target, ADDR["lqr"] + 20) - read_f(target, ADDR["lqr"] + 24)
            xint = read_f(target, ADDR["lqr"] + 28)
            phase_str = PHASES[phase] if phase < len(PHASES) else f"?{phase}"
            mode_str = MODES[mode] if mode < len(MODES) else f"?{mode}"
            print(f"{t:6.2f} {phase_str:10s} {mode_str:6s} {fault_str(fault_mask):9s} "
                  f"{pitch:+7.3f} {bpr:+7.3f} {yaw:+7.3f} {gz:+7.3f} "
                  f"{wl:+7.2f} {wr:+7.2f} {tau_l:+7.3f} {tau_r:+7.3f} "
                  f"{xerr:+7.3f} {xint:+7.3f}", flush=True)


if __name__ == "__main__":
    main()
