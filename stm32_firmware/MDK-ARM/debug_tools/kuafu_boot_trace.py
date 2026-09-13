"""Reset the target over SWD and trace the boot/balance transient at 20 Hz.

Captures startup phase, safety mode, pitch, wheel velocities and torque
commands from reset onward, so a runaway direction or bad attitude estimate
is visible sample-by-sample.

Usage:  python kuafu_boot_trace.py [seconds]   (default 15)
"""

import struct
import sys
import time

from pyocd.core.helpers import ConnectHelper

ADDR = {
    "ticks": 0x2000000c,
    "body_pitch": 0x20000014,
    "body_pitch_rate": 0x20000018,
    "tau_l": 0x2000001c,
    "tau_r": 0x20000020,
    "mahony": 0x200003cc,
    "lqr": 0x20000718,
    "ddsm_l": 0x20000404,
    "ddsm_r": 0x2000042c,
    "startup": 0x20000524,
    "safety": 0x20000888,
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
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    with ConnectHelper.session_with_chosen_probe(
            options={"connect_mode": "attach", "frequency": 1000000}) as session:
        target = session.target
        target.reset()
        target.resume()
        print("target reset; tracing %.1f s\n" % duration)
        print("t_s    phase      mode   fault     pitch    bpr     wl      wr      tau_l   tau_r")
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
            bpr = read_f(target, ADDR["body_pitch_rate"])
            wl = read_f(target, ADDR["ddsm_l"] + 8)
            wr = read_f(target, ADDR["ddsm_r"] + 8)
            tau_l = read_f(target, ADDR["tau_l"])
            tau_r = read_f(target, ADDR["tau_r"])
            phase_str = PHASES[phase] if phase < len(PHASES) else f"?{phase}"
            mode_str = MODES[mode] if mode < len(MODES) else f"?{mode}"
            print(f"{t:6.2f} {phase_str:10s} {mode_str:6s} {fault_str(fault_mask):9s} "
                  f"{pitch:+7.3f} {bpr:+7.3f} {wl:+7.2f} {wr:+7.2f} {tau_l:+7.3f} {tau_r:+7.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
