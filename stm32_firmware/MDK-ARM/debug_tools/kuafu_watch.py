"""Live KUAFU firmware state watch over SWD (pyocd).

Attaches to the running target WITHOUT reset or halt and prints the balance
control state at ~5 Hz.  Use it to diagnose stand/balance behaviour on the
bench: pitch estimate vs. reality, wheel velocity signs, torque commands,
safety mode/faults, startup phase.

Usage:
    python kuafu_watch.py [seconds]     # default: run until Ctrl+C

Addresses come from stm32_firmware.map; regenerate if the build changes.
"""

import struct
import sys
import time

from pyocd.core.helpers import ConnectHelper

ADDR = {
    "act_cfg": 0x20000008,
    "wheel_gate": 0x20000001,   # final wheel-output authorization (250 Hz verdict)
    "ticks": 0x2000000c,
    "pitch_filt": 0x20000010,
    "body_pitch": 0x20000014,
    "body_pitch_rate": 0x20000018,
    "tau_l": 0x2000001c,
    "tau_r": 0x20000020,
    "body_gyro": 0x200000a8,      # float[3]
    "imu": 0x2000033c,            # +4 accel[3] +16 gyro[3] +28 temp +32 health.last_valid
    "mahony": 0x200003cc,         # +36 roll +40 pitch +44 yaw
    "lqr": 0x20000718,            # +20 x_est +24 x_ref +28 x_int +32 v_ref
    "ddsm_l": 0x20000404,         # +4 torque +8 vel +12 pos +16 err +20 health.last_valid
    "ddsm_r": 0x2000042c,
    "startup": 0x20000524,        # +0 phase
    "safety": 0x20000888,         # +0 mode +12 fault_mask +16 gyro_calib[3] +28 calibrated
    "hb": 0x200009a8,             # +0 mode_request +4 vx +8 wz +12 d0 +16 last_hb_ms
}

MODES = ["INIT", "STAND", "ACTIVE", "CLIMB", "FAULT"]
PHASES = ["WAIT_POWER", "IMU_DISC", "GYRO_CALIB", "ACT_DISC", "READY", "FAILED"]
FAULT_BITS = [
    (0, "TILT"), (1, "HEARTBEAT"), (2, "OVERTEMP"), (3, "EMERGENCY"),
    (4, "SERVO"), (5, "IMU"), (6, "WHEEL_L"), (7, "WHEEL_R"),
    (8, "PITCH_RATE"), (9, "INIT"), (10, "INTERNAL"),
]


def read_f(target, addr):
    return struct.unpack("<f", bytes(target.read_memory_block8(addr, 4)))[0]


def read_u32(target, addr):
    return struct.unpack("<I", bytes(target.read_memory_block8(addr, 4)))[0]


def fault_str(mask):
    if mask == 0:
        return "-"
    names = [name for bit, name in FAULT_BITS if mask & (1 << bit)]
    return "|".join(names) if names else f"0x{mask:x}"


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else None
    with ConnectHelper.session_with_chosen_probe(
            options={"connect_mode": "attach", "frequency": 1000000}) as session:
        target = session.target
        target.init()
        print("attached (no reset). Ctrl+C to stop.\n")
        start = time.monotonic()
        while duration is None or time.monotonic() - start < duration:
            mode = read_u32(target, ADDR["safety"])
            fault_mask = read_u32(target, ADDR["safety"] + 12)
            calib = read_u32(target, ADDR["safety"] + 28) & 0xFF
            phase = read_u32(target, ADDR["startup"])
            roll = read_f(target, ADDR["mahony"] + 36)
            pitch = read_f(target, ADDR["mahony"] + 40)
            yaw = read_f(target, ADDR["mahony"] + 44)
            bp = read_f(target, ADDR["body_pitch"])
            bpr = read_f(target, ADDR["body_pitch_rate"])
            gyro = [read_f(target, ADDR["body_gyro"] + 4 * i) for i in range(3)]
            tau_l = read_f(target, ADDR["tau_l"])
            tau_r = read_f(target, ADDR["tau_r"])
            wl = read_f(target, ADDR["ddsm_l"] + 8)
            wr = read_f(target, ADDR["ddsm_r"] + 8)
            x_est = read_f(target, ADDR["lqr"] + 20)
            x_ref = read_f(target, ADDR["lqr"] + 24)
            x_int = read_f(target, ADDR["lqr"] + 28)
            gate = target.read8(ADDR["wheel_gate"])
            imu_age_owner = read_u32(target, ADDR["imu"] + 32)
            mode_str = MODES[mode] if mode < len(MODES) else f"?{mode}"
            phase_str = PHASES[phase] if phase < len(PHASES) else f"?{phase}"
            print(
                f"{phase_str:10s} {mode_str:6s} gate={gate} calib={calib} fault={fault_str(fault_mask):12s} "
                f"pitch={pitch:+7.3f} roll={roll:+7.3f} bp={bp:+7.3f} bpr={bpr:+7.3f} "
                f"gyro=({gyro[0]:+6.2f},{gyro[1]:+6.2f},{gyro[2]:+6.2f}) "
                f"wl={wl:+7.2f} wr={wr:+7.2f} tau=({tau_l:+6.3f},{tau_r:+6.3f}) "
                f"x={x_est:+.3f}/{x_ref:+.3f} xi={x_int:+.4f}",
                flush=True,
            )
            time.sleep(0.2)


if __name__ == "__main__":
    main()
