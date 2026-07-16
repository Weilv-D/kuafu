"""Servo mechanical-zero (dwell) calibration helper.

Reads the live ST3215 present-position of all four hip servos via SWD while the
firmware runs (in FAULT/SERVO the firmware disables torque, so servos are free
to be moved by hand). You manually pose each leg at the dwell posture, then the
script computes the raw tick each servo reports — that tick is SERVO_CENTER[i].

Convention (firmware, st3215.c):
    position_rad = (raw_tick - 2048) * (2*pi/4096)
  => raw_tick    = position_rad / (2*pi/4096) + 2048
"""
import struct, time, sys
from pyocd.core.helpers import ConnectHelper

PROBE = "LU_2022_8888"
TARGET = "stm32f407zgtx"

# g_servos[4] @ 0x200002cc, each ST3215_State_t is 0x20 bytes
A_SERVOS = 0x200002cc
SERVO_STRIDE = 0x20
OFF_POS_RAD = 0x04
OFF_IS_ONLINE = 0x1c
SERVO_NAMES = ["LF (id1, hip_A_l)", "RF (id2, hip_A_r)", "LB (id3, hip_B_l)", "RB (id4, hip_B_r)"]

TICK_TO_RAD = (2.0 * 3.14159265) / 4096.0
RAD_TO_TICK = 1.0 / TICK_TO_RAD

def read_servos(session):
    """Return list of (is_online, raw_tick, position_rad) for the 4 servos."""
    out = []
    for i in range(4):
        base = A_SERVOS + i * SERVO_STRIDE
        pos_bytes = bytes(session.target.read_memory_block8(base + OFF_POS_RAD, 4))
        onl_byte = bytes(session.target.read_memory_block8(base + OFF_IS_ONLINE, 1))
        pos_rad = struct.unpack("<f", pos_bytes)[0]
        is_online = onl_byte[0]
        raw_tick = pos_rad * RAD_TO_TICK + 2048.0
        out.append((is_online, raw_tick, pos_rad))
    return out

def open_session(tries=6):
    last = None
    for _ in range(tries):
        try:
            s = ConnectHelper.session_with_chosen_probe(
                blocking=False, unique_id=PROBE, target_override=TARGET, frequency=1_000_000)
            if s is None:
                raise RuntimeError("probe not found")
            s.options.set('connect_mode', 'under-reset')
            s.open()
            s.target.resume()
            return s
        except Exception as e:
            last = e
            try:
                if s: s.close()
            except Exception:
                pass
            time.sleep(0.3)
    print(f"ERROR: cannot connect: {last}")
    return None

def main():
    session = open_session()
    if session is None:
        sys.exit(1)
    print(f"connected: {session.target.part_number}")
    print("connect-under-reset rebooted the chip; servos re-enumerate after boot.\n")

    # Wait until at least one servo is online OR a timeout (they come up
    # individually as the firmware polls them). FAULT/SERVO still polls.
    print("waiting for servos to come online ...")
    deadline = time.time() + 30
    while time.time() < deadline:
        sv = read_servos(session)
        n_online = sum(1 for o, _, _ in sv if o)
        if n_online > 0:
            break
        time.sleep(0.5)
    sv = read_servos(session)

    print("\nServo present-position monitor. Pose each leg at DWELL (mechanical")
    print("center) by hand — servos are free (torque off in FAULT/SERVO).")
    print("Live tick values update below. Press ENTER to capture, Ctrl+C to quit.\n")

    try:
        while True:
            sv = read_servos(session)
            line = "  | ".join(
                f"{SERVO_NAMES[i].split()[0]}: {'ON ' if sv[i][0] else 'OFF'} tick={sv[i][1]:7.1f}"
                for i in range(4))
            print("\r" + line + "   ", end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n\n=== CAPTURED ===")
        sv = read_servos(session)
        ticks = []
        for i in range(4):
            onl, tk, pr = sv[i]
            tk_int = int(round(tk))
            tk_int = max(0, min(4095, tk_int))
            ticks.append(tk_int)
            print(f"  {SERVO_NAMES[i]}: online={onl}  raw_tick={tk:.1f} -> {tk_int}")
        print()
        print("Put these into pin_config.h (replace SERVO_CENTER_INIT):")
        print(f'  #define SERVO_CENTER_INIT  {{ {ticks[0]}, {ticks[1]}, {ticks[2]}, {ticks[3]} }}')
        print(f"  /* [LF, RF, LB, RB] — calibrated {time.strftime('%Y-%m-%d')} */")
    finally:
        session.close()

if __name__ == "__main__":
    main()
