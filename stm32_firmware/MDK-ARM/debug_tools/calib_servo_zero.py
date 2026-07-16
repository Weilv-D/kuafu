"""Servo mechanical-zero (dwell) calibration helper.

Reads the live ST3215 present-position of all four hip servos via SWD while the
firmware runs (in FAULT/SERVO the firmware disables torque, so servos are free
to be moved by hand). You manually pose each leg at the dwell posture, then the
script computes the raw tick each servo reports — that tick is SERVO_CENTER[i].

The helper reads ``position_tick`` directly. Shared-frame joint angles are
derived only by the firmware's calibrated ``servo_mapping`` module.
"""
import re, statistics, struct, time, sys
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

# Each ST3215_State_t is 0x28 bytes. Resolve the base from the latest build.
A_SERVOS = symbol_address("g_servos")
SERVO_STRIDE = 0x28
OFF_POS_TICK = 0x02
OFF_LAST_VALID_MS = 0x1C
OFF_IS_ONLINE = 0x27
SERVO_NAMES = ["LF (id1, hip_A_l)", "RF (id2, hip_A_r)", "LB (id3, hip_B_l)", "RB (id4, hip_B_r)"]

TICK_TO_RAD = (2.0 * 3.14159265) / 4096.0

def read_servos(session):
    """Return list of (is_online, raw_tick, position_rad) for the 4 servos."""
    # One contiguous SWD transfer is much more reliable than eight small reads
    # when the powered 1 Mbps servo bus is injecting noise near SWD wiring.
    data = bytes(session.target.read_memory_block8(A_SERVOS, 4 * SERVO_STRIDE))
    out = []
    for i in range(4):
        base = i * SERVO_STRIDE
        raw_tick = struct.unpack_from("<H", data, base + OFF_POS_TICK)[0]
        last_valid_ms = struct.unpack_from("<I", data, base + OFF_LAST_VALID_MS)[0]
        is_online = int(data[base + OFF_IS_ONLINE] != 0 and last_valid_ms != 0)
        out.append((is_online, float(raw_tick), raw_tick * TICK_TO_RAD))
    return out

def open_session(tries=6):
    last = None
    for _ in range(tries):
        try:
            s = ConnectHelper.session_with_chosen_probe(
                blocking=False, unique_id=PROBE, target_override=TARGET, frequency=100_000)
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
    # Allow several complete round-robin poll cycles after the first valid frame.
    time.sleep(1.0)
    sv = read_servos(session)

    if len(sys.argv) > 1 and sys.argv[1] == "--capture":
        samples = []
        for _ in range(9):
            samples.append(read_servos(session))
            time.sleep(0.1)
        print("\n=== CAPTURED (median of 9 samples) ===")
        ticks = []
        for i in range(4):
            online = all(sample[i][0] for sample in samples)
            raw_tick = statistics.median(sample[i][1] for sample in samples)
            tick = max(0, min(4095, int(round(raw_tick))))
            ticks.append(tick)
            print(f"  {SERVO_NAMES[i]}: online={int(online)}  raw_tick={raw_tick:.1f} -> {tick}")
        print("\nPut these into pin_config.h (replace SERVO_CENTER_INIT):")
        print(f'  #define SERVO_CENTER_INIT  {{ {ticks[0]}, {ticks[1]}, {ticks[2]}, {ticks[3]} }}')
        session.close()
        return

    print("\nServo present-position monitor. Pose each leg at DWELL (mechanical")
    print("center) by hand — servos are free (torque off in FAULT/SERVO).")
    print("Live tick values update below. Press Ctrl+C to capture.\n")

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
