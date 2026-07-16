"""Generate Pi-link HIL cases; dry-run unless --send is explicitly supplied."""

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pi5_runtime.protocol import command_frames, crc8_maxim, hello_frame

GENERATED = ROOT / "stm32_firmware" / "Core" / "Inc" / "kuafu_generated.h"


def generated_value(name: str) -> str:
    text = GENERATED.read_text(encoding="utf-8")
    match = re.search(rf"^#define\s+{re.escape(name)}\s+(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"missing {name} in {GENERATED}")
    return match.group(1).strip().strip('"').rstrip('f')


def cases() -> dict[str, list[tuple[str, bytes, float]]]:
    model_hash = generated_value("KUAFU_MODEL_HASH")
    d0_min = float(generated_value("D0_MIN_MM"))
    valid_hello = hello_frame(1, 0, model_hash).encode()
    wrong_hash = hello_frame(2, 0, "0000000000000000").encode()
    wrong_version = bytearray(valid_hello)
    wrong_version[1] = 2
    wrong_version[-2] = crc8_maxim(bytes(wrong_version[1:-2]))
    bad_crc = bytearray(valid_hello)
    bad_crc[-2] ^= 1
    heartbeat, action = command_frames(10, 0, 2, 0.0, 0.0, d0_min, [0.0] * 6)
    emergency, _ = command_frames(20, 0, 4, 0.0, 0.0, d0_min, [0.0] * 6)
    raw_hb, raw_action = heartbeat.encode(), action.encode()
    return {
        "valid": [("valid HELLO", valid_hello, 0.0), ("heartbeat", raw_hb, 0.0), ("action", raw_action, 0.0)],
        "wrong-hash": [("wrong hash", wrong_hash, 0.0)],
        "wrong-version": [("wrong version", bytes(wrong_version), 0.0)],
        "bad-crc": [("CRC corruption", bytes(bad_crc), 0.0)],
        "fragment": [("fragment 1", valid_hello[:5], 0.05), ("fragment 2", valid_hello[5:], 0.0)],
        "replay": [("heartbeat", raw_hb, 0.0), ("heartbeat replay", raw_hb, 0.0)],
        "stale-action": [("HELLO", valid_hello, 0.0), ("heartbeat", raw_hb, 0.10),
                         ("action then wait >80 ms", raw_action, 0.10)],
        "stale-heartbeat": [("HELLO", valid_hello, 0.0), ("heartbeat then wait >200 ms", raw_hb, 0.25)],
        "emergency": [("FAULT mode request", emergency.encode(), 0.0)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KUAFU Pi-link protocol HIL injector (dry-run by default)")
    parser.add_argument("--scenario", choices=["all", *cases().keys()], default="all")
    parser.add_argument("--port", help="serial port, required with --send")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--send", action="store_true", help="actually write frames to the selected serial port")
    args = parser.parse_args()
    if args.send and not args.port:
        parser.error("--port is required with --send")
    serial_port = None
    if args.send:
        import serial
        serial_port = serial.Serial(args.port, args.baudrate, timeout=0)
    selected = cases().items() if args.scenario == "all" else [(args.scenario, cases()[args.scenario])]
    try:
        for scenario, frames in selected:
            print(f"[{scenario}]")
            for label, data, delay in frames:
                print(f"  {label}: {data.hex(' ')}")
                if serial_port is not None:
                    serial_port.write(data)
                    serial_port.flush()
                if delay:
                    if serial_port is not None:
                        time.sleep(delay)
                    else:
                        print(f"  dry-run delay: {delay:.2f}s")
    finally:
        if serial_port is not None:
            serial_port.close()


if __name__ == "__main__":
    main()
