"""Read firmware health records over SWD without writing target memory."""

import argparse
import re
import struct
import time
from pathlib import Path

PROBE = "LU_2022_8888"
TARGET = "stm32f407zgtx"
MAP_PATH = Path(__file__).resolve().parents[1] / "stm32_firmware" / "stm32_firmware.map"


def symbol_address(name: str) -> int:
    text = MAP_PATH.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"^\s+{re.escape(name)}\s+(0x[0-9a-fA-F]+)\s+Data\b", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"cannot find {name} in {MAP_PATH}")
    return int(match.group(1), 16)


def decode_health(data: bytes, offset: int) -> dict[str, int]:
    last, timeout, checksum, protocol, consecutive, online = struct.unpack_from("<IHHHBB", data, offset)
    return {"last": last, "timeout": timeout, "checksum": checksum,
            "protocol": protocol, "consecutive": consecutive, "online": online}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only KUAFU firmware health over DAPLink")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--frequency", type=int, default=100_000)
    parser.add_argument("--under-reset", action="store_true",
                        help="explicitly reset during attach; default attaches without reset")
    args = parser.parse_args()

    from pyocd.core.helpers import ConnectHelper

    addresses = {name: symbol_address(name) for name in
                 ("g_system_ticks", "g_imu", "g_ddsm_left", "g_ddsm_right", "g_servos", "g_safety_state")}
    session = ConnectHelper.session_with_chosen_probe(
        blocking=False, unique_id=PROBE, target_override=TARGET, frequency=args.frequency)
    if session is None:
        raise SystemExit("DAPLink probe not found")
    if args.under_reset:
        session.options.set("connect_mode", "under-reset")
    session.open()
    try:
        session.target.resume()
        for _ in range(args.samples):
            tick = session.target.read32(addresses["g_system_ticks"])
            mode = session.target.read8(addresses["g_safety_state"])
            fault = session.target.read32(addresses["g_safety_state"] + 8)
            imu = bytes(session.target.read_memory_block8(addresses["g_imu"], 0x34))
            left = bytes(session.target.read_memory_block8(addresses["g_ddsm_left"], 0x20))
            right = bytes(session.target.read_memory_block8(addresses["g_ddsm_right"], 0x20))
            servos = bytes(session.target.read_memory_block8(addresses["g_servos"], 4 * 0x28))
            records = [("imu", decode_health(imu, 0x20)),
                       ("wheel_l", decode_health(left, 0x14)),
                       ("wheel_r", decode_health(right, 0x14))]
            records += [(f"servo_{i + 1}", decode_health(servos, i * 0x28 + 0x1C)) for i in range(4)]
            print(f"tick={tick} mode={mode} fault=0x{fault:08x}")
            for name, health in records:
                age = "never" if health["last"] == 0 else str((tick - health["last"]) & 0xFFFFFFFF)
                errors = health["timeout"] + health["checksum"] + health["protocol"]
                print(f"  {name:8s} online={health['online']} age_ms={age:>5} "
                      f"errors={errors} consecutive={health['consecutive']}")
            time.sleep(args.interval)
    finally:
        session.close()


if __name__ == "__main__":
    main()
