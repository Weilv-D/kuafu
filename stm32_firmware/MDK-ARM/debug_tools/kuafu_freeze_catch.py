"""Watch g_system_ticks; when it stops advancing, halt the core and dump
registers so we can see exactly where the main loop is stuck."""

import struct
import sys
import time

from pyocd.core.helpers import ConnectHelper

TICKS_ADDR = 0x20000008
FREEZE_S = 0.3


def read_u32(target, addr):
    return struct.unpack("<I", bytes(target.read_memory_block8(addr, 4)))[0]


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    with ConnectHelper.session_with_chosen_probe(
            options={"connect_mode": "attach", "frequency": 1000000}) as session:
        target = session.target
        target.init()
        print("watching for main-loop freeze (%.0f s)..." % duration)
        start = time.monotonic()
        last_ticks = read_u32(target, TICKS_ADDR)
        last_change = time.monotonic()
        while time.monotonic() - start < duration:
            time.sleep(0.05)
            t = read_u32(target, TICKS_ADDR)
            now = time.monotonic()
            if t != last_ticks:
                last_ticks = t
                last_change = now
                continue
            if now - last_change > FREEZE_S:
                print("freeze detected (ticks stuck at %d) — halting core" % t)
                target.halt()
                time.sleep(0.1)
                regs = {}
                for name in ("r0", "r1", "r2", "r3", "r12", "sp", "lr", "pc", "xpsr"):
                    try:
                        regs[name] = target.read_core_register(name)
                    except Exception as exc:  # noqa
                        regs[name] = str(exc)
                for name, value in regs.items():
                    print(f"  {name:5s} = 0x{value:08x}" if isinstance(value, int)
                          else f"  {name:5s} = {value}")
                target.resume()
                return
        print("no freeze in window")


if __name__ == "__main__":
    main()
