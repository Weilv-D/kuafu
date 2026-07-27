"""Read the HardFault post-mortem dump at 0x2001FFF0 (see stm32f4xx_it.c)."""

import struct

from pyocd.core.helpers import ConnectHelper

with ConnectHelper.session_with_chosen_probe(
        options={"connect_mode": "attach", "frequency": 1000000}) as session:
    target = session.target
    target.init()
    magic, pc, lr, cfsr = struct.unpack(
        "<IIII", bytes(target.read_memory_block8(0x2001FFF0, 16)))
    print(f"magic = 0x{magic:08x}")
    if magic == 0xDEADF417:
        print(f"PC    = 0x{pc:08x}  (faulting instruction)")
        print(f"LR    = 0x{lr:08x}")
        print(f"CFSR  = 0x{cfsr:08x}")
        bits = []
        if cfsr & (1 << 25): bits.append("DIVBYZERO")
        if cfsr & (1 << 24): bits.append("UNALIGNED")
        if cfsr & (1 << 19): bits.append("INVPC")
        if cfsr & (1 << 18): bits.append("INVSTATE")
        if cfsr & (1 << 17): bits.append("UNDEFINSTR")
        if cfsr & (1 << 16): bits.append("BFARVALID")
        if cfsr & (1 << 15): bits.append("BFARVALID?")
        if cfsr & (1 << 9): bits.append("IBUSERR")
        if cfsr & (1 << 10): bits.append("IMPRECISERR")
        if cfsr & (1 << 11): bits.append("PRECISERR")
        if cfsr & (1 << 1): bits.append("MMARVALID?")
        print("flags =", ", ".join(bits) if bits else "none")
        print("hint: find PC in stm32_firmware.map / .lst to identify the function")
    else:
        print("no fault recorded (magic mismatch) — the hangs are NOT CPU faults")
