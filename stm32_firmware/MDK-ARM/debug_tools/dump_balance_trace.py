"""Dump the SWD-visible balance trace without disturbing the motion CPU.

The firmware records a 256-sample, 250 Hz window (1.024 s) with a 64-sample
(256 ms) post-fault tail.  A snapshot is accepted only when the writer seqlock
is even and unchanged before/after the buffer read.  The probe ATTACHES: it
never resets, halts, or resumes the target, so a balancing robot keeps running.

Field semantics (see Core/Inc/balance_trace.h):
- mode/startup_state/actuator_state are raw firmware enum values
  (RobotMode_t / StartupPhase_t / ActuatorSupervisorPhase_t), not new classes.
- fault_mask is the full 32-bit SafetyState fault mask.
- All torques are wheel/body-frame Nm.  target = LQR output, sent = value of
  the frame the DDSM queue actually accepted, feedback = motor-reported torque
  mapped back to body frame.
- timestamp_ms is firmware HAL_GetTick time; seq is the writer sequence.

Usage: python dump_balance_trace.py [out.csv]
       Writes out.csv plus out.csv.meta.json (provenance sidecar).
"""
import csv
import importlib.util
import json
import struct
import sys
from pathlib import Path

TRACE_LEN = 256
TRACE_VERSION = 2
TEXT_LEN = 32
# C layout: uint32 ts, uint32 seq, 3*uint16 ages, 2 pad bytes, then six
# uint32 fields (mode, fault_mask, startup_state, actuator_state, validity,
# event), then 10 floats.  Total 80 bytes, matching the static assert in
# balance_trace.h.
SAMPLE_STRUCT = struct.Struct("<IIHHH2x6I10f")
SAMPLE_SIZE = SAMPLE_STRUCT.size

FIELD_NAMES = [
    "timestamp_ms", "seq", "imu_age_ms", "wheel_left_age_ms",
    "wheel_right_age_ms", "mode", "fault_mask", "startup_state",
    "actuator_state", "validity", "event", "pitch_rad", "pitch_rate_rad_s",
    "wheel_left_rad_s", "wheel_right_rad_s", "target_torque_left_nm",
    "target_torque_right_nm", "sent_torque_left_nm", "sent_torque_right_nm",
    "feedback_torque_left_nm", "feedback_torque_right_nm",
]


def decode_sample(raw, offset=0):
    values = SAMPLE_STRUCT.unpack_from(raw, offset)
    return dict(zip(FIELD_NAMES, values))


def decode_trace_buffer(raw, head, count, total_records, version=TRACE_VERSION,
                        frozen=False, firmware_version="unknown", build_id="unknown"):
    """Pure offline decoder used by tests and by the SWD dump path."""
    if version != TRACE_VERSION:
        raise ValueError("unsupported trace version: %s" % version)
    if not 0 <= head < TRACE_LEN or not 0 <= count <= TRACE_LEN:
        raise ValueError("invalid head/count")
    if len(raw) < TRACE_LEN * SAMPLE_SIZE:
        raise ValueError("trace buffer is truncated")
    # Partial ring: valid slots start at 0.  Full/wrapped ring: the oldest
    # live sample is the next-write head.
    start = head if count == TRACE_LEN else 0
    samples = [decode_sample(raw, ((start + i) % TRACE_LEN) * SAMPLE_SIZE)
               for i in range(count)]
    return {
        "version": version,
        "head": head,
        "count": count,
        "total_records": total_records,
        "is_full": count == TRACE_LEN,
        "frozen": bool(frozen),
        "firmware_version": firmware_version,
        "build_id": build_id,
        "samples": samples,
    }


def read_consistent_snapshot(target, addresses, retries=5):
    """Read a coherent snapshot using the firmware seqlock; attach only, never halt."""
    for _ in range(retries):
        seq0 = struct.unpack("<I", bytes(target.read_memory_block8(addresses["snapshot_seq"], 4)))[0]
        if seq0 & 1:
            continue
        head = struct.unpack("<H", bytes(target.read_memory_block8(addresses["head"], 2)))[0]
        count = struct.unpack("<H", bytes(target.read_memory_block8(addresses["count"], 2)))[0]
        total = struct.unpack("<I", bytes(target.read_memory_block8(addresses["seq"], 4)))[0]
        frozen = bytes(target.read_memory_block8(addresses["frozen"], 1))[0]
        version = struct.unpack("<I", bytes(target.read_memory_block8(addresses["version"], 4)))[0]
        raw = bytes(target.read_memory_block8(addresses["buf"], TRACE_LEN * SAMPLE_SIZE))
        seq1 = struct.unpack("<I", bytes(target.read_memory_block8(addresses["snapshot_seq"], 4)))[0]
        if seq0 == seq1 and not (seq1 & 1):
            fw = bytes(target.read_memory_block8(addresses["firmware"], TEXT_LEN)).split(b"\0", 1)[0].decode("ascii", "replace")
            build = bytes(target.read_memory_block8(addresses["build"], TEXT_LEN)).split(b"\0", 1)[0].decode("ascii", "replace")
            snapshot = decode_trace_buffer(raw, head, count, total, version, frozen, fw, build)
            snapshot["snapshot_seq"] = seq0
            return snapshot
    raise RuntimeError("trace changed during SWD snapshot; retry without halting target")


def write_csv(path, snapshot):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELD_NAMES)
        writer.writeheader()
        writer.writerows(snapshot["samples"])


def write_sidecar(path, snapshot):
    """Persist provenance next to the CSV so a dump is never anonymous."""
    meta = {key: snapshot[key] for key in
            ("version", "head", "count", "total_records", "is_full", "frozen",
             "firmware_version", "build_id")}
    meta["snapshot_seq"] = snapshot.get("snapshot_seq")
    meta["sample_size_bytes"] = SAMPLE_SIZE
    meta["field_names"] = FIELD_NAMES
    with open(str(path) + ".meta.json", "w") as stream:
        json.dump(meta, stream, indent=2)


def main():
    from pyocd.core.helpers import ConnectHelper

    spec = importlib.util.spec_from_file_location(
        "imu_tool", str(Path(__file__).resolve().parent / "read_imu_state.py"))
    imu_tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imu_tool)
    symbol_address = imu_tool.symbol_address
    probe = "LU_2022_8888"
    target_name = "stm32f407zgtx"
    addresses = {
        "buf": symbol_address("g_bt_buf"),
        "head": symbol_address("g_bt_head"),
        "count": symbol_address("g_bt_count"),
        "seq": symbol_address("g_bt_seq"),
        "snapshot_seq": symbol_address("g_bt_snapshot_seq"),
        "frozen": symbol_address("g_bt_frozen"),
        "version": symbol_address("g_bt_trace_version"),
        "firmware": symbol_address("g_bt_firmware_version"),
        "build": symbol_address("g_bt_build_id"),
    }
    out = sys.argv[1] if len(sys.argv) > 1 else "balance_trace.csv"
    # connect_mode="attach": the target must keep running.  No reset, no halt,
    # no resume — the robot may be balancing while this dump is taken.
    with ConnectHelper.session_with_chosen_probe(
            blocking=False, unique_id=probe, target_override=target_name,
            options={"connect_mode": "attach", "frequency": 1000000}) as session:
        if session is None:
            print("ERROR: probe not found")
            return 1
        snapshot = read_consistent_snapshot(session.target, addresses)

    write_csv(out, snapshot)
    write_sidecar(out, snapshot)
    if snapshot["count"] == 0:
        print("trace is EMPTY (count=0, total=%d); no control ticks recorded"
              % snapshot["total_records"])
        print("wrote %s + sidecar" % out)
        return 0

    print("trace_version=%d firmware=%s build=%s head=%d count=%d total=%d frozen=%s" % (
        snapshot["version"], snapshot["firmware_version"], snapshot["build_id"],
        snapshot["head"], snapshot["count"], snapshot["total_records"], snapshot["frozen"]))
    samples = snapshot["samples"]
    spans = [b - a for a, b in zip([s["timestamp_ms"] for s in samples[:-1]],
                                   [s["timestamp_ms"] for s in samples[1:]])]
    if spans:
        print("dt(ms) min/median/max = %d/%d/%d" %
              (min(spans), sorted(spans)[len(spans) // 2], max(spans)))
    max_pitch = max(abs(s["pitch_rad"]) for s in samples)
    max_target = max(max(abs(s["target_torque_left_nm"]), abs(s["target_torque_right_nm"])) for s in samples)
    max_feedback = max(max(abs(s["feedback_torque_left_nm"]), abs(s["feedback_torque_right_nm"])) for s in samples)
    print("max|pitch|=%.4f rad, max|target torque|=%.4f Nm, max|feedback torque|=%.4f Nm" %
          (max_pitch, max_target, max_feedback))
    print("wrote %s (%d samples) + sidecar" % (out, len(samples)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
