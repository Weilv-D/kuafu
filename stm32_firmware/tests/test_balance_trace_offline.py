import importlib.util
import struct
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "MDK-ARM" / "debug_tools" / "dump_balance_trace.py"
spec = importlib.util.spec_from_file_location("dump_balance_trace", SCRIPT)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def packed(seq, timestamp):
    values = [timestamp, seq, 2, 3, 4, 1, 0, 2, 2, 63, 0]
    values += [0.1, -0.2, 0.5, 0.6, seq + 0.1, seq + 0.2,
               seq + 0.3, seq + 0.4, seq + 0.5, seq + 0.6]
    return trace.SAMPLE_STRUCT.pack(*values)


def test_sample_struct_matches_firmware_layout():
    # balance_trace.h static-asserts sizeof(BalanceTraceSample_t) == 80.
    assert trace.SAMPLE_SIZE == 80


def buffer_with(*samples):
    raw = bytearray(trace.TRACE_LEN * trace.SAMPLE_SIZE)
    for index, sample in enumerate(samples):
        raw[index * trace.SAMPLE_SIZE:(index + 1) * trace.SAMPLE_SIZE] = sample
    return bytes(raw)


def test_empty_ring():
    decoded = trace.decode_trace_buffer(buffer_with(), 0, 0, 0)
    assert decoded["samples"] == []
    assert not decoded["is_full"]


def test_partial_ring_starts_at_zero_not_head():
    raw = buffer_with(packed(0, 100), packed(1, 104), packed(2, 108))
    decoded = trace.decode_trace_buffer(raw, 3, 3, 3)
    assert [sample["seq"] for sample in decoded["samples"]] == [0, 1, 2]


def test_full_wrapped_ring_starts_at_next_write_head():
    raw = bytearray(trace.TRACE_LEN * trace.SAMPLE_SIZE)
    for index in range(trace.TRACE_LEN):
        raw[index * trace.SAMPLE_SIZE:(index + 1) * trace.SAMPLE_SIZE] = packed(index + 5, index)
    decoded = trace.decode_trace_buffer(bytes(raw), 5, trace.TRACE_LEN, trace.TRACE_LEN + 5)
    assert decoded["samples"][0]["seq"] == 10
    assert decoded["samples"][-1]["seq"] == 9


def test_rejects_truncated_buffer():
    try:
        trace.decode_trace_buffer(b"", 0, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("truncated trace must be rejected")
