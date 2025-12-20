#!/usr/bin/env python3
"""Forge ingestion verifier (test harness)

This script performs a deterministic verification that feeding contract JSONL
into a PulseWriter-like sink preserves event count, order, seq, and timestamps.

It uses a local in-memory `TestPulseWriter` that records appended events
so assertions are strict without requiring production BinaryLog changes.

Usage:
  python scripts/forge_ingest_verify.py path/to/sample.contract.jsonl [more.jsonl]

Exit codes:
  0 = success (all checks passed)
  2 = verification failures
  1 = usage / error
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class TestBinaryLog:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.closed = False

    def append(self, record: Dict[str, Any]):
        # Store the raw event dict for verification
        self.records.append(record)

    def get_record_count(self) -> int:
        return len(self.records)

    def close(self):
        self.closed = True


class TestPulseWriter:
    def __init__(self, binary_log: Optional[TestBinaryLog] = None):
        self.binary_log = binary_log or TestBinaryLog()
        self._buffer: List[Dict[str, Any]] = []

    def submit_event(self, event: Dict[str, Any]):
        # In a real system, mapping/transformation might occur here.
        # For verification we store the event as-is.
        self._buffer.append(event)
        # Simulate immediate write (or could batch and flush)
        self._flush_buffer()

    def _flush_buffer(self):
        if not self._buffer:
            return
        for r in self._buffer:
            self.binary_log.append(r)
        self._buffer.clear()

    def flush(self):
        self._flush_buffer()

    def close(self):
        self.flush()
        self.binary_log.close()


def read_contract(path: Path) -> List[Dict[str, Any]]:
    events = []
    with path.open('r', encoding='utf-8') as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as ex:
                raise RuntimeError(f"invalid json in {path}:{i}: {ex}")
            events.append(obj)
    return events


def verify_events_preserved(input_events: List[Dict[str, Any]], stored_events: List[Dict[str, Any]]) -> List[str]:
    errs = []

    if len(input_events) != len(stored_events):
        errs.append(f"count_mismatch: input={len(input_events)} stored={len(stored_events)}")
        # still proceed to compare as much as possible

    n = min(len(input_events), len(stored_events))
    for i in range(n):
        inp = input_events[i]
        out = stored_events[i]

        # Check ordering: same seq and session_id (best-effort)
        s1 = inp.get('session_id')
        s2 = out.get('session_id')
        if s1 != s2:
            errs.append(f"session_id_mismatch at idx {i}: {s1} != {s2}")

        seq1 = inp.get('seq')
        seq2 = out.get('seq')
        if seq1 != seq2:
            errs.append(f"seq_mismatch at idx {i}: {seq1} != {seq2}")

        # Timestamps
        t1 = inp.get('t_monotonic_ns')
        t2 = out.get('t_monotonic_ns')
        if t1 != t2:
            errs.append(f"t_monotonic_ns_mismatch at idx {i}: {t1} != {t2}")

        u1 = inp.get('t_utc')
        u2 = out.get('t_utc')
        if u1 != u2:
            errs.append(f"t_utc_mismatch at idx {i}: {u1} != {u2}")

    # Check monotonic device seq per source if present
    # For generic check: ensure top-level seq is monotonically increasing per source
    last_seq = None
    for i, obj in enumerate(stored_events):
        seq = obj.get('seq')
        if not isinstance(seq, int):
            errs.append(f"bad_seq_type at idx {i}: {seq}")
            continue
        if last_seq is not None and seq <= last_seq:
            errs.append(f"non_monotonic_seq at idx {i}: {seq} <= {last_seq}")
        last_seq = seq

    return errs


def run_verify(paths: List[Path]) -> int:
    overall_errors = []
    for p in paths:
        print(f"[1/3] Loading contract: {p}")
        events = read_contract(p)
        print(f"    Loaded {len(events)} events")

        print("[2/3] Ingesting into real PulseWriter (production code path)")
        # Use the real build pipeline; write to temp directory under ./forge_verify/<basename>/
        from forge_memory.live_forge_harness import build_truevision_forge_pipeline, PulseConfig
        import tempfile
        tmpd = Path("forge_verify") / p.stem
        tmpd.mkdir(parents=True, exist_ok=True)
        pw = build_truevision_forge_pipeline(data_dir=str(tmpd), pulse_config=PulseConfig(max_records_per_pulse=100), test_mode=True)

        # Ingest and flush periodically to simulate pulses
        for i, evt in enumerate(events, start=1):
            pw.submit_event(evt)
            if i % 100 == 0:
                pw.flush()
        pw.close()

        # Read stored records back from the BinaryLog file
        records_path = tmpd / 'records.jsonl'
        if not records_path.exists():
            print(f"    ERROR: records file not found at {records_path}")
            overall_errors.append(f"missing_records:{tmpd}")
            continue
        with records_path.open('r', encoding='utf-8') as fh:
            stored = [json.loads(line) for line in fh if line.strip()]
        print(f"    Stored {len(stored)} records in BinaryLog at {records_path}")

        print("[3/3] Verifying preservation (count, order, seq, timestamps)")
        errs = verify_events_preserved(events, stored)
        if errs:
            print(f"Verification FAILED for {p}: {len(errs)} errors")
            for e in errs[:20]:
                print("  -", e)
            if len(errs) > 20:
                print(f"  ... and {len(errs)-20} more errors")
            overall_errors.extend([f"{p}:{e}" for e in errs])
        else:
            print(f"Verification OK: {p} (all checks passed)")

    if overall_errors:
        print(f"\nVERIFICATION SUMMARY: FAIL ({len(overall_errors)} total errors)")
        return 2
    print("\nVERIFICATION SUMMARY: OK")
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    paths = []
    for a in argv[1:]:
        p = Path(a)
        if not p.exists():
            print(f"Not found: {p}")
            return 1
        paths.append(p)
    return run_verify(paths)


if __name__ == '__main__':
    rc = main(sys.argv)
    sys.exit(rc)
