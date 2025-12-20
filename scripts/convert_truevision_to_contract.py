#!/usr/bin/env python3
"""Convert legacy TrueVision JSONL telemetry into Temporal Contract v1 JSONL.

Usage:
  python scripts/convert_truevision_to_contract.py input.jsonl [--out output.jsonl] [--session-id SESSION]

Behavior:
- Emits session_start event (seq=1)
- Converts each line from the input into a contract event with source='truevision'
- Emits session_end event at the end
- t_monotonic_ns captured at read time (time.monotonic_ns())
- t_utc is current UTC in ISO8601 with microsecond precision
- payload contains the original parsed JSON object (verbatim)
- meta.schema_version default '1.0'
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import uuid

CONVERTER_VERSION = "0.1"
DEFAULT_SCHEMA = "1.0"


def now_t_utc():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def make_event(source, session_id, seq, payload, schema_version=DEFAULT_SCHEMA, event_type=None):
    event = {
        'source': source,
        'session_id': session_id,
        'seq': seq,
        't_monotonic_ns': int(time.monotonic_ns()),
        't_utc': now_t_utc(),
        'event_id': f"{session_id}:{source}:{seq:012d}",
        'payload': payload,
        'meta': {
            'schema_version': schema_version,
            'converter_version': CONVERTER_VERSION,
        }
    }
    if event_type:
        # Record type in payload for session_start/session_end etc.
        event['payload'] = payload if isinstance(payload, dict) else { 'value': payload }
        event['payload'].setdefault('type', event_type)
    return event


def convert_file(input_path: Path, output_path: Path, session_id: str = None, source: str = 'truevision') -> int:
    session_id = session_id or f"session_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    seq = 1
    # Emit session_start
    start_payload = {
        'type': 'session_start',
        'start_time_utc': now_t_utc(),
    }
    with output_path.open('w', encoding='utf-8') as out_f, input_path.open('r', encoding='utf-8') as in_f:
        start_event = make_event(source, session_id, seq, start_payload, event_type='session_start')
        out_f.write(json.dumps(start_event, ensure_ascii=False) + "\n")
        seq += 1

        for line in in_f:
            line = line.strip()
            if not line:
                continue
            try:
                original = json.loads(line)
            except Exception as e:
                # Preserve error as an error event rather than crashing
                err_payload = {'type': 'error', 'message': f'parse_error: {e}', 'raw_line': line[:400]}
                evt = make_event(source, session_id, seq, err_payload, event_type='error')
                out_f.write(json.dumps(evt, ensure_ascii=False) + "\n")
                seq += 1
                continue

            # Construct framed event preserving original data
            payload = original if isinstance(original, dict) else { 'value': original }
            # Keep a note that this was wrapped
            payload.setdefault('_converted_from', input_path.name)
            evt = make_event(source, session_id, seq, payload)
            out_f.write(json.dumps(evt, ensure_ascii=False) + "\n")
            seq += 1

        # session_end
        end_payload = {
            'type': 'session_end',
            'final_seq': seq - 1,
            'reason': 'completed',
            'end_time_utc': now_t_utc()
        }
        end_evt = make_event(source, session_id, seq, end_payload, event_type='session_end')
        out_f.write(json.dumps(end_evt, ensure_ascii=False) + "\n")

    return seq - 1


def main():
    p = argparse.ArgumentParser(description='Convert legacy TrueVision JSONL to Temporal Contract v1 JSONL')
    p.add_argument('input', type=str, help='input legacy JSONL')
    p.add_argument('--out', type=str, default=None, help='output contract JSONL (default: <input>.contract.jsonl)')
    p.add_argument('--session-id', type=str, default=None, help='optional session id to use')
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input file not found: {inp}")
        return 1
    outp = Path(args.out) if args.out else inp.with_suffix('.contract.jsonl')

    print(f"Converting {inp} -> {outp}")
    total = convert_file(inp, outp, session_id=args.session_id)
    print(f"Wrote {total} events to {outp}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
