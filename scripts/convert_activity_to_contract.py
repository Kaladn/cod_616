#!/usr/bin/env python3
"""Convert legacy activity JSONL files into Temporal Contract v1 JSONL (activity stream).

Usage:
  python scripts/convert_activity_to_contract.py input.jsonl [--out output.jsonl] [--session-id SESSION]
  or
  python scripts/convert_activity_to_contract.py --dir CompuCogLogger/logs/activity

Behavior:
- Emits session_start (seq=1)
- Wraps each line as a contract 'activity' event
- Emits session_end
- t_monotonic_ns captured at read time; t_utc is current UTC
- payload contains original object under 'payload.activity'
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
        'payload': payload if isinstance(payload, dict) else {'value': payload},
        'meta': {
            'schema_version': schema_version,
            'converter_version': CONVERTER_VERSION,
        }
    }
    if event_type:
        event['payload']['type'] = event_type
    return event


def convert_file(input_path: Path, output_path: Path, session_id: str = None, source: str = 'activity') -> int:
    session_id = session_id or f"session_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    seq = 1

    # session_start
    start_payload = {'type':'session_start','start_time_utc': now_t_utc()}
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
                err_payload = {'type':'error','message': f'parse_error: {e}', 'raw_line': line[:400]}
                evt = make_event(source, session_id, seq, err_payload, event_type='error')
                out_f.write(json.dumps(evt, ensure_ascii=False) + "\n")
                seq += 1
                continue

            payload = {
                'type': 'activity',
                'activity': original
            }
            # mark converted origin
            payload['activity'].setdefault('_converted_from', input_path.name)
            evt = make_event(source, session_id, seq, payload, event_type='activity')
            out_f.write(json.dumps(evt, ensure_ascii=False) + "\n")
            seq += 1

        end_payload = {'type':'session_end', 'final_seq': seq - 1, 'reason':'completed', 'end_time_utc': now_t_utc()}
        end_evt = make_event(source, session_id, seq, end_payload, event_type='session_end')
        out_f.write(json.dumps(end_evt, ensure_ascii=False) + "\n")

    return seq - 1


def main():
    p = argparse.ArgumentParser(description='Convert legacy activity JSONL to Temporal Contract v1 JSONL')
    p.add_argument('input', type=str, nargs='?', default=None, help='input legacy JSONL file or directory (defaults to CompuCogLogger/logs/activity)')
    p.add_argument('--out', type=str, default=None, help='output contract JSONL (default: <input>.contract.jsonl)')
    p.add_argument('--session-id', type=str, default=None, help='optional session id to use')
    args = p.parse_args()

    if args.input:
        inp = Path(args.input)
        if not inp.exists():
            print(f"Input not found: {inp}")
            return 1
        inputs = [inp] if inp.is_file() else list(inp.glob('**/*.jsonl'))
    else:
        dirp = Path('CompuCogLogger/logs/activity')
        inputs = list(dirp.glob('**/*.jsonl'))

    if not inputs:
        print('No input files found')
        return 1

    for inp in inputs:
        outp = Path(args.out) if args.out else inp.with_suffix('.contract.jsonl')
        print(f"Converting {inp} -> {outp}")
        total = convert_file(inp, outp, session_id=args.session_id)
        print(f"Wrote {total} events to {outp}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())