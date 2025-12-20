#!/usr/bin/env python3
"""Convert legacy gamepad metric JSONL files into Temporal Contract v1 JSONL (gamepad stream).

Usage:
  python scripts/convert_gamepad_to_contract.py input.jsonl [--out output.jsonl] [--session-id SESSION]
  or
  python scripts/convert_gamepad_to_contract.py --dir CompuCogLogger/logs/gamepad

Behavior:
- Emits session_start event (seq=1)
- Emits a connect event for each detected device (based on gamepad_count) before state events
- Converts each metrics line into a contract 'state' event with payload.type='state'
- device_seq increments per device per session
- t_monotonic_ns captured at read time; t_utc is current UTC
- meta.schema_version default '1.0'
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import uuid
import hashlib

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


def deterministic_device_id(index=0):
    return f"gamepad::index_{index}"


def capabilities_hash_from_metrics(sample):
    # Derive a simple capabilities hash from sorted keys of sample
    keys = sorted(sample.keys())
    h = hashlib.sha256()
    h.update('\n'.join(keys).encode('utf-8'))
    return 'sha256:' + h.hexdigest()[:16]


def convert_file(input_path: Path, output_path: Path, session_id: str = None, source: str = 'gamepad') -> int:
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

        # For this legacy metrics stream, assume a single device when gamepad_count >=1
        device_guid = str(uuid.uuid4())
        device_id = deterministic_device_id(0)
        device_index = 0
        device_type = 'generic'
        capabilities_hash = None

        # Emit connect event
        connect_payload = {
            'type': 'connect',
            'gamepad': {
                'device_id': device_id,
                'device_guid': device_guid,
                'device_index': device_index,
                'device_type': device_type,
                'vendor_id': None,
                'product_id': None,
                'driver': None,
                'os_path': None,
                'capabilities_hash': None
            }
        }
        evt = make_event(source, session_id, seq, connect_payload, event_type='connect')
        out_f.write(json.dumps(evt, ensure_ascii=False) + "\n")
        seq += 1

        device_seq = 0
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            try:
                original = json.loads(line)
            except Exception as e:
                err_payload = {'type': 'error', 'message': f'parse_error: {e}', 'raw_line': line[:400]}
                evt = make_event(source, session_id, seq, err_payload, event_type='error')
                out_f.write(json.dumps(evt, ensure_ascii=False) + "\n")
                seq += 1
                continue

            # Build state payload
            device_seq += 1
            capabilities_hash = capabilities_hash_from_metrics(original)
            state_payload = {
                'type': 'state',
                'sample_count': original.get('left_stick_samples', 0) + original.get('right_stick_samples', 0),
                'sample_span_ns': None,
                'gamepad': {
                    'device_id': device_id,
                    'device_guid': device_guid,
                    'device_index': device_index,
                    'device_type': device_type,
                    'vendor_id': None,
                    'product_id': None,
                    'driver': None,
                    'os_path': None,
                    'capabilities_hash': capabilities_hash,
                    'device_seq': device_seq
                },
                'metrics': original
            }
            evt = make_event(source, session_id, seq, state_payload, event_type='state')
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
    p = argparse.ArgumentParser(description='Convert legacy gamepad metrics JSONL to Temporal Contract v1 JSONL')
    p.add_argument('input', type=str, nargs='?', default=None, help='input legacy JSONL file or directory (defaults to CompuCogLogger/logs/gamepad)')
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
        # default directory
        dirp = Path('CompuCogLogger/logs/gamepad')
        inputs = list(dirp.glob('**/*.jsonl'))

    if not inputs:
        print("No input files found")
        return 1

    for inp in inputs:
        outp = Path(args.out) if args.out else inp.with_suffix('.contract.jsonl')
        print(f"Converting {inp} -> {outp}")
        total = convert_file(inp, outp, session_id=args.session_id)
        print(f"Wrote {total} events to {outp}")

    return 0


if __name__ == '__main__':
    import argparse
    raise SystemExit(main())
