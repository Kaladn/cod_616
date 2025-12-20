#!/usr/bin/env python3
"""Validate JSONL event streams against Temporal Contract v1.

Usage: python scripts/validate_temporal_contract.py path/to/file.jsonl [more.jsonl]
Or: python scripts/validate_temporal_contract.py path/to/dir (will scan *.jsonl)

Exit code: 0 = all good, 2 = validation errors found, 1 = usage/error
"""
import sys
import json
from pathlib import Path
from datetime import datetime

REQ_FIELDS = [
    'source', 'session_id', 'seq', 't_monotonic_ns', 't_utc', 'event_id', 'payload', 'meta'
]

GAMEPAD_TYPES = {'button', 'axis', 'connection', 'heartbeat'}

errors = []


def parse_t_utc(s):
    try:
        # Accept trailing Z
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        datetime.fromisoformat(s)
        return True
    except Exception:
        return False


def validate_event(obj, fname, lineno):
    e = []
    for f in REQ_FIELDS:
        if f not in obj:
            e.append(f"missing_field:{f}")
    if e:
        return e
    # type checks
    if not isinstance(obj['source'], str):
        e.append('bad_type:source')
    if not isinstance(obj['session_id'], str):
        e.append('bad_type:session_id')
    if not isinstance(obj['event_id'], str):
        e.append('bad_type:event_id')
    if not isinstance(obj['seq'], int) or obj['seq'] <= 0:
        e.append('bad_seq')
    if not isinstance(obj['t_monotonic_ns'], int) or obj['t_monotonic_ns'] <= 0:
        e.append('bad_t_monotonic_ns')
    if not isinstance(obj['meta'], dict) or 'schema_version' not in obj['meta']:
        e.append('missing_meta.schema_version')
    if not isinstance(obj['payload'], dict):
        e.append('bad_type:payload')
    # t_utc parseable
    if not parse_t_utc(obj['t_utc']):
        e.append('bad_t_utc')
    # gamepad addendum checks
    if obj['source'] == 'gamepad':
        p = obj['payload']
        if not isinstance(p, dict):
            e.append('gamepad:bad_payload')
        else:
            device = p.get('device')
            if not (isinstance(device, dict) and device.get('device_class') == 'gamepad'):
                e.append('gamepad:missing_device')
            t = p.get('type')
            if t not in GAMEPAD_TYPES:
                e.append('gamepad:bad_type')
            if t == 'button':
                btn = p.get('button')
                if not (isinstance(btn, dict) and 'id' in btn and 'state' in btn):
                    e.append('gamepad:bad_button')
            if t == 'axis':
                axis = p.get('axis')
                if not (isinstance(axis, dict) and 'id' in axis and 'value' in axis):
                    e.append('gamepad:bad_axis')
    return e


def scan_path(p: Path):
    found = 0
    for path in (p.glob('**/*.jsonl') if p.is_dir() else [p]):
        if not path.exists():
            errors.append((str(path), 0, ['file_not_found']))
            continue
        with path.open('r', encoding='utf-8') as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                found += 1
                try:
                    obj = json.loads(line)
                except Exception as ex:
                    errors.append((str(path), i, [f'invalid_json:{ex}']))
                    continue
                e = validate_event(obj, str(path), i)
                if e:
                    errors.append((str(path), i, e))
    return found


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    total = 0
    for arg in argv[1:]:
        p = Path(arg)
        if p.is_dir():
            print(f"Scanning dir: {p}")
            for f in p.glob('**/*.jsonl'):
                print(f"  found: {f}")
            total += scan_path(p)
        else:
            total += scan_path(p)
    if not errors:
        print(f"Validation OK: {total} events scanned, no errors")
        return 0
    else:
        print(f"Validation FAILED: {len(errors)} errors across {total} events")
        for fname, lineno, errs in errors:
            print(f"{fname}:{lineno} -> {errs}")
        return 2


if __name__ == '__main__':
    rc = main(sys.argv)
    sys.exit(rc)
