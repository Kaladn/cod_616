import json
import os
import time
from tempfile import TemporaryDirectory
from hashlib import sha256

from loggers.writer import DailyJSONWriter


def test_writer_smoke_basic_chain_and_canonical():
    with TemporaryDirectory() as td:
        w = DailyJSONWriter(category='activity', prefix='act_smoke', data_root=td, pulse_interval=0.05)
        w.start()
        try:
            # deterministic events
            events = []
            for i in range(5):
                ev = {"id": i, "value": f"msg-{i}", "timestamp": 1700000000000 + i}
                events.append(ev)
                w.enqueue(ev)
            # allow writer to pulse a few times
            time.sleep(0.6)
        finally:
            w.stop()

        # locate file
        activity_dir = os.path.join(td, 'activity')
        files = [f for f in os.listdir(activity_dir) if f.startswith('act_smoke_') and f.endswith('.jsonl')]
        assert len(files) == 1, f"expected one file, found: {files}"
        path = os.path.join(activity_dir, files[0])

        lines = []
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                lines.append(line)

        assert len(lines) == len(events)

        prev_sha = '0' * 64
        for raw_line in lines:
            line = raw_line.rstrip('\n')
            raw_bytes = line.encode('utf-8')

            # parse object
            obj = json.loads(line)
            assert '_sha' in obj
            sha_given = obj['_sha']

            # canonicalize payload (object without _sha)
            payload_obj = dict(obj)
            payload_obj.pop('_sha')
            payload_bytes = json.dumps(payload_obj, separators=(',', ':'), sort_keys=True, ensure_ascii=False).encode('utf-8')

            # verify sha chain
            m = sha256()
            m.update(prev_sha.encode('ascii'))
            m.update(payload_bytes)
            assert m.hexdigest() == sha_given

            # verify raw line bytes match canonical serialization of full object
            full_canonical = json.dumps(obj, separators=(',', ':'), sort_keys=True, ensure_ascii=False).encode('utf-8')
            assert full_canonical == raw_bytes

            prev_sha = sha_given
