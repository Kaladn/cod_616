import os
import time
from tempfile import TemporaryDirectory
from collections import namedtuple

import json
from resilience.disk_guard import DiskGuard

NTuple = namedtuple('usage', ['total', 'used', 'free'])


def fake_disk_usage_factory(free_values):
    # returns a function that pops free values sequentially; when exhausted, returns last value repeatedly
    values = list(free_values)
    last = free_values[-1]
    def fn(path):
        if values:
            v = values.pop(0)
        else:
            v = last
        return NTuple(total=1000, used=1000 - v, free=v)
    return fn


def test_disk_guard_thresholds_and_warning_emission():
    with TemporaryDirectory() as td:
        # create fake logs_data structure with per-day files
        os.makedirs(os.path.join(td, 'activity'), exist_ok=True)
        # create three days of files with sizes 100, 200, 300 bytes
        days = ['12-24-25', '12-25-25', '12-26-25']
        sizes = [100, 200, 300]
        for d, s in zip(days, sizes):
            p = os.path.join(td, 'activity', f'act_{d}.jsonl')
            with open(p, 'wb') as f:
                f.write(b'x' * s)

        # avg_daily_usage = (100+200+300)/3 = 200
        avg = 200
        # thresholds: warning < 3*200=600, critical <1*200=200
        # simulate free values: 1000 (ok), 500 (warning), 150 (critical)
        fake_fn = fake_disk_usage_factory([1000, 500, 150])

        dg = DiskGuard(logs_path=td, check_interval=0.1, days=3, disk_usage_fn=fake_fn)
        dg.start()
        try:
            # allow a few cycles
            time.sleep(0.4)
        finally:
            dg.stop()

        # check that some severity was recorded (warning or critical)
        stats = dg.stats()
        assert stats['last_severity'] in ('warning', 'critical')

        # check system file exists and contains at least one warning and one critical entry
        sysdir = os.path.join(td, 'system')
        files = [os.path.join(sysdir, f) for f in os.listdir(sysdir) if f.startswith('syswarn_')]
        assert files
        severities = set()
        for fn in files:
            with open(fn, 'r', encoding='utf-8') as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        severities.add(obj.get('severity'))
                    except Exception:
                        continue
        assert severities & {'warning', 'critical'}


def test_disk_guard_handles_errors_and_continues():
    with TemporaryDirectory() as td:
        # no log files -> fallback avg
        # simulate error first, then low free value
        calls = [None, 100]  # first call raises, second returns free=100
        def flaky_fn(path):
            v = calls.pop(0)
            if v is None:
                raise RuntimeError('disk check failed')
            return NTuple(total=1000, used=1000-v, free=v)

        dg = DiskGuard(logs_path=td, check_interval=0.1, days=3, disk_usage_fn=flaky_fn)
        dg.start()
        try:
            time.sleep(0.3)
        finally:
            dg.stop()

        # ensure it recorded something
        stats = dg.stats()
        assert stats['last_avg_daily_usage_bytes'] is not None
        # system dir should exist and have a warning file
        sysdir = os.path.join(td, 'system')
        files = os.listdir(sysdir)
        assert any(f.startswith('syswarn_') for f in files)
