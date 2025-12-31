import time
from pathlib import Path
from datetime import datetime
import tempfile

import pytest
from unittest.mock import patch

from loggers.activity_service import ActivityService, IDLE_THRESHOLD_PULSES, STALL_THRESHOLD_PULSES


def test_detects_file_growth():
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / "logs" / "activity"
        log_dir.mkdir(parents=True)
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"user_activity_{today}.jsonl"
        log_file.write_text('{"test": "initial"}\n')

        with patch('loggers.activity_service.get_activity_log_path', return_value=log_file):
            svc = ActivityService()
            svc.start()

            # first poll - initial snapshot
            assert svc.poll() is None

            # append to file
            with open(log_file, 'a', encoding='utf-8') as fh:
                fh.write('{"test": "new_data"}\n')

            ev = svc.poll()
            assert ev is not None
            assert ev['event_type'] == 'activity_detected'
            assert ev['file_size'] > 0

            svc.stop()


def test_emits_idle_and_stalled_after_thresholds():
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / "logs" / "activity"
        log_dir.mkdir(parents=True)
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"user_activity_{today}.jsonl"
        log_file.write_text('{"test": "data"}\n')

        with patch('loggers.activity_service.get_activity_log_path', return_value=log_file):
            svc = ActivityService()
            svc.start()

            # initial snapshot consumes first poll
            assert svc.poll() is None

            ev = None
            # poll IDLE_THRESHOLD_PULSES times
            for _ in range(IDLE_THRESHOLD_PULSES):
                ev = svc.poll()
            assert ev is not None
            assert ev['event_type'] == 'activity_idle'

            # continue to STALL threshold
            for _ in range(STALL_THRESHOLD_PULSES - IDLE_THRESHOLD_PULSES):
                ev = svc.poll()
            assert ev is not None
            assert ev['event_type'] == 'activity_stalled'

            svc.stop()