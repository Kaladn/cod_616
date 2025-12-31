import tempfile
from pathlib import Path
from datetime import datetime
import time

from unittest.mock import patch

from loggers.input_service import InputService


def test_reads_appended_records():
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / "logs" / "input"
        log_dir.mkdir(parents=True)
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"input_events_{today}.jsonl"
        log_file.write_text('{"a": 1}\n')

        with patch('loggers.input_service.get_input_log_path', return_value=log_file):
            svc = InputService()
            svc.start()

            # initial poll consumes snapshot
            assert svc.poll() is None

            # append new records
            with open(log_file, 'a', encoding='utf-8') as fh:
                fh.write('{"b": 2}\n')
                fh.write('{"c": 3}\n')

            events = svc.poll()
            assert events is not None
            assert len(events) == 2
            assert events[0]['event_type'] == 'input_event'
            assert events[0]['record'] == {'b': 2}

            svc.stop()


def test_emits_input_idle_and_failed():
    with tempfile.TemporaryDirectory() as td:
        # missing file -> input_failed
        log_dir = Path(td) / "logs" / "input"
        log_dir.mkdir(parents=True)
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"input_events_{today}.jsonl"

        with patch('loggers.input_service.get_input_log_path', return_value=log_file):
            svc = InputService()
            svc.start()

            ev = svc.poll()
            assert ev is not None
            assert ev[0]['event_type'] == 'input_failed'

            # create file then poll initial snapshot
            log_file.write_text('{"x": 1}\n')
            assert svc.poll() is None

            # poll IDLE_THRESHOLD times to trigger idle
            for _ in range(51):
                ev = svc.poll()
            assert ev is not None
            assert ev[0]['event_type'] == 'input_idle'

            svc.stop()