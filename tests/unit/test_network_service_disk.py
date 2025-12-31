import tempfile
from pathlib import Path
from datetime import datetime

from unittest.mock import patch

from loggers.network_service import NetworkService


def test_detects_growth_and_missing():
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / "logs" / "network"
        log_dir.mkdir(parents=True)
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"network_capture_{today}.jsonl"

        # missing file -> network_failed
        with patch('loggers.network_service.get_network_log_path', return_value=log_file):
            svc = NetworkService()
            svc.start()

            ev = svc.poll()
            assert ev is not None
            assert ev['event_type'] == 'network_failed'

            # create file
            log_file.write_text('{"pkt": 1}\n')
            assert svc.poll() is None

            # append more data
            with open(log_file, 'a', encoding='utf-8') as fh:
                fh.write('{"pkt": 2}\n')

            ev = svc.poll()
            assert ev is not None
            assert ev['event_type'] == 'network_active'

            svc.stop()


def test_emits_idle_and_stalled():
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / "logs" / "network"
        log_dir.mkdir(parents=True)
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"network_capture_{today}.jsonl"
        log_file.write_text('{"pkt": 1}\n')

        with patch('loggers.network_service.get_network_log_path', return_value=log_file):
            svc = NetworkService()
            svc.start()

            # initial snapshot
            assert svc.poll() is None

            ev = None
            for _ in range(50):
                ev = svc.poll()
            assert ev is not None
            assert ev['event_type'] == 'network_idle'

            for _ in range(50):
                ev = svc.poll()
            assert ev is not None
            assert ev['event_type'] == 'network_stalled'

            svc.stop()