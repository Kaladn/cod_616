import time
import tempfile
from pathlib import Path
from datetime import datetime

from unittest.mock import patch

from loggers.activity_service import ActivityService
from loggers.input_service import InputService
from loggers.network_service import NetworkService


def test_services_integration_short_run():
    with tempfile.TemporaryDirectory() as td:
        # create dirs and files
        activity_dir = Path(td) / "logs" / "activity"
        input_dir = Path(td) / "logs" / "input"
        network_dir = Path(td) / "logs" / "network"
        activity_dir.mkdir(parents=True)
        input_dir.mkdir(parents=True)
        network_dir.mkdir(parents=True)

        today = datetime.now().strftime("%Y%m%d")
        activity_file = activity_dir / f"user_activity_{today}.jsonl"
        input_file = input_dir / f"input_events_{today}.jsonl"
        network_file = network_dir / f"network_capture_{today}.jsonl"

        # write initial content
        activity_file.write_text('{"a":1}\n')
        input_file.write_text('{"i":1}\n')
        network_file.write_text('{"n":1}\n')

        with patch('loggers.activity_service.get_activity_log_path', return_value=activity_file), \
             patch('loggers.input_service.get_input_log_path', return_value=input_file), \
             patch('loggers.network_service.get_network_log_path', return_value=network_file):

            act = ActivityService(); act.start()
            inp = InputService(); inp.start()
            net = NetworkService(); net.start()

            events = {"activity": [], "input": [], "network": []}

            # Append new lines and poll repeatedly for a short interval
            start = time.time()
            while time.time() - start < 1.0:
                # append to files occasionally
                activity_file.write_text(activity_file.read_text() + '{"a":2}\n')
                input_file.write_text(input_file.read_text() + '{"i":2}\n')
                network_file.write_text(network_file.read_text() + '{"n":2}\n')

                e = act.poll()
                if e:
                    events['activity'].append(e)
                ei = inp.poll()
                if ei:
                    events['input'].extend(ei)
                en = net.poll()
                if en:
                    events['network'].append(en)

                time.sleep(0.05)

            act.stop(); inp.stop(); net.stop()

            # ensure some events were observed
            assert len(events['activity']) >= 1
            assert len(events['input']) >= 1
            assert len(events['network']) >= 1

            valid_activity = {"activity_detected", "activity_idle", "activity_stalled", "activity_failed"}
            valid_input = {"input_event", "input_idle", "input_stalled", "input_failed"}
            valid_network = {"network_active", "network_idle", "network_stalled", "network_failed"}

            for a in events['activity']:
                assert a['event_type'] in valid_activity
            for i in events['input']:
                assert i['event_type'] in valid_input
            for n in events['network']:
                assert n['event_type'] in valid_network