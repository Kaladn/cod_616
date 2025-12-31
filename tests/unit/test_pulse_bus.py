import os
import time
from tempfile import TemporaryDirectory
from resilience.pulse_bus import PulseBus


def test_pulse_bus_basic_forward_and_drop():
    with TemporaryDirectory() as td:
        bus = PulseBus(data_root=td, pulse_interval=0.05, default_buffer_size=8)
        bus.start()
        try:
            # publish to two categories
            for i in range(10):
                bus.publish({'id': i, 'v': 'a'}, 'activity', 'act_test')
            for i in range(6):
                bus.publish({'id': i, 'v': 'b'}, 'input', 'inp_test')
            # force overflow on a small buffer by quickly publishing many events
            # using small maxlen for a new key
            with bus._global_lock:
                bus._buffers[('activity', 'act_test')]['maxlen'] = 5
            for i in range(50):
                bus.publish({'id': i, 'v': 'ovf'}, 'activity', 'act_test')
            # let pulses run
            time.sleep(0.5)
        finally:
            bus.stop()

        # stats
        stats = bus.stats()
        assert stats['published'] > 0
        assert stats['forwarded'] > 0
        assert stats['dropped'] > 0

        # check files written
        act_dir = os.path.join(td, 'activity')
        inp_dir = os.path.join(td, 'input')
        act_files = [f for f in os.listdir(act_dir) if f.startswith('act_test_')]
        inp_files = [f for f in os.listdir(inp_dir) if f.startswith('inp_test_')]
        assert len(act_files) == 1
        assert len(inp_files) == 1

        # ensure lines exist
        act_path = os.path.join(act_dir, act_files[0])
        inp_path = os.path.join(inp_dir, inp_files[0])
        with open(act_path, 'r', encoding='utf-8') as f:
            act_lines = f.readlines()
        with open(inp_path, 'r', encoding='utf-8') as f:
            inp_lines = f.readlines()

        assert len(act_lines) > 0
        assert len(inp_lines) == 6
