import time
from tempfile import TemporaryDirectory
from forge_memory.core.string_dict import StringDictionary
from forge_memory.core.binary_log import BinaryLog
from forge_memory.indexes.hash_index import HashIndex
from forge_memory.indexes.bitmap_index import BitmapIndex
from forge_memory.indexes.vcache import VCache
from forge_memory.pulse.worker_pool import WorkerPool
from forge_memory.pulse.pulse_writer import PulseWriter


def sample_hypothesis(wid: int):
    # Deterministic small generator
    i = int(time.time() * 1000) % 100000
    return {
        'pulse_id': 1,
        'worker_id': wid,
        'seq': i,
        'timestamp': time.time(),
        'success': (i % 2 == 0),
        'task_id': f'task/{wid}',
        'engine_id': 'engine/x',
        'transform_id': 't1',
        'failure_reason': None,
        'grid_shape_in': (8,8),
        'grid_shape_out': (6,6),
        'color_count': 1,
        'train_pair_indices': [0],
        'error_metrics': {},
        'params': {},
        'context': {},
    }


def test_end_to_end_write_and_query():
    with TemporaryDirectory() as td:
        sd = StringDictionary(td + '/strings.bin')
        bl = BinaryLog(td + '/records.bin', sd)
        # indexes
        index_engine_shapes = HashIndex()
        index_task = HashIndex()
        bitmap = BitmapIndex()
        vcache = VCache()

        wp = WorkerPool(worker_count=4, queue_size=100)
        indexes = {'engine_shapes': index_engine_shapes, 'task': index_task, 'success': bitmap}
        pw = PulseWriter(wp.get_queues(), bl, indexes)

        try:
            # push some records directly to queues
            for wid in range(4):
                for j in range(5):
                    rec = sample_hypothesis(wid)
                    rec['seq'] = j
                    wp.put_direct(wid, rec)

            pw.start()
            time.sleep(0.5)  # let pulse writer flush
            pw.stop()

            # validate indexes
            key = ('engine/x', 't1', (8,8), (6,6))
            offsets = index_engine_shapes.lookup(key)
            assert len(offsets) == 20

            # map offsets to record ids and filter successes
            offset_to_id = pw.get_offset_to_id_map()
            filtered = bitmap.filter_offsets(offsets, offset_to_id)
            # some successes expected
            assert len(filtered) > 0
        finally:
            bl.close(); sd.close()
