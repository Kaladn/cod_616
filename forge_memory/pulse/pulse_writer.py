import threading
import time
from typing import List

from forge_memory.core.binary_log import BinaryLog
from forge_memory.indexes.hash_index import HashIndex
from forge_memory.indexes.bitmap_index import BitmapIndex

class PulseWriter:
    """Drains worker queues and writes batches to BinaryLog and updates indexes."""

    def __init__(self, queues: List, binary_log: BinaryLog, indexes: dict, pulse_interval_ms: int = 10):
        self.queues = queues
        self.binary_log = binary_log
        self.index_engine_shapes: HashIndex = indexes['engine_shapes']
        self.index_task: HashIndex = indexes['task']
        self.bitmap: BitmapIndex = indexes['success']
        self.running = False
        self.thread = None
        self.pulse_interval_ms = pulse_interval_ms
        self._next_record_id = 0
        self._offset_to_id = {}

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while self.running:
            batch = []
            for q in self.queues:
                try:
                    rec = q.get(block=False)
                    batch.append(rec)
                except Exception:
                    continue
            if batch:
                offsets = []
                for rec in batch:
                    # rec is expected to be dict compatible with ForgeRecord
                    fr = self._dict_to_record(rec)
                    offset = self.binary_log.append_record(fr)
                    offsets.append(offset)
                    rec_id = self._next_record_id
                    self._offset_to_id[offset] = rec_id
                    self._next_record_id += 1
                    # update task index
                    self.index_task.insert(rec['task_id'], offset)
                    # update engine_shapes index
                    key = (rec['engine_id'], rec['transform_id'], tuple(rec['grid_shape_in']), tuple(rec['grid_shape_out']))
                    self.index_engine_shapes.insert(key, offset)
                    # update bitmap
                    self.bitmap.set_bit(rec_id, rec['success'])
            time.sleep(self.pulse_interval_ms / 1000.0)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _dict_to_record(self, rec: dict):
        from forge_memory.core.record import ForgeRecord
        return ForgeRecord(
            pulse_id=rec.get('pulse_id', 0),
            worker_id=rec.get('worker_id', 0),
            seq=rec.get('seq', 0),
            timestamp=rec.get('timestamp', time.time()),
            success=rec.get('success', False),
            task_id=rec['task_id'],
            engine_id=rec['engine_id'],
            transform_id=rec['transform_id'],
            failure_reason=rec.get('failure_reason', None),
            grid_shape_in=tuple(rec['grid_shape_in']),
            grid_shape_out=tuple(rec['grid_shape_out']),
            color_count=rec.get('color_count', 0),
            train_pair_indices=rec.get('train_pair_indices', []),
            error_metrics=rec.get('error_metrics', {}),
            params=rec.get('params', {}),
            context=rec.get('context', {}),
        )

    def get_offset_to_id_map(self):
        return dict(self._offset_to_id)
