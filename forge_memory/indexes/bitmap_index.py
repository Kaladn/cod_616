import threading
from typing import Set, List

class BitmapIndex:
    """Simple in-memory success bitmap implementation using Python set."""

    def __init__(self):
        self.lock = threading.Lock()
        self.success_set: Set[int] = set()

    def set_bit(self, record_id: int, success: bool):
        with self.lock:
            if success:
                self.success_set.add(record_id)
            else:
                self.success_set.discard(record_id)

    def batch_set(self, items: List[tuple[int, bool]]):
        with self.lock:
            for record_id, success in items:
                if success:
                    self.success_set.add(record_id)
                else:
                    self.success_set.discard(record_id)

    def filter_offsets(self, offsets: List[int], offset_to_id: dict) -> List[int]:
        # Given offsets and mapping offset->record_id, return offsets with success True
        with self.lock:
            return [o for o in offsets if offset_to_id.get(o) in self.success_set]

    def clear(self):
        with self.lock:
            self.success_set.clear()
