import threading
from typing import Any, List, Dict, Tuple

class HashIndex:
    """Simple in-memory hash index key -> list[offsets]."""

    def __init__(self):
        self.lock = threading.Lock()
        self.table: Dict[bytes, List[int]] = {}

    def _serialize_key(self, key: Any) -> bytes:
        # Keep keys small and deterministic.
        if isinstance(key, tuple):
            return b'|'.join([str(x).encode('utf-8') for x in key])
        return str(key).encode('utf-8')

    def insert(self, key: Any, offset: int):
        k = self._serialize_key(key)
        with self.lock:
            self.table.setdefault(k, []).append(offset)

    def lookup(self, key: Any) -> List[int]:
        k = self._serialize_key(key)
        with self.lock:
            return list(self.table.get(k, []))

    def batch_insert(self, pairs: List[Tuple[Any, int]]):
        with self.lock:
            for key, offset in pairs:
                k = self._serialize_key(key)
                self.table.setdefault(k, []).append(offset)

    def clear(self):
        with self.lock:
            self.table.clear()
