from collections import OrderedDict
from typing import Any, List

class VCache:
    """Simple LRU cache mapping key -> offsets list."""

    def __init__(self, max_entries=1000):
        self.cache = OrderedDict()
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    def lookup(self, key: Any):
        k = key
        if k in self.cache:
            self.cache.move_to_end(k)
            self.hits += 1
            return list(self.cache[k])
        self.misses += 1
        return None

    def insert(self, key: Any, offsets: List[int]):
        k = key
        self.cache[k] = list(offsets)
        self.cache.move_to_end(k)
        if len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {'hits': self.hits, 'misses': self.misses, 'hit_rate': hit_rate}
