"""ForgeRecord: immutable, canonical JSON payload container.

Contract (explicit):
- Fields:
  - offset: int | None (None until appended; offset is byte index where record magic starts)
  - timestamp: int (epoch milliseconds)
  - payload_bytes: bytes (canonical JSON bytes: UTF-8, no whitespace, sorted keys, no NaN/Inf)
  - crc32: int (unsigned 32-bit CRC32 over payload_bytes)

Construction modes supported (to satisfy existing call sites):
1) Explicit: ForgeRecord(offset, timestamp_ms, payload_bytes, crc32)
2) Dict-style convenience: ForgeRecord(**kwargs) — packs kwargs into canonical JSON payload and
   extracts timestamp (kwargs['timestamp'] in seconds or ms acceptable) into integer ms.

This class enforces canonical JSON payload generation and validates numeric finiteness.
"""

from typing import Optional, Any, Mapping
import json
import time
import zlib
import math


def _ensure_finite_numbers(obj: Any):
    """Recursively ensure there are no NaN or Infinity float values."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError('Non-finite float not allowed in payload')
    elif isinstance(obj, Mapping):
        for v in obj.values():
            _ensure_finite_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _ensure_finite_numbers(v)


def _canonical_json_bytes(obj: Any) -> bytes:
    # Validate numbers
    _ensure_finite_numbers(obj)
    # Deterministic JSON: sorted keys, no whitespace, stable floats via default encoder
    s = json.dumps(obj, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
    # json.dumps may still allow NaN/Inf if present; validation above prevents that
    return s.encode('utf-8')


class ForgeRecord:
    __slots__ = ('offset', 'timestamp', 'payload_bytes', 'crc32')

    def __init__(self, *args: Any, **kwargs: Any):
        # explicit construction: (offset, timestamp_ms, payload_bytes, crc32)
        if len(args) == 4 and not kwargs:
            offset, timestamp_ms, payload_bytes, crc32 = args
            if payload_bytes is None or not isinstance(payload_bytes, (bytes, bytearray)):
                raise TypeError('payload_bytes must be bytes')
            if not isinstance(timestamp_ms, int):
                raise TypeError('timestamp must be int (epoch ms)')
            if crc32 is None:
                crc32 = zlib.crc32(payload_bytes) & 0xffffffff
            object.__setattr__(self, 'offset', offset)
            object.__setattr__(self, 'timestamp', timestamp_ms)
            object.__setattr__(self, 'payload_bytes', bytes(payload_bytes))
            object.__setattr__(self, 'crc32', int(crc32))
            return

        # dict-style: pack kwargs into canonical JSON payload
        payload_dict = dict(kwargs)
        # normalize timestamp: accept seconds (float) or ms (int)
        ts = payload_dict.get('timestamp', None)
        if ts is None:
            timestamp_ms = int(time.time() * 1000)
        else:
            if isinstance(ts, float):
                timestamp_ms = int(ts * 1000)
            else:
                timestamp_ms = int(ts)
            # store normalized timestamp back into payload for determinism
            payload_dict['timestamp'] = timestamp_ms
        payload = _canonical_json_bytes(payload_dict)
        crc = zlib.crc32(payload) & 0xffffffff
        object.__setattr__(self, 'offset', None)
        object.__setattr__(self, 'timestamp', int(timestamp_ms))
        object.__setattr__(self, 'payload_bytes', payload)
        object.__setattr__(self, 'crc32', int(crc))

    def __repr__(self) -> str:
        return f"ForgeRecord(offset={self.offset}, timestamp={self.timestamp}, payload_len={len(self.payload_bytes)}, crc32={self.crc32})"
