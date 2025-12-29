"""BinaryLog: append-only binary record file with explicit frame.

SPEC (must be stable):
- Byte order: LITTLE-ENDIAN for all integer fields
- Record frame (sequence, offsets refer to start of magic):

  [magic: 4 bytes]           b'FREC'
  [version: u16]            (e.g., 1)
  [header_len: u16]         number of bytes in header following these fields
  [payload_len: u32]
  [timestamp_ms: u64]
  [crc32: u32]              CRC32(payload_bytes)
  [payload_bytes: payload_len bytes]

- Offsets returned by `append` point to the start of the 'magic' field.
- header_len in current format is 16 (payload_len(4) + timestamp_ms(8) + crc32(4)).
- CRC32 is computed over payload_bytes only and stored as uint32 LE.
- Files are fsync'd on append (best-effort).
- No WAL, no concurrency, no background threads, no caches.
"""

import os
import struct
import zlib
from typing import Optional, Iterator
from .record import ForgeRecord

MAGIC = b'FREC'
VERSION = 1
# header fields after version/header_len: payload_len(u32), timestamp_ms(u64), crc32(u32)
HEADER_FMT = '<IQI'  # payload_len (I), timestamp_ms (Q), crc32 (I)
HEADER_LEN = struct.calcsize(HEADER_FMT)  # should be 4+8+4 = 16

# fields for version and header_len
VER_HDR_FMT = '<HH'  # version u16, header_len u16
VER_HDR_SIZE = struct.calcsize(VER_HDR_FMT)

# full prefix size = magic(4) + VER_HDR_SIZE

class BinaryLog:
    """Minimal BinaryLog operating on a single `records.bin` file under a data directory.

    Methods implemented: append(record: ForgeRecord) -> int, read(offset: int) -> ForgeRecord,
    iter_records() -> Iterator[ForgeRecord] (useful for tests).
    """

    def __init__(self, data_dir_or_path: str, string_dict: Optional[object] = None):
        # decide on file path; if directory -> data_dir/records.bin
        if os.path.isdir(data_dir_or_path) or data_dir_or_path.endswith(os.path.sep):
            self.path = os.path.join(data_dir_or_path, 'records.bin')
        else:
            if data_dir_or_path.endswith('.bin'):
                self.path = data_dir_or_path
            else:
                self.path = os.path.join(data_dir_or_path, 'records.bin')
        self.string_dict = string_dict
        os.makedirs(os.path.dirname(self.path), exist_ok=True) if os.path.dirname(self.path) else None
        # open for reading and appending
        self._f = open(self.path, 'a+b')

    def append(self, record: ForgeRecord) -> int:
        """Append the given ForgeRecord to the file and return the starting offset (magic start).

        Enforces: monotonic offsets (by appending at file end), fsync on append.
        """
        payload = record.payload_bytes
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError('payload_bytes must be bytes')
        payload_len = len(payload)
        timestamp_ms = int(record.timestamp)
        crc = int(record.crc32) if record.crc32 is not None else (zlib.crc32(payload) & 0xffffffff)
        # compute header bytes
        ver_hdr = struct.pack(VER_HDR_FMT, VERSION, HEADER_LEN)
        header = struct.pack(HEADER_FMT, payload_len, timestamp_ms, crc)
        # seek to end, record offset
        self._f.seek(0, os.SEEK_END)
        offset = self._f.tell()
        # write frame
        self._f.write(MAGIC)
        self._f.write(ver_hdr)
        self._f.write(header)
        self._f.write(payload)
        self._f.flush()
        try:
            os.fsync(self._f.fileno())
        except Exception:
            pass
        return offset

    # Backwards-compatible helper expected by PulseWriter
    def append_record(self, record: ForgeRecord) -> int:
        return self.append(record)

    def read(self, offset: int) -> ForgeRecord:
        """Read the record starting at offset (magic start) and return ForgeRecord; verifies CRC."""
        self._f.flush()
        self._f.seek(offset)
        magic = self._f.read(4)
        if magic != MAGIC:
            raise ValueError('Invalid magic at offset')
        ver_hdr = self._f.read(VER_HDR_SIZE)
        if len(ver_hdr) < VER_HDR_SIZE:
            raise ValueError('Incomplete header')
        version, hdr_len = struct.unpack(VER_HDR_FMT, ver_hdr)
        if hdr_len < HEADER_LEN:
            raise ValueError('Unsupported header_len')
        header = self._f.read(HEADER_LEN)
        if len(header) < HEADER_LEN:
            raise ValueError('Incomplete header fields')
        payload_len, timestamp_ms, crc_stored = struct.unpack(HEADER_FMT, header)
        payload = self._f.read(payload_len)
        if len(payload) < payload_len:
            raise ValueError('Incomplete payload data')
        crc_calc = zlib.crc32(payload) & 0xffffffff
        if crc_calc != crc_stored:
            raise ValueError('CRC mismatch')
        return ForgeRecord(offset, int(timestamp_ms), payload, int(crc_stored))

    def iter_records(self) -> Iterator[ForgeRecord]:
        """Yield all records in file in order (useful for tests)."""
        self._f.flush()
        self._f.seek(0)
        while True:
            start = self._f.tell()
            magic = self._f.read(4)
            if not magic or len(magic) < 4:
                break
            if magic != MAGIC:
                raise ValueError('Unexpected magic while iterating')
            ver_hdr = self._f.read(VER_HDR_SIZE)
            if len(ver_hdr) < VER_HDR_SIZE:
                break
            _, hdr_len = struct.unpack(VER_HDR_FMT, ver_hdr)
            if hdr_len < HEADER_LEN:
                raise ValueError('Unsupported header_len in iteration')
            header = self._f.read(HEADER_LEN)
            if len(header) < HEADER_LEN:
                break
            payload_len, timestamp_ms, crc_stored = struct.unpack(HEADER_FMT, header)
            payload = self._f.read(payload_len)
            if len(payload) < payload_len:
                break
            crc_calc = zlib.crc32(payload) & 0xffffffff
            if crc_calc != crc_stored:
                raise ValueError('CRC mismatch during iteration')
            yield ForgeRecord(start, int(timestamp_ms), payload, int(crc_stored))

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
