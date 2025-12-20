from __future__ import annotations

import os
import struct
import threading
from typing import Iterable, List, Tuple

from forge_memory.core.mmap_file import MMapFile
from forge_memory.core.record import ForgeRecord
from forge_memory.core.string_dict import StringDictionary
from forge_memory.utils.checksum import Checksum
from forge_memory.utils.constants import MAGIC_WAL, WAL_VERSION


# WAL header layout (20 bytes)
# 0   4  MAGIC                    (uint32)  "WAL\0"  -> MAGIC_WAL
# 4   2  VERSION                  (uint16)  -> WAL_VERSION
# 6   2  RESERVED                 (uint16)  -> 0 for now
# 8   8  LAST_CHECKPOINT_PULSE_ID (uint64)
# 16  4  ENTRY_COUNT              (uint32)
_WAL_HEADER_FMT = "<I H H Q I"
_WAL_HEADER_SIZE = struct.calcsize(_WAL_HEADER_FMT)

# Entry layout:
# offset 0   4  ENTRY_LENGTH (uint32)  - total bytes including this field
# offset 4   8  PULSE_ID     (uint64)
# offset 12  4  RECORD_COUNT (uint32)
# offset 16  N  RECORD_DATA  (bytes, concatenated ForgeRecord binaries)
# offset 16+N 4 CHECKSUM     (uint32, CRC32 over entry_body)
#
# entry_body = PULSE_ID (8) + RECORD_COUNT (4) + RECORD_DATA (N)
# ENTRY_LENGTH = 4 + len(entry_body) + 4
#
# NOTE: We do *not* parse individual ForgeRecords here; that's up to recovery.
_ENTRY_HEADER_FMT = "<I Q I"  # length, pulse_id, record_count
_ENTRY_HEADER_SIZE = struct.calcsize(_ENTRY_HEADER_FMT)  # 4+8+4 = 16


class WALWriter:
    """
    WALWriter handles append-only, crash-resistant journaling of pulses.

    It owns wal.bin, maintains the WAL header, and appends entries of the form:
        [ENTRY_LENGTH][PULSE_ID][RECORD_COUNT][RECORD_DATA...][CHECKSUM]

    Responsibilities:
      - Initialize or validate WAL header.
      - Scan existing entries on startup to find current_offset and last_pulse_id.
      - Append new entries atomically (from the WAL point of view).
      - Update LAST_CHECKPOINT_PULSE_ID and ENTRY_COUNT in header.
      - Truncate the WAL file after checkpointing.
    """

    DEFAULT_INITIAL_SIZE = 64 * 1024 * 1024  # 64 MiB to start; will auto-grow

    def __init__(
        self,
        data_dir: str,
        string_dict: StringDictionary,
        filename: str = "wal.bin",
        initial_size: int | None = None,
    ) -> None:
        if initial_size is None:
            initial_size = self.DEFAULT_INITIAL_SIZE

        self.data_dir = data_dir
        self.wal_path = os.path.join(data_dir, filename)
        self.string_dict = string_dict

        os.makedirs(self.data_dir, exist_ok=True)

        # MMapFile handles creation/resizing; we open read/write.
        self.mmap_file = MMapFile(self.wal_path, initial_size, mode="r+")
        self._lock = threading.Lock()

        # Header state (in-memory mirror)
        self.last_checkpoint_pulse_id: int = 0
        self.entry_count: int = 0

        # File position for next entry
        self.current_offset: int = 0

        # Highest pulse_id we've written in this WAL
        self.last_pulse_id: int = 0

        # Initialize or validate header, then scan entries
        # Check if file is brand new by reading first 4 bytes (MAGIC)
        if self.mmap_file.size < _WAL_HEADER_SIZE:
            is_new = True
        else:
            magic_bytes = self.mmap_file.read_bytes(0, 4)
            magic_value = struct.unpack("<I", magic_bytes)[0]
            is_new = (magic_value == 0)
        
        if is_new:
            # Brand new file
            self._write_header(
                last_checkpoint_pulse_id=0,
                entry_count=0,
            )
            self.current_offset = _WAL_HEADER_SIZE
            self.last_checkpoint_pulse_id = 0
            self.entry_count = 0
            self.last_pulse_id = 0
        else:
            self._load_header()
            self._scan_existing_entries()

    # ------------------------------------------------------------------ #
    # Header helpers
    # ------------------------------------------------------------------ #

    def _write_header(self, last_checkpoint_pulse_id: int, entry_count: int) -> None:
        header_bytes = struct.pack(
            _WAL_HEADER_FMT,
            MAGIC_WAL,
            WAL_VERSION,
            0,  # reserved
            last_checkpoint_pulse_id,
            entry_count,
        )
        # Always write header at offset 0
        self.mmap_file.write_bytes(0, header_bytes)
        self.mmap_file.flush()

    def _load_header(self) -> None:
        if self.mmap_file.size < _WAL_HEADER_SIZE:
            raise ValueError("WAL file too small to contain a header")

        header_bytes = self.mmap_file.read_bytes(0, _WAL_HEADER_SIZE)
        magic, version, reserved, last_cp, entry_count = struct.unpack(
            _WAL_HEADER_FMT, header_bytes
        )

        if magic != MAGIC_WAL:
            raise ValueError(f"Invalid WAL MAGIC: {hex(magic)}")
        if version != WAL_VERSION:
            raise ValueError(f"Unsupported WAL version: {version}")
        if reserved != 0:
            # We don't require reserved==0 for forward compatibility, but it's nice to warn.
            # For now treat any non-zero as an error to keep behavior strict.
            raise ValueError(f"Unexpected WAL reserved field: {reserved}")

        self.last_checkpoint_pulse_id = last_cp
        self.entry_count = entry_count
        # current_offset/last_pulse_id will be set by _scan_existing_entries()

    # ------------------------------------------------------------------ #
    # Entry scan (startup / truncate)
    # ------------------------------------------------------------------ #

    def _scan_existing_entries(self) -> None:
        """
        Scan WAL to find all valid entries, set current_offset and last_pulse_id.

        We do not trust ENTRY_COUNT alone; we scan by ENTRY_LENGTH and checksum.

        Algorithm:
          offset = _WAL_HEADER_SIZE
          while offset + 4 <= file_size:
            read ENTRY_LENGTH
            validate boundaries and checksums
            if invalid/truncated -> break
            parse PULSE_ID
            update last_pulse_id
            offset += ENTRY_LENGTH

        All successfully parsed entries are considered valid; trailing partial
        entries are ignored.
        """
        file_size = self.mmap_file.size
        offset = _WAL_HEADER_SIZE
        valid_entries = 0
        last_pulse_id = 0

        while offset + 4 <= file_size:
            # Read ENTRY_LENGTH
            length_bytes = self.mmap_file.read_bytes(offset, 4)
            (entry_length,) = struct.unpack("<I", length_bytes)

            if entry_length == 0:
                break

            end = offset + entry_length
            if end > file_size:
                # Truncated entry at end of file
                break

            entry_bytes = self.mmap_file.read_bytes(offset, entry_length)

            # Validate checksum
            if len(entry_bytes) < _ENTRY_HEADER_SIZE + 4:
                # Not enough for header + checksum
                break

            entry_body = entry_bytes[4:-4]  # skip length, omit checksum
            (checksum_stored,) = struct.unpack("<I", entry_bytes[-4:])
            if not Checksum.verify(entry_body, checksum_stored):
                # Corrupt entry; stop at this boundary
                break

            # Parse PULSE_ID only (we don't need RECORD_COUNT here)
            _, pulse_id, _ = struct.unpack(
                _ENTRY_HEADER_FMT, entry_bytes[:_ENTRY_HEADER_SIZE]
            )
            last_pulse_id = max(last_pulse_id, pulse_id)
            valid_entries += 1

            offset = end

        self.current_offset = offset
        self.last_pulse_id = last_pulse_id
        # entry_count from header might be stale; update to real count
        self.entry_count = valid_entries
        self._write_header(self.last_checkpoint_pulse_id, self.entry_count)

    # ------------------------------------------------------------------ #
    # Public API — append / checkpoint / truncate
    # ------------------------------------------------------------------ #

    def write_entry(
        self, pulse_id: int, records: Iterable[ForgeRecord]
    ) -> int:
        """
        Append a WAL entry for a single pulse.

        Steps:
          1. Serialize all ForgeRecords.
          2. Build entry_body = [PULSE_ID][RECORD_COUNT][RECORD_DATA...]
          3. Compute checksum over entry_body.
          4. Write [ENTRY_LENGTH][entry_body][CHECKSUM] at current_offset.
          5. Update header (ENTRY_COUNT).
          6. Flush to disk.

        Returns:
            offset (int): the byte offset where this entry was written.
        """
        rec_list = list(records)
        if not rec_list:
            # Nothing to write
            return self.current_offset

        # Serialize records using shared StringDictionary
        record_binaries: List[bytes] = [
            r.serialize(self.string_dict) for r in rec_list
        ]
        record_data = b"".join(record_binaries)
        record_count = len(record_binaries)

        # Build entry body and checksum
        entry_body = struct.pack("<Q I", pulse_id, record_count) + record_data
        checksum_value = Checksum.calculate(entry_body)
        checksum_bytes = checksum_value.to_bytes(4, "little", signed=False)

        entry_without_len = entry_body + checksum_bytes
        entry_length = len(entry_without_len) + 4  # +4 for ENTRY_LENGTH field itself

        entry_bytes = struct.pack("<I", entry_length) + entry_body + checksum_bytes

        with self._lock:
            offset = self.current_offset
            self.mmap_file.write_bytes(offset, entry_bytes)
            self.mmap_file.flush()

            # Update header state
            self.entry_count += 1
            if pulse_id > self.last_pulse_id:
                self.last_pulse_id = pulse_id

            self.current_offset += entry_length

            self._write_header(
                last_checkpoint_pulse_id=self.last_checkpoint_pulse_id,
                entry_count=self.entry_count,
            )

        return offset

    def write_checkpoint(self, pulse_id: int) -> None:
        """
        Update LAST_CHECKPOINT_PULSE_ID in WAL header (no truncation).

        This should only be called after all pulses <= pulse_id are durably
        reflected in BinaryLog and any indexes have been updated.
        """
        with self._lock:
            if pulse_id < self.last_checkpoint_pulse_id:
                # Never move checkpoint backwards
                return
            self.last_checkpoint_pulse_id = pulse_id
            self._write_header(
                last_checkpoint_pulse_id=self.last_checkpoint_pulse_id,
                entry_count=self.entry_count,
            )

    def truncate(self, keep_after_pulse_id: int) -> None:
        """
        Truncate WAL to remove entries with pulse_id < keep_after_pulse_id.

        Implementation:
          1. Scan existing entries and gather those with pulse_id >= keep_after_pulse_id.
          2. Create a new WAL file (wal_new.bin).
          3. Write header with LAST_CHECKPOINT_PULSE_ID = keep_after_pulse_id,
             ENTRY_COUNT = number of kept entries.
          4. Append kept entries sequentially (back-to-back) after header.
          5. Flush, close old mmap, atomically replace wal.bin with wal_new.bin.
          6. Re-open mmap on the new file and reset state.

        If no entries are >= keep_after_pulse_id, we keep an empty WAL with
        LAST_CHECKPOINT_PULSE_ID = keep_after_pulse_id and ENTRY_COUNT = 0.
        """
        with self._lock:
            # First, gather entries from the current WAL
            entries: List[Tuple[int, bytes]] = []  # (pulse_id, entry_bytes)
            file_size = self.mmap_file.size
            offset = _WAL_HEADER_SIZE

            while offset + 4 <= file_size:
                length_bytes = self.mmap_file.read_bytes(offset, 4)
                (entry_length,) = struct.unpack("<I", length_bytes)
                if entry_length == 0:
                    break
                end = offset + entry_length
                if end > file_size:
                    break

                entry_bytes = self.mmap_file.read_bytes(offset, entry_length)
                if len(entry_bytes) < _ENTRY_HEADER_SIZE + 4:
                    break

                entry_body = entry_bytes[4:-4]
                (checksum_stored,) = struct.unpack("<I", entry_bytes[-4:])
                if not Checksum.verify(entry_body, checksum_stored):
                    break

                # Parse PULSE_ID
                _, pulse_id, _ = struct.unpack(
                    _ENTRY_HEADER_FMT, entry_bytes[:_ENTRY_HEADER_SIZE]
                )

                if pulse_id >= keep_after_pulse_id:
                    entries.append((pulse_id, entry_bytes))

                offset = end

            # Close existing mmap before replacing file
            self.mmap_file.close()

            # Build new WAL file path
            new_path = self.wal_path + ".new"

            # Create & initialize new WAL file
            # Pre-size to at least old size or minimum default
            initial_size = max(self.DEFAULT_INITIAL_SIZE, file_size)
            new_mmap = MMapFile(new_path, initial_size, mode="r+")

            # Header with updated checkpoint and entry_count
            new_last_cp = keep_after_pulse_id
            new_entry_count = len(entries)
            header_bytes = struct.pack(
                _WAL_HEADER_FMT,
                MAGIC_WAL,
                WAL_VERSION,
                0,
                new_last_cp,
                new_entry_count,
            )
            new_mmap.write_bytes(0, header_bytes)

            # Write kept entries back-to-back after header
            new_offset = _WAL_HEADER_SIZE
            max_pulse_id = 0
            for pulse_id, entry_bytes in entries:
                new_mmap.write_bytes(new_offset, entry_bytes)
                new_offset += len(entry_bytes)
                if pulse_id > max_pulse_id:
                    max_pulse_id = pulse_id

            new_mmap.flush()
            new_mmap.close()

            # Atomically replace old wal.bin with wal.bin.new
            os.replace(new_path, self.wal_path)

            # Reopen mmap on the new file
            self.mmap_file = MMapFile(self.wal_path, initial_size, mode="r+")
            self.last_checkpoint_pulse_id = new_last_cp
            self.entry_count = new_entry_count
            self.current_offset = _WAL_HEADER_SIZE + sum(
                len(e[1]) for e in entries
            )
            self.last_pulse_id = max_pulse_id

            # Ensure header is correct on the new file
            self._write_header(self.last_checkpoint_pulse_id, self.entry_count)

    # ------------------------------------------------------------------ #
    # Introspection / teardown
    # ------------------------------------------------------------------ #

    def get_header_state(self) -> tuple[int, int]:
        """
        Returns (last_checkpoint_pulse_id, entry_count).
        """
        return self.last_checkpoint_pulse_id, self.entry_count

    def close(self) -> None:
        self.mmap_file.close()
