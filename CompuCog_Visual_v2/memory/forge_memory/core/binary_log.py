"""
FORGE MEMORY SYSTEM - BINARY LOG
Append-only record log per FORGE_PSEUDOCODE_PHASE1_CRITICAL.txt
"""

from __future__ import annotations

import os
import struct
import threading
from typing import List, Iterable

from forge_memory.core.mmap_file import MMapFile
from forge_memory.core.record import ForgeRecord
from forge_memory.core.string_dict import StringDictionary
from forge_memory.utils.constants import MAGIC_RECORD


class BinaryLog:
    """
    Append-only binary log for ForgeRecord entries (records.bin).
    
    Responsibilities:
    - Owns records.bin via MMapFile
    - Provides atomic appends (single-process, multi-thread)
    - Supports batch appends
    - Supports random-access reads by byte offset
    - Maintains in-memory record_offsets for fast iteration
    
    Per FORGE_PSEUDOCODE_PHASE1_CRITICAL.txt:
    - Scans existing records at startup (_scan_existing_records)
    - Tracks all record offsets in memory
    - Uses threading.Lock for append atomicity
    - Auto-resizes via MMapFile when needed
    """
    
    DEFAULT_INITIAL_SIZE = 1 * 1024 * 1024 * 1024  # 1 GiB
    
    def __init__(
        self,
        data_dir: str,
        string_dict: StringDictionary,
        initial_size: int = None,
        filename: str = "records.bin",
    ) -> None:
        """
        Initialize BinaryLog.
        
        Algorithm (per spec):
        1. Ensure data_dir exists
        2. Determine log_path = data_dir/records.bin
        3. If file new → create with initial_size, current_offset = 0
        4. If file exists → open and scan existing records to find current_offset
        
        Args:
            data_dir: Directory for records.bin
            string_dict: StringDictionary for string compression
            initial_size: Initial file size (default 1 GiB)
            filename: Log filename (default "records.bin")
        """
        self.data_dir = data_dir
        self.log_path = os.path.join(data_dir, filename)
        self.string_dict = string_dict
        
        if initial_size is None:
            initial_size = self.DEFAULT_INITIAL_SIZE
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        # MMapFile will create/resize as needed per spec
        self.mmap_file = MMapFile(self.log_path, initial_size, mode="r+")
        self.lock = threading.Lock()
        self.record_offsets: List[int] = []
        self.current_offset: int = 0
        
        # If file is effectively empty (no MAGIC_RECORD at 0), treat as new
        if self._is_file_empty():
            self.current_offset = 0
            self.record_offsets = []
        else:
            self._scan_existing_records()
    
    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    
    def _is_file_empty(self) -> bool:
        """
        Check if file is empty (no valid record at offset 0).
        
        Algorithm (per spec):
        1. Read first 4 bytes
        2. If less than 4 bytes or MAGIC != MAGIC_RECORD → treat as empty
        
        Returns:
            True if file is empty/invalid
        """
        if self.mmap_file.size < 4:
            return True
        
        first_bytes = self.mmap_file.read_bytes(0, 4)
        magic = struct.unpack("<I", first_bytes)[0]
        return magic != MAGIC_RECORD
    
    def _scan_existing_records(self) -> None:
        """
        Scan file to discover all record offsets and current_offset.
        
        Algorithm (per FORGE_PSEUDOCODE_PHASE1_CRITICAL.txt):
        1. Start at offset = 0
        2. While offset < file_size:
           a. Read MAGIC (4 bytes)
           b. If MAGIC != MAGIC_RECORD → break (end of valid records)
           c. Read RECORD_LENGTH at offset+6 (uint16)
           d. Append offset to record_offsets
           e. offset += RECORD_LENGTH
        3. Set current_offset = final offset
        
        Notes:
        - O(n) scan, only done once at startup
        - Assumes records are written back-to-back with no gaps
        """
        offset = 0
        file_size = self.mmap_file.size
        self.record_offsets.clear()
        
        while offset + 10 <= file_size:  # enough for MAGIC + VERSION + LENGTH
            # Read MAGIC
            magic_bytes = self.mmap_file.read_bytes(offset, 4)
            magic = struct.unpack("<I", magic_bytes)[0]
            
            if magic != MAGIC_RECORD:
                break  # Reached end of valid records
            
            # Read RECORD_LENGTH (2 bytes at offset+6)
            length_bytes = self.mmap_file.read_bytes(offset + 6, 2)
            record_length = struct.unpack("<H", length_bytes)[0]
            
            # Sanity: avoid infinite loops on corrupt length
            if record_length == 0:
                break
            if offset + record_length > file_size:
                # Partial/corrupted trailing record: stop here
                break
            
            self.record_offsets.append(offset)
            offset += record_length
        
        self.current_offset = offset
    
    # ------------------------------------------------------------------ #
    # Public API — Appends
    # ------------------------------------------------------------------ #
    
    def append(self, record: ForgeRecord) -> int:
        """
        Append a single ForgeRecord to the log.
        
        Algorithm (per spec):
        1. Acquire lock
        2. Serialize record
        3. Check if file needs resizing
        4. Write binary data at current_offset
        5. Update current_offset
        6. Add offset to record_offsets
        7. Release lock
        8. Return offset
        
        Args:
            record: ForgeRecord to append
            
        Returns:
            Byte offset where record was written
        """
        binary_data = record.serialize(self.string_dict)
        length = len(binary_data)
        
        with self.lock:
            offset = self.current_offset
            # MMapFile.write_bytes will resize as needed per spec
            self.mmap_file.write_bytes(offset, binary_data)
            self.record_offsets.append(offset)
            self.current_offset += length
            return offset
    
    def append_batch(self, records: Iterable[ForgeRecord]) -> List[int]:
        """
        Append a batch of ForgeRecords as one atomic operation.
        
        Algorithm (per spec):
        1. Acquire lock
        2. Serialize all records
        3. Calculate total size
        4. Check if resize needed (once)
        5. Write all records sequentially
        6. Update current_offset
        7. Add all offsets to record_offsets
        8. Release lock
        9. Return list of offsets
        
        Optimization:
        - Single lock acquisition
        - Single resize check
        - Sequential writes (better cache locality)
        
        Args:
            records: Iterable of ForgeRecords
            
        Returns:
            List of byte offsets in append order
        """
        # Pre-serialize outside the lock to minimize lock duration
        serialized_records: List[bytes] = [
            r.serialize(self.string_dict) for r in records
        ]
        total_length = sum(len(b) for b in serialized_records)
        
        with self.lock:
            offsets: List[int] = []
            write_offset = self.current_offset
            
            for binary_data in serialized_records:
                offsets.append(write_offset)
                self.record_offsets.append(write_offset)
                
                self.mmap_file.write_bytes(write_offset, binary_data)
                write_offset += len(binary_data)
            
            self.current_offset = write_offset
            return offsets
    
    # ------------------------------------------------------------------ #
    # Public API — Reads
    # ------------------------------------------------------------------ #
    
    def read_at_offset(self, offset: int) -> ForgeRecord:
        """
        Read a record at a specific byte offset.
        
        Algorithm (per spec):
        1. Read RECORD_LENGTH at offset+6 (2 bytes)
        2. Read RECORD_LENGTH bytes starting at offset
        3. Deserialize to ForgeRecord
        
        Args:
            offset: Byte offset in file
            
        Returns:
            Deserialized ForgeRecord
            
        Raises:
            ValueError: If offset is invalid or record is corrupted
        """
        if offset < 0 or offset + 10 > self.mmap_file.size:
            raise ValueError(f"Invalid offset: {offset}")
        
        length_bytes = self.mmap_file.read_bytes(offset + 6, 2)
        record_length = struct.unpack("<H", length_bytes)[0]
        
        if record_length == 0:
            raise ValueError(f"Record at offset {offset} has zero length")
        if offset + record_length > self.mmap_file.size:
            raise ValueError(
                f"Record at offset {offset} extends beyond file size "
                f"({offset + record_length} > {self.mmap_file.size})"
            )
        
        record_bytes = self.mmap_file.read_bytes(offset, record_length)
        return ForgeRecord.deserialize(record_bytes, self.string_dict)
    
    def read_all(self) -> List[ForgeRecord]:
        """
        Read all records in the log in append order.
        
        Returns:
            List of all ForgeRecords
        """
        return [self.read_at_offset(off) for off in self.record_offsets]
    
    def read_multiple(self, offsets: List[int]) -> List[ForgeRecord]:
        """
        Read multiple records (optimized for batch reads).
        
        Algorithm (per spec):
        1. Sort offsets (for sequential reads)
        2. For each offset, read record
        3. Return list of records (in original order)
        
        Optimization:
        - Sort offsets for sequential disk access
        
        Args:
            offsets: List of byte offsets
            
        Returns:
            List of ForgeRecords in original order
        """
        # Create mapping: offset → original_index
        offset_to_index = {offset: i for i, offset in enumerate(offsets)}
        
        # Sort offsets for sequential reads
        sorted_offsets = sorted(offsets)
        
        # Read records
        records_dict = {}
        for offset in sorted_offsets:
            record = self.read_at_offset(offset)
            records_dict[offset] = record
        
        # Return in original order
        records = [records_dict[offset] for offset in offsets]
        return records
    
    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    
    def __len__(self) -> int:
        """Number of records currently tracked."""
        return len(self.record_offsets)
    
    def get_offsets(self) -> List[int]:
        """Return a copy of known record offsets."""
        return list(self.record_offsets)
    
    def get_current_offset(self) -> int:
        """Return current write offset (end of valid data)."""
        return self.current_offset
    
    def get_record_count(self) -> int:
        """Get total number of records (alias for __len__)."""
        return len(self.record_offsets)
    
    def flush(self) -> None:
        """Flush changes to disk."""
        self.mmap_file.flush()
    
    def close(self) -> None:
        """Close binary log."""
        self.mmap_file.close()
