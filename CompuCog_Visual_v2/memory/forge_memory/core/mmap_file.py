"""
FORGE MEMORY SYSTEM - MEMORY-MAPPED FILE
Cross-platform mmap abstraction per FORGE_PSEUDOCODE_PHASE1.txt
"""

import mmap
import os
from pathlib import Path


class MMapFile:
    """
    Cross-platform memory-mapped file wrapper.
    
    Handles:
    - File creation with pre-allocation
    - Dynamic resizing
    - Memory mapping (OS-managed caching)
    - Read/write operations
    - Flush to disk
    """
    
    def __init__(self, path: str, initial_size: int, mode: str = 'r+'):
        """
        Initialize memory-mapped file.
        
        Args:
            path: File path
            initial_size: Initial file size in bytes
            mode: 'r' (read-only) or 'r+' (read-write)
        """
        self.path = str(Path(path))
        self.mode = mode
        self.initial_size = initial_size
        
        # Create file if doesn't exist
        if not os.path.exists(self.path):
            self._create_file_with_size(self.path, initial_size)
        
        # Open file
        if mode == 'r':
            self.file_handle = open(self.path, 'rb')
        else:  # 'r+'
            self.file_handle = open(self.path, 'r+b')
        
        # Get current file size
        self.size = os.path.getsize(self.path)
        
        # Create memory map
        if self.size == 0:
            # Special case: empty file, can't mmap yet
            # Resize to initial_size first
            self._resize_file(initial_size)
            self.size = initial_size
        
        if mode == 'r':
            self.mmap_handle = mmap.mmap(
                self.file_handle.fileno(), 
                0,  # map entire file
                access=mmap.ACCESS_READ
            )
        else:  # 'r+'
            self.mmap_handle = mmap.mmap(
                self.file_handle.fileno(),
                0,  # map entire file
                access=mmap.ACCESS_WRITE
            )
    
    @staticmethod
    def _create_file_with_size(path: str, size: int) -> None:
        """
        Create file and pre-allocate space.
        
        Algorithm:
        1. Create parent directory if needed
        2. Open file in write mode
        3. Seek to (size - 1)
        4. Write single byte (forces allocation)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            if size > 0:
                f.seek(size - 1)
                f.write(b'\x00')
    
    def _resize_file(self, new_size: int) -> None:
        """
        Resize the underlying file.
        Called internally before remapping.
        """
        self.file_handle.truncate(new_size)
        self.file_handle.flush()
        os.fsync(self.file_handle.fileno())
    
    def read_bytes(self, offset: int, length: int) -> bytes:
        """
        Read bytes from memory-mapped file.
        
        Args:
            offset: Byte offset in file
            length: Number of bytes to read
            
        Returns:
            Bytes read
            
        Raises:
            ValueError: If read would exceed file size
        """
        if offset < 0:
            raise ValueError(f"Negative offset: {offset}")
        if offset + length > self.size:
            raise ValueError(
                f"Read beyond file size: {offset + length} > {self.size}"
            )
        
        self.mmap_handle.seek(offset)
        return self.mmap_handle.read(length)
    
    def write_bytes(self, offset: int, data: bytes) -> None:
        """
        Write bytes to memory-mapped file.
        
        Auto-resizes if write would exceed current file size.
        
        Args:
            offset: Byte offset in file
            data: Bytes to write
            
        Raises:
            ValueError: If mode is read-only
        """
        if self.mode == 'r':
            raise ValueError("Cannot write to read-only mmap")
        
        required_size = offset + len(data)
        if required_size > self.size:
            # Double size until we have enough space
            new_size = self.size
            while new_size < required_size:
                new_size *= 2
            self.resize(new_size)
        
        self.mmap_handle.seek(offset)
        self.mmap_handle.write(data)
    
    def resize(self, new_size: int) -> None:
        """
        Resize memory-mapped file.
        
        Algorithm:
        1. Close current mmap
        2. Resize underlying file
        3. Recreate mmap with new size
        4. Update self.size
        
        Args:
            new_size: New file size in bytes
            
        Notes:
            Expensive operation (remaps memory)
        """
        if self.mode == 'r':
            raise ValueError("Cannot resize read-only mmap")
        
        # Close current mmap
        self.mmap_handle.close()
        
        # Resize file
        self._resize_file(new_size)
        
        # Recreate mmap
        self.mmap_handle = mmap.mmap(
            self.file_handle.fileno(),
            0,
            access=mmap.ACCESS_WRITE
        )
        self.size = new_size
    
    def flush(self) -> None:
        """
        Force write of dirty pages to disk.
        
        Use after critical writes (e.g., WAL checkpoint).
        Blocks until I/O complete.
        """
        self.mmap_handle.flush()
        self.file_handle.flush()
        os.fsync(self.file_handle.fileno())
    
    def close(self) -> None:
        """
        Close memory map and file.
        
        Flushes dirty pages before closing.
        """
        if hasattr(self, 'mmap_handle') and self.mmap_handle:
            self.mmap_handle.close()
        if hasattr(self, 'file_handle') and self.file_handle:
            self.file_handle.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
