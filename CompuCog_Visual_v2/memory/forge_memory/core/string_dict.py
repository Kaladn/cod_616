"""
FORGE MEMORY SYSTEM - STRING DICTIONARY
String deduplication per FORGE_PSEUDOCODE_PHASE1.txt
"""

import struct
from typing import Dict, Optional
from pathlib import Path

from forge_memory.utils.constants import MAGIC_STRING_DICT, STRING_DICT_VERSION


class StringDictionary:
    """
    String deduplication via dictionary compression.
    
    Binary format:
      HEADER:
        0  4  MAGIC (uint32)
        4  4  ENTRY_COUNT (uint32)
        8  4  TOTAL_BYTES (uint32)
      
      ENTRIES (repeating):
        0  2  STRING_LENGTH (uint16)
        2  N  STRING_DATA (UTF-8)
        2+N 4  HASH (uint32, CRC32)
    """
    
    def __init__(self, filepath: Optional[Path] = None):
        """
        Initialize string dictionary.
        
        Args:
            filepath: Optional path to load/save dictionary
        """
        self.filepath = filepath
        self.strings: Dict[int, str] = {}  # ref_id -> string
        self.hashes: Dict[int, int] = {}  # hash -> ref_id
        self.next_ref_id = 1  # 0 is reserved for None/null
        
        if filepath and filepath.exists():
            self.load(filepath)
    
    def add_string(self, s: str) -> int:
        """
        Add string to dictionary, return reference ID.
        Deduplicates automatically.
        
        Args:
            s: String to add
            
        Returns:
            uint16 reference ID
        """
        if not s:
            return 0
        
        # Check if already exists
        import zlib
        hash_val = zlib.crc32(s.encode('utf-8')) & 0xFFFFFFFF
        
        if hash_val in self.hashes:
            return self.hashes[hash_val]
        
        # Add new string
        ref_id = self.next_ref_id
        if ref_id > 0xFFFF:
            raise OverflowError("String dictionary full (max 65535 strings)")
        
        self.strings[ref_id] = s
        self.hashes[hash_val] = ref_id
        self.next_ref_id += 1
        
        return ref_id
    
    def get_string(self, ref_id: int) -> str:
        """
        Get string by reference ID.
        
        Args:
            ref_id: Reference ID from add_string
            
        Returns:
            Original string
            
        Raises:
            KeyError: If ref_id not found
        """
        if ref_id == 0:
            return ""
        
        if ref_id not in self.strings:
            raise KeyError(f"String ref_id {ref_id} not found")
        
        return self.strings[ref_id]
    
    def save(self, filepath: Optional[Path] = None) -> None:
        """
        Save dictionary to binary file.
        
        Args:
            filepath: Path to save to (uses self.filepath if None)
        """
        path = filepath or self.filepath
        if not path:
            raise ValueError("No filepath specified")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build entries
        entries = []
        total_bytes = 12  # header size
        
        for ref_id in sorted(self.strings.keys()):
            s = self.strings[ref_id]
            s_bytes = s.encode('utf-8')
            
            if len(s_bytes) > 0xFFFF:
                raise ValueError(f"String too long: {len(s_bytes)} bytes")
            
            # Find hash for this string
            import zlib
            hash_val = zlib.crc32(s_bytes) & 0xFFFFFFFF
            
            entry = struct.pack('<H', len(s_bytes)) + s_bytes + struct.pack('<I', hash_val)
            entries.append(entry)
            total_bytes += len(entry)
        
        # Write file
        with open(path, 'wb') as f:
            # Header
            header = struct.pack('<III', 
                                MAGIC_STRING_DICT,
                                len(self.strings),
                                total_bytes)
            f.write(header)
            
            # Entries
            for entry in entries:
                f.write(entry)
    
    def load(self, filepath: Path) -> None:
        """
        Load dictionary from binary file.
        
        Args:
            filepath: Path to load from
        """
        with open(filepath, 'rb') as f:
            # Read header
            header = f.read(12)
            if len(header) < 12:
                raise ValueError("File too short for string dict header")
            
            magic, entry_count, total_bytes = struct.unpack('<III', header)
            
            if magic != MAGIC_STRING_DICT:
                raise ValueError(f"Invalid magic: {hex(magic)}")
            
            # Read entries
            self.strings.clear()
            self.hashes.clear()
            
            ref_id = 1
            for _ in range(entry_count):
                # String length
                len_bytes = f.read(2)
                if len(len_bytes) < 2:
                    raise ValueError("Truncated string length")
                string_len = struct.unpack('<H', len_bytes)[0]
                
                # String data
                string_bytes = f.read(string_len)
                if len(string_bytes) < string_len:
                    raise ValueError("Truncated string data")
                s = string_bytes.decode('utf-8')
                
                # Hash
                hash_bytes = f.read(4)
                if len(hash_bytes) < 4:
                    raise ValueError("Truncated hash")
                hash_val = struct.unpack('<I', hash_bytes)[0]
                
                self.strings[ref_id] = s
                self.hashes[hash_val] = ref_id
                ref_id += 1
            
            self.next_ref_id = ref_id
