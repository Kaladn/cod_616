"""
FORGE MEMORY SYSTEM - CHECKSUM
CRC32 checksum utilities per FORGE_PSEUDOCODE_PHASE1.txt
"""

import zlib


class Checksum:
    """CRC32 checksum utilities for data integrity."""
    
    @staticmethod
    def calculate(data: bytes) -> int:
        """
        Calculate CRC32 checksum for data.
        
        Args:
            data: Bytes to checksum
            
        Returns:
            uint32 checksum value
        """
        return zlib.crc32(data) & 0xFFFFFFFF
    
    @staticmethod
    def verify(data: bytes, expected_checksum: int) -> bool:
        """
        Verify data against expected checksum.
        
        Args:
            data: Data to verify
            expected_checksum: Expected CRC32 value
            
        Returns:
            True if checksum matches, False otherwise
        """
        actual = Checksum.calculate(data)
        return actual == expected_checksum
    
    @staticmethod
    def verify_and_strip(data_with_checksum: bytes) -> tuple[bool, bytes]:
        """
        Verify checksum and return data without checksum.
        
        Args:
            data_with_checksum: Data with 4-byte checksum at end
            
        Returns:
            Tuple of (is_valid, data_without_checksum)
        """
        if len(data_with_checksum) < 4:
            return False, b''
        
        data = data_with_checksum[:-4]
        checksum_bytes = data_with_checksum[-4:]
        expected = int.from_bytes(checksum_bytes, 'little', signed=False)
        
        is_valid = Checksum.verify(data, expected)
        return is_valid, data
