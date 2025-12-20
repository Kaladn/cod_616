"""
FORGE MEMORY SYSTEM - SERIALIZATION
MessagePack utilities with versioning per FORGE_PSEUDOCODE_PHASE1.txt
"""

import msgpack
from typing import Any, Dict, Tuple


class Serialization:
    """MessagePack serialization with schema versioning."""
    
    @staticmethod
    def encode_with_version(obj: Dict[str, Any], schema_version: int = 1) -> bytes:
        """
        Encode dict with MessagePack, prepending schema version.
        
        Format:
          [version: uint8][msgpack_data]
        
        Args:
            obj: Dictionary to encode
            schema_version: Schema version byte (default 1)
            
        Returns:
            Encoded bytes with version prefix
        """
        if schema_version < 0 or schema_version > 255:
            raise ValueError(f"Schema version must fit in uint8: {schema_version}")
        
        version_byte = schema_version.to_bytes(1, 'little')
        msgpack_data = msgpack.packb(obj, use_bin_type=True)
        
        return version_byte + msgpack_data
    
    @staticmethod
    def decode_with_version(data: bytes) -> Tuple[int, Dict[str, Any]]:
        """
        Decode versioned MessagePack data.
        
        Args:
            data: Encoded bytes with version prefix
            
        Returns:
            Tuple of (schema_version, decoded_dict)
            
        Raises:
            ValueError: If data is too short or decode fails
        """
        if len(data) < 1:
            raise ValueError("Data too short to contain version byte")
        
        schema_version = int(data[0])
        msgpack_data = data[1:]
        
        try:
            obj = msgpack.unpackb(msgpack_data, raw=False)
        except Exception as e:
            raise ValueError(f"MessagePack decode failed: {e}") from e
        
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict, got {type(obj)}")
        
        return schema_version, obj
