import os
import struct
from typing import Dict, List

"""StringDictionary

File format (stable):
- Stored file name: `strings.dict` (under provided data_dir or as given file path)
- On-disk layout: repeated records:
    [4-byte uint32 LE length][utf-8 bytes]
- IDs are sequential integers starting at 1 corresponding to the order of first appearance in the file.

Behavior:
- Deterministic mapping: the first time a string is seen it is appended and assigned the next id (1-based).
- get_id(s: str) -> int: returns existing id or appends and returns new id.
- get_string(i: int) -> str: returns string for id i (1-based).
"""

LENGTH_FMT = '<I'

class StringDictionary:
    def __init__(self, path_or_dir: str):
        if os.path.isdir(path_or_dir) or path_or_dir.endswith(os.path.sep):
            self.path = os.path.join(path_or_dir, 'strings.dict')
        else:
            # accept direct file path
            self.path = path_or_dir
        os.makedirs(os.path.dirname(self.path), exist_ok=True) if os.path.dirname(self.path) else None
        self._f = open(self.path, 'a+b')
        # id -> string (1-based indexing achieved by list with dummy at index 0)
        self._id_to_str: List[str] = [None]  # index 0 unused
        self._str_to_id: Dict[str, int] = {}
        self._load_existing()

    def _load_existing(self):
        self._f.flush()
        self._f.seek(0)
        while True:
            head = self._f.read(4)
            if not head or len(head) < 4:
                break
            length = struct.unpack(LENGTH_FMT, head)[0]
            data = self._f.read(length)
            if len(data) < length:
                break
            s = data.decode('utf-8')
            # assign next id (1-based)
            if s in self._str_to_id:
                continue
            id_ = len(self._id_to_str)
            self._id_to_str.append(s)
            self._str_to_id[s] = id_

    def get_id(self, s: str) -> int:
        if s in self._str_to_id:
            return self._str_to_id[s]
        encoded = s.encode('utf-8')
        length = len(encoded)
        self._f.seek(0, os.SEEK_END)
        self._f.write(struct.pack(LENGTH_FMT, length))
        self._f.write(encoded)
        self._f.flush()
        try:
            os.fsync(self._f.fileno())
        except Exception:
            pass
        id_ = len(self._id_to_str)
        self._id_to_str.append(s)
        self._str_to_id[s] = id_
        return id_

    def get_string(self, id: int) -> str:
        if id <= 0 or id >= len(self._id_to_str):
            raise KeyError('invalid string id')
        return self._id_to_str[id]

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
