from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import struct

from forge_memory.utils.checksum import Checksum
from forge_memory.utils.serialization import Serialization
from forge_memory.utils.constants import MAGIC_RECORD, RECORD_SCHEMA_VERSION
from forge_memory.core.string_dict import StringDictionary

# Header layout from spec:
# 0   4  MAGIC (uint32)
# 4   2  VERSION (uint16)
# 6   2  RECORD_LENGTH (uint16)
# 8   8  PULSE_ID (uint64)
# 16  1  WORKER_ID (uint8)
# 17  4  SEQ (uint32)
# 21  8  TIMESTAMP (double)
# 29  1  SUCCESS (uint8)
# 30  2  TASK_ID_REF (uint16)
# 32  2  ENGINE_ID_REF (uint16)
# 34  2  TRANSFORM_ID_REF (uint16)
# 36  2  FAILURE_REASON_REF (uint16)
# 38  1  GRID_SHAPE_IN_H (uint8)
# 39  1  GRID_SHAPE_IN_W (uint8)
# 40  1  GRID_SHAPE_OUT_H (uint8)
# 41  1  GRID_SHAPE_OUT_W (uint8)
# 42  1  COLOR_COUNT (uint8)
# 43  1  TRAIN_PAIR_COUNT (uint8)
#
# Total fixed header size: 44 bytes
_HEADER_FMT = "<IHHQBI dB HHHH 6B"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


@dataclass
class ForgeRecord:
    """
    Canonical Forge binary record.

    This is a direct Python implementation of the Phase-1 critical spec:
    - Fixed 44-byte header
    - Variable tail (train_pair_indices + 3 MessagePack blobs)
    - CRC32 checksum (4 bytes, little-endian) at the end
    """

    # Fixed fields
    pulse_id: int
    worker_id: int
    seq: int
    timestamp: float
    success: bool
    task_id: str
    engine_id: str
    transform_id: str
    failure_reason: Optional[str]
    grid_shape_in: Tuple[int, int]
    grid_shape_out: Tuple[int, int]
    color_count: int
    train_pair_indices: List[int]

    # Variable dicts
    error_metrics: Dict[str, Any]
    params: Dict[str, Any]
    context: Dict[str, Any]

    # Internal refs (populated during serialize/deserialize)
    task_id_ref: int = 0
    engine_id_ref: int = 0
    transform_id_ref: int = 0
    failure_reason_ref: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForgeRecord":
        """
        Initialize from a Python dict.

        REQUIRED:
          task_id, engine_id, transform_id, success,
          grid_shape_in, grid_shape_out, train_pair_indices,
          error_metrics, params, context, timestamp

        OPTIONAL:
          failure_reason, pulse_id, worker_id, seq, color_count
        """
        try:
            grid_in = tuple(data["grid_shape_in"])
            grid_out = tuple(data["grid_shape_out"])
        except KeyError as e:
            raise KeyError(f"Missing required grid field: {e}") from e

        if len(grid_in) != 2 or len(grid_out) != 2:
            raise ValueError("grid_shape_in/grid_shape_out must be (H, W) tuples")

        return cls(
            pulse_id=int(data.get("pulse_id", 0)),
            worker_id=int(data.get("worker_id", 0)),
            seq=int(data.get("seq", 0)),
            timestamp=float(data["timestamp"]),
            success=bool(data["success"]),
            task_id=str(data["task_id"]),
            engine_id=str(data["engine_id"]),
            transform_id=str(data["transform_id"]),
            failure_reason=data.get("failure_reason"),
            grid_shape_in=(int(grid_in[0]), int(grid_in[1])),
            grid_shape_out=(int(grid_out[0]), int(grid_out[1])),
            color_count=int(data.get("color_count", 0)),
            train_pair_indices=[int(x) for x in data["train_pair_indices"]],
            error_metrics=dict(data.get("error_metrics", {})),
            params=dict(data.get("params", {})),
            context=dict(data.get("context", {})),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self, string_dict: StringDictionary) -> bytes:
        """
        Serialize to Forge binary format.

        Steps:
          1) String compression → string_dict (task/engine/transform/failure_reason)
          2) Encode error_metrics/params/context with MessagePack+version
          3) Pack 44-byte header
          4) Append variable tail
          5) Append CRC32 checksum
        """
        # 1) String compression
        self.task_id_ref = string_dict.add_string(self.task_id)
        self.engine_id_ref = string_dict.add_string(self.engine_id)
        self.transform_id_ref = string_dict.add_string(self.transform_id)

        if self.failure_reason is not None:
            self.failure_reason_ref = string_dict.add_string(self.failure_reason)
        else:
            self.failure_reason_ref = 0

        # 2) MessagePack blobs (versioned)
        error_metrics_blob = Serialization.encode_with_version(
            self.error_metrics, schema_version=1
        )
        params_blob = Serialization.encode_with_version(
            self.params, schema_version=1
        )
        context_blob = Serialization.encode_with_version(
            self.context, schema_version=1
        )

        if len(self.train_pair_indices) > 255:
            raise ValueError("TRAIN_PAIR_COUNT must fit in uint8 (<= 255)")

        train_pair_count = len(self.train_pair_indices)
        train_pair_bytes = bytes(int(x) & 0xFF for x in self.train_pair_indices)

        def _u16_len(blob: bytes, label: str) -> int:
            if len(blob) > 0xFFFF:
                raise ValueError(f"{label} blob too large for uint16 length field")
            return len(blob)

        error_metrics_len = _u16_len(error_metrics_blob, "error_metrics")
        params_len = _u16_len(params_blob, "params")
        context_len = _u16_len(context_blob, "context")

        grid_h_in, grid_w_in = self.grid_shape_in
        grid_h_out, grid_w_out = self.grid_shape_out

        # 3) Initial header with RECORD_LENGTH=0 (we'll patch it)
        header = struct.pack(
            _HEADER_FMT,
            MAGIC_RECORD,
            RECORD_SCHEMA_VERSION,
            0,  # placeholder
            int(self.pulse_id),
            int(self.worker_id) & 0xFF,
            int(self.seq),
            float(self.timestamp),
            1 if self.success else 0,
            int(self.task_id_ref),
            int(self.engine_id_ref),
            int(self.transform_id_ref),
            int(self.failure_reason_ref),
            int(grid_h_in) & 0xFF,
            int(grid_w_in) & 0xFF,
            int(grid_h_out) & 0xFF,
            int(grid_w_out) & 0xFF,
            int(self.color_count) & 0xFF,
            train_pair_count & 0xFF,
        )
        assert len(header) == _HEADER_SIZE == 44

        # 4) Variable tail (no checksum yet)
        tail = b"".join(
            [
                train_pair_bytes,
                struct.pack("<H", error_metrics_len),
                error_metrics_blob,
                struct.pack("<H", params_len),
                params_blob,
                struct.pack("<H", context_len),
                context_blob,
            ]
        )

        # Now we know total record length INCLUDING checksum
        record_length = len(header) + len(tail) + 4

        # Re-pack header with real RECORD_LENGTH
        header = struct.pack(
            _HEADER_FMT,
            MAGIC_RECORD,
            RECORD_SCHEMA_VERSION,
            record_length,
            int(self.pulse_id),
            int(self.worker_id) & 0xFF,
            int(self.seq),
            float(self.timestamp),
            1 if self.success else 0,
            int(self.task_id_ref),
            int(self.engine_id_ref),
            int(self.transform_id_ref),
            int(self.failure_reason_ref),
            int(grid_h_in) & 0xFF,
            int(grid_w_in) & 0xFF,
            int(grid_h_out) & 0xFF,
            int(grid_w_out) & 0xFF,
            int(self.color_count) & 0xFF,
            train_pair_count & 0xFF,
        )

        record_data = header + tail

        # 5) CRC32
        checksum_value = Checksum.calculate(record_data)
        checksum_bytes = checksum_value.to_bytes(4, "little", signed=False)
        return record_data + checksum_bytes

    # ------------------------------------------------------------------
    # Deserialization
    # ------------------------------------------------------------------

    @classmethod
    def deserialize(cls, data: bytes, string_dict: StringDictionary) -> "ForgeRecord":
        """
        Deserialize from Forge binary format.

        Validates:
          - length
          - CRC32 checksum
          - MAGIC
        """
        if len(data) < _HEADER_SIZE + 4:
            raise ValueError("Data too short to be a valid Forge record")

        # 1) Verify checksum
        record_data = data[:-4]
        checksum_bytes = data[-4:]
        expected_checksum = int.from_bytes(checksum_bytes, "little", signed=False)

        if not Checksum.verify(record_data, expected_checksum):
            raise ValueError("Checksum mismatch - corrupted record")

        # 2) Header
        header = record_data[:_HEADER_SIZE]
        (
            magic,
            version,
            record_length,
            pulse_id,
            worker_id,
            seq,
            timestamp,
            success_byte,
            task_id_ref,
            engine_id_ref,
            transform_id_ref,
            failure_reason_ref,
            grid_h_in,
            grid_w_in,
            grid_h_out,
            grid_w_out,
            color_count,
            train_pair_count,
        ) = struct.unpack(_HEADER_FMT, header)

        if magic != MAGIC_RECORD:
            raise ValueError(f"Invalid MAGIC: {hex(magic)}")

        if record_length != len(data):
            raise ValueError(
                f"Record length mismatch: header={record_length}, actual={len(data)}"
            )

        success = (success_byte == 1)

        # 3) Tail
        offset = _HEADER_SIZE

        # Train pair indices
        if train_pair_count:
            train_pair_slice = record_data[offset : offset + train_pair_count]
            train_pair_indices = list(train_pair_slice)
            offset += train_pair_count
        else:
            train_pair_indices = []

        # Error metrics
        if offset + 2 > len(record_data):
            raise ValueError("Truncated record before error_metrics_len")
        (error_metrics_len,) = struct.unpack("<H", record_data[offset : offset + 2])
        offset += 2
        error_metrics_blob = record_data[offset : offset + error_metrics_len]
        offset += error_metrics_len

        # Params
        if offset + 2 > len(record_data):
            raise ValueError("Truncated record before params_len")
        (params_len,) = struct.unpack("<H", record_data[offset : offset + 2])
        offset += 2
        params_blob = record_data[offset : offset + params_len]
        offset += params_len

        # Context
        if offset + 2 > len(record_data):
            raise ValueError("Truncated record before context_len")
        (context_len,) = struct.unpack("<H", record_data[offset : offset + 2])
        offset += 2
        context_blob = record_data[offset : offset + context_len]
        offset += context_len

        # 4) Strings back out of string_dict
        task_id = string_dict.get_string(task_id_ref)
        engine_id = string_dict.get_string(engine_id_ref)
        transform_id = string_dict.get_string(transform_id_ref)
        if failure_reason_ref == 0:
            failure_reason = None
        else:
            failure_reason = string_dict.get_string(failure_reason_ref)

        # 5) Decode MessagePack dicts
        _, error_metrics = Serialization.decode_with_version(error_metrics_blob)
        _, params = Serialization.decode_with_version(params_blob)
        _, context = Serialization.decode_with_version(context_blob)

        record_dict: Dict[str, Any] = {
            "pulse_id": pulse_id,
            "worker_id": worker_id,
            "seq": seq,
            "timestamp": timestamp,
            "success": success,
            "task_id": task_id,
            "engine_id": engine_id,
            "transform_id": transform_id,
            "failure_reason": failure_reason,
            "grid_shape_in": (grid_h_in, grid_w_in),
            "grid_shape_out": (grid_h_out, grid_w_out),
            "color_count": color_count,
            "train_pair_indices": train_pair_indices,
            "error_metrics": error_metrics,
            "params": params,
            "context": context,
        }

        rec = cls.from_dict(record_dict)
        rec.task_id_ref = task_id_ref
        rec.engine_id_ref = engine_id_ref
        rec.transform_id_ref = transform_id_ref
        rec.failure_reason_ref = failure_reason_ref
        return rec

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict view for query side."""
        return {
            "pulse_id": self.pulse_id,
            "worker_id": self.worker_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "success": self.success,
            "task_id": self.task_id,
            "engine_id": self.engine_id,
            "transform_id": self.transform_id,
            "failure_reason": self.failure_reason,
            "grid_shape_in": self.grid_shape_in,
            "grid_shape_out": self.grid_shape_out,
            "color_count": self.color_count,
            "train_pair_indices": list(self.train_pair_indices),
            "error_metrics": dict(self.error_metrics),
            "params": dict(self.params),
            "context": dict(self.context),
        }
