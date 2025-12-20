from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import os
import hashlib


class BinaryLog:
    """Append-only JSONL BinaryLog storing canonical event metadata per line.

    Writes lines to `<data_dir>/records.jsonl`. Each line is a JSON object with
    fields: event_id, seq, t_monotonic_ns, t_utc, source, session_id, payload_hash
    """

    def __init__(self, data_dir: str, mode: str = 'prod'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / 'records.jsonl'
        self.mode = mode
        # Ensure file exists
        if not self.path.exists():
            self.path.write_text('', encoding='utf-8')
        self._closed = False

    def append(self, record: Dict[str, Any]):
        # Serialize record as compact JSON and append atomically
        line = json.dumps(record, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
        # Open in append-binary to ensure bytes and fsync available
        with open(self.path, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')
            fh.flush()
            # Ensure durability
            try:
                os.fsync(fh.fileno())
            except Exception:
                # Windows may raise on some file systems; best-effort
                pass

    def get_record_count(self) -> int:
        # Count lines in file
        with open(self.path, 'r', encoding='utf-8') as fh:
            return sum(1 for _ in fh)

    def read_all(self) -> List[Dict[str, Any]]:
        out = []
        with open(self.path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    def close(self):
        self._closed = True


class PulseWriter:
    """PulseWriter with explicit buffer and flush semantics.

    The writer buffers records and writes them to BinaryLog on flush. This
    preserves ordering and provides explicit flush boundaries.
    """

    def __init__(self, binary_log: BinaryLog, max_records_per_pulse: int = 128):
        self.binary_log = binary_log
        self.max_records_per_pulse = max_records_per_pulse
        self._buffer: List[Dict[str, Any]] = []
        self._closed = False

    def submit_event(self, event: Dict[str, Any]):
        # Map contract event to stored record with payload hash
        payload = event.get('payload', {})
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',',':'))
        payload_hash = 'sha256:' + hashlib.sha256(payload_json.encode('utf-8')).hexdigest()[:16]

        rec = {
            'event_id': event.get('event_id'),
            'seq': event.get('seq'),
            't_monotonic_ns': event.get('t_monotonic_ns'),
            't_utc': event.get('t_utc'),
            'source': event.get('source'),
            'session_id': event.get('session_id'),
            'payload_hash': payload_hash
        }
        self._buffer.append(rec)
        # Optional auto-flush when buffer reaches pulse size
        if len(self._buffer) >= self.max_records_per_pulse:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        for r in self._buffer:
            self.binary_log.append(r)
        self._buffer.clear()

    def close(self):
        self.flush()
        self.binary_log.close()
        self._closed = True


@dataclass
class PulseConfig:
    max_records_per_pulse: int = 128
    max_bytes_per_pulse: int = 512 * 1024
    max_age_ms_per_pulse: int = 250


def build_truevision_forge_pipeline(data_dir: str = 'forge_data', pulse_config: Optional[PulseConfig] = None, *, test_mode: bool = False) -> PulseWriter:
    """Construct the full Forge pipeline for ingestion.

    Parameters:
    - data_dir: target directory for BinaryLog
    - pulse_config: PulseConfig object
    - test_mode: if True, will use the same code path but may write to a temp dir
    """
    pulse_config = pulse_config or PulseConfig()
    # For test-mode, use provided data_dir as the sink (caller may pass a tempdir)
    binary_log = BinaryLog(data_dir, mode='test' if test_mode else 'prod')
    pw = PulseWriter(binary_log=binary_log, max_records_per_pulse=pulse_config.max_records_per_pulse)
    return pw
