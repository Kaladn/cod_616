from __future__ import annotations

import os
from typing import Optional, Callable

from forge_memory.core.string_dict import StringDictionary
from forge_memory.core.binary_log import BinaryLog
from forge_memory.bridge.schema_map import TrueVisionSchemaMap
from forge_memory.wal.wal_writer import WALWriter
from forge_memory.wal.pulse_writer import PulseWriter, PulseConfig


def build_truevision_forge_pipeline(
    data_dir: str,
    *,
    eomm_threshold: float = 0.5,
    engine_version: str = "truevision_v2.0.0",
    compositor_version: str = "eomm_v2.0.0",
    worker_id: int = 0,
    pulse_config: Optional[PulseConfig] = None,
    checkpoint_callback: Optional[
        Callable[[int, int, int], None]
    ] = None,  # (pulse_id, record_count, total_bytes)
) -> PulseWriter:
    """
    Construct the full Forge pipeline for TrueVision:

        TrueVision window
          → schema_map
          → PulseWriter
          → WALWriter
          → BinaryLog

    Returns a PulseWriter that you call for each window:
        pulse_writer.submit_window(window_dict)
    """
    os.makedirs(data_dir, exist_ok=True)

    # 1) Shared string dictionary
    from pathlib import Path
    strings_path = Path(data_dir) / "strings.dict"
    string_dict = StringDictionary(strings_path)

    # 2) BinaryLog (permanent storage)
    binary_log = BinaryLog(
        data_dir=data_dir,
        string_dict=string_dict,
        filename="records.bin",
    )

    # 3) WALWriter (durability / crash recovery)
    wal_writer = WALWriter(
        data_dir=data_dir,
        string_dict=string_dict,
        filename="wal.bin",
    )

    # 4) TrueVision schema translator
    schema_map = TrueVisionSchemaMap(
        eomm_threshold=eomm_threshold,
        engine_version=engine_version,
        compositor_version=compositor_version,
    )

    # 5) PulseWriter (batching + orchestration)
    if pulse_config is None:
        pulse_config = PulseConfig(
            max_records_per_pulse=128,
            max_bytes_per_pulse=512 * 1024,
            max_age_ms_per_pulse=250,
        )

    pulse_writer = PulseWriter(
        wal_writer=wal_writer,
        binary_log=binary_log,
        string_dict=string_dict,
        schema_map=schema_map,
        worker_id=worker_id,
        config=pulse_config,
        checkpoint_callback=checkpoint_callback,
    )

    return pulse_writer
