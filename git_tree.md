# Repository file tree (generated)

Root: d:/cod_616

```
.
├─ LICENSE
├─ syswarn_12-30-25.jsonl
├─ archive/
│  └─ README.md
├─ Copilot Question Set 1-5 with answers.md
├─ copilot and chatgpt round 2— Create missing core mod.md
├─ loggers/
│  ├─ __init__.py
│  ├─ writer.py
│  ├─ process_logger.py
│  ├─ Oggers-Info.md
│  ├─ network_service.py
│  ├─ network_logger.ps1
│  ├─ network_integration.py
│  ├─ input_service.py
│  ├─ input_logger.py
│  ├─ input_integration.py
│  ├─ gamepad_logger_continuous.py
│  ├─ activity_service.py
│  ├─ activity_logger.py
│  ├─ activity_integration.py
│  └─ __pycache__/
├─ resilience/
│  ├─ __init__.py
│  ├─ pulse_bus.py
│  ├─ hot_spare.py
│  ├─ disk_guard.py
│  └─ __pycache__/
├─ forge_memory/
│  ├─ __init__.py
│  ├─ core/
│  │  ├─ string_dict.py
│  │  ├─ record.py
│  │  ├─ binary_log.py
│  │  └─ __pycache__/
│  ├─ indexes/
│  │  ├─ __init__.py
│  │  ├─ vcache.py
│  │  ├─ hash_index.py
│  │  ├─ bitmap_index.py
│  │  └─ __pycache__/
│  ├─ pulse/
│  │  ├─ worker_pool.py
│  │  ├─ pulse_writer.py
│  │  └─ __pycache__/
│  ├─ utils/
│  │  └─ __pycache__/
│  └─ wal/
│     └─ __init__.py
├─ tests/
│  ├─ unit/
│  │  ├─ test_writer_smoke.py
│  │  ├─ test_vcache.py
│  │  ├─ test_pulse_bus.py
│  │  ├─ test_hot_spare.py
│  │  ├─ test_hash_index.py
│  │  ├─ test_disk_guard.py
│  │  ├─ test_bitmap_index.py
│  │  └─ __pycache__/
│  └─ integration/
│     ├─ test_end_to_end.py
│     └─ __pycache__/
├─ playground/
│  ├─ temp_dbg.py
│  ├─ temp_dbg_copy.py
│  ├─ temp_check.py
│  ├─ temp_check_all.py
│  └─ inspect_sys.py
├─ scripts/
│  └─ __pycache__/
├─ CompucCogLogger/
│  └─ __pycache__/
└─ other files / pycache artifacts
```

Notes:
- This tree focuses on tracked source files and useful artifacts discovered in the workspace, not every generated `*.pyc` file.
- New additions in the recent change set include `resilience/hot_spare.py` and `tests/unit/test_hot_spare.py`, and `playground/` relocations of debug scripts.
- Let me know if you want the tree expanded with a full recursive listing of every file (including all `__pycache__` entries), or if you prefer it saved to another filename or format.
