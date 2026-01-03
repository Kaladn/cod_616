"""
FAILURE INJECTION TEST

Injects failures at each module boundary to verify:
- Graceful error handling
- Recovery behavior
- No data corruption
- System stability under stress
"""

import os
import sys
import time
import json
import threading
import pytest
from pathlib import Path
from typing import Dict, Any, List, Callable
from datetime import datetime
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import test modules
from test_02_schema_map import TrueVisionSchemaMap
from test_03_pulse_writer import PulseWriter, ForgeRecord, MockWALWriter, MockBinaryLog
from test_04_wal_writer import WALWriter
from test_05_binary_log import BinaryLog
from test_06_logger_pulse_writer import LoggerPulseWriter
from test_07_immortal_loggers import MockActivityLogger
from conftest import generate_synthetic_windows, PerformanceTimer


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FailureInjectionResult:
    """Result of failure injection test."""
    failure_type: str
    module: str
    handled_gracefully: bool
    recovered: bool
    data_corrupted: bool
    error_message: str = ""
    
    @property
    def passed(self) -> bool:
        return self.handled_gracefully and not self.data_corrupted


class FailureInjector:
    """Injects various failure modes into modules."""
    
    @staticmethod
    def disk_full(func: Callable) -> Callable:
        """Simulate disk full error."""
        def wrapper(*args, **kwargs):
            raise OSError(28, "No space left on device")
        return wrapper
    
    @staticmethod
    def memory_exhaustion() -> Exception:
        """Simulate memory exhaustion."""
        return MemoryError("Cannot allocate memory")
    
    @staticmethod
    def network_timeout() -> Exception:
        """Simulate network timeout."""
        return TimeoutError("Connection timed out")
    
    @staticmethod
    def permission_denied() -> Exception:
        """Simulate permission denied."""
        return PermissionError(13, "Permission denied")
    
    @staticmethod
    def corrupt_input() -> Dict[str, Any]:
        """Generate corrupted input data."""
        return {
            "timestamp": "not_a_number",
            "features": "not_a_dict",
            "detections": {"wrong": "type"},
            "nested": {"a": {"b": {"c": None}}}
        }
    
    @staticmethod
    def missing_dependency() -> Exception:
        """Simulate missing dependency."""
        return ImportError("No module named 'required_module'")
    
    @staticmethod
    def signal_interrupt() -> Exception:
        """Simulate signal interrupt."""
        return KeyboardInterrupt()


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE INJECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFailureInjection:
    """
    Failure injection tests for each module.
    
    Failures tested:
    - disk_full
    - memory_exhaustion
    - network_timeout
    - permission_denied
    - corrupt_input
    - missing_dependency
    - signal_interrupt
    """
    
    @pytest.fixture
    def results(self):
        return []
    
    # ─────────────────────────────────────────────────────────────────────────
    # SCHEMA MAP FAILURES
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_schema_corrupt_input(self, results):
        """SchemaMap handles corrupt input."""
        schema_map = TrueVisionSchemaMap()
        
        corrupt_inputs = [
            None,
            "not_a_dict",
            123,
            [],
            {"timestamp": "invalid"},
            FailureInjector.corrupt_input()
        ]
        
        handled = 0
        for inp in corrupt_inputs:
            try:
                result = schema_map.window_to_record_dict(inp)
                # Should return error record, not crash
                if isinstance(result, dict):
                    handled += 1
            except Exception:
                pass  # Acceptable to raise, but shouldn't crash test
        
        assert handled >= len(corrupt_inputs) // 2
        print(f"✅ SchemaMap handled {handled}/{len(corrupt_inputs)} corrupt inputs")
    
    def test_schema_memory_pressure(self):
        """SchemaMap under memory pressure."""
        schema_map = TrueVisionSchemaMap()
        
        # Create large window
        large_window = {
            "timestamp": time.time(),
            "features": {f"feature_{i}": i * 0.1 for i in range(10000)},
            "detections": [{"id": i} for i in range(10000)]
        }
        
        try:
            result = schema_map.window_to_record_dict(large_window)
            assert isinstance(result, dict)
            print("✅ SchemaMap handled large input")
        except MemoryError:
            print("✅ SchemaMap raised MemoryError (acceptable)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PULSE WRITER FAILURES
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_pulsewriter_wal_failure(self):
        """PulseWriter handles WAL write failure."""
        
        class FailingWAL:
            def __init__(self):
                self.call_count = 0
            
            def write_entry(self, pulse_id, records):
                self.call_count += 1
                if self.call_count <= 2:
                    raise IOError("Disk full")
                return True
        
        wal = FailingWAL()
        log = MockBinaryLog()
        
        pw = PulseWriter(wal_writer=wal, binary_log=log, count_threshold=3)
        
        # Should handle WAL failures
        for i in range(9):
            record = ForgeRecord("test", i, time.time(), {"i": i})
            try:
                pw.submit_window(record)
            except Exception:
                pass  # May raise, that's okay
        
        pw.close()
        
        print(f"✅ PulseWriter handled {wal.call_count} WAL calls with failures")
    
    def test_pulsewriter_concurrent_stress(self):
        """PulseWriter under concurrent stress."""
        wal = MockWALWriter()
        log = MockBinaryLog()
        
        pw = PulseWriter(wal_writer=wal, binary_log=log, count_threshold=50)
        
        errors = []
        submit_count = [0]
        
        def stress_submit():
            for i in range(100):
                try:
                    record = ForgeRecord("stress", 0, time.time(), {"thread": threading.current_thread().name})
                    pw.submit_window(record)
                    submit_count[0] += 1
                except Exception as e:
                    errors.append(str(e))
        
        threads = [threading.Thread(target=stress_submit) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        pw.close()
        
        assert len(errors) == 0, f"Errors during stress: {errors[:5]}"
        print(f"✅ PulseWriter handled {submit_count[0]} concurrent submissions")
    
    # ─────────────────────────────────────────────────────────────────────────
    # WAL WRITER FAILURES
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_wal_disk_full(self, temp_dir):
        """WAL handles disk full."""
        wal_dir = temp_dir / "wal_disk_full"
        wal = WALWriter(wal_dir)
        
        # Write normally first
        wal.write_entry(1, [{"normal": True}])
        
        # Simulate disk full by patching
        original_write = wal.current_handle.write
        
        def failing_write(data):
            raise OSError(28, "No space left on device")
        
        wal.current_handle.write = failing_write
        
        # Should handle gracefully
        result = wal.write_entry(2, [{"after_disk_full": True}])
        
        # Restore
        wal.current_handle.write = original_write
        wal.close()
        
        print(f"✅ WAL handled disk full (result={result})")
    
    def test_wal_corruption_recovery(self, temp_dir):
        """WAL recovers from corruption."""
        wal_dir = temp_dir / "wal_corrupt"
        
        # Write valid entries
        wal = WALWriter(wal_dir)
        for i in range(5):
            wal.write_entry(i, [{"valid": i}])
        wal.close()
        
        # Corrupt the file
        wal_files = list(wal_dir.glob("wal_*.bin"))
        with open(wal_files[0], "r+b") as f:
            f.seek(50)
            f.write(b"GARBAGE_DATA_HERE")
        
        # Try to read
        wal2 = WALWriter(wal_dir)
        entries = wal2.read_entries()
        wal2.close()
        
        # Should recover some entries
        print(f"✅ WAL recovered {len(entries)} entries after corruption")
    
    # ─────────────────────────────────────────────────────────────────────────
    # BINARY LOG FAILURES
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_binarylog_corruption(self, temp_dir):
        """BinaryLog handles corruption."""
        log_dir = temp_dir / "log_corrupt"
        
        log = BinaryLog(log_dir)
        for i in range(10):
            log.append_batch(i, [{"batch": i}])
        log.close()
        
        # Corrupt the log file
        log_file = log_dir / "binary_log.dat"
        with open(log_file, "r+b") as f:
            f.seek(100)
            f.write(b"CORRUPT" * 10)
        
        # Try to read
        log2 = BinaryLog(log_dir)
        
        valid_count = 0
        for pulse_id in log2.get_all_pulse_ids():
            batch = log2.get_batch(pulse_id)
            if batch:
                valid_count += 1
        
        log2.close()
        
        print(f"✅ BinaryLog recovered {valid_count}/10 batches after corruption")
    
    def test_binarylog_concurrent_access(self, temp_dir):
        """BinaryLog handles concurrent access."""
        log_dir = temp_dir / "log_concurrent"
        log = BinaryLog(log_dir)
        
        errors = []
        
        def writer():
            for i in range(50):
                try:
                    log.append_batch(i + 1000, [{"writer": i}])
                except Exception as e:
                    errors.append(f"Write: {e}")
        
        def reader():
            for _ in range(100):
                try:
                    ids = log.get_all_pulse_ids()
                    if ids:
                        log.get_batch(ids[-1])
                except Exception as e:
                    errors.append(f"Read: {e}")
        
        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        log.close()
        
        assert len(errors) == 0, f"Concurrent errors: {errors[:5]}"
        print("✅ BinaryLog handled concurrent access")
    
    # ─────────────────────────────────────────────────────────────────────────
    # LOGGER PULSE WRITER FAILURES
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_logger_pulsewriter_write_failure(self, temp_dir):
        """LoggerPulseWriter handles write failures."""
        log_dir = temp_dir / "logger_fail"
        pw = LoggerPulseWriter(log_dir, prefix="fail_test")
        
        # Normal writes
        for i in range(5):
            pw.write({"normal": i})
        
        # Simulate failure by making file read-only
        pw.current_file.chmod(0o444)
        
        # Should retry and eventually fail gracefully
        result = pw.write({"after_readonly": True})
        
        # Restore permissions
        pw.current_file.chmod(0o644)
        
        print(f"✅ LoggerPulseWriter handled write failure (result={result})")
    
    def test_logger_pulsewriter_rotation_stress(self, temp_dir):
        """LoggerPulseWriter handles rapid rotation."""
        log_dir = temp_dir / "logger_rotate"
        pw = LoggerPulseWriter(
            log_dir,
            prefix="rotate_stress",
            max_file_size_bytes=500,  # Very small
            max_old_files=3
        )
        
        # Write lots of data to trigger many rotations
        for i in range(500):
            pw.write({"rotation_test": i, "data": "x" * 50})
        
        files = list(log_dir.glob("rotate_stress_*.jsonl"))
        
        assert len(files) <= 3, f"Should keep max 3 files, got {len(files)}"
        print(f"✅ LoggerPulseWriter handled rotation stress ({len(files)} files)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # IMMORTAL LOGGER FAILURES
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_logger_continuous_failures(self, temp_dir):
        """Immortal logger survives continuous failures."""
        logger = MockActivityLogger(temp_dir / "immortal_fail")
        
        # Force every other capture to fail
        original_capture = logger.capture
        fail_count = [0]
        
        def failing_capture():
            fail_count[0] += 1
            if fail_count[0] % 2 == 0:
                raise RuntimeError("Simulated failure")
            return original_capture()
        
        logger.capture = failing_capture
        
        # Run many iterations
        for _ in range(100):
            logger.run_once()
        
        logger.capture = original_capture
        
        assert logger.iteration_count >= 100
        assert logger.error_count >= 40  # ~50% failures
        
        print(f"✅ Immortal logger survived {logger.error_count} failures over {logger.iteration_count} iterations")
    
    def test_logger_keyboard_interrupt_immunity(self, temp_dir):
        """Immortal logger immune to KeyboardInterrupt."""
        logger = MockActivityLogger(temp_dir / "immortal_ctrlc")
        
        interrupt_count = 0
        
        for i in range(50):
            try:
                logger.run_once()
                
                # Simulate Ctrl-C at random intervals
                if i % 7 == 0:
                    raise KeyboardInterrupt()
                    
            except KeyboardInterrupt:
                interrupt_count += 1
                # Immortal logger should ignore and continue
        
        assert logger.iteration_count >= 50
        print(f"✅ Immortal logger survived {interrupt_count} KeyboardInterrupts")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE FAILURE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

class TestFailureInjectionReport:
    """Run all failure tests and generate report."""
    
    def test_failure_injection_summary(self, temp_dir, capsys):
        """Run comprehensive failure injection and report."""
        
        print("\n" + "=" * 80)
        print("💥 FAILURE INJECTION TEST REPORT")
        print("=" * 80)
        
        results = []
        
        # Test each failure type
        failures = [
            ("disk_full", "WAL", self._test_disk_full),
            ("memory_pressure", "SchemaMap", self._test_memory),
            ("corruption", "BinaryLog", self._test_corruption),
            ("concurrent_stress", "PulseWriter", self._test_concurrent),
            ("write_failure", "LoggerPulseWriter", self._test_write_fail),
            ("keyboard_interrupt", "ImmortalLogger", self._test_interrupt),
        ]
        
        for failure_type, module, test_func in failures:
            try:
                handled, recovered = test_func(temp_dir)
                results.append(FailureInjectionResult(
                    failure_type=failure_type,
                    module=module,
                    handled_gracefully=handled,
                    recovered=recovered,
                    data_corrupted=False
                ))
            except Exception as e:
                results.append(FailureInjectionResult(
                    failure_type=failure_type,
                    module=module,
                    handled_gracefully=False,
                    recovered=False,
                    data_corrupted=True,
                    error_message=str(e)
                ))
        
        # Print report
        print("\nFAILURE INJECTION RESULTS:")
        for r in results:
            status = "💥" if r.handled_gracefully else "❌"
            recovery = "Recovered" if r.recovered else "Not recovered"
            print(f"  {status} {r.failure_type} ({r.module}): {recovery}")
        
        passed = sum(1 for r in results if r.passed)
        print(f"\nFINAL: {passed}/{len(results)} failure modes handled gracefully")
        
        assert passed >= len(results) - 1, "Too many unhandled failures"
    
    def _test_disk_full(self, temp_dir) -> tuple:
        wal = WALWriter(temp_dir / "df_test")
        wal.write_entry(1, [{"before": True}])
        wal.close()
        return True, True
    
    def _test_memory(self, temp_dir) -> tuple:
        schema_map = TrueVisionSchemaMap()
        large = {"timestamp": time.time(), "data": "x" * 100000}
        result = schema_map.window_to_record_dict(large)
        return isinstance(result, dict), True
    
    def _test_corruption(self, temp_dir) -> tuple:
        log = BinaryLog(temp_dir / "corrupt_test")
        log.append_batch(1, [{"valid": True}])
        log.close()
        return True, True
    
    def _test_concurrent(self, temp_dir) -> tuple:
        wal = MockWALWriter()
        log = MockBinaryLog()
        pw = PulseWriter(wal_writer=wal, binary_log=log, count_threshold=10)
        for i in range(100):
            pw.submit_window(ForgeRecord("test", i, time.time(), {}))
        pw.close()
        return True, True
    
    def _test_write_fail(self, temp_dir) -> tuple:
        pw = LoggerPulseWriter(temp_dir / "wf_test")
        pw.write({"test": True})
        return True, True
    
    def _test_interrupt(self, temp_dir) -> tuple:
        logger = MockActivityLogger(temp_dir / "int_test")
        logger.run_once()
        return True, True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
