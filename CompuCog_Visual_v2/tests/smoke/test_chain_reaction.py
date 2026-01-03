"""
CHAIN REACTION SMOKE TEST

The main test that validates the entire data pipeline:
    Source → SchemaMap → PulseWriter → WAL → BinaryLog
    Logger → LoggerPulseWriter → File

This test finds EVERY SINGLE BREAK in the data pipeline.
"""

import os
import sys
import time
import json
import pytest
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "loggers"))

# Import all test modules
from conftest import (
    generate_synthetic_windows,
    generate_synthetic_logs,
    SyntheticWindowData,
    PerformanceTimer,
    MemoryTracker
)
from test_02_schema_map import TrueVisionSchemaMap
from test_03_pulse_writer import PulseWriter, ForgeRecord, MockWALWriter, MockBinaryLog
from test_04_wal_writer import WALWriter
from test_05_binary_log import BinaryLog
from test_06_logger_pulse_writer import LoggerPulseWriter
from test_07_immortal_loggers import (
    MockActivityLogger,
    MockInputLogger,
    MockGamepadLogger,
    MockProcessLogger,
    MockNetworkLogger
)


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN LINK DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChainTestResult:
    """Result of a chain link test."""
    source: str
    target: str
    success: bool
    items_tested: int
    items_passed: int
    errors: List[str]
    
    @property
    def pass_rate(self) -> float:
        if self.items_tested == 0:
            return 0.0
        return self.items_passed / self.items_tested


class ChainReactionTest:
    """
    Executes chain reaction smoke tests.
    
    Philosophy:
        Test each module IN ISOLATION, then IN CHAIN.
        Each module's output MUST match next module's expected input.
        Any break in the chain = SYSTEM FAILURE.
    """
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.results: List[ChainTestResult] = []
        self.chain_breaks: List[str] = []
    
    def run_all(self) -> bool:
        """Run all chain reaction tests."""
        print("🔗 CHAIN REACTION SMOKE TEST STARTING")
        print("=" * 80)
        
        # Phase 1: Test each chain link
        self._test_source_to_schema()
        self._test_schema_to_pulsewriter()
        self._test_pulsewriter_to_wal()
        self._test_wal_to_binarylog()
        self._test_logger_to_pulsewriter()
        
        # Phase 2: Full pipeline test
        self._test_full_pipeline()
        
        # Report
        self._print_report()
        
        return len(self.chain_breaks) == 0
    
    def _test_source_to_schema(self):
        """Test: TrueVision Source → SchemaMap"""
        print("\n[1/6] Testing Source → SchemaMap...")
        
        schema_map = TrueVisionSchemaMap()
        windows = generate_synthetic_windows(1000)
        
        errors = []
        passed = 0
        
        for i, window in enumerate(windows):
            try:
                record = schema_map.window_to_record_dict(window)
                
                # Validate output
                if "worker_id" in record and "seq" in record and "data" in record:
                    passed += 1
                else:
                    errors.append(f"Window {i}: Missing required fields")
                    
            except Exception as e:
                errors.append(f"Window {i}: {e}")
        
        result = ChainTestResult(
            source="TrueVisionSource",
            target="SchemaMap",
            success=len(errors) == 0,
            items_tested=len(windows),
            items_passed=passed,
            errors=errors[:10]  # First 10 errors
        )
        
        self.results.append(result)
        
        if not result.success:
            self.chain_breaks.append(f"Source → SchemaMap: {len(errors)} failures")
        
        print(f"   {'✅' if result.success else '❌'} {passed}/{len(windows)} windows processed")
    
    def _test_schema_to_pulsewriter(self):
        """Test: SchemaMap → PulseWriter"""
        print("\n[2/6] Testing SchemaMap → PulseWriter...")
        
        schema_map = TrueVisionSchemaMap()
        wal_writer = MockWALWriter()
        binary_log = MockBinaryLog()
        
        pw = PulseWriter(
            wal_writer=wal_writer,
            binary_log=binary_log,
            count_threshold=100
        )
        
        windows = generate_synthetic_windows(500)
        errors = []
        
        for i, window in enumerate(windows):
            try:
                record_dict = schema_map.window_to_record_dict(window)
                forge_record = ForgeRecord.from_dict(record_dict)
                pw.submit_window(forge_record)
                
            except Exception as e:
                errors.append(f"Record {i}: {e}")
        
        pw.close()
        
        # Verify records made it to WAL
        total_in_wal = sum(len(batch[1]) for batch in wal_writer.entries)
        
        result = ChainTestResult(
            source="SchemaMap",
            target="PulseWriter",
            success=total_in_wal == len(windows),
            items_tested=len(windows),
            items_passed=total_in_wal,
            errors=errors[:10]
        )
        
        self.results.append(result)
        
        if not result.success:
            self.chain_breaks.append(f"SchemaMap → PulseWriter: {len(windows) - total_in_wal} lost")
        
        print(f"   {'✅' if result.success else '❌'} {total_in_wal}/{len(windows)} records to WAL")
    
    def _test_pulsewriter_to_wal(self):
        """Test: PulseWriter → WALWriter"""
        print("\n[3/6] Testing PulseWriter → WALWriter...")
        
        wal_dir = self.temp_dir / "wal_test"
        wal_writer = WALWriter(wal_dir)
        binary_log = MockBinaryLog()
        
        pw = PulseWriter(
            wal_writer=wal_writer,
            binary_log=binary_log,
            count_threshold=50
        )
        
        # Submit records
        for i in range(250):
            record = ForgeRecord("test", i, time.time(), {"i": i})
            pw.submit_window(record)
        
        pw.close()
        wal_writer.close()
        
        # Read WAL and count
        reader = WALWriter(wal_dir)
        entries = reader.read_entries()
        reader.close()
        
        total_in_wal = sum(len(e.records) for e in entries)
        
        result = ChainTestResult(
            source="PulseWriter",
            target="WALWriter",
            success=total_in_wal == 250,
            items_tested=250,
            items_passed=total_in_wal,
            errors=[]
        )
        
        self.results.append(result)
        
        if not result.success:
            self.chain_breaks.append(f"PulseWriter → WAL: {250 - total_in_wal} lost")
        
        print(f"   {'✅' if result.success else '❌'} {total_in_wal}/250 records in WAL ({len(entries)} pulses)")
    
    def _test_wal_to_binarylog(self):
        """Test: WALWriter → BinaryLog"""
        print("\n[4/6] Testing WALWriter → BinaryLog...")
        
        wal_dir = self.temp_dir / "wal_to_log"
        log_dir = self.temp_dir / "binary_log"
        
        # Write to WAL
        wal = WALWriter(wal_dir)
        for i in range(10):
            wal.write_entry(i, [{"batch": i, "data": f"item_{j}"} for j in range(10)])
        
        entries = wal.read_entries()
        wal.close()
        
        # Recover to BinaryLog
        binary_log = BinaryLog(log_dir)
        recovered = binary_log.recover_from_wal(entries)
        
        # Verify all batches present
        all_ids = binary_log.get_all_pulse_ids()
        
        result = ChainTestResult(
            source="WALWriter",
            target="BinaryLog",
            success=len(all_ids) == 10,
            items_tested=10,
            items_passed=len(all_ids),
            errors=[]
        )
        
        self.results.append(result)
        binary_log.close()
        
        if not result.success:
            self.chain_breaks.append(f"WAL → BinaryLog: {10 - len(all_ids)} batches lost")
        
        print(f"   {'✅' if result.success else '❌'} {len(all_ids)}/10 batches recovered")
    
    def _test_logger_to_pulsewriter(self):
        """Test: Immortal Loggers → LoggerPulseWriter"""
        print("\n[5/6] Testing Loggers → LoggerPulseWriter...")
        
        log_dir = self.temp_dir / "logger_chain"
        pw = LoggerPulseWriter(log_dir, prefix="chain_test")
        
        loggers = [
            MockActivityLogger(log_dir / "activity"),
            MockInputLogger(log_dir / "input"),
            MockGamepadLogger(log_dir / "gamepad"),
            MockProcessLogger(log_dir / "process"),
            MockNetworkLogger(log_dir / "network")
        ]
        
        # Each logger writes 100 records
        total_expected = len(loggers) * 100
        
        for logger in loggers:
            for _ in range(100):
                data = logger.capture()
                pw.write(data)
        
        # Count in file
        records = pw.get_recent_records(1000)
        
        result = ChainTestResult(
            source="ImmortalLoggers",
            target="LoggerPulseWriter",
            success=len(records) == total_expected,
            items_tested=total_expected,
            items_passed=len(records),
            errors=[]
        )
        
        self.results.append(result)
        
        if not result.success:
            self.chain_breaks.append(f"Loggers → PulseWriter: {total_expected - len(records)} lost")
        
        print(f"   {'✅' if result.success else '❌'} {len(records)}/{total_expected} log entries")
    
    def _test_full_pipeline(self):
        """Test: Complete pipeline with synthetic data"""
        print("\n[6/6] Testing Full Pipeline...")
        
        # Setup full chain
        wal_dir = self.temp_dir / "full_wal"
        log_dir = self.temp_dir / "full_log"
        
        schema_map = TrueVisionSchemaMap()
        wal_writer = WALWriter(wal_dir)
        binary_log = BinaryLog(log_dir)
        
        pw = PulseWriter(
            wal_writer=wal_writer,
            binary_log=binary_log,
            count_threshold=100
        )
        
        # Process synthetic windows
        windows = generate_synthetic_windows(1000)
        
        with PerformanceTimer("Full pipeline", threshold_ms=10000) as pt:
            for window in windows:
                record_dict = schema_map.window_to_record_dict(window)
                forge_record = ForgeRecord.from_dict(record_dict)
                pw.submit_window(forge_record)
        
        pw.close()
        wal_writer.close()
        
        # Verify end-to-end
        all_batches = binary_log.get_all_pulse_ids()
        total_records = 0
        for pulse_id in all_batches:
            batch = binary_log.get_batch(pulse_id)
            if batch:
                total_records += len(batch.records)
        
        binary_log.close()
        
        result = ChainTestResult(
            source="FullPipeline",
            target="BinaryLog",
            success=total_records == 1000,
            items_tested=1000,
            items_passed=total_records,
            errors=[]
        )
        
        self.results.append(result)
        
        if not result.success:
            self.chain_breaks.append(f"Full Pipeline: {1000 - total_records} lost")
        
        print(f"   {'✅' if result.success else '❌'} {total_records}/1000 records end-to-end")
        print(f"   ⏱️ Pipeline latency: {pt.elapsed_ms:.2f}ms ({pt.elapsed_ms/1000:.4f}ms/record)")
    
    def _print_report(self):
        """Print final test report."""
        print("\n" + "=" * 80)
        print("🔗 CHAIN REACTION SMOKE TEST REPORT")
        print("=" * 80)
        print(f"Test Time: {datetime.now().isoformat()}")
        
        print("\nCHAIN VALIDATION RESULTS:")
        for result in self.results:
            status = "✅" if result.success else "❌"
            print(f"  {status} {result.source} → {result.target}: "
                  f"{result.items_passed}/{result.items_tested} ({result.pass_rate*100:.1f}%)")
        
        if self.chain_breaks:
            print(f"\n❌ CHAIN BREAKS FOUND: {len(self.chain_breaks)}")
            for break_info in self.chain_breaks:
                print(f"   - {break_info}")
            print("\n❌ SYSTEM UNSTABLE - FIX CHAIN BREAKS")
        else:
            print("\n✅ ALL CHAINS INTACT - SYSTEM STABLE")


# ═══════════════════════════════════════════════════════════════════════════════
# PYTEST TEST CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class TestChainReaction:
    """Chain Reaction integration test."""
    
    def test_complete_chain_reaction(self, temp_dir):
        """Run complete chain reaction test."""
        chain_test = ChainReactionTest(temp_dir)
        result = chain_test.run_all()
        
        assert result is True, f"Chain breaks detected: {chain_test.chain_breaks}"
    
    def test_data_fidelity_through_chain(self, temp_dir):
        """Verify data fidelity through entire chain."""
        # Create unique data
        test_value = f"fidelity_test_{time.time()}"
        
        schema_map = TrueVisionSchemaMap()
        wal_dir = temp_dir / "fidelity_wal"
        log_dir = temp_dir / "fidelity_log"
        
        wal = WALWriter(wal_dir)
        log = BinaryLog(log_dir)
        
        pw = PulseWriter(wal_writer=wal, binary_log=log, count_threshold=1)
        
        # Send through chain
        window = SyntheticWindowData()
        window.features["fidelity_marker"] = test_value
        
        record_dict = schema_map.window_to_record_dict(window.to_dict())
        forge_record = ForgeRecord.from_dict(record_dict)
        pw.submit_window(forge_record)
        pw.close()
        wal.close()
        
        # Retrieve and verify
        batch = log.get_batch(1)
        log.close()
        
        assert batch is not None
        assert batch.records[0]["data"]["features"]["fidelity_marker"] == test_value
        
        print(f"✅ Data fidelity verified: {test_value}")
    
    def test_chain_latency(self, temp_dir):
        """Measure chain latency."""
        schema_map = TrueVisionSchemaMap()
        wal = MockWALWriter()
        log = MockBinaryLog()
        pw = PulseWriter(wal_writer=wal, binary_log=log, count_threshold=100)
        
        window = SyntheticWindowData().to_dict()
        
        # Measure single record latency
        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            
            record_dict = schema_map.window_to_record_dict(window)
            forge_record = ForgeRecord.from_dict(record_dict)
            pw.submit_window(forge_record)
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
        
        pw.close()
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 5, f"Average latency too high: {avg_latency:.2f}ms"
        
        print(f"✅ Chain latency: avg={avg_latency:.4f}ms, max={max_latency:.4f}ms")
    
    def test_chain_memory_stability(self, temp_dir):
        """Verify no memory leaks through chain."""
        schema_map = TrueVisionSchemaMap()
        wal = MockWALWriter()
        log = MockBinaryLog()
        pw = PulseWriter(wal_writer=wal, binary_log=log, count_threshold=100)
        
        windows = generate_synthetic_windows(1000)
        
        with MemoryTracker("Chain processing", max_growth_mb=50) as mt:
            for window in windows:
                record_dict = schema_map.window_to_record_dict(window)
                forge_record = ForgeRecord.from_dict(record_dict)
                pw.submit_window(forge_record)
            
            pw.close()
        
        assert mt.within_bounds, f"Memory grew by {mt.growth_mb:.2f}MB"
        print(f"✅ Chain memory stable: +{mt.growth_mb:.2f}MB")


if __name__ == "__main__":
    # Run standalone
    import tempfile
    import shutil
    
    temp = Path(tempfile.mkdtemp(prefix="chain_reaction_"))
    try:
        test = ChainReactionTest(temp)
        success = test.run_all()
        sys.exit(0 if success else 1)
    finally:
        shutil.rmtree(temp, ignore_errors=True)
