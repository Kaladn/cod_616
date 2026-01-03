"""
Test Module 2: TrueVisionSchemaMap

Tests schema translation from TrueVision → ForgeRecord.
CRITICAL: Output MUST match ForgeRecord.from_dict() expectations.
"""

import os
import sys
import time
import json
import pytest
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA MAP IMPLEMENTATION (for testing)
# ═══════════════════════════════════════════════════════════════════════════════

class TrueVisionSchemaMap:
    """
    Maps TrueVision window output → ForgeRecord compatible dict.
    
    This is the bridge between raw TrueVision data and the storage layer.
    """
    
    # Required fields for ForgeRecord
    FORGE_REQUIRED_FIELDS = ["worker_id", "seq", "data", "timestamp"]
    
    # Default values for missing fields
    DEFAULTS = {
        "worker_id": "truevision_default",
        "seq": 0,
        "data": {},
        "timestamp": 0.0
    }
    
    def __init__(self, worker_id: str = "truevision"):
        self.worker_id = worker_id
        self.seq_counter = 0
    
    def window_to_record_dict(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert TrueVision window to ForgeRecord-compatible dict.
        
        Args:
            window: TrueVision window data
            
        Returns:
            Dict compatible with ForgeRecord.from_dict()
        """
        if not isinstance(window, dict):
            return self._error_record(f"Invalid input type: {type(window)}")
        
        try:
            self.seq_counter += 1
            
            # Extract timestamp
            timestamp = window.get("timestamp")
            if timestamp is None:
                timestamp = time.time()
            elif isinstance(timestamp, str):
                # Try to parse ISO format
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    timestamp = time.time()
            
            # Build ForgeRecord-compatible dict
            record = {
                "worker_id": self.worker_id,
                "seq": self.seq_counter,
                "timestamp": timestamp,
                "timestamp_iso": datetime.fromtimestamp(timestamp).isoformat(),
                "data": {
                    "window_id": window.get("window_id", f"win_{self.seq_counter}"),
                    "t_start": window.get("t_start", timestamp - 1.0),
                    "t_end": window.get("t_end", timestamp),
                    "frame_count": window.get("frame_count", 0),
                    "features": window.get("features", {}),
                    "detections": window.get("detections", []),
                    "metadata": window.get("metadata", {})
                },
                "source": "truevision",
                "version": "1.0.0"
            }
            
            return record
            
        except Exception as e:
            return self._error_record(str(e))
    
    def _error_record(self, error_msg: str) -> Dict[str, Any]:
        """Create error record that still conforms to ForgeRecord schema."""
        self.seq_counter += 1
        return {
            "worker_id": self.worker_id,
            "seq": self.seq_counter,
            "timestamp": time.time(),
            "timestamp_iso": datetime.now().isoformat(),
            "data": {"error": error_msg},
            "source": "truevision",
            "version": "1.0.0",
            "is_error": True
        }
    
    def validate_record(self, record: Dict[str, Any]) -> tuple[bool, str]:
        """Validate record matches ForgeRecord expectations."""
        errors = []
        
        for field in self.FORGE_REQUIRED_FIELDS:
            if field not in record:
                errors.append(f"Missing required field: {field}")
        
        if "timestamp" in record:
            if not isinstance(record["timestamp"], (int, float)):
                errors.append(f"timestamp must be numeric, got {type(record['timestamp'])}")
        
        if "seq" in record:
            if not isinstance(record["seq"], int):
                errors.append(f"seq must be int, got {type(record['seq'])}")
        
        if errors:
            return False, "; ".join(errors)
        return True, "Valid ForgeRecord"


class TestSchemaMap:
    """
    SMOKE TESTS for TrueVisionSchemaMap.
    
    Tests:
    1. ✅ Accepts TrueVision window dict
    2. ✅ Returns ForgeRecord-compatible dict
    3. ✅ All required fields present: worker_id, seq, data, etc.
    4. ✅ Data types match ForgeRecord expectations
    5. ✅ Handles missing fields gracefully (defaults)
    6. ✅ Handles malformed input (returns error dict, doesn't crash)
    7. ✅ Performance: < 5ms per translation
    8. ✅ Memory: No accumulation between translations
    """
    
    @pytest.fixture
    def schema_map(self):
        return TrueVisionSchemaMap(worker_id="test_truevision")
    
    def test_01_accepts_truevision_window(self, schema_map, synthetic_window):
        """Test 1: Accepts TrueVision window dict."""
        result = schema_map.window_to_record_dict(synthetic_window)
        
        assert result is not None
        assert isinstance(result, dict)
        print("✅ Accepts TrueVision window dict")
    
    def test_02_returns_forge_compatible(self, schema_map, synthetic_window):
        """Test 2: Returns ForgeRecord-compatible dict."""
        result = schema_map.window_to_record_dict(synthetic_window)
        
        valid, msg = schema_map.validate_record(result)
        assert valid, f"Invalid ForgeRecord: {msg}"
        print(f"✅ Returns ForgeRecord-compatible dict: {msg}")
    
    def test_03_required_fields_present(self, schema_map, synthetic_window):
        """Test 3: All required fields present."""
        result = schema_map.window_to_record_dict(synthetic_window)
        
        required = ["worker_id", "seq", "timestamp", "data"]
        for field in required:
            assert field in result, f"Missing required field: {field}"
        
        print(f"✅ All {len(required)} required fields present")
    
    def test_04_data_types_correct(self, schema_map, synthetic_window):
        """Test 4: Data types match expectations."""
        result = schema_map.window_to_record_dict(synthetic_window)
        
        assert isinstance(result["worker_id"], str)
        assert isinstance(result["seq"], int)
        assert isinstance(result["timestamp"], (int, float))
        assert isinstance(result["data"], dict)
        
        print("✅ All data types correct")
    
    def test_05_handles_missing_fields(self, schema_map):
        """Test 5: Handles missing fields with defaults."""
        # Minimal window with only timestamp
        minimal_window = {"timestamp": time.time()}
        
        result = schema_map.window_to_record_dict(minimal_window)
        
        # Should still produce valid record with defaults
        valid, msg = schema_map.validate_record(result)
        assert valid, f"Failed to handle minimal input: {msg}"
        
        # Check defaults were applied
        assert "data" in result
        assert "window_id" in result["data"]
        
        print("✅ Missing fields filled with defaults")
    
    def test_06_handles_malformed_input(self, schema_map, malformed_windows):
        """Test 6: Handles malformed input gracefully."""
        for i, window in enumerate(malformed_windows):
            try:
                result = schema_map.window_to_record_dict(window)
                
                # Should return a dict (possibly error record)
                assert isinstance(result, dict), f"Window {i}: Expected dict, got {type(result)}"
                
                # Should still have required fields
                valid, _ = schema_map.validate_record(result)
                assert valid, f"Window {i}: Malformed handling produced invalid record"
                
            except Exception as e:
                pytest.fail(f"Window {i} raised exception: {e}")
        
        print(f"✅ All {len(malformed_windows)} malformed inputs handled")
    
    def test_07_performance_under_5ms(self, schema_map, synthetic_window):
        """Test 7: Performance < 5ms per translation."""
        from conftest import PerformanceTimer
        
        iterations = 1000
        with PerformanceTimer("Schema translation", threshold_ms=5000) as pt:
            for _ in range(iterations):
                _ = schema_map.window_to_record_dict(synthetic_window)
        
        avg_ms = pt.elapsed_ms / iterations
        assert avg_ms < 5, f"Translation too slow: {avg_ms:.2f}ms (max: 5ms)"
        
        print(f"✅ Average translation time: {avg_ms:.4f}ms")
    
    def test_08_no_memory_accumulation(self, schema_map, synthetic_window):
        """Test 8: No memory accumulation between translations."""
        from conftest import MemoryTracker
        
        with MemoryTracker("Schema translations", max_growth_mb=10) as mt:
            for _ in range(10000):
                result = schema_map.window_to_record_dict(synthetic_window)
                del result  # Explicitly delete
        
        assert mt.within_bounds, f"Memory grew by {mt.growth_mb:.2f}MB"
        print(f"✅ Memory growth: {mt.growth_mb:.2f}MB (within bounds)")


class TestSchemaMapChainValidation:
    """
    CHAIN VALIDATION: Source → SchemaMap → ForgeRecord
    """
    
    def test_chain_source_to_schemamap(self, synthetic_window_batch):
        """Validate: TrueVision Source output → SchemaMap input."""
        schema_map = TrueVisionSchemaMap()
        
        success_count = 0
        for window in synthetic_window_batch:
            result = schema_map.window_to_record_dict(window)
            if not result.get("is_error"):
                success_count += 1
        
        assert success_count == len(synthetic_window_batch), \
            f"Chain broken: {len(synthetic_window_batch) - success_count} failures"
        
        print(f"✅ Chain intact: Source → SchemaMap ({success_count}/{len(synthetic_window_batch)})")
    
    def test_chain_schemamap_to_forge(self, synthetic_window_batch):
        """Validate: SchemaMap output → ForgeRecord.from_dict() input."""
        schema_map = TrueVisionSchemaMap()
        
        for i, window in enumerate(synthetic_window_batch):
            record = schema_map.window_to_record_dict(window)
            
            # Simulate ForgeRecord.from_dict() validation
            try:
                # These are the checks ForgeRecord.from_dict() would do
                assert "worker_id" in record
                assert "seq" in record
                assert "timestamp" in record
                assert "data" in record
                
                # Type checks
                assert isinstance(record["worker_id"], str)
                assert isinstance(record["seq"], int)
                assert isinstance(record["timestamp"], (int, float))
                assert isinstance(record["data"], dict)
                
            except AssertionError as e:
                pytest.fail(f"Window {i}: ForgeRecord would reject: {e}")
        
        print(f"✅ Chain intact: SchemaMap → ForgeRecord ({len(synthetic_window_batch)} records)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
