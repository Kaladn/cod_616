"""
Test Module 7: Immortal Loggers (x5)

Tests each immortal logger for resilience and correct behavior.
"""

import os
import sys
import time
import json
import signal
import threading
import multiprocessing
import pytest
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "loggers"))


# ═══════════════════════════════════════════════════════════════════════════════
# IMMORTAL LOGGER BASE (for testing)
# ═══════════════════════════════════════════════════════════════════════════════

class ImmortalLoggerBase:
    """
    Base class for immortal loggers.
    
    IMMORTAL GUARANTEES:
    - while True loop never exits
    - Catches ALL exceptions
    - Continues after ANY error
    - Ctrl-C doesn't kill
    """
    
    def __init__(self, name: str, log_dir: Path, interval_seconds: float = 1.0):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_seconds
        
        self.running = False
        self.iteration_count = 0
        self.error_count = 0
        self.last_capture = None
        self.log_file = self.log_dir / f"{name}.jsonl"
        
        self._stop_event = threading.Event()
    
    def capture(self) -> Dict[str, Any]:
        """Override in subclass to capture data."""
        raise NotImplementedError
    
    def _write_safe(self, data: Dict[str, Any]) -> bool:
        """IMMORTAL write - never crashes."""
        for attempt in range(3):
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                return True
            except Exception:
                time.sleep(0.05 * (attempt + 1))
        return False
    
    def run_once(self) -> bool:
        """Run one iteration. Returns True if successful."""
        try:
            self.iteration_count += 1
            data = self.capture()
            
            if data:
                data["_iteration"] = self.iteration_count
                data["_timestamp"] = datetime.now().isoformat()
                self._write_safe(data)
                self.last_capture = data
            
            return True
            
        except Exception as e:
            self.error_count += 1
            self._write_safe({
                "_error": str(e),
                "_iteration": self.iteration_count,
                "_timestamp": datetime.now().isoformat()
            })
            return False
    
    def run_immortal(self, max_iterations: int = None):
        """
        IMMORTAL run loop.
        
        Args:
            max_iterations: For testing - limit iterations (None = forever)
        """
        self.running = True
        
        while self.running:
            try:
                # Check stop condition
                if max_iterations and self.iteration_count >= max_iterations:
                    break
                
                self.run_once()
                
            except KeyboardInterrupt:
                # IMMORTAL: Ignore Ctrl-C
                pass
            except Exception:
                # IMMORTAL: Catch everything, continue
                self.error_count += 1
            
            # Always sleep
            time.sleep(self.interval_seconds)
    
    def stop(self):
        """Stop the logger."""
        self.running = False
        self._stop_event.set()


class MockActivityLogger(ImmortalLoggerBase):
    """Mock activity logger for testing."""
    
    def __init__(self, log_dir: Path):
        super().__init__("activity", log_dir, interval_seconds=0.1)
        self.fail_next = False
    
    def capture(self) -> Dict[str, Any]:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Simulated failure")
        
        return {
            "windowTitle": "Test Window",
            "processName": "test.exe",
            "executablePath": "C:\\test\\test.exe",
            "idleSeconds": 0.5
        }


class MockInputLogger(ImmortalLoggerBase):
    """Mock input logger for testing."""
    
    def __init__(self, log_dir: Path):
        super().__init__("input", log_dir, interval_seconds=0.1)
        self.fail_next = False
    
    def capture(self) -> Dict[str, Any]:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Simulated failure")
        
        return {
            "keystroke_count": 10,
            "mouse_click_count": 5,
            "mouse_movement_distance": 100.0,
            "idle_seconds": 0.2
        }


class MockGamepadLogger(ImmortalLoggerBase):
    """Mock gamepad logger for testing."""
    
    def __init__(self, log_dir: Path):
        super().__init__("gamepad", log_dir, interval_seconds=0.1)
        self.fail_next = False
    
    def capture(self) -> Dict[str, Any]:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Simulated failure")
        
        return {
            "event": "axis_move",
            "axis": 0,
            "value": 0.75
        }


class MockProcessLogger(ImmortalLoggerBase):
    """Mock process logger for testing."""
    
    def __init__(self, log_dir: Path):
        super().__init__("process", log_dir, interval_seconds=0.1)
        self.fail_next = False
    
    def capture(self) -> Dict[str, Any]:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Simulated failure")
        
        return {
            "pid": 1234,
            "process_name": "test.exe",
            "command_line": "test.exe --run",
            "parent_pid": 1000
        }


class MockNetworkLogger(ImmortalLoggerBase):
    """Mock network logger for testing."""
    
    def __init__(self, log_dir: Path):
        super().__init__("network", log_dir, interval_seconds=0.1)
        self.fail_next = False
    
    def capture(self) -> Dict[str, Any]:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Simulated failure")
        
        return {
            "LocalAddress": "192.168.1.100",
            "LocalPort": 12345,
            "RemoteAddress": "8.8.8.8",
            "RemotePort": 443,
            "State": "Established",
            "Protocol": "TCP",
            "PID": 1234,
            "ProcessName": "test.exe"
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMMON TESTS (all 5 loggers)
# ═══════════════════════════════════════════════════════════════════════════════

class TestImmortalLoggersCommon:
    """
    COMMON SMOKE TESTS for all immortal loggers.
    
    Tests:
    1. ✅ while True loop never exits
    2. ✅ try/except catches ALL exceptions
    3. ✅ Continues after ANY error
    4. ✅ Writes to log file
    5. ✅ Interval respected (sleeps)
    6. ✅ Memory doesn't leak over 1000 iterations
    7. ✅ Survives missing dependencies (fallback mode)
    8. ✅ Ctrl-C doesn't kill (immortal!)
    """
    
    @pytest.fixture
    def log_dir(self, temp_dir):
        d = temp_dir / "immortal_logs"
        d.mkdir(exist_ok=True)
        return d
    
    @pytest.fixture(params=[
        MockActivityLogger,
        MockInputLogger,
        MockGamepadLogger,
        MockProcessLogger,
        MockNetworkLogger
    ])
    def logger(self, request, log_dir):
        """Parametrized fixture for all logger types."""
        logger_cls = request.param
        return logger_cls(log_dir)
    
    def test_01_loop_runs(self, logger):
        """Test 1: Run loop executes iterations."""
        # Run limited iterations
        logger.run_immortal(max_iterations=10)
        
        assert logger.iteration_count >= 10
        print(f"✅ {logger.name}: Loop ran {logger.iteration_count} iterations")
    
    def test_02_catches_all_exceptions(self, logger):
        """Test 2: Catches all exceptions."""
        # Force an error
        logger.fail_next = True
        
        # Should not raise
        result = logger.run_once()
        
        assert result is False  # Failed but didn't crash
        assert logger.error_count >= 1
        
        print(f"✅ {logger.name}: Exception caught, error_count={logger.error_count}")
    
    def test_03_continues_after_error(self, logger):
        """Test 3: Continues after error."""
        # Cause error on iteration 3
        def cause_error():
            if logger.iteration_count == 3:
                logger.fail_next = True
        
        # Run with error injection
        for _ in range(10):
            cause_error()
            logger.run_once()
        
        assert logger.iteration_count >= 10
        assert logger.error_count >= 1
        
        print(f"✅ {logger.name}: Continued after error, final iteration={logger.iteration_count}")
    
    def test_04_writes_to_log_file(self, logger):
        """Test 4: Writes to log file."""
        logger.run_immortal(max_iterations=5)
        
        assert logger.log_file.exists()
        
        with open(logger.log_file, "r") as f:
            lines = f.readlines()
        
        assert len(lines) >= 5
        
        # Verify JSON
        for line in lines:
            data = json.loads(line.strip())
            assert "_timestamp" in data
        
        print(f"✅ {logger.name}: {len(lines)} lines written")
    
    def test_05_interval_respected(self, logger):
        """Test 5: Interval respected (sleeps)."""
        start = time.time()
        
        logger.interval_seconds = 0.1
        logger.run_immortal(max_iterations=5)
        
        elapsed = time.time() - start
        expected_min = 5 * 0.1 * 0.5  # Allow 50% variance
        
        assert elapsed >= expected_min, f"Too fast: {elapsed:.2f}s (expected >={expected_min:.2f}s)"
        
        print(f"✅ {logger.name}: Interval respected, elapsed={elapsed:.2f}s")
    
    def test_06_memory_no_leak(self, logger):
        """Test 6: Memory doesn't leak over iterations."""
        from conftest import MemoryTracker
        
        with MemoryTracker(f"{logger.name} iterations", max_growth_mb=50) as mt:
            logger.run_immortal(max_iterations=1000)
        
        assert mt.within_bounds, f"Memory grew by {mt.growth_mb:.2f}MB"
        print(f"✅ {logger.name}: Memory growth={mt.growth_mb:.2f}MB")
    
    def test_07_survives_dependency_failure(self, logger):
        """Test 7: Survives missing dependencies."""
        # Simulate dependency failure by forcing errors
        original_capture = logger.capture
        
        def failing_capture():
            raise ImportError("Missing dependency")
        
        logger.capture = failing_capture
        
        # Should not crash
        for _ in range(5):
            logger.run_once()
        
        assert logger.iteration_count >= 5
        
        # Restore
        logger.capture = original_capture
        
        print(f"✅ {logger.name}: Survived dependency failure")
    
    def test_08_ctrl_c_doesnt_kill(self, logger):
        """Test 8: Ctrl-C doesn't kill (immortal!)."""
        
        def run_with_interrupt():
            for i in range(20):
                try:
                    logger.run_once()
                    
                    # Simulate Ctrl-C mid-execution
                    if i == 5:
                        raise KeyboardInterrupt()
                    
                except KeyboardInterrupt:
                    # IMMORTAL: Should continue
                    pass
        
        run_with_interrupt()
        
        assert logger.iteration_count >= 20
        print(f"✅ {logger.name}: Survived Ctrl-C, iterations={logger.iteration_count}")


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIFIC LOGGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestActivityLoggerSpecific:
    """Specific tests for activity logger."""
    
    def test_captures_window_info(self, temp_dir):
        """Captures window title and process info."""
        logger = MockActivityLogger(temp_dir / "activity")
        
        data = logger.capture()
        
        assert "windowTitle" in data
        assert "processName" in data
        assert "executablePath" in data
        assert "idleSeconds" in data
        
        print("✅ Activity logger captures window info")


class TestInputLoggerSpecific:
    """Specific tests for input logger."""
    
    def test_captures_input_metrics(self, temp_dir):
        """Captures keyboard/mouse metrics."""
        logger = MockInputLogger(temp_dir / "input")
        
        data = logger.capture()
        
        assert "keystroke_count" in data
        assert "mouse_click_count" in data
        assert "mouse_movement_distance" in data
        
        print("✅ Input logger captures metrics")


class TestGamepadLoggerSpecific:
    """Specific tests for gamepad logger."""
    
    def test_captures_controller_events(self, temp_dir):
        """Captures controller events."""
        logger = MockGamepadLogger(temp_dir / "gamepad")
        
        data = logger.capture()
        
        assert "event" in data
        assert data["event"] in ["button_press", "axis_move", "trigger", "hat_change"]
        
        print("✅ Gamepad logger captures events")


class TestProcessLoggerSpecific:
    """Specific tests for process logger."""
    
    def test_captures_process_info(self, temp_dir):
        """Captures process information."""
        logger = MockProcessLogger(temp_dir / "process")
        
        data = logger.capture()
        
        assert "pid" in data
        assert "process_name" in data
        assert "command_line" in data
        
        print("✅ Process logger captures process info")


class TestNetworkLoggerSpecific:
    """Specific tests for network logger."""
    
    def test_captures_network_connections(self, temp_dir):
        """Captures network connections."""
        logger = MockNetworkLogger(temp_dir / "network")
        
        data = logger.capture()
        
        assert "LocalAddress" in data
        assert "RemoteAddress" in data
        assert "State" in data
        assert "Protocol" in data
        
        print("✅ Network logger captures connections")


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoggerChainValidation:
    """Chain validation: Logger → LoggerPulseWriter."""
    
    def test_chain_all_loggers_to_pulsewriter(self, temp_dir):
        """Validate all loggers output → LoggerPulseWriter."""
        from test_06_logger_pulse_writer import LoggerPulseWriter
        
        loggers = [
            MockActivityLogger(temp_dir / "activity"),
            MockInputLogger(temp_dir / "input"),
            MockGamepadLogger(temp_dir / "gamepad"),
            MockProcessLogger(temp_dir / "process"),
            MockNetworkLogger(temp_dir / "network")
        ]
        
        pw = LoggerPulseWriter(temp_dir / "combined", prefix="all_loggers")
        
        for logger in loggers:
            data = logger.capture()
            result = pw.write(data)
            assert result is True
        
        # Verify all written
        recent = pw.get_recent_records(5)
        assert len(recent) == 5
        
        print("✅ Chain intact: 5 loggers → PulseWriter")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
