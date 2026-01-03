"""
Pytest fixtures and test configuration for chain reaction smoke tests.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any, List, Generator
from datetime import datetime
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core"))
sys.path.insert(0, str(PROJECT_ROOT / "loggers"))
sys.path.insert(0, str(PROJECT_ROOT / "gaming"))


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SyntheticWindowData:
    """Synthetic TrueVision window data for testing."""
    timestamp: float = field(default_factory=time.time)
    window_id: str = "test_win_001"
    frame_count: int = 30
    features: Dict[str, float] = field(default_factory=lambda: {
        "motion_intensity": 0.75,
        "flicker_score": 0.1,
        "hud_detected": True,
        "crosshair_score": 0.85
    })
    detections: List[Dict] = field(default_factory=lambda: [
        {"type": "crosshair", "confidence": 0.92, "x": 960, "y": 540},
        {"type": "hud_element", "confidence": 0.88, "x": 50, "y": 50}
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "window_id": self.window_id,
            "t_start": self.timestamp - 1.0,
            "t_end": self.timestamp,
            "frame_count": self.frame_count,
            "features": self.features,
            "detections": self.detections,
            "metadata": {
                "source": "synthetic_test",
                "version": "1.0.0"
            }
        }


@dataclass
class SyntheticActivityData:
    """Synthetic activity logger data."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    window_title: str = "Call of Duty: Warzone"
    process_name: str = "cod.exe"
    executable_path: str = "C:\\Games\\COD\\cod.exe"
    idle_seconds: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "windowTitle": self.window_title,
            "processName": self.process_name,
            "executablePath": self.executable_path,
            "idleSeconds": self.idle_seconds
        }


@dataclass
class SyntheticInputData:
    """Synthetic input logger data."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    keystroke_count: int = 42
    mouse_click_count: int = 15
    mouse_movement_distance: float = 1250.5
    idle_seconds: float = 0.2
    audio_active: bool = True
    camera_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "keystroke_count": self.keystroke_count,
            "mouse_click_count": self.mouse_click_count,
            "mouse_movement_distance": self.mouse_movement_distance,
            "idle_seconds": self.idle_seconds,
            "audio_active": self.audio_active,
            "camera_active": self.camera_active,
            "audio_device_name": "Speakers (Realtek)",
            "camera_device_name": None
        }


@dataclass  
class SyntheticGamepadData:
    """Synthetic gamepad logger data."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    event: str = "axis_move"
    axis: int = 0
    value: float = 0.75
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event": self.event,
            "axis": self.axis,
            "value": self.value
        }


@dataclass
class SyntheticNetworkData:
    """Synthetic network logger data."""
    Timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    LocalAddress: str = "192.168.1.100"
    LocalPort: int = 56263
    RemoteAddress: str = "142.250.80.46"
    RemotePort: int = 443
    State: str = "Established"
    Protocol: str = "TCP"
    PID: int = 8184
    ProcessName: str = "cod.exe"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Timestamp": self.Timestamp,
            "LocalAddress": self.LocalAddress,
            "LocalPort": self.LocalPort,
            "RemoteAddress": self.RemoteAddress,
            "RemotePort": self.RemotePort,
            "State": self.State,
            "Protocol": self.Protocol,
            "PID": self.PID,
            "ProcessName": self.ProcessName
        }


@dataclass
class SyntheticProcessData:
    """Synthetic process logger data."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pid: int = 12345
    process_name: str = "cod.exe"
    command_line: str = "C:\\Games\\COD\\cod.exe --launch"
    parent_pid: int = 1000
    origin: str = "user_initiated"
    flagged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "pid": self.pid,
            "process_name": self.process_name,
            "command_line": self.command_line,
            "parent_pid": self.parent_pid,
            "origin": self.origin,
            "flagged": self.flagged
        }


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_windows(count: int = 100) -> List[Dict[str, Any]]:
    """Generate batch of synthetic TrueVision windows."""
    windows = []
    base_time = time.time()
    
    for i in range(count):
        window = SyntheticWindowData(
            timestamp=base_time + (i * 0.033),  # ~30fps
            window_id=f"test_win_{i:05d}",
            frame_count=30
        )
        windows.append(window.to_dict())
    
    return windows


def generate_synthetic_logs(logger_type: str, count: int = 100) -> List[Dict[str, Any]]:
    """Generate batch of synthetic logger data."""
    generators = {
        "activity": SyntheticActivityData,
        "input": SyntheticInputData,
        "gamepad": SyntheticGamepadData,
        "network": SyntheticNetworkData,
        "process": SyntheticProcessData
    }
    
    if logger_type not in generators:
        raise ValueError(f"Unknown logger type: {logger_type}")
    
    gen_class = generators[logger_type]
    return [gen_class().to_dict() for _ in range(count)]


# ═══════════════════════════════════════════════════════════════════════════════
# PYTEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    temp = Path(tempfile.mkdtemp(prefix="compucog_smoke_"))
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def temp_log_dir(temp_dir: Path) -> Path:
    """Create logs subdirectory."""
    logs = temp_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs


@pytest.fixture
def temp_wal_dir(temp_dir: Path) -> Path:
    """Create WAL subdirectory."""
    wal = temp_dir / "wal"
    wal.mkdir(parents=True, exist_ok=True)
    return wal


@pytest.fixture
def synthetic_window() -> Dict[str, Any]:
    """Single synthetic TrueVision window."""
    return SyntheticWindowData().to_dict()


@pytest.fixture
def synthetic_window_batch() -> List[Dict[str, Any]]:
    """Batch of 100 synthetic windows."""
    return generate_synthetic_windows(100)


@pytest.fixture
def malformed_windows() -> List[Dict[str, Any]]:
    """Collection of malformed/edge-case windows for testing."""
    return [
        {},  # Empty dict
        {"timestamp": None},  # Null timestamp
        {"timestamp": "invalid"},  # Invalid timestamp format
        {"timestamp": time.time(), "features": None},  # Null features
        {"timestamp": time.time(), "detections": "not_a_list"},  # Wrong type
        {"timestamp": time.time(), "extra_field": "unexpected"},  # Extra fields
        {"timestamp": time.time(), "frame_count": -1},  # Invalid value
        {"timestamp": time.time(), "frame_count": "thirty"},  # Wrong type
    ]


@pytest.fixture
def mock_truevision_source():
    """Mock TrueVision source for testing."""
    mock = MagicMock()
    mock.get_window.return_value = SyntheticWindowData().to_dict()
    mock.is_connected.return_value = True
    mock.reconnect.return_value = True
    return mock


@pytest.fixture
def mock_session_context(temp_dir: Path):
    """Mock SessionContext for testing."""
    mock = MagicMock()
    mock.session_id = "test_session_001"
    mock.session_epoch = time.time()
    mock.get_timestamp.return_value = {
        "session_id": "test_session_001",
        "session_epoch": time.time(),
        "event_epoch": time.time(),
        "event_offset_ms": 0.0,
        "timestamp_iso": datetime.now().isoformat()
    }
    mock.get_modality_dir.return_value = temp_dir / "modality"
    (temp_dir / "modality").mkdir(parents=True, exist_ok=True)
    return mock


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class ChainLink:
    """Represents a single link in the module chain."""
    
    def __init__(self, name: str, module: Any, input_schema: type = None, output_schema: type = None):
        self.name = name
        self.module = module
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.test_results: List[Dict] = []
        self.error_count = 0
        self.success_count = 0
    
    def validate_input(self, data: Any) -> bool:
        """Validate data matches expected input schema."""
        if self.input_schema is None:
            return True
        # Add schema validation logic here
        return True
    
    def validate_output(self, data: Any) -> bool:
        """Validate output matches expected schema."""
        if self.output_schema is None:
            return True
        # Add schema validation logic here
        return True
    
    def record_result(self, success: bool, details: str = ""):
        """Record test result."""
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.test_results.append({
            "success": success,
            "details": details,
            "timestamp": time.time()
        })


def validate_chain_link(source: ChainLink, target: ChainLink, data: Any) -> tuple[bool, str]:
    """
    Validate that source's output can be consumed by target's input.
    
    Returns:
        (success, message)
    """
    try:
        # Get source output
        if not source.validate_output(data):
            return False, f"{source.name} output validation failed"
        
        # Feed to target
        if not target.validate_input(data):
            return False, f"{target.name} cannot consume {source.name} output"
        
        return True, f"Chain intact: {source.name} → {target.name}"
    
    except Exception as e:
        return False, f"Chain break: {source.name} → {target.name}: {e}"


def log_chain_mismatch(source_name: str, target_name: str, output: Any, error: Exception):
    """Log detailed chain break information."""
    print(f"\n🔴 CHAIN BREAK: {source_name} → {target_name}")
    print(f"   Error: {error}")
    print(f"   Output type: {type(output)}")
    if isinstance(output, dict):
        print(f"   Output keys: {list(output.keys())}")
    print(f"   Output sample: {str(output)[:200]}...")


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self, label: str = "", threshold_ms: float = None):
        self.label = label
        self.threshold_ms = threshold_ms
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        
        if self.threshold_ms and self.elapsed_ms > self.threshold_ms:
            print(f"⚠️ {self.label}: {self.elapsed_ms:.2f}ms (threshold: {self.threshold_ms}ms)")
    
    @property
    def passed_threshold(self) -> bool:
        if self.threshold_ms is None:
            return True
        return self.elapsed_ms <= self.threshold_ms


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


class MemoryTracker:
    """Track memory usage over a test."""
    
    def __init__(self, label: str = "", max_growth_mb: float = None):
        self.label = label
        self.max_growth_mb = max_growth_mb
        self.start_mb = 0.0
        self.end_mb = 0.0
        self.growth_mb = 0.0
    
    def __enter__(self):
        self.start_mb = get_memory_usage_mb()
        return self
    
    def __exit__(self, *args):
        self.end_mb = get_memory_usage_mb()
        self.growth_mb = self.end_mb - self.start_mb
        
        if self.max_growth_mb and self.growth_mb > self.max_growth_mb:
            print(f"⚠️ Memory leak suspected: {self.label}: +{self.growth_mb:.2f}MB")
    
    @property
    def within_bounds(self) -> bool:
        if self.max_growth_mb is None:
            return True
        return self.growth_mb <= self.max_growth_mb
