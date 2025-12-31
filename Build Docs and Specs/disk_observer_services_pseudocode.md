# Disk-Observer Services Pseudocode Specification

**Target System:** Windows 11 Pro  
**Constraint:** ALL CURRENT FILES ARE READ-ONLY. NO MODIFICATIONS TO EXISTING CODE.  
**Implementation:** Replace subprocess-based services with disk-observer services.

---

## Files to Replace (Complete Rewrites)

- `loggers/activity_service.py`
- `loggers/input_service.py`
- `loggers/network_service.py`

---

## Alignment Requirements

### Preserve from Existing Code

1. **Class names:** `ActivityService`, `InputService`, `NetworkService`
2. **Public API methods:** `start()`, `stop()`, `is_running()`
3. **Constructor signature:** `__init__(self, config_path: Optional[str] = None, use_subprocess: bool = True)`
   - **IGNORE `use_subprocess` parameter** (kept for API compatibility only, always False internally)
4. **Import style:** Standard library imports, pathlib for paths, logging for output
5. **Logging format:** `[ServiceName] message`

### Remove Entirely

1. All subprocess/Popen usage
2. All stub mode logic
3. All `use_subprocess` branching
4. All process management (terminate, kill, wait)

### Add New

1. File watching logic (stat-based, no external dependencies)
2. Event emission (dict-based events)
3. Per-file state tracking (offsets, mtimes, sizes)
4. Idle/stall detection (N pulses with no change)

---

## Configuration Constants

```python
# Add to each service file
IDLE_THRESHOLD_PULSES = 50  # No change for 50 pulses = idle
STALL_THRESHOLD_PULSES = 100  # No change for 100 pulses = stalled
PULSE_INTERVAL_SECONDS = 0.1  # Assumed pulse rate for timeout calculations
```

---

## Common Disk-Observer Pattern

All three services follow this pattern:

### State Tracking

```python
class ServiceState:
    file_path: Path
    last_size: int
    last_mtime: float
    read_offset: int  # For input_service only
    pulses_unchanged: int
    last_event_type: str
```

### Core Loop (Called on Each Pulse)

```python
def poll(self) -> Optional[dict]:
    """
    Called once per pulse (~100ms).
    Returns event dict or None.
    """
    if not self._started:
        return None
    
    # Check file existence
    if not self.file_path.exists():
        return self._emit_failed("file_missing")
    
    # Get current file stats
    stat = self.file_path.stat()
    current_size = stat.st_size
    current_mtime = stat.st_mtime
    
    # Detect change
    if current_size != self.last_size or current_mtime != self.last_mtime:
        self.pulses_unchanged = 0
        self.last_size = current_size
        self.last_mtime = current_mtime
        return self._emit_active()
    else:
        self.pulses_unchanged += 1
        
        # Check thresholds
        if self.pulses_unchanged >= STALL_THRESHOLD_PULSES:
            return self._emit_stalled()
        elif self.pulses_unchanged >= IDLE_THRESHOLD_PULSES:
            return self._emit_idle()
    
    return None
```

---

## Service 1: activity_service.py

### Purpose
Monitor activity logger output files to detect user activity.

### File to Watch
```python
# Pattern: logs/activity/user_activity_YYYYMMDD.jsonl
# Example: d:/cod_616/logs/activity/user_activity_20251231.jsonl

def get_activity_log_path() -> Path:
    """
    Compute today's activity log path.
    Matches pattern used by activity_logger.py line 94-102.
    """
    script_dir = Path(__file__).parent
    logs_dir = script_dir / ".." / "logs" / "activity"
    logs_dir = logs_dir.resolve()
    
    today = datetime.now().strftime("%Y%m%d")
    return logs_dir / f"user_activity_{today}.jsonl"
```

### Event Schema

```python
# Event type: "activity_detected"
{
    "event_type": "activity_detected",
    "timestamp": float,  # time.time()
    "file_path": str,
    "file_size": int,
    "mtime": float,
}

# Event type: "activity_idle"
{
    "event_type": "activity_idle",
    "timestamp": float,
    "pulses_unchanged": int,
    "file_path": str,
}

# Event type: "activity_stalled"
{
    "event_type": "activity_stalled",
    "timestamp": float,
    "pulses_unchanged": int,
    "file_path": str,
}

# Event type: "activity_failed"
{
    "event_type": "activity_failed",
    "timestamp": float,
    "reason": str,  # "file_missing" | "dir_missing"
    "file_path": str,
}
```

### Implementation Requirements

```python
class ActivityService:
    def __init__(self, config_path: Optional[str] = None, use_subprocess: bool = True):
        # Ignore use_subprocess (kept for API compatibility)
        self.config_path = config_path
        self._started = False
        self._file_path: Optional[Path] = None
        self._last_size: int = 0
        self._last_mtime: float = 0.0
        self._pulses_unchanged: int = 0
        self._last_event_type: Optional[str] = None
    
    def start(self) -> None:
        """
        Initialize file watching.
        Does NOT start subprocess.
        """
        if self._started:
            return
        
        self._file_path = get_activity_log_path()
        
        # Check if parent directory exists
        if not self._file_path.parent.exists():
            logging.warning(f"[ActivityService] Directory not found: {self._file_path.parent}")
            # Still mark as started; poll() will emit failure events
        
        # Initialize state if file exists
        if self._file_path.exists():
            stat = self._file_path.stat()
            self._last_size = stat.st_size
            self._last_mtime = stat.st_mtime
        
        self._started = True
        logging.info(f"[ActivityService] Watching: {self._file_path}")
    
    def stop(self) -> None:
        """
        Stop watching.
        """
        if not self._started:
            return
        
        logging.info(f"[ActivityService] Stopped")
        self._started = False
    
    def is_running(self) -> bool:
        return self._started
    
    def poll(self) -> Optional[dict]:
        """
        Call once per pulse.
        Returns event dict or None.
        """
        # Implementation follows Common Disk-Observer Pattern above
        # Emit events based on file changes
```

### Day Rollover Handling

```python
def poll(self) -> Optional[dict]:
    # ... existing logic ...
    
    # Check for day rollover
    current_expected_path = get_activity_log_path()
    if current_expected_path != self._file_path:
        logging.info(f"[ActivityService] Day rollover: {self._file_path} -> {current_expected_path}")
        self._file_path = current_expected_path
        self._last_size = 0
        self._last_mtime = 0.0
        self._pulses_unchanged = 0
        
        # Re-initialize if new file exists
        if self._file_path.exists():
            stat = self._file_path.stat()
            self._last_size = stat.st_size
            self._last_mtime = stat.st_mtime
```

---

## Service 2: input_service.py

### Purpose
Read input log files and emit parsed events.

### File to Watch
```python
# Configurable via config_path or default
# Default: logs/input/input_events_YYYYMMDD.jsonl

def get_input_log_path(config_path: Optional[str]) -> Path:
    """
    Get input log path from config or default.
    """
    if config_path:
        return Path(config_path)
    
    script_dir = Path(__file__).parent
    logs_dir = script_dir / ".." / "logs" / "input"
    logs_dir = logs_dir.resolve()
    
    today = datetime.now().strftime("%Y%m%d")
    return logs_dir / f"input_events_{today}.jsonl"
```

### Event Schema

```python
# Event type: "input_event"
# Contains parsed JSONL record from input log
{
    "event_type": "input_event",
    "timestamp": float,  # time.time() when read
    "record": dict,  # Parsed JSON record from file
}

# Event type: "input_idle"
{
    "event_type": "input_idle",
    "timestamp": float,
    "pulses_unchanged": int,
    "file_path": str,
}

# Event type: "input_failed"
{
    "event_type": "input_failed",
    "timestamp": float,
    "reason": str,  # "file_missing" | "parse_error" | "read_error"
    "file_path": str,
    "details": Optional[str],
}
```

### Implementation Requirements

```python
class InputService:
    def __init__(self, config_path: Optional[str] = None, use_subprocess: bool = True):
        # Ignore use_subprocess
        self.config_path = config_path
        self._started = False
        self._file_path: Optional[Path] = None
        self._read_offset: int = 0  # Byte offset in file
        self._last_mtime: float = 0.0
        self._pulses_unchanged: int = 0
    
    def start(self) -> None:
        """
        Initialize file reading.
        """
        if self._started:
            return
        
        self._file_path = get_input_log_path(self.config_path)
        
        # Initialize offset to end of file if exists
        if self._file_path.exists():
            stat = self._file_path.stat()
            self._read_offset = stat.st_size  # Start reading from end
            self._last_mtime = stat.st_mtime
        
        self._started = True
        logging.info(f"[InputService] Watching: {self._file_path}")
    
    def stop(self) -> None:
        if not self._started:
            return
        
        logging.info(f"[InputService] Stopped")
        self._started = False
    
    def is_running(self) -> bool:
        return self._started
    
    def poll(self) -> Optional[list[dict]]:
        """
        Call once per pulse.
        Returns list of events (may be empty) or None if not started.
        """
        if not self._started:
            return None
        
        # Check file existence
        if not self._file_path.exists():
            self._pulses_unchanged += 1
            if self._pulses_unchanged >= IDLE_THRESHOLD_PULSES:
                return [self._emit_failed("file_missing")]
            return []
        
        # Get current file stats
        stat = self._file_path.stat()
        current_size = stat.st_size
        current_mtime = stat.st_mtime
        
        # Check if file has new data
        if current_size <= self._read_offset:
            # No new data
            self._pulses_unchanged += 1
            
            if self._pulses_unchanged >= IDLE_THRESHOLD_PULSES:
                return [self._emit_idle()]
            return []
        
        # Read new data
        events = []
        try:
            with open(self._file_path, 'r', encoding='utf-8') as f:
                f.seek(self._read_offset)
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        events.append({
                            "event_type": "input_event",
                            "timestamp": time.time(),
                            "record": record,
                        })
                    except json.JSONDecodeError as e:
                        events.append(self._emit_failed("parse_error", str(e)))
                
                # Update offset to current position
                self._read_offset = f.tell()
        
        except Exception as e:
            return [self._emit_failed("read_error", str(e))]
        
        # Reset idle counter if we read data
        if events:
            self._pulses_unchanged = 0
            self._last_mtime = current_mtime
        
        return events if events else []
```

### Day Rollover Handling

```python
def poll(self) -> Optional[list[dict]]:
    # ... existing logic ...
    
    # Check for day rollover (if using default path)
    if not self.config_path:
        current_expected_path = get_input_log_path(None)
        if current_expected_path != self._file_path:
            logging.info(f"[InputService] Day rollover: {self._file_path} -> {current_expected_path}")
            self._file_path = current_expected_path
            self._read_offset = 0
            self._last_mtime = 0.0
            self._pulses_unchanged = 0
```

---

## Service 3: network_service.py

### Purpose
Monitor network capture output files.

### File to Watch
```python
# Pattern: logs/network/network_capture_YYYYMMDD.jsonl
# Or output from network_logger.ps1

def get_network_log_path(config_path: Optional[str]) -> Path:
    """
    Get network log path from config or default.
    """
    if config_path:
        return Path(config_path)
    
    script_dir = Path(__file__).parent
    logs_dir = script_dir / ".." / "logs" / "network"
    logs_dir = logs_dir.resolve()
    
    today = datetime.now().strftime("%Y%m%d")
    return logs_dir / f"network_capture_{today}.jsonl"
```

### Event Schema

```python
# Event type: "network_active"
{
    "event_type": "network_active",
    "timestamp": float,
    "file_path": str,
    "file_size": int,
    "mtime": float,
}

# Event type: "network_stalled"
{
    "event_type": "network_stalled",
    "timestamp": float,
    "pulses_unchanged": int,
    "file_path": str,
}

# Event type: "network_failed"
{
    "event_type": "network_failed",
    "timestamp": float,
    "reason": str,
    "file_path": str,
}
```

### Implementation Requirements

```python
class NetworkService:
    def __init__(self, config_path: Optional[str] = None, use_subprocess: bool = True):
        # Ignore use_subprocess
        self.config_path = config_path
        self._started = False
        self._file_path: Optional[Path] = None
        self._last_size: int = 0
        self._last_mtime: float = 0.0
        self._pulses_unchanged: int = 0
    
    def start(self) -> None:
        """
        Initialize file watching.
        """
        if self._started:
            return
        
        self._file_path = get_network_log_path(self.config_path)
        
        if self._file_path.exists():
            stat = self._file_path.stat()
            self._last_size = stat.st_size
            self._last_mtime = stat.st_mtime
        
        self._started = True
        logging.info(f"[NetworkService] Watching: {self._file_path}")
    
    def stop(self) -> None:
        if not self._started:
            return
        
        logging.info(f"[NetworkService] Stopped")
        self._started = False
    
    def is_running(self) -> bool:
        return self._started
    
    def poll(self) -> Optional[dict]:
        """
        Call once per pulse.
        Returns event dict or None.
        """
        # Implementation follows Common Disk-Observer Pattern
        # Similar to ActivityService but with network-specific event names
```

---

## Testing Requirements

### Unit Tests (Disk-Backed, No Mocks)

```python
# tests/unit/test_activity_service_disk.py

def test_activity_service_detects_file_growth():
    """
    Test that ActivityService emits activity_detected when file grows.
    """
    # 1. Create real temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs" / "activity"
        log_dir.mkdir(parents=True)
        
        # 2. Create initial log file
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"user_activity_{today}.jsonl"
        log_file.write_text('{"test": "initial"}\n')
        
        # 3. Patch get_activity_log_path to return test path
        with patch('loggers.activity_service.get_activity_log_path', return_value=log_file):
            service = ActivityService()
            service.start()
            
            # 4. First poll - should initialize
            event = service.poll()
            assert event is None  # No change yet
            
            # 5. Append to file
            with open(log_file, 'a') as f:
                f.write('{"test": "new_data"}\n')
            
            # 6. Poll again - should detect change
            event = service.poll()
            assert event is not None
            assert event["event_type"] == "activity_detected"
            assert event["file_size"] > 0


def test_activity_service_emits_idle_after_threshold():
    """
    Test that ActivityService emits activity_idle after IDLE_THRESHOLD_PULSES.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs" / "activity"
        log_dir.mkdir(parents=True)
        
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"user_activity_{today}.jsonl"
        log_file.write_text('{"test": "data"}\n')
        
        with patch('loggers.activity_service.get_activity_log_path', return_value=log_file):
            service = ActivityService()
            service.start()
            
            # Poll IDLE_THRESHOLD_PULSES times without file change
            for i in range(IDLE_THRESHOLD_PULSES):
                event = service.poll()
                if i < IDLE_THRESHOLD_PULSES - 1:
                    assert event is None
            
            # Next poll should emit idle
            event = service.poll()
            assert event is not None
            assert event["event_type"] == "activity_idle"
            assert event["pulses_unchanged"] >= IDLE_THRESHOLD_PULSES


def test_activity_service_emits_failed_when_file_missing():
    """
    Test that ActivityService emits activity_failed when file doesn't exist.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs" / "activity"
        log_dir.mkdir(parents=True)
        
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"user_activity_{today}.jsonl"
        # Do NOT create file
        
        with patch('loggers.activity_service.get_activity_log_path', return_value=log_file):
            service = ActivityService()
            service.start()
            
            # Poll should emit failure after threshold
            for i in range(IDLE_THRESHOLD_PULSES):
                service.poll()
            
            event = service.poll()
            assert event is not None
            assert event["event_type"] == "activity_failed"
            assert event["reason"] == "file_missing"
```

### Integration Test (2-Minute Real Scenario)

```python
# tests/integration/test_services_integration.py

def test_activity_service_real_scenario():
    """
    Integration test: Watch real activity log for 2 minutes.
    """
    service = ActivityService()
    service.start()
    
    events_collected = []
    start_time = time.time()
    
    # Poll for 2 minutes
    while time.time() - start_time < 120:
        event = service.poll()
        if event:
            events_collected.append(event)
        time.sleep(0.1)  # Simulate pulse interval
    
    service.stop()
    
    # Assertions
    assert len(events_collected) > 0, "Should collect at least one event"
    
    # Check event types are valid
    valid_types = ["activity_detected", "activity_idle", "activity_stalled", "activity_failed"]
    for event in events_collected:
        assert event["event_type"] in valid_types
```

---

## CI Guard Implementation

```bash
# Add to CI pipeline or pre-commit hook

# File: scripts/check_no_runtime_stubs.sh

#!/bin/bash
set -e

echo "Checking for forbidden patterns in runtime code..."

# Search in loggers/ excluding tests
FORBIDDEN=$(grep -rn --include="*.py" \
    -e "stub" \
    -e "fake" \
    -e "mock" \
    -e "dry_run" \
    -e "test_mode" \
    loggers/ | grep -v "test_" | grep -v "__pycache__" || true)

if [ -n "$FORBIDDEN" ]; then
    echo "ERROR: Found forbidden patterns in runtime code:"
    echo "$FORBIDDEN"
    exit 1
fi

echo "✓ No forbidden patterns found"
```

---

## Implementation Checklist for Copilot

### activity_service.py
- [ ] Remove all subprocess imports and usage
- [ ] Remove all stub mode logic
- [ ] Implement `get_activity_log_path()` matching activity_logger.py pattern
- [ ] Implement file stat-based change detection
- [ ] Implement pulse-based idle/stall thresholds
- [ ] Implement day rollover detection
- [ ] Add event emission methods
- [ ] Preserve public API: `start()`, `stop()`, `is_running()`
- [ ] Add `poll()` method returning Optional[dict]

### input_service.py
- [ ] Remove all subprocess imports and usage
- [ ] Remove all stub mode logic
- [ ] Implement `get_input_log_path()` with config support
- [ ] Implement offset-based file reading
- [ ] Implement JSONL parsing with error handling
- [ ] Implement pulse-based idle detection
- [ ] Implement day rollover detection
- [ ] Add event emission methods
- [ ] Preserve public API: `start()`, `stop()`, `is_running()`
- [ ] Add `poll()` method returning Optional[list[dict]]

### network_service.py
- [ ] Remove all subprocess imports and usage
- [ ] Remove all stub mode logic
- [ ] Implement `get_network_log_path()` with config support
- [ ] Implement file stat-based change detection
- [ ] Implement pulse-based stall detection
- [ ] Implement day rollover detection
- [ ] Add event emission methods
- [ ] Preserve public API: `start()`, `stop()`, `is_running()`
- [ ] Add `poll()` method returning Optional[dict]

### Tests
- [ ] Create `tests/unit/test_activity_service_disk.py`
- [ ] Create `tests/unit/test_input_service_disk.py`
- [ ] Create `tests/unit/test_network_service_disk.py`
- [ ] Create `tests/integration/test_services_integration.py`
- [ ] All tests use real temp directories and files
- [ ] No mocks, no fakes, no dependency injection

### CI
- [ ] Add `scripts/check_no_runtime_stubs.sh`
- [ ] Integrate into CI pipeline

---

## Explicit Directives to Copilot

1. **DO NOT modify any existing files** except `activity_service.py`, `input_service.py`, `network_service.py`
2. **DO NOT add subprocess, Popen, or process management code**
3. **DO NOT add stub modes, fake paths, or test-only branches in runtime code**
4. **DO NOT add dependency injection, runner factories, or abstractions**
5. **DO use pathlib.Path for all file operations**
6. **DO use os.stat() or Path.stat() for file metadata**
7. **DO use standard library only (no external dependencies)**
8. **DO preserve existing class names and public API methods**
9. **DO add poll() method to each service**
10. **DO implement day rollover detection for date-based log files**
11. **DO write disk-backed tests with real temp directories**
12. **DO implement CI guard script to prevent runtime stubs**

---

## End of Pseudocode Specification

**Status:** Complete. Ready for Copilot implementation.

**Target:** Windows 11 Pro  
**Files to replace:** 3 service files  
**Tests to create:** 4 test files  
**CI guard:** 1 script  
**No modifications to:** All other existing files (READ-ONLY)
