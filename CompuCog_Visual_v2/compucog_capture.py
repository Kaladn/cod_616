"""
CompuCog Unified Capture System

Single orchestrator that starts ALL modalities with synchronized timestamps:
- TrueVision (visual analysis)
- Activity Logger (window/process tracking)
- Gamepad Logger (controller inputs)
- Input Logger (keyboard/mouse metrics)
- Network Logger (TCP/UDP connections)
- Process Logger (system process monitoring)

All modules share a SessionContext with microsecond-precision epoch for perfect
temporal alignment. This enables 6-1-6 fusion block building with exact causality.

Usage:
    python compucog_capture.py --session-name "warzone_session" --duration 1800
    python compucog_capture.py --session-name "ranked_grind" 
    python compucog_capture.py --stop
"""

import os
import sys
import time
import json
import signal
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "core"))
sys.path.insert(0, str(SCRIPT_DIR / "gaming"))
sys.path.insert(0, str(SCRIPT_DIR / "loggers"))

from core.session_context import SessionContext


class TrueVisionRunner:
    """
    Runs TrueVision capture in a separate thread with synchronized timestamps.
    """
    
    def __init__(self, ctx: SessionContext, event_threshold: float = 0.5):
        self.ctx = ctx
        self.event_threshold = event_threshold
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.harness = None
        self.error: Optional[Exception] = None
        
    def start(self):
        """Start TrueVision in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def _run(self):
        """Internal run loop."""
        try:
            # Import here to avoid circular imports
            from gaming.truevision_event_live import CognitiveHarness
            
            # Create output directory for TrueVision
            tv_dir = self.ctx.get_modality_dir("truevision")
            
            # Initialize harness with session context
            self.harness = CognitiveHarness(
                data_dir=str(tv_dir),
                enable_events=True,
                event_threshold=self.event_threshold,
                session_id=self.ctx.session_id
            )
            
            # Patch the harness to use synchronized timestamps
            self._patch_timestamps()
            
            # Run until stopped
            while self.running:
                try:
                    # Capture one frame cycle
                    frame = self.harness.frame_capture.capture()
                    
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    
                    # Get synchronized timestamp
                    ts = self.ctx.get_timestamp()
                    frame_timestamp = ts["event_epoch"]
                    
                    # Convert to grid
                    grid_result = self.harness.frame_to_grid.convert(
                        frame=frame,
                        frame_id=len(self.harness.frame_buffer),
                        t_sec=frame_timestamp,
                        source="truevision_sync"
                    )
                    
                    # Add to buffer
                    self.harness.frame_buffer.append(grid_result)
                    if len(self.harness.frame_buffer) > self.harness.max_buffer_size:
                        self.harness.frame_buffer.pop(0)
                    
                    # Run operators (need at least 3 frames)
                    if len(self.harness.frame_buffer) >= 3:
                        window = self.harness._build_detection_window(self.harness.windows_captured)
                        
                        if window:
                            # Inject synchronized timestamp fields
                            window["session_id"] = ts["session_id"]
                            window["session_epoch"] = ts["session_epoch"]
                            window["event_epoch"] = ts["event_epoch"]
                            window["event_offset_ms"] = ts["event_offset_ms"]
                            window["timestamp_iso"] = ts["timestamp_iso"]
                            
                            # Write to JSONL with sync timestamps
                            self._write_sync_window(window, ts)
                            
                            # Process window (skip Forge if shutting down)
                            if self.running:
                                self.harness.process_window(window)
                    
                    time.sleep(0.033)  # ~30 FPS
                    
                except Exception as e:
                    print(f"[TrueVision] Frame error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            self.error = e
            print(f"[TrueVision] Fatal error: {e}")
            import traceback
            traceback.print_exc()
    
    def _patch_timestamps(self):
        """Patch harness to use session epoch as reference."""
        # Override chronos to use session epoch
        if hasattr(self.harness, 'chronos'):
            original_now = self.harness.chronos.now
            session_epoch = self.ctx.session_epoch
            def synced_now():
                return time.time()
            self.harness.chronos.now = synced_now
    
    def _write_sync_window(self, window: Dict, ts: Dict):
        """Write window to synchronized JSONL file."""
        log_path = self.ctx.get_log_path("truevision", "truevision_live")
        
        # Build output record with sync fields at top level
        record = {
            "session_id": ts["session_id"],
            "session_epoch": ts["session_epoch"],
            "event_epoch": ts["event_epoch"],
            "event_offset_ms": ts["event_offset_ms"],
            "timestamp_iso": ts["timestamp_iso"],
            "window_start_epoch": window.get("ts_start"),
            "window_end_epoch": window.get("ts_end"),
            "eomm_composite_score": window.get("eomm_score", 0.0),
            "eomm_flags": window.get("operator_flags", []),
            "operator_results": [
                {"name": k, "confidence": v} 
                for k, v in window.get("operator_scores", {}).items()
            ],
            "grid_shape": window.get("grid_shape"),
            "window_id": window.get("session_context", {}).get("window_id", 0),
        }
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    
    def stop(self):
        """Stop TrueVision capture."""
        self.running = False
        # Give the loop time to exit cleanly before shutdown
        time.sleep(0.2)
        if self.harness:
            try:
                self.harness._shutdown()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=5)
    
    def get_stats(self) -> Dict:
        """Get capture statistics."""
        if self.harness:
            return {
                "windows_captured": self.harness.windows_captured,
                "events_recorded": self.harness.events_recorded,
                "high_eomm_count": self.harness.high_eomm_count,
            }
        return {}


class SyncLogger:
    """
    Wrapper for Python loggers that injects synchronized timestamps.
    """
    
    def __init__(self, name: str, ctx: SessionContext, sample_interval: float = 3.0):
        self.name = name
        self.ctx = ctx
        self.sample_interval = sample_interval
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.event_count = 0
        self.error: Optional[Exception] = None
        
    def start(self):
        """Start logger in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def _run(self):
        """Internal run loop - override in subclasses."""
        raise NotImplementedError("Subclasses must implement _run")
    
    def _write_event(self, data: Dict):
        """Write event with synchronized timestamps."""
        ts = self.ctx.get_timestamp()
        
        # Merge sync fields with event data
        record = {
            "session_id": ts["session_id"],
            "session_epoch": ts["session_epoch"],
            "event_epoch": ts["event_epoch"],
            "event_offset_ms": ts["event_offset_ms"],
            "timestamp": ts["timestamp_iso"],  # Keep 'timestamp' for backwards compat
            **data  # Original event data (may override timestamp)
        }
        
        log_path = self.ctx.get_log_path(self.name)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        
        self.event_count += 1
    
    def stop(self):
        """Stop logger."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def get_stats(self) -> Dict:
        """Get logger statistics."""
        return {"event_count": self.event_count}


class ActivityLogger(SyncLogger):
    """Synchronized activity logger (window/process tracking)."""
    
    def __init__(self, ctx: SessionContext):
        super().__init__("activity", ctx, sample_interval=3.0)
        
    def _run(self):
        try:
            import ctypes
            from ctypes import wintypes
            import psutil
            
            # Windows API for idle detection
            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [('cbSize', wintypes.UINT), ('dwTime', wintypes.DWORD)]
            
            def get_idle_duration():
                lii = LASTINPUTINFO()
                lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
                ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
                millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
                return millis / 1000.0
            
            def get_active_window():
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                if not hwnd:
                    return None, None, None
                pid = wintypes.DWORD()
                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                return hwnd, pid.value, buf.value
            
            def get_process_info(pid):
                try:
                    proc = psutil.Process(pid)
                    return proc.name(), proc.exe()
                except:
                    return None, None
            
            while self.running:
                time.sleep(self.sample_interval)
                
                try:
                    idle_seconds = get_idle_duration()
                    hwnd, pid, window_title = get_active_window()
                    process_name, process_path = get_process_info(pid) if pid else (None, None)
                    
                    self._write_event({
                        "windowTitle": window_title or "",
                        "processName": process_name or "Unknown",
                        "executablePath": process_path or "",
                        "idleSeconds": round(idle_seconds, 1)
                    })
                except Exception as e:
                    print(f"[Activity] Error: {e}")
                    
        except Exception as e:
            self.error = e
            print(f"[Activity] Fatal: {e}")


class InputLogger(SyncLogger):
    """Synchronized input logger (keyboard/mouse metrics)."""
    
    def __init__(self, ctx: SessionContext):
        super().__init__("input", ctx, sample_interval=3.0)
        
    def _run(self):
        try:
            import ctypes
            from ctypes import wintypes
            
            # Track input activity via polling
            def get_key_state():
                keys_pressed = 0
                for vk in range(256):
                    if ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000:
                        keys_pressed += 1
                return keys_pressed
            
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            
            last_pos = POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(last_pos))
            
            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [('cbSize', wintypes.UINT), ('dwTime', wintypes.DWORD)]
            
            def get_idle_duration():
                lii = LASTINPUTINFO()
                lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
                ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
                millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
                return millis / 1000.0
            
            keystroke_count = 0
            mouse_clicks = 0
            mouse_distance = 0.0
            
            while self.running:
                time.sleep(self.sample_interval)
                
                try:
                    # Get current mouse position
                    current_pos = POINT()
                    ctypes.windll.user32.GetCursorPos(ctypes.byref(current_pos))
                    
                    # Calculate movement
                    dx = current_pos.x - last_pos.x
                    dy = current_pos.y - last_pos.y
                    mouse_distance = (dx**2 + dy**2)**0.5
                    
                    last_pos.x = current_pos.x
                    last_pos.y = current_pos.y
                    
                    # Get idle time
                    idle_seconds = get_idle_duration()
                    
                    # Estimate activity (key polling isn't perfect but works)
                    keys_active = get_key_state()
                    
                    self._write_event({
                        "keystroke_count": keys_active,
                        "mouse_click_count": 0,  # Can't reliably detect without hooks
                        "mouse_movement_distance": round(mouse_distance, 1),
                        "idle_seconds": round(idle_seconds, 1),
                        "is_active": idle_seconds < 1.0
                    })
                except Exception as e:
                    print(f"[Input] Error: {e}")
                    
        except Exception as e:
            self.error = e
            print(f"[Input] Fatal: {e}")


class ProcessLogger(SyncLogger):
    """Synchronized process logger (system process monitoring)."""
    
    def __init__(self, ctx: SessionContext):
        super().__init__("process", ctx, sample_interval=3.0)
        self.seen_pids = set()
        
    def _run(self):
        try:
            import psutil
            
            SYSTEM_NOISE = {
                'System', 'Idle', 'Registry', 'smss.exe', 'csrss.exe', 
                'wininit.exe', 'services.exe', 'lsass.exe', 'svchost.exe',
                'dwm.exe', 'winlogon.exe', 'fontdrvhost.exe', 'sihost.exe'
            }
            
            GAME_PROCESSES = {
                'cod.exe', 'modernwarfare.exe', 'BlackOpsColdWar.exe',
                'Warzone.exe', 'FortniteClient-Win64-Shipping.exe'
            }
            
            while self.running:
                time.sleep(self.sample_interval)
                
                try:
                    for proc in psutil.process_iter(['pid', 'name', 'ppid', 'exe']):
                        pid = proc.info['pid']
                        name = proc.info['name']
                        
                        # Skip seen and system noise
                        if pid in self.seen_pids or name in SYSTEM_NOISE:
                            continue
                        
                        self.seen_pids.add(pid)
                        
                        # Determine if flagged (game-related or suspicious)
                        flagged = name.lower() in [g.lower() for g in GAME_PROCESSES]
                        
                        self._write_event({
                            "pid": pid,
                            "process_name": name,
                            "parent_pid": proc.info.get('ppid'),
                            "executable_path": proc.info.get('exe') or "",
                            "flagged": flagged
                        })
                except Exception as e:
                    print(f"[Process] Error: {e}")
                    
        except Exception as e:
            self.error = e
            print(f"[Process] Fatal: {e}")


class GamepadLogger(SyncLogger):
    """Synchronized gamepad logger (controller inputs)."""
    
    def __init__(self, ctx: SessionContext):
        super().__init__("gamepad", ctx, sample_interval=0.016)  # ~60Hz
        
    def _run(self):
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            
            if pygame.joystick.get_count() == 0:
                print("[Gamepad] No controller detected")
                return
            
            controller = pygame.joystick.Joystick(0)
            controller.init()
            print(f"[Gamepad] Found: {controller.get_name()}")
            
            # Track state for change detection
            prev_buttons = {}
            prev_axes = {}
            
            AXIS_DEADZONE = 0.15
            
            while self.running:
                pygame.event.pump()
                
                # Check buttons
                for i in range(controller.get_numbuttons()):
                    state = controller.get_button(i)
                    if prev_buttons.get(i) != state:
                        prev_buttons[i] = state
                        self._write_event({
                            "event": "button_press" if state else "button_release",
                            "button": i,
                            "state": state
                        })
                
                # Check axes
                for i in range(controller.get_numaxes()):
                    value = controller.get_axis(i)
                    if abs(value) > AXIS_DEADZONE:
                        if prev_axes.get(i) is None or abs(value - prev_axes.get(i, 0)) > 0.05:
                            prev_axes[i] = value
                            self._write_event({
                                "event": "axis_move",
                                "axis": i,
                                "value": round(value, 3)
                            })
                    elif prev_axes.get(i) is not None:
                        prev_axes[i] = None
                        self._write_event({
                            "event": "axis_release",
                            "axis": i,
                            "value": 0
                        })
                
                time.sleep(self.sample_interval)
                
        except ImportError:
            print("[Gamepad] pygame not installed, skipping")
        except Exception as e:
            self.error = e
            print(f"[Gamepad] Fatal: {e}")


class NetworkLogger:
    """
    Network logger - runs as PowerShell subprocess with synchronized timestamps.
    """
    
    def __init__(self, ctx: SessionContext):
        self.ctx = ctx
        self.process: Optional[subprocess.Popen] = None
        self.running = False
        
    def start(self):
        """Start network logger as subprocess."""
        script_path = SCRIPT_DIR / "loggers" / "network_logger.ps1"
        
        if not script_path.exists():
            print(f"[Network] Script not found: {script_path}")
            return
        
        output_path = self.ctx.get_log_path("network", "network_state")
        
        # Pass session epoch to PowerShell
        cmd = [
            "powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path),
            "-OutputPath", str(output_path),
            "-SessionEpoch", str(self.ctx.session_epoch),
            "-SessionId", self.ctx.session_id
        ]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            self.running = True
            print(f"[Network] Started (PID: {self.process.pid})")
        except Exception as e:
            print(f"[Network] Failed to start: {e}")
    
    def stop(self):
        """Stop network logger."""
        if self.process and self.process.poll() is None:
            try:
                if os.name == 'nt':
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
        self.running = False
    
    def get_stats(self) -> Dict:
        """Get logger statistics."""
        log_path = self.ctx.get_log_path("network", "network_state")
        if log_path.exists():
            with open(log_path) as f:
                return {"event_count": sum(1 for _ in f)}
        return {"event_count": 0}


class CompuCogCapture:
    """
    Main orchestrator for synchronized multi-modal capture.
    """
    
    def __init__(self, session_name: str = None, event_threshold: float = 0.5):
        # Create session context with precise epoch
        self.ctx = SessionContext.create(session_name=session_name, base_dir=SCRIPT_DIR / "logs")
        self.event_threshold = event_threshold
        
        # Initialize all capture modules
        self.truevision = TrueVisionRunner(self.ctx, event_threshold)
        self.activity = ActivityLogger(self.ctx)
        self.input = InputLogger(self.ctx)
        self.process = ProcessLogger(self.ctx)
        self.gamepad = GamepadLogger(self.ctx)
        self.network = NetworkLogger(self.ctx)
        
        self.running = False
        self._start_time = None
        
    def start(self):
        """Start all capture modules."""
        print("\n" + "="*70)
        print("CompuCog Unified Capture System")
        print("="*70)
        print(f"\nSession ID: {self.ctx.session_id}")
        print(f"Session Epoch: {self.ctx.session_epoch}")
        print(f"Output Directory: {self.ctx.output_dir}")
        print()
        
        # Create directories
        dirs = self.ctx.setup_directories()
        for name, path in dirs.items():
            print(f"  [DIR] {name}: {path}")
        print()
        
        # Start all modules
        print("[Starting Capture Modules]")
        
        print("  Starting TrueVision...")
        self.truevision.start()
        
        print("  Starting Activity Logger...")
        self.activity.start()
        
        print("  Starting Input Logger...")
        self.input.start()
        
        print("  Starting Process Logger...")
        self.process.start()
        
        print("  Starting Gamepad Logger...")
        self.gamepad.start()
        
        print("  Starting Network Logger...")
        self.network.start()
        
        self.running = True
        self._start_time = time.time()
        
        # Write session metadata
        self._write_metadata()
        
        print()
        print("="*70)
        print(f"[RUNNING] All modules active. Press Ctrl+C to stop.")
        print("="*70)
        print()
    
    def _write_metadata(self):
        """Write session metadata file."""
        metadata = {
            **self.ctx.to_dict(),
            "modules": ["truevision", "activity", "input", "process", "gamepad", "network"],
            "event_threshold": self.event_threshold,
            "start_time_iso": datetime.now().isoformat(),
        }
        
        meta_path = self.ctx.output_dir / "session_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run(self, duration: int = None):
        """Run capture until stopped or duration reached."""
        self.start()
        
        try:
            if duration:
                print(f"Running for {duration} seconds...")
                time.sleep(duration)
            else:
                while self.running:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n[INTERRUPT] Ctrl+C received")
        finally:
            self.stop()
    
    def stop(self):
        """Stop all capture modules and finalize."""
        print("\n" + "="*70)
        print("Stopping All Modules")
        print("="*70)
        
        print("  Stopping TrueVision...")
        self.truevision.stop()
        
        print("  Stopping Activity Logger...")
        self.activity.stop()
        
        print("  Stopping Input Logger...")
        self.input.stop()
        
        print("  Stopping Process Logger...")
        self.process.stop()
        
        print("  Stopping Gamepad Logger...")
        self.gamepad.stop()
        
        print("  Stopping Network Logger...")
        self.network.stop()
        
        self.running = False
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        # Update metadata with end time
        meta_path = self.ctx.output_dir / "session_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            metadata["end_time_iso"] = datetime.now().isoformat()
            metadata["duration_seconds"] = elapsed
            metadata["stats"] = {
                "truevision": self.truevision.get_stats(),
                "activity": self.activity.get_stats(),
                "input": self.input.get_stats(),
                "process": self.process.get_stats(),
                "gamepad": self.gamepad.get_stats(),
                "network": self.network.get_stats(),
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Print summary
        print()
        print("="*70)
        print("SESSION COMPLETE")
        print("="*70)
        print(f"\nSession: {self.ctx.session_id}")
        print(f"Duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"\nOutput: {self.ctx.output_dir}")
        print()
        print("Statistics:")
        print(f"  TrueVision: {self.truevision.get_stats()}")
        print(f"  Activity: {self.activity.get_stats()}")
        print(f"  Input: {self.input.get_stats()}")
        print(f"  Process: {self.process.get_stats()}")
        print(f"  Gamepad: {self.gamepad.get_stats()}")
        print(f"  Network: {self.network.get_stats()}")
        print()
        print("Next steps:")
        print(f"  python fusion/fusion.py --session \"{self.ctx.output_dir}\"")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="CompuCog Unified Capture System - Synchronized Multi-Modal Capture"
    )
    parser.add_argument(
        "--session-name", "-n",
        type=str,
        default=None,
        help="Session name (default: auto-generated)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=None,
        help="Capture duration in seconds (default: until Ctrl+C)"
    )
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=0.5,
        help="EOMM threshold for event recording (default: 0.5)"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop any running capture session"
    )
    
    args = parser.parse_args()
    
    if args.stop:
        print("Stop functionality not yet implemented")
        return 0
    
    capture = CompuCogCapture(
        session_name=args.session_name,
        event_threshold=args.event_threshold
    )
    
    capture.run(duration=args.duration)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
