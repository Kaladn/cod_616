"""
Unified Logger Launcher

Starts ALL loggers simultaneously for temporally-aligned data capture.
Creates a single session directory with synchronized timestamps.

Usage:
    python unified_logger.py --session-name "warzone_session_001"
    python unified_logger.py --duration 3600  # 1 hour capture
    python unified_logger.py --stop  # Stop all running loggers
"""

import os
import sys
import time
import json
import signal
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import threading


class UnifiedLogger:
    """
    Orchestrates all logger processes for synchronized capture.
    """
    
    def __init__(self, session_name: Optional[str] = None, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent
        self.loggers_dir = self.base_dir / "loggers"
        
        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"session_{timestamp}"
        self.session_dir = self.base_dir / "logs" / self.session_name
        
        # Create subdirectories for each modality
        self.dirs = {
            "activity": self.session_dir / "activity",
            "gamepad": self.session_dir / "gamepad",
            "input": self.session_dir / "input",
            "network": self.session_dir / "network",
            "process": self.session_dir / "process",
            "truevision": self.session_dir / "truevision",
        }
        
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = False
        self._start_time: Optional[float] = None
        
    def setup_directories(self):
        """Create all session directories."""
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  [DIR] {name}: {path}")
    
    def start_activity_logger(self) -> Optional[subprocess.Popen]:
        """Start the activity logger (window/process tracking)."""
        script = self.loggers_dir / "activity_logger.py"
        if not script.exists():
            print(f"  [WARN] activity_logger.py not found")
            return None
        
        output_file = self.dirs["activity"] / f"user_activity_{self.session_name}.jsonl"
        
        proc = subprocess.Popen(
            [sys.executable, str(script), "--output", str(output_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        print(f"  [START] activity_logger (PID: {proc.pid})")
        return proc
    
    def start_gamepad_logger(self) -> Optional[subprocess.Popen]:
        """Start the gamepad logger (controller inputs)."""
        script = self.loggers_dir / "gamepad_logger_continuous.py"
        if not script.exists():
            print(f"  [WARN] gamepad_logger_continuous.py not found")
            return None
        
        output_file = self.dirs["gamepad"] / f"gamepad_stream_{self.session_name}.jsonl"
        
        proc = subprocess.Popen(
            [sys.executable, str(script), "--output", str(output_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        print(f"  [START] gamepad_logger (PID: {proc.pid})")
        return proc
    
    def start_input_logger(self) -> Optional[subprocess.Popen]:
        """Start the input logger (keyboard/mouse telemetry)."""
        script = self.loggers_dir / "input_logger.py"
        if not script.exists():
            print(f"  [WARN] input_logger.py not found")
            return None
        
        output_file = self.dirs["input"] / f"telemetry_{self.session_name}.jsonl"
        
        proc = subprocess.Popen(
            [sys.executable, str(script), "--output", str(output_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        print(f"  [START] input_logger (PID: {proc.pid})")
        return proc
    
    def start_network_logger(self) -> Optional[subprocess.Popen]:
        """Start the network logger (TCP/UDP connections)."""
        script = self.loggers_dir / "network_logger.ps1"
        if not script.exists():
            print(f"  [WARN] network_logger.ps1 not found")
            return None
        
        output_file = self.dirs["network"] / f"network_state_{self.session_name}.jsonl"
        
        proc = subprocess.Popen(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script),
             "-OutputPath", str(output_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        print(f"  [START] network_logger (PID: {proc.pid})")
        return proc
    
    def start_process_logger(self) -> Optional[subprocess.Popen]:
        """Start the process logger (system process monitoring)."""
        script = self.loggers_dir / "process_logger.py"
        if not script.exists():
            print(f"  [WARN] process_logger.py not found")
            return None
        
        output_file = self.dirs["process"] / f"process_events_{self.session_name}.jsonl"
        
        proc = subprocess.Popen(
            [sys.executable, str(script), "--output", str(output_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        print(f"  [START] process_logger (PID: {proc.pid})")
        return proc
    
    def start_all(self):
        """Start all loggers."""
        print(f"\n{'='*60}")
        print(f"UNIFIED LOGGER - Starting Session: {self.session_name}")
        print(f"{'='*60}")
        print(f"\nSession Directory: {self.session_dir}\n")
        
        self.setup_directories()
        print()
        
        # Start all loggers
        self.processes["activity"] = self.start_activity_logger()
        self.processes["gamepad"] = self.start_gamepad_logger()
        self.processes["input"] = self.start_input_logger()
        self.processes["network"] = self.start_network_logger()
        self.processes["process"] = self.start_process_logger()
        
        # Remove None entries
        self.processes = {k: v for k, v in self.processes.items() if v is not None}
        
        self.running = True
        self._start_time = time.time()
        
        # Write session metadata
        self._write_session_metadata()
        
        print(f"\n{'='*60}")
        print(f"[RUNNING] {len(self.processes)} loggers active")
        print(f"          Press Ctrl+C to stop all loggers")
        print(f"{'='*60}\n")
    
    def _write_session_metadata(self):
        """Write session metadata file."""
        metadata = {
            "session_name": self.session_name,
            "start_time": datetime.now().isoformat(),
            "start_epoch": self._start_time,
            "loggers": list(self.processes.keys()),
            "pids": {k: v.pid for k, v in self.processes.items()},
            "directories": {k: str(v) for k, v in self.dirs.items()},
        }
        
        meta_file = self.session_dir / "session_metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  [META] Session metadata written to {meta_file}")
    
    def stop_all(self):
        """Stop all loggers gracefully."""
        print(f"\n{'='*60}")
        print("STOPPING ALL LOGGERS")
        print(f"{'='*60}\n")
        
        for name, proc in self.processes.items():
            if proc and proc.poll() is None:
                print(f"  [STOP] {name} (PID: {proc.pid})")
                try:
                    if os.name == 'nt':
                        # Windows: send CTRL_BREAK_EVENT
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  [KILL] {name} (force)")
                    proc.kill()
                except Exception as e:
                    print(f"  [ERROR] {name}: {e}")
        
        self.running = False
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        # Update session metadata
        meta_file = self.session_dir / "session_metadata.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            metadata["end_time"] = datetime.now().isoformat()
            metadata["duration_seconds"] = elapsed
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"\n  Session duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"  Logs saved to: {self.session_dir}")
        print(f"\n{'='*60}")
        print("ALL LOGGERS STOPPED")
        print(f"{'='*60}\n")
    
    def run(self, duration: Optional[int] = None):
        """Run loggers until stopped or duration reached."""
        self.start_all()
        
        try:
            if duration:
                print(f"  Running for {duration} seconds...")
                time.sleep(duration)
            else:
                # Run until Ctrl+C
                while self.running:
                    time.sleep(1)
                    # Check if any logger died
                    for name, proc in self.processes.items():
                        if proc.poll() is not None:
                            print(f"  [DIED] {name} (exit code: {proc.returncode})")
        except KeyboardInterrupt:
            print("\n\n  [INTERRUPT] Ctrl+C received")
        finally:
            self.stop_all()
    
    def status(self):
        """Check status of running loggers."""
        print(f"\n{'='*60}")
        print("LOGGER STATUS")
        print(f"{'='*60}\n")
        
        for name, proc in self.processes.items():
            if proc:
                status = "RUNNING" if proc.poll() is None else f"STOPPED (exit: {proc.returncode})"
                print(f"  {name:12}: {status} (PID: {proc.pid})")
        
        if self._start_time:
            elapsed = time.time() - self._start_time
            print(f"\n  Elapsed: {elapsed:.1f} seconds")


def find_running_loggers() -> List[Dict]:
    """Find any running logger sessions."""
    # Look for session_metadata.json files
    base = Path(__file__).parent / "logs"
    sessions = []
    
    if base.exists():
        for session_dir in base.iterdir():
            if session_dir.is_dir():
                meta_file = session_dir / "session_metadata.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = json.load(f)
                    if "end_time" not in meta:  # Still running
                        sessions.append(meta)
    
    return sessions


def main():
    parser = argparse.ArgumentParser(description="Unified Logger Launcher")
    parser.add_argument("--session-name", "-n", type=str, default=None,
                        help="Name for this capture session")
    parser.add_argument("--duration", "-d", type=int, default=None,
                        help="Capture duration in seconds (default: until Ctrl+C)")
    parser.add_argument("--stop", action="store_true",
                        help="Stop all running loggers")
    parser.add_argument("--status", action="store_true",
                        help="Show status of running loggers")
    
    args = parser.parse_args()
    
    if args.stop:
        sessions = find_running_loggers()
        if sessions:
            print(f"Found {len(sessions)} running sessions:")
            for s in sessions:
                print(f"  - {s['session_name']} (PIDs: {s['pids']})")
                for name, pid in s['pids'].items():
                    try:
                        os.kill(pid, signal.SIGTERM)
                        print(f"    [STOPPED] {name} (PID: {pid})")
                    except:
                        pass
        else:
            print("No running sessions found.")
        return 0
    
    if args.status:
        sessions = find_running_loggers()
        if sessions:
            for s in sessions:
                print(f"Session: {s['session_name']}")
                print(f"  Started: {s['start_time']}")
                print(f"  Loggers: {', '.join(s['loggers'])}")
        else:
            print("No running sessions.")
        return 0
    
    # Start new session
    logger = UnifiedLogger(session_name=args.session_name)
    logger.run(duration=args.duration)
    
    return 0


if __name__ == "__main__":
    exit(main())
