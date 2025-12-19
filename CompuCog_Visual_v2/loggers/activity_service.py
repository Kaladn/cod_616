"""Activity logger service wrapper.

Provides a simple start/stop wrapper around the existing
`loggers/activity_logger.py` script. Runs the script in a subprocess
by default but can be instantiated in "stub" mode for tests.
"""
from pathlib import Path
import subprocess
import sys
import os
import time
import logging
from typing import Optional

SCRIPT_PATH = Path(__file__).parent / "activity_logger.py"


class ActivityService:
    def __init__(self, config_path: Optional[str] = None, use_subprocess: bool = True):
        self.config_path = config_path
        self.use_subprocess = use_subprocess
        self._proc: Optional[subprocess.Popen] = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        if not self.use_subprocess:
            # Stub mode (useful for tests)
            logging.info("[ActivityService] Stub start (no subprocess)")
            self._started = True
            return

        if not SCRIPT_PATH.exists():
            logging.warning(f"[ActivityService] Script not found: {SCRIPT_PATH}")
            return

        cmd = [sys.executable, str(SCRIPT_PATH)]
        env = os.environ.copy()
        if self.config_path:
            env['COMPUCOG_ACTIVITY_CONFIG'] = str(self.config_path)

        logging.info(f"[ActivityService] Starting subprocess: {cmd}")
        self._proc = subprocess.Popen(cmd, env=env)
        # Small wait to allow process to initialize
        time.sleep(0.1)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return

        if not self.use_subprocess:
            logging.info("[ActivityService] Stub stop")
            self._started = False
            return

        if self._proc is not None:
            logging.info(f"[ActivityService] Terminating subprocess PID={self._proc.pid}")
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
                finally:
                    self._proc = None

        self._started = False

    def is_running(self) -> bool:
        return self._started
