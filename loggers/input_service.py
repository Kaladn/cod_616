"""Input logger service wrapper.

Provides start/stop wrapper around the existing loggers/input_logger.py
script. Offers stub mode for tests.
"""
from pathlib import Path
import subprocess
import sys
import os
import time
import logging
from typing import Optional

SCRIPT_PATH = Path(__file__).parent / "input_logger.py"


class InputService:
    def __init__(self, config_path: Optional[str] = None, use_subprocess: bool = True):
        self.config_path = config_path
        self.use_subprocess = use_subprocess
        self._proc: Optional[subprocess.Popen] = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        if not self.use_subprocess:
            logging.info("[InputService] Stub start (no subprocess)")
            self._started = True
            return

        if not SCRIPT_PATH.exists():
            logging.warning(f"[InputService] Script not found: {SCRIPT_PATH}")
            return

        cmd = [sys.executable, str(SCRIPT_PATH)]
        env = os.environ.copy()
        if self.config_path:
            env['COMPUCOG_INPUT_CONFIG'] = str(self.config_path)

        logging.info(f"[InputService] Starting subprocess: {cmd}")
        self._proc = subprocess.Popen(cmd, env=env)
        time.sleep(0.1)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return

        if not self.use_subprocess:
            logging.info("[InputService] Stub stop")
            self._started = False
            return

        if self._proc is not None:
            logging.info(f"[InputService] Terminating subprocess PID={self._proc.pid}")
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
