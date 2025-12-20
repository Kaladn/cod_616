import time
from enum import Enum

class ChronosMode(Enum):
    LIVE = "live"
    TEST = "test"

class ChronosManager:
    def __init__(self):
        self._start = time.time()
        self._mode = ChronosMode.TEST

    def initialize(self, mode: ChronosMode):
        self._mode = mode
        self._start = time.time()

    def now(self) -> float:
        return time.time()
