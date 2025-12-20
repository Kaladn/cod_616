from __future__ import annotations

import time
import threading
from enum import Enum
from typing import Any, Dict, Optional, Union


class ChronosMode(str, Enum):
    """
    Operating modes for ChronosManager.

    LIVE      → wall-clock based, monotonic
    REPLAY    → wall-clock * replay_speed + offsets, monotonic
    SIMULATED → fully controlled time, advanced manually
    """

    LIVE = "live"
    REPLAY = "replay"
    SIMULATED = "simulated"

    @classmethod
    def from_any(cls, value: Union["ChronosMode", str]) -> "ChronosMode":
        if isinstance(value, ChronosMode):
            return value
        v = str(value).strip().lower()
        if v in ("live", "l"):
            return cls.LIVE
        if v in ("replay", "r"):
            return cls.REPLAY
        if v in ("simulated", "sim", "s"):
            return cls.SIMULATED
        raise ValueError(f"Unknown ChronosMode: {value!r}")


class ChronosManager:
    """
    ChronosManager v1 — deterministic cognitive time source.

    Responsibilities:
      - Provide a single monotonic timestamp source for the system
      - Support LIVE / REPLAY / SIMULATED modes
      - Allow controlled replay (speed, offset, anchor)
      - Allow unit-test-friendly simulated time control
      - Be thread-safe and side-effect free (beyond time)
    """

    DEFAULT_REPLAY_SPEED = 1.0

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Core state
        self.mode: ChronosMode = ChronosMode.LIVE
        self.initialized: bool = False

        # Base references
        self.boot_wall_time: float = 0.0      # wall-clock when (re)initialized
        self.boot_timestamp: float = 0.0      # logical time at boot (for REPLAY/SIM)

        # LIVE / general monotonic control
        self.last_timestamp: float = 0.0

        # REPLAY state
        self.replay_start_time: float = 0.0   # logical start timestamp for replay
        self.replay_speed: float = self.DEFAULT_REPLAY_SPEED
        self.replay_offset: float = 0.0       # additive offset to logical time
        self.replay_anchor_wall: float = 0.0  # wall time when replay began

        # SIMULATED state
        self.simulated_time: float = 0.0

    # ------------------------------------------------------------------
    # Initialization / mode configuration
    # ------------------------------------------------------------------

    def initialize(
        self,
        mode: Union[ChronosMode, str] = ChronosMode.LIVE,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize or re-initialize ChronosManager.

        Args:
            mode: LIVE / REPLAY / SIMULATED (string or ChronosMode)
            config: optional per-mode configuration:
                REPLAY:
                    replay_start_time: float (base timestamp)
                    replay_speed: float
                    replay_offset: float
                SIMULATED:
                    simulated_time: float (initial time)
        """
        with self._lock:
            self.mode = ChronosMode.from_any(mode)
            cfg = config or {}

            wall_now = time.time()
            self.boot_wall_time = wall_now

            if self.mode is ChronosMode.LIVE:
                # LIVE mode: start from wall clock
                self.boot_timestamp = wall_now
                self.last_timestamp = self.boot_timestamp

            elif self.mode is ChronosMode.REPLAY:
                # REPLAY mode: logical time follows:
                #   T_logical = replay_start_time + (wall_now - replay_anchor_wall) * speed + offset
                self.replay_start_time = float(
                    cfg.get("replay_start_time", wall_now)
                )
                self.replay_speed = float(
                    cfg.get("replay_speed", self.DEFAULT_REPLAY_SPEED)
                )
                self.replay_offset = float(cfg.get("replay_offset", 0.0))
                self.replay_anchor_wall = wall_now

                self.boot_timestamp = self.replay_start_time + self.replay_offset
                self.last_timestamp = self.boot_timestamp

            else:  # SIMULATED
                self.simulated_time = float(cfg.get("simulated_time", 0.0))
                self.boot_timestamp = self.simulated_time
                self.last_timestamp = self.simulated_time

            self.initialized = True

    def set_mode(
        self,
        mode: Union[ChronosMode, str],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Switch modes at runtime.

        This is equivalent to re-calling initialize() with a new mode.
        """
        self.initialize(mode=mode, config=config)

    # ------------------------------------------------------------------
    # Time retrieval
    # ------------------------------------------------------------------

    def now(self) -> float:
        """
        Return the current logical timestamp in seconds.

        - LIVE: wall-clock based, monotonic
        - REPLAY: wall-clock scaled by replay_speed, plus offsets
        - SIMULATED: internal simulated_time
        """
        with self._lock:
            if not self.initialized:
                # Default to LIVE init for safety if user forgot
                self.initialize(ChronosMode.LIVE)

            if self.mode is ChronosMode.LIVE:
                timestamp = self._now_live_locked()
            elif self.mode is ChronosMode.REPLAY:
                timestamp = self._now_replay_locked()
            else:
                timestamp = self._now_simulated_locked()

            # Monotonic clamp: never go backwards
            if timestamp < self.last_timestamp:
                timestamp = self.last_timestamp
            else:
                self.last_timestamp = timestamp

            return timestamp

    def delta(self, previous_timestamp: float) -> float:
        """
        Return a non-negative delta between now() and a previous timestamp.
        """
        current = self.now()
        d = current - float(previous_timestamp)
        if d < 0.0:
            return 0.0
        return d

    # ------------------------------------------------------------------
    # Mode-specific time generators (caller holds lock)
    # ------------------------------------------------------------------

    def _now_live_locked(self) -> float:
        return time.time()

    def _now_replay_locked(self) -> float:
        wall_now = time.time()
        elapsed_wall = wall_now - self.replay_anchor_wall
        logical = (
            self.replay_start_time
            + elapsed_wall * self.replay_speed
            + self.replay_offset
        )
        return logical

    def _now_simulated_locked(self) -> float:
        # In SIMULATED mode, we never auto-advance time.
        return self.simulated_time

    # ------------------------------------------------------------------
    # REPLAY controls
    # ------------------------------------------------------------------

    def configure_replay(
        self,
        *,
        replay_start_time: Optional[float] = None,
        replay_speed: Optional[float] = None,
        replay_offset: Optional[float] = None,
        reset_anchor: bool = True,
    ) -> None:
        """
        Update replay parameters.

        Args:
            replay_start_time: base logical timestamp
            replay_speed: speed multiplier (1.0 = real-time, 2.0 = 2x)
            replay_offset: additive offset to logical time
            reset_anchor: if True, set replay_anchor_wall = current wall time
        """
        with self._lock:
            if replay_start_time is not None:
                self.replay_start_time = float(replay_start_time)
            if replay_speed is not None:
                self.replay_speed = float(replay_speed)
            if replay_offset is not None:
                self.replay_offset = float(replay_offset)
            if reset_anchor:
                self.replay_anchor_wall = time.time()

            # Recompute last_timestamp based on new parameters
            if self.mode is ChronosMode.REPLAY:
                t = self._now_replay_locked()
                if t < self.last_timestamp:
                    t = self.last_timestamp
                self.last_timestamp = t

    # ------------------------------------------------------------------
    # SIMULATED controls
    # ------------------------------------------------------------------

    def set_simulated_time(self, new_time: float) -> float:
        """
        Hard-set the simulated logical time.

        Monotonic guarantee:
          - if new_time < last_timestamp, we clamp to last_timestamp
        Returns the effective simulated time.
        """
        with self._lock:
            new_time = float(new_time)
            # We preserve global monotonicity
            if new_time < self.last_timestamp:
                self.simulated_time = self.last_timestamp
            else:
                self.simulated_time = new_time
                self.last_timestamp = new_time
            return self.simulated_time

    def advance_simulated(self, delta_seconds: float) -> float:
        """
        Advance simulated_time by a non-negative delta.

        Returns the new simulated_time.
        """
        if delta_seconds < 0.0:
            delta_seconds = 0.0

        with self._lock:
            self.simulated_time += float(delta_seconds)
            if self.simulated_time < self.last_timestamp:
                self.simulated_time = self.last_timestamp
            else:
                self.last_timestamp = self.simulated_time
            return self.simulated_time

    # ------------------------------------------------------------------
    # Diagnostics / metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return a snapshot of ChronosManager state for debugging and logging.
        """
        with self._lock:
            return {
                "mode": self.mode.value,
                "initialized": self.initialized,
                "boot_wall_time": self.boot_wall_time,
                "boot_timestamp": self.boot_timestamp,
                "last_timestamp": self.last_timestamp,
                "replay": {
                    "start_time": self.replay_start_time,
                    "speed": self.replay_speed,
                    "offset": self.replay_offset,
                    "anchor_wall": self.replay_anchor_wall,
                },
                "simulated_time": self.simulated_time,
            }

    def print_status(self) -> None:
        """
        Print a human-readable summary of the current Chronos state.
        """
        meta = self.get_metadata()
        mode = meta["mode"]
        print("[ChronosManager] Status:")
        print(f"  Mode: {mode}")
        print(f"  Initialized: {meta['initialized']}")
        print(f"  Last timestamp: {meta['last_timestamp']:.6f}")
        if mode == ChronosMode.REPLAY.value:
            r = meta["replay"]
            print("  Replay:")
            print(f"    start_time: {r['start_time']}")
            print(f"    speed     : {r['speed']}")
            print(f"    offset    : {r['offset']}")
        elif mode == ChronosMode.SIMULATED.value:
            print(f"  Simulated time: {meta['simulated_time']}")
