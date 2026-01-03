"""
Session Context - Shared timing reference for all capture modules.

Provides:
- Unique session ID
- Precise epoch timestamp (microsecond precision)
- Synchronized timestamp generation for all modalities
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any
from pathlib import Path


@dataclass
class SessionContext:
    """
    Shared context for synchronized multi-modal capture.
    
    All modules use this to generate aligned timestamps.
    """
    session_id: str
    session_epoch: float  # Unix timestamp with microsecond precision
    output_dir: Path
    
    # Runtime state
    _start_time: float = field(default=0.0, repr=False)
    
    def __post_init__(self):
        self._start_time = self.session_epoch
        self.output_dir = Path(self.output_dir)
    
    @classmethod
    def create(cls, session_name: str = None, base_dir: Path = None) -> "SessionContext":
        """
        Factory method to create a new session context.
        
        Args:
            session_name: Optional custom session name
            base_dir: Base directory for logs (default: CompuCog_Visual_v2/logs)
        
        Returns:
            Configured SessionContext with synchronized epoch
        """
        # Capture epoch with maximum precision FIRST
        session_epoch = time.time()
        
        # Generate session ID from timestamp if not provided
        if session_name:
            session_id = session_name
        else:
            dt = datetime.fromtimestamp(session_epoch)
            session_id = dt.strftime("session_%Y%m%d_%H%M%S")
        
        # Determine output directory
        if base_dir is None:
            base_dir = Path(__file__).parent.parent / "logs"
        else:
            base_dir = Path(base_dir)
        
        output_dir = base_dir / session_id
        
        return cls(
            session_id=session_id,
            session_epoch=session_epoch,
            output_dir=output_dir
        )
    
    def get_offset_ms(self) -> float:
        """Get milliseconds elapsed since session start."""
        return (time.time() - self.session_epoch) * 1000.0
    
    def get_timestamp(self) -> Dict[str, Any]:
        """
        Generate synchronized timestamp dict for any event.
        
        Returns dict with:
            - session_id: Links event to session
            - session_epoch: Session start time
            - event_epoch: Current time with microsecond precision
            - event_offset_ms: Milliseconds since session start
            - timestamp_iso: Human-readable ISO format
        """
        now = time.time()
        offset_ms = (now - self.session_epoch) * 1000.0
        
        return {
            "session_id": self.session_id,
            "session_epoch": self.session_epoch,
            "event_epoch": now,
            "event_offset_ms": round(offset_ms, 3),
            "timestamp_iso": datetime.fromtimestamp(now).isoformat()
        }
    
    def get_modality_dir(self, modality: str) -> Path:
        """Get output directory for a specific modality."""
        modality_dir = self.output_dir / modality
        modality_dir.mkdir(parents=True, exist_ok=True)
        return modality_dir
    
    def get_log_path(self, modality: str, prefix: str = None) -> Path:
        """Get log file path for a modality."""
        prefix = prefix or modality
        filename = f"{prefix}_{self.session_id}.jsonl"
        return self.get_modality_dir(modality) / filename
    
    def setup_directories(self) -> Dict[str, Path]:
        """Create all modality directories and return paths."""
        modalities = ["truevision", "activity", "gamepad", "input", "network", "process"]
        dirs = {}
        
        for mod in modalities:
            dirs[mod] = self.get_modality_dir(mod)
        
        return dirs
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dict (for metadata file)."""
        return {
            "session_id": self.session_id,
            "session_epoch": self.session_epoch,
            "session_start_iso": datetime.fromtimestamp(self.session_epoch).isoformat(),
            "output_dir": str(self.output_dir)
        }
    
    def to_powershell_args(self) -> str:
        """Generate PowerShell parameter string for network logger."""
        return f'-SessionId "{self.session_id}" -SessionEpoch {self.session_epoch}'
