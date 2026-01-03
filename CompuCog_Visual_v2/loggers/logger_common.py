"""
Common utilities for all loggers.

Features:
1. Unified timestamp format
2. Auto-cleanup of data older than 1 day
3. Evidence sealing when issues detected
4. Session metadata creation
5. System health checks
6. Rate limiting for high-frequency writes
7. Session compression for archival

ALL loggers import from here to ensure format consistency.
"""

import os
import json
import time
import gzip
import shutil
import tarfile
import logging
import platform
from pathlib import Path
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED TIMESTAMP FORMAT - All loggers use this
# ═══════════════════════════════════════════════════════════════════════════════

def get_timestamp() -> Dict[str, Any]:
    """
    Get unified timestamp dict for all loggers.
    
    Returns consistent format:
        {
            "timestamp": "2026-01-03T10:30:45.123456",  # ISO 8601
            "epoch": 1767450645.123456,                 # Unix epoch
            "date": "20260103"                          # For file naming
        }
    """
    now = datetime.now()
    return {
        "timestamp": now.isoformat(),
        "epoch": time.time(),
        "date": now.strftime("%Y%m%d")
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED WRITE - All loggers use this
# ═══════════════════════════════════════════════════════════════════════════════

def write_safe(log_file: Path, data: Dict[str, Any], max_retries: int = 3) -> bool:
    """
    IMMORTAL write - retry up to max_retries times, never crash.
    
    Args:
        log_file: Path to JSONL file
        data: Dict to write as JSON line
        max_retries: Number of retry attempts
        
    Returns:
        True if written, False if failed (but NEVER crashes)
    """
    for attempt in range(max_retries):
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
                f.flush()
            return True
        except Exception:
            time.sleep(0.05 * (attempt + 1))  # Backoff
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-CLEANUP - Delete data older than 1 day
# ═══════════════════════════════════════════════════════════════════════════════

def cleanup_old_logs(logs_dir: Path, max_age_days: int = 1) -> int:
    """
    Delete log files older than max_age_days.
    
    Args:
        logs_dir: Directory containing log files
        max_age_days: Maximum age in days (default: 1)
        
    Returns:
        Number of files deleted
    """
    if not logs_dir.exists():
        return 0
    
    cutoff = datetime.now() - timedelta(days=max_age_days)
    deleted = 0
    
    try:
        for file_path in logs_dir.glob("*.jsonl"):
            try:
                # Check file modification time
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff:
                    file_path.unlink()
                    deleted += 1
                    logging.debug(f"Deleted old log: {file_path.name}")
            except Exception as e:
                logging.debug(f"Could not delete {file_path}: {e}")
    except Exception as e:
        logging.debug(f"Cleanup error: {e}")
    
    return deleted


def cleanup_old_sessions(logs_base_dir: Path, max_age_days: int = 1) -> int:
    """
    Delete entire session directories older than max_age_days.
    
    Args:
        logs_base_dir: Base logs directory containing session subdirs
        max_age_days: Maximum age in days (default: 1)
        
    Returns:
        Number of sessions deleted
    """
    if not logs_base_dir.exists():
        return 0
    
    cutoff = datetime.now() - timedelta(days=max_age_days)
    deleted = 0
    
    try:
        for session_dir in logs_base_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            # Skip evidence directory
            if session_dir.name == "evidence":
                continue
            
            try:
                # Check directory modification time
                mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(session_dir, ignore_errors=True)
                    deleted += 1
                    logging.info(f"Deleted old session: {session_dir.name}")
            except Exception as e:
                logging.debug(f"Could not delete session {session_dir}: {e}")
    except Exception as e:
        logging.debug(f"Session cleanup error: {e}")
    
    return deleted


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENCE SEALING - Preserve data when issues found
# ═══════════════════════════════════════════════════════════════════════════════

def seal_evidence(
    session_dir: Path,
    reason: str,
    findings: Dict[str, Any] = None,
    evidence_base: Path = None
) -> Optional[Path]:
    """
    Seal a session as evidence when issues are detected.
    
    This copies the session to an evidence directory that is:
    - NOT subject to auto-cleanup
    - Tagged with reason and timestamp
    - Contains a report file
    
    Args:
        session_dir: Session directory to seal
        reason: Why this is being sealed (e.g., "HIGH_EOMM_DETECTED")
        findings: Dict of specific findings to include in report
        evidence_base: Base evidence directory (default: logs/evidence)
        
    Returns:
        Path to sealed evidence directory, or None if failed
    """
    if not session_dir.exists():
        logging.error(f"Cannot seal: session not found: {session_dir}")
        return None
    
    # Determine evidence location
    if evidence_base is None:
        evidence_base = session_dir.parent / "evidence"
    
    evidence_base.mkdir(parents=True, exist_ok=True)
    
    # Create evidence ID
    evidence_id = f"EVD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_dir.name}"
    evidence_dir = evidence_base / evidence_id
    
    try:
        # Copy session to evidence
        shutil.copytree(session_dir, evidence_dir)
        
        # Create evidence report
        report = {
            "evidence_id": evidence_id,
            "sealed_at": datetime.now().isoformat(),
            "sealed_epoch": time.time(),
            "original_session": session_dir.name,
            "reason": reason,
            "findings": findings or {},
            "files": [f.name for f in evidence_dir.glob("**/*") if f.is_file()]
        }
        
        report_path = evidence_dir / "EVIDENCE_REPORT.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create human-readable summary
        summary_path = evidence_dir / "EVIDENCE_SUMMARY.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"EVIDENCE SEALED\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Evidence ID: {evidence_id}\n")
            f.write(f"Sealed At: {report['sealed_at']}\n")
            f.write(f"Original Session: {session_dir.name}\n")
            f.write(f"Reason: {reason}\n\n")
            f.write(f"{'='*60}\n")
            f.write(f"FINDINGS\n")
            f.write(f"{'='*60}\n\n")
            if findings:
                for key, value in findings.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write("No specific findings recorded.\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"FILES ({len(report['files'])})\n")
            f.write(f"{'='*60}\n\n")
            for file in report['files']:
                f.write(f"  - {file}\n")
        
        logging.info(f"Evidence sealed: {evidence_id}")
        return evidence_dir
        
    except Exception as e:
        logging.error(f"Failed to seal evidence: {e}")
        return None


def check_and_seal_session(
    session_dir: Path,
    eomm_threshold: float = 0.7,
    min_high_eomm_events: int = 3
) -> Optional[Path]:
    """
    Check a session for issues and seal as evidence if needed.
    
    Args:
        session_dir: Session directory to check
        eomm_threshold: EOMM score threshold for "high"
        min_high_eomm_events: Minimum high-EOMM events to trigger seal
        
    Returns:
        Path to evidence if sealed, None otherwise
    """
    # Look for TrueVision data
    tv_dir = session_dir / "truevision"
    if not tv_dir.exists():
        return None
    
    high_eomm_count = 0
    max_eomm = 0.0
    findings = {
        "high_eomm_events": [],
        "max_eomm_score": 0.0,
        "total_events": 0
    }
    
    try:
        for jsonl_file in tv_dir.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8-sig") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        findings["total_events"] += 1
                        
                        eomm = event.get("eomm_composite_score", 0)
                        if eomm > max_eomm:
                            max_eomm = eomm
                        
                        if eomm >= eomm_threshold:
                            high_eomm_count += 1
                            findings["high_eomm_events"].append({
                                "offset_ms": event.get("event_offset_ms"),
                                "score": eomm,
                                "flags": event.get("eomm_flags", [])
                            })
                    except:
                        pass
        
        findings["max_eomm_score"] = max_eomm
        
        # Check if we should seal
        if high_eomm_count >= min_high_eomm_events:
            return seal_evidence(
                session_dir=session_dir,
                reason=f"HIGH_EOMM_DETECTED ({high_eomm_count} events >= {eomm_threshold})",
                findings=findings
            )
        
    except Exception as e:
        logging.error(f"Error checking session: {e}")
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP CLEANUP - Run on logger start
# ═══════════════════════════════════════════════════════════════════════════════

def startup_cleanup(logs_base_dir: Path) -> Dict[str, int]:
    """
    Run cleanup tasks on logger startup.
    
    1. Delete sessions older than 1 day (except evidence)
    2. Delete loose log files older than 1 day
    
    Args:
        logs_base_dir: Base logs directory
        
    Returns:
        Dict with cleanup stats
    """
    stats = {
        "sessions_deleted": 0,
        "files_deleted": 0
    }
    
    try:
        # Clean old sessions
        stats["sessions_deleted"] = cleanup_old_sessions(logs_base_dir, max_age_days=1)
        
        # Clean each modality subdirectory
        for subdir in ["activity", "input", "process", "gamepad", "network", "truevision"]:
            modality_dir = logs_base_dir / subdir
            if modality_dir.exists():
                stats["files_deleted"] += cleanup_old_logs(modality_dir, max_age_days=1)
        
        if stats["sessions_deleted"] > 0 or stats["files_deleted"] > 0:
            logging.info(f"Startup cleanup: {stats['sessions_deleted']} sessions, {stats['files_deleted']} files deleted")
    
    except Exception as e:
        logging.debug(f"Startup cleanup error: {e}")
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 1: SESSION METADATA
# ═══════════════════════════════════════════════════════════════════════════════

def create_session_metadata(session_dir: Path, 
                          game: str = "unknown",
                          additional_meta: Dict[str, Any] = None) -> bool:
    """
    Create metadata file at start of each session.
    
    Args:
        session_dir: Session directory path
        game: Game/application name
        additional_meta: Additional metadata to include
        
    Returns:
        True if successful
    """
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "session_id": session_dir.name,
            "game": game,
            "system": {
                "platform": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "start_time": get_timestamp(),
            "logger_versions": {
                "common": "2.1",
                "activity": "1.2",
                "input": "1.1",
                "process": "1.1", 
                "gamepad": "1.3",
                "network": "1.1"
            }
        }
        
        if additional_meta:
            metadata.update(additional_meta)
        
        # Write as JSON (not JSONL) for readability
        metadata_path = session_dir / "SESSION_METADATA.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to create session metadata: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 2: HEALTH CHECK ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def get_system_health() -> Dict[str, Any]:
    """
    Return comprehensive system health status.
    Safe to call - never raises exceptions.
    
    Returns:
        Dict with timestamp, status, and checks
    """
    health = {
        "timestamp": get_timestamp(),
        "status": "healthy",
        "checks": {}
    }
    
    try:
        import psutil
        
        # Disk health
        try:
            disk = psutil.disk_usage(".")
            health["checks"]["disk"] = {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": disk.percent,
                "healthy": disk.percent < 90
            }
        except Exception:
            health["checks"]["disk"] = {"healthy": False, "error": "check_failed"}
        
        # Memory health
        try:
            mem = psutil.virtual_memory()
            health["checks"]["memory"] = {
                "percent_used": mem.percent,
                "available_gb": round(mem.available / (1024**3), 2),
                "healthy": mem.percent < 85
            }
        except Exception:
            health["checks"]["memory"] = {"healthy": False, "error": "check_failed"}
        
        # Logger processes health
        try:
            logger_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'logger' in cmdline.lower() and 'python' in cmdline.lower():
                        logger_pids.append(proc.info['pid'])
                except Exception:
                    continue
            
            health["checks"]["loggers"] = {
                "count": len(logger_pids),
                "expected": 5,  # activity, input, process, gamepad, network
                "healthy": len(logger_pids) >= 3,  # At least 3/5 running
                "pids": logger_pids
            }
        except Exception:
            health["checks"]["loggers"] = {"healthy": False, "error": "check_failed"}
        
        # Determine overall status
        unhealthy_checks = [c for c in health["checks"].values() 
                          if isinstance(c, dict) and c.get("healthy") is False]
        if unhealthy_checks:
            health["status"] = "degraded"
            health["unhealthy_checks"] = len(unhealthy_checks)
            
    except ImportError:
        health["status"] = "unknown"
        health["note"] = "psutil not available for detailed checks"
    
    return health


def write_health_check(health_dir: Path = None) -> bool:
    """
    Write current health status to health directory.
    Creates rotating health logs.
    
    Args:
        health_dir: Directory for health logs (default: logs/health)
        
    Returns:
        True if successful
    """
    try:
        if health_dir is None:
            health_dir = Path("logs/health")
        
        health_dir.mkdir(parents=True, exist_ok=True)
        
        # Get health data
        health_data = get_system_health()
        
        # Write to daily health file
        date_str = health_data["timestamp"]["date"]
        health_file = health_dir / f"health_{date_str}.jsonl"
        
        success = write_safe(health_file, health_data)
        
        # Cleanup old health files (keep 7 days)
        cleanup_old_logs(health_dir, max_age_days=7)
        
        return success
    except Exception as e:
        logging.debug(f"Failed to write health check: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 3: RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════════

# Rate limiting state
_write_limiter_lock = Lock()
_write_count = 0
_last_reset_time = time.time()
MAX_WRITES_PER_SECOND = 1000  # Conservative limit


def _check_rate_limit() -> None:
    """
    Internal: Enforce write rate limiting.
    Sleeps if rate exceeded.
    """
    global _write_count, _last_reset_time
    
    with _write_limiter_lock:
        current_time = time.time()
        
        # Reset counter if more than 1 second passed
        if current_time - _last_reset_time >= 1.0:
            _write_count = 0
            _last_reset_time = current_time
        
        # If we're over limit, sleep a bit
        if _write_count >= MAX_WRITES_PER_SECOND:
            sleep_time = 0.01  # 10ms
            time.sleep(sleep_time)
            _write_count = 0
            _last_reset_time = time.time()
        
        _write_count += 1


def write_safe_with_limit(path: Path, data: Dict[str, Any], 
                         max_retries: int = 3) -> bool:
    """
    Enhanced write_safe with rate limiting.
    Use this for high-frequency loggers.
    
    Args:
        path: Path to JSONL file
        data: Dict to write as JSON line
        max_retries: Number of retry attempts
        
    Returns:
        True if written, False if failed (but NEVER crashes)
    """
    _check_rate_limit()
    return write_safe(path, data, max_retries)


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 4: OPTIONAL COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def compress_session(session_dir: Path, 
                    delete_original: bool = False,
                    compression_level: int = 6) -> bool:
    """
    Compress an entire session directory to .tar.gz
    Useful for archiving evidence or old sessions.
    
    Args:
        session_dir: Session directory to compress
        delete_original: Whether to delete original after compression
        compression_level: Compression level 1-9 (default: 6)
        
    Returns:
        True if successful
    """
    try:
        if not session_dir.exists():
            return False
        
        # Create tar.gz filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{session_dir.name}_{timestamp}.tar.gz"
        archive_path = session_dir.parent / archive_name
        
        # Create tar.gz
        with tarfile.open(archive_path, "w:gz", compresslevel=compression_level) as tar:
            tar.add(session_dir, arcname=session_dir.name)
        
        # Verify compression worked
        if archive_path.stat().st_size > 0:
            if delete_original:
                shutil.rmtree(session_dir)
            logging.info(f"Compressed session: {session_dir.name} -> {archive_name}")
            return True
        
        return False
        
    except Exception as e:
        logging.error(f"Failed to compress {session_dir}: {e}")
        return False


def compress_old_sessions(logs_dir: Path = None,
                         days_old: int = 7,
                         delete_after_compress: bool = True) -> int:
    """
    Find and compress session directories older than X days.
    
    Args:
        logs_dir: Logs directory (default: Path("logs"))
        days_old: Minimum age in days to compress
        delete_after_compress: Whether to delete original after compression
        
    Returns:
        Number of sessions compressed
    """
    if logs_dir is None:
        logs_dir = Path("logs")
    
    compressed_count = 0
    cutoff_time = datetime.now() - timedelta(days=days_old)
    
    try:
        # Look for session directories
        for session_dir in logs_dir.glob("*_session"):
            if session_dir.is_dir():
                # Check modification time
                mod_time = datetime.fromtimestamp(session_dir.stat().st_mtime)
                
                if mod_time < cutoff_time:
                    # Skip evidence directories
                    if "evidence" not in str(session_dir):
                        if compress_session(session_dir, delete_after_compress):
                            compressed_count += 1
        
        return compressed_count
        
    except Exception as e:
        logging.error(f"Error during batch compression: {e}")
        return compressed_count

