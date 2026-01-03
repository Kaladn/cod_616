"""
TRUΕVISIΟN LOGGER SYSTEM MONITOR

Simple monitoring dashboard for the logger system.
Shows real-time health status of all loggers and system resources.

Usage:
    python monitor_loggers.py              # Interactive dashboard
    python monitor_loggers.py --once       # Single check, exit
    python monitor_loggers.py --json       # JSON output for automation
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))

from logger_common import get_system_health, get_timestamp


class LoggerMonitor:
    """Real-time monitoring dashboard for logger system."""
    
    def __init__(self, refresh_interval: int = 10):
        self.refresh_interval = refresh_interval
        self.health_history = []
        self.logs_dir = Path("logs")
        
    def get_recent_activity(self, max_files: int = 5) -> list:
        """Get most recently modified log files."""
        recent = []
        
        if not self.logs_dir.exists():
            return recent
        
        try:
            all_logs = list(self.logs_dir.glob("**/*.jsonl"))
            all_logs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for log_file in all_logs[:max_files]:
                try:
                    stat = log_file.stat()
                    age_seconds = time.time() - stat.st_mtime
                    size_kb = stat.st_size / 1024
                    
                    recent.append({
                        "name": log_file.name,
                        "path": str(log_file.relative_to(self.logs_dir)),
                        "age_seconds": age_seconds,
                        "size_kb": size_kb
                    })
                except Exception:
                    continue
        except Exception:
            pass
        
        return recent
    
    def get_session_info(self) -> dict:
        """Get information about current sessions."""
        info = {
            "active_sessions": 0,
            "evidence_count": 0,
            "total_size_mb": 0
        }
        
        if not self.logs_dir.exists():
            return info
        
        try:
            for item in self.logs_dir.iterdir():
                if item.is_dir():
                    if item.name == "evidence":
                        info["evidence_count"] = len(list(item.glob("EVD_*")))
                    elif "_session" in item.name:
                        info["active_sessions"] += 1
            
            # Calculate total size
            total_bytes = sum(f.stat().st_size for f in self.logs_dir.glob("**/*") if f.is_file())
            info["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)
        except Exception:
            pass
        
        return info
    
    def display_dashboard(self):
        """Display real-time monitoring dashboard in terminal."""
        print("Starting TRUΕVISIΟN Logger Monitor...")
        print("Press Ctrl+C to exit\n")
        
        try:
            while True:
                self._render_frame()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    
    def _render_frame(self):
        """Render a single dashboard frame."""
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        health = get_system_health()
        self.health_history.append(health)
        
        # Keep only last 100 samples
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        # Header
        print("═" * 80)
        print("  TRUΕVISIΟN LOGGER SYSTEM MONITOR")
        print("═" * 80)
        print(f"  Last Update: {health['timestamp']['timestamp']}")
        print(f"  Overall Status: {self._status_icon(health['status'])} {health['status'].upper()}")
        print("═" * 80)
        
        # Health checks
        print("\n  SYSTEM HEALTH CHECKS")
        print("  " + "─" * 40)
        
        for check_name, check_data in health.get('checks', {}).items():
            if not isinstance(check_data, dict):
                continue
                
            healthy = check_data.get('healthy', False)
            icon = "✅" if healthy else "❌"
            
            if check_name == "loggers":
                count = check_data.get('count', 0)
                expected = check_data.get('expected', 5)
                print(f"  {icon} LOGGERS    Running: {count}/{expected} expected")
            elif check_name == "disk":
                percent = check_data.get('percent_used', 0)
                free_gb = check_data.get('free_gb', 0)
                print(f"  {icon} DISK       Used: {percent}% ({free_gb} GB free)")
            elif check_name == "memory":
                percent = check_data.get('percent_used', 0)
                avail_gb = check_data.get('available_gb', 0)
                print(f"  {icon} MEMORY     Used: {percent}% ({avail_gb} GB available)")
        
        # Session info
        session_info = self.get_session_info()
        print("\n  SESSION STATUS")
        print("  " + "─" * 40)
        print(f"  📁 Active Sessions:  {session_info['active_sessions']}")
        print(f"  🔒 Evidence Sealed:  {session_info['evidence_count']}")
        print(f"  💾 Total Log Size:   {session_info['total_size_mb']} MB")
        
        # Recent activity
        recent = self.get_recent_activity(5)
        print("\n  RECENT LOG ACTIVITY")
        print("  " + "─" * 40)
        
        if recent:
            for log in recent:
                age = log['age_seconds']
                if age < 60:
                    age_str = f"{age:.0f}s ago"
                elif age < 3600:
                    age_str = f"{age/60:.0f}m ago"
                else:
                    age_str = f"{age/3600:.1f}h ago"
                
                name = log['name'][:35].ljust(35)
                print(f"  📝 {name} {age_str:>10} {log['size_kb']:.1f}KB")
        else:
            print("  No recent log activity")
        
        # Footer
        print("\n" + "═" * 80)
        print(f"  Refresh: {self.refresh_interval}s | History: {len(self.health_history)} samples")
        print("  Press Ctrl+C to exit")
        print("═" * 80)
    
    def _status_icon(self, status: str) -> str:
        """Get icon for status."""
        icons = {
            "healthy": "🟢",
            "degraded": "🟡",
            "unhealthy": "🔴",
            "unknown": "⚪"
        }
        return icons.get(status, "⚪")
    
    def single_check(self, as_json: bool = False) -> dict:
        """Perform a single health check."""
        health = get_system_health()
        health["sessions"] = self.get_session_info()
        health["recent_logs"] = self.get_recent_activity(5)
        
        if as_json:
            print(json.dumps(health, indent=2, default=str))
        else:
            # Simple text output
            status = health['status']
            icon = self._status_icon(status)
            print(f"{icon} System Status: {status.upper()}")
            
            for check_name, check_data in health.get('checks', {}).items():
                if isinstance(check_data, dict):
                    healthy = "✅" if check_data.get('healthy') else "❌"
                    print(f"  {healthy} {check_name}")
        
        return health


def main():
    parser = argparse.ArgumentParser(
        description="TRUΕVISIΟN Logger System Monitor"
    )
    parser.add_argument(
        '--once', '-1',
        action='store_true',
        help='Single health check, then exit'
    )
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output as JSON (implies --once)'
    )
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=10,
        help='Refresh interval in seconds (default: 10)'
    )
    
    args = parser.parse_args()
    
    monitor = LoggerMonitor(refresh_interval=args.interval)
    
    if args.json or args.once:
        monitor.single_check(as_json=args.json)
    else:
        monitor.display_dashboard()


if __name__ == "__main__":
    main()
