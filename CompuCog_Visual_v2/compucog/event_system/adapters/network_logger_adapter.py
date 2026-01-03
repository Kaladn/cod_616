"""
NetworkLoggerAdapter - Direct Network Connection Monitoring
Monitors TCP/UDP connections and converts to Event objects per CONTRACT_ATLAS.md

Per Phase 2 Architecture:
- NO subprocess launches (no PowerShell script)
- NO JSONL file writes
- Direct data collection via psutil.net_connections()
- Returns Event objects → EventManager → Forge
"""

from typing import Dict, Any, Optional, Set, Tuple
import logging
import psutil

from event_system.sensor_registry import SensorAdapter, SensorConfig
from event_system.event_manager import Event
from event_system.chronos_manager import ChronosManager


class NetworkLoggerAdapter(SensorAdapter):
    """
    Direct network connection monitoring adapter.
    
    Per CONTRACT_ATLAS.md:
    - Monitors TCP/UDP connections with process info
    - Tracks new connections and state changes
    - Converts to Event objects (NEVER dicts or ForgeRecords)
    - EventManager writes Events → Forge via BinaryLog
    - source_id: "network_monitor"
    - tags: ["network", "tcp", "udp"]
    """
    
    def __init__(self, config: SensorConfig, chronos: ChronosManager, event_mgr):
        super().__init__(config, chronos, event_mgr)
        self.tracked_connections: Set[Tuple] = set()  # (laddr, raddr, status, pid)
        
        # Setup logging
        self.logger = logging.getLogger(f"NetworkAdapter[{config.source_id}]")
        
    def initialize(self) -> bool:
        """Initialize network monitoring."""
        try:
            # Get initial connection snapshot
            for conn in psutil.net_connections(kind='inet'):
                try:
                    if conn.raddr:
                        conn_key = (
                            f"{conn.laddr.ip}:{conn.laddr.port}",
                            f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "",
                            conn.status,
                            conn.pid
                        )
                        self.tracked_connections.add(conn_key)
                except (AttributeError, TypeError):
                    pass
            
            self.logger.info(f"NetworkLoggerAdapter initialized (tracking {len(self.tracked_connections)} connections)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NetworkLoggerAdapter: {e}")
            return False
    
    def start(self) -> bool:
        """Start monitoring (no subprocess needed)."""
        try:
            self.running = True
            self.logger.info("Network monitoring started (direct psutil mode)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start network monitoring: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop monitoring (no cleanup needed)."""
        try:
            self.running = False
            self.logger.info("Network monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop network monitoring: {e}")
            return False
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """
        Scan for new network connections or state changes.
        
        Returns:
            Dict with connection data, or None if no changes
        """
        try:
            current_connections = set()
            new_connections = []
            
            # Scan current connections
            for conn in psutil.net_connections(kind='inet'):
                try:
                    # Only track established connections with remote address
                    if not conn.raddr or conn.raddr.ip in ('0.0.0.0', '::', '127.0.0.1', '::1'):
                        continue
                    
                    conn_key = (
                        f"{conn.laddr.ip}:{conn.laddr.port}",
                        f"{conn.raddr.ip}:{conn.raddr.port}",
                        conn.status,
                        conn.pid
                    )
                    
                    current_connections.add(conn_key)
                    
                    # Detect new connection
                    if conn_key not in self.tracked_connections:
                        # Get process name
                        process_name = "unknown"
                        try:
                            if conn.pid:
                                proc = psutil.Process(conn.pid)
                                process_name = proc.name()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        
                        new_connections.append({
                            'local_address': conn.laddr.ip,
                            'local_port': conn.laddr.port,
                            'remote_address': conn.raddr.ip,
                            'remote_port': conn.raddr.port,
                            'status': conn.status,
                            'protocol': conn.type.name if hasattr(conn.type, 'name') else str(conn.type),
                            'pid': conn.pid or 0,
                            'process_name': process_name,
                            'family': conn.family.name if hasattr(conn.family, 'name') else str(conn.family)
                        })
                        
                except (AttributeError, TypeError, psutil.AccessDenied):
                    pass
            
            # Update tracking
            self.tracked_connections = current_connections
            
            # Return first new connection (if any)
            if new_connections:
                return new_connections[0]  # Report one at a time
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scanning network connections: {e}")
            return None
    
    def convert_to_event(self, data: Dict[str, Any]) -> Event:
        """
        Convert network data to Event object per CONTRACT_ATLAS.md.
        
        Per CONTRACT_ATLAS.md Contract #2:
        - MUST return Event object (NEVER dict or ForgeRecord)
        - timestamp from ChronosManager (deterministic)
        - source_id from config
        - tags include sensor type
        - metadata is JSON-serializable
        
        Args:
            data: Dict from get_latest_data()
        
        Returns:
            Event object (validated by SensorAdapter.record_event)
        """
        # Get deterministic timestamp from ChronosManager
        timestamp = self.chronos.now()
        
        # Build Event object per CONTRACT_ATLAS.md
        return Event(
            event_id=f"network_{int(timestamp * 1000)}",
            timestamp=timestamp,
            source_id=self.config.source_id,
            tags=self.config.tags + ["network", "connection", data.get('protocol', 'tcp').lower()],
            metadata={
                "local_address": data.get("local_address", ""),
                "local_port": data.get("local_port", 0),
                "remote_address": data.get("remote_address", ""),
                "remote_port": data.get("remote_port", 0),
                "status": data.get("status", ""),
                "protocol": data.get("protocol", ""),
                "pid": data.get("pid", 0),
                "process_name": data.get("process_name", "unknown"),
                "family": data.get("family", "")
            }
        )
