"""
CompuCog Organism - Phase 2 Production Entry Point

Live multi-sensor event pipeline with Forge Memory backend.

Architecture:
- 5 sensor adapters → EventManager → Forge BinaryLog
- NO subprocess, NO JSONL, direct API integration
- Single Gateway Pattern (per CONTRACT_ATLAS.md v1.1)

Usage:
    python organism.py --duration 60              # Run for 60 seconds
    python organism.py --forge-dir ./forge_data   # Custom Forge location
    python organism.py --sample-rate 10           # 10Hz polling

Author: Phase 2 Integration
Date: December 2025
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from event_system.chronos_manager import ChronosManager, ChronosMode
from event_system.event_manager import EventManager
from event_system.sensor_registry import SensorRegistry, SensorConfig, SensorType

# Sensor adapters
from event_system.activity_logger_adapter import ActivityLoggerAdapter
from event_system.input_logger_adapter import InputLoggerAdapter
from event_system.process_logger_adapter import ProcessLoggerAdapter
from event_system.network_logger_adapter import NetworkLoggerAdapter
from event_system.gamepad_logger_adapter import GamepadLoggerAdapter

# Forge Memory
import sys
from pathlib import Path
forge_path = Path(__file__).parent.parent / "memory"
if str(forge_path) not in sys.path:
    sys.path.insert(0, str(forge_path))
from forge_memory.core.binary_log import BinaryLog


class CompuCogOrganism:
    """
    Production entry point for CompuCog Phase 2 architecture.
    
    Manages lifecycle of all sensors, EventManager, and Forge writes.
    """
    
    def __init__(self, forge_dir: str, sample_rate_hz: float = 5.0):
        """
        Initialize organism with Forge backend.
        
        Args:
            forge_dir: Path to Forge Memory directory
            sample_rate_hz: Polling rate for all sensors (default 5Hz)
        """
        self.forge_dir = Path(forge_dir)
        self.sample_rate_hz = sample_rate_hz
        self.running = False
        
        # Core components
        self.chronos: Optional[ChronosManager] = None
        self.event_mgr: Optional[EventManager] = None
        self.registry: Optional[SensorRegistry] = None
        self.binary_log: Optional[BinaryLog] = None
        
        # Signal handling for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM for clean shutdown."""
        print("\n[Organism] Shutdown signal received, stopping gracefully...")
        self.stop()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        print("[Organism] Initializing CompuCog Phase 2 architecture...")
        
        # Step 1: Initialize ChronosManager (LIVE mode)
        print("  [1/5] ChronosManager (LIVE mode)...", end=" ")
        self.chronos = ChronosManager()
        self.chronos.initialize(mode=ChronosMode.LIVE)
        print("✅")
        
        # Step 2: Initialize Forge Memory (BinaryLog + StringDict)
        print(f"  [2/5] Forge Memory ({self.forge_dir})...", end=" ")
        self.forge_dir.mkdir(parents=True, exist_ok=True)
        
        from forge_memory.core.string_dict import StringDict
        string_dict = StringDict(str(self.forge_dir))
        self.binary_log = BinaryLog(str(self.forge_dir), string_dict)
        print("✅")
        
        # Step 3: Initialize EventManager (with Forge gateway)
        print("  [3/5] EventManager → Forge Gateway...", end=" ")
        self.event_mgr = EventManager(
            chronos_manager=self.chronos,
            binary_log=self.binary_log
        )
        print("✅")
        
        # Step 4: Initialize SensorRegistry
        print("  [4/5] SensorRegistry...", end=" ")
        self.registry = SensorRegistry(
            chronos=self.chronos,
            event_mgr=self.event_mgr
        )
        print("✅")
        
        # Step 5: Register all sensor adapters
        print("  [5/5] Registering sensor adapters...")
        success_count = self._register_all_adapters()
        print(f"        → {success_count}/5 adapters registered ✅")
        
        print("\n[Organism] ✅ Initialization complete\n")
        return success_count > 0
    
    def _register_all_adapters(self) -> int:
        """
        Register all 5 sensor adapters.
        
        Returns:
            Number of successfully registered adapters
        """
        success_count = 0
        
        # 1. Activity Monitor (window tracking + idle detection)
        try:
            config = SensorConfig(
                sensor_type=SensorType.ACTIVITY_MONITOR,
                source_id="activity_monitor",
                sample_rate_hz=self.sample_rate_hz,
                tags=["activity", "window"]
            )
            adapter = ActivityLoggerAdapter(config, self.chronos, self.event_mgr)
            if self.registry.register_sensor(adapter):
                success_count += 1
                print(f"        ✅ activity_monitor")
        except Exception as e:
            print(f"        ❌ activity_monitor: {e}")
        
        # 2. Input Monitor (keyboard/mouse via idle detection)
        try:
            config = SensorConfig(
                sensor_type=SensorType.KEYBOARD_INPUT,
                source_id="input_monitor",
                sample_rate_hz=self.sample_rate_hz,
                tags=["input", "keyboard", "mouse"]
            )
            adapter = InputLoggerAdapter(config, self.chronos, self.event_mgr)
            if self.registry.register_sensor(adapter):
                success_count += 1
                print(f"        ✅ input_monitor")
        except Exception as e:
            print(f"        ❌ input_monitor: {e}")
        
        # 3. Process Monitor (process spawn/exit tracking)
        try:
            config = SensorConfig(
                sensor_type=SensorType.PROCESS_MONITOR,
                source_id="process_monitor",
                sample_rate_hz=self.sample_rate_hz,
                tags=["process", "system"]
            )
            adapter = ProcessLoggerAdapter(config, self.chronos, self.event_mgr)
            if self.registry.register_sensor(adapter):
                success_count += 1
                print(f"        ✅ process_monitor")
        except Exception as e:
            print(f"        ❌ process_monitor: {e}")
        
        # 4. Network Monitor (TCP/UDP connection tracking)
        try:
            config = SensorConfig(
                sensor_type=SensorType.NETWORK_TRAFFIC,
                source_id="network_monitor",
                sample_rate_hz=self.sample_rate_hz,
                tags=["network", "connections"]
            )
            adapter = NetworkLoggerAdapter(config, self.chronos, self.event_mgr)
            if self.registry.register_sensor(adapter):
                success_count += 1
                print(f"        ✅ network_monitor")
        except Exception as e:
            print(f"        ❌ network_monitor: {e}")
        
        # 5. Gamepad Monitor (controller input tracking)
        try:
            config = SensorConfig(
                sensor_type=SensorType.GAMEPAD_INPUT,
                source_id="gamepad_monitor",
                sample_rate_hz=self.sample_rate_hz,
                tags=["gamepad", "controller"]
            )
            adapter = GamepadLoggerAdapter(config, self.chronos, self.event_mgr)
            if self.registry.register_sensor(adapter):
                success_count += 1
                print(f"        ✅ gamepad_monitor")
        except Exception as e:
            print(f"        ⚠️  gamepad_monitor: {e}")
        
        return success_count
    
    def start(self) -> bool:
        """
        Start all registered sensors.
        
        Returns:
            True if at least one sensor started, False otherwise
        """
        print("[Organism] Starting all sensors...")
        results = self.registry.start_all_sensors()
        
        started = sum(1 for success in results.values() if success)
        print(f"[Organism] ✅ {started}/{len(results)} sensors started\n")
        
        self.running = True
        return started > 0
    
    def run(self, duration_seconds: Optional[float] = None) -> None:
        """
        Main event loop - poll sensors and record events.
        
        Args:
            duration_seconds: Run duration (None = infinite)
        """
        start_time = time.time()
        poll_count = 0
        
        print("="*80)
        print("  🧠 COMPUCOG ORGANISM LIVE")
        print("="*80)
        print(f"  Sample rate: {self.sample_rate_hz} Hz")
        print(f"  Duration: {duration_seconds}s" if duration_seconds else "  Duration: Infinite (Ctrl+C to stop)")
        print(f"  Forge: {self.forge_dir}")
        print("="*80)
        print()
        
        try:
            while self.running:
                # Poll all sensors
                self.registry.poll_sensors()
                poll_count += 1
                
                # Print stats every 50 polls
                if poll_count % 50 == 0:
                    stats = self.registry.get_sensor_stats()
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.1f}s] Events: {stats['total_events']} | "
                          f"Active sensors: {stats['active_sensors']}")
                
                # Check duration limit
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    print(f"\n[Organism] Duration limit reached ({duration_seconds}s)")
                    break
                
                # Sleep to maintain target sample rate
                time.sleep(1.0 / self.sample_rate_hz)
                
        except KeyboardInterrupt:
            print("\n[Organism] Interrupted by user")
        except Exception as e:
            print(f"\n[Organism] ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self) -> None:
        """Stop all sensors and print final stats."""
        if not self.running:
            return
        
        self.running = False
        
        print("\n[Organism] Stopping all sensors...")
        self.registry.stop_all_sensors()
        
        # Print final statistics
        print("\n" + "="*80)
        print("  📊 FINAL STATISTICS")
        print("="*80)
        
        sensor_stats = self.registry.get_sensor_stats()
        print(f"  Total events captured: {sensor_stats['total_events']}")
        print(f"\n  Events per sensor:")
        for source_id, count in sensor_stats['events_per_sensor'].items():
            print(f"    {source_id:20s}: {count:6d} events")
        
        event_stats = self.event_mgr.get_stats()
        print(f"\n  EventManager:")
        print(f"    Total events: {event_stats['total_events']}")
        print(f"    Total streams: {event_stats['total_streams']}")
        print(f"    Total chains: {event_stats['total_chains']}")
        
        print("\n  Forge Memory:")
        print(f"    Location: {self.forge_dir}")
        print(f"    BinaryLog size: {self.binary_log.tell()} bytes")
        
        print("="*80)
        print()


def main():
    """Production entry point."""
    parser = argparse.ArgumentParser(
        description="CompuCog Organism - Phase 2 Multi-Sensor Event Pipeline"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run duration in seconds (default: infinite)"
    )
    
    parser.add_argument(
        "--forge-dir",
        type=str,
        default="./forge_data",
        help="Forge Memory directory (default: ./forge_data)"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=5.0,
        help="Sensor polling rate in Hz (default: 5.0)"
    )
    
    args = parser.parse_args()
    
    # Create and run organism
    organism = CompuCogOrganism(
        forge_dir=args.forge_dir,
        sample_rate_hz=args.sample_rate
    )
    
    if not organism.initialize():
        print("[Organism] ❌ Initialization failed")
        sys.exit(1)
    
    if not organism.start():
        print("[Organism] ❌ Failed to start sensors")
        sys.exit(1)
    
    organism.run(duration_seconds=args.duration)
    organism.stop()
    
    print("[Organism] ✅ Shutdown complete")


if __name__ == "__main__":
    main()
