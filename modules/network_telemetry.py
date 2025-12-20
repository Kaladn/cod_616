"""
Network Telemetry Module
CompuCog Multimodal Game Intelligence Engine

Real-time network metrics:
- RTT (round-trip time)
- Packet loss
- Jitter
- Latency patterns

Built: November 25, 2025
"""

import numpy as np
import time
import subprocess
import re
import platform
from typing import Dict, Optional
from collections import deque


class NetworkTelemetry:
    """
    Real-time network telemetry capture and feature extraction.
    
    Measures RTT, packet loss, jitter for 616 fusion.
    """
    
    def __init__(
        self,
        target_host: str = "8.8.8.8",
        ping_interval_ms: int = 50,
        timeout_ms: int = 1000,
        history_size: int = 100
    ):
        """
        Args:
            target_host: IP/hostname to ping
            ping_interval_ms: Time between pings (ms)
            timeout_ms: Ping timeout (ms)
            history_size: Number of measurements to keep
        """
        self.target_host = target_host
        self.ping_interval_ms = ping_interval_ms
        self.timeout_ms = timeout_ms
        self.history_size = history_size
        self.ping_delay = ping_interval_ms / 1000.0
        
        # Detect OS for ping command
        self.os_type = platform.system()
        
        # RTT history
        self.rtt_history = deque(maxlen=history_size)
        
        # Statistics
        self.pings_sent = 0
        self.pings_received = 0
        self.total_rtt = 0.0
        
        print(f"[616 Network Telemetry]")
        print(f"  Target: {target_host}")
        print(f"  Ping interval: {ping_interval_ms}ms")
        print(f"  Timeout: {timeout_ms}ms")
        print(f"  OS: {self.os_type}")
    
    def ping(self) -> Optional[float]:
        """
        Send single ping and measure RTT.
        
        Returns:
            float: RTT in milliseconds, or None if failed
        """
        try:
            # Build ping command based on OS
            if self.os_type == "Windows":
                # Windows: ping -n 1 -w <timeout_ms> <host>
                cmd = ["ping", "-n", "1", "-w", str(self.timeout_ms), self.target_host]
                
                # Run ping
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_ms / 1000.0 + 0.5
                )
                
                # Parse output (look for "time=XXms")
                match = re.search(r"time[=<](\d+)ms", result.stdout)
                if match:
                    rtt = float(match.group(1))
                    return rtt
                else:
                    return None
            
            else:
                # Linux/Mac: ping -c 1 -W <timeout_sec> <host>
                timeout_sec = self.timeout_ms / 1000.0
                cmd = ["ping", "-c", "1", "-W", str(int(timeout_sec)), self.target_host]
                
                # Run ping
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec + 0.5
                )
                
                # Parse output (look for "time=XX.X ms")
                match = re.search(r"time=([\d.]+)\s*ms", result.stdout)
                if match:
                    rtt = float(match.group(1))
                    return rtt
                else:
                    return None
        
        except (subprocess.TimeoutExpired, Exception) as e:
            return None
    
    def capture(self) -> Dict[str, Optional[float]]:
        """
        Capture network metrics.
        
        Returns:
            Dict with:
                - 'rtt': RTT in ms (None if failed)
                - 'timestamp': Capture timestamp
        """
        start_time = time.perf_counter()
        
        # Send ping
        rtt = self.ping()
        
        # Update statistics
        self.pings_sent += 1
        if rtt is not None:
            self.pings_received += 1
            self.total_rtt += rtt
            self.rtt_history.append(rtt)
        else:
            # Use timeout value for failed pings
            self.rtt_history.append(float(self.timeout_ms))
        
        return {
            'rtt': rtt,
            'timestamp': start_time
        }
    
    def extract_features(self) -> Dict[str, np.ndarray]:
        """
        Capture network state and extract features.
        
        Returns:
            Dict with:
                - 'rtt': Current RTT (or timeout if failed)
                - 'packet_loss': Packet loss percentage
                - 'jitter': RTT standard deviation
                - 'rtt_mean': Mean RTT (last N samples)
                - 'rtt_std': Std RTT (last N samples)
                - 'rtt_max': Max RTT (last N samples)
                - 'rtt_min': Min RTT (last N samples)
                - 'feature_vector': Flattened feature vector (8 features)
                - 'timestamp': Capture timestamp
        """
        # Capture current state
        state = self.capture()
        rtt = state['rtt'] if state['rtt'] is not None else float(self.timeout_ms)
        timestamp = state['timestamp']
        
        # Compute statistics
        if len(self.rtt_history) > 0:
            rtt_mean = np.mean(self.rtt_history)
            rtt_std = np.std(self.rtt_history)
            rtt_max = np.max(self.rtt_history)
            rtt_min = np.min(self.rtt_history)
            jitter = rtt_std  # Jitter = RTT variation
        else:
            rtt_mean = rtt
            rtt_std = 0.0
            rtt_max = rtt
            rtt_min = rtt
            jitter = 0.0
        
        # Packet loss percentage
        if self.pings_sent > 0:
            packet_loss = (1.0 - self.pings_received / self.pings_sent) * 100.0
        else:
            packet_loss = 0.0
        
        # Detect anomalies (RTT spike = >2x mean)
        is_spike = 1.0 if (rtt > 2.0 * rtt_mean) else 0.0
        
        # Build feature vector
        feature_vector = np.array([
            rtt,  # 1
            packet_loss,  # 1
            jitter,  # 1
            rtt_mean,  # 1
            rtt_std,  # 1
            rtt_max,  # 1
            rtt_min,  # 1
            is_spike  # 1
        ], dtype=np.float32)  # Total: 8 features
        
        return {
            'rtt': rtt,
            'packet_loss': packet_loss,
            'jitter': jitter,
            'rtt_mean': rtt_mean,
            'rtt_std': rtt_std,
            'rtt_max': rtt_max,
            'rtt_min': rtt_min,
            'feature_vector': feature_vector,
            'timestamp': timestamp
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get telemetry statistics.
        
        Returns:
            Dict with ping counts, loss rate, average RTT
        """
        if self.pings_sent == 0:
            return {
                'pings_sent': 0,
                'pings_received': 0,
                'packet_loss_pct': 0.0,
                'avg_rtt_ms': 0.0
            }
        
        packet_loss_pct = (1.0 - self.pings_received / self.pings_sent) * 100.0
        avg_rtt = self.total_rtt / self.pings_received if self.pings_received > 0 else 0.0
        
        return {
            'pings_sent': self.pings_sent,
            'pings_received': self.pings_received,
            'packet_loss_pct': packet_loss_pct,
            'avg_rtt_ms': avg_rtt
        }
    
    def close(self):
        """Release resources (none needed for ping)."""
        pass


if __name__ == "__main__":
    """Test network telemetry."""
    
    print("Testing 616 Network Telemetry...")
    print("Press Ctrl+C to quit\n")
    
    # Initialize telemetry
    telemetry = NetworkTelemetry(
        target_host="8.8.8.8",
        ping_interval_ms=100,
        timeout_ms=1000
    )
    
    try:
        while True:
            # Extract features
            features = telemetry.extract_features()
            
            # Print every ping
            if telemetry.pings_sent % 10 == 0:
                stats = telemetry.get_statistics()
                print(f"[Ping {stats['pings_sent']:>5}] "
                      f"RTT: {features['rtt']:>6.1f}ms | "
                      f"Mean: {features['rtt_mean']:>6.1f}ms | "
                      f"Jitter: {features['jitter']:>5.2f}ms | "
                      f"Loss: {stats['packet_loss_pct']:>5.1f}%")
            
            # Rate limiting
            time.sleep(max(0, telemetry.ping_delay - 0.01))
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        stats = telemetry.get_statistics()
        print(f"\n[Final Stats]")
        print(f"  Pings sent: {stats['pings_sent']}")
        print(f"  Pings received: {stats['pings_received']}")
        print(f"  Packet loss: {stats['packet_loss_pct']:.1f}%")
        print(f"  Average RTT: {stats['avg_rtt_ms']:.1f}ms")
        
        telemetry.close()
