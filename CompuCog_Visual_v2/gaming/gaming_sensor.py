"""
Gaming Domain: Sensor Orchestrator

CORE LOOP: Full 6-stage unified reasoning cycle
  PERCEIVE → REPRESENT → SELECT → APPLY → EVALUATE → UPDATE

Purpose:
  Main daemon that coordinates all gaming operators.
  Captures screen frames, runs detectors, builds fingerprints, logs results.

Components:
  - Frame capture (mss)
  - Grid conversion (NATIVE resolution - NO 32×32 downsampling)
  - Operator orchestration (all 5 detectors)
  - Fingerprint construction
  - JSONL logging

Output:
  gaming/logs/gaming_fingerprint_{date}.jsonl
  One record per second with 6 feature scores + metadata

Usage:
  python gaming_sensor.py [--duration SECONDS] [--smoke]
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import yaml

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from frame_to_grid import FrameCapture, FrameToGrid, FrameGrid

# Import all operators
sys.path.insert(0, str(Path(__file__).parent / "operators"))
from flicker_detector import FlickerDetector, FrameSequence
from hud_stability import HUDStabilityDetector
from crosshair_motion import CrosshairMotionAnalyzer
from peripheral_flash import PeripheralFlashDetector
from color_shift import ColorShiftDetector


class GamingSensor:
    """
    CORE LOOP orchestrator for gaming domain visual reasoning.
    """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        
        # Load master config
        with open(self.config_path, 'r') as f:
            self.master_config = yaml.safe_load(f)
        
        # Load gaming domain config
        gaming_config_path = Path(__file__).parent / "config.yaml"
        with open(gaming_config_path, 'r') as f:
            self.gaming_config = yaml.safe_load(f)
        
        # Check if gaming domain enabled
        if not self.master_config.get("domains", {}).get("gaming", {}).get("enabled", False):
            raise RuntimeError("Gaming domain is disabled in master config")
        
        # PERCEIVE stage: Frame capture
        self.capturer = FrameCapture(self.master_config)
        self.grid_converter = FrameToGrid(self.master_config)
        
        # APPLY stage: Initialize all operators
        print("[*] Initializing gaming operators...")
        self.operators = {
            "flicker": FlickerDetector(self.gaming_config),
            "hud_stability": HUDStabilityDetector(self.gaming_config),
            "crosshair_motion": CrosshairMotionAnalyzer(self.gaming_config),
            "peripheral_flash": PeripheralFlashDetector(self.gaming_config),
            "color_shift": ColorShiftDetector(self.gaming_config)
        }
        
        # Output configuration
        output_config = self.gaming_config.get("output", {})
        self.fingerprint_interval = output_config.get("fingerprint_interval_sec", 1.0)
        self.log_dir = Path(__file__).parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Frame buffer for windowing
        self.frame_buffer: List[FrameGrid] = []
        self.buffer_duration = self.fingerprint_interval
        
        print(f"[+] GamingSensor initialized")
        print(f"    Fingerprint interval: {self.fingerprint_interval}s")
        print(f"    Log directory: {self.log_dir}")
    
    def _build_fingerprint(self, seq: FrameSequence) -> Dict:
        """
        APPLY + EVALUATE stages: Run all operators and build feature vector.
        
        Returns fingerprint dict with 6 feature scores.
        """
        fingerprint = {
            "t_start": seq.t_start,
            "t_end": seq.t_end,
            "frame_count": len(seq.frames),
            "features": {
                "flicker_score": 0.0,
                "hud_instability_score": 0.0,
                "crosshair_anomaly_score": 0.0,
                "peripheral_flash_score": 0.0,
                "color_shift_score": 0.0,
                "overall_anomaly_score": 0.0
            },
            "detections": []
        }
        
        # Run each operator
        for op_name, operator in self.operators.items():
            result = operator.analyze(seq)
            
            if result:
                # Map to feature score
                if op_name == "flicker":
                    fingerprint["features"]["flicker_score"] = result.confidence
                elif op_name == "hud_stability":
                    fingerprint["features"]["hud_instability_score"] = result.confidence
                elif op_name == "crosshair_motion":
                    fingerprint["features"]["crosshair_anomaly_score"] = result.confidence
                elif op_name == "peripheral_flash":
                    fingerprint["features"]["peripheral_flash_score"] = result.confidence
                elif op_name == "color_shift":
                    fingerprint["features"]["color_shift_score"] = result.confidence
                
                # Store full detection
                fingerprint["detections"].append({
                    "operator": op_name,
                    "confidence": result.confidence,
                    "features": result.features
                })
        
        # Compute overall anomaly score (max of all individual scores)
        individual_scores = [
            fingerprint["features"]["flicker_score"],
            fingerprint["features"]["hud_instability_score"],
            fingerprint["features"]["crosshair_anomaly_score"],
            fingerprint["features"]["peripheral_flash_score"],
            fingerprint["features"]["color_shift_score"]
        ]
        fingerprint["features"]["overall_anomaly_score"] = max(individual_scores)
        
        return fingerprint
    
    def _log_fingerprint(self, fingerprint: Dict):
        """
        UPDATE stage: Write fingerprint to JSONL log.
        
        Adds ISO 8601 timestamp for CompuCog schema compliance.
        """
        # Add ISO 8601 timestamp (CompuCog schema requirement)
        fingerprint['timestamp'] = datetime.fromtimestamp(fingerprint['t_start']).isoformat()
        
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"gaming_fingerprint_{date_str}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(fingerprint) + "\n")
    
    def run(self, duration: Optional[float] = None, smoke: bool = False):
        """
        Main CORE LOOP execution.
        
        Args:
          duration: Run for N seconds (None = infinite)
          smoke: Smoke test mode (1 fingerprint then exit)
        """
        print(f"\n[*] Starting GamingSensor (duration={duration}, smoke={smoke})")
        print(f"[*] Capturing at 30 fps, analyzing every {self.fingerprint_interval}s window\n")
        
        start_time = time.time()
        frame_count = 0
        fingerprint_count = 0
        
        try:
            while True:
                loop_start = time.time()
                
                # PERCEIVE: Capture frame
                pil_img = self.capturer.capture()
                
                # REPRESENT: Convert to ARC grid
                frame_grid = self.grid_converter.convert(
                    pil_img,
                    frame_id=frame_count,
                    t_sec=time.time(),
                    source="GamingSensor"
                )
                
                # Add to buffer
                self.frame_buffer.append(frame_grid)
                frame_count += 1
                
                # Check if we have enough frames for analysis
                if len(self.frame_buffer) >= 2:
                    buffer_duration = self.frame_buffer[-1].t_sec - self.frame_buffer[0].t_sec
                    
                    if buffer_duration >= self.buffer_duration:
                        # SELECT: Create frame sequence for operators
                        seq = FrameSequence(
                            frames=list(self.frame_buffer),
                            t_start=self.frame_buffer[0].t_sec,
                            t_end=self.frame_buffer[-1].t_sec,
                            src="GamingSensor"
                        )
                        
                        # APPLY + EVALUATE: Build fingerprint
                        fingerprint = self._build_fingerprint(seq)
                        
                        # UPDATE: Log fingerprint
                        self._log_fingerprint(fingerprint)
                        fingerprint_count += 1
                        
                        # Print summary
                        overall = fingerprint["features"]["overall_anomaly_score"]
                        detection_count = len(fingerprint["detections"])
                        print(f"[{fingerprint_count}] Fingerprint: overall={overall:.3f}, detections={detection_count}, frames={len(seq.frames)}")
                        
                        # Clear buffer
                        self.frame_buffer = []
                        
                        # Exit if smoke test
                        if smoke:
                            print(f"\n[+] Smoke test complete (1 fingerprint)")
                            break
                
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    print(f"\n[+] Duration limit reached ({duration}s)")
                    break
                
                # Sleep to maintain ~30 fps
                elapsed = time.time() - loop_start
                target_interval = 1.0 / 30.0
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
        
        except KeyboardInterrupt:
            print(f"\n[!] Interrupted by user")
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n[*] Session complete:")
        print(f"    Total time: {total_time:.1f}s")
        print(f"    Frames captured: {frame_count}")
        print(f"    Fingerprints logged: {fingerprint_count}")
        print(f"    Avg FPS: {frame_count / total_time:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Gaming Domain Sensor Orchestrator")
    parser.add_argument("--duration", type=float, help="Run for N seconds (default: infinite)")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 1 fingerprint then exit")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "truevision_config.yaml"
    
    sensor = GamingSensor(config_path)
    sensor.run(duration=args.duration, smoke=args.smoke)


if __name__ == "__main__":
    main()
