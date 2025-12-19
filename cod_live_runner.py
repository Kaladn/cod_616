"""
COD 616 Live Match Runner
CompuCog Multimodal Game Intelligence Engine

Real-time multimodal capture and 616 fusion during COD matches.

Modes:
- baseline: Capture bot lobby (manipulation-free reference)
- real: Capture real match (detect manipulation)
- compare: Compare baseline vs real signatures

Built: November 25, 2025
"""

import numpy as np
import time
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Import 616 modules
import sys
sys.path.append(str(Path(__file__).parent))

from modules.screen_grid_mapper import ScreenGridMapper
from modules.yolo_detector import YOLODetector
from modules.gamepad_capture import GamepadCapture
from modules.network_telemetry import NetworkTelemetry
from modules.fusion_616_engine import Fusion616Engine
from match_fingerprint_builder import MatchFingerprintBuilder


class COD616Runner:
    """
    Real-time COD match capture and 616 analysis.
    
    Captures all modalities and fuses into 616 signature.
    """
    
    def __init__(self, config_path: str = "cod_616/config_616.yaml"):
        """
        Args:
            config_path: Path to configuration file
        """
        # Load config and extract active profile
        with open(config_path, 'r') as f:
            config_full = yaml.safe_load(f)
        
        active_profile = config_full.get('active_profile', 'play_nice')
        self.config = config_full['profiles'][active_profile]
        self.config['output'] = config_full['output']
        self.config['logging'] = config_full['logging']
        
        print(f"[616 COD Runner]")
        print(f"  Config: {config_path}")
        print(f"  Active profile: {active_profile.upper()}\n")
        
        # Initialize modules
        self.screen_mapper = None
        self.yolo_detector = None
        self.gamepad_capture = None
        self.network_telemetry = None
        self.fusion_engine = None
        
        self._init_modules()
        
        # Telemetry storage
        self.telemetry_data = []
        self.save_interval = self.config['output']['save_interval_sec']
        self.last_save_time = time.time()
        
        # Phase 2: Match fingerprint builder
        self.fingerprint_builder = MatchFingerprintBuilder()
        
        print(f"\n✓ 616 COD Runner initialized\n")
    
    def _init_modules(self):
        """Initialize all capture modules."""
        
        # Screen grid mapper
        screen_cfg = self.config['screen']
        capture_res = screen_cfg.get('capture_resolution')
        self.screen_mapper = ScreenGridMapper(
            grid_size=tuple(screen_cfg['grid_size']),
            block_size=tuple(screen_cfg['block_size']),
            monitor=screen_cfg['monitor'],
            capture_fps=screen_cfg['capture_fps'],
            capture_resolution=tuple(capture_res) if capture_res else None
        )
        print()
        
        # YOLO detector
        yolo_cfg = self.config['yolo']
        if yolo_cfg.get('enabled', False):
            try:
                input_res = yolo_cfg.get('input_resolution')
                self.yolo_detector = YOLODetector(
                    model_path=yolo_cfg['model'],
                    confidence=yolo_cfg['confidence'],
                    iou=yolo_cfg['iou'],
                    device=yolo_cfg['device'],
                    classes=yolo_cfg['classes'],
                    skip_frames=yolo_cfg.get('skip_frames', 1),
                    input_resolution=tuple(input_res) if input_res else None
                )
                print()
            except Exception as e:
                print(f"[WARNING] YOLO detector failed to initialize: {e}")
                print(f"[INFO] Continuing without YOLO features\n")
                self.yolo_detector = None
        else:
            print("[INFO] YOLO disabled by config\n")
            self.yolo_detector = None
        
        # Gamepad capture
        gamepad_cfg = self.config['gamepad']
        if gamepad_cfg['enabled']:
            try:
                self.gamepad_capture = GamepadCapture(
                    poll_rate_hz=gamepad_cfg['poll_rate_hz'],
                    deadzone=gamepad_cfg['deadzone']
                )
                print()
            except Exception as e:
                print(f"[WARNING] Gamepad capture failed to initialize: {e}")
                print(f"[INFO] Continuing without gamepad features\n")
                self.gamepad_capture = None
        
        # Network telemetry
        network_cfg = self.config['network']
        if network_cfg['enabled']:
            self.network_telemetry = NetworkTelemetry(
                target_host=network_cfg['target_host'],
                ping_interval_ms=network_cfg['ping_interval_ms'],
                timeout_ms=network_cfg['timeout_ms']
            )
            print()
        
        # 616 Fusion Engine
        fusion_cfg = self.config['fusion_616']
        self.fusion_engine = Fusion616Engine(
            anchor_frequencies=fusion_cfg['anchor_frequencies'],
            window_size_ms=fusion_cfg['window_size_ms'],
            overlap_ms=fusion_cfg['overlap_ms']
        )
        print()

        # Audio capture and processing
        audio_cfg = self.config.get('audio', {'enabled': False})
        if audio_cfg.get('enabled', False):
            try:
                self.audio_capture = AudioCapture(
                    sample_rate=audio_cfg.get('sample_rate', 48000),
                    channels=audio_cfg.get('channels', 1),
                    block_duration=audio_cfg.get('block_duration', 0.5)
                )
                self.audio_resonance_state = AudioResonanceState(
                    sample_rate=audio_cfg.get('sample_rate', 48000),
                    block_duration=audio_cfg.get('block_duration', 0.5)
                )
                print()
            except Exception as e:
                print(f"[WARNING] Audio capture failed to initialize: {e}")
                print(f"[INFO] Continuing without audio features\n")
                self.audio_capture = None
                self.audio_resonance_state = None
        else:
            self.audio_capture = None
            self.audio_resonance_state = None
            print("[INFO] Audio capture disabled by config\n")
    
    def capture_frame(self) -> Dict:
        """
        Capture one multimodal frame.
        
        Returns:
            Dict with all features and fusion result
        """
        # Screen
        screen_features_dict = self.screen_mapper.extract_features()
        screen_features = screen_features_dict['block_vector']
        
        # YOLO (on screen frame)
        if self.yolo_detector is not None:
            yolo_detections = self.yolo_detector.detect(screen_features_dict['frame'])
            yolo_features_dict = self.yolo_detector.extract_features(
                yolo_detections, screen_features_dict['frame'].shape[:2]
            )
            yolo_features = yolo_features_dict['feature_vector']
        else:
            yolo_features = None
        
        # Gamepad
        if self.gamepad_capture is not None:
            gamepad_features_dict = self.gamepad_capture.extract_features()
            gamepad_features = gamepad_features_dict['feature_vector']
        else:
            gamepad_features = None
        
        # Network (non-blocking - use cached values)
        if self.network_telemetry is not None:
            network_features_dict = self.network_telemetry.extract_features()
            network_features = network_features_dict['feature_vector']
        else:
            network_features = None
        
        # Audio: capture and compute audio resonance (if available)
        audio_resonance = None
        if self.audio_capture is not None and self.audio_resonance_state is not None:
            audio_block = self.audio_capture.get_block()
            if audio_block is not None:
                audio_resonance = self.audio_resonance_state.update(audio_block)
            else:
                # Use silence profile when no block available (keeps consistent dims)
                audio_resonance = self.audio_resonance_state._get_silence_state()

        # Fuse (include audio_resonance)
        fusion_result = self.fusion_engine.fuse(
            screen_features=screen_features,
            visual_resonance=screen_features_dict.get('visual_resonance', {}),
            audio_resonance=audio_resonance,
            gamepad_features=gamepad_features,
            network_features=network_features
        )

        # Build frame dict for telemetry and fingerprint
        frame_data = {
            'screen': screen_features_dict,
            'yolo': yolo_features if yolo_features is not None else None,
            'gamepad': gamepad_features if gamepad_features is not None else None,
            'network': network_features if network_features is not None else None,
            'audio_resonance': audio_resonance,
            'fusion': fusion_result,
            'timestamp': time.time()
        }

        # Phase 2: Update fingerprint builder (accumulate match signature)
        fused_frame_for_builder = {
            'timestamp': frame_data['timestamp'],
            'visual_resonance': screen_features_dict.get('visual_resonance', {}),
            'gamepad': gamepad_features_dict if self.gamepad_capture is not None else {},
            'network': network_features_dict if self.network_telemetry is not None else {},
            'audio_resonance': audio_resonance if audio_resonance is not None else {}
        }
        self.fingerprint_builder.update(fused_frame_for_builder)

        return frame_data
    
    def run_capture(self, duration_sec: int = None, mode: str = "baseline"):
        """
        Run live capture for specified duration or until Ctrl+C.
        
        Args:
            duration_sec: Capture duration (seconds), None = infinite
            mode: 'baseline' or 'real'
        """
        print(f"[616 COD Runner - {mode.upper()} MODE]")
        if duration_sec is not None:
            print(f"  Duration: {duration_sec}s")
        else:
            print(f"  Duration: Until Ctrl+C (open-ended)")
        print(f"  Starting in 3 seconds...\n")
        time.sleep(3)
        
        # Phase 2: Reset fingerprint builder for new match
        self.fingerprint_builder.reset()
        
        print(f"✓ CAPTURE STARTED (Press Ctrl+C to stop)\n")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                # Check duration if specified
                if duration_sec is not None and (time.time() - start_time) >= duration_sec:
                    break
                # Capture frame
                frame_data = self.capture_frame()
                
                # Store telemetry
                if self.config['output']['save_telemetry']:
                    self.telemetry_data.append({
                        'timestamp': frame_data['timestamp'],
                        'manipulation_score': float(frame_data['fusion']['manipulation_score']),
                        'resonance_phases': frame_data['fusion']['resonance_vector'][:6].tolist(),
                        'screen_energy': float(np.sum(frame_data['screen']['block_vector'])),
                        'yolo_count': int(frame_data['yolo'][0]) if frame_data['yolo'] is not None else 0,
                        'gamepad_buttons': int(frame_data['gamepad'][48]) if frame_data['gamepad'] is not None else 0,
                        'network_rtt': float(frame_data['network'][0]) if frame_data['network'] is not None else 0.0
                    })
                
                # Save periodically
                if time.time() - self.last_save_time > self.save_interval:
                    self._save_telemetry(mode)
                    self.last_save_time = time.time()
                
                # Print status
                frame_count += 1
                if frame_count % 60 == 0:
                    elapsed = time.time() - start_time
                    if duration_sec is not None:
                        remaining = duration_sec - elapsed
                        print(f"[{elapsed:>6.1f}s / {duration_sec}s] "
                              f"Frame {frame_count:>5} | "
                              f"Manipulation: {frame_data['fusion']['manipulation_score']:.3f} | "
                              f"FPS: {frame_count / elapsed:.1f}")
                    else:
                        print(f"[{elapsed:>6.1f}s] "
                              f"Frame {frame_count:>5} | "
                              f"Manipulation: {frame_data['fusion']['manipulation_score']:.3f} | "
                              f"FPS: {frame_count / elapsed:.1f}")
        
        except KeyboardInterrupt:
            print("\n\n[STOPPED BY USER]")
        
        finally:
            # Final save
            self._save_telemetry(mode)
            
            # Phase 2: Build and save match fingerprint
            self._save_fingerprint(mode, frame_count, time.time() - start_time)
            
            # Print statistics
            print(f"\n[FINAL STATS]")
            print(f"  Frames captured: {frame_count}")
            print(f"  Duration: {time.time() - start_time:.1f}s")
            print(f"  Average FPS: {frame_count / (time.time() - start_time):.1f}")
            
            screen_stats = self.screen_mapper.get_statistics()
            print(f"  Screen FPS: {screen_stats['avg_fps']:.1f}")
            
            if self.yolo_detector is not None:
                yolo_stats = self.yolo_detector.get_statistics()
                print(f"  YOLO FPS: {yolo_stats['avg_fps']:.1f}")
            
            fusion_stats = self.fusion_engine.get_statistics()
            print(f"  Fusion FPS: {fusion_stats['avg_fps']:.1f}")
    
    def _save_telemetry(self, mode: str):
        """Save telemetry data to file."""
        if not self.telemetry_data:
            return
        
        # Create output directory
        output_dir = Path(self.config['output']['telemetry_dir']) / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"telemetry_{timestamp}.json"
        
        # Save
        with open(filename, 'w') as f:
            json.dump(self.telemetry_data, f, indent=2)
        
        print(f"  ✓ Saved telemetry: {filename} ({len(self.telemetry_data)} frames)")
    
    def _save_fingerprint(self, mode: str, frame_count: int, duration: float):
        """Build and save 365-dim match fingerprint."""
        if frame_count == 0:
            return
        
        # Build fingerprint
        fingerprint = self.fingerprint_builder.build()
        
        # Add metadata
        fingerprint['mode'] = mode
        fingerprint['profile'] = self.config.get('profile_name', 'unknown')
        fingerprint['capture_timestamp'] = datetime.now().isoformat()
        
        # Create output directory
        output_dir = Path(self.config['output']['telemetry_dir']) / 'fingerprints' / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"fingerprint_{timestamp}.json"
        
        # Save
        # Convert numpy arrays to lists for JSON serialization
        fingerprint_json = {
            'vector': fingerprint['vector'].tolist(),
            'layout_version': fingerprint['layout_version'],
            'frame_count': fingerprint['frame_count'],
            'duration': fingerprint['duration'],
            'meta': fingerprint['meta'],
            'mode': fingerprint['mode'],
            'profile': fingerprint['profile'],
            'capture_timestamp': fingerprint['capture_timestamp']
        }
        
        with open(filename, 'w') as f:
            json.dump(fingerprint_json, f, indent=2)
        
        print(f"  ✓ Saved fingerprint: {filename} (365 dims)")
        print(f"    Anomaly scores: visual={fingerprint['meta']['visual_anomaly_fraction']:.3f}, "
              f"input={fingerprint['meta']['input_anomaly_fraction']:.3f}, "
              f"network={fingerprint['meta']['network_anomaly_fraction']:.3f}")
        print(f"    Suspect scores: aim={fingerprint['meta']['aim_assist_suspect_score']:.3f}, "
              f"recoil={fingerprint['meta']['recoil_compensation_suspect_score']:.3f}")
    
    def compare_modes(self):
        """Compare baseline vs real match signatures."""
        print(f"[616 COMPARISON MODE]")
        print(f"  Loading baseline and real telemetry...\n")
        
        baseline_dir = Path(self.config['output']['telemetry_dir']) / 'baseline'
        real_dir = Path(self.config['output']['telemetry_dir']) / 'real'
        
        # Load latest files
        baseline_files = sorted(baseline_dir.glob('telemetry_*.json'))
        real_files = sorted(real_dir.glob('telemetry_*.json'))
        
        if not baseline_files:
            print(f"  [ERROR] No baseline telemetry found")
            return
        
        if not real_files:
            print(f"  [ERROR] No real telemetry found")
            return
        
        # Load data
        with open(baseline_files[-1], 'r') as f:
            baseline_data = json.load(f)
        
        with open(real_files[-1], 'r') as f:
            real_data = json.load(f)
        
        print(f"  Baseline: {baseline_files[-1].name} ({len(baseline_data)} frames)")
        print(f"  Real: {real_files[-1].name} ({len(real_data)} frames)\n")
        
        # Compare manipulation scores
        baseline_scores = [d['manipulation_score'] for d in baseline_data]
        real_scores = [d['manipulation_score'] for d in real_data]
        
        print(f"[MANIPULATION SCORES]")
        print(f"  Baseline: mean={np.mean(baseline_scores):.3f}, std={np.std(baseline_scores):.3f}, max={np.max(baseline_scores):.3f}")
        print(f"  Real:     mean={np.mean(real_scores):.3f}, std={np.std(real_scores):.3f}, max={np.max(real_scores):.3f}")
        
        # Detect anomalies (real >> baseline)
        threshold = np.mean(baseline_scores) + 2 * np.std(baseline_scores)
        anomalies = [s for s in real_scores if s > threshold]
        
        print(f"\n[ANOMALY DETECTION]")
        print(f"  Threshold: {threshold:.3f}")
        print(f"  Anomalies detected: {len(anomalies)} / {len(real_scores)} frames ({len(anomalies) / len(real_scores) * 100:.1f}%)")
        
        if anomalies:
            print(f"  Max anomaly score: {max(anomalies):.3f}")
    
    def close(self):
        """Release all resources."""
        if self.screen_mapper is not None:
            self.screen_mapper.close()
        
        if self.gamepad_capture is not None:
            self.gamepad_capture.close()
        
        if self.network_telemetry is not None:
            self.network_telemetry.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="616 COD Live Match Runner")
    parser.add_argument('--mode', type=str, choices=['baseline', 'real', 'compare'], required=True,
                        help='Capture mode: baseline (bot lobby), real (live match), compare (analysis)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Capture duration in seconds (default: None = until Ctrl+C)')
    parser.add_argument('--config', type=str, default='cod_616/config_616.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = COD616Runner(config_path=args.config)
    
    try:
        if args.mode == 'compare':
            runner.compare_modes()
        else:
            runner.run_capture(duration_sec=args.duration, mode=args.mode)
    
    finally:
        runner.close()


if __name__ == "__main__":
    main()
