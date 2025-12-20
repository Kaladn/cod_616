"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     CompuCog — Sovereign Cognitive Defense System                           ║
║     Intellectual Property of Cortex Evolved / L.A. Mercey                   ║
║                                                                              ║
║     Copyright © 2025 Cortex Evolved. All Rights Reserved.                   ║
║                                                                              ║
║     "We use unconventional digital wisdom —                                  ║
║        because conventional digital wisdom doesn't protect anyone."         ║
║                                                                              ║
║     This software is proprietary and confidential.                           ║
║     Unauthorized access, copying, modification, or distribution             ║
║     is strictly prohibited and may violate applicable laws.                  ║
║                                                                              ║
║     File automatically watermarked on: 2025-12-02 00:00:00                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

TrueVision v1.0.0 - Timed Detection Pipeline

Purpose:
  Run TrueVision detection pipeline for specified duration.
  Captures gameplay, runs all operators, exports telemetry.
  Shows prominent Windows notification when complete.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
import subprocess

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "operators"))
sys.path.insert(0, str(Path(__file__).parent.parent / "compositor"))
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))
sys.path.insert(0, str(Path(__file__).parent.parent / "memory"))
sys.path.insert(0, str(Path(__file__).parent))

from frame_to_grid import FrameCapture, FrameToGrid
from crosshair_lock import CrosshairLockOperator, FrameSequence
from hit_registration import HitRegistrationOperator
from death_event import DeathEventOperator
from edge_entry import EdgeEntryOperator
from eomm_compositor import EommCompositor
from session_baseline import SessionBaselineTracker
from truevision_version import TRUEVISION_VERSION
import yaml

from forge_memory.live_forge_harness import build_truevision_forge_pipeline
from forge_memory.wal.pulse_writer import PulseWriter


def show_windows_notification(title: str, message: str, duration: int = 10):
    """Show Windows 10/11 toast notification with high priority"""
    # PowerShell command for prominent notification
    ps_script = f"""
    Add-Type -AssemblyName System.Windows.Forms
    $notification = New-Object System.Windows.Forms.NotifyIcon
    $notification.Icon = [System.Drawing.SystemIcons]::Information
    $notification.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Info
    $notification.BalloonTipText = "{message}"
    $notification.BalloonTipTitle = "{title}"
    $notification.Visible = $True
    $notification.ShowBalloonTip({duration * 1000})
    Start-Sleep -Seconds {duration}
    $notification.Dispose()
    """
    
    try:
        subprocess.run(["powershell", "-Command", ps_script], 
                      capture_output=True, 
                      timeout=duration + 5)
    except Exception as e:
        print(f"[!] Notification failed: {e}")


def run_timed_pipeline(duration_minutes: int):
    """Run TrueVision pipeline for specified duration"""
    print(f"╔{'═' * 78}╗")
    print(f"║ {'TrueVision v' + TRUEVISION_VERSION + ' - Timed Detection Run':^76} ║")
    print(f"╚{'═' * 78}╝")
    print()
    
    # Calculate end time
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    # Initialize components
    config_path = str(Path(__file__).parent / "truevision_config.yaml")
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("[1/5] Initializing frame capture...")
    capturer = FrameCapture(config)
    converter = FrameToGrid(config)
    
    print("[2/5] Initializing detection operators...")
    crosshair_op = CrosshairLockOperator(config_path)
    hit_reg_op = HitRegistrationOperator(config_path)
    death_op = DeathEventOperator(config_path)
    edge_op = EdgeEntryOperator(config_path)
    
    print("[3/5] Initializing EOMM compositor...")
    compositor = EommCompositor(config_path)
    
    print(f"[4/5] Initializing session baseline tracker...")
    session_tracker = SessionBaselineTracker(min_samples_for_warmup=5)
    
    print("[5/6] Initializing Forge memory pipeline...")
    forge_data_dir = Path(__file__).parent.parent / "memory" / "forge_data"
    forge_data_dir.mkdir(parents=True, exist_ok=True)
    
    pulse_writer: PulseWriter = build_truevision_forge_pipeline(
        data_dir=str(forge_data_dir),
        eomm_threshold=0.5,
        engine_version="truevision_v2.0.0",
        compositor_version="eomm_v2.0.0",
        worker_id=0,
    )
    
    # Frame buffer for 1-second windows
    frame_buffer = []
    frame_buffer_start_time = None
    
    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"live_session_{timestamp}"
    output_dir = Path(__file__).parent / "telemetry"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"truevision_live_{timestamp}.jsonl"
    
    print(f"[6/6] Starting {duration_minutes}-minute detection run...")
    print(f"      Session ID: {session_id}")
    print(f"      Output: {output_file.name}")
    print(f"      Forge data: {forge_data_dir}")
    print(f"      End time: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
    print()
    print("=" * 80)
    print()
    
    window_count = 0
    manipulation_count = 0
    total_flags = []
    
    try:
        with open(output_file, 'w') as f:
            print("      Capturing frames...")
            
            while time.time() < end_time:
                remaining_sec = int(end_time - time.time())
                remaining_min = remaining_sec // 60
                remaining_sec_display = remaining_sec % 60
                
                # Capture frame
                raw_frame = capturer.capture()
                if raw_frame:
                    current_time = time.time()
                    frame_id = int((current_time - start_time) * 30)  # Approximate frame number at 30fps
                    grid_frame = converter.convert(raw_frame, frame_id, current_time, "live_capture")
                    frame_buffer.append(grid_frame)
                    
                    if frame_buffer_start_time is None:
                        frame_buffer_start_time = grid_frame.t_sec
                    
                    # Process when we have 1 second of frames (~30 frames at 30fps)
                    if len(frame_buffer) >= 30:
                        # Create frame sequence
                        seq = FrameSequence(
                            frames=frame_buffer,
                            t_start=frame_buffer_start_time,
                            t_end=grid_frame.t_sec,
                            src="live_capture"
                        )
                        
                        # Run all operators
                        op_results = []
                        
                        result = crosshair_op.analyze(seq)
                        if result:
                            op_results.append(result)
                        
                        result = hit_reg_op.analyze(seq)
                        if result:
                            op_results.append(result)
                        
                        result = death_op.analyze(seq)
                        if result:
                            op_results.append(result)
                        
                        result = edge_op.analyze(seq)
                        if result:
                            op_results.append(result)
                        
                        # Compose telemetry window
                        if op_results:
                            window = compositor.compose_window(
                                operator_results=op_results,
                                window_start_epoch=seq.t_start,
                                window_end_epoch=seq.t_end,
                                session_id=session_id,
                                frame_count=len(seq.frames),
                                session_tracker=session_tracker
                            )
                            
                            # Write to JSONL (legacy)
                            window_dict = window.to_dict()
                            f.write(json.dumps(window_dict) + "\n")
                            f.flush()  # Ensure data is written
                            
                            # Write to Forge memory (permanent)
                            # Translate TrueVision schema to Forge schema
                            forge_window = {
                                "ts_start": window_dict["window_start_epoch"],
                                "ts_end": window_dict["window_end_epoch"],
                                "grid_shape": [32, 32],  # From config
                                "grid_color_count": 10,   # Palette 0-9
                                "operator_scores": {
                                    op["operator_name"]: op["confidence"]
                                    for op in window_dict["operator_results"]
                                },
                                "operator_flags": window_dict.get("eomm_flags", []),
                                "eomm_score": window_dict["eomm_composite_score"],
                                "baseline_stats": window_dict.get("metadata", {}),
                                "telemetry": {
                                    "window_duration_ms": window_dict["window_duration_ms"],
                                    "frame_count": window_dict["frame_count"],
                                },
                                "session_context": {
                                    "session_id": window_dict["session_id"],
                                    "match_id": window_dict["session_id"],
                                    "source": "live",
                                    "map_name": None,
                                    "mode_name": None,
                                    "bot_difficulty": None,
                                    "perspective": "pov",
                                    "run_id": timestamp,
                                },
                            }
                            pulse_writer.submit_window(forge_window)
                            
                            window_count += 1
                            
                            # Track manipulation
                            if window.eomm_composite_score > 0.5:
                                manipulation_count += 1
                            
                            for flag in window.eomm_flags:
                                if flag not in total_flags:
                                    total_flags.append(flag)
                            
                            # Progress update
                            flags_str = ", ".join(window.eomm_flags[:3]) if window.eomm_flags else "NONE"
                            if len(window.eomm_flags) > 3:
                                flags_str += f" +{len(window.eomm_flags) - 3} more"
                            
                            print(f"[{remaining_min:02d}:{remaining_sec_display:02d}] Window {window_count:03d} | "
                                  f"EOMM={window.eomm_composite_score:.2f} | "
                                  f"Flags=[{flags_str}]")
                        
                        # Clear buffer for next window
                        frame_buffer = []
                        frame_buffer_start_time = None
                
                # Brief sleep to avoid CPU hammering (capture at ~30fps)
                time.sleep(1.0 / 30.0)
        
    except KeyboardInterrupt:
        print("\n[!] Detection interrupted by user")
    finally:
        print("\n[→] Flushing Forge memory buffer...")
        pulse_writer.close()
        print("[✓] Forge memory closed")
    
    # Final summary
    print()
    print("=" * 80)
    print("DETECTION RUN COMPLETE")
    print("=" * 80)
    print(f"Duration: {duration_minutes} minutes")
    print(f"Windows analyzed: {window_count}")
    print(f"Manipulation windows (EOMM > 0.5): {manipulation_count} ({manipulation_count/window_count*100:.1f}%)" if window_count > 0 else "No windows analyzed")
    print(f"Unique flags detected: {len(total_flags)}")
    if total_flags:
        print(f"  {', '.join(total_flags)}")
    print(f"Output file: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
    print("=" * 80)
    
    # Show prominent notification
    notification_title = "🎯 TrueVision Detection Complete"
    notification_message = (
        f"{duration_minutes}min run finished\\n"
        f"{window_count} windows analyzed\\n"
        f"{manipulation_count} manipulation events detected\\n"
        f"Output: {output_file.name}"
    )
    
    print("\n[✓] Showing Windows notification...")
    show_windows_notification(notification_title, notification_message, duration=15)
    
    # Also play system sound
    print("\a")  # ASCII bell - plays system sound
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TrueVision detection pipeline for specified duration")
    parser.add_argument("--duration", "-d", type=int, default=15, 
                       help="Duration in minutes (default: 15)")
    
    args = parser.parse_args()
    
    try:
        output_path = run_timed_pipeline(args.duration)
        print(f"\n✅ Detection complete. Telemetry saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
