"""
Analyze correlations between visual detection and logger telemetry.
Shows input/network/activity patterns during high EOMM windows.
"""

import json
from pathlib import Path
from datetime import datetime

def load_jsonl(path):
    """Load JSONL file into list of dicts"""
    data = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip invalid lines silently
                    continue
    return data

def find_closest_telemetry(target_epoch, telemetry_list, max_delta=10):
    """Find closest telemetry entry within max_delta seconds"""
    closest = None
    min_delta = max_delta
    
    for entry in telemetry_list:
        # Handle different timestamp formats
        entry_epoch = entry.get('timestamp', entry.get('epoch_time', 0))
        
        # If timestamp is ISO string, convert to epoch
        if isinstance(entry_epoch, str):
            try:
                dt = datetime.fromisoformat(entry_epoch.replace('Z', '+00:00'))
                entry_epoch = dt.timestamp()
            except:
                continue
        
        delta = abs(entry_epoch - target_epoch)
        if delta < min_delta:
            min_delta = delta
            closest = entry
    
    return closest, min_delta

def analyze_window_correlations(visual_path, input_path, network_path, activity_path):
    """Analyze correlations for high EOMM windows"""
    
    # Load all telemetry
    print(f"[+] Loading visual telemetry: {visual_path}")
    visual_windows = load_jsonl(visual_path)
    
    print(f"[+] Loading input telemetry: {input_path}")
    input_logs = load_jsonl(input_path)
    
    print(f"[+] Loading network telemetry: {network_path}")
    network_logs = load_jsonl(network_path)
    
    print(f"[+] Loading activity telemetry: {activity_path}")
    activity_logs = load_jsonl(activity_path)
    
    print(f"\n{'='*80}")
    print(f"TELEMETRY CORRELATION ANALYSIS")
    print(f"{'='*80}\n")
    
    # Focus on high EOMM windows (>= 0.8)
    high_eomm_windows = [w for w in visual_windows if w['eomm_composite_score'] >= 0.8]
    
    print(f"Total windows: {len(visual_windows)}")
    print(f"High EOMM windows (>= 0.8): {len(high_eomm_windows)}")
    print(f"Input log entries: {len(input_logs)}")
    print(f"Network log entries: {len(network_logs)}")
    print(f"Activity log entries: {len(activity_logs)}")
    print(f"\n{'='*80}\n")
    
    # Analyze first 10 high EOMM windows
    for i, window in enumerate(high_eomm_windows[:10], 1):
        window_epoch = window['window_start_epoch']
        eomm_score = window['eomm_composite_score']
        flags = window['eomm_flags']
        
        print(f"{'='*80}")
        print(f"High EOMM Window #{i} (EOMM={eomm_score:.2f})")
        print(f"Timestamp: {datetime.fromtimestamp(window_epoch).strftime('%H:%M:%S')}")
        print(f"Flags: {', '.join(flags)}")
        print(f"{'='*80}\n")
        
        # Find closest input telemetry
        input_entry, input_delta = find_closest_telemetry(window_epoch, input_logs)
        if input_entry:
            print(f"[INPUT] (Δ={input_delta:.1f}s)")
            print(f"  Idle seconds: {input_entry.get('idle_seconds', 'N/A')}")
            print(f"  Keystrokes: {input_entry.get('keystroke_count', 'N/A')}")
            print(f"  Mouse clicks: {input_entry.get('mouse_click_count', 'N/A')}")
            print(f"  Mouse movement: {input_entry.get('mouse_movement_distance', 'N/A')}")
            print(f"  Audio active: {input_entry.get('audio_active', 'N/A')}")
            print(f"  Camera active: {input_entry.get('camera_active', 'N/A')}")
        else:
            print(f"[INPUT] No match within 10 seconds")
        
        print()
        
        # Find closest network telemetry (connection snapshot)
        network_entry, network_delta = find_closest_telemetry(window_epoch, network_logs)
        if network_entry:
            print(f"[NETWORK] (Δ={network_delta:.1f}s)")
            print(f"  Connection: {network_entry.get('LocalAddress', 'N/A')}:{network_entry.get('LocalPort', 'N/A')}")
            print(f"  -> {network_entry.get('RemoteAddress', 'N/A')}:{network_entry.get('RemotePort', 'N/A')}")
            print(f"  State: {network_entry.get('State', 'N/A')}")
            print(f"  Process: {network_entry.get('ProcessName', 'N/A')} (PID {network_entry.get('PID', 'N/A')})")
        else:
            print(f"[NETWORK] No match within 10 seconds")
        
        print()
        
        # Find closest activity telemetry
        activity_entry, activity_delta = find_closest_telemetry(window_epoch, activity_logs)
        if activity_entry:
            print(f"[ACTIVITY] (Δ={activity_delta:.1f}s)")
            print(f"  Active window: {activity_entry.get('windowTitle', 'N/A')}")
            print(f"  Process: {activity_entry.get('processName', 'N/A')}")
            print(f"  Idle seconds: {activity_entry.get('idleSeconds', 'N/A')}")
        else:
            print(f"[ACTIVITY] No match within 10 seconds")
        
        print(f"\n")
    
    print(f"{'='*80}")
    print(f"PATTERN SUMMARY")
    print(f"{'='*80}\n")
    
    # Statistical summary
    high_eomm_epochs = [w['window_start_epoch'] for w in high_eomm_windows]
    
    # Check input patterns
    idle_times = []
    for epoch in high_eomm_epochs:
        entry, delta = find_closest_telemetry(epoch, input_logs, max_delta=5)
        if entry and delta < 5:
            idle_times.append(entry.get('idle_seconds', 0))
    
    if idle_times:
        avg_idle = sum(idle_times) / len(idle_times)
        print(f"Average idle time during high EOMM: {avg_idle:.1f}s")
        print(f"  (Shows if user was actively playing or AFK)")
    
    # Check network patterns (connection states)
    connection_states = []
    processes = []
    for epoch in high_eomm_epochs:
        entry, delta = find_closest_telemetry(epoch, network_logs, max_delta=5)
        if entry and delta < 5:
            connection_states.append(entry.get('State', 'Unknown'))
            processes.append(entry.get('ProcessName', 'Unknown'))
    
    if connection_states:
        from collections import Counter
        state_counts = Counter(connection_states)
        process_counts = Counter(processes)
        
        print(f"\nNetwork during high EOMM:")
        print(f"  Connection states: {dict(state_counts)}")
        print(f"  Top processes: {dict(list(process_counts.most_common(3)))}")
    
    # Check active window patterns
    windows = []
    for epoch in high_eomm_epochs:
        entry, delta = find_closest_telemetry(epoch, activity_logs, max_delta=5)
        if entry and delta < 5:
            windows.append(entry.get('windowTitle', 'Unknown'))
    
    if windows:
        from collections import Counter
        window_counts = Counter(windows)
        print(f"\nActive windows during high EOMM:")
        for window, count in window_counts.most_common(5):
            print(f"  {window}: {count} times")
    
    # Check input activity
    total_keys = 0
    total_clicks = 0
    total_movement = 0
    for epoch in high_eomm_epochs:
        entry, delta = find_closest_telemetry(epoch, input_logs, max_delta=5)
        if entry and delta < 5:
            total_keys += entry.get('keystroke_count', 0)
            total_clicks += entry.get('mouse_click_count', 0)
            total_movement += entry.get('mouse_movement_distance', 0)
    
    if total_keys or total_clicks or total_movement:
        print(f"\nInput activity during high EOMM:")
        print(f"  Total keystrokes: {total_keys}")
        print(f"  Total mouse clicks: {total_clicks}")
        print(f"  Total mouse movement: {total_movement:.0f} pixels")
        print(f"  (Shows if user was actively controlling the game)")

if __name__ == "__main__":
    # Get latest files
    visual_file = Path("telemetry/truevision_live_20251202_155530.jsonl")
    input_file = Path("../logs/input/input_activity_20251202.jsonl")
    network_file = Path("../logs/network/telemetry_20251202.jsonl")
    activity_file = Path("../logs/activity/user_activity_20251202.jsonl")
    
    if not all([visual_file.exists(), input_file.exists(), network_file.exists(), activity_file.exists()]):
        print("[ERROR] Missing telemetry files!")
        print(f"  Visual: {visual_file.exists()}")
        print(f"  Input: {input_file.exists()}")
        print(f"  Network: {network_file.exists()}")
        print(f"  Activity: {activity_file.exists()}")
    else:
        analyze_window_correlations(visual_file, input_file, network_file, activity_file)
