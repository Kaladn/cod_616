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
║     File automatically watermarked on: 2025-11-29 19:21:12                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

"""

"""
Retroactively add ISO 8601 timestamps to existing gaming fingerprint logs.
Reads log files, adds timestamp field from t_start, writes to same file.
"""
import json
from datetime import datetime
from pathlib import Path

def migrate_log(log_path: Path):
    """Add timestamps to all fingerprints in log file"""
    if not log_path.exists():
        print(f"[!] File not found: {log_path}")
        return
    
    # Read all fingerprints
    fingerprints = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip():
                fp = json.loads(line)
                # Add timestamp if missing
                if 'timestamp' not in fp:
                    fp['timestamp'] = datetime.fromtimestamp(fp['t_start']).isoformat()
                fingerprints.append(fp)
    
    # Overwrite file with updated fingerprints
    with open(log_path, 'w') as f:
        for fp in fingerprints:
            f.write(json.dumps(fp) + "\n")
    
    print(f"[+] Migrated {len(fingerprints)} fingerprints in {log_path.name}")

if __name__ == "__main__":
    logs_dir = Path(__file__).parent / "logs"
    
    # Migrate all gaming_fingerprint_*.jsonl files
    for log_file in logs_dir.glob("gaming_fingerprint_*.jsonl"):
        migrate_log(log_file)
