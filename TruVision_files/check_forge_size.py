"""Quick script to check actual data written to Forge Memory"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "memory"))

from forge_memory.core.mmap_file import MMapFile

forge_data = Path(__file__).parent / "forge_data"
records_path = forge_data / "records.bin"

if not records_path.exists():
    print("❌ records.bin not found")
    sys.exit(1)

# Open the mmap file
mmap_file = MMapFile(str(records_path), initial_size=1024*1024*1024, mode="r")

# Read first 4 bytes to check for records
first_bytes = mmap_file.read_bytes(0, 4)
magic = int.from_bytes(first_bytes, 'little')

print(f"🔍 Forge Memory Analysis")
print(f"=" * 60)
print(f"File size on disk: {records_path.stat().st_size / (1024*1024):.2f} MB")
print(f"File size (allocated): {mmap_file.size / (1024*1024):.2f} MB")
print(f"First 4 bytes (magic): 0x{magic:08X}")

# Try to find actual data extent by scanning for non-zero bytes
print(f"\nScanning for actual data extent...")

# Check first 1MB in 4KB chunks
chunk_size = 4096
max_check = 1 * 1024 * 1024  # Check first 1MB
last_nonzero = 0

for offset in range(0, min(max_check, mmap_file.size), chunk_size):
    chunk = mmap_file.read_bytes(offset, min(chunk_size, mmap_file.size - offset))
    if any(b != 0 for b in chunk):
        last_nonzero = offset + chunk_size

print(f"Last non-zero data found at: ~{last_nonzero / 1024:.2f} KB")
print(f"Estimated actual data: ~{last_nonzero / 1024:.2f} KB")
print(f"File allocation overhead: {(mmap_file.size - last_nonzero) / (1024*1024):.2f} MB")
print(f"\n✅ This is NORMAL for memory-mapped files - pre-allocation for performance")

mmap_file.close()
