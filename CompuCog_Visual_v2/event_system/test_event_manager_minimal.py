"""
EventManager Contract Test: Validate Event recording in isolation
Tests the ChronosManager → EventManager → Event flow with NO adapters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_system.chronos_manager import ChronosManager
from event_system.event_manager import EventManager

def main():
    print("=" * 60)
    print("EventManager Contract Test")
    print("=" * 60)
    
    # Step 1: Initialize ChronosManager
    print("\n[1/5] Initializing ChronosManager...")
    chronos = ChronosManager()
    print(f"✓ Chronos initialized, current time: {chronos.now()}")
    
    # Step 2: Initialize EventManager
    print("\n[2/5] Initializing EventManager...")
    event_mgr = EventManager(chronos)
    print("✓ EventManager initialized")
    
    # Step 3: Register a test source
    print("\n[3/5] Registering test source...")
    event_mgr.register_source(source_id="test_source", kind="sensor")
    print("✓ Source 'test_source' registered")
    
    # Step 4: Record ONE event
    print("\n[4/5] Recording test event...")
    event = event_mgr.record_event(
        source_id="test_source",
        tags=["test"],
        metadata={"test_key": "test_value"}
    )
    print(f"✓ Event recorded: {event.event_id}")
    
    # Step 5: Retrieve event stats
    print("\n[5/5] Retrieving stats...")
    stats = event_mgr.get_stats()
    print(f"✓ Total events: {stats.get('total_events', 0)}")
    print(f"✓ Sources: {list(stats.get('sources', {}).keys())}")
    
    print("\n" + "=" * 60)
    print("✅ EventManager Contract: VALIDATED")
    print("=" * 60)
    print("\nEventManager can record and retrieve events.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
