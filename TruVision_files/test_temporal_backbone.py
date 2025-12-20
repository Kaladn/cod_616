"""
Test Temporal Backbone — ChronosManager + EventManager + 6-1-6 Capsules

This test forces event recording to demonstrate the temporal backbone.
"""
import sys
from pathlib import Path

# Add paths
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from event_system.chronos_manager import ChronosManager, ChronosMode
from event_system.event_manager import EventManager

def test_temporal_backbone():
    """Test ChronosManager + EventManager + 6-1-6 Capsules"""
    
    print("="*80)
    print("Testing Temporal Backbone")
    print("="*80)
    print()
    
    # Initialize Chronos
    print("[1/4] Initializing ChronosManager...")
    chronos = ChronosManager()
    chronos.initialize(ChronosMode.LIVE)
    print(f"  ✓ Timestamp: {chronos.now():.6f}")
    print()
    
    # Initialize EventManager
    print("[2/4] Initializing EventManager...")
    event_mgr = EventManager(chronos_manager=chronos)
    event_mgr.register_source("test_vision", "sensor", {"type": "test"})
    event_mgr.register_source("test_operators", "detector", {"type": "test"})
    print("  ✓ EventManager initialized")
    print()
    
    # Record 20 events (simulating TrueVision windows)
    print("[3/4] Recording 20 test events...")
    session_id = f"test_session_{int(chronos.now())}"
    session_chain = event_mgr.create_chain(session_id, {"test": True})
    
    events = []
    for i in range(20):
        # Simulate vision event
        event = event_mgr.record_event(
            source_id="test_vision",
            tags=["vision", "window", f"seq_{i}"],
            metadata={
                "window_id": i,
                "eomm_score": i * 0.05,  # 0.00 → 0.95
                "test": True
            }
        )
        events.append(event)
        
        # Attach to chain
        event_mgr.attach_event_to_chain(event.event_id, session_id)
        
        # Brief delay
        import time
        time.sleep(0.05)
    
    print(f"  ✓ Recorded {len(events)} events")
    print()
    
    # Extract 6-1-6 capsule for middle event
    print("[4/4] Extracting 6-1-6 capsule...")
    middle_event = events[10]  # Event 10 (0-indexed)
    
    capsule = event_mgr.get_capsule(middle_event.event_id)
    
    print(f"  Anchor Event: {capsule.anchor_event.event_id}")
    print(f"  Timestamp: {capsule.anchor_event.timestamp:.6f}")
    print(f"  EOMM Score: {capsule.anchor_event.metadata['eomm_score']:.2f}")
    print()
    print(f"  ✓ Precursors: {len(capsule.events_before)} events")
    for i, evt in enumerate(capsule.events_before):
        print(f"      [{i+1}] {evt.event_id} @ {evt.timestamp:.6f} (EOMM: {evt.metadata['eomm_score']:.2f})")
    print()
    print(f"  ✓ Consequences: {len(capsule.events_after)} events")
    for i, evt in enumerate(capsule.events_after):
        print(f"      [{i+1}] {evt.event_id} @ {evt.timestamp:.6f} (EOMM: {evt.metadata['eomm_score']:.2f})")
    print()
    print(f"  ✓ Time Span: {capsule.get_time_span():.3f} seconds")
    print()
    
    # Test temporal queries
    print("="*80)
    print("Testing Temporal Queries")
    print("="*80)
    print()
    
    # Query: Last 5 events
    print("[Query 1] Last 5 events:")
    recent = event_mgr.get_recent_events(limit=5)
    for evt in recent:
        print(f"  {evt.event_id} @ {evt.timestamp:.6f} (EOMM: {evt.metadata['eomm_score']:.2f})")
    print()
    
    # Query: Events in time range
    print("[Query 2] Events in middle 0.5 seconds:")
    start_time = events[5].timestamp
    end_time = events[15].timestamp
    range_events = event_mgr.get_events_in_range(start_time, end_time, source_id="test_vision")
    print(f"  Found {len(range_events)} events in range")
    print()
    
    # Query: Session chain
    print("[Query 3] Session chain:")
    chain = event_mgr.get_chain(session_id)
    print(f"  Chain ID: {chain.chain_id}")
    print(f"  Events: {len(chain.event_ids)}")
    print(f"  Duration: {chain.get_duration():.3f} seconds")
    print()
    
    # Statistics
    print("="*80)
    print("EventManager Statistics")
    print("="*80)
    event_mgr.print_stats()
    print()
    
    print("="*80)
    print("✅ TEMPORAL BACKBONE OPERATIONAL")
    print("="*80)
    print()
    print("🔥 ChronosManager: Deterministic time")
    print("🔥 EventManager: Events + Chains")
    print("🔥 6-1-6 Capsules: Causal memory")
    print("🔥 Temporal Queries: Range + Recent + Chain")
    print()


if __name__ == "__main__":
    test_temporal_backbone()
