"""
EventManager v1 — Comprehensive Smoke Tests

Tests all core functionality:
1. Event creation and recording
2. Capsule building (6-1-6 logic)
3. Cross-stream capsules
4. Time range queries
5. Chain management
6. Statistics

Author: Manus AI
Date: December 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from event_system.event_manager import EventManager, Event, Capsule, Stream, Chain
from event_system.chronos_manager import ChronosManager, ChronosMode
import time


def test_event_creation():
    """Test 1: Event creation and recording"""
    print("\n" + "="*80)
    print("TEST 1: Event Creation and Recording")
    print("="*80)
    
    chronos = ChronosManager()
    chronos.initialize(ChronosMode.SIMULATED)
    chronos.set_simulated_time(1000.0)
    
    event_mgr = EventManager(chronos_manager=chronos)
    
    # Register sources
    event_mgr.register_source("vision", "sensor", {"camera_id": "cam_01"})
    event_mgr.register_source("network", "logger", {"interface": "eth0"})
    event_mgr.register_source("input", "sensor", {"device": "mouse"})
    
    # Record events
    event1 = event_mgr.record_event(
        source_id="vision",
        tags=["motion_detected"],
        metadata={"region": "top_left", "intensity": 0.75}
    )
    
    chronos.advance_simulated(0.1)
    
    event2 = event_mgr.record_event(
        source_id="network",
        tags=["lag_spike"],
        metadata={"latency_ms": 250, "packet_loss": 0.05}
    )
    
    chronos.advance_simulated(0.1)
    
    event3 = event_mgr.record_event(
        source_id="input",
        tags=["click", "left_button"],
        metadata={"x": 1024, "y": 768}
    )
    
    # Verify events
    assert event1.event_id == "evt_00000001"
    assert event1.source_id == "vision"
    assert "motion_detected" in event1.tags
    assert event1.metadata["region"] == "top_left"
    
    assert event2.event_id == "evt_00000002"
    assert event2.source_id == "network"
    assert "lag_spike" in event2.tags
    
    assert event3.event_id == "evt_00000003"
    assert event3.source_id == "input"
    assert "click" in event3.tags
    
    # Verify event lookup
    retrieved = event_mgr.get_event("evt_00000002")
    assert retrieved is not None
    assert retrieved.event_id == event2.event_id
    assert retrieved.timestamp == event2.timestamp
    
    # Verify statistics
    stats = event_mgr.get_stats()
    assert stats["total_events"] == 3
    assert stats["total_streams"] == 3
    assert stats["stream_breakdown"]["vision"] == 1
    assert stats["stream_breakdown"]["network"] == 1
    assert stats["stream_breakdown"]["input"] == 1
    
    print("\n✓ Event creation test PASSED")
    print(f"  - Created {stats['total_events']} events across {stats['total_streams']} streams")
    print(f"  - Event IDs: {event1.event_id}, {event2.event_id}, {event3.event_id}")
    print(f"  - Timestamps: {event1.timestamp:.3f}, {event2.timestamp:.3f}, {event3.timestamp:.3f}")
    
    return event_mgr


def test_capsule_building(event_mgr: EventManager):
    """Test 2: Capsule building (6-1-6 logic)"""
    print("\n" + "="*80)
    print("TEST 2: Capsule Building (6-1-6 Logic)")
    print("="*80)
    
    # Create 13 events in vision stream
    chronos = event_mgr.chronos
    event_mgr.register_source("test", "sensor", {"purpose": "testing"})
    
    events = []
    for i in range(13):
        chronos.advance_simulated(0.01)
        event = event_mgr.record_event(
            source_id="test",
            tags=[f"event_{i}"],
            metadata={"sequence": i}
        )
        events.append(event)
    
    # Get capsule for middle event (index 6)
    middle_event = events[6]
    capsule = event_mgr.get_capsule(middle_event.event_id)
    
    # Verify capsule structure
    assert capsule.anchor_event.event_id == middle_event.event_id
    assert len(capsule.events_before) == 6  # events 0-5
    assert len(capsule.events_after) == 6   # events 7-12
    
    # Verify order (events_before is newest-first, events_after is oldest-first)
    assert capsule.events_before[0].event_id == events[5].event_id  # Newest of before
    assert capsule.events_before[-1].event_id == events[0].event_id  # Oldest of before
    assert capsule.events_after[0].event_id == events[7].event_id   # Oldest of after
    assert capsule.events_after[-1].event_id == events[12].event_id  # Newest of after
    
    # Verify get_all_events() returns chronological order
    all_events = capsule.get_all_events()
    assert len(all_events) == 13
    assert all_events[0].event_id == events[0].event_id
    assert all_events[6].event_id == events[6].event_id
    assert all_events[12].event_id == events[12].event_id
    
    # Verify time span
    time_span = capsule.get_time_span()
    assert time_span > 0.0
    expected_span = all_events[-1].timestamp - all_events[0].timestamp
    assert abs(time_span - expected_span) < 0.001
    
    # Test edge case: First event (should have 0 before, 6 after)
    first_capsule = event_mgr.get_capsule(events[0].event_id)
    assert len(first_capsule.events_before) == 0
    assert len(first_capsule.events_after) == 6
    
    # Test edge case: Last event (should have 6 before, 0 after)
    last_capsule = event_mgr.get_capsule(events[12].event_id)
    assert len(last_capsule.events_before) == 6
    assert len(last_capsule.events_after) == 0
    
    print("\n✓ Capsule building test PASSED")
    print(f"  - Middle capsule: 6 before + anchor + 6 after = 13 total")
    print(f"  - Time span: {time_span:.3f} seconds")
    print(f"  - First capsule: 0 before + anchor + 6 after")
    print(f"  - Last capsule: 6 before + anchor + 0 after")


def test_cross_stream_capsule(event_mgr: EventManager):
    """Test 3: Cross-stream capsules"""
    print("\n" + "="*80)
    print("TEST 3: Cross-Stream Capsules")
    print("="*80)
    
    chronos = event_mgr.chronos
    
    # Record events across multiple streams at similar times
    chronos.set_simulated_time(2000.0)
    
    vision_event = event_mgr.record_event(
        source_id="vision",
        tags=["enemy_spotted"],
        metadata={"location": "north"}
    )
    
    chronos.advance_simulated(0.05)  # 50ms later
    
    input_event = event_mgr.record_event(
        source_id="input",
        tags=["mouse_move"],
        metadata={"dx": 100, "dy": 50}
    )
    
    chronos.advance_simulated(0.1)  # 100ms later
    
    network_event = event_mgr.record_event(
        source_id="network",
        tags=["packet_sent"],
        metadata={"size": 256}
    )
    
    chronos.advance_simulated(0.05)  # 50ms later
    
    test_event = event_mgr.record_event(
        source_id="test",
        tags=["test_marker"],
        metadata={"marker": "anchor"}
    )
    
    # Get cross-stream capsule with 200ms window
    cross_stream = event_mgr.get_cross_stream_capsule(
        anchor_event_id=network_event.event_id,
        time_window_ms=200.0
    )
    
    # Verify all streams have events
    assert "vision" in cross_stream
    assert "input" in cross_stream
    assert "network" in cross_stream
    
    # Verify events are correct
    assert vision_event.event_id in [e.event_id for e in cross_stream["vision"]]
    assert input_event.event_id in [e.event_id for e in cross_stream["input"]]
    assert network_event.event_id in [e.event_id for e in cross_stream["network"]]
    
    print("\n✓ Cross-stream capsule test PASSED")
    print(f"  - Anchor: {network_event.event_id} at {network_event.timestamp:.3f}")
    print(f"  - Window: ±200ms")
    print(f"  - Streams captured: {list(cross_stream.keys())}")
    for source_id, events in cross_stream.items():
        print(f"    {source_id}: {len(events)} events")


def test_time_range_queries(event_mgr: EventManager):
    """Test 4: Time range queries"""
    print("\n" + "="*80)
    print("TEST 4: Time Range Queries")
    print("="*80)
    
    chronos = event_mgr.chronos
    
    # Record events at specific times
    chronos.set_simulated_time(3000.0)
    early_event = event_mgr.record_event(
        source_id="test",
        tags=["early"],
        metadata={}
    )
    
    chronos.set_simulated_time(3500.0)
    middle_event = event_mgr.record_event(
        source_id="test",
        tags=["middle"],
        metadata={}
    )
    
    chronos.set_simulated_time(4000.0)
    late_event = event_mgr.record_event(
        source_id="test",
        tags=["late"],
        metadata={}
    )
    
    # Query range [3200, 3800] - should only get middle event
    events = event_mgr.get_events_in_range(3200.0, 3800.0)
    assert len(events) == 1
    assert events[0].event_id == middle_event.event_id
    
    # Query range [2900, 4100] - should get all three
    events = event_mgr.get_events_in_range(2900.0, 4100.0)
    event_ids = [e.event_id for e in events]
    assert early_event.event_id in event_ids
    assert middle_event.event_id in event_ids
    assert late_event.event_id in event_ids
    
    # Query with source filter
    test_events = event_mgr.get_events_in_range(
        start_time=0.0,
        end_time=5000.0,
        source_id="test"
    )
    assert len(test_events) > 0
    assert all(e.source_id == "test" for e in test_events)
    
    # Query with tag filter
    middle_tagged = event_mgr.get_events_in_range(
        start_time=0.0,
        end_time=5000.0,
        tags=["middle"]
    )
    assert len(middle_tagged) == 1
    assert middle_tagged[0].event_id == middle_event.event_id
    
    # Test get_recent_events
    recent = event_mgr.get_recent_events(limit=5)
    assert len(recent) <= 5
    assert recent[0].timestamp >= recent[-1].timestamp  # Newest first
    
    # Test get_recent_events with source filter
    recent_test = event_mgr.get_recent_events(limit=10, source_id="test")
    assert all(e.source_id == "test" for e in recent_test)
    
    print("\n✓ Time range query test PASSED")
    print(f"  - Range [3200, 3800]: {len(events)} event (middle)")
    print(f"  - Range [2900, 4100]: {len(event_mgr.get_events_in_range(2900.0, 4100.0))} events (early, middle, late)")
    print(f"  - Source filter 'test': {len(test_events)} events")
    print(f"  - Tag filter 'middle': {len(middle_tagged)} event")
    print(f"  - Recent events (limit 5): {len(recent)} events")


def test_chain_management(event_mgr: EventManager):
    """Test 5: Chain management"""
    print("\n" + "="*80)
    print("TEST 5: Chain Management")
    print("="*80)
    
    chronos = event_mgr.chronos
    
    # Create events for a gaming match
    chronos.set_simulated_time(5000.0)
    match_start = event_mgr.record_event(
        source_id="test",
        tags=["match_start"],
        metadata={"map": "dust2"}
    )
    
    chronos.advance_simulated(10.0)
    kill_event = event_mgr.record_event(
        source_id="test",
        tags=["kill"],
        metadata={"weapon": "ak47"}
    )
    
    chronos.advance_simulated(5.0)
    death_event = event_mgr.record_event(
        source_id="test",
        tags=["death"],
        metadata={"killer": "enemy_01"}
    )
    
    chronos.advance_simulated(10.0)
    match_end = event_mgr.record_event(
        source_id="test",
        tags=["match_end"],
        metadata={"result": "win"}
    )
    
    # Create chain
    chain = event_mgr.create_chain(
        chain_id="match_12345",
        metadata={"map": "dust2", "mode": "competitive"}
    )
    
    # Attach events to chain
    event_mgr.attach_event_to_chain(match_start.event_id, "match_12345")
    event_mgr.attach_event_to_chain(kill_event.event_id, "match_12345")
    event_mgr.attach_event_to_chain(death_event.event_id, "match_12345")
    event_mgr.attach_event_to_chain(match_end.event_id, "match_12345")
    
    # Verify chain
    retrieved_chain = event_mgr.get_chain("match_12345")
    assert retrieved_chain is not None
    assert retrieved_chain.chain_id == "match_12345"
    assert len(retrieved_chain.event_ids) == 4
    
    # Get chain events
    chain_events = event_mgr.get_chain_events("match_12345")
    assert len(chain_events) == 4
    assert chain_events[0].event_id == match_start.event_id
    assert chain_events[1].event_id == kill_event.event_id
    assert chain_events[2].event_id == death_event.event_id
    assert chain_events[3].event_id == match_end.event_id
    
    # Verify chronological order
    for i in range(len(chain_events) - 1):
        assert chain_events[i].timestamp <= chain_events[i+1].timestamp
    
    # Verify duration
    duration = chain.get_duration()
    expected_duration = match_end.timestamp - match_start.timestamp
    assert abs(duration - expected_duration) < 0.001
    
    print("\n✓ Chain management test PASSED")
    print(f"  - Chain ID: {chain.chain_id}")
    print(f"  - Events in chain: {len(chain_events)}")
    print(f"  - Duration: {duration:.3f} seconds")
    print(f"  - Start time: {chain.start_time:.3f}")
    print(f"  - End time: {chain.end_time:.3f}")


def test_statistics():
    """Test 6: Statistics and diagnostics"""
    print("\n" + "="*80)
    print("TEST 6: Statistics and Diagnostics")
    print("="*80)
    
    chronos = ChronosManager()
    chronos.initialize(ChronosMode.SIMULATED)
    chronos.set_simulated_time(6000.0)
    
    event_mgr = EventManager(chronos_manager=chronos)
    
    # Register multiple sources
    sources = ["vision", "audio", "network", "input", "process"]
    for source in sources:
        event_mgr.register_source(source, "sensor", {})
    
    # Record varying numbers of events per source
    for i, source in enumerate(sources):
        num_events = (i + 1) * 2  # 2, 4, 6, 8, 10 events
        for j in range(num_events):
            chronos.advance_simulated(0.01)
            event_mgr.record_event(
                source_id=source,
                tags=[f"event_{j}"],
                metadata={}
            )
    
    # Create chains
    chain1 = event_mgr.create_chain("chain_1", {})
    chain2 = event_mgr.create_chain("chain_2", {})
    
    # Get statistics
    stats = event_mgr.get_stats()
    
    assert stats["total_events"] == 30  # 2+4+6+8+10
    assert stats["total_streams"] == 5
    assert stats["total_chains"] == 2
    
    assert stats["stream_breakdown"]["vision"] == 2
    assert stats["stream_breakdown"]["audio"] == 4
    assert stats["stream_breakdown"]["network"] == 6
    assert stats["stream_breakdown"]["input"] == 8
    assert stats["stream_breakdown"]["process"] == 10
    
    # Print stats (for visual verification)
    event_mgr.print_stats()
    
    print("\n✓ Statistics test PASSED")
    print(f"  - Total events: {stats['total_events']}")
    print(f"  - Total streams: {stats['total_streams']}")
    print(f"  - Total chains: {stats['total_chains']}")
    print(f"  - Capsule cache size: {stats['capsule_cache_size']}")


def main():
    """Run all smoke tests"""
    print("\n" + "="*80)
    print("EventManager v1 — Comprehensive Smoke Tests")
    print("="*80)
    print("\nRunning 6 comprehensive tests...")
    
    start_time = time.time()
    
    try:
        # Test 1: Event creation
        event_mgr = test_event_creation()
        
        # Test 2: Capsule building
        test_capsule_building(event_mgr)
        
        # Test 3: Cross-stream capsules
        test_cross_stream_capsule(event_mgr)
        
        # Test 4: Time range queries
        test_time_range_queries(event_mgr)
        
        # Test 5: Chain management
        test_chain_management(event_mgr)
        
        # Test 6: Statistics
        test_statistics()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("🔥 ALL TESTS PASSED")
        print("="*80)
        print(f"\n✓ 6 tests completed in {elapsed:.3f} seconds")
        print("\nEventManager v1 is operational and ready for integration.")
        print("\nNext steps:")
        print("  1. Integrate with CompuCog organs")
        print("  2. Start recording real events")
        print("  3. Build episodic memory with capsules")
        print("  4. Enable forensic timeline reconstruction")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
