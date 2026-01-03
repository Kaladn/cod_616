"""
Smoke tests for ChronosManager v1.

Tests basic functionality across all three modes.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from event_system.chronos_manager import ChronosManager, ChronosMode
import time


def test_live_monotonic():
    """Test LIVE mode maintains monotonic time."""
    c = ChronosManager()
    c.initialize(ChronosMode.LIVE)
    t1 = c.now()
    time.sleep(0.01)
    t2 = c.now()
    assert t2 >= t1, f"Time went backwards: {t1} -> {t2}"
    print("✓ LIVE mode monotonic")


def test_simulated_advance():
    """Test SIMULATED mode manual time control."""
    c = ChronosManager()
    c.initialize(ChronosMode.SIMULATED, {"simulated_time": 100.0})
    assert c.now() == 100.0, "Initial simulated time wrong"
    
    c.advance_simulated(5.0)
    assert c.now() == 105.0, "Advance failed"
    
    c.set_simulated_time(200.0)
    assert c.now() == 200.0, "Set time failed"
    print("✓ SIMULATED mode time control")


def test_replay_speed():
    """Test REPLAY mode speed multiplier."""
    c = ChronosManager()
    c.initialize(ChronosMode.REPLAY, {
        "replay_start_time": 1000.0,
        "replay_speed": 2.0,
        "replay_offset": 0.0,
    })
    t1 = c.now()
    time.sleep(0.05)
    t2 = c.now()
    assert t2 > t1, "Time didn't advance"
    # At 2x speed, 0.05 real seconds = ~0.1 cognitive seconds
    delta = t2 - t1
    assert delta > 0.05, f"Speed too slow: {delta}s over 0.05s"
    print(f"✓ REPLAY mode 2x speed (delta={delta:.3f}s)")


def test_mode_switching():
    """Test switching between modes."""
    c = ChronosManager()
    
    # Start LIVE
    c.initialize(ChronosMode.LIVE)
    live_time = c.now()
    assert live_time > 0
    
    # Switch to SIMULATED
    c.set_mode(ChronosMode.SIMULATED, {"simulated_time": 5000.0})
    sim_time = c.now()
    assert sim_time == 5000.0, f"Expected 5000.0, got {sim_time}"
    
    # Switch back to LIVE
    c.set_mode(ChronosMode.LIVE)
    live_time2 = c.now()
    assert live_time2 >= live_time, "Time went backwards after mode switch"
    print("✓ Mode switching")


def test_metadata():
    """Test metadata retrieval."""
    c = ChronosManager()
    c.initialize(ChronosMode.REPLAY, {
        "replay_start_time": 1000.0,
        "replay_speed": 3.0,
        "replay_offset": 10.0,
    })
    
    meta = c.get_metadata()
    assert meta["mode"] == "replay"
    assert meta["initialized"] is True
    assert meta["replay"]["speed"] == 3.0
    assert meta["replay"]["offset"] == 10.0
    print("✓ Metadata retrieval")


def test_delta():
    """Test time delta calculation."""
    c = ChronosManager()
    c.initialize(ChronosMode.SIMULATED, {"simulated_time": 100.0})
    
    t1 = c.now()  # 100.0
    c.advance_simulated(5.0)  # Now 105.0
    
    delta = c.delta(t1)
    assert delta == 5.0, f"Expected delta=5.0, got {delta}"
    print("✓ Delta calculation")


if __name__ == "__main__":
    print("=" * 60)
    print("ChronosManager v1 Smoke Tests")
    print("=" * 60)
    
    tests = [
        test_live_monotonic,
        test_simulated_advance,
        test_replay_speed,
        test_mode_switching,
        test_metadata,
        test_delta,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🔥 All tests passed. Time fabric operational.")
    else:
        print(f"\n⚠️  {failed} test(s) failed.")
        sys.exit(1)
