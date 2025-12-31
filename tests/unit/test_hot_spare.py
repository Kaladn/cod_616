import time
import types

import pytest

import resilience.hot_spare as hs_mod
from resilience.hot_spare import HotSpare


def test_window_prune_correctness():
    # choose small window for test determinism
    w = 10.0
    h = HotSpare(window_seconds=w)
    base = 1000.0
    # events at base-20 (should be pruned), base-5 (kept), base (kept)
    h.mirror({'v': 1}, ts=base - 20.0)
    h.mirror({'v': 2}, ts=base - 5.0)
    h.mirror({'v': 3}, ts=base)
    # now add a new event at ts=base + 0.1 which triggers prune with now=base+0.1
    h.mirror({'v': 4}, ts=base + 0.1)
    s = h.stats()
    assert s['buffer_count'] == 3  # items with ts >= base-10.0
    assert s['oldest_ts'] >= base - 5.0


def test_drain_ordering():
    h = HotSpare(window_seconds=30.0)
    # add out-of-order timestamps
    h.mirror({'id': 'a'}, ts=200.0)
    h.mirror({'id': 'b'}, ts=100.0)
    h.mirror({'id': 'c'}, ts=150.0)
    out = h.drain()
    assert [o['id'] for o in out] == ['b', 'c', 'a']


def test_takeover_drain_behavior():
    h = HotSpare(window_seconds=30.0)
    h.mirror({'x': 1}, ts=10.0)
    h.mirror({'x': 2}, ts=20.0)
    h.takeover()
    out1 = h.drain()
    assert len(out1) == 2
    # after drain in takeover mode buffer cleared
    s = h.stats()
    assert s['buffer_count'] == 0
    # mirror more events, drain returns those
    h.mirror({'x': 3}, ts=30.0)
    out2 = h.drain()
    assert [o['x'] for o in out2] == [3]


def test_non_takeover_drain_keeps_buffer():
    h = HotSpare(window_seconds=30.0)
    h.mirror({'a': 1}, ts=1.0)
    h.mirror({'a': 2}, ts=2.0)
    s_before = h.stats()
    out = h.drain()
    s_after = h.stats()
    # buffer unchanged
    assert s_before['buffer_count'] == s_after['buffer_count']


def test_max_cap_drop_newest():
    # temporarily lower the MAX_EVENTS for this test
    orig = hs_mod.MAX_EVENTS
    try:
        hs_mod.MAX_EVENTS = 10
        h = HotSpare(window_seconds=30.0)
        base = 0.0
        # fill up to MAX
        for i in range(hs_mod.MAX_EVENTS):
            h.mirror({'i': i}, ts=base + i)
        s = h.stats()
        assert s['buffer_count'] == hs_mod.MAX_EVENTS
        # add one more -> should be dropped (use a timestamp close to the window
        # so pruning does not remove the existing buffer)
        h.mirror({'i': 9999}, ts=base + hs_mod.MAX_EVENTS - 0.1)
        s2 = h.stats()
        assert s2['buffer_count'] == hs_mod.MAX_EVENTS
        assert s2['dropped_newest'] == 1
    finally:
        hs_mod.MAX_EVENTS = orig