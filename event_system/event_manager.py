import time
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Event:
    event_id: str
    source_id: str
    tags: List[str]
    metadata: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class Chain:
    def __init__(self, chain_id: str, metadata: Dict[str, Any] = None):
        self.chain_id = chain_id
        self.metadata = metadata or {}
        self.event_ids: List[str] = []

    def get_duration(self) -> float:
        return 0.0

class EventManager:
    def __init__(self, chronos_manager=None):
        self.chronos = chronos_manager
        self._events: Dict[str, Event] = {}
        self._chains: Dict[str, Chain] = {}
        self._next_id = 1

    def register_source(self, *args, **kwargs):
        return True

    def create_chain(self, chain_id: str, metadata: Dict[str, Any] = None) -> Chain:
        chain = Chain(chain_id, metadata)
        self._chains[chain_id] = chain
        return chain

    def record_event(self, source_id: str, tags: List[str], metadata: Dict[str, Any]) -> Event:
        eid = f"evt_{self._next_id}"
        self._next_id += 1
        event = Event(event_id=eid, source_id=source_id, tags=tags, metadata=metadata)
        self._events[eid] = event
        return event

    def attach_event_to_chain(self, event_id: str, chain_id: str):
        if chain_id not in self._chains:
            self._chains[chain_id] = Chain(chain_id)
        self._chains[chain_id].event_ids.append(event_id)

    def get_chain(self, chain_id: str) -> Chain:
        return self._chains.get(chain_id, Chain(chain_id))

    def get_recent_events(self, limit: int = 5) -> List[Event]:
        return list(self._events.values())[-limit:]

    def get_capsule(self, event_id: str):
        # Construct a 6-1-6 capsule based on the chain that contains the event.
        anchor = self._events.get(event_id)
        if anchor is None:
            return None

        # Find the chain that contains this event
        chain_id = None
        for cid, chain in self._chains.items():
            if event_id in chain.event_ids:
                chain_id = cid
                break

        # If not part of a chain, return minimal capsule with only the anchor
        class Capsule:
            def __init__(self, anchor, before, after):
                self.anchor_event = anchor
                self.events_before = before
                self.events_after = after
            def get_time_span(self):
                times = [self.anchor_event.timestamp]
                times += [e.timestamp for e in self.events_before]
                times += [e.timestamp for e in self.events_after]
                if not times:
                    return 0.0
                return max(times) - min(times)

        if chain_id is None:
            return Capsule(anchor, [], [])

        chain = self._chains[chain_id]
        # Build ordered list of Event objects for chain
        event_objs = [self._events[eid] for eid in chain.event_ids if eid in self._events]
        # Find index of anchor
        idx = None
        for i, ev in enumerate(event_objs):
            if ev.event_id == event_id:
                idx = i
                break

        if idx is None:
            return Capsule(anchor, [], [])

        before = event_objs[max(0, idx - 6): idx]
        after = event_objs[idx + 1: idx + 1 + 6]

        return Capsule(anchor, before, after)

    def get_events_in_range(self, start_time: float, end_time: float, source_id: str = None):
        """Return events whose timestamp is within [start_time, end_time].
        If `source_id` is provided, filter by event.source_id.
        """
        results = []
        for ev in self._events.values():
            if start_time <= ev.timestamp <= end_time:
                if source_id is None or ev.source_id == source_id:
                    results.append(ev)
        # Sort by timestamp ascending
        results.sort(key=lambda e: e.timestamp)
        return results

    def print_stats(self):
        print(f"EventManager: {len(self._events)} events, {len(self._chains)} chains")
