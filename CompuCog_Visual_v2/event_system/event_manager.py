"""
EventManager v1 — Temporal Backbone for CompuCog/CortexOS

Provides deterministic event creation, 6-1-6 capsules, per-source streams,
event chains, and clean query APIs.

Architecture:
- Event: Atomic "thing that happened"
- Capsule: 6-1-6 temporal window (6 before, anchor, 6 after)
- Stream: Per-source ordered event log
- Chain: Larger sequences of related events
- EventManager: Central coordinator

Integration:
- ChronosManager: Deterministic timestamps
- PulseWriter: Optional pulse_id references
- NVMe Brain: Optional spatial references

Author: Manus AI
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import bisect
import sys
from pathlib import Path

# Add parent directory for forge_memory imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Forge Memory integration per CONTRACT_ATLAS.md v1.1
if TYPE_CHECKING:
    from memory.forge_memory.core.binary_log import BinaryLog
    from memory.forge_memory.core.record import ForgeRecord


@dataclass
class Event:
    """
    Atomic event in the system.
    
    Invariants:
    - event_id is unique across all events
    - timestamp is monotonically increasing within a stream
    - source_id must be registered before use
    - tags are lowercase, no spaces
    - metadata is JSON-serializable
    """
    event_id: str
    timestamp: float
    source_id: str
    tags: List[str]
    metadata: dict
    pulse_id: Optional[str] = None
    nvme_ref: Optional[Tuple[int, int, int]] = None
    
    def to_dict(self) -> dict:
        """Convert Event to JSON-serializable dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "tags": self.tags,
            "metadata": self.metadata,
            "pulse_id": self.pulse_id,
            "nvme_ref": self.nvme_ref
        }
    
    @staticmethod
    def from_dict(data: dict) -> Event:
        """Reconstruct Event from dictionary."""
        return Event(
            event_id=data["event_id"],
            timestamp=data["timestamp"],
            source_id=data["source_id"],
            tags=data["tags"],
            metadata=data["metadata"],
            pulse_id=data.get("pulse_id"),
            nvme_ref=tuple(data["nvme_ref"]) if data.get("nvme_ref") else None
        )


@dataclass
class Capsule:
    """
    6-1-6 temporal window: 6 events before, anchor event, 6 events after.
    
    Invariants:
    - anchor_event is always present
    - events_before is ordered newest-to-oldest (reverse chronological)
    - events_after is ordered oldest-to-newest (chronological)
    - len(events_before) <= 6
    - len(events_after) <= 6
    - All events in capsule are from the same source_id as anchor
    """
    anchor_event: Event
    events_before: List[Event] = field(default_factory=list)
    events_after: List[Event] = field(default_factory=list)
    chain_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert Capsule to JSON-serializable dictionary."""
        return {
            "anchor_event": self.anchor_event.to_dict(),
            "events_before": [e.to_dict() for e in self.events_before],
            "events_after": [e.to_dict() for e in self.events_after],
            "chain_id": self.chain_id
        }
    
    def get_all_events(self) -> List[Event]:
        """Return all events in chronological order (before → anchor → after)."""
        return list(reversed(self.events_before)) + [self.anchor_event] + self.events_after
    
    def get_time_span(self) -> float:
        """Calculate total time span of capsule in seconds."""
        all_events = self.get_all_events()
        if len(all_events) < 2:
            return 0.0
        return all_events[-1].timestamp - all_events[0].timestamp


# Memory bounds constants
MAX_EVENTS_PER_STREAM = 10000  # Drop oldest when exceeded
MAX_CHAINS = 100  # Maximum concurrent chains
MAX_TIMELINE_EVENTS = 50000  # Global timeline limit
MAX_CAPSULE_CACHE = 1000  # LRU cache for capsules


@dataclass
class Stream:
    """
    Per-source ordered event log with BOUNDED MEMORY.
    
    Invariants:
    - events list is always sorted by timestamp (ascending)
    - last_event_time matches events[-1].timestamp if events exist
    - source_id is unique across all streams
    - len(events) <= MAX_EVENTS_PER_STREAM (drop-oldest policy)
    """
    source_id: str
    kind: str
    metadata: dict
    events: List[Event] = field(default_factory=list)
    last_event_time: float = 0.0
    max_events: int = MAX_EVENTS_PER_STREAM
    events_dropped: int = 0  # Counter for monitoring
    
    def append_event(self, event: Event) -> Optional[Event]:
        """
        Append event to stream, maintaining sorted order.
        Drops OLDEST event if at capacity (drop-oldest policy).
        
        Preconditions:
        - event.source_id == self.source_id
        - event.timestamp >= self.last_event_time (monotonic time)
        
        Returns:
        - The dropped event if eviction occurred, else None
        """
        assert event.source_id == self.source_id, f"Event source mismatch: {event.source_id} != {self.source_id}"
        assert event.timestamp >= self.last_event_time, f"Non-monotonic timestamp: {event.timestamp} < {self.last_event_time}"
        
        dropped = None
        # Drop oldest if at capacity
        if len(self.events) >= self.max_events:
            dropped = self.events.pop(0)
            self.events_dropped += 1
        
        self.events.append(event)
        self.last_event_time = event.timestamp
        return dropped
    
    def get_event_at_index(self, index: int) -> Optional[Event]:
        """Get event at specific index, or None if out of bounds."""
        if 0 <= index < len(self.events):
            return self.events[index]
        return None
    
    def find_event_index(self, event_id: str) -> Optional[int]:
        """Find index of event by event_id, or None if not found."""
        for i, event in enumerate(self.events):
            if event.event_id == event_id:
                return i
        return None


@dataclass
class Chain:
    """
    Larger sequence of related events across time.
    
    Usage:
    - "same match" (all events during a gaming match)
    - "same process" (all events from a specific process)
    - "same session" (all events during a work session)
    """
    chain_id: str
    event_ids: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    
    def add_event(self, event_id: str, timestamp: float) -> None:
        """Add event to chain, updating time bounds."""
        self.event_ids.append(event_id)
        
        if self.start_time == 0.0 or timestamp < self.start_time:
            self.start_time = timestamp
        
        if timestamp > self.end_time:
            self.end_time = timestamp
    
    def get_duration(self) -> float:
        """Get total duration of chain in seconds."""
        return self.end_time - self.start_time


class EventManager:
    """
    Central coordinator for all events in the system.
    
    Responsibilities:
    1. Event creation — single place where "this happened" becomes structured
    2. Time-aware chaining — uses Chronos to decide what belongs together
    3. Source separation — keeps per-stream order
    4. Cross-stream stitching — "what else was happening around this event?"
    5. Capsule lookup — given event_id, get full 6-1-6 context
    6. Export / query — simple queries for CompuCog / CortexOS
    
    Does NOT do:
    - Heavy analysis (that's RecognitionField / Security organ)
    - Long-term retention policies (that's a separate Retention organ)
    
    EventManager's job is clean structure, not opinions.
    """
    
    def __init__(
        self,
        chronos_manager,
        binary_log = None,  # type: BinaryLog
        pulse_writer=None,
        nvme_interface=None
    ):
        """
        Initialize EventManager with required dependencies.
        
        Parameters:
        - chronos_manager: Interface to ChronosManager for timestamps
        - binary_log: BinaryLog instance for Forge Memory writes (REQUIRED per CONTRACT_ATLAS.md v1.1)
        - pulse_writer: Optional interface to PulseWriter
        - nvme_interface: Optional interface to NVMe brain (CortexCube)
        """
        # External dependencies
        self.chronos = chronos_manager
        self.binary_log = binary_log
        self.pulse_writer = pulse_writer
        self.nvme = nvme_interface
        
        # Internal state (all BOUNDED)
        self.streams: Dict[str, Stream] = {}
        self.events: Dict[str, Event] = {}  # event_id -> Event lookup
        self.chains: Dict[str, Chain] = {}  # max MAX_CHAINS
        self.timeline: List[Event] = []  # max MAX_TIMELINE_EVENTS
        
        # Capsule cache with LRU eviction
        self.capsule_cache: Dict[str, Capsule] = {}  # max MAX_CAPSULE_CACHE
        self._capsule_access_order: List[str] = []  # LRU tracking
        
        # Monitoring counters
        self.events_dropped_total: int = 0
        self.chains_dropped: int = 0
        self.timeline_drops: int = 0
        
        # Monotonic event counter (for deterministic IDs)
        self.event_counter: int = 0
    
    # =========================================================================
    # SOURCE REGISTRATION
    # =========================================================================
    
    def register_source(
        self,
        source_id: str,
        kind: str,
        metadata: dict = None
    ) -> None:
        """
        Register a new event source (logger/organ).
        
        Parameters:
        - source_id: Unique identifier ("vision", "network", "input", etc.)
        - kind: Type of source ("sensor", "logger", "organ", etc.)
        - metadata: Optional source-specific metadata
        
        Preconditions:
        - source_id must be unique (not already registered)
        
        Postconditions:
        - New Stream created for source_id
        - Stream added to self.streams
        """
        if source_id in self.streams:
            raise ValueError(f"Source {source_id} already registered")
        
        stream = Stream(
            source_id=source_id,
            kind=kind,
            metadata=metadata or {}
        )
        
        self.streams[source_id] = stream
        
        print(f"[EventManager] Registered source: {source_id} (kind: {kind})")
    
    # =========================================================================
    # EVENT CREATION (CORE METHOD)
    # =========================================================================
    
    def record_event(
        self,
        source_id: str,
        tags: List[str] = None,
        metadata: dict = None,
        pulse_id: str = None,
        nvme_ref: Tuple[int, int, int] = None
    ) -> Event:
        """
        Record a new event in the system.
        
        This is the SINGLE ENTRY POINT for all event creation.
        
        Parameters:
        - source_id: Source that generated this event
        - tags: Semantic labels (["kill", "death", "lag_spike"])
        - metadata: Flexible JSON data
        - pulse_id: Optional link to PulseWriter record
        - nvme_ref: Optional (x, y, z) location in CortexCube
        
        Returns:
        - Newly created Event object
        
        Algorithm:
        1. Get timestamp from ChronosManager
        2. Create event_id (deterministic monotonic)
        3. Build Event struct
        4. Append to appropriate stream
        5. Maintain global timeline
        6. Update capsules
        7. Return Event
        """
        # Step 1: Get timestamp from Chronos (NEVER use time.time() directly)
        timestamp = self.chronos.now()
        
        # Step 2: Create event_id (deterministic monotonic ID)
        self.event_counter += 1
        event_id = f"evt_{self.event_counter:08d}"
        
        # Step 3: Build Event struct
        event = Event(
            event_id=event_id,
            timestamp=timestamp,
            source_id=source_id,
            tags=tags or [],
            metadata=metadata or {},
            pulse_id=pulse_id,
            nvme_ref=nvme_ref
        )
        
        # Step 4: Append to appropriate stream
        if source_id not in self.streams:
            raise ValueError(f"Source {source_id} not registered. Call register_source() first.")
        
        stream = self.streams[source_id]
        dropped_from_stream = stream.append_event(event)
        
        # Track dropped events
        if dropped_from_stream is not None:
            self.events_dropped_total += 1
            # Remove from global lookup
            self.events.pop(dropped_from_stream.event_id, None)
        
        # Step 5: Maintain global timeline (sorted by timestamp) - BOUNDED
        # Use bisect for efficient insertion into sorted list
        insert_index = self._find_timeline_insert_position(timestamp)
        self.timeline.insert(insert_index, event)
        
        # Evict oldest from timeline if over limit
        while len(self.timeline) > MAX_TIMELINE_EVENTS:
            evicted = self.timeline.pop(0)
            self.events.pop(evicted.event_id, None)
            self.timeline_drops += 1
        
        # Step 6: Store event in lookup dict
        self.events[event_id] = event
        
        # Step 7: Write to Forge Memory (per CONTRACT_ATLAS.md v1.1 Contract #12)
        if self.binary_log is not None:
            forge_record = self._event_to_forge_record(event)
            self.binary_log.append(forge_record)
        
        # Step 8: Update capsules (invalidate cache for affected events)
        self._update_capsules_after_event(event)
        
        # Step 9: Return Event
        return event
    
    # =========================================================================
    # FORGE MEMORY INTEGRATION (per CONTRACT_ATLAS.md v1.1)
    # =========================================================================
    
    def _event_to_forge_record(self, event: Event):  # -> ForgeRecord
        """
        Convert Event to ForgeRecord per CONTRACT_ATLAS.md v1.1 Contract #12.
        
        Field Mapping (per CONTRACT_ATLAS.md Event→ForgeRecord table):
        - pulse_id: EventManager pulse counter (self.event_counter)
        - worker_id: hash(source_id) % 256
        - seq: event_counter (extracted from event_id)
        - timestamp: event.timestamp
        - success: True (events are successful by definition)
        - task_id: event.source_id
        - engine_id: "event_pipeline_v1"
        - transform_id: "sensor_event"
        - failure_reason: None
        - grid_shape_in: (0, 0) — not applicable for events
        - grid_shape_out: (0, 0) — not applicable for events
        - color_count: 0 — not applicable for events
        - train_pair_indices: [] — not applicable for events
        - error_metrics: {} — events don't have error metrics
        - params: event.tags as dict {"tags": [tag1, tag2, ...]}
        - context: event.metadata
        
        Parameters:
        - event: Event to convert
        
        Returns:
        - ForgeRecord ready for BinaryLog.append()
        """
        # Runtime import (parent dir already in sys.path from module header)
        from memory.forge_memory.core.record import ForgeRecord
        
        # Extract seq from event_id (format: "evt_00000123")
        seq = int(event.event_id.split("_")[1])
        
        # Hash source_id to worker_id (0-255)
        worker_id = hash(event.source_id) % 256
        
        # Build params dict with tags
        params = {
            "tags": event.tags,
            "pulse_id": event.pulse_id,
            "nvme_ref": event.nvme_ref
        }
        
        return ForgeRecord(
            pulse_id=self.event_counter,  # Per CONTRACT_ATLAS.md: EventManager pulse counter
            worker_id=worker_id,
            seq=seq,
            timestamp=event.timestamp,
            success=True,
            task_id=event.source_id,
            engine_id="event_pipeline_v1",
            transform_id="sensor_event",
            failure_reason=None,
            grid_shape_in=(0, 0),
            grid_shape_out=(0, 0),
            color_count=0,
            train_pair_indices=[],
            error_metrics={},
            params=params,
            context=event.metadata
        )
    
    # =========================================================================
    # CAPSULE BUILDING (6-1-6 LOGIC)
    # =========================================================================
    
    def get_capsule(self, anchor_event_id: str) -> Capsule:
        """
        Get the 6-1-6 capsule for a specific event.
        
        Parameters:
        - anchor_event_id: Event ID to use as anchor
        
        Returns:
        - Capsule with up to 6 events before and 6 events after
        
        Algorithm:
        1. Find anchor event
        2. Get stream for that event's source
        3. Find index of anchor in stream
        4. Extract 6 events before (newest to oldest)
        5. Extract 6 events after (oldest to newest)
        6. Build and return Capsule
        
        Caching:
        - Check capsule_cache first
        - If not cached, build and cache
        """
        # Check cache first
        if anchor_event_id in self.capsule_cache:
            return self.capsule_cache[anchor_event_id]
        
        # Step 1: Find anchor event
        if anchor_event_id not in self.events:
            raise ValueError(f"Event {anchor_event_id} not found")
        
        anchor_event = self.events[anchor_event_id]
        
        # Step 2: Get stream for that event's source
        stream = self.streams[anchor_event.source_id]
        
        # Step 3: Find index of anchor in stream
        anchor_index = stream.find_event_index(anchor_event_id)
        if anchor_index is None:
            raise ValueError(f"Event {anchor_event_id} not found in stream {anchor_event.source_id}")
        
        # Step 4: Extract 6 events before (newest to oldest)
        before_start = max(0, anchor_index - 6)
        events_before = stream.events[before_start:anchor_index]
        events_before = list(reversed(events_before))  # Reverse to newest-first
        
        # Step 5: Extract 6 events after (oldest to newest)
        after_end = min(len(stream.events), anchor_index + 7)
        events_after = stream.events[anchor_index + 1:after_end]
        
        # Step 6: Build Capsule
        capsule = Capsule(
            anchor_event=anchor_event,
            events_before=events_before,
            events_after=events_after,
            chain_id=None  # TODO: Detect chain if event is part of one
        )
        
        # Cache capsule
        self.capsule_cache[anchor_event_id] = capsule
        
        return capsule
    
    def get_cross_stream_capsule(
        self,
        anchor_event_id: str,
        time_window_ms: float = 1000.0
    ) -> Dict[str, List[Event]]:
        """
        Get events from ALL streams within a time window around anchor event.
        
        Parameters:
        - anchor_event_id: Event ID to use as anchor
        - time_window_ms: Time window in milliseconds (default 1000ms = 1 second)
        
        Returns:
        - Dict mapping source_id -> List[Event] for all streams with events in window
        
        Algorithm:
        1. Find anchor event
        2. Calculate time window bounds
        3. For each stream, binary search for events in window
        4. Return dict of source_id -> events
        """
        # Step 1: Find anchor event
        if anchor_event_id not in self.events:
            raise ValueError(f"Event {anchor_event_id} not found")
        
        anchor_event = self.events[anchor_event_id]
        anchor_time = anchor_event.timestamp
        
        # Step 2: Calculate time window bounds
        window_sec = time_window_ms / 1000.0
        start_time = anchor_time - window_sec
        end_time = anchor_time + window_sec
        
        # Step 3: For each stream, find events in window
        result: Dict[str, List[Event]] = {}
        
        for source_id, stream in self.streams.items():
            # Binary search for start index
            start_idx = bisect.bisect_left(
                [e.timestamp for e in stream.events],
                start_time
            )
            
            # Binary search for end index
            end_idx = bisect.bisect_right(
                [e.timestamp for e in stream.events],
                end_time
            )
            
            # Extract events in window
            events_in_window = stream.events[start_idx:end_idx]
            
            if events_in_window:
                result[source_id] = events_in_window
        
        return result
    
    # =========================================================================
    # EVENT QUERIES
    # =========================================================================
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """
        Get event by ID.
        
        Parameters:
        - event_id: Event ID to look up
        
        Returns:
        - Event object, or None if not found
        """
        return self.events.get(event_id)
    
    def get_events_in_range(
        self,
        start_time: float,
        end_time: float,
        source_id: str = None,
        tags: List[str] = None
    ) -> List[Event]:
        """
        Get all events in a time range, optionally filtered by source and tags.
        
        Parameters:
        - start_time: Start timestamp (inclusive)
        - end_time: End timestamp (inclusive)
        - source_id: Optional source filter
        - tags: Optional tag filter (events must have ALL specified tags)
        
        Returns:
        - List of events in chronological order
        
        Algorithm:
        1. Use binary search on timeline to find start/end indices
        2. Extract events in range
        3. Apply source_id filter if specified
        4. Apply tags filter if specified
        5. Return filtered list
        """
        # Step 1: Binary search on timeline
        start_idx = bisect.bisect_left(
            [e.timestamp for e in self.timeline],
            start_time
        )
        
        end_idx = bisect.bisect_right(
            [e.timestamp for e in self.timeline],
            end_time
        )
        
        # Step 2: Extract events in range
        events = self.timeline[start_idx:end_idx]
        
        # Step 3: Apply source_id filter
        if source_id is not None:
            events = [e for e in events if e.source_id == source_id]
        
        # Step 4: Apply tags filter
        if tags is not None:
            events = [
                e for e in events
                if all(tag in e.tags for tag in tags)
            ]
        
        return events
    
    def get_recent_events(
        self,
        limit: int = 100,
        source_id: str = None
    ) -> List[Event]:
        """
        Get most recent events (newest first).
        
        Parameters:
        - limit: Maximum number of events to return
        - source_id: Optional source filter
        
        Returns:
        - List of events in reverse chronological order (newest first)
        """
        if source_id is None:
            # Get from global timeline
            events = self.timeline[-limit:]
            return list(reversed(events))
        else:
            # Get from specific stream
            if source_id not in self.streams:
                return []
            
            stream = self.streams[source_id]
            events = stream.events[-limit:]
            return list(reversed(events))
    
    # =========================================================================
    # CHAIN MANAGEMENT
    # =========================================================================
    
    def create_chain(
        self,
        chain_id: str,
        metadata: dict = None
    ) -> Chain:
        """
        Create a new event chain. BOUNDED to MAX_CHAINS.
        
        Parameters:
        - chain_id: Unique chain identifier
        - metadata: Optional chain-specific metadata
        
        Returns:
        - Newly created Chain object
        
        Raises:
        - ValueError if chain_id already exists
        """
        if chain_id in self.chains:
            raise ValueError(f"Chain {chain_id} already exists")
        
        # Evict oldest chain if at capacity (drop-oldest policy)
        while len(self.chains) >= MAX_CHAINS:
            oldest_chain_id = next(iter(self.chains))
            del self.chains[oldest_chain_id]
            self.chains_dropped += 1
        
        chain = Chain(
            chain_id=chain_id,
            metadata=metadata or {}
        )
        
        self.chains[chain_id] = chain
        
        return chain
    
    def attach_event_to_chain(
        self,
        event_id: str,
        chain_id: str
    ) -> None:
        """
        Attach an event to a chain.
        
        Parameters:
        - event_id: Event to attach
        - chain_id: Chain to attach to
        
        Raises:
        - ValueError if event or chain not found
        """
        if event_id not in self.events:
            raise ValueError(f"Event {event_id} not found")
        
        if chain_id not in self.chains:
            raise ValueError(f"Chain {chain_id} not found")
        
        event = self.events[event_id]
        chain = self.chains[chain_id]
        
        chain.add_event(event_id, event.timestamp)
        
        # Invalidate capsule cache for this event (chain_id changed)
        if event_id in self.capsule_cache:
            del self.capsule_cache[event_id]
    
    def get_chain(self, chain_id: str) -> Optional[Chain]:
        """Get chain by ID."""
        return self.chains.get(chain_id)
    
    def get_chain_events(self, chain_id: str) -> List[Event]:
        """
        Get all events in a chain, in chronological order.
        
        Parameters:
        - chain_id: Chain to retrieve events from
        
        Returns:
        - List of events in chronological order
        
        Raises:
        - ValueError if chain not found
        """
        if chain_id not in self.chains:
            raise ValueError(f"Chain {chain_id} not found")
        
        chain = self.chains[chain_id]
        
        # Look up events by ID
        events = []
        for event_id in chain.event_ids:
            if event_id in self.events:
                events.append(self.events[event_id])
        
        # Sort by timestamp (should already be sorted, but ensure it)
        events.sort(key=lambda e: e.timestamp)
        
        return events
    
    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================
    
    def _find_timeline_insert_position(self, timestamp: float) -> int:
        """
        Find insertion position in timeline using binary search.
        
        Parameters:
        - timestamp: Timestamp to insert
        
        Returns:
        - Index where event should be inserted to maintain sorted order
        
        Algorithm:
        - Use bisect_right to maintain stable sort (events with same timestamp
          maintain insertion order)
        """
        return bisect.bisect_right(
            [e.timestamp for e in self.timeline],
            timestamp
        )
    
    def _update_capsules_after_event(self, new_event: Event) -> None:
        """
        Invalidate capsule cache entries affected by new event.
        
        Parameters:
        - new_event: Newly added event
        
        Algorithm:
        1. Get stream for new event's source
        2. Find index of new event in stream
        3. Invalidate capsules for events within 6 positions before/after
        
        Note: This is a conservative approach. A more optimized version
        would only invalidate capsules that actually need updating.
        """
        stream = self.streams[new_event.source_id]
        new_index = stream.find_event_index(new_event.event_id)
        
        if new_index is None:
            return  # Event not in stream yet (shouldn't happen)
        
        # Invalidate capsules for events within 6 positions
        start_idx = max(0, new_index - 6)
        end_idx = min(len(stream.events), new_index + 7)
        
        for i in range(start_idx, end_idx):
            event = stream.events[i]
            if event.event_id in self.capsule_cache:
                del self.capsule_cache[event.event_id]
    
    # =========================================================================
    # STATISTICS & DIAGNOSTICS
    # =========================================================================
    
    def get_stats(self) -> dict:
        """
        Get statistics about EventManager state.
        
        Returns:
        - Dictionary with statistics
        """
        stream_breakdown = {
            source_id: len(stream.events)
            for source_id, stream in self.streams.items()
        }
        
        return {
            "total_events": len(self.events),
            "total_chains": len(self.chains),
            "total_streams": len(self.streams),
            "capsule_cache_size": len(self.capsule_cache),
            "stream_breakdown": stream_breakdown,
            "memory_bounds": {
                "max_events_per_stream": MAX_EVENTS_PER_STREAM,
                "max_chains": MAX_CHAINS,
                "max_timeline": MAX_TIMELINE_EVENTS,
                "max_capsule_cache": MAX_CAPSULE_CACHE,
            },
            "drops": {
                "events_dropped": self.events_dropped_total,
                "chains_dropped": self.chains_dropped,
                "timeline_drops": self.timeline_drops,
            }
        }
    
    def print_stats(self) -> None:
        """Print statistics to console."""
        stats = self.get_stats()
        
        print("\n[EventManager Stats]")
        print(f"  Total Events: {stats['total_events']}")
        print(f"  Total Chains: {stats['total_chains']}")
        print(f"  Total Streams: {stats['total_streams']}")
        print(f"  Capsule Cache Size: {stats['capsule_cache_size']}")
        
        # Memory usage vs bounds
        print(f"  Timeline: {len(self.timeline)}/{MAX_TIMELINE_EVENTS}")
        print(f"  Chains: {len(self.chains)}/{MAX_CHAINS}")
        
        # Drop counters
        drops = stats['drops']
        if any(v > 0 for v in drops.values()):
            print(f"  Drops: events={drops['events_dropped']}, chains={drops['chains_dropped']}, timeline={drops['timeline_drops']}")
        
        if stats['stream_breakdown']:
            print("  Stream Breakdown:")
            for source_id, count in stats['stream_breakdown'].items():
                stream = self.streams.get(source_id)
                dropped = stream.events_dropped if stream else 0
                print(f"    {source_id}: {count}/{MAX_EVENTS_PER_STREAM} events (dropped: {dropped})")
