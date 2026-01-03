"""GameEventProducer: wires gaming services to PulseBus.

This module bridges the gap between disk-observer services (Activity/Input/Network)
and the PulseBus event routing system, enabling gaming data to flow to JSONL files.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Optional, Dict, Any

_logger = logging.getLogger(__name__)


class GameEventProducer:
    """Polls gaming services and publishes events to PulseBus.
    
    Design:
    - Single background thread polls services at fixed interval
    - Enriches events with session_id, timestamp, source metadata
    - Never crashes: all exceptions caught and logged
    - Clean lifecycle: start/stop with proper service management
    """
    
    def __init__(self, pulse_bus, poll_interval: float = 0.1):
        """Initialize producer with PulseBus reference.
        
        Args:
            pulse_bus: PulseBus instance to publish events to
            poll_interval: Seconds between service polls (default 0.1 = 100ms)
        """
        self.pulse_bus = pulse_bus
        self.poll_interval = float(poll_interval)
        self.session_id = self._create_session_id()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._services: Dict[str, Any] = {}
    
    def start(self) -> None:
        """Start background polling thread and initialize services."""
        if self._running:
            _logger.warning('GameEventProducer already running')
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name='GameEventProducer',
            daemon=True
        )
        self._thread.start()
        _logger.info(f'GameEventProducer started (session_id={self.session_id})')
    
    def stop(self) -> None:
        """Stop polling thread and cleanup services."""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        
        # Stop all services
        for name, service in self._services.items():
            try:
                service.stop()
            except Exception:
                _logger.exception(f'Failed to stop service: {name}')
        
        _logger.info('GameEventProducer stopped')
    
    def emit(self, event_type: str, data: Dict[str, Any], category: str = 'game', prefix: str = 'events') -> None:
        """Publish single event to PulseBus.
        
        Enriches event with metadata:
        - session_id: unique session identifier
        - timestamp: current epoch time
        - event_type: type of event
        - source: always 'game_producer'
        """
        try:
            enriched_event = {
                'session_id': self.session_id,
                'timestamp': time.time(),
                'event_type': event_type,
                'source': 'game_producer',
                **data  # Merge original data
            }
            
            self.pulse_bus.publish(
                event=enriched_event,
                category=category,
                prefix=prefix
            )
        except Exception:
            _logger.exception(f'Failed to emit event: {event_type}')
            # Never crash - event lost but producer continues
    
    def _poll_loop(self) -> None:
        """Background loop: poll services and emit events."""
        # Lazy import to avoid circular dependencies
        try:
            from loggers.activity_service import ActivityService
            from loggers.input_service import InputService
            from loggers.network_service import NetworkService
        except Exception:
            _logger.exception('Failed to import gaming services')
            return
        
        # Initialize services
        try:
            self._services = {
                'activity': ActivityService(),
                'input': InputService(),
                'network': NetworkService()
            }
            
            for name, service in self._services.items():
                service.start()
                _logger.info(f'Started service: {name}')
        except Exception:
            _logger.exception('Failed to initialize services')
            return
        
        # Poll loop
        while self._running:
            for service_name, service in self._services.items():
                try:
                    result = service.poll()
                    if result:
                        # Handle list or single dict
                        events = result if isinstance(result, list) else [result]
                        for event in events:
                            self.emit(
                                event_type=f'{service_name}_event',
                                data={'service': service_name, 'payload': event}
                            )
                except Exception:
                    _logger.exception(f'Service poll failed: {service_name}')
                    # Continue with other services
            
            time.sleep(self.poll_interval)
    
    def _create_session_id(self) -> str:
        """Generate unique session identifier."""
        return f'session_{uuid.uuid4().hex[:8]}_{int(time.time())}'
