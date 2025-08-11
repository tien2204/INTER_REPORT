"""
Event-driven communication system for system-wide events and notifications
"""

import threading
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class EventType(Enum):
    """System event types"""
    # Workflow events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_PAUSED = "workflow_paused"
    
    # Agent events
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_ERROR = "agent_error"
    AGENT_IDLE = "agent_idle"
    
    # Design events
    DESIGN_CREATED = "design_created"
    DESIGN_UPDATED = "design_updated"
    DESIGN_APPROVED = "design_approved"
    DESIGN_REJECTED = "design_rejected"
    
    # System events
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_INFO = "system_info"
    
    # Custom events
    CUSTOM = "custom"

@dataclass
class Event:
    """Event data structure"""
    event_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    event_type: EventType = EventType.CUSTOM
    source: str = ""
    target: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    priority: int = 0  # Higher number = higher priority
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'source': self.source,
            'target': self.target,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'priority': self.priority,
            'metadata': self.metadata
        }

class EventDispatcher:
    """
    Event dispatcher for system-wide event handling
    Supports both synchronous and asynchronous event processing
    """
    
    def __init__(self, max_event_history: int = 1000):
        self.max_event_history = max_event_history
        self._listeners: Dict[EventType, List[Callable[[Event], None]]] = defaultdict(list)
        self._async_listeners: Dict[EventType, List[Callable[[Event], Any]]] = defaultdict(list)
        self._wildcard_listeners: List[Callable[[Event], None]] = []
        self._event_history: List[Event] = []
        self._event_filters: List[Callable[[Event], bool]] = []
        self._lock = threading.RLock()
        self._stats = {
            'events_dispatched': 0,
            'events_filtered': 0,
            'listener_errors': 0
        }
    
    def subscribe(self, 
                 event_type: EventType, 
                 listener: Callable[[Event], None],
                 async_listener: bool = False) -> None:
        """Subscribe to specific event type"""
        with self._lock:
            if async_listener:
                self._async_listeners[event_type].append(listener)
            else:
                self._listeners[event_type].append(listener)
            logger.debug(f"Subscribed to {event_type.value} events")
    
    def subscribe_all(self, listener: Callable[[Event], None]) -> None:
        """Subscribe to all events (wildcard listener)"""
        with self._lock:
            self._wildcard_listeners.append(listener)
            logger.debug("Subscribed to all events")
    
    def unsubscribe(self, 
                   event_type: EventType, 
                   listener: Callable[[Event], None],
                   async_listener: bool = False) -> bool:
        """Unsubscribe from event type"""
        with self._lock:
            listeners = self._async_listeners if async_listener else self._listeners
            if event_type in listeners and listener in listeners[event_type]:
                listeners[event_type].remove(listener)
                logger.debug(f"Unsubscribed from {event_type.value} events")
                return True
            return False
    
    def unsubscribe_all(self, listener: Callable[[Event], None]) -> bool:
        """Unsubscribe from all events"""
        with self._lock:
            if listener in self._wildcard_listeners:
                self._wildcard_listeners.remove(listener)
                logger.debug("Unsubscribed from all events")
                return True
            return False
    
    def add_filter(self, filter_func: Callable[[Event], bool]) -> None:
        """Add event filter (return True to allow event)"""
        with self._lock:
            self._event_filters.append(filter_func)
            logger.debug("Added event filter")
    
    def remove_filter(self, filter_func: Callable[[Event], bool]) -> bool:
        """Remove event filter"""
        with self._lock:
            if filter_func in self._event_filters:
                self._event_filters.remove(filter_func)
                logger.debug("Removed event filter")
                return True
            return False
    
    def dispatch(self, event: Event) -> None:
        """Dispatch event to all subscribers"""
        with self._lock:
            # Apply filters
            for filter_func in self._event_filters:
                try:
                    if not filter_func(event):
                        self._stats['events_filtered'] += 1
                        logger.debug(f"Event filtered: {event.event_id}")
                        return
                except Exception as e:
                    logger.error(f"Error in event filter: {e}")
            
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self.max_event_history:
                self._event_history.pop(0)
            
            self._stats['events_dispatched'] += 1
            logger.debug(f"Dispatching event: {event.event_type.value}")
        
        # Notify synchronous listeners
        self._notify_sync_listeners(event)
        
        # Notify asynchronous listeners
        self._notify_async_listeners(event)
    
    def dispatch_event(self, 
                      event_type: EventType,
                      source: str,
                      data: Optional[Dict[str, Any]] = None,
                      target: Optional[str] = None,
                      session_id: Optional[str] = None,
                      priority: int = 0) -> str:
        """Convenience method to create and dispatch event"""
        event = Event(
            event_type=event_type,
            source=source,
            target=target,
            data=data or {},
            session_id=session_id,
            priority=priority
        )
        
        self.dispatch(event)
        return event.event_id
    
    def get_event_history(self, 
                         event_type: Optional[EventType] = None,
                         source: Optional[str] = None,
                         session_id: Optional[str] = None,
                         limit: Optional[int] = None) -> List[Event]:
        """Get event history with optional filtering"""
        with self._lock:
            events = self._event_history.copy()
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        
        # Sort by timestamp (most recent first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        if limit:
            events = events[:limit]
        
        return events
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event dispatcher statistics"""
        with self._lock:
            return {
                **self._stats,
                'total_listeners': sum(len(listeners) for listeners in self._listeners.values()),
                'total_async_listeners': sum(len(listeners) for listeners in self._async_listeners.values()),
                'wildcard_listeners': len(self._wildcard_listeners),
                'event_history_size': len(self._event_history),
                'active_filters': len(self._event_filters)
            }
    
    def clear_history(self) -> None:
        """Clear event history"""
        with self._lock:
            self._event_history.clear()
            logger.info("Event history cleared")
    
    def _notify_sync_listeners(self, event: Event) -> None:
        """Notify synchronous listeners"""
        # Specific listeners
        listeners = []
        with self._lock:
            listeners.extend(self._listeners.get(event.event_type, []))
            listeners.extend(self._wildcard_listeners)
        
        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                self._stats['listener_errors'] += 1
                logger.error(f"Error in event listener: {e}")
    
    def _notify_async_listeners(self, event: Event) -> None:
        """Notify asynchronous listeners"""
        async_listeners = []
        with self._lock:
            async_listeners.extend(self._async_listeners.get(event.event_type, []))
        
        if async_listeners:
            # Run async listeners in background
            threading.Thread(
                target=self._run_async_listeners,
                args=(event, async_listeners),
                daemon=True
            ).start()
    
    def _run_async_listeners(self, event: Event, listeners: List[Callable[[Event], Any]]) -> None:
        """Run asynchronous listeners"""
        async def run_listeners():
            for listener in listeners:
                try:
                    result = listener(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    self._stats['listener_errors'] += 1
                    logger.error(f"Error in async event listener: {e}")
        
        # Run async listeners
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, schedule the coroutine
                asyncio.create_task(run_listeners())
            else:
                # If no loop is running, run it
                loop.run_until_complete(run_listeners())
        except RuntimeError:
            # No event loop, create new one
            asyncio.run(run_listeners())
