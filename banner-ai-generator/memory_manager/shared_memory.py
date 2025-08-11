"""
Shared Memory Management for inter-agent communication and data sharing
"""

import threading
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MemoryEventType(Enum):
    """Types of memory events for observers"""
    DATA_UPDATED = "data_updated"
    SESSION_CREATED = "session_created"
    SESSION_ENDED = "session_ended"
    BLUEPRINT_CREATED = "blueprint_created"
    FEEDBACK_ADDED = "feedback_added"

@dataclass
class MemoryEvent:
    """Memory event for observer pattern"""
    event_type: MemoryEventType
    session_id: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    agent_id: Optional[str] = None

class SharedMemoryManager:
    """
    Thread-safe shared memory manager for multi-agent communication
    Implements observer pattern for event-driven updates
    """
    
    def __init__(self):
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()
        self._observers: List[Callable[[MemoryEvent], None]] = []
        self._session_metadata: Dict[str, Dict[str, Any]] = {}
        
    def create_session_memory(self, session_id: str, initial_data: Optional[Dict[str, Any]] = None) -> None:
        """Create isolated memory space for a session"""
        with self._global_lock:
            if session_id in self._memory:
                logger.warning(f"Session {session_id} memory already exists")
                return
                
            self._memory[session_id] = initial_data or {}
            self._locks[session_id] = threading.RLock()
            self._session_metadata[session_id] = {
                'created_at': time.time(),
                'last_accessed': time.time(),
                'access_count': 0
            }
            
            # Notify observers
            event = MemoryEvent(
                event_type=MemoryEventType.SESSION_CREATED,
                session_id=session_id,
                data={'initial_data': initial_data}
            )
            self._notify_observers(event)
            
        logger.info(f"Created session memory for: {session_id}")
    
    @contextmanager
    def get_session_lock(self, session_id: str):
        """Context manager for thread-safe session access"""
        if session_id not in self._locks:
            with self._global_lock:
                if session_id not in self._locks:
                    raise KeyError(f"Session {session_id} not found")
        
        lock = self._locks[session_id]
        lock.acquire()
        try:
            # Update access metadata
            if session_id in self._session_metadata:
                self._session_metadata[session_id]['last_accessed'] = time.time()
                self._session_metadata[session_id]['access_count'] += 1
            yield
        finally:
            lock.release()
    
    def set_data(self, session_id: str, key: str, value: Any, agent_id: Optional[str] = None) -> None:
        """Set data in session memory with thread safety"""
        with self.get_session_lock(session_id):
            if session_id not in self._memory:
                raise KeyError(f"Session {session_id} not found")
                
            old_value = self._memory[session_id].get(key)
            self._memory[session_id][key] = value
            
            # Notify observers only if value changed
            if old_value != value:
                event = MemoryEvent(
                    event_type=MemoryEventType.DATA_UPDATED,
                    session_id=session_id,
                    data={'key': key, 'value': value, 'old_value': old_value},
                    agent_id=agent_id
                )
                self._notify_observers(event)
                
        logger.debug(f"Set {key} in session {session_id}")
    
    def get_data(self, session_id: str, key: str, default: Any = None) -> Any:
        """Get data from session memory with thread safety"""
        with self.get_session_lock(session_id):
            if session_id not in self._memory:
                raise KeyError(f"Session {session_id} not found")
            return self._memory[session_id].get(key, default)
    
    def get_all_data(self, session_id: str) -> Dict[str, Any]:
        """Get all data from session memory"""
        with self.get_session_lock(session_id):
            if session_id not in self._memory:
                raise KeyError(f"Session {session_id} not found")
            return self._memory[session_id].copy()
    
    def update_data(self, session_id: str, data: Dict[str, Any], agent_id: Optional[str] = None) -> None:
        """Update multiple keys in session memory"""
        with self.get_session_lock(session_id):
            if session_id not in self._memory:
                raise KeyError(f"Session {session_id} not found")
                
            old_data = self._memory[session_id].copy()
            self._memory[session_id].update(data)
            
            # Notify observers
            event = MemoryEvent(
                event_type=MemoryEventType.DATA_UPDATED,
                session_id=session_id,
                data={'updated_keys': list(data.keys()), 'old_data': old_data},
                agent_id=agent_id
            )
            self._notify_observers(event)
    
    def delete_data(self, session_id: str, key: str) -> bool:
        """Delete data from session memory"""
        with self.get_session_lock(session_id):
            if session_id not in self._memory:
                return False
            return self._memory[session_id].pop(key, None) is not None
    
    def clear_session(self, session_id: str) -> None:
        """Clear all data from session memory"""
        with self.get_session_lock(session_id):
            if session_id in self._memory:
                self._memory[session_id].clear()
                logger.info(f"Cleared session memory: {session_id}")
    
    def destroy_session(self, session_id: str) -> None:
        """Completely remove session memory and locks"""
        with self._global_lock:
            if session_id in self._memory:
                del self._memory[session_id]
            if session_id in self._locks:
                del self._locks[session_id]
            if session_id in self._session_metadata:
                del self._session_metadata[session_id]
                
            # Notify observers
            event = MemoryEvent(
                event_type=MemoryEventType.SESSION_ENDED,
                session_id=session_id,
                data={}
            )
            self._notify_observers(event)
            
        logger.info(f"Destroyed session memory: {session_id}")
    
    def list_sessions(self) -> List[str]:
        """List all active sessions"""
        with self._global_lock:
            return list(self._memory.keys())
    
    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata"""
        return self._session_metadata.get(session_id)
    
    def add_observer(self, observer: Callable[[MemoryEvent], None]) -> None:
        """Add observer for memory events"""
        self._observers.append(observer)
    
    def remove_observer(self, observer: Callable[[MemoryEvent], None]) -> None:
        """Remove observer"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers(self, event: MemoryEvent) -> None:
        """Notify all observers about memory events"""
        for observer in self._observers:
            try:
                observer(event)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")
