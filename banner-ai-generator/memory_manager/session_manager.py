"""
Session management for campaigns and design workflows
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

from .shared_memory import SharedMemoryManager, MemoryEvent, MemoryEventType
from .memory_store import MemoryStore, PersistentStore, CampaignData

logger = logging.getLogger(__name__)

class SessionStatus(Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"
    ERROR = "error"

@dataclass
class Session:
    """Session data structure"""
    session_id: str
    campaign_id: Optional[str]
    name: str
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.now()

class SessionManager:
    """
    Manages sessions for campaigns and design workflows
    Handles session lifecycle, expiration, and cleanup
    """
    
    def __init__(self, 
                 shared_memory: SharedMemoryManager,
                 memory_store: Optional[MemoryStore] = None,
                 persistent_store: Optional[PersistentStore] = None,
                 default_timeout_hours: int = 24):
        
        self.shared_memory = shared_memory
        self.memory_store = memory_store or MemoryStore()
        self.persistent_store = persistent_store
        self.default_timeout_hours = default_timeout_hours
        
        self._sessions: Dict[str, Session] = {}
        self._campaign_sessions: Dict[str, List[str]] = {}  # campaign_id -> session_ids
        
        # Subscribe to memory events
        self.shared_memory.add_observer(self._handle_memory_event)
    
    def create_session(self, 
                      name: str,
                      campaign_id: Optional[str] = None,
                      timeout_hours: Optional[int] = None,
                      initial_data: Optional[Dict[str, Any]] = None) -> Session:
        """Create a new session"""
        
        session_id = str(uuid.uuid4())
        timeout = timeout_hours or self.default_timeout_hours
        expires_at = datetime.now() + timedelta(hours=timeout)
        
        session = Session(
            session_id=session_id,
            campaign_id=campaign_id,
            name=name,
            expires_at=expires_at,
            metadata=initial_data or {}
        )
        
        # Store session
        self._sessions[session_id] = session
        
        # Track campaign sessions
        if campaign_id:
            if campaign_id not in self._campaign_sessions:
                self._campaign_sessions[campaign_id] = []
            self._campaign_sessions[campaign_id].append(session_id)
        
        # Create shared memory for session
        self.shared_memory.create_session_memory(session_id, initial_data)
        
        logger.info(f"Created session: {session_id} for campaign: {campaign_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        session = self._sessions.get(session_id)
        if session and not session.is_expired():
            session.update_activity()
            return session
        elif session and session.is_expired():
            self._expire_session(session_id)
        return None
    
    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session metadata"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)
            else:
                session.metadata[key] = value
        
        session.update_activity()
        logger.debug(f"Updated session: {session_id}")
        return True
    
    def extend_session(self, session_id: str, hours: int) -> bool:
        """Extend session expiration time"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.expires_at = datetime.now() + timedelta(hours=hours)
        session.update_activity()
        logger.info(f"Extended session {session_id} by {hours} hours")
        return True
    
    def pause_session(self, session_id: str) -> bool:
        """Pause a session"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.status = SessionStatus.PAUSED
        session.update_activity()
        logger.info(f"Paused session: {session_id}")
        return True
    
    def resume_session(self, session_id: str) -> bool:
        """Resume a paused session"""
        session = self._sessions.get(session_id)
        if not session or session.status != SessionStatus.PAUSED:
            return False
        
        if session.is_expired():
            self._expire_session(session_id)
            return False
        
        session.status = SessionStatus.ACTIVE
        session.update_activity()
        logger.info(f"Resumed session: {session_id}")
        return True
    
    def complete_session(self, session_id: str, save_to_persistent: bool = True) -> bool:
        """Mark session as completed"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.status = SessionStatus.COMPLETED
        session.update_activity()
        
        # Save to persistent storage if requested
        if save_to_persistent and self.persistent_store:
            self._save_session_data(session_id)
        
        logger.info(f"Completed session: {session_id}")
        return True
    
    def end_session(self, session_id: str, save_to_persistent: bool = False) -> bool:
        """End and cleanup session"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        # Save to persistent storage if requested
        if save_to_persistent and self.persistent_store:
            self._save_session_data(session_id)
        
        # Remove from campaign tracking
        if session.campaign_id and session.campaign_id in self._campaign_sessions:
            campaign_sessions = self._campaign_sessions[session.campaign_id]
            if session_id in campaign_sessions:
                campaign_sessions.remove(session_id)
                if not campaign_sessions:
                    del self._campaign_sessions[session.campaign_id]
        
        # Cleanup shared memory
        self.shared_memory.destroy_session(session_id)
        
        # Remove session
        del self._sessions[session_id]
        
        logger.info(f"Ended session: {session_id}")
        return True
    
    def list_sessions(self, campaign_id: Optional[str] = None, 
                     status: Optional[SessionStatus] = None) -> List[Session]:
        """List sessions with optional filtering"""
        sessions = []
        
        if campaign_id:
            session_ids = self._campaign_sessions.get(campaign_id, [])
            sessions = [self._sessions[sid] for sid in session_ids if sid in self._sessions]
        else:
            sessions = list(self._sessions.values())
        
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        # Filter out expired sessions
        active_sessions = []
        for session in sessions:
            if session.is_expired():
                self._expire_session(session.session_id)
            else:
                active_sessions.append(session)
        
        return active_sessions
    
    def get_campaign_sessions(self, campaign_id: str) -> List[Session]:
        """Get all sessions for a campaign"""
        return self.list_sessions(campaign_id=campaign_id)
    
    def cleanup_expired_sessions(self) -> int:
        """Cleanup expired sessions"""
        expired_count = 0
        expired_sessions = []
        
        for session_id, session in self._sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self._expire_session(session_id)
            expired_count += 1
        
        logger.info(f"Cleaned up {expired_count} expired sessions")
        return expired_count
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        total_sessions = len(self._sessions)
        active_sessions = len([s for s in self._sessions.values() if s.status == SessionStatus.ACTIVE])
        paused_sessions = len([s for s in self._sessions.values() if s.status == SessionStatus.PAUSED])
        completed_sessions = len([s for s in self._sessions.values() if s.status == SessionStatus.COMPLETED])
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'paused_sessions': paused_sessions,
            'completed_sessions': completed_sessions,
            'total_campaigns': len(self._campaign_sessions),
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _expire_session(self, session_id: str) -> None:
        """Mark session as expired and cleanup if needed"""
        if session_id in self._sessions:
            self._sessions[session_id].status = SessionStatus.EXPIRED
            logger.info(f"Session expired: {session_id}")
            
            # Optionally cleanup expired session after some time
            # For now, just mark as expired but keep in memory
    
    def _save_session_data(self, session_id: str) -> None:
        """Save session data to persistent storage"""
        if not self.persistent_store:
            return
        
        try:
            # Get all session data from shared memory
            session_data = self.shared_memory.get_all_data(session_id)
            
            # Save design versions if any
            if 'design_versions' in session_data:
                for version_data in session_data['design_versions']:
                    # This would be handled by the specific agents
                    pass
            
            # Save feedback if any
            if 'feedback' in session_data:
                for feedback in session_data['feedback']:
                    self.persistent_store.save_feedback(session_id, feedback)
            
            logger.debug(f"Saved session data to persistent storage: {session_id}")
            
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
    
    def _handle_memory_event(self, event: MemoryEvent) -> None:
        """Handle memory events from shared memory"""
        try:
            # Update session activity on any memory event
            if event.session_id in self._sessions:
                self._sessions[event.session_id].update_activity()
            
            # Handle specific event types
            if event.event_type == MemoryEventType.BLUEPRINT_CREATED:
                logger.info(f"Blueprint created in session: {event.session_id}")
            elif event.event_type == MemoryEventType.FEEDBACK_ADDED:
                logger.info(f"Feedback added to session: {event.session_id}")
                
        except Exception as e:
            logger.error(f"Error handling memory event: {e}")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB (rough approximation)"""
        import sys
        total_size = 0
        
        # Estimate size of sessions
        total_size += sys.getsizeof(self._sessions)
        for session in self._sessions.values():
            total_size += sys.getsizeof(session.__dict__)
        
        # Estimate shared memory usage
        for session_id in self._sessions.keys():
            try:
                data = self.shared_memory.get_all_data(session_id)
                total_size += sys.getsizeof(str(data))
            except:
                pass
        
        return total_size / (1024 * 1024)  # Convert to MB
