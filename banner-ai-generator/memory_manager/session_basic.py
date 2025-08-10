from typing import Dict, Optional, List
from datetime import datetime, timedelta
import threading

class BasicSessionManager:
    """Basic session management for agent workflows"""
    
    def __init__(self, memory_manager: BasicMemoryManager, session_timeout_hours: int = 24):
        self.memory_manager = memory_manager
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def create_session(self, campaign_data: CampaignData) -> str:
        """Create new session for campaign"""
        session_id = campaign_data.session_id
        
        with self._lock:
            self._sessions[session_id] = {
                'campaign_id': campaign_data.campaign_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'status': 'active',
                'workflow_state': WorkflowState.INITIALIZED
            }
        
        # Initialize session in memory
        self.memory_manager.create_session(session_id)
        self.memory_manager.store(session_id, MemoryKey.CAMPAIGN_DATA, campaign_data)
        
        return session_id
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        with self._lock:
            session_info = self._sessions.get(session_id)
            if session_info:
                # Check if session is expired
                if datetime.now() - session_info['last_activity'] > self.session_timeout:
                    session_info['status'] = 'expired'
            return session_info.copy() if session_info else None
    
    def update_session_activity(self, session_id: str) -> None:
        """Update last activity timestamp"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]['last_activity'] = datetime.now()
    
    def update_workflow_state(self, session_id: str, new_state: WorkflowState) -> bool:
        """Update workflow state for session"""
        try:
            with self._lock:
                if session_id in self._sessions:
                    self._sessions[session_id]['workflow_state'] = new_state
                    self._sessions[session_id]['last_activity'] = datetime.now()
            
            # Also update in memory
            campaign_data = self.memory_manager.retrieve(session_id, MemoryKey.CAMPAIGN_DATA)
            if campaign_data:
                campaign_data.workflow_state = new_state
                campaign_data.updated_at = datetime.now()
                self.memory_manager.store(session_id, MemoryKey.CAMPAIGN_DATA, campaign_data)
            
            return True
        except Exception as e:
            print(f"Error updating workflow state: {e}")
            return False
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if session is active and not expired"""
        session_info = self.get_session_info(session_id)
        if not session_info:
            return False
        
        return (session_info['status'] == 'active' and 
                datetime.now() - session_info['last_activity'] <= self.session_timeout)
    
    def close_session(self, session_id: str) -> bool:
        """Close session and mark as completed"""
        try:
            with self._lock:
                if session_id in self._sessions:
                    self._sessions[session_id]['status'] = 'completed'
                    self._sessions[session_id]['last_activity'] = datetime.now()
            
            self.update_workflow_state(session_id, WorkflowState.COMPLETED)
            return True
        except Exception as e:
            print(f"Error closing session: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """Cleanup expired sessions and return count of cleaned sessions"""
        expired_count = 0
        current_time = datetime.now()
        
        with self._lock:
            expired_sessions = []
            for session_id, session_info in self._sessions.items():
                if current_time - session_info['last_activity'] > self.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                # Clean memory
                self.memory_manager.clear_session(session_id)
                # Remove session info
                del self._sessions[session_id]
                expired_count += 1
        
        return expired_count
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of all active sessions"""
        active_sessions = []
        current_time = datetime.now()
        
        with self._lock:
            for session_id, session_info in self._sessions.items():
                if (session_info['status'] == 'active' and 
                    current_time - session_info['last_activity'] <= self.session_timeout):
                    
                    session_copy = session_info.copy()
                    session_copy['session_id'] = session_id
                    active_sessions.append(session_copy)
        
        return active_sessions
