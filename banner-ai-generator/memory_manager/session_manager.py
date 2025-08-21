"""
Session Manager Module

Manages sessions and state for campaigns and agent workflows.
Provides session-based isolation and state persistence.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from structlog import get_logger

logger = get_logger(__name__)


class SessionStatus(Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class AgentSession:
    """Agent session data structure"""
    session_id: str
    agent_id: str
    campaign_id: str
    status: SessionStatus
    state: Dict[str, Any]
    context: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime]


@dataclass
class WorkflowSession:
    """Workflow session data structure"""
    session_id: str
    campaign_id: str
    workflow_type: str
    current_step: str
    steps_completed: List[str]
    agent_sessions: List[str]
    status: SessionStatus
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class SessionManager:
    """
    Session manager for campaigns and agent workflows
    """
    
    def __init__(self, shared_memory, session_timeout_hours: int = 24):
        self.shared_memory = shared_memory
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self._agent_sessions: Dict[str, AgentSession] = {}
        self._workflow_sessions: Dict[str, WorkflowSession] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
    
    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a session"""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]
    
    async def create_workflow_session(self, campaign_id: str, workflow_type: str, 
                                    metadata: Dict[str, Any] = None) -> str:
        """Create a new workflow session"""
        session_id = str(uuid.uuid4())
        
        workflow_session = WorkflowSession(
            session_id=session_id,
            campaign_id=campaign_id,
            workflow_type=workflow_type,
            current_step="",
            steps_completed=[],
            agent_sessions=[],
            status=SessionStatus.ACTIVE,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self._workflow_sessions[session_id] = workflow_session
        
        # Store in shared memory
        await self.shared_memory.set_agent_state(
            f"workflow_session:{session_id}",
            asdict(workflow_session)
        )
        
        logger.info(f"Workflow session created: {session_id}")
        return session_id
    
    async def create_agent_session(self, agent_id: str, campaign_id: str, 
                                 workflow_session_id: str = None,
                                 context: Dict[str, Any] = None) -> str:
        """Create a new agent session"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + self.session_timeout
        
        agent_session = AgentSession(
            session_id=session_id,
            agent_id=agent_id,
            campaign_id=campaign_id,
            status=SessionStatus.ACTIVE,
            state={},
            context=context or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            expires_at=expires_at
        )
        
        self._agent_sessions[session_id] = agent_session
        
        # Store in shared memory
        await self.shared_memory.set_agent_state(
            f"agent_session:{session_id}",
            asdict(agent_session)
        )
        
        # Add to workflow session if provided
        if workflow_session_id:
            await self.add_agent_to_workflow(workflow_session_id, session_id)
        
        logger.info(f"Agent session created: {session_id} for agent {agent_id}")
        return session_id
    
    async def get_agent_session(self, session_id: str) -> Optional[AgentSession]:
        """Get agent session by ID"""
        try:
            # Check local cache first
            if session_id in self._agent_sessions:
                session = self._agent_sessions[session_id]
                if self._is_session_expired(session):
                    await self.expire_session(session_id)
                    return None
                return session
            
            # Get from shared memory
            session_data = await self.shared_memory.get_agent_state(
                f"agent_session:{session_id}"
            )
            
            if session_data:
                # Convert datetime strings back to datetime objects
                session_data["created_at"] = datetime.fromisoformat(session_data["created_at"])
                session_data["updated_at"] = datetime.fromisoformat(session_data["updated_at"])
                if session_data.get("expires_at"):
                    session_data["expires_at"] = datetime.fromisoformat(session_data["expires_at"])
                
                session = AgentSession(**session_data)
                
                if self._is_session_expired(session):
                    await self.expire_session(session_id)
                    return None
                
                self._agent_sessions[session_id] = session
                return session
            
            return None
        except Exception as e:
            logger.error(f"Failed to get agent session {session_id}: {e}")
            return None
    
    async def get_workflow_session(self, session_id: str) -> Optional[WorkflowSession]:
        """Get workflow session by ID"""
        try:
            # Check local cache first
            if session_id in self._workflow_sessions:
                return self._workflow_sessions[session_id]
            
            # Get from shared memory
            session_data = await self.shared_memory.get_agent_state(
                f"workflow_session:{session_id}"
            )
            
            if session_data:
                # Convert datetime strings back to datetime objects
                session_data["created_at"] = datetime.fromisoformat(session_data["created_at"])
                session_data["updated_at"] = datetime.fromisoformat(session_data["updated_at"])
                
                session = WorkflowSession(**session_data)
                self._workflow_sessions[session_id] = session
                return session
            
            return None
        except Exception as e:
            logger.error(f"Failed to get workflow session {session_id}: {e}")
            return None
    
    async def update_agent_session(self, session_id: str, **updates):
        """Update agent session"""
        async with self._get_session_lock(session_id):
            session = await self.get_agent_session(session_id)
            if session:
                # Update session data
                for key, value in updates.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                
                session.updated_at = datetime.utcnow()
                
                # Update local cache
                self._agent_sessions[session_id] = session
                
                # Update shared memory
                await self.shared_memory.set_agent_state(
                    f"agent_session:{session_id}",
                    asdict(session)
                )
                
                logger.debug(f"Agent session {session_id} updated")
    
    async def update_workflow_session(self, session_id: str, **updates):
        """Update workflow session"""
        async with self._get_session_lock(session_id):
            session = await self.get_workflow_session(session_id)
            if session:
                # Update session data
                for key, value in updates.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                
                session.updated_at = datetime.utcnow()
                
                # Update local cache
                self._workflow_sessions[session_id] = session
                
                # Update shared memory
                await self.shared_memory.set_agent_state(
                    f"workflow_session:{session_id}",
                    asdict(session)
                )
                
                logger.debug(f"Workflow session {session_id} updated")
    
    async def add_agent_to_workflow(self, workflow_session_id: str, agent_session_id: str):
        """Add agent session to workflow session"""
        workflow_session = await self.get_workflow_session(workflow_session_id)
        if workflow_session:
            if agent_session_id not in workflow_session.agent_sessions:
                workflow_session.agent_sessions.append(agent_session_id)
                await self.update_workflow_session(
                    workflow_session_id,
                    agent_sessions=workflow_session.agent_sessions
                )
    
    async def complete_workflow_step(self, session_id: str, step_name: str):
        """Mark a workflow step as completed"""
        workflow_session = await self.get_workflow_session(session_id)
        if workflow_session:
            if step_name not in workflow_session.steps_completed:
                workflow_session.steps_completed.append(step_name)
                await self.update_workflow_session(
                    session_id,
                    steps_completed=workflow_session.steps_completed
                )
                
                logger.info(f"Workflow step '{step_name}' completed in session {session_id}")
    
    async def set_workflow_step(self, session_id: str, step_name: str):
        """Set current workflow step"""
        await self.update_workflow_session(session_id, current_step=step_name)
        logger.info(f"Workflow session {session_id} moved to step '{step_name}'")
    
    def _is_session_expired(self, session: AgentSession) -> bool:
        """Check if agent session is expired"""
        if session.expires_at:
            return datetime.utcnow() > session.expires_at
        return False
    
    async def expire_session(self, session_id: str):
        """Mark session as expired"""
        await self.update_agent_session(session_id, status=SessionStatus.EXPIRED)
        logger.info(f"Agent session {session_id} expired")
    
    async def pause_session(self, session_id: str):
        """Pause a session"""
        await self.update_agent_session(session_id, status=SessionStatus.PAUSED)
        logger.info(f"Agent session {session_id} paused")
    
    async def resume_session(self, session_id: str):
        """Resume a paused session"""
        await self.update_agent_session(session_id, status=SessionStatus.ACTIVE)
        logger.info(f"Agent session {session_id} resumed")
    
    async def complete_session(self, session_id: str):
        """Mark session as completed"""
        await self.update_agent_session(session_id, status=SessionStatus.COMPLETED)
        logger.info(f"Agent session {session_id} completed")
    
    async def fail_session(self, session_id: str, error_message: str = None):
        """Mark session as failed"""
        updates = {"status": SessionStatus.FAILED}
        if error_message:
            updates["context"] = {"error": error_message}
        
        await self.update_agent_session(session_id, **updates)
        logger.error(f"Agent session {session_id} failed: {error_message}")
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self._agent_sessions.items():
            if self._is_session_expired(session):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.expire_session(session_id)
            del self._agent_sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def get_campaign_sessions(self, campaign_id: str) -> List[AgentSession]:
        """Get all active sessions for a campaign"""
        sessions = []
        for session in self._agent_sessions.values():
            if (session.campaign_id == campaign_id and 
                session.status == SessionStatus.ACTIVE and
                not self._is_session_expired(session)):
                sessions.append(session)
        
        return sessions
    
    async def get_agent_sessions_by_type(self, agent_id: str) -> List[AgentSession]:
        """Get all sessions for a specific agent type"""
        sessions = []
        for session in self._agent_sessions.values():
            if (session.agent_id == agent_id and 
                session.status == SessionStatus.ACTIVE and
                not self._is_session_expired(session)):
                sessions.append(session)
        
        return sessions
