"""
Communication Protocols Module

Defines communication protocols and message structures
for inter-agent communication.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class MessageType(Enum):
    """Message types for inter-agent communication"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class Priority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventType(Enum):
    """Event types for system events"""
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_ERROR = "agent_error"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    ITERATION_CREATED = "iteration_created"
    FEEDBACK_RECEIVED = "feedback_received"


@dataclass
class Message:
    """Base message structure for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    sender: str = ""
    recipient: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl: Optional[int] = None  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Event:
    """Event structure for system events"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.AGENT_STARTED
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None


class MessageProtocol:
    """
    Protocol definitions for inter-agent messages
    """
    
    @staticmethod
    def create_request(sender: str, recipient: str, action: str, 
                      data: Dict[str, Any] = None, 
                      priority: Priority = Priority.NORMAL,
                      correlation_id: str = None) -> Message:
        """Create a request message"""
        payload = {
            "action": action,
            "data": data or {}
        }
        
        return Message(
            type=MessageType.REQUEST,
            sender=sender,
            recipient=recipient,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id
        )
    
    @staticmethod
    def create_response(original_message: Message, sender: str,
                       success: bool, data: Dict[str, Any] = None,
                       error: str = None) -> Message:
        """Create a response message"""
        payload = {
            "success": success,
            "data": data or {},
            "error": error
        }
        
        return Message(
            type=MessageType.RESPONSE,
            sender=sender,
            recipient=original_message.sender,
            payload=payload,
            correlation_id=original_message.id,
            reply_to=original_message.id
        )
    
    @staticmethod
    def create_notification(sender: str, event_type: str,
                          data: Dict[str, Any] = None,
                          recipients: List[str] = None) -> Message:
        """Create a notification message"""
        payload = {
            "event_type": event_type,
            "data": data or {},
            "recipients": recipients or []
        }
        
        return Message(
            type=MessageType.NOTIFICATION,
            sender=sender,
            payload=payload
        )
    
    @staticmethod
    def create_error(sender: str, recipient: str, error_code: str,
                    error_message: str, original_message_id: str = None) -> Message:
        """Create an error message"""
        payload = {
            "error_code": error_code,
            "error_message": error_message,
            "original_message_id": original_message_id
        }
        
        return Message(
            type=MessageType.ERROR,
            sender=sender,
            recipient=recipient,
            payload=payload,
            priority=Priority.HIGH
        )
    
    @staticmethod
    def create_heartbeat(sender: str, status: str = "alive",
                        data: Dict[str, Any] = None) -> Message:
        """Create a heartbeat message"""
        payload = {
            "status": status,
            "data": data or {}
        }
        
        return Message(
            type=MessageType.HEARTBEAT,
            sender=sender,
            payload=payload,
            priority=Priority.LOW
        )


class EventProtocol:
    """
    Protocol definitions for system events
    """
    
    @staticmethod
    def create_agent_event(event_type: EventType, agent_id: str,
                          data: Dict[str, Any] = None,
                          correlation_id: str = None) -> Event:
        """Create an agent-related event"""
        return Event(
            type=event_type,
            source=agent_id,
            data=data or {},
            correlation_id=correlation_id
        )
    
    @staticmethod
    def create_task_event(event_type: EventType, agent_id: str,
                         task_id: str, task_type: str,
                         data: Dict[str, Any] = None) -> Event:
        """Create a task-related event"""
        additional_data = data or {}
        event_data = {
            "task_id": task_id,
            "task_type": task_type,
        }
        event_data.update(additional_data)  # Fixed syntax error
        
        return Event(
            type=event_type,
            source=agent_id,
            data=event_data
        )
    
    @staticmethod
    def create_workflow_event(event_type: EventType, workflow_id: str,
                            campaign_id: str, current_step: str = None,
                            data: Dict[str, Any] = None) -> Event:
        """Create a workflow-related event"""
        additional_data = data or {}
        event_data = {
            "workflow_id": workflow_id,
            "campaign_id": campaign_id,
            "current_step": current_step,
        }
        event_data.update(additional_data)  # Fixed syntax error
        
        return Event(
            type=event_type,
            source="workflow_manager",
            data=event_data
        )
    
    @staticmethod
    def create_iteration_event(campaign_id: str, iteration_id: str,
                             iteration_number: int, status: str,
                             data: Dict[str, Any] = None) -> Event:
        """Create an iteration-related event"""
        additional_data = data or {}
        event_data = {
            "campaign_id": campaign_id,
            "iteration_id": iteration_id,
            "iteration_number": iteration_number,
            "status": status,
        }
        event_data.update(additional_data)  # Fixed syntax error
        
        return Event(
            type=EventType.ITERATION_CREATED,
            source="design_system",
            data=event_data
        )


class WorkflowProtocol:
    """
    Protocol definitions for workflow-specific messages
    """
    # Define workflow channels
    CHANNELS = {
        "background_design": "workflow.background_design",
        "foreground_design": "workflow.foreground_design", 
        "error_handling": "workflow.error_handling",
        "strategist": "workflow.strategist",
        "composition": "workflow.composition",
        "finalization": "workflow.finalization"
    }

    
    # Strategist Agent Messages
    ANALYZE_BRIEF = "analyze_brief"
    PROCESS_LOGO = "process_logo"
    EXTRACT_BRAND_INFO = "extract_brand_info"
    DEFINE_STRATEGY = "define_strategy"
    
    # Background Designer Messages
    GENERATE_BACKGROUND = "generate_background"
    VALIDATE_BACKGROUND = "validate_background"
    REFINE_BACKGROUND = "refine_background"
    
    # Foreground Designer Messages
    CREATE_LAYOUT = "create_layout"
    GENERATE_BLUEPRINT = "generate_blueprint"
    SELECT_TYPOGRAPHY = "select_typography"
    DESIGN_CTA = "design_cta"
    
    # Developer Messages
    COMPILE_BLUEPRINT = "compile_blueprint"
    GENERATE_SVG = "generate_svg"
    GENERATE_FIGMA_CODE = "generate_figma_code"
    RENDER_PREVIEW = "render_preview"
    
    # Reviewer Messages
    REVIEW_DESIGN = "review_design"
    PROVIDE_FEEDBACK = "provide_feedback"
    VALIDATE_DESIGN = "validate_design"
    
    # Workflow states
    STATES = {
        "INITIALIZING": "initializing",
        "STRATEGY_PLANNING": "strategy_planning", 
        "BACKGROUND_DESIGN": "background_design",
        "FOREGROUND_DESIGN": "foreground_design",
        "COMPOSITION": "composition",
        "FINALIZATION": "finalization",
        "COMPLETED": "completed",
        "FAILED": "failed"
    }

    @classmethod
    def create_workflow_request(cls, sender: str, recipient: str, action: str,
                              campaign_id: str, iteration_id: str = None,
                              data: Dict[str, Any] = None) -> Message:
        """Create a workflow-specific request"""
        payload = {
            "action": action,
            "campaign_id": campaign_id,
            "iteration_id": iteration_id,
            "data": data or {}
        }
        
        return Message(
            type=MessageType.REQUEST,
            sender=sender,
            recipient=recipient,
            payload=payload,
            priority=Priority.NORMAL
        )

    @staticmethod
    def create_workflow_message(workflow_action: str, session_id: str, 
                               agent_id: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a workflow message"""
        return {
            "type": "workflow",
            "workflow_action": workflow_action,
            "session_id": session_id,
            "agent_id": agent_id,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat(),
            "message_id": str(uuid.uuid4())
        }
    
    @staticmethod
    def create_state_transition_message(session_id: str, from_state: str, 
                                      to_state: str, agent_id: str) -> Dict[str, Any]:
        """Create a state transition message"""
        return WorkflowProtocol.create_workflow_message(
            workflow_action="state_transition",
            session_id=session_id,
            agent_id=agent_id,
            data={
                "from_state": from_state,
                "to_state": to_state,
                "transition_time": datetime.utcnow().isoformat()
            }
        )


class AgentIdentifiers:
    """Standard agent identifiers"""
    STRATEGIST = "strategist_agent"
    BACKGROUND_DESIGNER = "background_designer_agent"
    FOREGROUND_DESIGNER = "foreground_designer_agent"
    DEVELOPER = "developer_agent"
    DESIGN_REVIEWER = "design_reviewer_agent"
    WORKFLOW_MANAGER = "workflow_manager"
    API_GATEWAY = "api_gateway"


class ChannelNames:
    """Standard channel names for pub/sub"""
    AGENT_EVENTS = "agent_events"
    TASK_EVENTS = "task_events"
    WORKFLOW_EVENTS = "workflow_events"
    SYSTEM_EVENTS = "system_events"
    ERROR_EVENTS = "error_events"
    NOTIFICATIONS = "notifications"
