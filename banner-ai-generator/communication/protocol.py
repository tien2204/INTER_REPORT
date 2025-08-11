"""
Communication protocols and message structures for inter-agent communication
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages between agents"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    STATUS_UPDATE = "status_update"
    WORKFLOW_CONTROL = "workflow_control"

class AgentType(Enum):
    """Types of agents in the system"""
    STRATEGIST = "strategist"
    BACKGROUND_DESIGNER = "background_designer"
    FOREGROUND_DESIGNER = "foreground_designer"
    DEVELOPER = "developer"
    DESIGN_REVIEWER = "design_reviewer"

class ResponseStatus(Enum):
    """Response status codes"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"
    TIMEOUT = "timeout"

@dataclass
class AgentMessage:
    """Base message structure for inter-agent communication"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = ""
    to_agent: str = ""
    message_type: MessageType = MessageType.REQUEST
    action: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    timeout: Optional[int] = None  # seconds
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            'message_id': self.message_id,
            'from_agent': self.from_agent,
            'to_agent': self.to_agent,
            'message_type': self.message_type.value,
            'action': self.action,
            'payload': self.payload,
            'session_id': self.session_id,
            'correlation_id': self.correlation_id,
            'timestamp': self.timestamp.isoformat(),
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary"""
        return cls(
            message_id=data.get('message_id', str(uuid.uuid4())),
            from_agent=data.get('from_agent', ''),
            to_agent=data.get('to_agent', ''),
            message_type=MessageType(data.get('message_type', 'request')),
            action=data.get('action', ''),
            payload=data.get('payload', {}),
            session_id=data.get('session_id'),
            correlation_id=data.get('correlation_id'),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            timeout=data.get('timeout'),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            metadata=data.get('metadata', {})
        )

@dataclass
class AgentResponse:
    """Response message structure"""
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    status: ResponseStatus = ResponseStatus.SUCCESS
    result: Any = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            'response_id': self.response_id,
            'request_id': self.request_id,
            'from_agent': self.from_agent,
            'to_agent': self.to_agent,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'error_details': self.error_details,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentResponse':
        """Create response from dictionary"""
        return cls(
            response_id=data.get('response_id', str(uuid.uuid4())),
            request_id=data.get('request_id', ''),
            from_agent=data.get('from_agent', ''),
            to_agent=data.get('to_agent', ''),
            status=ResponseStatus(data.get('status', 'success')),
            result=data.get('result'),
            error=data.get('error'),
            error_details=data.get('error_details'),
            session_id=data.get('session_id'),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            processing_time=data.get('processing_time'),
            metadata=data.get('metadata', {})
        )

class CommunicationProtocol:
    """Base communication protocol interface"""
    
    def __init__(self):
        self.message_handlers: Dict[str, callable] = {}
    
    def register_handler(self, action: str, handler: callable) -> None:
        """Register message handler for specific action"""
        self.message_handlers[action] = handler
        logger.debug(f"Registered handler for action: {action}")
    
    def unregister_handler(self, action: str) -> None:
        """Unregister message handler"""
        if action in self.message_handlers:
            del self.message_handlers[action]
            logger.debug(f"Unregistered handler for action: {action}")
    
    def get_handler(self, action: str) -> Optional[callable]:
        """Get handler for action"""
        return self.message_handlers.get(action)
    
    def validate_message(self, message: AgentMessage) -> bool:
        """Validate message format and content"""
        if not message.from_agent or not message.to_agent:
            return False
        if not message.action:
            return False
        if not isinstance(message.payload, dict):
            return False
        return True

class MessageProtocol(CommunicationProtocol):
    """Protocol for standard message handling"""
    
    def __init__(self):
        super().__init__()
        self._setup_standard_handlers()
    
    def _setup_standard_handlers(self) -> None:
        """Setup standard message handlers"""
        self.register_handler("ping", self._handle_ping)
        self.register_handler("status", self._handle_status)
        self.register_handler("error", self._handle_error)
    
    def _handle_ping(self, message: AgentMessage) -> AgentResponse:
        """Handle ping message"""
        return AgentResponse(
            request_id=message.message_id,
            from_agent=message.to_agent,
            to_agent=message.from_agent,
            status=ResponseStatus.SUCCESS,
            result={"pong": True, "timestamp": datetime.now().isoformat()}
        )
    
    def _handle_status(self, message: AgentMessage) -> AgentResponse:
        """Handle status request"""
        return AgentResponse(
            request_id=message.message_id,
            from_agent=message.to_agent,
            to_agent=message.from_agent,
            status=ResponseStatus.SUCCESS,
            result={"status": "active", "timestamp": datetime.now().isoformat()}
        )
    
    def _handle_error(self, message: AgentMessage) -> AgentResponse:
        """Handle error message"""
        error_info = message.payload.get("error", "Unknown error")
        logger.error(f"Received error from {message.from_agent}: {error_info}")
        
        return AgentResponse(
            request_id=message.message_id,
            from_agent=message.to_agent,
            to_agent=message.from_agent,
            status=ResponseStatus.SUCCESS,
            result={"acknowledged": True}
        )

class WorkflowProtocol(CommunicationProtocol):
    """Protocol for workflow-specific communication"""
    
    def __init__(self):
        super().__init__()
        self._setup_workflow_handlers()
    
    def _setup_workflow_handlers(self) -> None:
        """Setup workflow-specific handlers"""
        self.register_handler("analyze_brief", self._handle_analyze_brief)
        self.register_handler("generate_background", self._handle_generate_background)
        self.register_handler("create_blueprint", self._handle_create_blueprint)
        self.register_handler("generate_code", self._handle_generate_code)
        self.register_handler("review_design", self._handle_review_design)
        self.register_handler("provide_feedback", self._handle_provide_feedback)
    
    def _handle_analyze_brief(self, message: AgentMessage) -> AgentResponse:
        """Handle brief analysis request"""
        # This would be implemented by the Strategist agent
        return self._create_workflow_response(message, "Brief analysis completed")
    
    def _handle_generate_background(self, message: AgentMessage) -> AgentResponse:
        """Handle background generation request"""
        # This would be implemented by the Background Designer agent
        return self._create_workflow_response(message, "Background generated")
    
    def _handle_create_blueprint(self, message: AgentMessage) -> AgentResponse:
        """Handle blueprint creation request"""
        # This would be implemented by the Foreground Designer agent
        return self._create_workflow_response(message, "Blueprint created")
    
    def _handle_generate_code(self, message: AgentMessage) -> AgentResponse:
        """Handle code generation request"""
        # This would be implemented by the Developer agent
        return self._create_workflow_response(message, "Code generated")
    
    def _handle_review_design(self, message: AgentMessage) -> AgentResponse:
        """Handle design review request"""
        # This would be implemented by the Design Reviewer agent
        return self._create_workflow_response(message, "Design reviewed")
    
    def _handle_provide_feedback(self, message: AgentMessage) -> AgentResponse:
        """Handle feedback provision"""
        return self._create_workflow_response(message, "Feedback provided")
    
    def _create_workflow_response(self, message: AgentMessage, result_message: str) -> AgentResponse:
        """Create standard workflow response"""
        return AgentResponse(
            request_id=message.message_id,
            from_agent=message.to_agent,
            to_agent=message.from_agent,
            status=ResponseStatus.SUCCESS,
            result={
                "message": result_message,
                "session_id": message.session_id,
                "action": message.action
            }
        )
