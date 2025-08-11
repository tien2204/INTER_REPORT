"""
Agent coordinator for orchestrating multi-agent workflows
"""

import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid

from .message_queue import MessageQueue, Message, MessagePriority
from .event_dispatcher import EventDispatcher, Event, EventType
from .protocol import AgentMessage, AgentResponse, MessageType, ResponseStatus, AgentType

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepStatus(Enum):
    """Individual step status"""
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    step_id: str
    name: str
    agent_type: AgentType
    action: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.WAITING
    depends_on: List[str] = field(default_factory=list)  # Step IDs this step depends on
    timeout: Optional[int] = None  # seconds
    retry_count: int = 0
    max_retries: int = 3
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_execute(self, completed_steps: set) -> bool:
        """Check if step can be executed based on dependencies"""
        return all(dep_id in completed_steps for dep_id in self.depends_on)
    
    def is_timed_out(self) -> bool:
        """Check if step has timed out"""
        if not self.timeout or not self.started_at:
            return False
        return datetime.now() > self.started_at + timedelta(seconds=self.timeout)

@dataclass
class Workflow:
    """Workflow definition and state"""
    workflow_id: str
    name: str
    session_id: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_executable_steps(self) -> List[WorkflowStep]:
        """Get steps that can be executed now"""
        completed_steps = {
            step.step_id for step in self.steps 
            if step.status == StepStatus.COMPLETED
        }
        
        return [
            step for step in self.steps
            if step.status == StepStatus.WAITING and step.can_execute(completed_steps)
        ]
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID"""
        return next((step for step in self.steps if step.step_id == step_id), None)
    
    def is_completed(self) -> bool:
        """Check if workflow is completed"""
        return all(step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED] for step in self.steps)
    
    def has_failed_steps(self) -> bool:
        """Check if workflow has failed steps"""
        return any(step.status == StepStatus.FAILED for step in self.steps)

class AgentCoordinator:
    """
    Coordinates multi-agent workflows and manages inter-agent communication
    Handles workflow orchestration, error recovery, and agent lifecycle
    """
    
    def __init__(self, 
                 message_queue: MessageQueue,
                 event_dispatcher: EventDispatcher,
                 default_timeout: int = 300):  # 5 minutes default
        
        self.message_queue = message_queue
        self.event_dispatcher = event_dispatcher
        self.default_timeout = default_timeout
        
        self._workflows: Dict[str, Workflow] = {}
        self._agents: Dict[str, Dict[str, Any]] = {}  # agent_id -> agent_info
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'workflows_started': 0,
            'workflows_completed': 0,
            'workflows_failed': 0,
            'steps_executed': 0,
            'steps_failed': 0,
            'messages_sent': 0
        }
        
        # Setup message handling
        self.message_queue.subscribe("coordinator", self._handle_message)
        
        # Setup event handling
        self.event_dispatcher.subscribe(EventType.AGENT_ERROR, self._handle_agent_error)
        self.event_dispatcher.subscribe(EventType.AGENT_STARTED, self._handle_agent_started)
        self.event_dispatcher.subscribe(EventType.AGENT_STOPPED, self._handle_agent_stopped)
        
        # Start workflow processor
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        self.start()
    
    def start(self) -> None:
        """Start the coordinator"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._processor_thread = threading.Thread(
                target=self._process_workflows,
                daemon=True,
                name="WorkflowProcessor"
            )
            self._processor_thread.start()
            logger.info("Agent coordinator started")
    
    def stop(self) -> None:
        """Stop the coordinator"""
        with self._lock:
            self._running = False
            if self._processor_thread and self._processor_thread.is_alive():
                self._processor_thread.join(timeout=5.0)
            logger.info("Agent coordinator stopped")
    
    def register_agent(self, 
                      agent_id: str, 
                      agent_type: AgentType,
                      capabilities: List[str],
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register an agent with the coordinator"""
        with self._lock:
            self._agents[agent_id] = {
                'agent_type': agent_type,
                'capabilities': capabilities,
                'status': 'active',
                'registered_at': datetime.now(),
                'last_seen': datetime.now(),
                'metadata': metadata or {}
            }
            
        logger.info(f"Registered agent: {agent_id} ({agent_type.value})")
        
        # Dispatch event
        self.event_dispatcher.dispatch_event(
            EventType.AGENT_STARTED,
            source="coordinator",
            data={
                'agent_id': agent_id,
                'agent_type': agent_type.value,
                'capabilities': capabilities
            }
        )
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        with self._lock:
            if agent_id not in self._agents:
                return False
            
            agent_info = self._agents[agent_id]
            del self._agents[agent_id]
        
        logger.info(f"Unregistered agent: {agent_id}")
        
        # Dispatch event
        self.event_dispatcher.dispatch_event(
            EventType.AGENT_STOPPED,
            source="coordinator",
            data={
                'agent_id': agent_id,
                'agent_type': agent_info['agent_type'].value
            }
        )
        
        return True
    
    def create_workflow(self, 
                       name: str,
                       session_id: str,
                       steps: List[WorkflowStep],
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new workflow"""
        workflow_id = str(uuid.uuid4())
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            session_id=session_id,
            steps=steps,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._workflows[workflow_id] = workflow
            self._stats['workflows_started'] += 1
        
        logger.info(f"Created workflow: {workflow_id} ({name})")
        
        # Dispatch event
        self.event_dispatcher.dispatch_event(
            EventType.WORKFLOW_STARTED,
            source="coordinator",
            data={
                'workflow_id': workflow_id,
                'name': name,
                'session_id': session_id,
                'steps_count': len(steps)
            },
            session_id=session_id
        )
        
        return workflow_id
    
    def start_workflow(self, workflow_id: str) -> bool:
        """Start workflow execution"""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                logger.error(f"Workflow not found: {workflow_id}")
                return False
            
            if workflow.status != WorkflowStatus.PENDING:
                logger.warning(f"Workflow {workflow_id} is not in pending state")
                return False
            
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now()
        
        logger.info(f"Started workflow: {workflow_id}")
        return True
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause workflow execution"""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow or workflow.status != WorkflowStatus.RUNNING:
                return False
            
            workflow.status = WorkflowStatus.PAUSED
        
        logger.info(f"Paused workflow: {workflow_id}")
        
        # Dispatch event
        self.event_dispatcher.dispatch_event(
            EventType.WORKFLOW_PAUSED,
            source="coordinator",
            data={'workflow_id': workflow_id},
            session_id=workflow.session_id
        )
        
        return True
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume paused workflow"""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow or workflow.status != WorkflowStatus.PAUSED:
                return False
            
            workflow.status = WorkflowStatus.RUNNING
        
        logger.info(f"Resumed workflow: {workflow_id}")
        return True
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel workflow execution"""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                return False
            
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.now()
        
        logger.info(f"Cancelled workflow: {workflow_id}")
        return True
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID"""
        return self._workflows.get(workflow_id)
    
    def list_workflows(self, 
                      session_id: Optional[str] = None,
                      status: Optional[WorkflowStatus] = None) -> List[Workflow]:
        """List workflows with optional filtering"""
        with self._lock:
            workflows = list(self._workflows.values())
        
        if session_id:
            workflows = [w for w in workflows if w.session_id == session_id]
        
        if status:
            workflows = [w for w in workflows if w.status == status]
        
        return workflows
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information"""
        return self._agents.get(agent_id)
    
    def list_agents(self, agent_type: Optional[AgentType] = None) -> List[Dict[str, Any]]:
        """List registered agents"""
        with self._lock:
            agents = [
                {'agent_id': aid, **info} 
                for aid, info in self._agents.items()
            ]
        
        if agent_type:
            agents = [a for a in agents if a['agent_type'] == agent_type]
        
        return agents
    
    def send_message_to_agent(self, 
                             target_agent: str,
                             action: str,
                             payload: Dict[str, Any],
                             session_id: Optional[str] = None,
                             timeout: Optional[int] = None,
                             priority: MessagePriority = MessagePriority.NORMAL) -> Optional[str]:
        """Send message to specific agent"""
        message = AgentMessage(
            from_agent="coordinator",
            to_agent=target_agent,
            message_type=MessageType.REQUEST,
            action=action,
            payload=payload,
            session_id=session_id,
            timeout=timeout or self.default_timeout
        )
        
        success = self.message_queue.send_message(message, priority)
        if success:
            with self._lock:
                self._stats['messages_sent'] += 1
                self._active_requests[message.message_id] = {
                    'target_agent': target_agent,
                    'action': action,
                    'sent_at': datetime.now(),
                    'timeout': timeout or self.default_timeout
                }
            
            logger.debug(f"Sent message to {target_agent}: {message.message_id}")
            return message.message_id
        
        return None
    
    def wait_for_response(self, request_id: str, timeout: Optional[int] = None) -> Optional[AgentResponse]:
        """Wait for response to a request"""
        response = self.message_queue.wait_for_response(
            request_id, 
            timeout or self.default_timeout
        )
        
        with self._lock:
            if request_id in self._active_requests:
                del self._active_requests[request_id]
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        with self._lock:
            return {
                **self._stats,
                'active_workflows': len([w for w in self._workflows.values() 
                                       if w.status == WorkflowStatus.RUNNING]),
                'total_workflows': len(self._workflows),
                'registered_agents': len(self._agents),
                'active_requests': len(self._active_requests)
            }
    
    def _process_workflows(self) -> None:
        """Process active workflows"""
        while self._running:
            try:
                with self._lock:
                    active_workflows = [
                        w for w in self._workflows.values()
                        if w.status == WorkflowStatus.RUNNING
                    ]
                
                for workflow in active_workflows:
                    self._process_workflow(workflow)
                
                time.sleep(1.0)  # Process every second
                
            except Exception as e:
                logger.error(f"Error processing workflows: {e}")
    
    def _process_workflow(self, workflow: Workflow) -> None:
        """Process individual workflow"""
        try:
            # Check for timed out steps
            self._check_step_timeouts(workflow)
            
            # Get executable steps
            executable_steps = workflow.get_executable_steps()
            
            for step in executable_steps:
                self._execute_step(workflow, step)
            
            # Check if workflow is completed
            if workflow.is_completed():
                self._complete_workflow(workflow)
            elif workflow.has_failed_steps():
                self._fail_workflow(workflow)
                
        except Exception as e:
            logger.error(f"Error processing workflow {workflow.workflow_id}: {e}")
            self._fail_workflow(workflow, str(e))
    
    def _execute_step(self, workflow: Workflow, step: WorkflowStep) -> None:
        """Execute individual workflow step"""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()
        
        logger.info(f"Executing step: {step.name} ({step.step_id})")
        
        # Find agent for this step
        target_agent = self._find_agent_for_step(step)
        if not target_agent:
            step.status = StepStatus.FAILED
            step.error = f"No available agent for {step.agent_type.value}"
            logger.error(f"No agent found for step: {step.step_id}")
            return
        
        # Send message to agent
        request_id = self.send_message_to_agent(
            target_agent=target_agent,
            action=step.action,
            payload=step.inputs,
            session_id=workflow.session_id,
            timeout=step.timeout
        )
        
        if not request_id:
            step.status = StepStatus.FAILED
            step.error = "Failed to send message to agent"
            return
        
        # Store request info for tracking
        step.metadata['request_id'] = request_id
        step.metadata['target_agent'] = target_agent
        
        with self._lock:
            self._stats['steps_executed'] += 1
    
    def _find_agent_for_step(self, step: WorkflowStep) -> Optional[str]:
        """Find available agent for workflow step"""
        with self._lock:
            candidates = [
                agent_id for agent_id, info in self._agents.items()
                if (info['agent_type'] == step.agent_type and 
                    info['status'] == 'active' and
                    step.action in info['capabilities'])
            ]
        
        # Simple round-robin selection (could be improved with load balancing)
        return candidates[0] if candidates else None
    
    def _check_step_timeouts(self, workflow: Workflow) -> None:
        """Check for timed out steps"""
        for step in workflow.steps:
            if step.status == StepStatus.RUNNING and step.is_timed_out():
                step.status = StepStatus.FAILED
                step.error = "Step timed out"
                step.completed_at = datetime.now()
                
                logger.warning(f"Step timed out: {step.step_id}")
                
                with self._lock:
                    self._stats['steps_failed'] += 1
    
    def _complete_workflow(self, workflow: Workflow) -> None:
        """Mark workflow as completed"""
        workflow.status = WorkflowStatus.COMPLETED
        workflow.completed_at = datetime.now()
        
        with self._lock:
            self._stats['workflows_completed'] += 1
        
        logger.info(f"Workflow completed: {workflow.workflow_id}")
        
        # Dispatch event
        self.event_dispatcher.dispatch_event(
            EventType.WORKFLOW_COMPLETED,
            source="coordinator",
            data={
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'duration': (workflow.completed_at - workflow.started_at).total_seconds() if workflow.started_at else 0
            },
            session_id=workflow.session_id
        )
    
    def _fail_workflow(self, workflow: Workflow, error: Optional[str] = None) -> None:
        """Mark workflow as failed"""
        workflow.status = WorkflowStatus.FAILED
        workflow.completed_at = datetime.now()
        workflow.error = error
        
        with self._lock:
            self._stats['workflows_failed'] += 1
        
        logger.error(f"Workflow failed: {workflow.workflow_id} - {error}")
        
        # Dispatch event
        self.event_dispatcher.dispatch_event(
            EventType.WORKFLOW_FAILED,
            source="coordinator",
            data={
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'error': error
            },
            session_id=workflow.session_id
        )
    
    def _handle_message(self, message: Message) -> None:
        """Handle incoming messages"""
        try:
            agent_message = message.content
            
            if agent_message.message_type == MessageType.RESPONSE:
                self._handle_agent_response(agent_message)
            elif agent_message.message_type == MessageType.STATUS_UPDATE:
                self._handle_status_update(agent_message)
            elif agent_message.message_type == MessageType.ERROR:
                self._handle_agent_error_message(agent_message)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _handle_agent_response(self, message: AgentMessage) -> None:
        """Handle agent response"""
        # Find the workflow step this response belongs to
        workflow_step = self._find_step_by_request_id(message.correlation_id or message.message_id)
        
        if workflow_step:
            workflow, step = workflow_step
            
            # Update step status based on response
            response_data = message.payload
            if response_data.get('status') == 'success':
                step.status = StepStatus.COMPLETED
                step.outputs = response_data.get('result', {})
            else:
                step.status = StepStatus.FAILED
                step.error = response_data.get('error', 'Unknown error')
                
                with self._lock:
                    self._stats['steps_failed'] += 1
            
            step.completed_at = datetime.now()
            logger.debug(f"Updated step {step.step_id} with response")
    
    def _handle_status_update(self, message: AgentMessage) -> None:
        """Handle agent status update"""
        agent_id = message.from_agent
        status_data = message.payload
        
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id]['last_seen'] = datetime.now()
                self._agents[agent_id]['status'] = status_data.get('status', 'active')
    
    def _handle_agent_error_message(self, message: AgentMessage) -> None:
        """Handle agent error message"""
        logger.error(f"Agent error from {message.from_agent}: {message.payload}")
        
        # Mark agent as error state
        with self._lock:
            if message.from_agent in self._agents:
                self._agents[message.from_agent]['status'] = 'error'
                self._agents[message.from_agent]['last_error'] = message.payload
    
    def _handle_agent_error(self, event: Event) -> None:
        """Handle agent error event"""
        agent_id = event.data.get('agent_id')
        if agent_id:
            with self._lock:
                if agent_id in self._agents:
                    self._agents[agent_id]['status'] = 'error'
                    self._agents[agent_id]['last_error'] = event.data
    
    def _handle_agent_started(self, event: Event) -> None:
        """Handle agent started event"""
        agent_id = event.data.get('agent_id')
        if agent_id:
            with self._lock:
                if agent_id in self._agents:
                    self._agents[agent_id]['status'] = 'active'
                    self._agents[agent_id]['last_seen'] = datetime.now()
    
    def _handle_agent_stopped(self, event: Event) -> None:
        """Handle agent stopped event"""
        agent_id = event.data.get('agent_id')
        if agent_id:
            with self._lock:
                if agent_id in self._agents:
                    self._agents[agent_id]['status'] = 'stopped'
    
    def _find_step_by_request_id(self, request_id: str) -> Optional[tuple]:
        """Find workflow step by request ID"""
        for workflow in self._workflows.values():
            for step in workflow.steps:
                if step.metadata.get('request_id') == request_id:
                    return workflow, step
        return None
