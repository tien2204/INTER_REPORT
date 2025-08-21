"""
Agent Coordinator

Central orchestration system for managing multi-agent workflows,
communication, and coordination across the banner generation pipeline.
"""

import asyncio
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from structlog import get_logger

from .protocol import Message, MessageType, Priority  # Fixed import
from .message_queue import MessageQueue
from memory_manager.shared_memory import SharedMemory

logger = get_logger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(Enum):
    """Individual agent status"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class WorkflowStep:
    """Represents a single step in the workflow"""
    step_id: str
    agent_id: str
    action: str
    dependencies: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class Workflow:
    """Represents a complete multi-agent workflow"""
    workflow_id: str
    workflow_type: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    current_step_index: int = 0


@dataclass
class AgentInfo:
    """Information about a registered agent"""
    agent_id: str
    agent_name: str
    capabilities: List[str]
    status: AgentStatus = AgentStatus.IDLE
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    current_workflow: Optional[str] = None
    current_step: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class AgentCoordinator:
    """
    Central coordinator for multi-agent workflows
    
    Capabilities:
    - Workflow orchestration and management
    - Agent registration and health monitoring
    - Dependency resolution and execution order
    - Error handling and recovery
    - Performance monitoring and optimization
    - Dynamic load balancing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core components
        self.message_queue: Optional[MessageQueue] = None
        self.shared_memory: Optional[SharedMemory] = None
        
        # Agent management
        self.registered_agents: Dict[str, AgentInfo] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # Workflow management
        self.active_workflows: Dict[str, Workflow] = {}
        self.workflow_templates: Dict[str, List[WorkflowStep]] = {}
        self.workflow_history: List[Workflow] = []
        
        # Coordination settings
        self.heartbeat_interval = config.get("heartbeat_interval", 30)  # seconds
        self.step_timeout = config.get("step_timeout", 300)  # seconds
        self.max_concurrent_workflows = config.get("max_concurrent_workflows", 10)
        self.retry_delay = config.get("retry_delay", 5)  # seconds
        
        # Performance tracking
        self.workflow_metrics: Dict[str, Any] = {}
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Control flags
        self._running = False
        self._coordinator_task: Optional[asyncio.Task] = None
        
        logger.info("Agent Coordinator initialized")
    
    async def start(self):
        """Start the agent coordinator"""
        try:
            logger.info("Starting Agent Coordinator")
            
            # Initialize workflow templates
            await self._initialize_workflow_templates()
            
            # Start coordination loop
            self._running = True
            self._coordinator_task = asyncio.create_task(self._coordination_loop())
            
            # Start heartbeat monitoring
            asyncio.create_task(self._heartbeat_monitor())
            
            # Start workflow cleanup
            asyncio.create_task(self._workflow_cleanup())
            
            logger.info("Agent Coordinator started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Agent Coordinator: {e}")
            raise
    
    async def stop(self):
        """Stop the agent coordinator"""
        try:
            logger.info("Stopping Agent Coordinator")
            
            self._running = False
            
            # Cancel coordination task
            if self._coordinator_task:
                self._coordinator_task.cancel()
                try:
                    await self._coordinator_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel active workflows
            for workflow in self.active_workflows.values():
                await self._cancel_workflow(workflow.workflow_id, "Coordinator shutdown")
            
            logger.info("Agent Coordinator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Agent Coordinator: {e}")
    
    def set_communication(self, message_queue: MessageQueue, shared_memory: SharedMemory):
        """Set communication interfaces"""
        self.message_queue = message_queue
        self.shared_memory = shared_memory
    
    async def register_agent(self, agent_id: str, agent_name: str, 
                           capabilities: List[str]) -> bool:
        """Register an agent with the coordinator"""
        try:
            logger.info(f"Registering agent: {agent_id}")
            
            agent_info = AgentInfo(
                agent_id=agent_id,
                agent_name=agent_name,
                capabilities=capabilities,
                status=AgentStatus.IDLE,
                last_heartbeat=datetime.utcnow()
            )
            
            self.registered_agents[agent_id] = agent_info
            self.agent_capabilities[agent_id] = capabilities
            
            # Initialize performance metrics
            self.agent_performance[agent_id] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "avg_execution_time": 0.0,
                "last_task_time": None
            }
            
            # Subscribe to agent messages
            if self.message_queue:
                await self.message_queue.subscribe(
                    f"agent.{agent_id}",
                    self._handle_agent_message
                )
            
            logger.info(f"Agent {agent_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering agent {agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the coordinator"""
        try:
            logger.info(f"Unregistering agent: {agent_id}")
            
            # Cancel any active workflows involving this agent
            for workflow in list(self.active_workflows.values()):
                for step in workflow.steps:
                    if step.agent_id == agent_id and step.status in ["pending", "running"]:
                        await self._cancel_workflow(workflow.workflow_id, f"Agent {agent_id} unregistered")
                        break
            
            # Remove agent
            if agent_id in self.registered_agents:
                del self.registered_agents[agent_id]
            if agent_id in self.agent_capabilities:
                del self.agent_capabilities[agent_id]
            if agent_id in self.agent_performance:
                del self.agent_performance[agent_id]
            
            logger.info(f"Agent {agent_id} unregistered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering agent {agent_id}: {e}")
            return False
    
    async def start_workflow(self, workflow_type: str, context: Dict[str, Any]) -> str:
        """Start a new workflow"""
        try:
            # Check workflow limit
            if len(self.active_workflows) >= self.max_concurrent_workflows:
                raise Exception("Maximum concurrent workflows exceeded")
            
            # Generate workflow ID
            workflow_id = str(uuid.uuid4())
            
            # Get workflow template
            template_steps = self.workflow_templates.get(workflow_type)
            if not template_steps:
                raise Exception(f"Unknown workflow type: {workflow_type}")
            
            # Create workflow steps
            steps = []
            for template_step in template_steps:
                step = WorkflowStep(
                    step_id=f"{workflow_id}_{template_step.step_id}",
                    agent_id=template_step.agent_id,
                    action=template_step.action,
                    dependencies=template_step.dependencies.copy(),
                    data=template_step.data.copy(),
                    timeout_seconds=template_step.timeout_seconds,
                    max_retries=template_step.max_retries
                )
                steps.append(step)
            
            # Create workflow
            workflow = Workflow(
                workflow_id=workflow_id,
                workflow_type=workflow_type,
                steps=steps,
                context=context
            )
            
            # Add to active workflows
            self.active_workflows[workflow_id] = workflow
            
            # Trigger workflow execution
            await self._trigger_event("workflow_started", {"workflow_id": workflow_id})
            
            logger.info(f"Started workflow {workflow_id} of type {workflow_type}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error starting workflow: {e}")
            raise
    
    async def cancel_workflow(self, workflow_id: str, reason: str = "User cancelled") -> bool:
        """Cancel an active workflow"""
        return await self._cancel_workflow(workflow_id, reason)
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                # Check history
                for historical_workflow in self.workflow_history:
                    if historical_workflow.workflow_id == workflow_id:
                        workflow = historical_workflow
                        break
            
            if not workflow:
                return None
            
            # Calculate progress
            total_steps = len(workflow.steps)
            completed_steps = len([s for s in workflow.steps if s.status == "completed"])
            progress_percentage = (completed_steps / total_steps * 100) if total_steps > 0 else 0
            
            # Get current step
            current_step = None
            if workflow.current_step_index < len(workflow.steps):
                current_step = workflow.steps[workflow.current_step_index]
            
            return {
                "workflow_id": workflow_id,
                "workflow_type": workflow.workflow_type,
                "status": workflow.status.value,
                "progress_percentage": progress_percentage,
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "current_step": {
                    "step_id": current_step.step_id if current_step else None,
                    "agent_id": current_step.agent_id if current_step else None,
                    "action": current_step.action if current_step else None,
                    "status": current_step.status if current_step else None
                } if current_step else None,
                "created_at": workflow.created_at.isoformat(),
                "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
                "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
                "error_message": workflow.error_message
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return None
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an agent"""
        try:
            agent = self.registered_agents.get(agent_id)
            if not agent:
                return None
            
            performance = self.agent_performance.get(agent_id, {})
            
            return {
                "agent_id": agent_id,
                "agent_name": agent.agent_name,
                "status": agent.status.value,
                "capabilities": agent.capabilities,
                "last_heartbeat": agent.last_heartbeat.isoformat(),
                "current_workflow": agent.current_workflow,
                "current_step": agent.current_step,
                "performance": {
                    "total_tasks": performance.get("total_tasks", 0),
                    "success_rate": (performance.get("successful_tasks", 0) / 
                                   max(performance.get("total_tasks", 1), 1) * 100),
                    "avg_execution_time": performance.get("avg_execution_time", 0.0),
                    "last_task_time": performance.get("last_task_time")
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting agent status: {e}")
            return None
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            active_agents = len([a for a in self.registered_agents.values() 
                               if a.status != AgentStatus.OFFLINE])
            
            total_workflows = len(self.active_workflows) + len(self.workflow_history)
            
            return {
                "coordinator_running": self._running,
                "registered_agents": len(self.registered_agents),
                "active_agents": active_agents,
                "active_workflows": len(self.active_workflows),
                "total_workflows": total_workflows,
                "workflow_types": list(self.workflow_templates.keys()),
                "system_metrics": {
                    "avg_workflow_duration": await self._calculate_avg_workflow_duration(),
                    "success_rate": await self._calculate_system_success_rate(),
                    "throughput": await self._calculate_system_throughput()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"coordinator_running": False, "error": str(e)}
    
    # Event system
    
    async def subscribe_to_event(self, event_type: str, handler: Callable):
        """Subscribe to coordination events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger an event to all subscribers"""
        try:
            handlers = self.event_handlers.get(event_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
        except Exception as e:
            logger.error(f"Error triggering event {event_type}: {e}")
    
    # Internal coordination methods
    
    async def _coordination_loop(self):
        """Main coordination loop"""
        while self._running:
            try:
                # Process active workflows
                for workflow_id in list(self.active_workflows.keys()):
                    await self._process_workflow(workflow_id)
                
                # Update agent statuses
                await self._update_agent_statuses()
                
                # Sleep briefly to prevent tight loop
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(5.0)  # Longer sleep on error
    
    async def _process_workflow(self, workflow_id: str):
        """Process a single workflow"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow or workflow.status != WorkflowStatus.PENDING:
                return
            
            # Start workflow if not started
            if workflow.status == WorkflowStatus.PENDING:
                workflow.status = WorkflowStatus.RUNNING
                workflow.started_at = datetime.utcnow()
                await self._trigger_event("workflow_running", {"workflow_id": workflow_id})
            
            # Find next executable step
            next_step = await self._find_next_executable_step(workflow)
            
            if next_step:
                await self._execute_workflow_step(workflow, next_step)
            elif await self._is_workflow_complete(workflow):
                await self._complete_workflow(workflow_id)
            elif await self._is_workflow_stuck(workflow):
                await self._handle_stuck_workflow(workflow_id)
                
        except Exception as e:
            logger.error(f"Error processing workflow {workflow_id}: {e}")
            await self._fail_workflow(workflow_id, str(e))
    
    async def _find_next_executable_step(self, workflow: Workflow) -> Optional[WorkflowStep]:
        """Find the next step that can be executed"""
        try:
            for step in workflow.steps:
                if step.status != "pending":
                    continue
                
                # Check if dependencies are satisfied
                dependencies_satisfied = True
                for dep_step_id in step.dependencies:
                    dep_step = next((s for s in workflow.steps if s.step_id.endswith(dep_step_id)), None)
                    if not dep_step or dep_step.status != "completed":
                        dependencies_satisfied = False
                        break
                
                if dependencies_satisfied:
                    # Check if agent is available
                    agent = self.registered_agents.get(step.agent_id)
                    if agent and agent.status == AgentStatus.IDLE:
                        return step
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding next executable step: {e}")
            return None
    
    async def _execute_workflow_step(self, workflow: Workflow, step: WorkflowStep):
        """Execute a workflow step"""
        try:
            logger.info(f"Executing step {step.step_id} on agent {step.agent_id}")
            
            # Update step status
            step.status = "running"
            step.started_at = datetime.utcnow()
            
            # Update agent status
            agent = self.registered_agents.get(step.agent_id)
            if agent:
                agent.status = AgentStatus.BUSY
                agent.current_workflow = workflow.workflow_id
                agent.current_step = step.step_id
            
            # Prepare message data
            message_data = step.data.copy()
            message_data.update(workflow.context)
            message_data["workflow_id"] = workflow.workflow_id
            message_data["step_id"] = step.step_id
            
            # Send message to agent using the correct Message class
            if self.message_queue:
                message = Message(
                    type=MessageType.REQUEST,
                    sender="coordinator",
                    recipient=step.agent_id,
                    payload={
                        "action": step.action,
                        "data": message_data
                    },
                    priority=Priority.NORMAL,
                    timestamp=datetime.utcnow()
                )
                
                await self.message_queue.publish(f"agent.{step.agent_id}", message.__dict__)
            
            # Set timeout for step
            asyncio.create_task(self._monitor_step_timeout(workflow.workflow_id, step.step_id))
            
            await self._trigger_event("step_started", {
                "workflow_id": workflow.workflow_id,
                "step_id": step.step_id,
                "agent_id": step.agent_id
            })
            
        except Exception as e:
            logger.error(f"Error executing workflow step {step.step_id}: {e}")
            await self._fail_workflow_step(workflow.workflow_id, step.step_id, str(e))
    
    async def _handle_agent_message(self, message_data: Dict[str, Any]):
        """Handle messages from agents"""
        try:
            sender = message_data.get("sender")
            message_type_value = message_data.get("type")
            payload = message_data.get("payload", {})
            action = payload.get("action")
            data = payload.get("data", {})
            
            # Update agent heartbeat
            if sender in self.registered_agents:
                self.registered_agents[sender].last_heartbeat = datetime.utcnow()
            
            # Handle different message types
            if message_type_value == MessageType.RESPONSE.value:
                await self._handle_task_result(sender, data)
            elif message_type_value == MessageType.ERROR.value:
                await self._handle_agent_error(sender, data)
            elif message_type_value == MessageType.HEARTBEAT.value:
                await self._handle_heartbeat(sender, data)
            elif message_type_value == MessageType.NOTIFICATION.value:
                await self._handle_status_update(sender, data)
                
        except Exception as e:
            logger.error(f"Error handling agent message: {e}")
    
    async def _handle_status_update(self, agent_id: str, data: Dict[str, Any]):
        """Handle agent status update"""
        try:
            workflow_id = data.get("workflow_id")
            step_id = data.get("step_id")
            progress = data.get("progress", 0)
            current_step = data.get("current_step", "")
            
            # Update workflow progress if available
            if workflow_id and self.shared_memory:
                await self.shared_memory.update_design_progress(
                    workflow_id,
                    {
                        "progress_percentage": progress,
                        "current_step": current_step,
                        "current_agent": agent_id
                    }
                )
            
            await self._trigger_event("agent_status_update", {
                "agent_id": agent_id,
                "workflow_id": workflow_id,
                "step_id": step_id,
                "progress": progress
            })
            
        except Exception as e:
            logger.error(f"Error handling status update from {agent_id}: {e}")
    
    async def _handle_task_result(self, agent_id: str, data: Dict[str, Any]):
        """Handle task completion result from agent"""
        try:
            workflow_id = data.get("workflow_id")
            step_id = data.get("step_id")
            success = data.get("success", False)
            result_data = data.get("result", {})
            error_message = data.get("error")
            
            if workflow_id and step_id:
                if success:
                    await self._complete_workflow_step(workflow_id, step_id, result_data)
                else:
                    await self._fail_workflow_step(workflow_id, step_id, error_message or "Unknown error")
            
            # Update agent performance
            await self._update_agent_performance(agent_id, success)
            
            # Reset agent status
            agent = self.registered_agents.get(agent_id)
            if agent:
                agent.status = AgentStatus.IDLE
                agent.current_workflow = None
                agent.current_step = None
            
        except Exception as e:
            logger.error(f"Error handling task result from {agent_id}: {e}")
    
    async def _handle_agent_error(self, agent_id: str, data: Dict[str, Any]):
        """Handle error from agent"""
        try:
            workflow_id = data.get("workflow_id")
            step_id = data.get("step_id")
            error_message = data.get("error", "Unknown agent error")
            
            # Mark agent as error state
            agent = self.registered_agents.get(agent_id)
            if agent:
                agent.status = AgentStatus.ERROR
            
            # Fail the step/workflow
            if workflow_id and step_id:
                await self._fail_workflow_step(workflow_id, step_id, error_message)
            
            await self._trigger_event("agent_error", {
                "agent_id": agent_id,
                "workflow_id": workflow_id,
                "error": error_message
            })
            
        except Exception as e:
            logger.error(f"Error handling agent error from {agent_id}: {e}")
    
    async def _handle_heartbeat(self, agent_id: str, data: Dict[str, Any]):
        """Handle agent heartbeat"""
        try:
            agent = self.registered_agents.get(agent_id)
            if agent:
                agent.last_heartbeat = datetime.utcnow()
                if agent.status == AgentStatus.OFFLINE:
                    agent.status = AgentStatus.IDLE
                    logger.info(f"Agent {agent_id} came back online")
                    
        except Exception as e:
            logger.error(f"Error handling heartbeat from {agent_id}: {e}")
    
    async def _complete_workflow_step(self, workflow_id: str, step_id: str, result_data: Dict[str, Any]):
        """Mark a workflow step as completed"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                return
            
            # Find and update step
            step = next((s for s in workflow.steps if s.step_id == step_id), None)
            if not step:
                return
            
            step.status = "completed"
            step.completed_at = datetime.utcnow()
            
            # Update workflow context with results
            if result_data:
                workflow.context.update(result_data)
            
            logger.info(f"Completed step {step_id} in workflow {workflow_id}")
            
            await self._trigger_event("step_completed", {
                "workflow_id": workflow_id,
                "step_id": step_id,
                "agent_id": step.agent_id
            })
            
        except Exception as e:
            logger.error(f"Error completing workflow step {step_id}: {e}")
    
    async def _fail_workflow_step(self, workflow_id: str, step_id: str, error_message: str):
        """Mark a workflow step as failed"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                return
            
            # Find step
            step = next((s for s in workflow.steps if s.step_id == step_id), None)
            if not step:
                return
            
            # Check if we can retry
            if step.retry_count < step.max_retries:
                step.retry_count += 1
                step.status = "pending"
                step.error_message = error_message
                
                logger.info(f"Retrying step {step_id} (attempt {step.retry_count}/{step.max_retries})")
                
                # Wait before retry
                await asyncio.sleep(self.retry_delay)
                
                await self._trigger_event("step_retry", {
                    "workflow_id": workflow_id,
                    "step_id": step_id,
                    "retry_count": step.retry_count,
                    "error": error_message
                })
            else:
                # Max retries exceeded
                step.status = "failed"
                step.error_message = error_message
                step.completed_at = datetime.utcnow()
                
                logger.error(f"Step {step_id} failed after {step.max_retries} retries: {error_message}")
                
                # Fail entire workflow
                await self._fail_workflow(workflow_id, f"Step {step_id} failed: {error_message}")
                
        except Exception as e:
            logger.error(f"Error failing workflow step {step_id}: {e}")
    
    async def _complete_workflow(self, workflow_id: str):
        """Mark a workflow as completed"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                return
            
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()
            
            # Move to history
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]
            
            # Update metrics
            await self._update_workflow_metrics(workflow)
            
            logger.info(f"Completed workflow {workflow_id}")
            
            await self._trigger_event("workflow_completed", {
                "workflow_id": workflow_id,
                "workflow_type": workflow.workflow_type,
                "duration": (workflow.completed_at - workflow.started_at).total_seconds() if workflow.started_at else 0
            })
            
        except Exception as e:
            logger.error(f"Error completing workflow {workflow_id}: {e}")
    
    async def _fail_workflow(self, workflow_id: str, error_message: str):
        """Mark a workflow as failed"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                return
            
            workflow.status = WorkflowStatus.FAILED
            workflow.error_message = error_message
            workflow.completed_at = datetime.utcnow()
            
            # Move to history
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]
            
            logger.error(f"Failed workflow {workflow_id}: {error_message}")
            
            await self._trigger_event("workflow_failed", {
                "workflow_id": workflow_id,
                "error": error_message
            })
            
        except Exception as e:
            logger.error(f"Error failing workflow {workflow_id}: {e}")
    
    async def _cancel_workflow(self, workflow_id: str, reason: str) -> bool:
        """Cancel a workflow"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                return False
            
            workflow.status = WorkflowStatus.CANCELLED
            workflow.error_message = f"Cancelled: {reason}"
            workflow.completed_at = datetime.utcnow()
            
            # Cancel running steps
            for step in workflow.steps:
                if step.status == "running":
                    # Send cancellation message to agent
                    if self.message_queue:
                        message = Message(
                            type=MessageType.NOTIFICATION,
                            sender="coordinator",
                            recipient=step.agent_id,
                            payload={
                                "action": "cancel_task",
                                "data": {"workflow_id": workflow_id, "step_id": step.step_id}
                            },
                            priority=Priority.HIGH,
                            timestamp=datetime.utcnow()
                        )
                        await self.message_queue.publish(f"agent.{step.agent_id}", message.__dict__)
                    
                    step.status = "cancelled"
                    step.completed_at = datetime.utcnow()
            
            # Move to history
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]
            
            logger.info(f"Cancelled workflow {workflow_id}: {reason}")
            
            await self._trigger_event("workflow_cancelled", {
                "workflow_id": workflow_id,
                "reason": reason
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling workflow {workflow_id}: {e}")
            return False
    
    # Monitoring and maintenance
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats"""
        while self._running:
            try:
                current_time = datetime.utcnow()
                timeout_threshold = timedelta(seconds=self.heartbeat_interval * 2)
                
                for agent_id, agent in self.registered_agents.items():
                    if current_time - agent.last_heartbeat > timeout_threshold:
                        if agent.status != AgentStatus.OFFLINE:
                            logger.warning(f"Agent {agent_id} missed heartbeat, marking as offline")
                            agent.status = AgentStatus.OFFLINE
                            
                            # Handle workflows involving offline agent
                            await self._handle_offline_agent(agent_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _workflow_cleanup(self):
        """Clean up old workflow history"""
        while self._running:
            try:
                # Keep last 1000 workflows in history
                if len(self.workflow_history) > 1000:
                    self.workflow_history = self.workflow_history[-1000:]
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in workflow cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def _monitor_step_timeout(self, workflow_id: str, step_id: str):
        """Monitor step timeout"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                return
            
            step = next((s for s in workflow.steps if s.step_id == step_id), None)
            if not step:
                return
            
            # Wait for timeout
            await asyncio.sleep(step.timeout_seconds)
            
            # Check if step is still running
            if step.status == "running":
                logger.warning(f"Step {step_id} timed out after {step.timeout_seconds} seconds")
                await self._fail_workflow_step(workflow_id, step_id, f"Timeout after {step.timeout_seconds} seconds")
                
        except Exception as e:
            logger.error(f"Error monitoring step timeout: {e}")
    
    # Utility methods
    
    async def _is_workflow_complete(self, workflow: Workflow) -> bool:
        """Check if workflow is complete"""
        return all(step.status == "completed" for step in workflow.steps)
    
    async def _is_workflow_stuck(self, workflow: Workflow) -> bool:
        """Check if workflow is stuck"""
        # Workflow is stuck if no steps are running and no steps can be executed
        running_steps = [s for s in workflow.steps if s.status == "running"]
        if running_steps:
            return False
        
        next_step = await self._find_next_executable_step(workflow)
        return next_step is None and not await self._is_workflow_complete(workflow)
    
    async def _handle_stuck_workflow(self, workflow_id: str):
        """Handle stuck workflow"""
        await self._fail_workflow(workflow_id, "Workflow stuck - no executable steps available")
    
    async def _handle_offline_agent(self, agent_id: str):
        """Handle agent going offline"""
        try:
            # Find workflows with steps assigned to offline agent
            for workflow in list(self.active_workflows.values()):
                for step in workflow.steps:
                    if step.agent_id == agent_id and step.status == "running":
                        await self._fail_workflow_step(
                            workflow.workflow_id, 
                            step.step_id, 
                            f"Agent {agent_id} went offline"
                        )
                        
        except Exception as e:
            logger.error(f"Error handling offline agent {agent_id}: {e}")
    
    async def _update_agent_performance(self, agent_id: str, success: bool):
        """Update agent performance metrics"""
        try:
            if agent_id not in self.agent_performance:
                self.agent_performance[agent_id] = {
                    "total_tasks": 0,
                    "successful_tasks": 0,
                    "failed_tasks": 0,
                    "avg_execution_time": 0.0,
                    "last_task_time": None
                }
            
            metrics = self.agent_performance[agent_id]
            metrics["total_tasks"] += 1
            metrics["last_task_time"] = datetime.utcnow().isoformat()
            
            if success:
                metrics["successful_tasks"] += 1
            else:
                metrics["failed_tasks"] += 1
                
        except Exception as e:
            logger.error(f"Error updating agent performance: {e}")
    
    async def _update_agent_statuses(self):
        """Update agent statuses based on current activity"""
        try:
            # Reset agent statuses
            for agent in self.registered_agents.values():
                if agent.status == AgentStatus.BUSY:
                    # Check if agent is actually busy
                    is_busy = False
                    for workflow in self.active_workflows.values():
                        for step in workflow.steps:
                            if step.agent_id == agent.agent_id and step.status == "running":
                                is_busy = True
                                break
                        if is_busy:
                            break
                    
                    if not is_busy:
                        agent.status = AgentStatus.IDLE
                        agent.current_workflow = None
                        agent.current_step = None
                        
        except Exception as e:
            logger.error(f"Error updating agent statuses: {e}")
    
    async def _update_workflow_metrics(self, workflow: Workflow):
        """Update workflow performance metrics"""
        try:
            workflow_type = workflow.workflow_type
            
            if workflow_type not in self.workflow_metrics:
                self.workflow_metrics[workflow_type] = {
                    "total_count": 0,
                    "successful_count": 0,
                    "failed_count": 0,
                    "avg_duration": 0.0,
                    "total_duration": 0.0
                }
            
            metrics = self.workflow_metrics[workflow_type]
            metrics["total_count"] += 1
            
            if workflow.status == WorkflowStatus.COMPLETED:
                metrics["successful_count"] += 1
            else:
                metrics["failed_count"] += 1
            
            if workflow.started_at and workflow.completed_at:
                duration = (workflow.completed_at - workflow.started_at).total_seconds()
                metrics["total_duration"] += duration
                metrics["avg_duration"] = metrics["total_duration"] / metrics["total_count"]
                
        except Exception as e:
            logger.error(f"Error updating workflow metrics: {e}")
    
    async def _calculate_avg_workflow_duration(self) -> float:
        """Calculate average workflow duration"""
        try:
            total_duration = 0.0
            total_count = 0
            
            for metrics in self.workflow_metrics.values():
                total_duration += metrics.get("total_duration", 0.0)
                total_count += metrics.get("total_count", 0)
            
            return total_duration / total_count if total_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating average workflow duration: {e}")
            return 0.0
    
    async def _calculate_system_success_rate(self) -> float:
        """Calculate overall system success rate"""
        try:
            total_successful = 0
            total_count = 0
            
            for metrics in self.workflow_metrics.values():
                total_successful += metrics.get("successful_count", 0)
                total_count += metrics.get("total_count", 0)
            
            return (total_successful / total_count * 100) if total_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating system success rate: {e}")
            return 0.0
    
    async def _calculate_system_throughput(self) -> float:
        """Calculate system throughput (workflows per hour)"""
        try:
            if not self.workflow_history:
                return 0.0
            
            # Calculate throughput based on last 24 hours
            current_time = datetime.utcnow()
            one_day_ago = current_time - timedelta(days=1)
            
            recent_workflows = [
                w for w in self.workflow_history 
                if w.completed_at and w.completed_at >= one_day_ago
            ]
            
            return len(recent_workflows) / 24.0  # Workflows per hour
            
        except Exception as e:
            logger.error(f"Error calculating system throughput: {e}")
            return 0.0
    
    async def _initialize_workflow_templates(self):
        """Initialize workflow templates"""
        try:
            # Banner generation workflow template
            banner_workflow_steps = [
                WorkflowStep(
                    step_id="strategist",
                    agent_id="strategist",
                    action="start_strategy_workflow",
                    dependencies=[],
                    timeout_seconds=120
                ),
                WorkflowStep(
                    step_id="background_designer",
                    agent_id="background_designer",
                    action="start_background_workflow",
                    dependencies=["strategist"],
                    timeout_seconds=300
                ),
                WorkflowStep(
                    step_id="foreground_designer",
                    agent_id="foreground_designer",
                    action="start_foreground_workflow",
                    dependencies=["strategist", "background_designer"],
                    timeout_seconds=240
                ),
                WorkflowStep(
                    step_id="developer",
                    agent_id="developer",
                    action="start_code_generation_workflow",
                    dependencies=["foreground_designer"],
                    timeout_seconds=180
                ),
                WorkflowStep(
                    step_id="design_reviewer",
                    agent_id="design_reviewer",
                    action="start_design_review_workflow",
                    dependencies=["developer"],
                    timeout_seconds=120
                )
            ]
            
            self.workflow_templates["banner_generation"] = banner_workflow_steps
            
            # Quick banner workflow (without review)
            quick_workflow_steps = banner_workflow_steps[:-1]  # Remove review step
            self.workflow_templates["quick_banner_generation"] = quick_workflow_steps
            
            logger.info("Workflow templates initialized")
            
        except Exception as e:
            logger.error(f"Error initializing workflow templates: {e}")
            raise
