"""
Communication Manager

Central manager for all communication components, providing unified
interface for inter-agent communication, coordination, and monitoring.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from structlog import get_logger

from .agent_coordinator import AgentCoordinator, WorkflowStatus, AgentStatus
from .message_queue import MessageQueue
from .enhanced_event_dispatcher import EnhancedEventDispatcher
from .protocol import Message, MessageType  # Fixed import: AgentMessage -> Message
from memory_manager.shared_memory import SharedMemory

logger = get_logger(__name__)


class CommunicationManager:
    """
    Unified communication management system
    
    Capabilities:
    - Centralized communication setup and management
    - Integrated agent coordination and workflow management
    - Event-driven communication with advanced features
    - Message routing and delivery guarantees
    - System monitoring and health checking
    - Performance optimization and load balancing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core communication components
        self.message_queue: Optional[MessageQueue] = None
        self.event_dispatcher: Optional[EnhancedEventDispatcher] = None
        self.agent_coordinator: Optional[AgentCoordinator] = None
        self.shared_memory: Optional[SharedMemory] = None
        
        # System state
        self._running = False
        self._health_monitor_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.system_metrics = {
            "message_throughput": 0.0,
            "event_throughput": 0.0,
            "workflow_success_rate": 0.0,
            "avg_response_time": 0.0,
            "active_agents": 0,
            "active_workflows": 0
        }
        
        # Configuration
        self.health_check_interval = config.get("health_check_interval", 30)
        self.metrics_update_interval = config.get("metrics_update_interval", 60)
        
        logger.info("Communication Manager initialized")
    
    async def initialize(self, shared_memory: SharedMemory):
        """Initialize all communication components"""
        try:
            logger.info("Initializing Communication Manager")
            
            self.shared_memory = shared_memory
            
            # Initialize message queue
            message_queue_config = self.config.get("message_queue", {})
            self.message_queue = MessageQueue(message_queue_config)
            await self.message_queue.start()
            
            # Initialize event dispatcher
            event_dispatcher_config = self.config.get("event_dispatcher", {})
            self.event_dispatcher = EnhancedEventDispatcher(event_dispatcher_config)
            await self.event_dispatcher.start()
            
            # Initialize agent coordinator
            coordinator_config = self.config.get("agent_coordinator", {})
            self.agent_coordinator = AgentCoordinator(coordinator_config)
            self.agent_coordinator.set_communication(self.message_queue, shared_memory)
            await self.agent_coordinator.start()
            
            # Connect components
            self.event_dispatcher.set_agent_coordinator(self.agent_coordinator)
            
            # Set up system event subscriptions
            await self._setup_system_event_subscriptions()
            
            logger.info("Communication Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Communication Manager: {e}")
            raise
    
    async def start(self):
        """Start the communication manager"""
        try:
            logger.info("Starting Communication Manager")
            
            self._running = True
            
            # Start health monitoring
            self._health_monitor_task = asyncio.create_task(self._health_monitor())
            
            # Start metrics collection
            asyncio.create_task(self._metrics_collector())
            
            logger.info("Communication Manager started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Communication Manager: {e}")
            raise
    
    async def stop(self):
        """Stop the communication manager"""
        try:
            logger.info("Stopping Communication Manager")
            
            self._running = False
            
            # Cancel health monitor
            if self._health_monitor_task:
                self._health_monitor_task.cancel()
                try:
                    await self._health_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Stop components in reverse order
            if self.agent_coordinator:
                await self.agent_coordinator.stop()
            
            if self.event_dispatcher:
                await self.event_dispatcher.stop()
            
            if self.message_queue:
                await self.message_queue.stop()
            
            logger.info("Communication Manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Communication Manager: {e}")
    
    # Agent Management
    
    async def register_agent(self, agent_id: str, agent_name: str, 
                           capabilities: List[str]) -> bool:
        """Register an agent with the communication system"""
        try:
            if not self.agent_coordinator:
                logger.error("Agent coordinator not initialized")
                return False
            
            # Register with coordinator
            success = await self.agent_coordinator.register_agent(agent_id, agent_name, capabilities)
            
            if success:
                # Publish agent registration event
                await self.publish_event("agent_registered", {
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "capabilities": capabilities
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Error registering agent {agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the communication system"""
        try:
            if not self.agent_coordinator:
                return False
            
            success = await self.agent_coordinator.unregister_agent(agent_id)
            
            if success:
                await self.publish_event("agent_unregistered", {
                    "agent_id": agent_id
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Error unregistering agent {agent_id}: {e}")
            return False
    
    # Workflow Management
    
    async def start_workflow(self, workflow_type: str, context: Dict[str, Any]) -> Optional[str]:
        """Start a new workflow"""
        try:
            if not self.agent_coordinator:
                logger.error("Agent coordinator not initialized")
                return None
            
            workflow_id = await self.agent_coordinator.start_workflow(workflow_type, context)
            
            # Publish workflow start event
            await self.publish_event("workflow_started", {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "context": context
            })
            
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error starting workflow: {e}")
            return None
    
    async def cancel_workflow(self, workflow_id: str, reason: str = "User cancelled") -> bool:
        """Cancel an active workflow"""
        try:
            if not self.agent_coordinator:
                return False
            
            success = await self.agent_coordinator.cancel_workflow(workflow_id, reason)
            
            if success:
                await self.publish_event("workflow_cancelled", {
                    "workflow_id": workflow_id,
                    "reason": reason
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Error cancelling workflow {workflow_id}: {e}")
            return False
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        try:
            if not self.agent_coordinator:
                return None
            
            return await self.agent_coordinator.get_workflow_status(workflow_id)
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return None
    
    # Message Management
    
    async def send_message(self, message: Message) -> bool:  # Fixed type: AgentMessage -> Message
        """Send a message through the communication system"""
        try:
            if not self.message_queue:
                logger.error("Message queue not initialized")
                return False
            
            channel = f"agent.{message.recipient}"
            # Convert message to dict - assuming Message has a dict() method or similar
            message_dict = {
                "id": message.id,
                "type": message.type.value,
                "sender": message.sender,
                "recipient": message.recipient,
                "payload": message.payload,
                "priority": message.priority.value,
                "correlation_id": message.correlation_id,
                "reply_to": message.reply_to,
                "timestamp": message.timestamp.isoformat(),
                "ttl": message.ttl,
                "retry_count": message.retry_count,
                "max_retries": message.max_retries
            }
            await self.message_queue.publish(channel, message_dict)
            
            # Publish message sent event
            await self.publish_event("message_sent", {
                "sender": message.sender,
                "recipient": message.recipient,
                "message_type": message.type.value,
                "payload": message.payload
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def subscribe_to_messages(self, channel: str, handler: Callable) -> bool:
        """Subscribe to messages on a channel"""
        try:
            if not self.message_queue:
                return False
            
            await self.message_queue.subscribe(channel, handler)
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to messages: {e}")
            return False
    
    # Event Management
    
    async def publish_event(self, event_type: str, data: Dict[str, Any], 
                          source: str = "communication_manager", priority: int = 0) -> bool:
        """Publish an event through the event system"""
        try:
            if not self.event_dispatcher:
                logger.error("Event dispatcher not initialized")
                return False
            
            await self.event_dispatcher.publish(event_type, data, source, priority)
            return True
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            return False
    
    async def subscribe_to_event(self, event_type: str, handler: Callable,
                               event_filter: Optional[Callable] = None,
                               priority: int = 0) -> bool:
        """Subscribe to events"""
        try:
            if not self.event_dispatcher:
                return False
            
            await self.event_dispatcher.subscribe(event_type, handler, event_filter, priority)
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to event: {e}")
            return False
    
    # System Status and Monitoring
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                "communication_manager": {
                    "running": self._running,
                    "components": {
                        "message_queue": self.message_queue is not None,
                        "event_dispatcher": self.event_dispatcher is not None,
                        "agent_coordinator": self.agent_coordinator is not None
                    }
                },
                "metrics": self.system_metrics.copy()
            }
            
            # Get component-specific status
            if self.agent_coordinator:
                coordinator_status = await self.agent_coordinator.get_system_status()
                status["agent_coordinator"] = coordinator_status
            
            if self.event_dispatcher:
                event_metrics = await self.event_dispatcher.get_event_metrics()
                status["event_dispatcher"] = event_metrics
            
            if self.message_queue:
                message_status = await self.message_queue.get_status()
                status["message_queue"] = message_status
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent"""
        try:
            if not self.agent_coordinator:
                return None
            
            return await self.agent_coordinator.get_agent_status(agent_id)
            
        except Exception as e:
            logger.error(f"Error getting agent status: {e}")
            return None
    
    async def get_all_agents_status(self) -> Dict[str, Any]:
        """Get status of all registered agents"""
        try:
            if not self.agent_coordinator:
                return {}
            
            # Get list of registered agents
            coordinator_status = await self.agent_coordinator.get_system_status()
            agent_count = coordinator_status.get("registered_agents", 0)
            
            # For now, return summary - in production, would get individual statuses
            return {
                "total_agents": agent_count,
                "active_agents": coordinator_status.get("active_agents", 0),
                "summary": "Use get_agent_status(agent_id) for individual agent details"
            }
            
        except Exception as e:
            logger.error(f"Error getting all agents status: {e}")
            return {}
    
    # Performance and Health Monitoring
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            metrics = self.system_metrics.copy()
            
            # Add component-specific metrics
            if self.event_dispatcher:
                event_metrics = await self.event_dispatcher.get_event_metrics()
                metrics["event_processing"] = event_metrics.get("global", {})
            
            if self.agent_coordinator:
                coordinator_status = await self.agent_coordinator.get_system_status()
                metrics["workflow_metrics"] = coordinator_status.get("system_metrics", {})
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_status = {
                "overall_health": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {}
            }
            
            issues = []
            
            # Check message queue health
            if self.message_queue:
                try:
                    mq_status = await self.message_queue.get_status()
                    health_status["components"]["message_queue"] = "healthy"
                except Exception as e:
                    health_status["components"]["message_queue"] = "unhealthy"
                    issues.append(f"Message queue issue: {e}")
            else:
                health_status["components"]["message_queue"] = "not_initialized"
                issues.append("Message queue not initialized")
            
            # Check event dispatcher health
            if self.event_dispatcher:
                try:
                    event_metrics = await self.event_dispatcher.get_event_metrics()
                    queue_size = event_metrics.get("global", {}).get("queue_size", 0)
                    if queue_size > 1000:  # Threshold for concern
                        issues.append(f"Event queue backing up: {queue_size} events")
                    health_status["components"]["event_dispatcher"] = "healthy"
                except Exception as e:
                    health_status["components"]["event_dispatcher"] = "unhealthy"
                    issues.append(f"Event dispatcher issue: {e}")
            else:
                health_status["components"]["event_dispatcher"] = "not_initialized"
                issues.append("Event dispatcher not initialized")
            
            # Check agent coordinator health
            if self.agent_coordinator:
                try:
                    coordinator_status = await self.agent_coordinator.get_system_status()
                    active_workflows = coordinator_status.get("active_workflows", 0)
                    if active_workflows > 50:  # Threshold for concern
                        issues.append(f"High workflow load: {active_workflows} active workflows")
                    health_status["components"]["agent_coordinator"] = "healthy"
                except Exception as e:
                    health_status["components"]["agent_coordinator"] = "unhealthy"
                    issues.append(f"Agent coordinator issue: {e}")
            else:
                health_status["components"]["agent_coordinator"] = "not_initialized"
                issues.append("Agent coordinator not initialized")
            
            # Determine overall health
            if issues:
                if len(issues) >= 2:
                    health_status["overall_health"] = "critical"
                else:
                    health_status["overall_health"] = "degraded"
                health_status["issues"] = issues
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "overall_health": "critical",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Internal Methods
    
    async def _setup_system_event_subscriptions(self):
        """Set up system-level event subscriptions"""
        try:
            if not self.event_dispatcher:
                return
            
            # Subscribe to workflow events
            await self.event_dispatcher.subscribe(
                "workflow_completed",
                self._handle_workflow_completed,
                priority=10
            )
            
            await self.event_dispatcher.subscribe(
                "workflow_failed",
                self._handle_workflow_failed,
                priority=10
            )
            
            # Subscribe to agent events
            await self.event_dispatcher.subscribe(
                "agent_error",
                self._handle_agent_error,
                priority=10
            )
            
            logger.info("System event subscriptions set up")
            
        except Exception as e:
            logger.error(f"Error setting up system event subscriptions: {e}")
    
    async def _handle_workflow_completed(self, event: Dict[str, Any]):
        """Handle workflow completion events"""
        try:
            data = event.get("data", {})
            workflow_id = data.get("workflow_id")
            
            logger.info(f"Workflow completed: {workflow_id}")
            
            # Update metrics
            self.system_metrics["workflow_success_rate"] = await self._calculate_workflow_success_rate()
            
        except Exception as e:
            logger.error(f"Error handling workflow completed event: {e}")
    
    async def _handle_workflow_failed(self, event: Dict[str, Any]):
        """Handle workflow failure events"""
        try:
            data = event.get("data", {})
            workflow_id = data.get("workflow_id")
            error = data.get("error")
            
            logger.warning(f"Workflow failed: {workflow_id} - {error}")
            
            # Update metrics
            self.system_metrics["workflow_success_rate"] = await self._calculate_workflow_success_rate()
            
        except Exception as e:
            logger.error(f"Error handling workflow failed event: {e}")
    
    async def _handle_agent_error(self, event: Dict[str, Any]):
        """Handle agent error events"""
        try:
            data = event.get("data", {})
            agent_id = data.get("agent_id")
            error = data.get("error")
            
            logger.error(f"Agent error: {agent_id} - {error}")
            
            # Could implement automatic recovery mechanisms here
            
        except Exception as e:
            logger.error(f"Error handling agent error event: {e}")
    
    async def _health_monitor(self):
        """Continuous health monitoring"""
        while self._running:
            try:
                health_status = await self.health_check()
                
                if health_status["overall_health"] != "healthy":
                    logger.warning(f"System health issue detected: {health_status}")
                    
                    # Publish health alert event
                    await self.publish_event("system_health_alert", health_status)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _metrics_collector(self):
        """Collect and update system metrics"""
        while self._running:
            try:
                await asyncio.sleep(self.metrics_update_interval)
                
                # Update metrics
                await self._update_system_metrics()
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(self.metrics_update_interval)
    
    async def _update_system_metrics(self):
        """Update system performance metrics"""
        try:
            # Get component metrics
            if self.agent_coordinator:
                coordinator_status = await self.agent_coordinator.get_system_status()
                self.system_metrics["active_agents"] = coordinator_status.get("active_agents", 0)
                self.system_metrics["active_workflows"] = coordinator_status.get("active_workflows", 0)
            
            if self.event_dispatcher:
                event_metrics = await self.event_dispatcher.get_event_metrics()
                global_metrics = event_metrics.get("global", {})
                
                # Calculate event throughput (events per minute)
                total_handled = global_metrics.get("total_handled", 0)
                self.system_metrics["event_throughput"] = total_handled / max(1, self.metrics_update_interval / 60)
            
            # Update workflow success rate
            self.system_metrics["workflow_success_rate"] = await self._calculate_workflow_success_rate()
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    async def _calculate_workflow_success_rate(self) -> float:
        """Calculate workflow success rate"""
        try:
            if not self.agent_coordinator:
                return 0.0
            
            coordinator_status = await self.agent_coordinator.get_system_status()
            metrics = coordinator_status.get("system_metrics", {})
            
            return metrics.get("success_rate", 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating workflow success rate: {e}")
            return 0.0
