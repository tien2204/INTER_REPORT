"""
Foreground Designer Agent

Responsible for layout design, typography, component placement,
and generating design blueprints for banner advertisements.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from structlog import get_logger

from communication.protocol import Message, MessageType
from memory_manager.shared_memory import SharedMemory
from communication.message_queue import MessageQueue
from .layout_engine import LayoutEngine
from .typography_manager import TypographyManager
from .component_placer import ComponentPlacer
from .blueprint_generator import BlueprintGenerator

logger = get_logger(__name__)


class ForegroundDesignerAgent:
    """
    AI Agent for foreground design and layout generation
    
    Capabilities:
    - Layout analysis and optimization
    - Typography selection and sizing
    - Component placement and hierarchy
    - Blueprint generation for developers
    - Brand consistency checking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = "foreground_designer"
        self.agent_name = "Foreground Designer Agent"
        
        # Core components
        self.layout_engine = LayoutEngine(config.get("layout_engine", {}))
        self.typography_manager = TypographyManager(config.get("typography", {}))
        self.component_placer = ComponentPlacer(config.get("component_placer", {}))
        self.blueprint_generator = BlueprintGenerator(config.get("blueprint_generator", {}))
        
        # Communication
        self.shared_memory: Optional[SharedMemory] = None
        self.message_queue: Optional[MessageQueue] = None
        
        # State management
        self._running = False
        self._active_sessions = {}
        self._processing_queue = asyncio.Queue()
        
        # Performance tracking
        self._total_processed = 0
        self._successful_generations = 0
        self._avg_processing_time = 0.0
        
        logger.info(f"Foreground Designer Agent initialized: {self.agent_id}")
    
    async def start(self):
        """Start the agent"""
        try:
            logger.info("Starting Foreground Designer Agent")
            
            # Initialize components
            await self.layout_engine.initialize()
            await self.typography_manager.initialize()
            await self.component_placer.initialize()
            await self.blueprint_generator.initialize()
            
            self._running = True
            
            # Start message processing
            asyncio.create_task(self._process_messages())
            asyncio.create_task(self._process_queue())
            
            logger.info("Foreground Designer Agent started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Foreground Designer Agent: {e}")
            raise
    
    async def stop(self):
        """Stop the agent"""
        try:
            logger.info("Stopping Foreground Designer Agent")
            
            self._running = False
            
            # Wait for active sessions to complete
            for session_id in list(self._active_sessions.keys()):
                await self._cleanup_session(session_id)
            
            logger.info("Foreground Designer Agent stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Foreground Designer Agent: {e}")
    
    def set_communication(self, shared_memory: SharedMemory, message_queue: MessageQueue):
        """Set communication interfaces"""
        self.shared_memory = shared_memory
        self.message_queue = message_queue
        
        # Initialize components with communication
        self.layout_engine.set_communication(shared_memory, message_queue)
        self.typography_manager.set_communication(shared_memory, message_queue)
        self.component_placer.set_communication(shared_memory, message_queue)
        self.blueprint_generator.set_communication(shared_memory, message_queue)
    
    async def _process_messages(self):
        """Process incoming messages"""
        if not self.message_queue:
            return
        
        try:
            await self.message_queue.subscribe(
                f"agent.{self.agent_id}",
                self._handle_message
            )
        except Exception as e:
            logger.error(f"Error in message processing: {e}")
    
    async def _handle_message(self, message: Message):
        """Handle incoming agent message"""
        try:
            # Extract action from message payload
            action = message.payload.get("action", "")
            logger.info(f"Foreground Designer Agent received message: {action}")
            
            if action == "start_foreground_design_workflow":
                await self._start_foreground_design_workflow(message.payload.get("data", {}))
            
            elif action == "generate_layout":
                await self._generate_layout(message.payload.get("data", {}))
            
            elif action == "optimize_typography":
                await self._optimize_typography(message.payload.get("data", {}))
            
            elif action == "place_components":
                await self._place_components(message.payload.get("data", {}))
            
            elif action == "generate_blueprint":
                await self._generate_blueprint(message.payload.get("data", {}))
            
            elif action == "iterate_design":
                await self._iterate_design(message.payload.get("data", {}))
            
            else:
                logger.warning(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _process_queue(self):
        """Process the internal processing queue"""
        while self._running:
            try:
                task = await asyncio.wait_for(self._processing_queue.get(), timeout=1.0)
                await self._process_task(task)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing queue task: {e}")
    
    async def _start_foreground_design_workflow(self, data: Dict[str, Any]):
        """Start the foreground design workflow"""
        try:
            session_id = data.get("session_id")
            design_id = data.get("design_id")
            
            logger.info(f"Starting foreground design workflow: session={session_id}, design={design_id}")
            
            # Create session
            session = {
                "session_id": session_id,
                "design_id": design_id,
                "started_at": datetime.utcnow(),
                "current_step": "initialization",
                "progress": 0,
                "context": data.get("context", {}),
                "results": {}
            }
            
            self._active_sessions[session_id] = session
            
            # Add to processing queue
            await self._processing_queue.put({
                "type": "workflow",
                "session_id": session_id,
                "data": data
            })
            
        except Exception as e:
            logger.error(f"Error starting foreground design workflow: {e}")
    
    async def _process_task(self, task: Dict[str, Any]):
        """Process a queued task"""
        try:
            task_type = task.get("type")
            session_id = task.get("session_id")
            
            if task_type == "workflow":
                await self._execute_foreground_workflow(session_id, task["data"])
            
            elif task_type == "layout_generation":
                await self._execute_layout_generation(session_id, task["data"])
            
            elif task_type == "typography_optimization":
                await self._execute_typography_optimization(session_id, task["data"])
            
            elif task_type == "component_placement":
                await self._execute_component_placement(session_id, task["data"])
            
            elif task_type == "blueprint_generation":
                await self._execute_blueprint_generation(session_id, task["data"])
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
    
    async def _execute_foreground_workflow(self, session_id: str, data: Dict[str, Any]):
        """Execute the complete foreground design workflow"""
        try:
            session = self._active_sessions.get(session_id)
            if not session:
                logger.error(f"Session not found: {session_id}")
                return
            
            design_id = session["design_id"]
            
            # Update progress
            await self._update_progress(session_id, 10, "Analyzing strategic direction")
            
            # Step 1: Get strategic direction and background
            strategic_direction = await self._get_strategic_direction(design_id)
            background_data = await self._get_background_data(design_id)
            
            if not strategic_direction or not background_data:
                logger.error(f"Missing strategic direction or background data for design {design_id}")
                await self._mark_session_failed(session_id, "Missing prerequisite data")
                return
            
            # Step 2: Generate layout
            await self._update_progress(session_id, 25, "Generating layout structure")
            layout_result = await self.layout_engine.generate_layout(
                strategic_direction, background_data, session["context"]
            )
            
            if not layout_result.get("success"):
                await self._mark_session_failed(session_id, "Layout generation failed")
                return
            
            session["results"]["layout"] = layout_result
            
            # Step 3: Optimize typography
            await self._update_progress(session_id, 50, "Optimizing typography")
            typography_result = await self.typography_manager.optimize_typography(
                strategic_direction, layout_result, session["context"]
            )
            
            if not typography_result.get("success"):
                await self._mark_session_failed(session_id, "Typography optimization failed")
                return
            
            session["results"]["typography"] = typography_result
            
            # Step 4: Place components
            await self._update_progress(session_id, 75, "Placing components")
            placement_result = await self.component_placer.place_components(
                layout_result, typography_result, session["context"]
            )
            
            if not placement_result.get("success"):
                await self._mark_session_failed(session_id, "Component placement failed")
                return
            
            session["results"]["placement"] = placement_result
            
            # Step 5: Generate blueprint
            await self._update_progress(session_id, 90, "Generating design blueprint")
            blueprint_result = await self.blueprint_generator.generate_blueprint(
                layout_result, typography_result, placement_result, session["context"]
            )
            
            if not blueprint_result.get("success"):
                await self._mark_session_failed(session_id, "Blueprint generation failed")
                return
            
            session["results"]["blueprint"] = blueprint_result
            
            # Step 6: Finalize
            await self._update_progress(session_id, 100, "Finalizing design")
            await self._complete_session(session_id)
            
            # Update design in shared memory
            await self._update_design_data(design_id, session["results"])
            
            # Notify next agent (Developer Agent)
            await self._notify_next_agent(design_id, session_id)
            
            logger.info(f"Foreground design workflow completed: session={session_id}")
            
        except Exception as e:
            logger.error(f"Error in foreground workflow: {e}")
            await self._mark_session_failed(session_id, str(e))
    
    async def _get_strategic_direction(self, design_id: str) -> Optional[Dict[str, Any]]:
        """Get strategic direction from shared memory"""
        try:
            if not self.shared_memory:
                return None
            
            design_data = await self.shared_memory.get_design_data(design_id)
            return design_data.get("strategic_direction") if design_data else None
            
        except Exception as e:
            logger.error(f"Error getting strategic direction: {e}")
            return None
    
    async def _get_background_data(self, design_id: str) -> Optional[Dict[str, Any]]:
        """Get background data from shared memory"""
        try:
            if not self.shared_memory:
                return None
            
            design_data = await self.shared_memory.get_design_data(design_id)
            return design_data.get("background_data") if design_data else None
            
        except Exception as e:
            logger.error(f"Error getting background data: {e}")
            return None
    
    async def _update_progress(self, session_id: str, progress: int, current_step: str):
        """Update session progress"""
        try:
            session = self._active_sessions.get(session_id)
            if not session:
                return
            
            session["progress"] = progress
            session["current_step"] = current_step
            session["updated_at"] = datetime.utcnow()
            
            # Update design progress in shared memory
            if self.shared_memory and session.get("design_id"):
                await self.shared_memory.update_design_progress(
                    session["design_id"],
                    {
                        "progress_percentage": progress,
                        "current_step": current_step,
                        "current_agent": self.agent_id
                    }
                )
            
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    async def _mark_session_failed(self, session_id: str, error_message: str):
        """Mark session as failed"""
        try:
            session = self._active_sessions.get(session_id)
            if not session:
                return
            
            session["status"] = "failed"
            session["error_message"] = error_message
            session["completed_at"] = datetime.utcnow()
            
            # Update design status
            if self.shared_memory and session.get("design_id"):
                await self.shared_memory.update_design_progress(
                    session["design_id"],
                    {
                        "status": "failed",
                        "error_message": error_message,
                        "current_agent": self.agent_id
                    }
                )
            
            logger.error(f"Session failed: {session_id} - {error_message}")
            
        except Exception as e:
            logger.error(f"Error marking session failed: {e}")
    
    async def _complete_session(self, session_id: str):
        """Mark session as completed"""
        try:
            session = self._active_sessions.get(session_id)
            if not session:
                return
            
            session["status"] = "completed"
            session["completed_at"] = datetime.utcnow()
            
            # Update statistics
            self._total_processed += 1
            self._successful_generations += 1
            
            # Calculate processing time
            processing_time = (session["completed_at"] - session["started_at"]).total_seconds()
            self._avg_processing_time = (
                (self._avg_processing_time * (self._total_processed - 1) + processing_time) 
                / self._total_processed
            )
            
            logger.info(f"Session completed: {session_id} (processing time: {processing_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"Error completing session: {e}")
    
    async def _update_design_data(self, design_id: str, results: Dict[str, Any]):
        """Update design data with foreground results"""
        try:
            if not self.shared_memory:
                return
            
            design_data = await self.shared_memory.get_design_data(design_id)
            if not design_data:
                return
            
            # Update with foreground design results
            design_data["foreground_data"] = {
                "layout": results.get("layout"),
                "typography": results.get("typography"), 
                "placement": results.get("placement"),
                "blueprint": results.get("blueprint"),
                "generated_at": datetime.utcnow().isoformat(),
                "agent": self.agent_id
            }
            
            design_data["status"] = "foreground_design"
            design_data["progress_percentage"] = 70  # Ready for developer
            design_data["current_step"] = "Code generation"
            design_data["current_agent"] = "developer"
            
            await self.shared_memory.set_design_data(design_id, design_data)
            
        except Exception as e:
            logger.error(f"Error updating design data: {e}")
    
    async def _notify_next_agent(self, design_id: str, session_id: str):
        """Notify the next agent in the pipeline"""
        try:
            if not self.message_queue:
                return
            
            # Create message using Message class
            message = Message(
                sender=self.agent_id,
                recipient="developer",
                type=MessageType.REQUEST,
                payload={
                    "action": "start_code_generation_workflow",
                    "data": {
                        "session_id": session_id,
                        "design_id": design_id,
                        "previous_agent": self.agent_id
                    }
                },
                timestamp=datetime.utcnow()
            )
            
            # Convert to dict for publishing
            message_dict = {
                "sender": message.sender,
                "recipient": message.recipient,
                "type": message.type.value,
                "payload": message.payload,
                "timestamp": message.timestamp.isoformat()
            }
            
            await self.message_queue.publish("agent.developer", message_dict)
            
        except Exception as e:
            logger.error(f"Error notifying next agent: {e}")
    
    async def _cleanup_session(self, session_id: str):
        """Clean up session resources"""
        try:
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
            
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
    
    # Public API methods
    async def generate_layout(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate layout for design"""
        try:
            return await self.layout_engine.generate_layout(
                design_data.get("strategic_direction", {}),
                design_data.get("background_data", {}),
                design_data.get("context", {})
            )
        except Exception as e:
            logger.error(f"Error generating layout: {e}")
            return {"success": False, "error": str(e)}
    
    async def optimize_typography(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize typography for design"""
        try:
            return await self.typography_manager.optimize_typography(
                design_data.get("strategic_direction", {}),
                design_data.get("layout", {}),
                design_data.get("context", {})
            )
        except Exception as e:
            logger.error(f"Error optimizing typography: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a session"""
        session = self._active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "design_id": session.get("design_id"),
            "status": session.get("status", "active"),
            "progress": session.get("progress", 0),
            "current_step": session.get("current_step"),
            "started_at": session.get("started_at"),
            "updated_at": session.get("updated_at"),
            "error_message": session.get("error_message")
        }
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and statistics"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "running": self._running,
            "active_sessions": len(self._active_sessions),
            "total_processed": self._total_processed,
            "successful_generations": self._successful_generations,
            "success_rate": (self._successful_generations / self._total_processed * 100) if self._total_processed > 0 else 0,
            "avg_processing_time": self._avg_processing_time,
            "queue_size": self._processing_queue.qsize()
        }
    
    # Missing method implementations (placeholder methods)
    async def _generate_layout(self, data: Dict[str, Any]):
        """Handle generate layout message"""
        try:
            # Implementation placeholder
            logger.info("Generate layout message received")
        except Exception as e:
            logger.error(f"Error in generate layout: {e}")
    
    async def _optimize_typography(self, data: Dict[str, Any]):
        """Handle optimize typography message"""
        try:
            # Implementation placeholder
            logger.info("Optimize typography message received")
        except Exception as e:
            logger.error(f"Error in optimize typography: {e}")
    
    async def _place_components(self, data: Dict[str, Any]):
        """Handle place components message"""
        try:
            # Implementation placeholder
            logger.info("Place components message received")
        except Exception as e:
            logger.error(f"Error in place components: {e}")
    
    async def _generate_blueprint(self, data: Dict[str, Any]):
        """Handle generate blueprint message"""
        try:
            # Implementation placeholder
            logger.info("Generate blueprint message received")
        except Exception as e:
            logger.error(f"Error in generate blueprint: {e}")
    
    async def _iterate_design(self, data: Dict[str, Any]):
        """Handle iterate design message"""
        try:
            # Implementation placeholder
            logger.info("Iterate design message received")
        except Exception as e:
            logger.error(f"Error in iterate design: {e}")
    
    async def _execute_layout_generation(self, session_id: str, data: Dict[str, Any]):
        """Execute layout generation task"""
        try:
            # Implementation placeholder
            logger.info(f"Execute layout generation for session: {session_id}")
        except Exception as e:
            logger.error(f"Error in execute layout generation: {e}")
    
    async def _execute_typography_optimization(self, session_id: str, data: Dict[str, Any]):
        """Execute typography optimization task"""
        try:
            # Implementation placeholder
            logger.info(f"Execute typography optimization for session: {session_id}")
        except Exception as e:
            logger.error(f"Error in execute typography optimization: {e}")
    
    async def _execute_component_placement(self, session_id: str, data: Dict[str, Any]):
        """Execute component placement task"""
        try:
            # Implementation placeholder
            logger.info(f"Execute component placement for session: {session_id}")
        except Exception as e:
            logger.error(f"Error in execute component placement: {e}")
    
    async def _execute_blueprint_generation(self, session_id: str, data: Dict[str, Any]):
        """Execute blueprint generation task"""
        try:
            # Implementation placeholder
            logger.info(f"Execute blueprint generation for session: {session_id}")
        except Exception as e:
            logger.error(f"Error in execute blueprint generation: {e}")
