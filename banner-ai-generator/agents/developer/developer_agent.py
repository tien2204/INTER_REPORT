"""
Developer Agent

Responsible for converting design blueprints into executable code
across multiple formats (SVG, Figma, HTML/CSS, PNG).
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from structlog import get_logger

from communication.protocol import Message, MessageType
from memory_manager.shared_memory import SharedMemory
from communication.message_queue import MessageQueue
from .svg_generator import SVGGenerator
from .figma_generator import FigmaGenerator
from .html_generator import HTMLGenerator
from .code_optimizer import CodeOptimizer

logger = get_logger(__name__)


class DeveloperAgent:
    """
    AI Agent for code generation from design blueprints
    
    Capabilities:
    - SVG code generation
    - Figma plugin code generation
    - HTML/CSS generation
    - PNG export coordination
    - Code optimization
    - Multi-format export
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = "developer"
        self.agent_name = "Developer Agent"
        
        # Core generators
        self.svg_generator = SVGGenerator(config.get("svg_generator", {}))
        self.figma_generator = FigmaGenerator(config.get("figma_generator", {}))
        self.html_generator = HTMLGenerator(config.get("html_generator", {}))
        self.code_optimizer = CodeOptimizer(config.get("code_optimizer", {}))
        
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
        
        logger.info(f"Developer Agent initialized: {self.agent_id}")
    
    async def start(self):
        """Start the agent"""
        try:
            logger.info("Starting Developer Agent")
            
            # Initialize generators
            await self.svg_generator.initialize()
            await self.figma_generator.initialize()
            await self.html_generator.initialize()
            await self.code_optimizer.initialize()
            
            self._running = True
            
            # Start message processing
            asyncio.create_task(self._process_messages())
            asyncio.create_task(self._process_queue())
            
            logger.info("Developer Agent started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Developer Agent: {e}")
            raise
    
    async def stop(self):
        """Stop the agent"""
        try:
            logger.info("Stopping Developer Agent")
            
            self._running = False
            
            # Wait for active sessions to complete
            for session_id in list(self._active_sessions.keys()):
                await self._cleanup_session(session_id)
            
            logger.info("Developer Agent stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Developer Agent: {e}")
    
    def set_communication(self, shared_memory: SharedMemory, message_queue: MessageQueue):
        """Set communication interfaces"""
        self.shared_memory = shared_memory
        self.message_queue = message_queue
        
        # Initialize generators with communication
        self.svg_generator.set_communication(shared_memory, message_queue)
        self.figma_generator.set_communication(shared_memory, message_queue)
        self.html_generator.set_communication(shared_memory, message_queue)
        self.code_optimizer.set_communication(shared_memory, message_queue)
    
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
            logger.info(f"Developer Agent received message: {action}")
            
            if action == "start_code_generation_workflow":
                await self._start_code_generation_workflow(message.payload.get("data", {}))
            
            elif action == "generate_svg":
                await self._generate_svg(message.payload.get("data", {}))
            
            elif action == "generate_figma":
                await self._generate_figma(message.payload.get("data", {}))
            
            elif action == "generate_html":
                await self._generate_html(message.payload.get("data", {}))
            
            elif action == "optimize_code":
                await self._optimize_code(message.payload.get("data", {}))
            
            elif action == "export_all_formats":
                await self._export_all_formats(message.payload.get("data", {}))
            
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
    
    async def _start_code_generation_workflow(self, data: Dict[str, Any]):
        """Start the code generation workflow"""
        try:
            session_id = data.get("session_id")
            design_id = data.get("design_id")
            
            logger.info(f"Starting code generation workflow: session={session_id}, design={design_id}")
            
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
            logger.error(f"Error starting code generation workflow: {e}")
    
    async def _process_task(self, task: Dict[str, Any]):
        """Process a queued task"""
        try:
            task_type = task.get("type")
            session_id = task.get("session_id")
            
            if task_type == "workflow":
                await self._execute_code_generation_workflow(session_id, task["data"])
            
            elif task_type == "svg_generation":
                await self._execute_svg_generation(session_id, task["data"])
            
            elif task_type == "figma_generation":
                await self._execute_figma_generation(session_id, task["data"])
            
            elif task_type == "html_generation":
                await self._execute_html_generation(session_id, task["data"])
            
            elif task_type == "code_optimization":
                await self._execute_code_optimization(session_id, task["data"])
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
    
    async def _execute_code_generation_workflow(self, session_id: str, data: Dict[str, Any]):
        """Execute the complete code generation workflow"""
        try:
            session = self._active_sessions.get(session_id)
            if not session:
                logger.error(f"Session not found: {session_id}")
                return
            
            design_id = session["design_id"]
            
            # Update progress
            await self._update_progress(session_id, 10, "Loading design blueprint")
            
            # Step 1: Get design blueprint
            blueprint = await self._get_design_blueprint(design_id)
            
            if not blueprint:
                logger.error(f"No blueprint found for design {design_id}")
                await self._mark_session_failed(session_id, "Missing design blueprint")
                return
            
            session["blueprint"] = blueprint
            
            # Step 2: Determine target formats
            target_formats = blueprint.get("exports", {}).keys()
            if not target_formats:
                target_formats = ["svg", "png"]  # Default formats
            
            await self._update_progress(session_id, 20, f"Generating {len(target_formats)} formats")
            
            # Step 3: Generate code for each format
            generation_results = {}
            
            for i, format_name in enumerate(target_formats):
                progress = 20 + (i / len(target_formats)) * 60
                await self._update_progress(session_id, int(progress), f"Generating {format_name.upper()}")
                
                try:
                    if format_name == "svg":
                        result = await self.svg_generator.generate_svg(blueprint)
                    elif format_name == "figma":
                        result = await self.figma_generator.generate_figma(blueprint)
                    elif format_name == "html":
                        result = await self.html_generator.generate_html(blueprint)
                    elif format_name == "png":
                        result = await self._generate_png_from_svg(blueprint, generation_results.get("svg"))
                    else:
                        logger.warning(f"Unsupported format: {format_name}")
                        continue
                    
                    if result.get("success"):
                        generation_results[format_name] = result
                    else:
                        logger.error(f"Failed to generate {format_name}: {result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"Error generating {format_name}: {e}")
                    continue
            
            # Step 4: Optimize generated code
            await self._update_progress(session_id, 85, "Optimizing generated code")
            
            optimization_results = {}
            for format_name, result in generation_results.items():
                try:
                    optimized = await self.code_optimizer.optimize_code(result, format_name)
                    if optimized.get("success"):
                        optimization_results[format_name] = optimized
                    else:
                        optimization_results[format_name] = result  # Use original if optimization fails
                except Exception as e:
                    logger.error(f"Error optimizing {format_name}: {e}")
                    optimization_results[format_name] = result  # Use original
            
            # Step 5: Finalize and store results
            await self._update_progress(session_id, 95, "Finalizing results")
            
            session["results"] = {
                "formats": optimization_results,
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "target_formats": list(target_formats),
                    "successful_formats": list(optimization_results.keys()),
                    "blueprint_id": blueprint.get("blueprint_id")
                }
            }
            
            # Step 6: Update design with generated code
            await self._update_design_with_code(design_id, session["results"])
            
            # Step 7: Complete session
            await self._update_progress(session_id, 100, "Code generation completed")
            await self._complete_session(session_id)
            
            # Notify Design Reviewer Agent
            await self._notify_next_agent(design_id, session_id)
            
            logger.info(f"Code generation workflow completed: session={session_id}")
            
        except Exception as e:
            logger.error(f"Error in code generation workflow: {e}")
            await self._mark_session_failed(session_id, str(e))
    
    async def _get_design_blueprint(self, design_id: str) -> Optional[Dict[str, Any]]:
        """Get design blueprint from shared memory"""
        try:
            if not self.shared_memory:
                return None
            
            design_data = await self.shared_memory.get_design_data(design_id)
            
            if not design_data:
                return None
            
            foreground_data = design_data.get("foreground_data", {})
            return foreground_data.get("blueprint")
            
        except Exception as e:
            logger.error(f"Error getting design blueprint: {e}")
            return None
    
    async def _generate_png_from_svg(self, blueprint: Dict[str, Any], 
                                   svg_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate PNG from SVG using headless browser or conversion service"""
        try:
            if not svg_result or not svg_result.get("success"):
                return {"success": False, "error": "No SVG available for PNG conversion"}
            
            svg_code = svg_result.get("svg_code", "")
            if not svg_code:
                return {"success": False, "error": "Empty SVG code"}
            
            # For now, return a placeholder PNG generation result
            # In production, this would use a service like Puppeteer or similar
            return {
                "success": True,
                "format": "png",
                "png_data": "base64_encoded_png_data_placeholder",
                "png_url": f"/generated/design_{blueprint.get('blueprint_id', 'unknown')}.png",
                "metadata": {
                    "width": blueprint.get("structure", {}).get("document", {}).get("dimensions", {}).get("width", 800),
                    "height": blueprint.get("structure", {}).get("document", {}).get("dimensions", {}).get("height", 600),
                    "format": "png",
                    "generated_from": "svg"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating PNG: {e}")
            return {"success": False, "error": str(e)}
    
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
    
    async def _update_design_with_code(self, design_id: str, results: Dict[str, Any]):
        """Update design data with generated code"""
        try:
            if not self.shared_memory:
                return
            
            design_data = await self.shared_memory.get_design_data(design_id)
            if not design_data:
                return
            
            # Update with generated code
            design_data["generated_code"] = {
                "formats": results.get("formats", {}),
                "metadata": results.get("metadata", {}),
                "generated_at": datetime.utcnow().isoformat(),
                "agent": self.agent_id
            }
            
            # Extract URLs for quick access
            code_urls = {}
            for format_name, format_result in results.get("formats", {}).items():
                if format_name == "svg" and "svg_code" in format_result:
                    code_urls["svg"] = f"/generated/{design_id}.svg"
                elif format_name == "html" and "html_code" in format_result:
                    code_urls["html"] = f"/generated/{design_id}.html"
                elif format_name == "figma" and "figma_code" in format_result:
                    code_urls["figma"] = f"/generated/{design_id}_figma.json"
                elif format_name == "png" and "png_url" in format_result:
                    code_urls["png"] = format_result["png_url"]
            
            design_data["export_urls"] = code_urls
            
            design_data["status"] = "code_generation"
            design_data["progress_percentage"] = 90  # Ready for review
            design_data["current_step"] = "Design review"
            design_data["current_agent"] = "design_reviewer"
            
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
                recipient="design_reviewer",
                type=MessageType.REQUEST,
                payload={
                    "action": "start_design_review_workflow",
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
            
            await self.message_queue.publish("agent.design_reviewer", message_dict)
            
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
    async def generate_code(self, blueprint: Dict[str, Any], 
                          target_formats: List[str]) -> Dict[str, Any]:
        """Generate code for specified formats"""
        try:
            results = {}
            
            for format_name in target_formats:
                if format_name == "svg":
                    result = await self.svg_generator.generate_svg(blueprint)
                elif format_name == "figma":
                    result = await self.figma_generator.generate_figma(blueprint)
                elif format_name == "html":
                    result = await self.html_generator.generate_html(blueprint)
                else:
                    continue
                
                if result.get("success"):
                    results[format_name] = result
            
            return {
                "success": len(results) > 0,
                "formats": results,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return {"success": False, "error": str(e)}
    
    async def optimize_generated_code(self, code_results: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize generated code"""
        try:
            optimized_results = {}
            
            for format_name, result in code_results.items():
                optimized = await self.code_optimizer.optimize_code(result, format_name)
                if optimized.get("success"):
                    optimized_results[format_name] = optimized
                else:
                    optimized_results[format_name] = result  # Use original
            
            return {
                "success": True,
                "formats": optimized_results,
                "optimized_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing code: {e}")
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
            "queue_size": self._processing_queue.qsize(),
            "supported_formats": ["svg", "figma", "html", "png"]
        }
    
    # Missing method implementations (placeholder methods)
    async def _generate_svg(self, data: Dict[str, Any]):
        """Handle generate SVG message"""
        try:
            # Implementation placeholder
            logger.info("Generate SVG message received")
        except Exception as e:
            logger.error(f"Error in generate SVG: {e}")
    
    async def _generate_figma(self, data: Dict[str, Any]):
        """Handle generate Figma message"""
        try:
            # Implementation placeholder
            logger.info("Generate Figma message received")
        except Exception as e:
            logger.error(f"Error in generate Figma: {e}")
    
    async def _generate_html(self, data: Dict[str, Any]):
        """Handle generate HTML message"""
        try:
            # Implementation placeholder
            logger.info("Generate HTML message received")
        except Exception as e:
            logger.error(f"Error in generate HTML: {e}")
    
    async def _optimize_code(self, data: Dict[str, Any]):
        """Handle optimize code message"""
        try:
            # Implementation placeholder
            logger.info("Optimize code message received")
        except Exception as e:
            logger.error(f"Error in optimize code: {e}")
    
    async def _export_all_formats(self, data: Dict[str, Any]):
        """Handle export all formats message"""
        try:
            # Implementation placeholder
            logger.info("Export all formats message received")
        except Exception as e:
            logger.error(f"Error in export all formats: {e}")
    
    async def _execute_svg_generation(self, session_id: str, data: Dict[str, Any]):
        """Execute SVG generation task"""
        try:
            # Implementation placeholder
            logger.info(f"Execute SVG generation for session: {session_id}")
        except Exception as e:
            logger.error(f"Error in execute SVG generation: {e}")
    
    async def _execute_figma_generation(self, session_id: str, data: Dict[str, Any]):
        """Execute Figma generation task"""
        try:
            # Implementation placeholder
            logger.info(f"Execute Figma generation for session: {session_id}")
        except Exception as e:
            logger.error(f"Error in execute Figma generation: {e}")
    
    async def _execute_html_generation(self, session_id: str, data: Dict[str, Any]):
        """Execute HTML generation task"""
        try:
            # Implementation placeholder
            logger.info(f"Execute HTML generation for session: {session_id}")
        except Exception as e:
            logger.error(f"Error in execute HTML generation: {e}")
    
    async def _execute_code_optimization(self, session_id: str, data: Dict[str, Any]):
        """Execute code optimization task"""
        try:
            # Implementation placeholder
            logger.info(f"Execute code optimization for session: {session_id}")
        except Exception as e:
            logger.error(f"Error in execute code optimization: {e}")
