"""
Background Designer Agent

ReAct agent for generating text-free backgrounds using Text-to-Image models.
Implements self-refinement loop to ensure high-quality outputs.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import uuid
import base64
from structlog import get_logger

from .prompt_generator import PromptGenerator
from .image_validator import ImageValidator
from .refinement_loop import RefinementLoop
from .tools.find_image_path import FindImagePathTool
from .tools.t2i_interface import TextToImageInterface
from .tools.text_checker import TextChecker
from .tools.image_resizer import ImageResizer
from communication.protocol import MessageProtocol, AgentIdentifiers, WorkflowProtocol
from communication.message_queue import MessageQueue
from memory_manager.shared_memory import SharedMemory
from memory_manager.session_manager import SessionManager

logger = get_logger(__name__)


class BackgroundDesignerAgent:
    """
    Background Designer Agent - Generates text-free backgrounds using ReAct pattern
    """
    
    def __init__(self, shared_memory: SharedMemory, message_queue: MessageQueue,
                 session_manager: SessionManager, config: Dict[str, Any] = None):
        self.agent_id = AgentIdentifiers.BACKGROUND_DESIGNER
        self.shared_memory = shared_memory
        self.message_queue = message_queue
        self.session_manager = session_manager
        self.config = config or {}
        
        # Initialize components
        self.prompt_generator = PromptGenerator(config.get("prompt_generator", {}))
        self.image_validator = ImageValidator(config.get("image_validator", {}))
        self.refinement_loop = RefinementLoop(config.get("refinement_loop", {}))
        
        # Initialize tools
        self.find_image_tool = FindImagePathTool(config.get("find_image_tool", {}))
        self.t2i_interface = TextToImageInterface(config.get("t2i_interface", {}))
        self.text_checker = TextChecker(config.get("text_checker", {}))
        self.image_resizer = ImageResizer(config.get("image_resizer", {}))
        
        self._running = False
        self._current_session = None
        
        # ReAct state
        self.thoughts = []
        self.actions = []
        self.observations = []
        
    async def start(self):
        """Start the background designer agent"""
        try:
            self._running = True
            logger.info(f"Background Designer Agent {self.agent_id} starting...")
            
            # Subscribe to messages
            await self.message_queue.subscribe(
                f"agent.{self.agent_id}",
                self._handle_message
            )
            
            # Subscribe to workflow events
            await self.message_queue.subscribe(
                WorkflowProtocol.CHANNELS["background_design"],
                self._handle_workflow_message
            )
            
            logger.info(f"Background Designer Agent {self.agent_id} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Background Designer Agent: {e}")
            raise
    
    async def stop(self):
        """Stop the background designer agent"""
        try:
            self._running = False
            logger.info(f"Background Designer Agent {self.agent_id} stopping...")
            
            # Unsubscribe from messages
            await self.message_queue.unsubscribe(f"agent.{self.agent_id}")
            await self.message_queue.unsubscribe(WorkflowProtocol.CHANNELS["background_design"])
            
            logger.info(f"Background Designer Agent {self.agent_id} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Background Designer Agent: {e}")
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle direct messages to the agent"""
        try:
            action = message.get("action")
            
            if action == "generate_background":
                await self._generate_background(message)
            elif action == "validate_background":
                await self._validate_background(message)
            elif action == "get_status":
                await self._send_status(message)
            else:
                logger.warning(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_workflow_message(self, message: Dict[str, Any]):
        """Handle workflow messages"""
        try:
            workflow_action = message.get("workflow_action")
            
            if workflow_action == "start_background_design":
                await self._start_background_design_workflow(message)
            elif workflow_action == "refine_background":
                await self._refine_background_workflow(message)
            else:
                logger.warning(f"Unknown workflow action: {workflow_action}")
                
        except Exception as e:
            logger.error(f"Error handling workflow message: {e}")
    
    async def _start_background_design_workflow(self, message: Dict[str, Any]):
        """Start background design workflow using ReAct pattern"""
        try:
            session_id = message.get("session_id")
            campaign_data = await self.shared_memory.get_campaign_data(session_id)
            
            if not campaign_data:
                logger.error(f"No campaign data found for session {session_id}")
                return
            
            self._current_session = session_id
            
            # Reset ReAct state
            self.thoughts = []
            self.actions = []
            self.observations = []
            
            # Start ReAct loop
            await self._react_loop(campaign_data)
            
        except Exception as e:
            logger.error(f"Error in background design workflow: {e}")
            await self._send_error_message(session_id, str(e))
    
    async def _react_loop(self, campaign_data: Dict[str, Any]):
        """Main ReAct loop for background generation"""
        max_iterations = self.config.get("max_react_iterations", 5)
        
        for iteration in range(max_iterations):
            try:
                logger.info(f"ReAct iteration {iteration + 1}/{max_iterations}")
                
                # Think phase
                thought = await self._think(campaign_data, iteration)
                self.thoughts.append(thought)
                logger.info(f"Thought: {thought}")
                
                # Act phase
                action_result = await self._act(thought, campaign_data)
                self.actions.append(action_result)
                logger.info(f"Action: {action_result['action']}")
                
                # Observe phase
                observation = await self._observe(action_result)
                self.observations.append(observation)
                logger.info(f"Observation: {observation}")
                
                # Check if we have a successful result
                if observation.get("success") and observation.get("quality_score", 0) >= 0.8:
                    logger.info("Background generation completed successfully")
                    await self._complete_background_design(observation)
                    break
                    
                # If we're on the last iteration and still no success, use best result
                if iteration == max_iterations - 1:
                    logger.warning("Max iterations reached, using best available result")
                    await self._complete_background_design(observation)
                    
            except Exception as e:
                logger.error(f"Error in ReAct iteration {iteration + 1}: {e}")
                continue
    
    async def _think(self, campaign_data: Dict[str, Any], iteration: int) -> str:
        """Think phase - analyze situation and plan next action"""
        try:
            strategic_direction = campaign_data.get("strategic_direction", {})
            previous_observations = self.observations[-2:] if len(self.observations) >= 2 else self.observations
            
            context = {
                "iteration": iteration,
                "campaign_brief": campaign_data.get("brief", {}),
                "strategic_direction": strategic_direction,
                "previous_observations": previous_observations,
                "target_dimensions": campaign_data.get("brief", {}).get("dimensions", {}),
                "brand_colors": strategic_direction.get("color_palette", []),
                "mood": strategic_direction.get("mood", ""),
                "style": strategic_direction.get("visual_style", "")
            }
            
            # Generate thought using LLM
            thought_prompt = self._build_thinking_prompt(context)
            thought = await self._generate_thought(thought_prompt)
            
            return thought
            
        except Exception as e:
            logger.error(f"Error in think phase: {e}")
            return f"Error in thinking: {str(e)}"
    
    async def _act(self, thought: str, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Act phase - execute action based on thought"""
        try:
            # Determine action from thought
            action_type = self._parse_action_from_thought(thought)
            
            if action_type == "find_existing":
                return await self._find_existing_images(campaign_data)
            elif action_type == "generate_new":
                return await self._generate_new_background(thought, campaign_data)
            elif action_type == "refine_prompt":
                return await self._refine_generation_prompt(thought, campaign_data)
            elif action_type == "resize_image":
                return await self._resize_existing_image(thought, campaign_data)
            else:
                return await self._generate_new_background(thought, campaign_data)
                
        except Exception as e:
            logger.error(f"Error in act phase: {e}")
            return {"action": "error", "error": str(e)}
    
    async def _observe(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Observe phase - evaluate action result"""
        try:
            action_type = action_result.get("action")
            
            if action_type == "generate_image":
                return await self._observe_generated_image(action_result)
            elif action_type == "find_images":
                return await self._observe_found_images(action_result)
            elif action_type == "resize_image":
                return await self._observe_resized_image(action_result)
            else:
                return {"success": False, "message": f"Unknown action type: {action_type}"}
                
        except Exception as e:
            logger.error(f"Error in observe phase: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_new_background(self, thought: str, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new background using T2I model"""
        try:
            strategic_direction = campaign_data.get("strategic_direction", {})
            dimensions = campaign_data.get("brief", {}).get("dimensions", {})
            
            # Generate optimized prompt
            prompt_data = {
                "mood": strategic_direction.get("mood", ""),
                "style": strategic_direction.get("visual_style", ""),
                "colors": strategic_direction.get("color_palette", []),
                "industry": campaign_data.get("brief", {}).get("industry", ""),
                "thought": thought,
                "ensure_no_text": True
            }
            
            prompt = await self.prompt_generator.generate_background_prompt(prompt_data)
            
            # Generate image
            image_data = await self.t2i_interface.generate_image(
                prompt=prompt,
                width=dimensions.get("width", 1024),
                height=dimensions.get("height", 1024),
                guidance_scale=7.5,
                num_inference_steps=25
            )
            
            return {
                "action": "generate_image",
                "prompt": prompt,
                "image_data": image_data,
                "dimensions": dimensions
            }
            
        except Exception as e:
            logger.error(f"Error generating background: {e}")
            return {"action": "generate_image", "error": str(e)}
    
    async def _observe_generated_image(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Observe and validate generated image"""
        try:
            image_data = action_result.get("image_data")
            if not image_data:
                return {"success": False, "message": "No image data generated"}
            
            # Validate image quality
            validation_result = await self.image_validator.validate_background(image_data)
            
            # Check for text in image
            has_text = await self.text_checker.detect_text(image_data)
            
            # Calculate overall quality score
            quality_score = validation_result.get("quality_score", 0)
            if has_text:
                quality_score *= 0.5  # Heavily penalize text presence
            
            return {
                "success": quality_score >= 0.7,
                "quality_score": quality_score,
                "has_text": has_text,
                "validation_details": validation_result,
                "image_data": image_data,
                "message": f"Generated image with quality score: {quality_score:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Error observing generated image: {e}")
            return {"success": False, "error": str(e)}
    
    async def _complete_background_design(self, final_observation: Dict[str, Any]):
        """Complete background design and notify other agents"""
        try:
            session_id = self._current_session
            
            # Store background data in shared memory
            background_data = {
                "image_data": final_observation.get("image_data"),
                "quality_score": final_observation.get("quality_score", 0),
                "validation_details": final_observation.get("validation_details", {}),
                "generation_metadata": {
                    "thoughts": self.thoughts,
                    "actions": self.actions,
                    "observations": self.observations,
                    "completed_at": datetime.utcnow().isoformat()
                }
            }
            
            await self.shared_memory.set_background_data(session_id, background_data)
            
            # Notify workflow completion
            completion_message = MessageProtocol.create_workflow_message(
                workflow_action="background_design_completed",
                session_id=session_id,
                agent_id=self.agent_id,
                data={
                    "background_ready": True,
                    "quality_score": final_observation.get("quality_score", 0)
                }
            )
            
            await self.message_queue.publish(
                WorkflowProtocol.CHANNELS["foreground_design"],
                completion_message
            )
            
            logger.info(f"Background design completed for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error completing background design: {e}")
    
    def _build_thinking_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for thinking phase"""
        return f"""
You are an AI background designer agent using ReAct reasoning. Analyze the current situation and plan your next action.

Context:
- Iteration: {context['iteration']}
- Campaign Brief: {context['campaign_brief']}
- Strategic Direction: {context['strategic_direction']}
- Previous Observations: {context['previous_observations']}
- Target Dimensions: {context['target_dimensions']}

Your goal is to create a high-quality, text-free background that matches the strategic direction.

Think about:
1. What has been tried before (if any)?
2. What worked or didn't work?
3. What should be the next action?
4. How to improve quality or avoid previous issues?

Output a clear thought about what to do next and why.
"""
    
    async def _generate_thought(self, prompt: str) -> str:
        """Generate thought using LLM"""
        # This would integrate with your LLM interface
        # For now, return a placeholder
        return "I need to generate a background that matches the strategic direction while ensuring no text is present."
    
    def _parse_action_from_thought(self, thought: str) -> str:
        """Parse action type from thought"""
        thought_lower = thought.lower()
        
        if "find" in thought_lower or "existing" in thought_lower:
            return "find_existing"
        elif "refine" in thought_lower or "improve prompt" in thought_lower:
            return "refine_prompt"
        elif "resize" in thought_lower or "scale" in thought_lower:
            return "resize_image"
        else:
            return "generate_new"
    
    async def _find_existing_images(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find existing suitable images"""
        try:
            # Use find image tool
            search_criteria = {
                "industry": campaign_data.get("brief", {}).get("industry", ""),
                "mood": campaign_data.get("strategic_direction", {}).get("mood", ""),
                "dimensions": campaign_data.get("brief", {}).get("dimensions", {})
            }
            
            found_images = await self.find_image_tool.find_images(search_criteria)
            
            return {
                "action": "find_images",
                "found_images": found_images,
                "count": len(found_images)
            }
            
        except Exception as e:
            logger.error(f"Error finding existing images: {e}")
            return {"action": "find_images", "error": str(e)}
    
    async def _observe_found_images(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Observe found images"""
        found_images = action_result.get("found_images", [])
        
        if not found_images:
            return {
                "success": False,
                "message": "No suitable existing images found"
            }
        
        # Evaluate best image
        best_image = found_images[0]  # Simplified selection
        
        return {
            "success": True,
            "quality_score": 0.8,  # Placeholder
            "image_data": best_image.get("data"),
            "message": f"Found {len(found_images)} suitable images"
        }
    
    async def _refine_generation_prompt(self, thought: str, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine the generation prompt based on thought"""
        # Implementation would refine prompt based on previous failures
        return await self._generate_new_background(thought, campaign_data)
    
    async def _resize_existing_image(self, thought: str, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resize existing image"""
        # Implementation would resize an existing image
        return {"action": "resize_image", "message": "Image resized"}
    
    async def _observe_resized_image(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Observe resized image"""
        return {
            "success": True,
            "quality_score": 0.8,
            "message": "Image resized successfully"
        }
    
    async def _send_error_message(self, session_id: str, error: str):
        """Send error message"""
        error_message = MessageProtocol.create_workflow_message(
            workflow_action="background_design_failed",
            session_id=session_id,
            agent_id=self.agent_id,
            data={"error": error}
        )
        
        await self.message_queue.publish(
            WorkflowProtocol.CHANNELS["error_handling"],
            error_message
        )
    
    async def _send_status(self, message: Dict[str, Any]):
        """Send agent status"""
        status = {
            "agent_id": self.agent_id,
            "running": self._running,
            "current_session": self._current_session,
            "react_state": {
                "thoughts_count": len(self.thoughts),
                "actions_count": len(self.actions),
                "observations_count": len(self.observations)
            }
        }
        
        response = MessageProtocol.create_response(
            request_id=message.get("request_id"),
            agent_id=self.agent_id,
            data=status
        )
        
        await self.message_queue.publish(
            message.get("reply_to", f"agent.{message.get('from_agent')}"),
            response
        )
