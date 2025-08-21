"""
Refinement Loop for Background Designer

Implements self-refinement logic to iteratively improve
background generation quality.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from structlog import get_logger

logger = get_logger(__name__)


class RefinementLoop:
    """
    Handles iterative refinement of background generation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Refinement parameters
        self.max_iterations = config.get("max_iterations", 3)
        self.quality_threshold = config.get("quality_threshold", 0.8)
        self.improvement_threshold = config.get("improvement_threshold", 0.1)
        
        # Track refinement history
        self.refinement_history = []
    
    async def should_refine(self, 
                           current_result: Dict[str, Any],
                           iteration: int) -> Tuple[bool, str]:
        """
        Determine if refinement is needed
        
        Args:
            current_result: Current generation result with quality score
            iteration: Current iteration number
        
        Returns:
            (should_refine, reason)
        """
        try:
            quality_score = current_result.get("quality_score", 0)
            has_text = current_result.get("has_text", False)
            issues = current_result.get("validation_details", {}).get("issues", [])
            
            # Always refine if text is detected
            if has_text:
                return True, "Text detected in background - must refine"
            
            # Refine if quality is below threshold
            if quality_score < self.quality_threshold:
                return True, f"Quality score {quality_score:.2f} below threshold {self.quality_threshold}"
            
            # Refine if there are significant issues
            critical_issues = [issue for issue in issues if self._is_critical_issue(issue)]
            if critical_issues:
                return True, f"Critical issues detected: {', '.join(critical_issues)}"
            
            # Check if we've reached max iterations
            if iteration >= self.max_iterations:
                return False, f"Max iterations ({self.max_iterations}) reached"
            
            # Check for potential improvement based on history
            if self._has_improvement_potential(current_result):
                return True, "Potential for improvement detected"
            
            return False, "Quality acceptable, no refinement needed"
            
        except Exception as e:
            logger.error(f"Error determining refinement need: {e}")
            return False, f"Error in refinement decision: {str(e)}"
    
    def generate_refinement_strategy(self, 
                                   current_result: Dict[str, Any],
                                   iteration: int) -> Dict[str, Any]:
        """
        Generate strategy for next refinement iteration
        
        Args:
            current_result: Current result to improve
            iteration: Current iteration number
        
        Returns:
            Refinement strategy with specific actions
        """
        try:
            strategy = {
                "iteration": iteration + 1,
                "timestamp": datetime.utcnow().isoformat(),
                "actions": [],
                "prompt_modifications": [],
                "parameter_adjustments": {},
                "focus_areas": []
            }
            
            quality_score = current_result.get("quality_score", 0)
            has_text = current_result.get("has_text", False)
            validation_details = current_result.get("validation_details", {})
            issues = validation_details.get("issues", [])
            
            # Address text detection
            if has_text:
                strategy["actions"].append("strengthen_text_avoidance")
                strategy["prompt_modifications"].extend([
                    "Add stronger negative prompts for text",
                    "Emphasize abstract patterns only",
                    "Specify 'completely text-free background'"
                ])
                strategy["focus_areas"].append("text_elimination")
            
            # Address quality issues
            if quality_score < 0.7:
                strategy["actions"].append("improve_overall_quality")
                strategy["prompt_modifications"].extend([
                    "Add quality enhancement keywords",
                    "Specify professional photography style",
                    "Request higher resolution details"
                ])
                strategy["parameter_adjustments"]["guidance_scale"] = 8.0
                strategy["parameter_adjustments"]["num_inference_steps"] = 30
                strategy["focus_areas"].append("quality_enhancement")
            
            # Address specific validation issues
            for issue in issues:
                if "contrast" in issue.lower():
                    strategy["actions"].append("improve_contrast")
                    strategy["prompt_modifications"].append("high contrast, vibrant colors")
                    strategy["focus_areas"].append("contrast_improvement")
                
                elif "composition" in issue.lower():
                    strategy["actions"].append("improve_composition")
                    strategy["prompt_modifications"].append("clean composition, balanced layout")
                    strategy["focus_areas"].append("composition_refinement")
                
                elif "noise" in issue.lower():
                    strategy["actions"].append("reduce_noise")
                    strategy["prompt_modifications"].append("clean, smooth, professional")
                    strategy["parameter_adjustments"]["guidance_scale"] = 6.0
                    strategy["focus_areas"].append("noise_reduction")
                
                elif "sharpness" in issue.lower():
                    strategy["actions"].append("improve_sharpness")
                    strategy["prompt_modifications"].append("sharp focus, crisp details")
                    strategy["focus_areas"].append("sharpness_enhancement")
            
            # Add iteration-specific adjustments
            strategy.update(self._get_iteration_specific_adjustments(iteration))
            
            # Store strategy in history
            self.refinement_history.append(strategy)
            
            logger.info(f"Generated refinement strategy for iteration {iteration + 1}: {strategy['actions']}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating refinement strategy: {e}")
            return self._get_fallback_strategy(iteration)
    
    def apply_refinement_strategy(self, 
                                 strategy: Dict[str, Any],
                                 original_prompt: str,
                                 original_params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Apply refinement strategy to generate new prompt and parameters
        
        Args:
            strategy: Refinement strategy
            original_prompt: Original generation prompt
            original_params: Original generation parameters
        
        Returns:
            (refined_prompt, refined_parameters)
        """
        try:
            refined_prompt = original_prompt
            refined_params = original_params.copy()
            
            # Apply prompt modifications
            prompt_modifications = strategy.get("prompt_modifications", [])
            if prompt_modifications:
                # Add modifications to prompt
                modifications_text = ", ".join(prompt_modifications)
                refined_prompt = f"{original_prompt}, {modifications_text}"
            
            # Apply parameter adjustments
            parameter_adjustments = strategy.get("parameter_adjustments", {})
            refined_params.update(parameter_adjustments)
            
            # Apply action-specific refinements
            actions = strategy.get("actions", [])
            
            if "strengthen_text_avoidance" in actions:
                refined_prompt = self._strengthen_text_avoidance(refined_prompt)
            
            if "improve_overall_quality" in actions:
                refined_prompt = self._add_quality_enhancers(refined_prompt)
            
            if "improve_contrast" in actions:
                refined_prompt = self._add_contrast_enhancers(refined_prompt)
                refined_params["guidance_scale"] = max(refined_params.get("guidance_scale", 7.5), 8.0)
            
            if "reduce_noise" in actions:
                refined_params["guidance_scale"] = min(refined_params.get("guidance_scale", 7.5), 6.0)
                refined_params["num_inference_steps"] = max(refined_params.get("num_inference_steps", 25), 30)
            
            logger.info(f"Applied refinement strategy: {len(prompt_modifications)} prompt mods, {len(parameter_adjustments)} param adjustments")
            return refined_prompt, refined_params
            
        except Exception as e:
            logger.error(f"Error applying refinement strategy: {e}")
            return original_prompt, original_params
    
    def analyze_refinement_progress(self) -> Dict[str, Any]:
        """
        Analyze the progress of refinement iterations
        
        Returns:
            Progress analysis with trends and recommendations
        """
        try:
            if not self.refinement_history:
                return {"status": "no_refinements", "message": "No refinement history available"}
            
            analysis = {
                "total_iterations": len(self.refinement_history),
                "strategies_used": [],
                "focus_areas": [],
                "parameter_trends": {},
                "effectiveness": {},
                "recommendations": []
            }
            
            # Analyze strategies and focus areas
            all_actions = []
            all_focus_areas = []
            
            for strategy in self.refinement_history:
                all_actions.extend(strategy.get("actions", []))
                all_focus_areas.extend(strategy.get("focus_areas", []))
            
            analysis["strategies_used"] = list(set(all_actions))
            analysis["focus_areas"] = list(set(all_focus_areas))
            
            # Analyze parameter trends
            guidance_scales = [s.get("parameter_adjustments", {}).get("guidance_scale") 
                             for s in self.refinement_history if s.get("parameter_adjustments", {}).get("guidance_scale")]
            if guidance_scales:
                analysis["parameter_trends"]["guidance_scale"] = {
                    "values": guidance_scales,
                    "trend": "increasing" if guidance_scales[-1] > guidance_scales[0] else "decreasing"
                }
            
            # Generate recommendations
            if "text_elimination" in all_focus_areas:
                analysis["recommendations"].append("Consider using different T2I model parameters for text-free generation")
            
            if len(self.refinement_history) >= self.max_iterations:
                analysis["recommendations"].append("Max iterations reached - consider adjusting base prompt strategy")
            
            if "quality_enhancement" in all_focus_areas:
                analysis["recommendations"].append("Base prompt may need fundamental revision for better quality")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing refinement progress: {e}")
            return {"status": "error", "error": str(e)}
    
    def reset_refinement_history(self):
        """Reset refinement history for new generation session"""
        self.refinement_history = []
        logger.info("Refinement history reset")
    
    def _is_critical_issue(self, issue: str) -> bool:
        """Determine if an issue is critical and requires refinement"""
        critical_keywords = [
            "text detected", "text visible", "readable text",
            "low quality", "poor quality", "artifacts",
            "too dark", "too bright", "no suitable space"
        ]
        
        issue_lower = issue.lower()
        return any(keyword in issue_lower for keyword in critical_keywords)
    
    def _has_improvement_potential(self, current_result: Dict[str, Any]) -> bool:
        """Check if there's potential for improvement based on history"""
        if len(self.refinement_history) < 2:
            return True  # Always try to improve in early iterations
        
        # Check if quality has been stagnant
        current_quality = current_result.get("quality_score", 0)
        
        # If we haven't seen significant improvement, might still be worth trying
        return current_quality < 0.9  # Always room for improvement below 90%
    
    def _get_iteration_specific_adjustments(self, iteration: int) -> Dict[str, Any]:
        """Get adjustments specific to iteration number"""
        adjustments = {}
        
        if iteration == 1:
            # Second iteration - moderate adjustments
            adjustments["approach"] = "moderate_refinement"
        elif iteration == 2:
            # Third iteration - more aggressive changes
            adjustments["approach"] = "aggressive_refinement"
            adjustments["parameter_adjustments"] = {"guidance_scale": 9.0}
        else:
            # Later iterations - experimental approaches
            adjustments["approach"] = "experimental_refinement"
            adjustments["parameter_adjustments"] = {"guidance_scale": 5.0, "num_inference_steps": 40}
        
        return adjustments
    
    def _strengthen_text_avoidance(self, prompt: str) -> str:
        """Strengthen text avoidance in prompt"""
        text_avoidance_terms = [
            "completely text-free",
            "no typography",
            "abstract patterns only",
            "pure background design",
            "zero text elements"
        ]
        
        additional_terms = ", ".join(text_avoidance_terms)
        return f"{prompt}, {additional_terms}"
    
    def _add_quality_enhancers(self, prompt: str) -> str:
        """Add quality enhancement terms"""
        quality_terms = [
            "ultra high quality",
            "professional grade",
            "studio quality",
            "perfect details",
            "masterpiece"
        ]
        
        additional_terms = ", ".join(quality_terms)
        return f"{prompt}, {additional_terms}"
    
    def _add_contrast_enhancers(self, prompt: str) -> str:
        """Add contrast enhancement terms"""
        contrast_terms = [
            "high contrast",
            "vibrant colors",
            "rich saturation",
            "dynamic range"
        ]
        
        additional_terms = ", ".join(contrast_terms)
        return f"{prompt}, {additional_terms}"
    
    def _get_fallback_strategy(self, iteration: int) -> Dict[str, Any]:
        """Get fallback strategy when generation fails"""
        return {
            "iteration": iteration + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "actions": ["improve_overall_quality"],
            "prompt_modifications": ["enhance quality", "professional design"],
            "parameter_adjustments": {"guidance_scale": 7.5},
            "focus_areas": ["general_improvement"],
            "approach": "fallback_strategy"
        }
