"""
Multimodal Large Language Model Interface

Interface for multimodal models that can process both text and images,
used for design review, image analysis, and visual understanding tasks.
"""

import base64
import io
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from PIL import Image
import json
import asyncio
from structlog import get_logger

logger = get_logger(__name__)


class MLLMProvider(Enum):
    """Supported multimodal LLM providers"""
    OPENAI_VISION = "openai_vision"
    ANTHROPIC_VISION = "anthropic_vision"
    GOOGLE_VISION = "google_vision"
    LOCAL_VISION = "local_vision"


class MultimodalLLMInterface:
    """
    Interface for Multimodal Large Language Models
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Provider configurations
        self.providers_config = {
            MLLMProvider.OPENAI_VISION: config.get("openai_vision", {}),
            MLLMProvider.ANTHROPIC_VISION: config.get("anthropic_vision", {}),
            MLLMProvider.GOOGLE_VISION: config.get("google_vision", {}),
            MLLMProvider.LOCAL_VISION: config.get("local_vision", {})
        }
        
        # Default provider
        self.default_provider = MLLMProvider(config.get("default_provider", "openai_vision"))
        
        # Model mappings
        self.model_mappings = {
            "gpt-4-vision-preview": MLLMProvider.OPENAI_VISION,
            "gpt-4-vision": MLLMProvider.OPENAI_VISION,
            "claude-3-vision": MLLMProvider.ANTHROPIC_VISION,
            "gemini-pro-vision": MLLMProvider.GOOGLE_VISION,
            "llava": MLLMProvider.LOCAL_VISION
        }
        
        # Initialize clients
        self._clients = {}
        asyncio.create_task(self._initialize_clients())
    
    async def analyze_image(self, 
                           image_data: str,
                           prompt: str,
                           model: Optional[str] = None,
                           max_tokens: int = 1000,
                           **kwargs) -> str:
        """
        Analyze image with text prompt
        
        Args:
            image_data: Base64 encoded image data
            prompt: Analysis prompt
            model: Model name (optional)
            max_tokens: Maximum response tokens
            **kwargs: Additional parameters
        
        Returns:
            Analysis result as text
        """
        try:
            provider, model_name = self._resolve_model(model)
            
            logger.info(f"Analyzing image with {provider.value}:{model_name}")
            
            # Prepare image for API
            image_url = self._prepare_image_for_api(image_data)
            
            # Analyze based on provider
            if provider == MLLMProvider.OPENAI_VISION:
                result = await self._analyze_openai_vision(
                    image_url, prompt, model_name, max_tokens, **kwargs
                )
            elif provider == MLLMProvider.ANTHROPIC_VISION:
                result = await self._analyze_anthropic_vision(
                    image_url, prompt, model_name, max_tokens, **kwargs
                )
            elif provider == MLLMProvider.GOOGLE_VISION:
                result = await self._analyze_google_vision(
                    image_url, prompt, model_name, max_tokens, **kwargs
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            logger.info(f"Image analysis completed: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
    
    async def analyze_design_quality(self, 
                                   image_data: str,
                                   criteria: List[str] = None) -> Dict[str, Any]:
        """
        Analyze design quality of an image
        
        Args:
            image_data: Base64 encoded image data
            criteria: Specific criteria to evaluate
        
        Returns:
            Detailed quality analysis
        """
        try:
            default_criteria = [
                "visual hierarchy",
                "color harmony", 
                "typography quality",
                "composition balance",
                "brand consistency",
                "readability",
                "aesthetic appeal",
                "professional appearance"
            ]
            
            evaluation_criteria = criteria or default_criteria
            
            prompt = f"""
            Please analyze this design image for quality across these criteria:
            {', '.join(evaluation_criteria)}
            
            For each criterion, provide:
            1. Score (1-10)
            2. Brief explanation
            3. Specific observations
            4. Improvement suggestions (if score < 8)
            
            Also provide:
            - Overall quality score (1-10)
            - Strengths (top 3)
            - Areas for improvement (top 3)
            - Target audience assessment
            - Design style classification
            
            Respond in JSON format with this structure:
            {{
                "overall_score": number,
                "criteria_scores": {{
                    "criterion_name": {{
                        "score": number,
                        "explanation": "string",
                        "observations": ["string"],
                        "improvements": ["string"]
                    }}
                }},
                "strengths": ["string"],
                "improvements": ["string"],
                "target_audience": "string",
                "design_style": "string",
                "professional_level": "string"
            }}
            """
            
            response = await self.analyze_image(image_data, prompt, max_tokens=2000)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                result["analysis_type"] = "design_quality"
                return result
            except json.JSONDecodeError:
                # Fallback parsing
                return await self._parse_design_analysis(response)
                
        except Exception as e:
            logger.error(f"Error analyzing design quality: {e}")
            raise
    
    async def detect_design_elements(self, 
                                   image_data: str,
                                   element_types: List[str] = None) -> Dict[str, Any]:
        """
        Detect and analyze specific design elements
        
        Args:
            image_data: Base64 encoded image data
            element_types: Types of elements to detect
        
        Returns:
            Detected elements analysis
        """
        try:
            default_elements = [
                "text/typography",
                "logos/branding", 
                "buttons/CTAs",
                "images/graphics",
                "colors/gradients",
                "shapes/lines",
                "icons/symbols"
            ]
            
            elements_to_detect = element_types or default_elements
            
            prompt = f"""
            Analyze this image and detect the following design elements:
            {', '.join(elements_to_detect)}
            
            For each element type found, provide:
            1. Presence (true/false)
            2. Count (if multiple)
            3. Locations (describe where they appear)
            4. Descriptions (what they look like)
            5. Quality assessment (1-10)
            6. Effectiveness (how well they serve their purpose)
            
            Also analyze:
            - Layout structure
            - Visual flow/hierarchy
            - Color scheme
            - Space utilization
            - Accessibility considerations
            
            Respond in JSON format:
            {{
                "elements": {{
                    "element_type": {{
                        "present": boolean,
                        "count": number,
                        "locations": ["string"],
                        "descriptions": ["string"],
                        "quality_score": number,
                        "effectiveness_score": number
                    }}
                }},
                "layout": {{
                    "structure": "string",
                    "visual_flow": "string",
                    "hierarchy_score": number
                }},
                "color_analysis": {{
                    "scheme_type": "string",
                    "dominant_colors": ["string"],
                    "harmony_score": number
                }},
                "accessibility": {{
                    "contrast_score": number,
                    "readability_score": number,
                    "issues": ["string"]
                }}
            }}
            """
            
            response = await self.analyze_image(image_data, prompt, max_tokens=2000)
            
            try:
                result = json.loads(response)
                result["analysis_type"] = "design_elements"
                return result
            except json.JSONDecodeError:
                return await self._parse_elements_analysis(response)
                
        except Exception as e:
            logger.error(f"Error detecting design elements: {e}")
            raise
    
    async def compare_designs(self, 
                            image_data_1: str,
                            image_data_2: str,
                            comparison_criteria: List[str] = None) -> Dict[str, Any]:
        """
        Compare two design images
        
        Args:
            image_data_1: First image (base64)
            image_data_2: Second image (base64)
            comparison_criteria: Criteria for comparison
        
        Returns:
            Detailed comparison analysis
        """
        try:
            default_criteria = [
                "visual appeal",
                "brand consistency",
                "message clarity",
                "professional quality",
                "target audience fit",
                "effectiveness"
            ]
            
            criteria = comparison_criteria or default_criteria
            
            # For simplicity, analyze each image separately first
            # In a real implementation, you might use a model that can handle multiple images
            
            analysis_1 = await self.analyze_design_quality(image_data_1, criteria)
            analysis_2 = await self.analyze_design_quality(image_data_2, criteria)
            
            # Generate comparison
            comparison_prompt = f"""
            Compare these two design analyses and provide a detailed comparison:
            
            Design 1 Analysis: {json.dumps(analysis_1, indent=2)}
            Design 2 Analysis: {json.dumps(analysis_2, indent=2)}
            
            Provide comparison in JSON format:
            {{
                "winner": {{
                    "overall": "design_1|design_2|tie",
                    "reasoning": "string"
                }},
                "criteria_comparison": {{
                    "criterion_name": {{
                        "design_1_score": number,
                        "design_2_score": number,
                        "winner": "design_1|design_2|tie",
                        "reasoning": "string"
                    }}
                }},
                "strengths": {{
                    "design_1": ["string"],
                    "design_2": ["string"]
                }},
                "recommendations": {{
                    "design_1": ["string"],
                    "design_2": ["string"]
                }},
                "overall_assessment": "string"
            }}
            """
            
            comparison_result = await self.analyze_image(
                image_data_1,  # Use first image as reference
                comparison_prompt,
                max_tokens=2000
            )
            
            try:
                result = json.loads(comparison_result)
                result["analysis_type"] = "design_comparison"
                result["individual_analyses"] = {
                    "design_1": analysis_1,
                    "design_2": analysis_2
                }
                return result
            except json.JSONDecodeError:
                return {
                    "analysis_type": "design_comparison",
                    "individual_analyses": {
                        "design_1": analysis_1,
                        "design_2": analysis_2
                    },
                    "comparison_text": comparison_result
                }
                
        except Exception as e:
            logger.error(f"Error comparing designs: {e}")
            raise
    
    async def extract_text_content(self, image_data: str) -> Dict[str, Any]:
        """
        Extract and analyze text content from image
        
        Args:
            image_data: Base64 encoded image data
        
        Returns:
            Extracted text analysis
        """
        try:
            prompt = """
            Analyze this image and extract all visible text content.
            
            Provide detailed analysis:
            1. All text found (exact transcription)
            2. Text locations and hierarchy
            3. Font styles and sizes (approximate)
            4. Text readability assessment
            5. Typography quality evaluation
            
            Also identify:
            - Headlines/titles
            - Body text
            - CTAs/buttons
            - Captions/labels
            - Watermarks/signatures
            
            Respond in JSON format:
            {
                "has_text": boolean,
                "text_elements": [
                    {
                        "content": "string",
                        "type": "headline|body|cta|caption|other",
                        "location": "string",
                        "font_style": "string",
                        "readability_score": number
                    }
                ],
                "typography_analysis": {
                    "overall_quality": number,
                    "hierarchy_clarity": number,
                    "consistency": number,
                    "readability": number
                },
                "accessibility": {
                    "contrast_sufficient": boolean,
                    "font_size_adequate": boolean,
                    "issues": ["string"]
                }
            }
            """
            
            response = await self.analyze_image(image_data, prompt, max_tokens=1500)
            
            try:
                result = json.loads(response)
                result["analysis_type"] = "text_extraction"
                return result
            except json.JSONDecodeError:
                return await self._parse_text_analysis(response)
                
        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
            raise
    
    async def _analyze_openai_vision(self, 
                                   image_url: str,
                                   prompt: str, 
                                   model: str,
                                   max_tokens: int,
                                   **kwargs) -> str:
        """Analyze with OpenAI Vision"""
        try:
            client = self._clients.get(MLLMProvider.OPENAI_VISION)
            if not client:
                raise ValueError("OpenAI Vision client not initialized")
            
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error with OpenAI Vision: {e}")
            raise
    
    async def _analyze_anthropic_vision(self, 
                                      image_url: str,
                                      prompt: str, 
                                      model: str,
                                      max_tokens: int,
                                      **kwargs) -> str:
        """Analyze with Anthropic Vision"""
        try:
            # Anthropic vision implementation would go here
            logger.warning("Anthropic Vision not fully implemented")
            return "Anthropic Vision analysis placeholder"
            
        except Exception as e:
            logger.error(f"Error with Anthropic Vision: {e}")
            raise
    
    async def _analyze_google_vision(self, 
                                   image_url: str,
                                   prompt: str, 
                                   model: str,
                                   max_tokens: int,
                                   **kwargs) -> str:
        """Analyze with Google Vision"""
        try:
            # Google Vision implementation would go here
            logger.warning("Google Vision not fully implemented")
            return "Google Vision analysis placeholder"
            
        except Exception as e:
            logger.error(f"Error with Google Vision: {e}")
            raise
    
    def _resolve_model(self, model: Optional[str]) -> tuple[MLLMProvider, str]:
        """Resolve model to provider and model name"""
        if not model:
            provider = self.default_provider
            model_name = self.providers_config[provider].get("default_model", "gpt-4-vision-preview")
        else:
            provider = self.model_mappings.get(model, self.default_provider)
            model_name = model
        
        return provider, model_name
    
    def _prepare_image_for_api(self, image_data: str) -> str:
        """Prepare image data for API call"""
        if image_data.startswith('data:image'):
            return image_data
        else:
            return f"data:image/png;base64,{image_data}"
    
    async def _parse_design_analysis(self, response: str) -> Dict[str, Any]:
        """Parse design analysis from text response"""
        try:
            # Basic parsing fallback
            return {
                "analysis_type": "design_quality",
                "overall_score": 7.0,  # Default score
                "raw_analysis": response,
                "parsing_method": "fallback"
            }
        except Exception as e:
            logger.error(f"Error parsing design analysis: {e}")
            return {"error": str(e), "raw_response": response}
    
    async def _parse_elements_analysis(self, response: str) -> Dict[str, Any]:
        """Parse elements analysis from text response"""
        try:
            return {
                "analysis_type": "design_elements",
                "raw_analysis": response,
                "parsing_method": "fallback"
            }
        except Exception as e:
            logger.error(f"Error parsing elements analysis: {e}")
            return {"error": str(e), "raw_response": response}
    
    async def _parse_text_analysis(self, response: str) -> Dict[str, Any]:
        """Parse text analysis from response"""
        try:
            # Check if any text keywords are mentioned
            has_text = any(keyword in response.lower() for keyword in [
                "text", "words", "letters", "typography", "font"
            ])
            
            return {
                "analysis_type": "text_extraction",
                "has_text": has_text,
                "raw_analysis": response,
                "parsing_method": "fallback"
            }
        except Exception as e:
            logger.error(f"Error parsing text analysis: {e}")
            return {"error": str(e), "raw_response": response}
    
    async def _initialize_clients(self):
        """Initialize multimodal LLM clients"""
        try:
            import asyncio
            
            # Initialize OpenAI Vision client
            openai_config = self.providers_config[MLLMProvider.OPENAI_VISION]
            if openai_config.get("enabled", False):
                import openai
                self._clients[MLLMProvider.OPENAI_VISION] = openai.AsyncOpenAI(
                    api_key=openai_config.get("api_key")
                )
                logger.info("OpenAI Vision client initialized")
            
            # Initialize other providers...
            logger.info("Multimodal LLM interface initialized")
            
        except Exception as e:
            logger.error(f"Error initializing MLLM clients: {e}")
    
    async def batch_analyze_images(self, 
                                 image_data_list: List[str],
                                 prompt: str,
                                 model: Optional[str] = None) -> List[str]:
        """
        Analyze multiple images with same prompt
        
        Args:
            image_data_list: List of base64 image data
            prompt: Analysis prompt
            model: Model name
        
        Returns:
            List of analysis results
        """
        try:
            results = []
            
            # Process in batches to avoid rate limits
            batch_size = self.config.get("batch_size", 3)
            
            for i in range(0, len(image_data_list), batch_size):
                batch = image_data_list[i:i + batch_size]
                
                # Process batch concurrently
                import asyncio
                batch_results = await asyncio.gather(
                    *[self.analyze_image(image_data, prompt, model) for image_data in batch],
                    return_exceptions=True
                )
                
                # Handle results and exceptions
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in batch processing: {result}")
                        results.append(f"Error: {str(result)}")
                    else:
                        results.append(result)
                
                # Brief pause between batches
                if i + batch_size < len(image_data_list):
                    await asyncio.sleep(1)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch image analysis: {e}")
            return [f"Error: {str(e)}"] * len(image_data_list)
    
    async def get_model_capabilities(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get capabilities of multimodal model"""
        provider, model_name = self._resolve_model(model)
        
        capabilities = {
            MLLMProvider.OPENAI_VISION: {
                "max_image_size": "20MB",
                "supported_formats": ["PNG", "JPEG", "WEBP", "GIF"],
                "max_tokens": 4096,
                "supports_multiple_images": False,
                "supports_streaming": False,
                "quality": "high"
            },
            MLLMProvider.ANTHROPIC_VISION: {
                "max_image_size": "5MB",
                "supported_formats": ["PNG", "JPEG", "WEBP"],
                "max_tokens": 4096,
                "supports_multiple_images": True,
                "supports_streaming": False,
                "quality": "high"
            }
        }
        
        return capabilities.get(provider, {
            "max_image_size": "Unknown",
            "supported_formats": ["PNG", "JPEG"],
            "max_tokens": 1000,
            "supports_multiple_images": False,
            "supports_streaming": False,
            "quality": "medium"
        })
