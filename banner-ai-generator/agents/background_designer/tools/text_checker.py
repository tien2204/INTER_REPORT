"""
Text Checker Tool

Uses Multimodal LLM to detect text elements in generated backgrounds
to ensure text-free requirements are met.
"""

import base64
import io
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import json
from structlog import get_logger

logger = get_logger(__name__)


class TextChecker:
    """
    Detects text elements in images using multimodal AI models
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Model configuration
        self.primary_model = config.get("primary_model", "gpt-4-vision-preview")
        self.fallback_model = config.get("fallback_model", "claude-3-vision")
        
        # API configurations
        self.openai_config = config.get("openai_config", {})
        self.anthropic_config = config.get("anthropic_config", {})
        
        # Detection thresholds
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.max_retries = config.get("max_retries", 2)
        
        # Initialize clients
        self._openai_client = None
        self._anthropic_client = None
        self._initialize_clients()
    
    async def detect_text(self, image_data: str) -> bool:
        """
        Detect if image contains any text elements
        
        Args:
            image_data: Base64 encoded image data
        
        Returns:
            True if text is detected, False otherwise
        """
        try:
            # Try primary model first
            try:
                result = await self._detect_with_primary_model(image_data)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Primary model failed: {e}")
            
            # Fallback to secondary model
            try:
                result = await self._detect_with_fallback_model(image_data)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Fallback model failed: {e}")
            
            # If both models fail, use OCR fallback
            return await self._detect_with_ocr_fallback(image_data)
            
        except Exception as e:
            logger.error(f"Error detecting text: {e}")
            # Conservative approach - assume text is present if detection fails
            return True
    
    async def analyze_text_elements(self, image_data: str) -> Dict[str, Any]:
        """
        Detailed analysis of text elements in image
        
        Args:
            image_data: Base64 encoded image data
        
        Returns:
            Detailed analysis result
        """
        try:
            # Prepare analysis prompt
            analysis_prompt = self._build_analysis_prompt()
            
            # Try with primary model
            try:
                result = await self._analyze_with_model(image_data, analysis_prompt, "primary")
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Primary model analysis failed: {e}")
            
            # Fallback analysis
            try:
                result = await self._analyze_with_model(image_data, analysis_prompt, "fallback")
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Fallback model analysis failed: {e}")
            
            # Basic OCR analysis
            return await self._basic_ocr_analysis(image_data)
            
        except Exception as e:
            logger.error(f"Error analyzing text elements: {e}")
            return {
                "has_text": True,
                "confidence": 0.5,
                "error": str(e)
            }
    
    async def _detect_with_primary_model(self, image_data: str) -> Optional[bool]:
        """Detect text using primary model (GPT-4 Vision)"""
        try:
            if not self._openai_client:
                return None
            
            # Prepare the image
            image_url = self._prepare_image_for_api(image_data)
            
            # Detection prompt
            prompt = """
            Analyze this image and determine if it contains any text elements.
            
            Look for:
            - Readable text or letters
            - Numbers or symbols that could be text
            - Logos with text
            - Watermarks
            - Any typography elements
            
            Respond with only "YES" if text is detected or "NO" if no text is found.
            Be very thorough - even small or partially visible text should be reported as YES.
            """
            
            response = await self._openai_client.chat.completions.create(
                model=self.primary_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=10
            )
            
            result_text = response.choices[0].message.content.strip().upper()
            
            if "YES" in result_text:
                return True
            elif "NO" in result_text:
                return False
            else:
                logger.warning(f"Unclear response from primary model: {result_text}")
                return None
                
        except Exception as e:
            logger.error(f"Error with primary model text detection: {e}")
            return None
    
    async def _detect_with_fallback_model(self, image_data: str) -> Optional[bool]:
        """Detect text using fallback model (Claude Vision)"""
        try:
            if not self._anthropic_client:
                return None
            
            # For Claude, we need to handle image differently
            # This is a simplified implementation
            prompt = """
            Please analyze this image and tell me if it contains any visible text, letters, numbers, or typography elements.
            
            Answer with just "YES" if any text is visible, or "NO" if the image is completely text-free.
            """
            
            # Claude API call would go here
            # This is a placeholder since Claude API might have different image handling
            
            logger.warning("Claude Vision integration not fully implemented")
            return None
            
        except Exception as e:
            logger.error(f"Error with fallback model text detection: {e}")
            return None
    
    async def _detect_with_ocr_fallback(self, image_data: str) -> bool:
        """Fallback OCR-based text detection"""
        try:
            # Try with pytesseract if available
            try:
                import pytesseract
                from PIL import Image
                
                # Decode image
                image = self._decode_image(image_data)
                if not image:
                    return True  # Conservative assumption
                
                # Extract text
                extracted_text = pytesseract.image_to_string(image).strip()
                
                # Check if meaningful text was found
                if len(extracted_text) > 3:  # More than just noise
                    return True
                
                return False
                
            except ImportError:
                logger.warning("pytesseract not available, using basic detection")
                return await self._basic_text_detection(image_data)
                
        except Exception as e:
            logger.error(f"Error with OCR fallback: {e}")
            return True  # Conservative assumption
    
    async def _basic_text_detection(self, image_data: str) -> bool:
        """Very basic text detection using image analysis"""
        try:
            image = self._decode_image(image_data)
            if not image:
                return True
            
            import numpy as np
            
            # Convert to grayscale
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Look for text-like patterns (high contrast regions)
            # This is very basic and not reliable
            
            # Edge detection
            edges = np.abs(np.diff(img_array, axis=0)).sum() + np.abs(np.diff(img_array, axis=1)).sum()
            edge_density = edges / (img_array.shape[0] * img_array.shape[1])
            
            # High edge density might indicate text
            if edge_density > 50:  # Threshold for potential text
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in basic text detection: {e}")
            return True
    
    async def _analyze_with_model(self, 
                                 image_data: str, 
                                 prompt: str, 
                                 model_type: str) -> Optional[Dict[str, Any]]:
        """Analyze image with specified model"""
        try:
            if model_type == "primary" and self._openai_client:
                return await self._analyze_with_openai(image_data, prompt)
            elif model_type == "fallback" and self._anthropic_client:
                return await self._analyze_with_anthropic(image_data, prompt)
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing with {model_type} model: {e}")
            return None
    
    async def _analyze_with_openai(self, image_data: str, prompt: str) -> Dict[str, Any]:
        """Detailed analysis with OpenAI GPT-4 Vision"""
        try:
            image_url = self._prepare_image_for_api(image_data)
            
            response = await self._openai_client.chat.completions.create(
                model=self.primary_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            # Parse the response
            return self._parse_analysis_response(result_text)
            
        except Exception as e:
            logger.error(f"Error with OpenAI analysis: {e}")
            return None
    
    async def _analyze_with_anthropic(self, image_data: str, prompt: str) -> Dict[str, Any]:
        """Detailed analysis with Claude Vision"""
        try:
            # Placeholder for Claude implementation
            logger.warning("Claude analysis not implemented")
            return None
            
        except Exception as e:
            logger.error(f"Error with Anthropic analysis: {e}")
            return None
    
    async def _basic_ocr_analysis(self, image_data: str) -> Dict[str, Any]:
        """Basic OCR-based analysis"""
        try:
            has_text = await self._detect_with_ocr_fallback(image_data)
            
            return {
                "has_text": has_text,
                "confidence": 0.6,  # Lower confidence for OCR
                "method": "ocr_fallback",
                "details": "Basic OCR detection used"
            }
            
        except Exception as e:
            logger.error(f"Error in basic OCR analysis: {e}")
            return {
                "has_text": True,
                "confidence": 0.5,
                "error": str(e)
            }
    
    def _build_analysis_prompt(self) -> str:
        """Build detailed analysis prompt"""
        return """
        Please provide a detailed analysis of any text elements in this image.

        Analyze the following:
        1. Is there any visible text, letters, or numbers?
        2. Are there any logos with text components?
        3. Are there any watermarks or signatures?
        4. How confident are you in your assessment (scale 1-10)?
        5. Describe the location of any text found

        Please respond in JSON format:
        {
            "has_text": true/false,
            "confidence": 1-10,
            "text_elements": ["description of any text found"],
            "locations": ["where text appears"],
            "severity": "none/minor/major"
        }
        """
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse model response into structured format"""
        try:
            # Try to extract JSON from response
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validate and normalize
                return {
                    "has_text": bool(result.get("has_text", False)),
                    "confidence": float(result.get("confidence", 5)) / 10.0,
                    "text_elements": result.get("text_elements", []),
                    "locations": result.get("locations", []),
                    "severity": result.get("severity", "unknown"),
                    "raw_response": response_text
                }
            
            # Fallback parsing
            has_text = any(word in response_text.lower() for word in ["yes", "true", "found", "visible"])
            
            return {
                "has_text": has_text,
                "confidence": 0.7,
                "raw_response": response_text,
                "parsing_method": "fallback"
            }
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}")
            return {
                "has_text": True,  # Conservative assumption
                "confidence": 0.5,
                "error": str(e),
                "raw_response": response_text
            }
    
    def _prepare_image_for_api(self, image_data: str) -> str:
        """Prepare image data for API call"""
        if image_data.startswith('data:image'):
            return image_data
        else:
            return f"data:image/png;base64,{image_data}"
    
    def _decode_image(self, image_data: str) -> Optional[Image.Image]:
        """Decode base64 image to PIL Image"""
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None
    
    def _initialize_clients(self):
        """Initialize API clients"""
        try:
            # Initialize OpenAI client
            openai_api_key = self.openai_config.get("api_key")
            if openai_api_key:
                import openai
                self._openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized for text detection")
            
            # Initialize Anthropic client
            anthropic_api_key = self.anthropic_config.get("api_key")
            if anthropic_api_key:
                # Anthropic client initialization would go here
                logger.info("Anthropic client configuration found")
            
        except Exception as e:
            logger.error(f"Error initializing text checker clients: {e}")
    
    async def batch_detect_text(self, image_data_list: List[str]) -> List[bool]:
        """Detect text in multiple images efficiently"""
        try:
            results = []
            
            # Process in batches to avoid rate limits
            batch_size = self.config.get("batch_size", 5)
            
            for i in range(0, len(image_data_list), batch_size):
                batch = image_data_list[i:i + batch_size]
                
                # Process batch concurrently
                import asyncio
                batch_results = await asyncio.gather(
                    *[self.detect_text(image_data) for image_data in batch],
                    return_exceptions=True
                )
                
                # Handle results and exceptions
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in batch processing: {result}")
                        results.append(True)  # Conservative assumption
                    else:
                        results.append(result)
                
                # Brief pause between batches
                if i + batch_size < len(image_data_list):
                    await asyncio.sleep(0.5)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch text detection: {e}")
            return [True] * len(image_data_list)  # Conservative fallback
