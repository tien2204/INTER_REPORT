"""
OpenAI Model Manager

Manages OpenAI models including GPT and DALL-E,
handles authentication, rate limiting, and optimization.
"""

import asyncio
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
from datetime import datetime, timedelta
import time
from structlog import get_logger

logger = get_logger(__name__)


class OpenAIManager:
    """
    Manager for OpenAI models and API interactions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # API configuration
        self.api_key = config.get("api_key", "")
        self.organization = config.get("organization", "")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        
        # Rate limiting
        self.rate_limits = config.get("rate_limits", {
            "requests_per_minute": 60,
            "tokens_per_minute": 150000,
            "requests_per_day": 1000
        })
        
        # Model configurations
        self.model_configs = {
            "gpt-4": {
                "max_tokens": 8192,
                "context_window": 8192,
                "cost_per_1k_input": 0.03,
                "cost_per_1k_output": 0.06,
                "supports_streaming": True,
                "supports_functions": True
            },
            "gpt-4-turbo": {
                "max_tokens": 4096,
                "context_window": 128000,
                "cost_per_1k_input": 0.01,
                "cost_per_1k_output": 0.03,
                "supports_streaming": True,
                "supports_functions": True
            },
            "gpt-3.5-turbo": {
                "max_tokens": 4096,
                "context_window": 16385,
                "cost_per_1k_input": 0.0015,
                "cost_per_1k_output": 0.002,
                "supports_streaming": True,
                "supports_functions": True
            },
            "gpt-4-vision-preview": {
                "max_tokens": 4096,
                "context_window": 128000,
                "cost_per_1k_input": 0.01,
                "cost_per_1k_output": 0.03,
                "supports_streaming": False,
                "supports_images": True
            },
            "dall-e-3": {
                "supported_sizes": ["1024x1024", "1024x1792", "1792x1024"],
                "cost_per_image": 0.04,
                "supports_style": True,
                "supports_quality": True
            }
        }
        
        # Initialize client
        self._client = None
        self._rate_limiter = RateLimiter(self.rate_limits)
        
        # Usage tracking
        self._usage_stats = {
            "requests_today": 0,
            "tokens_used": 0,
            "cost_estimate": 0.0,
            "last_reset": datetime.now().date()
        }
        
        asyncio.create_task(self._initialize())
    
    async def generate_text(self, 
                           messages: List[Dict[str, str]],
                           model: str = "gpt-3.5-turbo",
                           max_tokens: Optional[int] = None,
                           temperature: float = 0.7,
                           stream: bool = False,
                           **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text using OpenAI LLM
        
        Args:
            messages: Conversation messages
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response
            **kwargs: Additional parameters
        
        Returns:
            Generated text or async generator for streaming
        """
        try:
            # Validate model
            if model not in self.model_configs:
                raise ValueError(f"Unsupported model: {model}")
            
            model_config = self.model_configs[model]
            
            # Set max_tokens if not provided
            if max_tokens is None:
                max_tokens = min(1000, model_config["max_tokens"])
            
            # Check rate limits
            await self._rate_limiter.wait_if_needed()
            
            # Prepare request
            request_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
                **kwargs
            }
            
            # Track usage
            input_tokens = self._estimate_tokens(messages)
            self._update_usage_stats(input_tokens, max_tokens, model)
            
            if stream:
                return self._stream_chat_completion(request_params)
            else:
                response = await self._client.chat.completions.create(**request_params)
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            raise
    
    async def generate_image(self, 
                           prompt: str,
                           model: str = "dall-e-3",
                           size: str = "1024x1024",
                           quality: str = "standard",
                           style: str = "vivid",
                           **kwargs) -> str:
        """
        Generate image using DALL-E
        
        Args:
            prompt: Image generation prompt
            model: DALL-E model name
            size: Image size
            quality: Image quality
            style: Image style
            **kwargs: Additional parameters
        
        Returns:
            Base64 encoded image data
        """
        try:
            # Validate parameters
            if model not in self.model_configs or "cost_per_image" not in self.model_configs[model]:
                raise ValueError(f"Unsupported image model: {model}")
            
            model_config = self.model_configs[model]
            
            if size not in model_config["supported_sizes"]:
                logger.warning(f"Size {size} not supported, using 1024x1024")
                size = "1024x1024"
            
            # Check rate limits
            await self._rate_limiter.wait_if_needed()
            
            # Generate image
            response = await self._client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                response_format="b64_json",
                n=1,
                **kwargs
            )
            
            # Track usage
            self._update_image_usage_stats(model)
            
            image_data = response.data[0].b64_json
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error generating image with OpenAI: {e}")
            raise
    
    async def analyze_image(self, 
                          image_data: str,
                          prompt: str,
                          model: str = "gpt-4-vision-preview",
                          max_tokens: int = 1000,
                          **kwargs) -> str:
        """
        Analyze image using GPT-4 Vision
        
        Args:
            image_data: Base64 encoded image or URL
            prompt: Analysis prompt
            model: Vision model name
            max_tokens: Maximum response tokens
            **kwargs: Additional parameters
        
        Returns:
            Analysis result
        """
        try:
            # Validate model
            if model not in self.model_configs or not self.model_configs[model].get("supports_images"):
                raise ValueError(f"Model {model} doesn't support image analysis")
            
            # Check rate limits
            await self._rate_limiter.wait_if_needed()
            
            # Prepare image URL
            if not image_data.startswith('http'):
                image_url = f"data:image/jpeg;base64,{image_data.split(',')[-1]}"
            else:
                image_url = image_data
            
            # Analyze image
            response = await self._client.chat.completions.create(
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
            
            # Track usage
            input_tokens = self._estimate_tokens([{"role": "user", "content": prompt}])
            self._update_usage_stats(input_tokens, max_tokens, model)
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error analyzing image with OpenAI: {e}")
            raise
    
    async def _stream_chat_completion(self, request_params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream chat completion response"""
        try:
            stream = await self._client.chat.completions.create(**request_params)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error streaming chat completion: {e}")
            raise
    
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count for messages"""
        try:
            # Rough estimation: 1 token ≈ 0.75 words
            total_chars = sum(len(msg.get("content", "")) for msg in messages)
            estimated_tokens = int(total_chars / 3)  # Rough approximation
            return max(1, estimated_tokens)
        except Exception:
            return 100  # Fallback estimate
    
    def _update_usage_stats(self, input_tokens: int, max_output_tokens: int, model: str):
        """Update usage statistics"""
        try:
            # Reset daily stats if needed
            today = datetime.now().date()
            if self._usage_stats["last_reset"] != today:
                self._usage_stats["requests_today"] = 0
                self._usage_stats["tokens_used"] = 0
                self._usage_stats["cost_estimate"] = 0.0
                self._usage_stats["last_reset"] = today
            
            # Update stats
            self._usage_stats["requests_today"] += 1
            self._usage_stats["tokens_used"] += input_tokens + max_output_tokens
            
            # Estimate cost
            model_config = self.model_configs.get(model, {})
            input_cost = (input_tokens / 1000) * model_config.get("cost_per_1k_input", 0.001)
            output_cost = (max_output_tokens / 1000) * model_config.get("cost_per_1k_output", 0.002)
            self._usage_stats["cost_estimate"] += input_cost + output_cost
            
        except Exception as e:
            logger.error(f"Error updating usage stats: {e}")
    
    def _update_image_usage_stats(self, model: str):
        """Update image generation usage stats"""
        try:
            today = datetime.now().date()
            if self._usage_stats["last_reset"] != today:
                self._usage_stats["requests_today"] = 0
                self._usage_stats["cost_estimate"] = 0.0
                self._usage_stats["last_reset"] = today
            
            self._usage_stats["requests_today"] += 1
            
            model_config = self.model_configs.get(model, {})
            image_cost = model_config.get("cost_per_image", 0.04)
            self._usage_stats["cost_estimate"] += image_cost
            
        except Exception as e:
            logger.error(f"Error updating image usage stats: {e}")
    
    async def _initialize(self):
        """Initialize OpenAI client"""
        try:
            if not self.api_key:
                logger.warning("OpenAI API key not provided")
                return
            
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization if self.organization else None,
                base_url=self.base_url
            )
            
            # Test connection
            await self._test_connection()
            
            logger.info("OpenAI manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI manager: {e}")
    
    async def _test_connection(self):
        """Test OpenAI connection"""
        try:
            # Simple test request
            response = await self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            logger.info("OpenAI connection test successful")
            
        except Exception as e:
            logger.warning(f"OpenAI connection test failed: {e}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return self._usage_stats.copy()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.model_configs.keys())
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about specific model"""
        return self.model_configs.get(model, {})
    
    async def estimate_cost(self, 
                          messages: List[Dict[str, str]],
                          model: str,
                          max_tokens: int = 1000) -> Dict[str, Any]:
        """Estimate cost for a request"""
        try:
            input_tokens = self._estimate_tokens(messages)
            model_config = self.model_configs.get(model, {})
            
            input_cost = (input_tokens / 1000) * model_config.get("cost_per_1k_input", 0.001)
            output_cost = (max_tokens / 1000) * model_config.get("cost_per_1k_output", 0.002)
            total_cost = input_cost + output_cost
            
            return {
                "model": model,
                "input_tokens": input_tokens,
                "max_output_tokens": max_tokens,
                "input_cost_usd": round(input_cost, 4),
                "output_cost_usd": round(output_cost, 4),
                "total_cost_usd": round(total_cost, 4)
            }
            
        except Exception as e:
            logger.error(f"Error estimating cost: {e}")
            return {"error": str(e)}


class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self, limits: Dict[str, int]):
        self.requests_per_minute = limits.get("requests_per_minute", 60)
        self.tokens_per_minute = limits.get("tokens_per_minute", 150000)
        
        self.request_times = []
        self.token_usage = []
        
    async def wait_if_needed(self, tokens: int = 0):
        """Wait if rate limit would be exceeded"""
        try:
            now = time.time()
            minute_ago = now - 60
            
            # Clean old entries
            self.request_times = [t for t in self.request_times if t > minute_ago]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > minute_ago]
            
            # Check request rate
            if len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
            
            # Check token rate
            total_tokens = sum(tokens for _, tokens in self.token_usage)
            if total_tokens + tokens > self.tokens_per_minute:
                # Calculate sleep time based on oldest token usage
                if self.token_usage:
                    sleep_time = 60 - (now - self.token_usage[0][0])
                    if sleep_time > 0:
                        logger.info(f"Token rate limit reached, sleeping for {sleep_time:.1f}s")
                        await asyncio.sleep(sleep_time)
            
            # Record this request
            self.request_times.append(now)
            if tokens > 0:
                self.token_usage.append((now, tokens))
                
        except Exception as e:
            logger.error(f"Error in rate limiter: {e}")
            # Small delay as fallback
            await asyncio.sleep(0.1)
