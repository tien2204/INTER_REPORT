"""
Text-to-Image Interface

Unified interface for text-to-image generation models including
FLUX.1-schnell, DALL-E, and other T2I models.
"""

import asyncio
import base64
import io
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from PIL import Image
import requests
from structlog import get_logger

logger = get_logger(__name__)


class T2IProvider(Enum):
    """Text-to-image model providers"""
    FLUX = "flux"
    OPENAI_DALLE = "openai_dalle"
    STABILITY_AI = "stability_ai"
    LOCAL = "local"


@dataclass
class T2IRequest:
    """Text-to-image generation request"""
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    seed: Optional[int] = None
    num_images: int = 1
    style: Optional[str] = None
    quality: str = "standard"  # standard, hd


@dataclass
class T2IResponse:
    """Text-to-image generation response"""
    success: bool
    images: List[str]  # Base64 encoded images
    metadata: Dict[str, Any]
    provider: str
    processing_time: float
    error_message: Optional[str] = None


class TextToImageInterface:
    """
    Unified text-to-image generation interface
    
    Features:
    - Multi-provider support (FLUX, DALL-E, Stability AI)
    - Automatic quality optimization
    - Batch generation support
    - Style transfer capabilities
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.default_provider = T2IProvider(config.get("default_provider", "flux"))
        
        # Provider configurations
        self.providers_config = {
            T2IProvider.FLUX: config.get("flux_config", {}),
            T2IProvider.OPENAI_DALLE: config.get("openai_config", {}),
            T2IProvider.STABILITY_AI: config.get("stability_config", {}),
            T2IProvider.LOCAL: config.get("local_config", {})
        }
        
        # Generation settings
        self.max_retries = config.get("max_retries", 3)
        self.timeout_seconds = config.get("timeout_seconds", 120)
        self.quality_threshold = config.get("quality_threshold", 0.7)
        
        # Performance tracking
        self.generation_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_processing_time": 0.0,
            "provider_usage": {provider.value: 0 for provider in T2IProvider}
        }
        
        logger.info("Text-to-Image Interface initialized")
    
    async def generate_image(self, request: T2IRequest, 
                           provider: Optional[T2IProvider] = None) -> T2IResponse:
        """
        Generate image from text prompt
        
        Args:
            request: T2I generation request
            provider: Specific provider to use (optional)
            
        Returns:
            T2IResponse with generated images
        """
        start_time = datetime.utcnow()
        
        try:
            # Use specified provider or default
            selected_provider = provider or self.default_provider
            
            # Validate provider configuration
            if not self._is_provider_available(selected_provider):
                # Fallback to available provider
                selected_provider = self._get_available_provider()
                if not selected_provider:
                    raise Exception("No available T2I providers configured")
            
            # Generate image based on provider
            response = await self._generate_with_provider(request, selected_provider)
            
            # Update statistics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_stats(selected_provider, processing_time, response.success)
            
            return response
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_stats(provider, processing_time, False)
            
            logger.error(f"T2I generation failed: {e}")
            return T2IResponse(
                success=False,
                images=[],
                metadata={},
                provider=provider.value if provider else "unknown",
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def generate_batch(self, requests: List[T2IRequest],
                           provider: Optional[T2IProvider] = None) -> List[T2IResponse]:
        """
        Generate multiple images in batch
        
        Args:
            requests: List of T2I requests
            provider: Provider to use for all requests
            
        Returns:
            List of T2I responses
        """
        try:
            # Process requests concurrently
            tasks = [
                self.generate_image(request, provider)
                for request in requests
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            results = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    results.append(T2IResponse(
                        success=False,
                        images=[],
                        metadata={},
                        provider=provider.value if provider else "unknown",
                        processing_time=0.0,
                        error_message=str(response)
                    ))
                else:
                    results.append(response)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch T2I generation failed: {e}")
            return [
                T2IResponse(
                    success=False,
                    images=[],
                    metadata={},
                    provider="unknown",
                    processing_time=0.0,
                    error_message=str(e)
                )
                for _ in requests
            ]
    
    async def optimize_prompt(self, prompt: str, style: Optional[str] = None) -> str:
        """
        Optimize prompt for better image generation
        
        Args:
            prompt: Original prompt
            style: Target style (optional)
            
        Returns:
            Optimized prompt
        """
        try:
            # Basic prompt optimization
            optimized = prompt.strip()
            
            # Add style modifiers
            if style:
                style_modifiers = {
                    "photorealistic": "photorealistic, highly detailed, professional photography",
                    "artistic": "artistic, creative, beautiful composition",
                    "minimalist": "minimalist, clean, simple, elegant",
                    "modern": "modern, contemporary, sleek design",
                    "vintage": "vintage, retro, classic style",
                    "abstract": "abstract, artistic interpretation, creative"
                }
                
                if style.lower() in style_modifiers:
                    optimized += f", {style_modifiers[style.lower()]}"
            
            # Add quality modifiers
            optimized += ", high quality, detailed, sharp focus"
            
            return optimized
            
        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            return prompt
    
    async def validate_image_quality(self, image_data: str) -> float:
        """
        Validate generated image quality
        
        Args:
            image_data: Base64 encoded image
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            # Decode image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Basic quality checks
            width, height = image.size
            
            # Resolution check
            resolution_score = min(width * height / (1024 * 1024), 1.0)
            
            # Aspect ratio check
            aspect_ratio = max(width, height) / min(width, height)
            aspect_score = max(0.0, 1.0 - (aspect_ratio - 1.0) / 2.0)
            
            # File size check (proxy for compression quality)
            size_score = min(len(image_bytes) / (500 * 1024), 1.0)  # 500KB baseline
            
            # Combined quality score
            quality_score = (resolution_score * 0.5 + aspect_score * 0.3 + size_score * 0.2)
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Image quality validation failed: {e}")
            return 0.5  # Neutral score on error
    
    async def get_available_providers(self) -> List[T2IProvider]:
        """Get list of available providers"""
        available = []
        
        for provider in T2IProvider:
            if self._is_provider_available(provider):
                available.append(provider)
        
        return available
    
    async def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        stats = self.generation_stats.copy()
        
        # Calculate success rate
        total = stats["total_requests"]
        if total > 0:
            stats["success_rate"] = (stats["successful_generations"] / total) * 100
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    # Provider-specific generation methods
    
    async def _generate_with_provider(self, request: T2IRequest, 
                                    provider: T2IProvider) -> T2IResponse:
        """Generate with specific provider"""
        
        if provider == T2IProvider.FLUX:
            return await self._generate_with_flux(request)
        elif provider == T2IProvider.OPENAI_DALLE:
            return await self._generate_with_dalle(request)
        elif provider == T2IProvider.STABILITY_AI:
            return await self._generate_with_stability(request)
        elif provider == T2IProvider.LOCAL:
            return await self._generate_with_local(request)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _generate_with_flux(self, request: T2IRequest) -> T2IResponse:
        """Generate with FLUX model"""
        try:
            config = self.providers_config[T2IProvider.FLUX]
            
            if config.get("use_local", False):
                # Local FLUX generation
                return await self._generate_flux_local(request, config)
            else:
                # API-based FLUX generation
                return await self._generate_flux_api(request, config)
                
        except Exception as e:
            logger.error(f"FLUX generation failed: {e}")
            raise
    
    async def _generate_flux_api(self, request: T2IRequest, config: Dict[str, Any]) -> T2IResponse:
        """Generate with FLUX API"""
        try:
            api_url = config.get("api_url", "")
            api_key = config.get("api_key", "")
            
            if not api_url:
                raise ValueError("FLUX API URL not configured")
            
            # Prepare request payload
            payload = {
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "num_images": request.num_images
            }
            
            if request.negative_prompt:
                payload["negative_prompt"] = request.negative_prompt
            
            if request.seed:
                payload["seed"] = request.seed
            
            # Make API request
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            async with asyncio.timeout(self.timeout_seconds):
                response = requests.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                
                result = response.json()
                
                return T2IResponse(
                    success=True,
                    images=result.get("images", []),
                    metadata=result.get("metadata", {}),
                    provider="flux",
                    processing_time=result.get("processing_time", 0.0)
                )
                
        except Exception as e:
            logger.error(f"FLUX API generation failed: {e}")
            raise
    
    async def _generate_flux_local(self, request: T2IRequest, config: Dict[str, Any]) -> T2IResponse:
        """Generate with local FLUX model"""
        try:
            # Placeholder for local FLUX implementation
            # This would require torch, diffusers, and the FLUX model
            
            logger.warning("Local FLUX generation not implemented")
            raise NotImplementedError("Local FLUX generation not yet implemented")
            
        except Exception as e:
            logger.error(f"Local FLUX generation failed: {e}")
            raise
    
    async def _generate_with_dalle(self, request: T2IRequest) -> T2IResponse:
        """Generate with DALL-E"""
        try:
            config = self.providers_config[T2IProvider.OPENAI_DALLE]
            api_key = config.get("api_key", "")
            
            if not api_key:
                raise ValueError("OpenAI API key not configured")
            
            # Use OpenAI client
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)
            
            # Determine model and size
            model = "dall-e-3"
            size = f"{request.width}x{request.height}"
            
            # Validate size for DALL-E
            valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
            if size not in valid_sizes:
                size = "1024x1024"  # Default fallback
            
            # Generate image
            response = await client.images.generate(
                model=model,
                prompt=request.prompt,
                size=size,
                quality=request.quality,
                n=request.num_images
            )
            
            # Convert to base64
            images = []
            for image in response.data:
                if hasattr(image, 'b64_json') and image.b64_json:
                    images.append(image.b64_json)
                elif hasattr(image, 'url') and image.url:
                    # Download and convert URL to base64
                    img_response = requests.get(image.url)
                    img_b64 = base64.b64encode(img_response.content).decode()
                    images.append(img_b64)
            
            return T2IResponse(
                success=True,
                images=images,
                metadata={"model": model, "size": size},
                provider="openai_dalle",
                processing_time=0.0  # OpenAI doesn't provide timing
            )
            
        except Exception as e:
            logger.error(f"DALL-E generation failed: {e}")
            raise
    
    async def _generate_with_stability(self, request: T2IRequest) -> T2IResponse:
        """Generate with Stability AI"""
        try:
            # Placeholder for Stability AI implementation
            logger.warning("Stability AI generation not implemented")
            raise NotImplementedError("Stability AI generation not yet implemented")
            
        except Exception as e:
            logger.error(f"Stability AI generation failed: {e}")
            raise
    
    async def _generate_with_local(self, request: T2IRequest) -> T2IResponse:
        """Generate with local model"""
        try:
            # Placeholder for local model implementation
            logger.warning("Local model generation not implemented")
            raise NotImplementedError("Local model generation not yet implemented")
            
        except Exception as e:
            logger.error(f"Local model generation failed: {e}")
            raise
    
    # Utility methods
    
    def _is_provider_available(self, provider: T2IProvider) -> bool:
        """Check if provider is available and configured"""
        config = self.providers_config.get(provider, {})
        
        if not config.get("enabled", False):
            return False
        
        if provider == T2IProvider.FLUX:
            return bool(config.get("api_url") or config.get("use_local", False))
        elif provider == T2IProvider.OPENAI_DALLE:
            return bool(config.get("api_key"))
        elif provider == T2IProvider.STABILITY_AI:
            return bool(config.get("api_key"))
        elif provider == T2IProvider.LOCAL:
            return bool(config.get("model_path"))
        
        return False
    
    def _get_available_provider(self) -> Optional[T2IProvider]:
        """Get first available provider"""
        for provider in T2IProvider:
            if self._is_provider_available(provider):
                return provider
        return None
    
    async def _update_stats(self, provider: Optional[T2IProvider], 
                          processing_time: float, success: bool):
        """Update generation statistics"""
        try:
            self.generation_stats["total_requests"] += 1
            
            if success:
                self.generation_stats["successful_generations"] += 1
            else:
                self.generation_stats["failed_generations"] += 1
            
            if provider:
                self.generation_stats["provider_usage"][provider.value] += 1
            
            # Update average processing time
            total_requests = self.generation_stats["total_requests"]
            current_avg = self.generation_stats["avg_processing_time"]
            self.generation_stats["avg_processing_time"] = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )
            
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")


