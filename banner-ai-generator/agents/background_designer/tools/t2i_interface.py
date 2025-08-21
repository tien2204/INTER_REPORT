"""
Text-to-Image Interface

Interface for Text-to-Image models including FLUX.1-schnell
and other diffusion models for background generation.
"""

import asyncio
import base64
import io
from typing import Any, Dict, List, Optional, Union
import json
from structlog import get_logger

logger = get_logger(__name__)


class TextToImageInterface:
    """
    Interface for Text-to-Image model integration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Model configuration
        self.default_model = config.get("default_model", "flux.1-schnell")
        self.models_config = config.get("models", {})
        
        # API configurations
        self.api_configs = {
            "flux": config.get("flux_config", {}),
            "openai": config.get("openai_config", {}),
            "local": config.get("local_config", {})
        }
        
        # Generation parameters
        self.default_params = {
            "width": 1024,
            "height": 1024,
            "guidance_scale": 7.5,
            "num_inference_steps": 25,
            "num_images": 1,
            "safety_check": True
        }
        
        # Initialize model clients
        self._clients = {}
        asyncio.create_task(self._initialize_clients())
    
    async def generate_image(self, 
                           prompt: str,
                           width: int = 1024,
                           height: int = 1024,
                           guidance_scale: float = 7.5,
                           num_inference_steps: int = 25,
                           model: Optional[str] = None,
                           **kwargs) -> str:
        """
        Generate image using Text-to-Image model
        
        Args:
            prompt: Text prompt for image generation
            width: Image width in pixels
            height: Image height in pixels
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            model: Specific model to use (optional)
            **kwargs: Additional model-specific parameters
        
        Returns:
            Base64 encoded image data
        """
        try:
            # Validate parameters
            width, height = self._validate_dimensions(width, height)
            guidance_scale = max(1.0, min(20.0, guidance_scale))
            num_inference_steps = max(10, min(100, num_inference_steps))
            
            # Select model
            model_name = model or self.default_model
            
            # Prepare generation parameters
            params = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                **kwargs
            }
            
            logger.info(f"Generating image with {model_name}: {width}x{height}, prompt: {prompt[:50]}...")
            
            # Generate based on model type
            if model_name.startswith("flux"):
                image_data = await self._generate_with_flux(params)
            elif model_name.startswith("openai"):
                image_data = await self._generate_with_openai(params)
            elif model_name == "local":
                image_data = await self._generate_with_local_model(params)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # Validate generated image
            if not image_data:
                raise RuntimeError("No image data generated")
            
            logger.info(f"Image generated successfully with {model_name}")
            return image_data
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            # Return placeholder image on error
            return await self._generate_placeholder_image(width, height)
    
    async def _generate_with_flux(self, params: Dict[str, Any]) -> str:
        """Generate image using FLUX.1-schnell model"""
        try:
            flux_config = self.api_configs.get("flux", {})
            
            # FLUX.1-schnell specific optimizations
            if "flux.1-schnell" in self.default_model:
                # Schnell model works best with fewer steps
                params["num_inference_steps"] = min(params.get("num_inference_steps", 25), 8)
                params["guidance_scale"] = min(params.get("guidance_scale", 7.5), 3.5)
            
            # Check if we have local FLUX implementation
            if flux_config.get("use_local", False):
                return await self._generate_flux_local(params)
            else:
                return await self._generate_flux_api(params)
                
        except Exception as e:
            logger.error(f"Error generating with FLUX: {e}")
            raise
    
    async def _generate_flux_local(self, params: Dict[str, Any]) -> str:
        """Generate using local FLUX implementation"""
        try:
            # This would integrate with local FLUX installation
            # For now, simulate the generation
            
            import torch
            from diffusers import FluxPipeline
            
            # Load model (cache for subsequent uses)
            if "flux_pipeline" not in self._clients:
                logger.info("Loading FLUX.1-schnell pipeline...")
                self._clients["flux_pipeline"] = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    torch_dtype=torch.float16
                ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            pipeline = self._clients["flux_pipeline"]
            
            # Generate image
            result = pipeline(
                prompt=params["prompt"],
                width=params["width"],
                height=params["height"],
                guidance_scale=params["guidance_scale"],
                num_inference_steps=params["num_inference_steps"],
                generator=torch.manual_seed(42)  # For reproducibility
            )
            
            # Convert to base64
            image = result.images[0]
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            
            image_data = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{image_data}"
            
        except ImportError:
            logger.warning("FLUX dependencies not available, falling back to API")
            return await self._generate_flux_api(params)
        except Exception as e:
            logger.error(f"Error with local FLUX generation: {e}")
            raise
    
    async def _generate_flux_api(self, params: Dict[str, Any]) -> str:
        """Generate using FLUX API"""
        try:
            flux_config = self.api_configs.get("flux", {})
            api_url = flux_config.get("api_url", "")
            api_key = flux_config.get("api_key", "")
            
            if not api_url or not api_key:
                raise ValueError("FLUX API configuration missing")
            
            # Prepare API request
            request_data = {
                "model": "flux.1-schnell",
                "prompt": params["prompt"],
                "width": params["width"],
                "height": params["height"],
                "guidance_scale": params["guidance_scale"],
                "num_inference_steps": params["num_inference_steps"]
            }
            
            # Make API request
            import aiohttp
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                async with session.post(api_url, json=request_data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("image_data", "")
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"API error {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Error with FLUX API generation: {e}")
            raise
    
    async def _generate_with_openai(self, params: Dict[str, Any]) -> str:
        """Generate image using OpenAI DALL-E"""
        try:
            openai_config = self.api_configs.get("openai", {})
            api_key = openai_config.get("api_key", "")
            
            if not api_key:
                raise ValueError("OpenAI API key not configured")
            
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)
            
            # DALL-E 3 parameters
            size = self._get_dalle_size(params["width"], params["height"])
            quality = openai_config.get("quality", "standard")
            
            response = await client.images.generate(
                model="dall-e-3",
                prompt=params["prompt"],
                size=size,
                quality=quality,
                response_format="b64_json",
                n=1
            )
            
            image_data = response.data[0].b64_json
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            raise
    
    async def _generate_with_local_model(self, params: Dict[str, Any]) -> str:
        """Generate using local diffusion model"""
        try:
            # This would integrate with local Stable Diffusion or similar
            local_config = self.api_configs.get("local", {})
            
            # Placeholder implementation
            # In real implementation, this would use local model inference
            
            logger.warning("Local model generation not implemented, using placeholder")
            return await self._generate_placeholder_image(params["width"], params["height"])
            
        except Exception as e:
            logger.error(f"Error with local model generation: {e}")
            raise
    
    def _validate_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Validate and adjust image dimensions"""
        # Ensure dimensions are multiples of 8 (common requirement for diffusion models)
        width = max(256, min(2048, (width // 8) * 8))
        height = max(256, min(2048, (height // 8) * 8))
        
        # Check aspect ratio limits
        ratio = max(width, height) / min(width, height)
        if ratio > 4.0:
            # Adjust to maintain reasonable aspect ratio
            if width > height:
                width = height * 4
            else:
                height = width * 4
        
        return width, height
    
    def _get_dalle_size(self, width: int, height: int) -> str:
        """Convert dimensions to DALL-E compatible size"""
        # DALL-E 3 supported sizes
        sizes = ["1024x1024", "1024x1792", "1792x1024"]
        
        ratio = width / height
        
        if abs(ratio - 1.0) < 0.1:  # Square-ish
            return "1024x1024"
        elif ratio > 1.2:  # Landscape
            return "1792x1024"
        else:  # Portrait
            return "1024x1792"
    
    async def _generate_placeholder_image(self, width: int, height: int) -> str:
        """Generate placeholder image when models fail"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create solid color background
            image = Image.new('RGB', (width, height), color='#f0f0f0')
            draw = ImageDraw.Draw(image)
            
            # Add placeholder text
            try:
                font = ImageFont.truetype("arial.ttf", 36)
            except OSError:
                font = ImageFont.load_default()
            
            text = "Placeholder Background"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            draw.text((x, y), text, fill='#888888', font=font)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            
            image_data = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error creating placeholder image: {e}")
            # Return minimal base64 image
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    async def _initialize_clients(self):
        """Initialize model clients"""
        try:
            # Initialize based on configuration
            for model_type, config in self.api_configs.items():
                if config.get("enabled", False):
                    logger.info(f"Initializing {model_type} client...")
                    # Client initialization would go here
            
            logger.info("T2I interface initialized")
            
        except Exception as e:
            logger.error(f"Error initializing T2I clients: {e}")
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models"""
        models = []
        
        if self.api_configs.get("flux", {}).get("enabled", False):
            models.extend(["flux.1-schnell", "flux.1-dev"])
        
        if self.api_configs.get("openai", {}).get("enabled", False):
            models.extend(["dall-e-3", "dall-e-2"])
        
        if self.api_configs.get("local", {}).get("enabled", False):
            models.extend(["stable-diffusion", "local-model"])
        
        return models
    
    async def estimate_generation_time(self, 
                                     width: int, 
                                     height: int, 
                                     num_inference_steps: int,
                                     model: Optional[str] = None) -> float:
        """Estimate generation time in seconds"""
        model_name = model or self.default_model
        
        # Base time estimates (in seconds)
        base_times = {
            "flux.1-schnell": 3.0,   # Very fast
            "flux.1-dev": 15.0,      # Slower but higher quality
            "dall-e-3": 10.0,        # API dependent
            "stable-diffusion": 8.0,  # Local model
        }
        
        base_time = base_times.get(model_name, 10.0)
        
        # Adjust for parameters
        pixel_factor = (width * height) / (1024 * 1024)
        step_factor = num_inference_steps / 25
        
        estimated_time = base_time * pixel_factor * step_factor
        
        return max(1.0, estimated_time)
    
    def get_model_capabilities(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get capabilities of specific model"""
        model_name = model or self.default_model
        
        capabilities = {
            "flux.1-schnell": {
                "max_resolution": (2048, 2048),
                "min_resolution": (256, 256),
                "supports_guidance": True,
                "max_guidance_scale": 3.5,
                "optimal_steps": 4,
                "max_steps": 8,
                "speed": "very_fast",
                "quality": "high"
            },
            "flux.1-dev": {
                "max_resolution": (2048, 2048),
                "min_resolution": (256, 256),
                "supports_guidance": True,
                "max_guidance_scale": 20.0,
                "optimal_steps": 25,
                "max_steps": 50,
                "speed": "medium",
                "quality": "very_high"
            },
            "dall-e-3": {
                "max_resolution": (1792, 1792),
                "min_resolution": (1024, 1024),
                "supports_guidance": False,
                "fixed_sizes": ["1024x1024", "1024x1792", "1792x1024"],
                "speed": "medium",
                "quality": "very_high"
            }
        }
        
        return capabilities.get(model_name, {
            "max_resolution": (1024, 1024),
            "min_resolution": (512, 512),
            "supports_guidance": True,
            "speed": "medium",
            "quality": "medium"
        })
