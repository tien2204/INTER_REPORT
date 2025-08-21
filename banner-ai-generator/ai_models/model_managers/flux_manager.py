"""
FLUX Model Manager

Manages FLUX.1-schnell and FLUX.1-dev models for high-quality
text-to-image generation, with optimization for speed and quality.
"""

import asyncio
import torch
from typing import Any, Dict, List, Optional, Union
import base64
import io
from PIL import Image
from datetime import datetime
from structlog import get_logger

logger = get_logger(__name__)


class FluxManager:
    """
    Manager for FLUX text-to-image models
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Model configurations
        self.models_config = {
            "flux.1-schnell": {
                "model_path": "black-forest-labs/FLUX.1-schnell",
                "max_width": 2048,
                "max_height": 2048,
                "optimal_steps": 4,
                "max_steps": 8,
                "guidance_scale_range": (1.0, 3.5),
                "speed": "very_fast",
                "quality": "high",
                "memory_requirement": "8GB"
            },
            "flux.1-dev": {
                "model_path": "black-forest-labs/FLUX.1-dev",
                "max_width": 2048,
                "max_height": 2048,
                "optimal_steps": 25,
                "max_steps": 50,
                "guidance_scale_range": (1.0, 20.0),
                "speed": "medium",
                "quality": "very_high",
                "memory_requirement": "12GB"
            }
        }
        
        # Runtime configuration
        self.device = config.get("device", "auto")
        self.use_cpu_offload = config.get("cpu_offload", False)
        self.enable_attention_slicing = config.get("attention_slicing", True)
        self.enable_memory_efficient_attention = config.get("memory_efficient_attention", True)
        
        # API configuration for external FLUX services
        self.api_config = config.get("api", {})
        self.use_local = config.get("use_local", True)
        
        # Initialize pipelines
        self._pipelines = {}
        self._device = self._setup_device()
        self._loaded_models = set()
        
        # Usage tracking
        self._usage_stats = {
            "generations_today": 0,
            "total_inference_time": 0.0,
            "last_reset": datetime.now().date()
        }
        
        if self.use_local:
            asyncio.create_task(self._initialize_local())
    
    async def generate_image(self, 
                           prompt: str,
                           model: str = "flux.1-schnell",
                           width: int = 1024,
                           height: int = 1024,
                           num_inference_steps: Optional[int] = None,
                           guidance_scale: Optional[float] = None,
                           seed: Optional[int] = None,
                           **kwargs) -> str:
        """
        Generate image using FLUX model
        
        Args:
            prompt: Text prompt for image generation
            model: FLUX model name
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducibility
            **kwargs: Additional parameters
        
        Returns:
            Base64 encoded image data
        """
        try:
            # Validate model
            if model not in self.models_config:
                raise ValueError(f"Unsupported FLUX model: {model}")
            
            model_config = self.models_config[model]
            
            # Validate and adjust parameters
            width, height = self._validate_dimensions(width, height, model_config)
            num_inference_steps = self._validate_steps(num_inference_steps, model_config)
            guidance_scale = self._validate_guidance_scale(guidance_scale, model_config)
            
            logger.info(f"Generating image with {model}: {width}x{height}, {num_inference_steps} steps")
            
            start_time = datetime.now()
            
            if self.use_local:
                image_data = await self._generate_local(
                    prompt, model, width, height, num_inference_steps, 
                    guidance_scale, seed, **kwargs
                )
            else:
                image_data = await self._generate_api(
                    prompt, model, width, height, num_inference_steps,
                    guidance_scale, seed, **kwargs
                )
            
            # Track usage
            inference_time = (datetime.now() - start_time).total_seconds()
            self._update_usage_stats(inference_time)
            
            logger.info(f"Image generation completed in {inference_time:.2f}s")
            return image_data
            
        except Exception as e:
            logger.error(f"Error generating image with FLUX: {e}")
            raise
    
    async def _generate_local(self, 
                            prompt: str,
                            model: str,
                            width: int,
                            height: int,
                            num_inference_steps: int,
                            guidance_scale: float,
                            seed: Optional[int],
                            **kwargs) -> str:
        """Generate image using local FLUX pipeline"""
        try:
            # Load pipeline if needed
            pipeline = await self._get_pipeline(model)
            
            # Set random seed if provided
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(seed)
            
            # Generate image
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    **kwargs
                )
            
            # Convert to base64
            image = result.images[0]
            return self._image_to_base64(image)
            
        except Exception as e:
            logger.error(f"Error in local FLUX generation: {e}")
            raise
    
    async def _generate_api(self, 
                          prompt: str,
                          model: str,
                          width: int,
                          height: int,
                          num_inference_steps: int,
                          guidance_scale: float,
                          seed: Optional[int],
                          **kwargs) -> str:
        """Generate image using FLUX API"""
        try:
            api_url = self.api_config.get("url", "")
            api_key = self.api_config.get("key", "")
            
            if not api_url:
                raise ValueError("FLUX API URL not configured")
            
            # Prepare request
            request_data = {
                "model": model,
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }
            
            if seed is not None:
                request_data["seed"] = seed
            
            # Make API request
            import aiohttp
            async with aiohttp.ClientSession() as session:
                headers = {}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                
                async with session.post(api_url, json=request_data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("image_data", "")
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"API error {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Error with FLUX API: {e}")
            raise
    
    async def _get_pipeline(self, model: str):
        """Get or load FLUX pipeline"""
        try:
            if model in self._pipelines:
                return self._pipelines[model]
            
            # Load pipeline
            logger.info(f"Loading FLUX pipeline: {model}")
            
            from diffusers import FluxPipeline
            
            model_config = self.models_config[model]
            model_path = model_config["model_path"]
            
            # Load with optimizations
            pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self._device.type == "cuda" else torch.float32,
                device_map="auto" if self.use_cpu_offload else None
            )
            
            # Move to device if not using cpu offload
            if not self.use_cpu_offload:
                pipeline = pipeline.to(self._device)
            
            # Apply optimizations
            self._apply_optimizations(pipeline)
            
            # Cache pipeline
            self._pipelines[model] = pipeline
            self._loaded_models.add(model)
            
            logger.info(f"FLUX pipeline {model} loaded successfully")
            return pipeline
            
        except ImportError:
            logger.error("FLUX dependencies not available. Please install diffusers and related packages.")
            raise
        except Exception as e:
            logger.error(f"Error loading FLUX pipeline {model}: {e}")
            raise
    
    def _apply_optimizations(self, pipeline):
        """Apply performance optimizations to pipeline"""
        try:
            # Enable attention slicing for memory efficiency
            if self.enable_attention_slicing:
                pipeline.enable_attention_slicing()
            
            # Enable memory efficient attention
            if self.enable_memory_efficient_attention:
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except Exception:
                    # xformers might not be available
                    pass
            
            # Enable CPU offloading if configured
            if self.use_cpu_offload:
                pipeline.enable_sequential_cpu_offload()
            
            logger.info("Applied FLUX pipeline optimizations")
            
        except Exception as e:
            logger.error(f"Error applying optimizations: {e}")
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        try:
            if self.device == "auto":
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
                elif torch.backends.mps.is_available():
                    device = torch.device("mps")
                    logger.info("Using MPS device")
                else:
                    device = torch.device("cpu")
                    logger.info("Using CPU device")
            else:
                device = torch.device(self.device)
                logger.info(f"Using specified device: {device}")
            
            return device
            
        except Exception as e:
            logger.error(f"Error setting up device: {e}")
            return torch.device("cpu")
    
    def _validate_dimensions(self, width: int, height: int, model_config: Dict[str, Any]) -> tuple[int, int]:
        """Validate and adjust image dimensions"""
        max_width = model_config["max_width"]
        max_height = model_config["max_height"]
        
        # Ensure dimensions are multiples of 8
        width = min(max_width, max(256, (width // 8) * 8))
        height = min(max_height, max(256, (height // 8) * 8))
        
        # Check aspect ratio limits
        ratio = max(width, height) / min(width, height)
        if ratio > 4.0:
            # Adjust to maintain reasonable aspect ratio
            if width > height:
                width = height * 4
            else:
                height = width * 4
        
        return width, height
    
    def _validate_steps(self, steps: Optional[int], model_config: Dict[str, Any]) -> int:
        """Validate and adjust inference steps"""
        if steps is None:
            return model_config["optimal_steps"]
        
        max_steps = model_config["max_steps"]
        return max(1, min(max_steps, steps))
    
    def _validate_guidance_scale(self, guidance_scale: Optional[float], model_config: Dict[str, Any]) -> float:
        """Validate and adjust guidance scale"""
        if guidance_scale is None:
            # Use optimal guidance scale
            guidance_range = model_config["guidance_scale_range"]
            return (guidance_range[0] + guidance_range[1]) / 2
        
        guidance_range = model_config["guidance_scale_range"]
        return max(guidance_range[0], min(guidance_range[1], guidance_scale))
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            
            image_data = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise
    
    def _update_usage_stats(self, inference_time: float):
        """Update usage statistics"""
        try:
            today = datetime.now().date()
            if self._usage_stats["last_reset"] != today:
                self._usage_stats["generations_today"] = 0
                self._usage_stats["total_inference_time"] = 0.0
                self._usage_stats["last_reset"] = today
            
            self._usage_stats["generations_today"] += 1
            self._usage_stats["total_inference_time"] += inference_time
            
        except Exception as e:
            logger.error(f"Error updating usage stats: {e}")
    
    async def _initialize_local(self):
        """Initialize local FLUX setup"""
        try:
            # Check if torch is available
            if not torch.cuda.is_available() and self.device != "cpu":
                logger.warning("CUDA not available, falling back to CPU")
                self._device = torch.device("cpu")
            
            # Pre-load default model if configured
            default_model = self.config.get("preload_model")
            if default_model and default_model in self.models_config:
                logger.info(f"Pre-loading FLUX model: {default_model}")
                await self._get_pipeline(default_model)
            
            logger.info("FLUX manager initialized for local inference")
            
        except Exception as e:
            logger.error(f"Error initializing local FLUX: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available FLUX models"""
        return list(self.models_config.keys())
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about specific FLUX model"""
        return self.models_config.get(model, {})
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self._usage_stats.copy()
        stats["loaded_models"] = list(self._loaded_models)
        stats["device"] = str(self._device)
        return stats
    
    async def estimate_generation_time(self, 
                                     model: str,
                                     width: int,
                                     height: int,
                                     num_inference_steps: int) -> float:
        """Estimate generation time in seconds"""
        try:
            model_config = self.models_config.get(model, {})
            
            # Base time per step (approximate)
            if model == "flux.1-schnell":
                base_time_per_step = 0.5  # Very fast
            elif model == "flux.1-dev":
                base_time_per_step = 1.2  # Higher quality but slower
            else:
                base_time_per_step = 1.0
            
            # Adjust for resolution
            pixel_factor = (width * height) / (1024 * 1024)
            
            # Adjust for device
            device_factor = 1.0
            if self._device.type == "cpu":
                device_factor = 5.0  # Much slower on CPU
            elif self._device.type == "mps":
                device_factor = 2.0  # Slower than CUDA
            
            estimated_time = base_time_per_step * num_inference_steps * pixel_factor * device_factor
            
            return max(1.0, estimated_time)
            
        except Exception as e:
            logger.error(f"Error estimating generation time: {e}")
            return 30.0  # Conservative fallback
    
    async def optimize_for_speed(self, model: str):
        """Optimize specific model for speed"""
        try:
            if model in self._pipelines:
                pipeline = self._pipelines[model]
                
                # Apply additional speed optimizations
                try:
                    # Compile model for faster inference (PyTorch 2.0+)
                    if hasattr(torch, 'compile'):
                        pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
                        logger.info(f"Compiled {model} UNet for faster inference")
                except Exception as e:
                    logger.warning(f"Could not compile model: {e}")
                
        except Exception as e:
            logger.error(f"Error optimizing model for speed: {e}")
    
    async def clear_cache(self):
        """Clear model cache to free memory"""
        try:
            # Clear pipelines
            for model in list(self._pipelines.keys()):
                del self._pipelines[model]
            
            self._pipelines.clear()
            self._loaded_models.clear()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Cleared FLUX model cache")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        try:
            memory_info = {
                "loaded_models": list(self._loaded_models),
                "device": str(self._device)
            }
            
            if torch.cuda.is_available():
                memory_info["gpu_memory"] = {
                    "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                    "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
                    "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
                }
            
            return memory_info
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}
