"""
Custom Model Manager

Manages custom and local AI models including local LLMs,
custom fine-tuned models, and specialized models.
"""

import asyncio
import json
import subprocess
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
from structlog import get_logger

logger = get_logger(__name__)


class CustomModelManager:
    """
    Manager for custom and local AI models
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Model configurations
        self.models_config = config.get("models", {})
        self.model_directories = config.get("model_directories", ["./models", "~/.cache/models"])
        
        # Runtime configurations
        self.max_memory_gb = config.get("max_memory_gb", 8)
        self.max_concurrent_models = config.get("max_concurrent_models", 2)
        
        # Model types supported
        self.supported_types = {
            "llm": {
                "frameworks": ["transformers", "llama.cpp", "ollama"],
                "formats": [".bin", ".gguf", ".safetensors"]
            },
            "diffusion": {
                "frameworks": ["diffusers", "comfyui"],
                "formats": [".bin", ".safetensors", ".ckpt"]
            },
            "vision": {
                "frameworks": ["transformers", "clip"],
                "formats": [".bin", ".safetensors"]
            }
        }
        
        # Loaded models tracking
        self._loaded_models = {}
        self._model_processes = {}
        self._available_models = {}
        
        # Initialize
        asyncio.create_task(self._initialize())
    
    async def load_model(self, 
                        model_name: str,
                        model_type: str = "llm",
                        framework: str = "auto",
                        **kwargs) -> bool:
        """
        Load a custom model
        
        Args:
            model_name: Name or path of the model
            model_type: Type of model (llm, diffusion, vision)
            framework: Framework to use (auto-detect if not specified)
            **kwargs: Additional loading parameters
        
        Returns:
            True if model loaded successfully
        """
        try:
            if model_name in self._loaded_models:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            # Check memory constraints
            if len(self._loaded_models) >= self.max_concurrent_models:
                logger.warning(f"Max concurrent models ({self.max_concurrent_models}) reached")
                return False
            
            logger.info(f"Loading custom model: {model_name}")
            
            # Detect framework if auto
            if framework == "auto":
                framework = await self._detect_framework(model_name, model_type)
            
            # Load based on framework
            if framework == "transformers":
                success = await self._load_transformers_model(model_name, model_type, **kwargs)
            elif framework == "llama.cpp":
                success = await self._load_llamacpp_model(model_name, **kwargs)
            elif framework == "ollama":
                success = await self._load_ollama_model(model_name, **kwargs)
            elif framework == "diffusers":
                success = await self._load_diffusers_model(model_name, **kwargs)
            else:
                raise ValueError(f"Unsupported framework: {framework}")
            
            if success:
                self._loaded_models[model_name] = {
                    "type": model_type,
                    "framework": framework,
                    "loaded_at": datetime.now(),
                    "config": kwargs
                }
                logger.info(f"Successfully loaded model: {model_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    async def generate_text(self, 
                           model_name: str,
                           prompt: str,
                           max_tokens: int = 512,
                           temperature: float = 0.7,
                           **kwargs) -> str:
        """
        Generate text using a loaded model
        
        Args:
            model_name: Name of the loaded model
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        try:
            if model_name not in self._loaded_models:
                raise ValueError(f"Model {model_name} not loaded")
            
            model_info = self._loaded_models[model_name]
            framework = model_info["framework"]
            
            if framework == "transformers":
                return await self._generate_transformers(
                    model_name, prompt, max_tokens, temperature, **kwargs
                )
            elif framework == "llama.cpp":
                return await self._generate_llamacpp(
                    model_name, prompt, max_tokens, temperature, **kwargs
                )
            elif framework == "ollama":
                return await self._generate_ollama(
                    model_name, prompt, max_tokens, temperature, **kwargs
                )
            else:
                raise ValueError(f"Text generation not supported for framework: {framework}")
                
        except Exception as e:
            logger.error(f"Error generating text with {model_name}: {e}")
            raise
    
    async def generate_image(self, 
                           model_name: str,
                           prompt: str,
                           width: int = 512,
                           height: int = 512,
                           **kwargs) -> str:
        """
        Generate image using a loaded diffusion model
        
        Args:
            model_name: Name of the loaded model
            prompt: Text prompt
            width: Image width
            height: Image height
            **kwargs: Additional parameters
        
        Returns:
            Base64 encoded image
        """
        try:
            if model_name not in self._loaded_models:
                raise ValueError(f"Model {model_name} not loaded")
            
            model_info = self._loaded_models[model_name]
            
            if model_info["type"] != "diffusion":
                raise ValueError(f"Model {model_name} is not a diffusion model")
            
            framework = model_info["framework"]
            
            if framework == "diffusers":
                return await self._generate_diffusers_image(
                    model_name, prompt, width, height, **kwargs
                )
            else:
                raise ValueError(f"Image generation not supported for framework: {framework}")
                
        except Exception as e:
            logger.error(f"Error generating image with {model_name}: {e}")
            raise
    
    async def _load_transformers_model(self, model_name: str, model_type: str, **kwargs) -> bool:
        """Load model using Transformers library"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            # Determine device
            device = kwargs.get("device", "auto")
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if model_type == "llm":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    **kwargs
                )
                
                # Create pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
                
                self._loaded_models[model_name + "_pipeline"] = pipe
            
            return True
            
        except ImportError:
            logger.error("Transformers library not available")
            return False
        except Exception as e:
            logger.error(f"Error loading Transformers model: {e}")
            return False
    
    async def _load_llamacpp_model(self, model_name: str, **kwargs) -> bool:
        """Load model using llama.cpp"""
        try:
            # Check if llama.cpp is available
            model_path = await self._find_model_path(model_name, [".gguf", ".bin"])
            if not model_path:
                logger.error(f"Model file not found for {model_name}")
                return False
            
            # Start llama.cpp server process
            cmd = [
                "llama-server",
                "--model", str(model_path),
                "--port", str(kwargs.get("port", 8080)),
                "--threads", str(kwargs.get("threads", 4))
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self._model_processes[model_name] = process
            
            # Wait a bit for server to start
            await asyncio.sleep(2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading llama.cpp model: {e}")
            return False
    
    async def _load_ollama_model(self, model_name: str, **kwargs) -> bool:
        """Load model using Ollama"""
        try:
            # Check if Ollama is available
            result = await asyncio.create_subprocess_exec(
                "ollama", "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.error("Ollama not available")
                return False
            
            # Check if model is available
            available_models = stdout.decode().strip().split('\n')[1:]  # Skip header
            model_available = any(model_name in line for line in available_models)
            
            if not model_available:
                # Try to pull the model
                logger.info(f"Pulling Ollama model: {model_name}")
                result = await asyncio.create_subprocess_exec(
                    "ollama", "pull", model_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await result.communicate()
                
                if result.returncode != 0:
                    logger.error(f"Failed to pull Ollama model: {model_name}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading Ollama model: {e}")
            return False
    
    async def _load_diffusers_model(self, model_name: str, **kwargs) -> bool:
        """Load diffusion model using Diffusers"""
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            device = kwargs.get("device", "auto")
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            self._loaded_models[model_name + "_pipeline"] = pipeline
            
            return True
            
        except ImportError:
            logger.error("Diffusers library not available")
            return False
        except Exception as e:
            logger.error(f"Error loading Diffusers model: {e}")
            return False
    
    async def _generate_transformers(self, model_name: str, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate text using Transformers pipeline"""
        try:
            pipeline = self._loaded_models.get(model_name + "_pipeline")
            if not pipeline:
                raise ValueError(f"Pipeline not found for {model_name}")
            
            result = pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                **kwargs
            )
            
            return result[0]["generated_text"][len(prompt):]  # Remove input prompt
            
        except Exception as e:
            logger.error(f"Error generating with Transformers: {e}")
            raise
    
    async def _generate_llamacpp(self, model_name: str, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate text using llama.cpp server"""
        try:
            import aiohttp
            
            port = self._loaded_models[model_name]["config"].get("port", 8080)
            url = f"http://localhost:{port}/completion"
            
            data = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("content", "")
                    else:
                        raise RuntimeError(f"llama.cpp server error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error generating with llama.cpp: {e}")
            raise
    
    async def _generate_ollama(self, model_name: str, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate text using Ollama"""
        try:
            import aiohttp
            
            url = "http://localhost:11434/api/generate"
            
            data = {
                "model": model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    **kwargs
                },
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        raise RuntimeError(f"Ollama error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise
    
    async def _generate_diffusers_image(self, model_name: str, prompt: str, width: int, height: int, **kwargs) -> str:
        """Generate image using Diffusers pipeline"""
        try:
            import base64
            import io
            
            pipeline = self._loaded_models.get(model_name + "_pipeline")
            if not pipeline:
                raise ValueError(f"Pipeline not found for {model_name}")
            
            result = pipeline(
                prompt,
                width=width,
                height=height,
                **kwargs
            )
            
            image = result.images[0]
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            
            image_data = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error generating image with Diffusers: {e}")
            raise
    
    async def _detect_framework(self, model_name: str, model_type: str) -> str:
        """Auto-detect appropriate framework for model"""
        try:
            # Check if it's a Hugging Face model
            if "/" in model_name and not Path(model_name).exists():
                return "transformers"
            
            # Check file extensions
            model_path = await self._find_model_path(model_name)
            if model_path:
                suffix = model_path.suffix.lower()
                
                if suffix in [".gguf"]:
                    return "llama.cpp"
                elif suffix in [".bin", ".safetensors"] and model_type == "llm":
                    return "transformers"
                elif suffix in [".bin", ".safetensors", ".ckpt"] and model_type == "diffusion":
                    return "diffusers"
            
            # Check if Ollama model exists
            try:
                result = await asyncio.create_subprocess_exec(
                    "ollama", "list",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await result.communicate()
                
                if result.returncode == 0:
                    available_models = stdout.decode()
                    if model_name in available_models:
                        return "ollama"
            except:
                pass
            
            # Default fallback
            return "transformers" if model_type == "llm" else "diffusers"
            
        except Exception as e:
            logger.error(f"Error detecting framework: {e}")
            return "transformers"
    
    async def _find_model_path(self, model_name: str, extensions: List[str] = None) -> Optional[Path]:
        """Find model file path"""
        try:
            # If it's already a path
            if Path(model_name).exists():
                return Path(model_name)
            
            # Search in model directories
            for directory in self.model_directories:
                dir_path = Path(directory).expanduser()
                if not dir_path.exists():
                    continue
                
                # Direct file match
                model_path = dir_path / model_name
                if model_path.exists():
                    return model_path
                
                # Search with extensions
                if extensions:
                    for ext in extensions:
                        file_path = dir_path / f"{model_name}{ext}"
                        if file_path.exists():
                            return file_path
                
                # Search subdirectories
                for subdir in dir_path.iterdir():
                    if subdir.is_dir() and model_name in subdir.name:
                        for file in subdir.iterdir():
                            if extensions and file.suffix in extensions:
                                return file
                            elif not extensions and file.is_file():
                                return file
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding model path: {e}")
            return None
    
    async def _scan_available_models(self):
        """Scan for available models in configured directories"""
        try:
            available = {}
            
            for directory in self.model_directories:
                dir_path = Path(directory).expanduser()
                if not dir_path.exists():
                    continue
                
                for item in dir_path.iterdir():
                    if item.is_file():
                        # Single model file
                        model_type = self._detect_model_type(item)
                        if model_type:
                            available[item.stem] = {
                                "path": str(item),
                                "type": model_type,
                                "size": item.stat().st_size
                            }
                    elif item.is_dir():
                        # Model directory
                        model_files = list(item.glob("*.bin")) + list(item.glob("*.safetensors"))
                        if model_files:
                            model_type = self._detect_model_type(item)
                            available[item.name] = {
                                "path": str(item),
                                "type": model_type,
                                "size": sum(f.stat().st_size for f in model_files)
                            }
            
            self._available_models = available
            logger.info(f"Found {len(available)} available models")
            
        except Exception as e:
            logger.error(f"Error scanning models: {e}")
    
    def _detect_model_type(self, path: Path) -> Optional[str]:
        """Detect model type from path"""
        try:
            name = path.name.lower()
            
            if any(keyword in name for keyword in ["llama", "gpt", "mistral", "chat"]):
                return "llm"
            elif any(keyword in name for keyword in ["stable-diffusion", "sd", "diffusion"]):
                return "diffusion"
            elif any(keyword in name for keyword in ["clip", "vision", "blip"]):
                return "vision"
            
            return "unknown"
            
        except Exception:
            return None
    
    async def _initialize(self):
        """Initialize custom model manager"""
        try:
            # Scan for available models
            await self._scan_available_models()
            
            # Load any configured models
            auto_load = self.config.get("auto_load", [])
            for model_config in auto_load:
                model_name = model_config["name"]
                model_type = model_config.get("type", "llm")
                framework = model_config.get("framework", "auto")
                
                success = await self.load_model(model_name, model_type, framework)
                if success:
                    logger.info(f"Auto-loaded model: {model_name}")
                else:
                    logger.warning(f"Failed to auto-load model: {model_name}")
            
            logger.info("Custom model manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing custom model manager: {e}")
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """Get information about currently loaded models"""
        return {
            name: {
                "type": info["type"],
                "framework": info["framework"],
                "loaded_at": info["loaded_at"].isoformat(),
                "memory_usage": self._estimate_memory_usage(name)
            }
            for name, info in self._loaded_models.items()
            if not name.endswith("_pipeline")
        }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models"""
        return self._available_models.copy()
    
    def _estimate_memory_usage(self, model_name: str) -> str:
        """Estimate memory usage of loaded model"""
        try:
            # This is a rough estimation
            model_info = self._available_models.get(model_name, {})
            size_bytes = model_info.get("size", 0)
            
            if size_bytes > 0:
                # Models typically use 2-4x their file size in memory
                memory_gb = (size_bytes * 3) / (1024**3)
                return f"{memory_gb:.1f}GB"
            
            return "Unknown"
            
        except Exception:
            return "Unknown"
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory"""
        try:
            if model_name not in self._loaded_models:
                logger.warning(f"Model {model_name} not loaded")
                return False
            
            # Stop process if it exists
            if model_name in self._model_processes:
                process = self._model_processes[model_name]
                process.terminate()
                await process.wait()
                del self._model_processes[model_name]
            
            # Remove from loaded models
            del self._loaded_models[model_name]
            
            # Remove pipeline if it exists
            pipeline_key = model_name + "_pipeline"
            if pipeline_key in self._loaded_models:
                del self._loaded_models[pipeline_key]
            
            logger.info(f"Unloaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup all loaded models and processes"""
        try:
            # Stop all processes
            for process in self._model_processes.values():
                process.terminate()
                await process.wait()
            
            # Clear all loaded models
            self._loaded_models.clear()
            self._model_processes.clear()
            
            logger.info("Custom model manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
