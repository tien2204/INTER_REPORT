"""
AI Models configuration management
"""

import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    GOOGLE = "google"
    AZURE = "azure"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"
    STABILITY = "stability"
    LOCAL = "local"

@dataclass
class LLMConfig:
    """Large Language Model configuration"""
    # Model settings
    provider: ModelProvider = ModelProvider.OPENAI
    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    organization: Optional[str] = None
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Request settings
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Streaming and async
    stream: bool = False
    async_enabled: bool = True
    
    # Cost optimization
    use_cache: bool = True
    cache_ttl: int = 3600
    
    # Fallback models
    fallback_models: List[str] = field(default_factory=list)
    
    def get_provider_config(self) -> Dict[str, Any]:
        """Get provider-specific configuration"""
        if self.provider == ModelProvider.OPENAI:
            return {
                "api_key": self.api_key,
                "organization": self.organization,
                "api_base": self.api_base
            }
        elif self.provider == ModelProvider.ANTHROPIC:
            return {
                "api_key": self.api_key
            }
        elif self.provider == ModelProvider.AZURE:
            return {
                "api_key": self.api_key,
                "api_base": self.api_base,
                "api_version": "2023-05-15"
            }
        else:
            return {"api_key": self.api_key}

@dataclass
class T2IConfig:
    """Text-to-Image model configuration"""
    # Model settings
    provider: ModelProvider = ModelProvider.REPLICATE
    model_name: str = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
    api_key: Optional[str] = None
    
    # Generation parameters
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    
    # Quality settings
    scheduler: str = "DPMSolverMultistep"
    negative_prompt: str = "ugly, blurry, poor quality, distorted, text, watermark"
    
    # Output settings
    output_format: str = "png"
    output_quality: int = 95
    
    # Performance
    batch_size: int = 1
    timeout: int = 300
    max_retries: int = 2
    
    # Cost optimization
    use_cache: bool = True
    cache_ttl: int = 86400  # 24 hours
    
    # Model-specific settings
    flux_settings: Dict[str, Any] = field(default_factory=lambda: {
        "num_outputs": 1,
        "aspect_ratio": "16:9",
        "output_format": "webp",
        "output_quality": 80
    })
    
    sdxl_settings: Dict[str, Any] = field(default_factory=lambda: {
        "refine": "expert_ensemble_refiner",
        "high_noise_frac": 0.8,
        "apply_watermark": False
    })

@dataclass
class MLLMConfig:
    """Multimodal Large Language Model configuration"""
    # Model settings
    provider: ModelProvider = ModelProvider.OPENAI
    model_name: str = "gpt-4-vision-preview"
    api_key: Optional[str] = None
    
    # Vision parameters
    max_image_size: tuple = (1024, 1024)
    image_detail: str = "auto"  # low, high, auto
    supported_formats: List[str] = field(default_factory=lambda: ["png", "jpeg", "jpg", "webp"])
    
    # Text generation
    temperature: float = 0.3
    max_tokens: int = 1500
    
    # Request settings
    timeout: int = 90
    max_retries: int = 3
    
    # Capabilities
    text_detection: bool = True
    object_detection: bool = True
    scene_analysis: bool = True
    design_critique: bool = True
    
    # Cost optimization
    use_cache: bool = True
    cache_ttl: int = 1800

@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    # Model settings
    provider: ModelProvider = ModelProvider.OPENAI
    model_name: str = "text-embedding-ada-002"
    api_key: Optional[str] = None
    
    # Parameters
    dimensions: int = 1536
    batch_size: int = 100
    
    # Performance
    timeout: int = 30
    max_retries: int = 3
    
    # Caching
    use_cache: bool = True
    cache_ttl: int = 604800  # 1 week

@dataclass
class LocalModelConfig:
    """Local model configuration"""
    # Model paths
    model_path: str = "models/"
    cache_dir: str = "model_cache/"
    
    # Hardware settings
    device: str = "auto"  # auto, cpu, cuda, mps
    gpu_memory_fraction: float = 0.8
    cpu_threads: int = 4
    
    # Model loading
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "float16"
    
    # Performance
    batch_size: int = 1
    max_memory: Dict[str, str] = field(default_factory=dict)

class ModelConfig:
    """AI Models configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("MODEL_CONFIG_PATH", "config/models.json")
        
        # Initialize default configurations
        self.llm = LLMConfig()
        self.t2i = T2IConfig()
        self.mllm = MLLMConfig()
        self.embedding = EmbeddingConfig()
        self.local = LocalModelConfig()
        
        # Model registry for different use cases
        self._model_registry = {
            "strategist": {
                "brief_analysis": LLMConfig(model_name="gpt-4", temperature=0.3),
                "brand_analysis": MLLMConfig(model_name="gpt-4-vision-preview", temperature=0.2)
            },
            "background_designer": {
                "image_generation": T2IConfig(
                    provider=ModelProvider.REPLICATE,
                    model_name="black-forest-labs/flux-schnell",
                    width=1200,
                    height=628
                ),
                "text_detection": MLLMConfig(model_name="gpt-4-vision-preview", temperature=0.1)
            },
            "foreground_designer": {
                "blueprint_generation": LLMConfig(model_name="gpt-4", temperature=0.4),
                "color_analysis": MLLMConfig(model_name="gpt-4-vision-preview", temperature=0.2)
            },
            "developer": {
                "code_generation": LLMConfig(model_name="gpt-4", temperature=0.2),
                "code_optimization": LLMConfig(model_name="gpt-4", temperature=0.1)
            },
            "design_reviewer": {
                "design_critique": MLLMConfig(model_name="gpt-4-vision-preview", temperature=0.4),
                "feedback_generation": LLMConfig(model_name="gpt-4", temperature=0.6)
            }
        }
        
        # Load configuration
        self._load_from_env()
        if os.path.exists(self.config_path):
            self._load_from_file()
        
        self._validate_config()
    
    def _load_from_env(self) -> None:
        """Load model configuration from environment variables"""
        # API Keys
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.llm.api_key = openai_key
            self.mllm.api_key = openai_key
            self.embedding.api_key = openai_key
        
        self.t2i.api_key = os.getenv("REPLICATE_API_TOKEN")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Model overrides
        self.llm.model_name = os.getenv("DEFAULT_LLM_MODEL", self.llm.model_name)
        self.t2i.model_name = os.getenv("DEFAULT_T2I_MODEL", self.t2i.model_name)
        self.mllm.model_name = os.getenv("DEFAULT_MLLM_MODEL", self.mllm.model_name)
        
        # Performance settings
        if os.getenv("GPU_ENABLED", "true").lower() == "false":
            self.local.device = "cpu"
        
        logger.debug("Loaded model configuration from environment")
    
    def _load_from_file(self) -> None:
        """Load model configuration from file"""
        try:
            import json
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            for section, data in config_data.items():
                if hasattr(self, section):
                    section_config = getattr(self, section)
                    for key, value in data.items():
                        if hasattr(section_config, key):
                            if key == 'provider' and isinstance(value, str):
                                setattr(section_config, key, ModelProvider(value))
                            else:
                                setattr(section_config, key, value)
                elif section == "registry":
                    # Update model registry
                    self._update_model_registry(data)
            
            logger.info(f"Loaded model configuration from: {self.config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load model config file: {e}")
    
    def _update_model_registry(self, registry_data: Dict[str, Any]) -> None:
        """Update model registry from configuration"""
        for agent_type, models in registry_data.items():
            if agent_type not in self._model_registry:
                self._model_registry[agent_type] = {}
            
            for model_use, model_config in models.items():
                if model_config.get('type') == 'llm':
                    config = LLMConfig(**model_config.get('config', {}))
                elif model_config.get('type') == 't2i':
                    config = T2IConfig(**model_config.get('config', {}))
                elif model_config.get('type') == 'mllm':
                    config = MLLMConfig(**model_config.get('config', {}))
                else:
                    continue
                
                self._model_registry[agent_type][model_use] = config
    
    def _validate_config(self) -> None:
        """Validate model configurations"""
        # Check required API keys
        if not self.llm.api_key:
            logger.warning("No LLM API key configured")
        
        if not self.t2i.api_key:
            logger.warning("No T2I API key configured")
        
        # Validate parameters
        if self.llm.temperature < 0 or self.llm.temperature > 2:
            logger.error("LLM temperature must be between 0 and 2")
        
        if self.t2i.guidance_scale < 1 or self.t2i.guidance_scale > 20:
            logger.error("T2I guidance scale should be between 1 and 20")
        
        # Validate local model paths
        if not os.path.exists(self.local.model_path):
            os.makedirs(self.local.model_path, exist_ok=True)
        
        if not os.path.exists(self.local.cache_dir):
            os.makedirs(self.local.cache_dir, exist_ok=True)
    
    def get_model_config(self, agent_type: str, use_case: str) -> Optional[Union[LLMConfig, T2IConfig, MLLMConfig]]:
        """Get model configuration for specific agent and use case"""
        return self._model_registry.get(agent_type, {}).get(use_case)
    
    def set_model_config(self, agent_type: str, use_case: str, config: Union[LLMConfig, T2IConfig, MLLMConfig]) -> None:
        """Set model configuration for agent and use case"""
        if agent_type not in self._model_registry:
            self._model_registry[agent_type] = {}
        
        self._model_registry[agent_type][use_case] = config
        logger.info(f"Set model config for {agent_type}.{use_case}")
    
    def get_available_models(self, provider: ModelProvider) -> List[str]:
        """Get list of available models for provider"""
        model_lists = {
            ModelProvider.OPENAI: [
                "gpt-4", "gpt-4-turbo-preview", "gpt-4-vision-preview",
                "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
                "text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"
            ],
            ModelProvider.ANTHROPIC: [
                "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
            ],
            ModelProvider.REPLICATE: [
                "black-forest-labs/flux-schnell",
                "stability-ai/sdxl",
                "stability-ai/stable-diffusion"
            ],
            ModelProvider.STABILITY: [
                "stable-diffusion-xl-1024-v1-0",
                "stable-diffusion-v1-6"
            ]
        }
        
        return model_lists.get(provider, [])
    
    def update_api_key(self, provider: ModelProvider, api_key: str) -> None:
        """Update API key for provider"""
        if provider == ModelProvider.OPENAI:
            self.llm.api_key = api_key
            self.mllm.api_key = api_key
            self.embedding.api_key = api_key
        elif provider == ModelProvider.REPLICATE:
            self.t2i.api_key = api_key
        
        # Update registry configs
        for agent_configs in self._model_registry.values():
            for config in agent_configs.values():
                if hasattr(config, 'provider') and config.provider == provider:
                    config.api_key = api_key
        
        logger.info(f"Updated API key for provider: {provider.value}")
    
    def get_cost_estimate(self, agent_type: str, use_case: str, 
                         input_tokens: int = 0, output_tokens: int = 0,
                         images_generated: int = 0) -> float:
        """Estimate cost for model usage"""
        config = self.get_model_config(agent_type, use_case)
        if not config:
            return 0.0
        
        # Cost per 1K tokens (approximate)
        cost_mapping = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015}
        }
        
        if isinstance(config, LLMConfig) or isinstance(config, MLLMConfig):
            model_costs = cost_mapping.get(config.model_name, {"input": 0.001, "output": 0.002})
            return (input_tokens / 1000 * model_costs["input"] + 
                   output_tokens / 1000 * model_costs["output"])
        
        elif isinstance(config, T2IConfig):
            # Approximate T2I costs
            image_cost = 0.02  # per image
            return images_generated * image_cost
        
        return 0.0
    
    def save_config(self, path: Optional[str] = None) -> bool:
        """Save model configuration to file"""
        try:
            import json
            save_path = path or self.config_path
            
            # Prepare config data
            config_data = {
                "llm": {k: v.value if isinstance(v, Enum) else v for k, v in self.llm.__dict__.items()},
                "t2i": {k: v.value if isinstance(v, Enum) else v for k, v in self.t2i.__dict__.items()},
                "mllm": {k: v.value if isinstance(v, Enum) else v for k, v in self.mllm.__dict__.items()},
                "embedding": {k: v.value if isinstance(v, Enum) else v for k, v in self.embedding.__dict__.items()},
                "local": self.local.__dict__
            }
            
            # Add model registry
            registry_data = {}
            for agent_type, models in self._model_registry.items():
                registry_data[agent_type] = {}
                for use_case, config in models.items():
                    if isinstance(config, LLMConfig):
                        config_type = "llm"
                    elif isinstance(config, T2IConfig):
                        config_type = "t2i"
                    elif isinstance(config, MLLMConfig):
                        config_type = "mllm"
                    else:
                        continue
                    
                    registry_data[agent_type][use_case] = {
                        "type": config_type,
                        "config": {k: v.value if isinstance(v, Enum) else v 
                                 for k, v in config.__dict__.items() 
                                 if not k.endswith('_key')}
                    }
            
            config_data["registry"] = registry_data
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Saved model configuration to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model configuration: {e}")
            return False
