"""
Agent configuration management
"""

import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of agents in the system"""
    STRATEGIST = "strategist"
    BACKGROUND_DESIGNER = "background_designer"
    FOREGROUND_DESIGNER = "foreground_designer"
    DEVELOPER = "developer"
    DESIGN_REVIEWER = "design_reviewer"

@dataclass
class BaseAgentConfig:
    """Base configuration for all agents"""
    agent_id: str
    agent_type: AgentType
    name: str
    enabled: bool = True
    max_concurrent_tasks: int = 5
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay: float = 1.0
    memory_limit_mb: int = 512
    log_level: str = "INFO"
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategistConfig(BaseAgentConfig):
    """Configuration for Strategist Agent"""
    # Brief analysis settings
    brief_analysis_model: str = "gpt-4"
    brief_analysis_temperature: float = 0.3
    brief_max_tokens: int = 2000
    
    # Brand analysis settings
    brand_analysis_enabled: bool = True
    color_extraction_enabled: bool = True
    logo_processing_enabled: bool = True
    
    # Target audience analysis
    audience_analysis_model: str = "gpt-4"
    audience_temperature: float = 0.4
    
    # Logo processing
    logo_padding_removal: bool = True
    logo_background_removal: bool = True
    logo_max_size_mb: int = 10
    supported_logo_formats: List[str] = field(default_factory=lambda: ["png", "jpg", "jpeg", "svg"])
    
    # Validation settings
    validate_brand_consistency: bool = True
    validate_target_alignment: bool = True
    
    def __post_init__(self):
        if not self.capabilities:
            self.capabilities = [
                "analyze_brief",
                "process_logo", 
                "extract_brand_colors",
                "analyze_target_audience",
                "validate_brand_assets"
            ]

@dataclass
class BackgroundDesignerConfig(BaseAgentConfig):
    """Configuration for Background Designer Agent"""
    # Text-to-Image model settings
    t2i_model: str = "flux.1-schnell"
    t2i_guidance_scale: float = 7.5
    t2i_num_inference_steps: int = 20
    t2i_seed: Optional[int] = None
    
    # Image generation settings
    default_width: int = 1200
    default_height: int = 628
    max_image_size: int = 2048
    image_format: str = "png"
    image_quality: int = 95
    
    # Self-refinement settings
    refinement_enabled: bool = True
    max_refinement_iterations: int = 3
    text_detection_enabled: bool = True
    text_detection_model: str = "gpt-4-vision-preview"
    
    # Prompt engineering
    prompt_template_path: str = "prompts/background_generation.txt"
    negative_prompt_default: str = "text, letters, words, watermark, signature"
    style_modifiers: List[str] = field(default_factory=lambda: [
        "professional", "clean", "modern", "minimalist", "gradient"
    ])
    
    # Image validation
    min_resolution: tuple = (800, 600)
    max_file_size_mb: int = 5
    allowed_formats: List[str] = field(default_factory=lambda: ["png", "jpg", "jpeg"])
    
    def __post_init__(self):
        if not self.capabilities:
            self.capabilities = [
                "generate_background",
                "validate_image",
                "detect_text",
                "refine_image",
                "resize_image"
            ]

@dataclass
class ForegroundDesignerConfig(BaseAgentConfig):
    """Configuration for Foreground Designer Agent"""
    # Blueprint generation
    blueprint_version: str = "1.0"
    blueprint_template_path: str = "templates/blueprint_template.json"
    
    # Layout settings
    layout_engine: str = "relative_positioning"
    grid_enabled: bool = True
    grid_columns: int = 12
    responsive_design: bool = True
    
    # Typography settings
    font_library_path: str = "assets/fonts/"
    default_font_family: str = "Inter"
    fallback_fonts: List[str] = field(default_factory=lambda: ["Arial", "Helvetica", "sans-serif"])
    font_size_scale: Dict[str, int] = field(default_factory=lambda: {
        "xs": 12, "sm": 14, "base": 16, "lg": 18, "xl": 20, "2xl": 24, "3xl": 30
    })
    
    # Color management
    color_palette_size: int = 5
    contrast_ratio_threshold: float = 4.5
    accessibility_check: bool = True
    
    # Component settings
    button_styles: List[str] = field(default_factory=lambda: [
        "primary", "secondary", "outline", "ghost", "gradient"
    ])
    text_alignment_options: List[str] = field(default_factory=lambda: [
        "left", "center", "right", "justify"
    ])
    
    # CTA optimization
    cta_a_b_testing: bool = True
    cta_conversion_optimization: bool = True
    
    # Validation
    validate_hierarchy: bool = True
    validate_readability: bool = True
    validate_accessibility: bool = True
    
    def __post_init__(self):
        if not self.capabilities:
            self.capabilities = [
                "create_blueprint",
                "design_layout",
                "select_typography",
                "design_cta",
                "validate_design",
                "generate_color_palette"
            ]

@dataclass
class DeveloperConfig(BaseAgentConfig):
    """Configuration for Developer Agent"""
    # Code generation settings
    code_generation_model: str = "gpt-4"
    code_temperature: float = 0.2
    code_max_tokens: int = 4000
    
    # SVG generation
    svg_version: str = "1.1"
    svg_optimization: bool = True
    svg_minification: bool = True
    embed_fonts: bool = True
    
    # Figma integration
    figma_api_enabled: bool = True
    figma_plugin_version: str = "1.0"
    figma_auto_layout: bool = True
    
    # Output formats
    supported_formats: List[str] = field(default_factory=lambda: [
        "svg", "html", "css", "figma", "sketch", "pdf"
    ])
    
    # Code quality
    code_validation: bool = True
    lint_enabled: bool = True
    minification: bool = True
    compression_enabled: bool = True
    
    # Performance settings
    optimize_images: bool = True
    lazy_loading: bool = True
    cache_assets: bool = True
    
    # Browser compatibility
    target_browsers: List[str] = field(default_factory=lambda: [
        "chrome >= 90", "firefox >= 88", "safari >= 14", "edge >= 90"
    ])
    
    def __post_init__(self):
        if not self.capabilities:
            self.capabilities = [
                "generate_svg",
                "generate_html",
                "generate_css", 
                "create_figma_plugin",
                "optimize_code",
                "validate_code"
            ]

@dataclass
class DesignReviewerConfig(BaseAgentConfig):
    """Configuration for Design Reviewer Agent"""
    # Review model settings
    review_model: str = "gpt-4-vision-preview"
    review_temperature: float = 0.4
    review_max_tokens: int = 1500
    
    # Evaluation criteria
    visual_hierarchy_weight: float = 0.25
    brand_consistency_weight: float = 0.25
    readability_weight: float = 0.25
    aesthetic_quality_weight: float = 0.25
    
    # Scoring system
    min_acceptable_score: float = 7.0
    max_score: float = 10.0
    detailed_feedback: bool = True
    
    # Analysis settings
    color_analysis: bool = True
    typography_analysis: bool = True
    layout_analysis: bool = True
    accessibility_check: bool = True
    
    # Feedback generation
    constructive_feedback: bool = True
    improvement_suggestions: bool = True
    examples_enabled: bool = True
    
    # Automation settings
    auto_approve_threshold: float = 8.5
    auto_reject_threshold: float = 5.0
    human_review_required: bool = False
    
    def __post_init__(self):
        if not self.capabilities:
            self.capabilities = [
                "review_design",
                "evaluate_hierarchy",
                "check_brand_consistency",
                "assess_readability",
                "provide_feedback",
                "score_design"
            ]

class AgentConfig:
    """Agent configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("AGENT_CONFIG_PATH", "config/agents.json")
        self._configs: Dict[str, BaseAgentConfig] = {}
        self._load_default_configs()
        if os.path.exists(self.config_path):
            self._load_configs_from_file()
    
    def _load_default_configs(self) -> None:
        """Load default configurations for all agents"""
        self._configs.update({
            "strategist": StrategistConfig(
                agent_id="strategist_001",
                agent_type=AgentType.STRATEGIST,
                name="Campaign Strategist"
            ),
            "background_designer": BackgroundDesignerConfig(
                agent_id="bg_designer_001", 
                agent_type=AgentType.BACKGROUND_DESIGNER,
                name="Background Designer"
            ),
            "foreground_designer": ForegroundDesignerConfig(
                agent_id="fg_designer_001",
                agent_type=AgentType.FOREGROUND_DESIGNER, 
                name="Foreground Designer"
            ),
            "developer": DeveloperConfig(
                agent_id="developer_001",
                agent_type=AgentType.DEVELOPER,
                name="Code Developer"
            ),
            "design_reviewer": DesignReviewerConfig(
                agent_id="reviewer_001",
                agent_type=AgentType.DESIGN_REVIEWER,
                name="Design Reviewer"
            )
        })
    
    def _load_configs_from_file(self) -> None:
        """Load configurations from file (JSON/YAML)"""
        try:
            import json
            with open(self.config_path, 'r') as f:
                file_configs = json.load(f)
            
            # Merge file configs with defaults
            for agent_key, config_data in file_configs.items():
                if agent_key in self._configs:
                    # Update existing config
                    for key, value in config_data.items():
                        if hasattr(self._configs[agent_key], key):
                            setattr(self._configs[agent_key], key, value)
                            
            logger.info(f"Loaded agent configurations from: {self.config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load config file {self.config_path}: {e}")
    
    def get_config(self, agent_type: Union[AgentType, str]) -> Optional[BaseAgentConfig]:
        """Get configuration for agent type"""
        if isinstance(agent_type, AgentType):
            agent_type = agent_type.value
        return self._configs.get(agent_type)
    
    def update_config(self, agent_type: Union[AgentType, str], **kwargs) -> bool:
        """Update agent configuration"""
        if isinstance(agent_type, AgentType):
            agent_type = agent_type.value
            
        if agent_type not in self._configs:
            return False
            
        config = self._configs[agent_type]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.debug(f"Updated {agent_type}.{key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key} for agent {agent_type}")
        
        return True
    
    def set_config(self, agent_type: Union[AgentType, str], config: BaseAgentConfig) -> None:
        """Set complete configuration for agent"""
        if isinstance(agent_type, AgentType):
            agent_type = agent_type.value
        self._configs[agent_type] = config
        logger.info(f"Set configuration for agent: {agent_type}")
    
    def list_agents(self) -> List[str]:
        """List all configured agents"""
        return list(self._configs.keys())
    
    def validate_config(self, agent_type: Union[AgentType, str]) -> bool:
        """Validate agent configuration"""
        if isinstance(agent_type, AgentType):
            agent_type = agent_type.value
            
        config = self._configs.get(agent_type)
        if not config:
            return False
        
        # Basic validation
        if not config.agent_id or not config.name:
            return False
        
        if config.max_concurrent_tasks <= 0:
            return False
            
        if config.timeout_seconds <= 0:
            return False
        
        # Agent-specific validation
        if isinstance(config, StrategistConfig):
            return self._validate_strategist_config(config)
        elif isinstance(config, BackgroundDesignerConfig):
            return self._validate_background_designer_config(config)
        elif isinstance(config, ForegroundDesignerConfig):
            return self._validate_foreground_designer_config(config)
        elif isinstance(config, DeveloperConfig):
            return self._validate_developer_config(config)
        elif isinstance(config, DesignReviewerConfig):
            return self._validate_design_reviewer_config(config)
        
        return True
    
    def _validate_strategist_config(self, config: StrategistConfig) -> bool:
        """Validate strategist-specific configuration"""
        if config.brief_analysis_temperature < 0 or config.brief_analysis_temperature > 2:
            return False
        if config.logo_max_size_mb <= 0:
            return False
        return True
    
    def _validate_background_designer_config(self, config: BackgroundDesignerConfig) -> bool:
        """Validate background designer-specific configuration"""
        if config.default_width <= 0 or config.default_height <= 0:
            return False
        if config.max_refinement_iterations <= 0:
            return False
        return True
    
    def _validate_foreground_designer_config(self, config: ForegroundDesignerConfig) -> bool:
        """Validate foreground designer-specific configuration"""
        if config.grid_columns <= 0:
            return False
        if config.contrast_ratio_threshold <= 0:
            return False
        return True
    
    def _validate_developer_config(self, config: DeveloperConfig) -> bool:
        """Validate developer-specific configuration"""
        if config.code_temperature < 0 or config.code_temperature > 2:
            return False
        return True
    
    def _validate_design_reviewer_config(self, config: DesignReviewerConfig) -> bool:
        """Validate design reviewer-specific configuration"""
        weights = [
            config.visual_hierarchy_weight,
            config.brand_consistency_weight,
            config.readability_weight,
            config.aesthetic_quality_weight
        ]
        if abs(sum(weights) - 1.0) > 0.001:  # Should sum to 1.0
            return False
        return True
    
    def save_configs(self, path: Optional[str] = None) -> bool:
        """Save current configurations to file"""
        try:
            import json
            save_path = path or self.config_path
            
            # Convert configs to serializable format
            serializable_configs = {}
            for agent_type, config in self._configs.items():
                config_dict = {}
                for key, value in config.__dict__.items():
                    if isinstance(value, Enum):
                        config_dict[key] = value.value
                    else:
                        config_dict[key] = value
                serializable_configs[agent_type] = config_dict
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(serializable_configs, f, indent=2, default=str)
            
            logger.info(f"Saved agent configurations to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configurations: {e}")
            return False
