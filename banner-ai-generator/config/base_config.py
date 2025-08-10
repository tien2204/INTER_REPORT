import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import timedelta
import json

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    # Connection settings
    HOST: str = "localhost"
    PORT: int = 5432
    NAME: str = "banner_generator"
    USER: str = "banner_user"
    PASSWORD: str = ""
    
    # Connection pool settings
    MIN_CONNECTIONS: int = 1
    MAX_CONNECTIONS: int = 10
    CONNECTION_TIMEOUT: int = 30
    
    # Redis settings (for caching and sessions)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    
    # Performance settings
    ENABLE_QUERY_LOGGING: bool = False
    CONNECTION_POOL_SIZE: int = 5
    STATEMENT_TIMEOUT: int = 300

@dataclass
class ModelConfig:
    """AI Model configuration settings"""
    # Language Models
    DEFAULT_LLM_MODEL: str = "gpt-4"
    DEFAULT_LLM_API_KEY: str = ""
    DEFAULT_LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.7
    LLM_TIMEOUT: int = 120
    
    # Text-to-Image Models
    DEFAULT_T2I_MODEL: str = "flux-1-schnell"
    T2I_API_KEY: str = ""
    T2I_BASE_URL: str = ""
    T2I_MAX_SIZE: int = 1024
    T2I_STEPS: int = 4
    T2I_GUIDANCE_SCALE: float = 7.5
    T2I_TIMEOUT: int = 300
    
    # Multimodal Models (for image analysis)
    DEFAULT_MLLM_MODEL: str = "gpt-4-vision-preview"
    MLLM_API_KEY: str = ""
    MLLM_BASE_URL: str = "https://api.openai.com/v1"
    MLLM_MAX_TOKENS: int = 2048
    MLLM_TIMEOUT: int = 60
    
    # Model fallbacks and retries
    MODEL_RETRY_ATTEMPTS: int = 3
    MODEL_RETRY_DELAY: float = 1.0
    ENABLE_MODEL_FALLBACK: bool = True
    
    # Rate limiting
    LLM_REQUESTS_PER_MINUTE: int = 60
    T2I_REQUESTS_PER_MINUTE: int = 10
    MLLM_REQUESTS_PER_MINUTE: int = 30

@dataclass 
class AgentConfig:
    """Agent-specific configuration settings"""
    # General agent settings
    MAX_RETRIES: int = 3
    TIMEOUT_SECONDS: int = 300
    ENABLE_PARALLEL_EXECUTION: bool = False
    
    # Strategist Agent
    STRATEGIST_ANALYSIS_DEPTH: str = "detailed"  # basic, detailed, comprehensive
    STRATEGIST_BRAND_ANALYSIS: bool = True
    STRATEGIST_COMPETITOR_ANALYSIS: bool = False
    
    # Background Designer Agent  
    BACKGROUND_MAX_ITERATIONS: int = 5
    BACKGROUND_TEXT_CHECK_ENABLED: bool = True
    BACKGROUND_AUTO_RESIZE: bool = True
    BACKGROUND_QUALITY_THRESHOLD: float = 0.8
    
    # Foreground Designer Agent
    FOREGROUND_LAYOUT_ALGORITHM: str = "grid_based"  # grid_based, free_form, template_based
    FOREGROUND_AUTO_SPACING: bool = True
    FOREGROUND_RESPONSIVE_DESIGN: bool = True
    FOREGROUND_COLOR_HARMONY_CHECK: bool = True
    
    # Developer Agent
    DEVELOPER_OUTPUT_FORMATS: List[str] = field(default_factory=lambda: ["svg", "figma"])
    DEVELOPER_CODE_OPTIMIZATION: bool = True
    DEVELOPER_MINIFY_OUTPUT: bool = False
    
    # Design Reviewer Agent
    REVIEWER_ENABLED: bool = True
    REVIEWER_STRICT_MODE: bool = False
    REVIEWER_AUTO_APPROVE_THRESHOLD: float = 8.5
    REVIEWER_MAX_FEEDBACK_LENGTH: int = 500

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    # API Security
    ENABLE_AUTHENTICATION: bool = True
    ENABLE_RATE_LIMITING: bool = True
    API_KEY_REQUIRED: bool = False
    JWT_SECRET_KEY: str = ""
    JWT_EXPIRATION_HOURS: int = 24
    
    # File Upload Security
    ALLOWED_FILE_EXTENSIONS: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".svg", ".pdf"])
    MAX_FILE_SIZE_MB: int = 50
    SCAN_UPLOADS_FOR_MALWARE: bool = False
    QUARANTINE_SUSPICIOUS_FILES: bool = True
    
    # Data Protection
    ENCRYPT_SENSITIVE_DATA: bool = False
    ENCRYPTION_KEY: str = ""
    ENABLE_AUDIT_LOGGING: bool = True
    
    # Network Security
    ALLOWED_ORIGINS: List[str] = field(default_factory=lambda: ["*"])
    ENABLE_CORS: bool = True
    TRUST_PROXY: bool = False

@dataclass 
class PerformanceConfig:
    """Performance and optimization settings"""
    # Memory Management
    MAX_MEMORY_SIZE_MB: int = 1024
    MEMORY_WARNING_THRESHOLD: float = 0.8
    ENABLE_MEMORY_MONITORING: bool = True
    AUTO_CLEANUP_EXPIRED_SESSIONS: bool = True
    
    # Caching
    ENABLE_CACHING: bool = True
    CACHE_TTL_SECONDS: int = 3600
    CACHE_MAX_SIZE_MB: int = 256
    
    # Processing
    MAX_CONCURRENT_CAMPAIGNS: int = 10
    WORKER_POOL_SIZE: int = 4
    QUEUE_MAX_SIZE: int = 100
    
    # File Processing
    IMAGE_PROCESSING_QUALITY: int = 95
    ENABLE_IMAGE_COMPRESSION: bool = True
    THUMBNAIL_GENERATION: bool = True

@dataclass
class BaseConfig:
    """
    Main configuration class combining all configuration aspects
    
    This class serves as the central configuration object that aggregates
    all configuration categories and provides unified access to settings.
    """
    
    # Basic System Settings
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    VERSION: str = "1.0.0"
    
    # Core System Settings
    SESSION_TIMEOUT_HOURS: int = 24
    MAX_ITERATIONS: int = 5
    ENABLE_ITERATIVE_REFINEMENT: bool = True
    
    # Directory Settings
    BASE_DIR: str = "."
    UPLOAD_DIR: str = "./uploads"
    OUTPUT_DIR: str = "./output"
    TEMP_DIR: str = "./temp"
    LOGS_DIR: str = "./logs"
    EXPORTS_DIR: str = "./exports"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_TITLE: str = "Banner AI Generator API"
    API_VERSION: str = "v1"
    
    # Nested configuration objects
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Custom settings (for extensibility)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls, prefix: str = "") -> 'BaseConfig':
        """
        Create configuration from environment variables
        
        Args:
            prefix: Optional prefix for environment variables (e.g., "BANNER_")
        """
        config = cls()
        
        # Load main config fields
        config._load_main_fields_from_env(prefix)
        
        # Load nested config objects
        config.database = config._load_nested_config_from_env(DatabaseConfig, prefix + "DB_")
        config.models = config._load_nested_config_from_env(ModelConfig, prefix + "MODEL_")
        config.agents = config._load_nested_config_from_env(AgentConfig, prefix + "AGENT_")
        config.security = config._load_nested_config_from_env(SecurityConfig, prefix + "SECURITY_")
        config.performance = config._load_nested_config_from_env(PerformanceConfig, prefix + "PERF_")
        
        return config
    
    def _load_main_fields_from_env(self, prefix: str) -> None:
        """Load main configuration fields from environment variables"""
        main_fields = {
            'DEBUG', 'ENVIRONMENT', 'LOG_LEVEL', 'VERSION',
            'SESSION_TIMEOUT_HOURS', 'MAX_ITERATIONS', 'ENABLE_ITERATIVE_REFINEMENT',
            'BASE_DIR', 'UPLOAD_DIR', 'OUTPUT_DIR', 'TEMP_DIR', 'LOGS_DIR', 'EXPORTS_DIR',
            'API_HOST', 'API_PORT', 'API_WORKERS', 'API_TITLE', 'API_VERSION'
        }
        
        for field_name in main_fields:
            env_key = f"{prefix}{field_name}"
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                self._set_field_value(field_name, env_value)
    
    def _load_nested_config_from_env(self, config_class: type, prefix: str):
        """Load nested configuration object from environment variables"""
        config_instance = config_class()
        
        for field_name in config_class.__annotations__:
            env_key = f"{prefix}{field_name}"
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                field_type = config_class.__annotations__[field_name]
                converted_value = self._convert_env_value(env_value, field_type)
                setattr(config_instance, field_name, converted_value)
        
        return config_instance
    
    def _set_field_value(self, field_name: str, env_value: str) -> None:
        """Set field value with proper type conversion"""
        if hasattr(self, field_name):
            field_type = self.__annotations__.get(field_name, str)
            converted_value = self._convert_env_value(env_value, field_type)
            setattr(self, field_name, converted_value)
    
    def _convert_env_value(self, env_value: str, field_type: type) -> Any:
        """Convert environment variable string to appropriate type"""
        try:
            if field_type == bool:
                return env_value.lower() in ('true', '1', 'yes', 'on', 'enabled')
            elif field_type == int:
                return int(env_value)
            elif field_type == float:
                return float(env_value)
            elif field_type == List[str]:
                # Parse comma-separated values
                return [item.strip() for item in env_value.split(',') if item.strip()]
            elif field_type == str:
                return env_value
            else:
                # For complex types, try JSON parsing
                try:
                    return json.loads(env_value)
                except:
                    return env_value
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert {env_value} to {field_type}: {e}")
            return env_value
    
    @classmethod
    def from_file(cls, config_path: str, merge_env: bool = True) -> 'BaseConfig':
        """
        Load configuration from JSON/YAML file
        
        Args:
            config_path: Path to configuration file
            merge_env: Whether to merge with environment variables
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load from file
        if config_file.suffix.lower() == '.json':
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        elif config_file.suffix.lower() in ['.yml', '.yaml']:
            try:
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML configuration files")
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
        
        # Create config object
        config = cls()
        config._apply_config_dict(config_data)
        
        # Merge with environment variables if requested
        if merge_env:
            env_config = cls.from_env()
            config._merge_with_env_config(env_config)
        
        return config
    
    def _apply_config_dict(self, config_data: Dict[str, Any]) -> None:
        """Apply configuration data from dictionary"""
        for key, value in config_data.items():
            if hasattr(self, key):
                if key in ['database', 'models', 'agents', 'security', 'performance']:
                    # Handle nested configuration objects
                    nested_config = getattr(self, key)
                    if isinstance(value, dict):
                        for nested_key, nested_value in value.items():
                            if hasattr(nested_config, nested_key):
                                setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(self, key, value)
            else:
                # Store in custom_settings for unknown keys
                self.custom_settings[key] = value
    
    def _merge_with_env_config(self, env_config: 'BaseConfig') -> None:
        """Merge with environment-based configuration"""
        # Only merge fields that are explicitly set in environment
        env_vars = os.environ
        
        for field_name in self.__annotations__:
            if field_name.upper() in env_vars:
                setattr(self, field_name, getattr(env_config, field_name))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        
        # Add main fields
        for field_name in self.__annotations__:
            if not field_name.startswith('_'):
                value = getattr(self, field_name)
                if hasattr(value, '__dict__'):
                    # Convert nested dataclass to dict
                    result[field_name] = {
                        k: v for k, v in value.__dict__.items()
                        if not k.startswith('_')
                    }
                else:
                    result[field_name] = value
        
        return result
    
    def save_to_file(self, config_path: str, format: str = 'json') -> bool:
        """
        Save configuration to file
        
        Args:
            config_path: Output file path
            format: File format ('json' or 'yaml')
        """
        try:
            config_dict = self.to_dict()
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
            elif format.lower() in ['yml', 'yaml']:
                try:
                    import yaml
                    with open(config_file, 'w', encoding='utf-8') as f:
                        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML output")
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def validate(self) -> List[str]:
        """Basic configuration validation"""
        errors = []
        
        # Validate basic fields
        if self.SESSION_TIMEOUT_HOURS <= 0:
            errors.append("SESSION_TIMEOUT_HOURS must be positive")
        
        if self.MAX_ITERATIONS <= 0:
            errors.append("MAX_ITERATIONS must be positive")
        
        if self.API_PORT <= 0 or self.API_PORT > 65535:
            errors.append("API_PORT must be between 1 and 65535")
        
        if self.API_WORKERS <= 0:
            errors.append("API_WORKERS must be positive")
        
        # Validate directories
        directory_fields = ['UPLOAD_DIR', 'OUTPUT_DIR', 'TEMP_DIR', 'LOGS_DIR', 'EXPORTS_DIR']
        for dir_field in directory_fields:
            dir_path = getattr(self, dir_field)
            if not dir_path or len(dir_path.strip()) == 0:
                errors.append(f"{dir_field} cannot be empty")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.LOG_LEVEL not in valid_log_levels:
            errors.append(f"LOG_LEVEL must be one of: {valid_log_levels}")
        
        # Validate environment
        valid_environments = ['development', 'staging', 'production']
        if self.ENVIRONMENT not in valid_environments:
            errors.append(f"ENVIRONMENT must be one of: {valid_environments}")
        
        return errors
    
    def setup_directories(self) -> bool:
        """Create all necessary directories"""
        try:
            directory_fields = ['UPLOAD_DIR', 'OUTPUT_DIR', 'TEMP_DIR', 'LOGS_DIR', 'EXPORTS_DIR']
            
            for dir_field in directory_fields:
                dir_path = Path(getattr(self, dir_field))
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Create .gitkeep file for empty directories
                gitkeep_file = dir_path / '.gitkeep'
                if not gitkeep_file.exists():
                    gitkeep_file.touch()
            
            return True
        except Exception as e:
            print(f"Error setting up directories: {e}")
            return False
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        if self.database.PASSWORD:
            return f"postgresql://{self.database.USER}:{self.database.PASSWORD}@{self.database.HOST}:{self.database.PORT}/{self.database.NAME}"
        else:
            return f"postgresql://{self.database.USER}@{self.database.HOST}:{self.database.PORT}/{self.database.NAME}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.database.REDIS_PASSWORD:
            return f"redis://:{self.database.REDIS_PASSWORD}@{self.database.REDIS_HOST}:{self.database.REDIS_PORT}/{self.database.REDIS_DB}"
        else:
            return f"redis://{self.database.REDIS_HOST}:{self.database.REDIS_PORT}/{self.database.REDIS_DB}"
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'level': self.LOG_LEVEL,
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard' if not self.DEBUG else 'detailed'
                },
                'file': {
                    'level': 'INFO',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': f'{self.LOGS_DIR}/app.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'formatter': 'detailed'
                }
            },
            'loggers': {
                '': {  # Root logger
                    'handlers': ['console', 'file'],
                    'level': self.LOG_LEVEL,
                    'propagate': False
                }
            }
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT.lower() == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT.lower() == 'development'
    
    def is_staging(self) -> bool:
        """Check if running in staging environment"""
        return self.ENVIRONMENT.lower() == 'staging'
