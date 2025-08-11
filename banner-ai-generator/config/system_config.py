"""
System-wide configuration management
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    # SQLite settings (default)
    sqlite_path: str = "data/banner_ai.db"
    sqlite_timeout: int = 30
    sqlite_check_same_thread: bool = False
    
    # PostgreSQL settings (optional)
    postgres_enabled: bool = False
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "banner_ai"
    postgres_username: str = ""
    postgres_password: str = ""
    postgres_ssl_mode: str = "prefer"
    
    # Connection pooling
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Backup settings
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    backup_path: str = "backups/"
    
    def get_database_url(self) -> str:
        """Get database URL for connection"""
        if self.postgres_enabled:
            return f"postgresql://{self.postgres_username}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        else:
            return f"sqlite:///{self.sqlite_path}"

@dataclass
class CacheConfig:
    """Caching configuration"""
    # Redis settings
    redis_enabled: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    
    # In-memory cache settings
    memory_cache_enabled: bool = True
    memory_cache_size_mb: int = 256
    memory_cache_ttl: int = 3600  # seconds
    
    # Cache policies
    default_ttl: int = 1800  # 30 minutes
    design_cache_ttl: int = 7200  # 2 hours
    image_cache_ttl: int = 86400  # 24 hours
    feedback_cache_ttl: int = 3600  # 1 hour
    
    # Cache prefixes
    cache_prefix: str = "banner_ai:"
    session_prefix: str = "session:"
    design_prefix: str = "design:"
    image_prefix: str = "image:"
    
    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL for connection"""
        if not self.redis_enabled:
            return None
        
        auth = f":{self.redis_password}@" if self.redis_password else ""
        protocol = "rediss" if self.redis_ssl else "redis"
        return f"{protocol}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

@dataclass
class SecurityConfig:
    """Security configuration"""
    # Authentication
    auth_enabled: bool = True
    jwt_secret_key: str = "your-super-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    
    # API security
    api_key_required: bool = False
    api_rate_limiting: bool = True
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 1000
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    cors_headers: List[str] = field(default_factory=lambda: ["Content-Type", "Authorization"])
    
    # File upload security
    max_file_size_mb: int = 10
    allowed_file_types: List[str] = field(default_factory=lambda: [
        "image/png", "image/jpeg", "image/jpg", "image/svg+xml", "application/json"
    ])
    scan_uploads: bool = True
    
    # Content security
    content_filtering: bool = True
    xss_protection: bool = True
    sql_injection_protection: bool = True
    
    # Encryption
    encrypt_sensitive_data: bool = True
    encryption_key: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate security configuration"""
        if self.auth_enabled and len(self.jwt_secret_key) < 32:
            logger.warning("JWT secret key should be at least 32 characters long")
            return False
        
        if self.max_file_size_mb > 50:
            logger.warning("Max file size is very large, consider security implications")
        
        return True

@dataclass
class LoggingConfig:
    """Logging configuration"""
    # General logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_to_console: bool = True
    
    # File logging
    log_file_path: str = "logs/banner_ai.log"
    log_file_max_size_mb: int = 10
    log_file_backup_count: int = 5
    log_file_rotation: bool = True
    
    # Structured logging
    structured_logging: bool = False
    json_logging: bool = False
    
    # Component-specific logging
    agent_log_level: str = "INFO"
    api_log_level: str = "INFO"
    database_log_level: str = "WARNING"
    
    # Performance logging
    performance_logging: bool = True
    slow_query_threshold: float = 1.0  # seconds
    request_logging: bool = True
    
    # Security logging
    security_logging: bool = True
    audit_logging: bool = True

@dataclass
class PerformanceConfig:
    """Performance and optimization configuration"""
    # Threading
    max_worker_threads: int = 10
    thread_pool_size: int = 5
    
    # Processing limits
    max_concurrent_requests: int = 50
    request_timeout_seconds: int = 300
    max_queue_size: int = 1000
    
    # Memory management
    memory_limit_mb: int = 2048
    gc_threshold: int = 700
    memory_profiling: bool = False
    
    # Image processing
    image_processing_threads: int = 2
    max_image_dimensions: tuple = (4096, 4096)
    image_compression_quality: int = 85
    
    # AI model optimization
    model_cache_enabled: bool = True
    model_quantization: bool = False
    gpu_acceleration: bool = True
    batch_processing: bool = True
    
    # Caching strategy
    aggressive_caching: bool = False
    cache_prewarming: bool = True
    
    # Monitoring
    metrics_enabled: bool = True
    health_checks: bool = True
    profiling_enabled: bool = False

@dataclass
class IntegrationConfig:
    """External integration configuration"""
    # OpenAI API
    openai_api_key: Optional[str] = None
    openai_org_id: Optional[str] = None
    openai_timeout: int = 60
    openai_max_retries: int = 3
    
    # Other LLM providers
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_key: Optional[str] = None
    
    # Text-to-Image services
    replicate_api_key: Optional[str] = None
    stability_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    
    # Figma integration
    figma_api_token: Optional[str] = None
    figma_team_id: Optional[str] = None
    
    # Cloud storage
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket: Optional[str] = None
    
    # Google Cloud
    gcp_credentials_path: Optional[str] = None
    gcs_bucket: Optional[str] = None
    
    # Monitoring services
    sentry_dsn: Optional[str] = None
    datadog_api_key: Optional[str] = None
    newrelic_license_key: Optional[str] = None

class SystemConfig:
    """System configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("SYSTEM_CONFIG_PATH", "config/system.json")
        
        # Initialize with defaults
        self.database = DatabaseConfig()
        self.cache = CacheConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
        self.performance = PerformanceConfig()
        self.integration = IntegrationConfig()
        
        # Load from environment variables
        self._load_from_env()
        
        # Load from config file if exists
        if os.path.exists(self.config_path):
            self._load_from_file()
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables"""
        # Database
        if os.getenv("DATABASE_URL"):
            self.database.postgres_enabled = True
            # Parse DATABASE_URL if needed
        self.database.sqlite_path = os.getenv("SQLITE_PATH", self.database.sqlite_path)
        
        # Cache
        self.cache.redis_enabled = os.getenv("REDIS_ENABLED", "false").lower() == "true"
        self.cache.redis_host = os.getenv("REDIS_HOST", self.cache.redis_host)
        self.cache.redis_port = int(os.getenv("REDIS_PORT", str(self.cache.redis_port)))
        self.cache.redis_password = os.getenv("REDIS_PASSWORD")
        
        # Security
        self.security.jwt_secret_key = os.getenv("JWT_SECRET_KEY", self.security.jwt_secret_key)
        self.security.auth_enabled = os.getenv("AUTH_ENABLED", "true").lower() == "true"
        
        # API Keys
        self.integration.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.integration.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.integration.replicate_api_key = os.getenv("REPLICATE_API_TOKEN")
        self.integration.figma_api_token = os.getenv("FIGMA_API_TOKEN")
        
        # Logging
        self.logging.log_level = os.getenv("LOG_LEVEL", self.logging.log_level)
        
        logger.debug("Loaded configuration from environment variables")
    
    def _load_from_file(self) -> None:
        """Load configuration from JSON file"""
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
                            setattr(section_config, key, value)
            
            logger.info(f"Loaded system configuration from: {self.config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load config file {self.config_path}: {e}")
    
    def _validate_config(self) -> None:
        """Validate system configuration"""
        # Validate security config
        if not self.security.validate():
            logger.error("Security configuration validation failed")
        
        # Validate directories exist
        os.makedirs(os.path.dirname(self.database.sqlite_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.logging.log_file_path), exist_ok=True)
        os.makedirs(self.database.backup_path, exist_ok=True)
        
        # Validate API keys for enabled services
        if self.cache.redis_enabled and not self.cache.redis_host:
            logger.warning("Redis enabled but no host configured")
        
        if not self.integration.openai_api_key:
            logger.warning("No OpenAI API key configured")
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return self.database
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration"""
        return self.cache
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return self.security
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        return self.logging
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration"""
        return self.performance
    
    def get_integration_config(self) -> IntegrationConfig:
        """Get integration configuration"""
        return self.integration
    
    def update_config(self, section: str, **kwargs) -> bool:
        """Update configuration section"""
        if not hasattr(self, section):
            logger.error(f"Unknown configuration section: {section}")
            return False
        
        section_config = getattr(self, section)
        for key, value in kwargs.items():
            if hasattr(section_config, key):
                setattr(section_config, key, value)
                logger.debug(f"Updated {section}.{key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key} in section {section}")
        
        return True
    
    def save_config(self, path: Optional[str] = None) -> bool:
        """Save current configuration to file"""
        try:
            import json
            save_path = path or self.config_path
            
            config_data = {
                'database': self.database.__dict__,
                'cache': self.cache.__dict__,
                'security': {k: v for k, v in self.security.__dict__.items() 
                           if not k.startswith('jwt_secret') and not k.startswith('encryption_key')},
                'logging': self.logging.__dict__,
                'performance': self.performance.__dict__,
                'integration': {k: v for k, v in self.integration.__dict__.items() 
                              if not k.endswith('_key') and not k.endswith('_token')}
            }
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Saved system configuration to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for monitoring"""
        return {
            'database': {
                'type': 'postgres' if self.database.postgres_enabled else 'sqlite',
                'path': self.database.sqlite_path if not self.database.postgres_enabled else 'postgres',
                'backup_enabled': self.database.backup_enabled
            },
            'cache': {
                'redis_enabled': self.cache.redis_enabled,
                'memory_cache_enabled': self.cache.memory_cache_enabled,
                'memory_cache_size_mb': self.cache.memory_cache_size_mb
            },
            'security': {
                'auth_enabled': self.security.auth_enabled,
                'rate_limiting': self.security.api_rate_limiting,
                'cors_enabled': self.security.cors_enabled
            },
            'performance': {
                'max_worker_threads': self.performance.max_worker_threads,
                'max_concurrent_requests': self.performance.max_concurrent_requests,
                'gpu_acceleration': self.performance.gpu_acceleration
            },
            'integrations': {
                'openai_configured': bool(self.integration.openai_api_key),
                'figma_configured': bool(self.integration.figma_api_token),
                'cloud_storage_configured': bool(self.integration.s3_bucket or self.integration.gcs_bucket)
            }
        }
