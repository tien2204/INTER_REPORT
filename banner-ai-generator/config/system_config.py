"""
System Configuration

Manages system-wide configuration settings.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "sqlite:///./banner_generator.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30


@dataclass 
class RedisConfig:
    """Redis configuration"""
    url: str = "redis://localhost:6379"
    max_connections: int = 10
    retry_on_timeout: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = "your-secret-key-here"
    jwt_secret: str = "your-jwt-secret-here"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"])


@dataclass
class FileConfig:
    """File handling configuration"""
    upload_dir: str = "uploads"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list = field(default_factory=lambda: ["png", "jpg", "jpeg", "svg", "pdf"])
    temp_dir: str = "temp"
    output_dir: str = "output"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    prometheus_port: int = 8090
    health_check_interval: int = 30
    metrics_collection_interval: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "memory_usage": 0.8,
        "cpu_usage": 0.8,
        "queue_size": 1000,
        "error_rate": 0.05
    })


@dataclass
class WorkflowConfig:
    """Workflow configuration"""
    max_iterations: int = 5
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay: int = 5
    parallel_processing: bool = True
    max_concurrent_workflows: int = 10


class SystemConfig:
    """
    System configuration manager
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._load_from_environment()
        
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        # Database configuration
        self.database = DatabaseConfig(
            url=os.getenv("DATABASE_URL", "sqlite:///./banner_generator.db"),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
        )
        
        # Redis configuration
        self.redis = RedisConfig(
            url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "10")),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
        )
        
        # Security configuration
        self.security = SecurityConfig(
            secret_key=os.getenv("SECRET_KEY", "your-secret-key-here"),
            jwt_secret=os.getenv("JWT_SECRET", "your-jwt-secret-here"),
            access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        )
        
        # File configuration
        self.file = FileConfig(
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024))),
            temp_dir=os.getenv("TEMP_DIR", "temp"),
            output_dir=os.getenv("OUTPUT_DIR", "output")
        )
        
        # Logging configuration
        self.logging = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file_path=os.getenv("LOG_FILE_PATH"),
            max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024)))
        )
        
        # Monitoring configuration
        self.monitoring = MonitoringConfig(
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "8090")),
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            metrics_collection_interval=int(os.getenv("METRICS_COLLECTION_INTERVAL", "60"))
        )
        
        # Workflow configuration
        self.workflow = WorkflowConfig(
            max_iterations=int(os.getenv("MAX_ITERATIONS", "5")),
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "300")),
            retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "3")),
            max_concurrent_workflows=int(os.getenv("MAX_CONCURRENT_WORKFLOWS", "10"))
        )
        
        # Application configuration
        self.app = {
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000")),
            "max_workers": int(os.getenv("MAX_WORKERS", "4")),
            "environment": os.getenv("ENVIRONMENT", "development")
        }

    def get(self, key: str, default=None):
        """
        Allow dict-like access to config values.
        Supports top-level attributes (e.g., 'app', 'database', 'redis', ...)
        """
        if hasattr(self, key):
            return getattr(self, key)
        return default

    
    def _load_from_file(self, config_file: str):
        """Load configuration from file"""
        # Implementation for loading from YAML/JSON config file
        # This can be extended based on requirements
        pass
    
    def get_database_url(self) -> str:
        """Get database URL"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis URL"""
        return self.redis.url
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.app["environment"] == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.app["environment"] == "production"
    
    def get_upload_directory(self) -> Path:
        """Get upload directory path"""
        upload_dir = Path(self.file.upload_dir)
        upload_dir.mkdir(exist_ok=True)
        return upload_dir
    
    def get_temp_directory(self) -> Path:
        """Get temporary directory path"""
        temp_dir = Path(self.file.temp_dir)
        temp_dir.mkdir(exist_ok=True)
        return temp_dir
    
    def get_output_directory(self) -> Path:
        """Get output directory path"""
        output_dir = Path(self.file.output_dir)
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def validate_configuration(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate required settings
        if self.security.secret_key == "your-secret-key-here":
            errors.append("SECRET_KEY must be set to a secure value")
        
        if self.security.jwt_secret == "your-jwt-secret-here":
            errors.append("JWT_SECRET must be set to a secure value")
        
        # Validate file size limits
        if self.file.max_file_size <= 0:
            errors.append("MAX_FILE_SIZE must be greater than 0")
        
        # Validate timeouts
        if self.workflow.timeout_seconds <= 0:
            errors.append("TIMEOUT_SECONDS must be greater than 0")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "database": self.database.__dict__,
            "redis": self.redis.__dict__,
            "security": {
                **self.security.__dict__,
                "secret_key": "***",  # Hide sensitive data
                "jwt_secret": "***"
            },
            "file": self.file.__dict__,
            "logging": self.logging.__dict__,
            "monitoring": self.monitoring.__dict__,
            "workflow": self.workflow.__dict__,
            "app": self.app
        }


# Global configuration instance
_config_instance: Optional[SystemConfig] = None


def get_system_config() -> SystemConfig:
    """Get global system configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = SystemConfig()
    return _config_instance


def init_system_config(config_file: Optional[str] = None) -> SystemConfig:
    """Initialize system configuration"""
    global _config_instance
    _config_instance = SystemConfig(config_file)
    return _config_instance
