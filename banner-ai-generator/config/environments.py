"""
Environment-specific configuration management
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from .agent_config import AgentConfig
from .system_config import SystemConfig
from .model_config import ModelConfig

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration container"""
    environment: Environment
    debug: bool
    agent_config: AgentConfig
    system_config: SystemConfig
    model_config: ModelConfig
    
    def __post_init__(self):
        """Post-initialization setup"""
        self._apply_environment_overrides()
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides"""
        if self.environment == Environment.DEVELOPMENT:
            self._apply_development_config()
        elif self.environment == Environment.TESTING:
            self._apply_testing_config()
        elif self.environment == Environment.STAGING:
            self._apply_staging_config()
        elif self.environment == Environment.PRODUCTION:
            self._apply_production_config()
    
    def _apply_development_config(self) -> None:
        """Apply development environment settings"""
        self.debug = True
        
        # System config
        self.system_config.logging.log_level = "DEBUG"
        self.system_config.logging.log_to_console = True
        self.system_config.security.auth_enabled = False
        self.system_config.security.cors_origins = ["*"]
        self.system_config.cache.memory_cache_enabled = True
        self.system_config.cache.redis_enabled = False
        self.system_config.performance.metrics_enabled = True
        self.system_config.performance.profiling_enabled = True
        
        # Agent config - faster timeouts for development
        for agent_type in ["strategist", "background_designer", "foreground_designer", "developer", "design_reviewer"]:
            config = self.agent_config.get_config(agent_type)
            if config:
                config.timeout_seconds = 60
                config.log_level = "DEBUG"
        
        # Model config - use faster/cheaper models for development
        self.model_config.llm.model_name = "gpt-3.5-turbo"
        self.model_config.llm.temperature = 0.5
        self.model_config.llm.use_cache = True
        self.model_config.t2i.num_inference_steps = 10  # Faster generation
        
        logger.info("Applied development environment configuration")
    
    def _apply_testing_config(self) -> None:
        """Apply testing environment settings"""
        self.debug = True
        
        # System config
        self.system_config.logging.log_level = "INFO"
        self.system_config.logging.log_to_file = False
        self.system_config.database.sqlite_path = ":memory:"  # In-memory database
        self.system_config.cache.memory_cache_enabled = True
        self.system_config.cache.redis_enabled = False
        self.system_config.security.auth_enabled = False
        self.system_config.performance.metrics_enabled = False
        
        # Agent config - very fast timeouts for testing
        for agent_type in ["strategist", "background_designer", "foreground_designer", "developer", "design_reviewer"]:
            config = self.agent_config.get_config(agent_type)
            if config:
                config.timeout_seconds = 30
                config.max_concurrent_tasks = 2
                config.retry_attempts = 1
        
        # Model config - use mock/cached responses
        self.model_config.llm.use_cache = True
        self.model_config.llm.cache_ttl = 86400  # Long cache for testing
        self.model_config.t2i.use_cache = True
        self.model_config.mllm.use_cache = True
        
        logger.info("Applied testing environment configuration")
    
    def _apply_staging_config(self) -> None:
        """Apply staging environment settings"""
        self.debug = False
        
        # System config
        self.system_config.logging.log_level = "INFO"
        self.system_config.logging.log_to_file = True
        self.system_config.security.auth_enabled = True
        self.system_config.security.api_rate_limiting = True
        self.system_config.security.rate_limit_per_minute = 200
        self.system_config.cache.redis_enabled = True
        self.system_config.performance.metrics_enabled = True
        self.system_config.performance.profiling_enabled = False
        
        # Agent config - production-like but more lenient
        for agent_type in ["strategist", "background_designer", "foreground_designer", "developer", "design_reviewer"]:
            config = self.agent_config.get_config(agent_type)
            if config:
                config.timeout_seconds = 180
                config.max_concurrent_tasks = 3
                config.retry_attempts = 2
        
        # Model config - production models but with cost optimization
        self.model_config.llm.model_name = "gpt-4"
        self.model_config.llm.use_cache = True
        self.model_config.llm.cache_ttl = 3600
        self.model_config.t2i.use_cache = True
        self.model_config.mllm.use_cache = True
        
        logger.info("Applied staging environment configuration")
    
    def _apply_production_config(self) -> None:
        """Apply production environment settings"""
        self.debug = False
        
        # System config
        self.system_config.logging.log_level = "INFO"
        self.system_config.logging.log_to_file = True
        self.system_config.logging.structured_logging = True
        self.system_config.logging.security_logging = True
        self.system_config.logging.audit_logging = True
        
        self.system_config.security.auth_enabled = True
        self.system_config.security.api_rate_limiting = True
        self.system_config.security.rate_limit_per_minute = 100
        self.system_config.security.rate_limit_per_hour = 1000
        self.system_config.security.content_filtering = True
        self.system_config.security.encrypt_sensitive_data = True
        
        self.system_config.cache.redis_enabled = True
        self.system_config.cache.memory_cache_enabled = True
        
        self.system_config.performance.metrics_enabled = True
        self.system_config.performance.profiling_enabled = False
        self.system_config.performance.max_concurrent_requests = 100
        
        self.system_config.database.postgres_enabled = True
        self.system_config.database.backup_enabled = True
        self.system_config.database.backup_interval_hours = 12
        
        # Agent config - production settings
        for agent_type in ["strategist", "background_designer", "foreground_designer", "developer", "design_reviewer"]:
            config = self.agent_config.get_config(agent_type)
            if config:
                config.timeout_seconds = 300
                config.max_concurrent_tasks = 5
                config.retry_attempts = 3
                config.log_level = "INFO"
        
        # Model config - full production models
        self.model_config.llm.model_name = "gpt-4"
        self.model_config.llm.use_cache = True
        self.model_config.llm.cache_ttl = 1800
        self.model_config.t2i.use_cache = True
        self.model_config.t2i.cache_ttl = 86400
        self.model_config.mllm.use_cache = True
        
        logger.info("Applied production environment configuration")

class ConfigManager:
    """Centralized configuration manager for all environments"""
    
    _instance = None
    _config: Optional[EnvironmentConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._initialize_config()
    
    def _initialize_config(self) -> None:
        """Initialize configuration based on environment"""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        
        try:
            environment = Environment(env_name)
        except ValueError:
            logger.warning(f"Unknown environment '{env_name}', defaulting to development")
            environment = Environment.DEVELOPMENT
        
        # Load base configurations
        agent_config = AgentConfig()
        system_config = SystemConfig()
        model_config = ModelConfig()
        
        # Create environment config
        self._config = EnvironmentConfig(
            environment=environment,
            debug=environment in [Environment.DEVELOPMENT, Environment.TESTING],
            agent_config=agent_config,
            system_config=system_config,
            model_config=model_config
        )
        
        logger.info(f"Initialized configuration for environment: {environment.value}")
    
    def get_config(self) -> EnvironmentConfig:
        """Get current environment configuration"""
        if self._config is None:
            self._initialize_config()
        return self._config
    
    def reload_config(self) -> None:
        """Reload configuration from files and environment"""
        self._config = None
        self._initialize_config()
        logger.info("Configuration reloaded")
    
    def switch_environment(self, environment: Environment) -> None:
        """Switch to different environment configuration"""
        if self._config is None:
            self._initialize_config()
        
        self._config.environment = environment
        self._config._apply_environment_overrides()
        logger.info(f"Switched to environment: {environment.value}")
    
    def get_agent_config(self, agent_type: str = None):
        """Get agent configuration"""
        config = self.get_config()
        if agent_type:
            return config.agent_config.get_config(agent_type)
        return config.agent_config
    
    def get_system_config(self):
        """Get system configuration"""
        return self.get_config().system_config
    
    def get_model_config(self):
        """Get model configuration"""
        return self.get_config().model_config
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get_config().debug
    
    def get_environment(self) -> Environment:
        """Get current environment"""
        return self.get_config().environment

# Global configuration instance
_config_manager = ConfigManager()

def get_config() -> EnvironmentConfig:
    """Get global configuration instance"""
    return _config_manager.get_config()

def get_agent_config(agent_type: str = None):
    """Get agent configuration"""
    return _config_manager.get_agent_config(agent_type)

def get_system_config():
    """Get system configuration"""
    return _config_manager.get_system_config()

def get_model_config():
    """Get model configuration"""
    return _config_manager.get_model_config()

def reload_config():
    """Reload configuration"""
    _config_manager.reload_config()

def switch_environment(environment: Environment):
    """Switch environment"""
    _config_manager.switch_environment(environment)

def is_debug() -> bool:
    """Check if debug mode"""
    return _config_manager.is_debug()

def get_environment() -> Environment:
    """Get current environment"""
    return _config_manager.get_environment()

# Environment validation
def validate_environment_setup() -> Dict[str, Any]:
    """Validate environment setup and return status"""
    config = get_config()
    validation_results = {
        "environment": config.environment.value,
        "debug": config.debug,
        "issues": [],
        "warnings": []
    }
    
    # Check system configuration
    system_config = config.system_config
    
    # Database validation
    if system_config.database.postgres_enabled:
        if not system_config.database.postgres_username:
            validation_results["issues"].append("PostgreSQL enabled but no username configured")
    else:
        db_dir = os.path.dirname(system_config.database.sqlite_path)
        if not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                validation_results["issues"].append(f"Cannot create database directory: {e}")
    
    # Cache validation
    if system_config.cache.redis_enabled:
        if not system_config.cache.redis_host:
            validation_results["issues"].append("Redis enabled but no host configured")
    
    # Model validation
    model_config = config.model_config
    if not model_config.llm.api_key:
        validation_results["warnings"].append("No LLM API key configured")
    
    if not model_config.t2i.api_key:
        validation_results["warnings"].append("No T2I API key configured")
    
    # Security validation
    security_config = system_config.security
    if config.environment == Environment.PRODUCTION:
        if security_config.jwt_secret_key == "your-super-secret-key-change-in-production":
            validation_results["issues"].append("Default JWT secret key in production")
        
        if not security_config.auth_enabled:
            validation_results["issues"].append("Authentication disabled in production")
        
        if not security_config.encrypt_sensitive_data:
            validation_results["warnings"].append("Sensitive data encryption disabled in production")
    
    # Log directory validation
    log_dir = os.path.dirname(system_config.logging.log_file_path)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            validation_results["issues"].append(f"Cannot create log directory: {e}")
    
    # Agent configuration validation
    agent_config = config.agent_config
    for agent_type in ["strategist", "background_designer", "foreground_designer", "developer", "design_reviewer"]:
        if not agent_config.validate_config(agent_type):
            validation_results["issues"].append(f"Invalid configuration for agent: {agent_type}")
    
    validation_results["valid"] = len(validation_results["issues"]) == 0
    validation_results["total_issues"] = len(validation_results["issues"])
    validation_results["total_warnings"] = len(validation_results["warnings"])
    
    return validation_results

def print_config_summary():
    """Print configuration summary for debugging"""
    config = get_config()
    
    print(f"=== Configuration Summary ===")
    print(f"Environment: {config.environment.value}")
    print(f"Debug Mode: {config.debug}")
    print(f"Database: {'PostgreSQL' if config.system_config.database.postgres_enabled else 'SQLite'}")
    print(f"Cache: {'Redis' if config.system_config.cache.redis_enabled else 'Memory'}")
    print(f"Authentication: {'Enabled' if config.system_config.security.auth_enabled else 'Disabled'}")
    print(f"Default LLM: {config.model_config.llm.model_name}")
    print(f"Default T2I: {config.model_config.t2i.model_name}")
    print(f"Log Level: {config.system_config.logging.log_level}")
    print("="*30)
