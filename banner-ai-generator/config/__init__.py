from .base_config import BaseConfig, AgentConfig, ModelConfig, DatabaseConfig
from .environment_config import EnvironmentConfig, ConfigProfile, ConfigTemplate
from .validation import ConfigValidator, ValidationError, ValidationReport
from .config_manager import ConfigManager
from .secrets_manager import SecretsManager

import os
from typing import Optional

# Global configuration instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config(environment: Optional[str] = None, profile: Optional[str] = None) -> BaseConfig:
    """
    Get configuration for specified environment and profile
    
    Args:
        environment: Target environment (dev/staging/prod). Defaults to ENV var or 'development'
        profile: Configuration profile (minimal/standard/enterprise). Defaults to 'standard'
    
    Returns:
        BaseConfig: Loaded and validated configuration
    """
    manager = get_config_manager()
    return manager.get_config(environment, profile)

def reload_config() -> BaseConfig:
    """Reload configuration from environment variables and files"""
    global _config_manager
    _config_manager = None
    return get_config()

def validate_current_config() -> ValidationReport:
    """Validate current configuration and return detailed report"""
    config = get_config()
    validator = ConfigValidator()
    return validator.validate_full_config(config)

__all__ = [
    # Core classes
    'BaseConfig',
    'AgentConfig', 
    'ModelConfig',
    'DatabaseConfig',
    'EnvironmentConfig',
    'ConfigProfile',
    'ConfigTemplate',
    'ConfigValidator',
    'ConfigManager',
    'SecretsManager',
    
    # Utility functions
    'get_config',
    'get_config_manager',
    'reload_config',
    'validate_current_config',
    
    # Exceptions
    'ValidationError',
    'ValidationReport'
]
