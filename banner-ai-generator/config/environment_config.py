from typing import Dict, Any, Optional, List
from enum import Enum
import copy

class ConfigProfile(Enum):
    """Configuration profiles for different use cases"""
    MINIMAL = "minimal"      # Lightweight setup for testing
    STANDARD = "standard"    # Default production setup
    ENTERPRISE = "enterprise" # Full-featured enterprise setup
    DEVELOPMENT = "development" # Development with debugging features

class ConfigTemplate:
    """Pre-defined configuration templates"""
    
    MINIMAL_TEMPLATE = {
        'DEBUG': False,
        'LOG_LEVEL': 'WARNING',
        'SESSION_TIMEOUT_HOURS': 2,
        'MAX_ITERATIONS': 3,
        'API_WORKERS': 1,
        'agents': {
            'MAX_RETRIES': 1,
            'TIMEOUT_SECONDS': 120,
            'BACKGROUND_MAX_ITERATIONS': 3,
            'REVIEWER_ENABLED': False
        },
        'performance': {
            'MAX_MEMORY_SIZE_MB': 256,
            'MAX_CONCURRENT_CAMPAIGNS': 3,
            'WORKER_POOL_SIZE': 1,
            'ENABLE_CACHING': False
        },
        'security': {
            'ENABLE_AUTHENTICATION': False,
            'ENABLE_RATE_LIMITING': False,
            'ENABLE_AUDIT_LOGGING': False
        }
    }
    
    STANDARD_TEMPLATE = {
        'DEBUG': False,
        'LOG_LEVEL': 'INFO',
        'SESSION_TIMEOUT_HOURS': 24,
        'MAX_ITERATIONS': 5,
        'API_WORKERS': 2,
        'agents': {
            'MAX_RETRIES': 3,
            'TIMEOUT_SECONDS': 300,
            'BACKGROUND_MAX_ITERATIONS': 5,
            'REVIEWER_ENABLED': True
        },
        'performance': {
            'MAX_MEMORY_SIZE_MB': 1024,
            'MAX_CONCURRENT_CAMPAIGNS': 10,
            'WORKER_POOL_SIZE': 4,
            'ENABLE_CACHING': True
        },
        'security': {
            'ENABLE_AUTHENTICATION': True,
            'ENABLE_RATE_LIMITING': True,
            'ENABLE_AUDIT_LOGGING': True
        }
    }
    
    ENTERPRISE_TEMPLATE = {
        'DEBUG': False,
        'LOG_LEVEL': 'INFO',
        'SESSION_TIMEOUT_HOURS': 48,
        'MAX_ITERATIONS': 10,
        'API_WORKERS': 8,
        'agents': {
            'MAX_RETRIES': 5,
            'TIMEOUT_SECONDS': 600,
            'BACKGROUND_MAX_ITERATIONS': 10,
            'REVIEWER_ENABLED': True,
            'REVIEWER_STRICT_MODE': True,
            'ENABLE_PARALLEL_EXECUTION': True
        },
        'performance': {
            'MAX_MEMORY_SIZE_MB': 4096,
            'MAX_CONCURRENT_CAMPAIGNS': 50,
            'WORKER_POOL_SIZE': 16,
            'ENABLE_CACHING': True,
            'CACHE_TTL_SECONDS': 7200
        },
        'security': {
            'ENABLE_AUTHENTICATION': True,
            'ENABLE_RATE_LIMITING': True,
            'ENABLE_AUDIT_LOGGING': True,
            'ENCRYPT_SENSITIVE_DATA': True,
            'SCAN_UPLOADS_FOR_MALWARE': True
        },
        'database': {
            'MIN_CONNECTIONS': 5,
            'MAX_CONNECTIONS': 50,
            'CONNECTION_TIMEOUT': 60
        }
    }
    
    DEVELOPMENT_TEMPLATE = {
        'DEBUG': True,
        'LOG_LEVEL': 'DEBUG',
        'SESSION_TIMEOUT_HOURS': 1,
        'MAX_ITERATIONS': 3,
        'API_WORKERS': 1,
        'agents': {
            'MAX_RETRIES': 1,
            'TIMEOUT_SECONDS': 60,
            'BACKGROUND_MAX_ITERATIONS': 2,
            'REVIEWER_ENABLED': True,
            'REVIEWER_STRICT_MODE': False
        },
        'performance': {
            'MAX_MEMORY_SIZE_MB': 512,
            'MAX_CONCURRENT_CAMPAIGNS': 3,
            'WORKER_POOL_SIZE': 2,
            'ENABLE_CACHING': False,
            'ENABLE_MEMORY_MONITORING': True
        },
        'security': {
            'ENABLE_AUTHENTICATION': False,
            'ENABLE_RATE_LIMITING': False,
            'ENABLE_AUDIT_LOGGING': True
        }
    }

class EnvironmentConfig:
    """Environment-specific configuration management"""
    
    # Environment-specific overrides
    ENVIRONMENTS = {
        'development': {
            'DEBUG': True,
            'LOG_LEVEL': 'DEBUG',
            'SESSION_TIMEOUT_HOURS': 1,
            'ENABLE_ITERATIVE_REFINEMENT': True,
            'API_WORKERS': 1,
            'database': {
                'NAME': 'banner_generator_dev',
                'ENABLE_QUERY_LOGGING': True,
                'MIN_CONNECTIONS': 1,
                'MAX_CONNECTIONS': 5
            },
            'models': {
                'LLM_TEMPERATURE': 0.9,
                'MODEL_RETRY_ATTEMPTS': 1,
                'ENABLE_MODEL_FALLBACK': False
            },
            'performance': {
                'MAX_MEMORY_SIZE_MB': 512,
                'MAX_CONCURRENT_CAMPAIGNS': 3,
                'ENABLE_MEMORY_MONITORING': True
            },
            'security': {
                'ENABLE_AUTHENTICATION': False,
                'ENABLE_RATE_LIMITING': False
            }
        },
        
        'staging': {
            'DEBUG': False,
            'LOG_LEVEL': 'INFO',
            'SESSION_TIMEOUT_HOURS': 12,
            'API_WORKERS': 2,
            'database': {
                'NAME': 'banner_generator_staging',
                'ENABLE_QUERY_LOGGING': False,
                'MIN_CONNECTIONS': 2,
                'MAX_CONNECTIONS': 15
            },
            'models': {
                'LLM_TEMPERATURE': 0.7,
                'MODEL_RETRY_ATTEMPTS': 2,
                'ENABLE_MODEL_FALLBACK': True
            },
            'performance': {
                'MAX_MEMORY_SIZE_MB': 1024,
                'MAX_CONCURRENT_CAMPAIGNS': 15,
                'ENABLE_CACHING': True
            },
            'security': {
                'ENABLE_AUTHENTICATION': True,
                'ENABLE_RATE_LIMITING': True,
                'ENABLE_AUDIT_LOGGING': True
            }
        },
        
        'production': {
            'DEBUG': False,
            'LOG_LEVEL': 'WARNING',
            'SESSION_TIMEOUT_HOURS': 48,
            'API_WORKERS': 4,
            'database': {
                'NAME': 'banner_generator_prod',
                'ENABLE_QUERY_LOGGING': False,
                'MIN_CONNECTIONS': 5,
                'MAX_CONNECTIONS': 25,
                'CONNECTION_TIMEOUT': 60
            },
            'models': {
                'LLM_TEMPERATURE': 0.7,
                'MODEL_RETRY_ATTEMPTS': 3,
                'ENABLE_MODEL_FALLBACK': True,
                'LLM_TIMEOUT': 180,
                'T2I_TIMEOUT': 600
            },
            'performance': {
                'MAX_MEMORY_SIZE_MB': 2048,
                'MAX_CONCURRENT_CAMPAIGNS': 25,
                'WORKER_POOL_SIZE': 8,
                'ENABLE_CACHING': True,
                'CACHE_TTL_SECONDS': 3600,
                'AUTO_CLEANUP_EXPIRED_SESSIONS': True
            },
            'security': {
                'ENABLE_AUTHENTICATION': True,
                'ENABLE_RATE_LIMITING': True,
                'ENABLE_AUDIT_LOGGING': True,
                'ENCRYPT_SENSITIVE_DATA': True,
                'SCAN_UPLOADS_FOR_MALWARE': True,
                'MAX_FILE_SIZE_MB': 25
            }
        }
    }
    
    # Profile-specific templates
    PROFILE_TEMPLATES = {
        ConfigProfile.MINIMAL: MINIMAL_TEMPLATE,
        ConfigProfile.STANDARD: STANDARD_TEMPLATE,
        ConfigProfile.ENTERPRISE: ENTERPRISE_TEMPLATE,
        ConfigProfile.DEVELOPMENT: DEVELOPMENT_TEMPLATE
    }
    
    @classmethod
    def get_template(cls, profile: ConfigProfile) -> Dict[str, Any]:
        """Get configuration template for profile"""
        return copy.deepcopy(cls.PROFILE_TEMPLATES.get(profile, cls.STANDARD_TEMPLATE))
    
    @classmethod
    def get_environment_config(cls, environment: str) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        return copy.deepcopy(cls.ENVIRONMENTS.get(environment.lower(), {}))
    
    @classmethod
    def merge_configs(cls, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries recursively"""
        result = copy.deepcopy(base_config)
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursive merge for nested dictionaries
                result[key] = cls.merge_configs(result[key], value)
            else:
                # Direct override for non-dict values or new keys
                result[key] = copy.deepcopy(value)
        
        return result
    
    @classmethod
    def apply_environment_overrides(cls, config: 'BaseConfig', environment: str) -> 'BaseConfig':
        """Apply environment-specific overrides to configuration"""
        from .base_config import BaseConfig
        
        env_overrides = cls.get_environment_config(environment)
        if not env_overrides:
            return config
        
        # Convert config to dict, merge, and convert back
        config_dict = config.to_dict()
        merged_dict = cls.merge_configs(config_dict, env_overrides)
        
        # Apply merged configuration
        config._apply_config_dict(merged_dict)
        return config
    
    @classmethod
    def apply_profile_template(cls, config: 'BaseConfig', profile: ConfigProfile) -> 'BaseConfig':
        """Apply profile template to configuration"""
        template = cls.get_template(profile)
        
        # Convert config to dict, merge, and convert back
        config_dict = config.to_dict()
        merged_dict = cls.merge_configs(config_dict, template)
        
        # Apply merged configuration
        config._apply_config_dict(merged_dict)
        return config
    
    @classmethod
    def create_config_for_environment(cls, environment: str, profile: Optional[ConfigProfile] = None) -> 'BaseConfig':
        """Create optimized configuration for specific environment and profile"""
        from .base_config import BaseConfig
        
        # Start with base configuration
        config = BaseConfig()
        
        # Apply profile template if specified
        if profile:
            config = cls.apply_profile_template(config, profile)
        
        # Apply environment-specific overrides
        config = cls.apply_environment_overrides(config, environment)
        
        # Set environment
        config.ENVIRONMENT = environment
        
        return config
    
    @classmethod
    def get_recommended_profile(cls, environment: str) -> ConfigProfile:
        """Get recommended profile for environment"""
        recommendations = {
            'development': ConfigProfile.DEVELOPMENT,
            'staging': ConfigProfile.STANDARD,
            'production': ConfigProfile.ENTERPRISE
        }
        return recommendations.get(environment.lower(), ConfigProfile.STANDARD)
    
    @classmethod
    def validate_environment_compatibility(cls, config: 'BaseConfig') -> List[str]:
        """Validate configuration compatibility with environment"""
        warnings = []
        environment = config.ENVIRONMENT.lower()
        
        # Development environment warnings
        if environment == 'development':
            if not config.DEBUG:
                warnings.append("DEBUG should be True in development environment")
            if config.LOG_LEVEL not in ['DEBUG', 'INFO']:
                warnings.append("LOG_LEVEL should be DEBUG or INFO in development")
            if config.security.ENABLE_AUTHENTICATION:
                warnings.append("Authentication not typically needed in development")
        
        # Production environment warnings
        elif environment == 'production':
            if config.DEBUG:
                warnings.append("DEBUG should be False in production environment")
            if config.LOG_LEVEL == 'DEBUG':
                warnings.append("LOG_LEVEL should not be DEBUG in production")
            if not config.security.ENABLE_AUTHENTICATION:
                warnings.append("Authentication should be enabled in production")
            if not config.security.ENABLE_RATE_LIMITING:
                warnings.append("Rate limiting should be enabled in production")
            if config.performance.MAX_CONCURRENT_CAMPAIGNS < 10:
                warnings.append("Consider increasing MAX_CONCURRENT_CAMPAIGNS for production")
        
        # Staging environment warnings
        elif environment == 'staging':
            if config.DEBUG:
                warnings.append("DEBUG should typically be False in staging")
            if not config.security.ENABLE_AUTHENTICATION:
                warnings.append("Authentication should be enabled in staging for production testing")
        
        return warnings
    
    @classmethod
    def optimize_for_deployment(cls, config: 'BaseConfig', deployment_type: str) -> 'BaseConfig':
        """Optimize configuration for specific deployment type"""
        
        if deployment_type.lower() == 'docker':
            # Docker-specific optimizations
            config.API_HOST = "0.0.0.0"  # Bind to all interfaces
            config.database.HOST = "postgres"  # Docker service name
            config.database.REDIS_HOST = "redis"  # Docker service name
            
        elif deployment_type.lower() == 'kubernetes':
            # Kubernetes-specific optimizations
            config.API_HOST = "0.0.0.0"
            config.performance.ENABLE_MEMORY_MONITORING = True
            config.security.TRUST_PROXY = True  # Behind ingress
            
        elif deployment_type.lower() == 'serverless':
            # Serverless-specific optimizations
            config.SESSION_TIMEOUT_HOURS = 1  # Shorter timeout
            config.performance.MAX_CONCURRENT_CAMPAIGNS = 5
            config.performance.WORKER_POOL_SIZE = 1
            config.database.MIN_CONNECTIONS = 0  # No persistent connections
            config.database.MAX_CONNECTIONS = 3
            
        return config

class EnvironmentConfig:
    """Environment-specific configuration management"""
    
    # Environment-specific overrides
    ENVIRONMENTS = {
        'development': {
            'DEBUG': True,
            'LOG_LEVEL': 'DEBUG',
            'SESSION_TIMEOUT_HOURS': 1,
            'ENABLE_ITERATIVE_REFINEMENT': True,
            'API_WORKERS': 1,
            'database': {
                'NAME': 'banner_generator_dev',
                'ENABLE_QUERY_LOGGING': True,
                'MIN_CONNECTIONS': 1,
                'MAX_CONNECTIONS': 5
            },
            'models': {
                'LLM_TEMPERATURE': 0.9,
                'MODEL_RETRY_ATTEMPTS': 1,
                'ENABLE_MODEL_FALLBACK': False
            },
            'performance': {
                'MAX_MEMORY_SIZE_MB': 512,
                'MAX_CONCURRENT_CAMPAIGNS': 3,
                'ENABLE_MEMORY_MONITORING': True
            },
            'security': {
                'ENABLE_AUTHENTICATION': False,
                'ENABLE_RATE_LIMITING': False
            }
        },
        
        'staging': {
            'DEBUG': False,
            'LOG_LEVEL': 'INFO',
            'SESSION_TIMEOUT_HOURS': 12,
            'API_WORKERS': 2,
            'database': {
                'NAME': 'banner_generator_staging',
                'ENABLE_QUERY_LOGGING': False,
                'MIN_CONNECTIONS': 2,
                'MAX_CONNECTIONS': 15
            },
            'models': {
                'LLM_TEMPERATURE': 0.7,
                'MODEL_RETRY_ATTEMPTS': 2,
                'ENABLE_MODEL_FALLBACK': True
            },
            'performance': {
                'MAX_MEMORY_SIZE_MB': 1024,
                'MAX_CONCURRENT_CAMPAIGNS': 15,
                'ENABLE_CACHING': True
            },
            'security': {
                'ENABLE_AUTHENTICATION': True,
                'ENABLE_RATE_LIMITING': True,
                'ENABLE_AUDIT_LOGGING': True
            }
        },
        
        'production': {
            'DEBUG': False,
            'LOG_LEVEL': 'WARNING',
            'SESSION_TIMEOUT_HOURS': 48,
            'API_WORKERS': 4,
            'database': {
                'NAME': 'banner_generator_prod',
                'ENABLE_QUERY_LOGGING': False,
                'MIN_CONNECTIONS': 5,
                'MAX_CONNECTIONS': 25,
                'CONNECTION_TIMEOUT': 60
            },
            'models': {
                'LLM_TEMPERATURE': 0.7,
                'MODEL_RETRY_ATTEMPTS': 3,
                'ENABLE_MODEL_FALLBACK': True,
                'LLM_TIMEOUT': 180,
                'T2I_TIMEOUT': 600
            },
            'performance': {
                'MAX_MEMORY_SIZE_MB': 2048,
                'MAX_CONCURRENT_CAMPAIGNS': 25,
                'WORKER_POOL_SIZE': 8,
                'ENABLE_CACHING': True,
                'CACHE_TTL_SECONDS': 3600,
                'AUTO_CLEANUP_EXPIRED_SESSIONS': True
            },
            'security': {
                'ENABLE_AUTHENTICATION': True,
                'ENABLE_RATE_LIMITING': True,
                'ENABLE_AUDIT_LOGGING': True,
                'ENCRYPT_SENSITIVE_DATA': True,
                'SCAN_UPLOADS_FOR_MALWARE': True,
                'MAX_FILE_SIZE_MB': 25
            }
        },
        
        'testing': {
            'DEBUG': True,
            'LOG_LEVEL': 'DEBUG',
            'SESSION_TIMEOUT_HOURS': 0.5,  # 30 minutes
            'MAX_ITERATIONS': 1,
            'API_WORKERS': 1,
            'database': {
                'NAME': 'banner_generator_test',
                'ENABLE_QUERY_LOGGING': True,
                'MIN_CONNECTIONS': 1,
                'MAX_CONNECTIONS': 3
            },
            'models': {
                'MODEL_RETRY_ATTEMPTS': 1,
                'ENABLE_MODEL_FALLBACK': False,
                'LLM_TIMEOUT': 30,
                'T2I_TIMEOUT': 60
            },
            'performance': {
                'MAX_MEMORY_SIZE_MB': 128,
                'MAX_CONCURRENT_CAMPAIGNS': 1,
                'WORKER_POOL_SIZE': 1,
                'ENABLE_CACHING': False
            },
            'security': {
                'ENABLE_AUTHENTICATION': False,
                'ENABLE_RATE_LIMITING': False,
                'ENABLE_AUDIT_LOGGING': False
            }
        }
    }
    
    @classmethod
    def get_environment_list(cls) -> List[str]:
        """Get list of available environments"""
        return list(cls.ENVIRONMENTS.keys())
    
    @classmethod
    def is_valid_environment(cls, environment: str) -> bool:
        """Check if environment is valid"""
        return environment.lower() in cls.ENVIRONMENTS
    
    @classmethod
    def get_environment_description(cls, environment: str) -> str:
        """Get description of environment"""
        descriptions = {
            'development': 'Local development with debugging features enabled',
            'staging': 'Pre-production testing environment with production-like settings',
            'production': 'Live production environment with security and performance optimizations',
            'testing': 'Automated testing environment with minimal resource usage'
        }
        return descriptions.get(environment.lower(), 'Unknown environment')
    
    @classmethod
    def get_profile_description(cls, profile: ConfigProfile) -> str:
        """Get description of configuration profile"""
        descriptions = {
            ConfigProfile.MINIMAL: 'Lightweight setup with minimal features for testing',
            ConfigProfile.STANDARD: 'Balanced setup suitable for most production deployments',
            ConfigProfile.ENTERPRISE: 'Full-featured setup with all enterprise capabilities',
            ConfigProfile.DEVELOPMENT: 'Development-optimized setup with debugging features'
        }
        return descriptions.get(profile, 'Unknown profile')
    
    @classmethod
    def get_environment_requirements(cls, environment: str) -> Dict[str, Any]:
        """Get infrastructure requirements for environment"""
        requirements = {
            'development': {
                'cpu_cores': 2,
                'memory_gb': 4,
                'storage_gb': 10,
                'database': 'sqlite/postgresql',
                'redis': 'optional',
                'load_balancer': False
            },
            'staging': {
                'cpu_cores': 4,
                'memory_gb': 8,
                'storage_gb': 50,
                'database': 'postgresql',
                'redis': 'required',
                'load_balancer': True
            },
            'production': {
                'cpu_cores': 8,
                'memory_gb': 16,
                'storage_gb': 200,
                'database': 'postgresql_cluster',
                'redis': 'redis_cluster',
                'load_balancer': True,
                'monitoring': 'required',
                'backup': 'required'
            },
            'testing': {
                'cpu_cores': 1,
                'memory_gb': 2,
                'storage_gb': 5,
                'database': 'sqlite',
                'redis': 'optional',
                'load_balancer': False
            }
        }
        return requirements.get(environment.lower(), {})
    
    @classmethod
    def suggest_optimizations(cls, config: 'BaseConfig') -> List[str]:
        """Suggest optimizations based on environment and current configuration"""
        suggestions = []
        environment = config.ENVIRONMENT.lower()
        
        # Performance suggestions
        if environment == 'production':
            if config.performance.WORKER_POOL_SIZE < 4:
                suggestions.append("Consider increasing WORKER_POOL_SIZE for better performance")
            if not config.performance.ENABLE_CACHING:
                suggestions.append("Enable caching for better performance in production")
            if config.database.MAX_CONNECTIONS < 20:
                suggestions.append("Consider increasing database connection pool for production load")
        
        # Security suggestions
        if environment in ['staging', 'production']:
            if not config.security.ENABLE_AUTHENTICATION:
                suggestions.append("Enable authentication for non-development environments")
            if not config.security.ENABLE_RATE_LIMITING:
                suggestions.append("Enable rate limiting to prevent abuse")
            if environment == 'production' and not config.security.ENCRYPT_SENSITIVE_DATA:
                suggestions.append("Consider enabling data encryption for production")
        
        # Memory suggestions
        if config.performance.MAX_MEMORY_SIZE_MB > 2048 and environment == 'development':
            suggestions.append("Consider reducing memory allocation for development environment")
        elif config.performance.MAX_MEMORY_SIZE_MB < 1024 and environment == 'production':
            suggestions.append("Consider increasing memory allocation for production workload")
        
        # Agent configuration suggestions
        if config.agents.MAX_RETRIES > 3 and environment == 'development':
            suggestions.append("Consider reducing retry attempts for faster development feedback")
        elif config.agents.MAX_RETRIES < 3 and environment == 'production':
            suggestions.append("Consider increasing retry attempts for production reliability")
        
        return suggestions
