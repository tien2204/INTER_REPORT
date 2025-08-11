import os
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import threading
import copy

from .base_config import BaseConfig
from .environment_config import EnvironmentConfig, ConfigProfile, ConfigTemplate
from .validation import ConfigValidator, ValidationReport, ValidationSeverity
from .secrets_manager import SecretsManager

class ConfigManager:
    """
    Central configuration manager for the Banner AI Generator system
    
    Provides unified access to configuration with support for:
    - Environment-specific configurations
    - Configuration profiles
    - Dynamic configuration updates
    - Validation and health monitoring
    - Secrets management integration
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("./config")
        self.secrets_manager = SecretsManager()
        self.validator = ConfigValidator()
        
        self._current_config: Optional[BaseConfig] = None
        self._config_cache: Dict[str, BaseConfig] = {}
        self._lock = threading.Lock()
        
        # Configuration change callbacks
        self._change_callbacks: List[callable] = []
        
        # Load initial configuration
        self._load_initial_config()
    
    def _load_initial_config(self) -> None:
        """Load initial configuration from environment and files"""
        try:
            # Determine environment and profile
            environment = os.getenv('ENVIRONMENT', os.getenv('ENV', 'development'))
            profile_name = os.getenv('CONFIG_PROFILE', 'standard')
            
            try:
                profile = ConfigProfile(profile_name)
            except ValueError:
                profile = ConfigProfile.STANDARD
            
            # Load configuration
            self._current_config = self.load_config(environment, profile)
            
        except Exception as e:
            print(f"Warning: Could not load initial configuration: {e}")
            # Fallback to default configuration
            self._current_config = BaseConfig()
    
    def get_config(self, environment: Optional[str] = None, profile: Optional[str] = None) -> BaseConfig:
        """
        Get configuration for specified environment and profile
        
        Args:
            environment: Target environment (development/staging/production)
            profile: Configuration profile (minimal/standard/enterprise/development)
        
        Returns:
            BaseConfig: Loaded and validated configuration
        """
        # Use current config if no specific environment/profile requested
        if environment is None and profile is None:
            if self._current_config is not None:
                return copy.deepcopy(self._current_config)
        
        # Determine environment and profile
        env = environment or os.getenv('ENVIRONMENT', 'development')
        profile_name = profile or os.getenv('CONFIG_PROFILE', 'standard')
        
        try:
            config_profile = ConfigProfile(profile_name)
        except ValueError:
            config_profile = ConfigProfile.STANDARD
        
        # Load configuration
        return self.load_config(env, config_profile)
    
    def load_config(self, environment: str, profile: ConfigProfile) -> BaseConfig:
        """
        Load configuration for specific environment and profile
        
        Args:
            environment: Target environment
            profile: Configuration profile
        
        Returns:
            BaseConfig: Loaded configuration
        """
        cache_key = f"{environment}_{profile.value}"
        
        with self._lock:
            # Check cache first
            if cache_key in self._config_cache:
                return copy.deepcopy(self._config_cache[cache_key])
        
        # Load base configuration
        config = BaseConfig()
        
        # Apply profile template
        config = EnvironmentConfig.apply_profile_template(config, profile)
        
        # Apply environment-specific overrides
        config = EnvironmentConfig.apply_environment_overrides(config, environment)
        
        # Load from configuration files
        config = self._load_from_files(config, environment, profile)
        
        # Load from environment variables
        env_config = BaseConfig.from_env()
        config = self._merge_configs(config, env_config)
        
        # Load secrets
        config = self._load_secrets(config)
        
        # Validate configuration
        validation_report = self.validator.validate_full_config(config)
        
        # Handle validation errors
        if validation_report.has_critical_issues():
            critical_issues = validation_report.get_issues_by_severity(ValidationSeverity.CRITICAL)
            error_messages = [issue.message for issue in critical_issues]
            raise ValueError(f"Critical configuration errors: {'; '.join(error_messages)}")
        
        # Cache configuration
        with self._lock:
            self._config_cache[cache_key] = copy.deepcopy(config)
        
        return config
    
    def _load_from_files(self, config: BaseConfig, environment: str, profile: ConfigProfile) -> BaseConfig:
        """Load configuration from files"""
        
        # Configuration file precedence (last one wins):
        # 1. config/default.json
        # 2. config/{profile}.json
        # 3. config/{environment}.json
        # 4. config/{environment}-{profile}.json
        
        config_files = [
            self.config_dir / "default.json",
            self.config_dir / f"{profile.value}.json",
            self.config_dir / f"{environment}.json",
            self.config_dir / f"{environment}-{profile.value}.json"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    file_config = BaseConfig.from_file(str(config_file), merge_env=False)
                    config = self._merge_configs(config, file_config)
                except Exception as e:
                    print(f"Warning: Could not load config file {config_file}: {e}")
        
        return config
    
    def _load_secrets(self, config: BaseConfig) -> BaseConfig:
        """Load secrets into configuration"""
        try:
            # Load API keys
            if not config.models.DEFAULT_LLM_API_KEY:
                config.models.DEFAULT_LLM_API_KEY = self.secrets_manager.get_secret("llm_api_key", "")
            
            if not config.models.T2I_API_KEY:
                config.models.T2I_API_KEY = self.secrets_manager.get_secret("t2i_api_key", "")
            
            if not config.models.MLLM_API_KEY:
                config.models.MLLM_API_KEY = self.secrets_manager.get_secret("mllm_api_key", "")
            
            # Load database password
            if not config.database.PASSWORD:
                config.database.PASSWORD = self.secrets_manager.get_secret("database_password", "")
            
            # Load JWT secret
            if not config.security.JWT_SECRET_KEY:
                config.security.JWT_SECRET_KEY = self.secrets_manager.get_secret("jwt_secret_key", "")
            
            # Load encryption key
            if not config.security.ENCRYPTION_KEY:
                config.security.ENCRYPTION_KEY = self.secrets_manager.get_secret("encryption_key", "")
            
        except Exception as e:
            print(f"Warning: Could not load secrets: {e}")
        
        return config
    
    def _merge_configs(self, base_config: BaseConfig, override_config: BaseConfig) -> BaseConfig:
        """Merge two configuration objects"""
        base_dict = base_config.to_dict()
        override_dict = override_config.to_dict()
        
        merged_dict = EnvironmentConfig.merge_configs(base_dict, override_dict)
        
        result_config = BaseConfig()
        result_config._apply_config_dict(merged_dict)
        
        return result_config
    
    def update_config(self, updates: Dict[str, Any], persist: bool = False) -> ValidationReport:
        """
        Update current configuration with new values
        
        Args:
            updates: Dictionary of configuration updates
            persist: Whether to persist changes to file
        
        Returns:
            ValidationReport: Validation report for updated configuration
        """
        with self._lock:
            if self._current_config is None:
                raise ValueError("No current configuration loaded")
            
            # Create updated configuration
            updated_config = copy.deepcopy(self._current_config)
            updated_config._apply_config_dict(updates)
            
            # Validate updated configuration
            validation_report = self.validator.validate_full_config(updated_config)
            
            # Only apply if validation passes
            if not validation_report.has_critical_issues():
                old_config = copy.deepcopy(self._current_config)
                self._current_config = updated_config
                
                # Clear cache
                self._config_cache.clear()
                
                # Notify callbacks
                self._notify_config_change(old_config, updated_config)
                
                # Persist if requested
                if persist:
                    self._persist_config_changes(updates)
            
            return validation_report
    
    def reload_config(self, clear_cache: bool = True) -> BaseConfig:
        """
        Reload configuration from sources
        
        Args:
            clear_cache: Whether to clear configuration cache
        
        Returns:
            BaseConfig: Reloaded configuration
        """
        with self._lock:
            if clear_cache:
                self._config_cache.clear()
            
            # Reload from environment
            self._load_initial_config()
            
            return copy.deepcopy(self._current_config)
    
    def validate_current_config(self) -> ValidationReport:
        """Validate current configuration"""
        if self._current_config is None:
            raise ValueError("No configuration loaded")
        
        return self.validator.validate_full_config(self._current_config)
    
    def get_config_health(self) -> Dict[str, Any]:
        """Get current configuration health report"""
        if self._current_config is None:
            raise ValueError("No configuration loaded")
        
        return self.validator.generate_config_health_report(self._current_config)
    
    def export_config(self, filepath: str, format: str = 'json', include_secrets: bool = False) -> bool:
        """
        Export current configuration to file
        
        Args:
            filepath: Output file path
            format: Export format ('json' or 'yaml')
            include_secrets: Whether to include sensitive data
        
        Returns:
            bool: Success status
        """
        if self._current_config is None:
            return False
        
        try:
            export_config = copy.deepcopy(self._current_config)
            
            # Remove secrets if not requested
            if not include_secrets:
                export_config = self._sanitize_config_for_export(export_config)
            
            return export_config.save_to_file(filepath, format)
            
        except Exception as e:
            print(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, filepath: str, validate: bool = True) -> ValidationReport:
        """
        Import configuration from file
        
        Args:
            filepath: Configuration file path
            validate: Whether to validate imported configuration
        
        Returns:
            ValidationReport: Validation report
        """
        try:
            imported_config = BaseConfig.from_file(filepath)
            
            if validate:
                validation_report = self.validator.validate_full_config(imported_config)
                
                if validation_report.has_critical_issues():
                    return validation_report
            
            # Apply imported configuration
            with self._lock:
                self._current_config = imported_config
                self._config_cache.clear()
            
            return validation_report if validate else ValidationReport(is_valid=True, total_issues=0)
            
        except Exception as e:
            # Create error report
            from .validation import ValidationIssue
            report = ValidationReport(is_valid=False, total_issues=1)
            report.add_issue(ValidationIssue(
                field_path="import",
                severity=ValidationSeverity.CRITICAL,
                message=f"Failed to import configuration: {str(e)}",
                category="import"
            ))
            return report
    
    def register_change_callback(self, callback: callable) -> None:
        """Register callback for configuration changes"""
        self._change_callbacks.append(callback)
    
    def _notify_config_change(self, old_config: BaseConfig, new_config: BaseConfig) -> None:
        """Notify registered callbacks of configuration changes"""
        for callback in self._change_callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                print(f"Error in config change callback: {e}")
    
    def _persist_config_changes(self, changes: Dict[str, Any]) -> None:
        """Persist configuration changes to file"""
        try:
            # Create config override file
            override_file = self.config_dir / "runtime_overrides.json"
            
            # Load existing overrides
            existing_overrides = {}
            if override_file.exists():
                with open(override_file, 'r') as f:
                    existing_overrides = json.load(f)
            
            # Merge changes
            merged_overrides = EnvironmentConfig.merge_configs(existing_overrides, changes)
            
            # Add metadata
            merged_overrides['_metadata'] = {
                'last_updated': datetime.now().isoformat(),
                'updated_by': 'config_manager'
            }
            
            # Save to file
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(override_file, 'w') as f:
                json.dump(merged_overrides, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Warning: Could not persist configuration changes: {e}")
    
    def _sanitize_config_for_export(self, config: BaseConfig) -> BaseConfig:
        """Remove sensitive data from configuration for export"""
        # Clear API keys
        config.models.DEFAULT_LLM_API_KEY = ""
        config.models.T2I_API_KEY = ""
        config.models.MLLM_API_KEY = ""
        
        # Clear database password
        config.database.PASSWORD = ""
        config.database.REDIS_PASSWORD = ""
        
        # Clear security keys
        config.security.JWT_SECRET_KEY = ""
        config.security.ENCRYPTION_KEY = ""
        
        return config
    
    def create_config_template(self, profile: ConfigProfile, output_path: str) -> bool:
        """
        Create configuration template file for specified profile
        
        Args:
            profile: Configuration profile
            output_path: Output file path
        
        Returns:
            bool: Success status
        """
        try:
            # Create base configuration
            config = BaseConfig()
            
            # Apply profile template
            config = EnvironmentConfig.apply_profile_template(config, profile)
            
            # Sanitize for template
            config = self._sanitize_config_for_export(config)
            
            # Add template metadata
            config.custom_settings['_template'] = {
                'profile': profile.value,
                'created_at': datetime.now().isoformat(),
                'description': EnvironmentConfig.get_profile_description(profile)
            }
            
            # Save template
            return config.save_to_file(output_path, 'json')
            
        except Exception as e:
            print(f"Error creating config template: {e}")
            return False
    
    def get_config_diff(self, other_config: BaseConfig) -> Dict[str, Any]:
        """
        Get differences between current config and another config
        
        Args:
            other_config: Configuration to compare against
        
        Returns:
            Dict containing differences
        """
        if self._current_config is None:
            raise ValueError("No current configuration loaded")
        
        current_dict = self._current_config.to_dict()
        other_dict = other_config.to_dict()
        
        return self._compute_config_diff(current_dict, other_dict)
    
    def _compute_config_diff(self, dict1: Dict[str, Any], dict2: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Recursively compute differences between configuration dictionaries"""
        diff = {}
        
        # Check all keys in both dictionaries
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in dict1:
                diff[full_key] = {"status": "added", "new_value": dict2[key]}
            elif key not in dict2:
                diff[full_key] = {"status": "removed", "old_value": dict1[key]}
            elif dict1[key] != dict2[key]:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    # Recursively diff nested dictionaries
                    nested_diff = self._compute_config_diff(dict1[key], dict2[key], full_key)
                    diff.update(nested_diff)
                else:
                    diff[full_key] = {
                        "status": "changed",
                        "old_value": dict1[key],
                        "new_value": dict2[key]
                    }
        
        return diff
    
    def backup_config(self, backup_dir: Optional[str] = None) -> str:
        """
        Create backup of current configuration
        
        Args:
            backup_dir: Backup directory (defaults to config_dir/backups)
        
        Returns:
            str: Path to backup file
        """
        if self._current_config is None:
            raise ValueError("No configuration to backup")
        
        # Determine backup directory
        if backup_dir:
            backup_path = Path(backup_dir)
        else:
            backup_path = self.config_dir / "backups"
        
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"config_backup_{timestamp}.json"
        backup_filepath = backup_path / backup_filename
        
        # Export configuration (including secrets for backup)
        if self.export_config(str(backup_filepath), include_secrets=True):
            return str(backup_filepath)
        else:
            raise RuntimeError("Failed to create configuration backup")
    
    def restore_from_backup(self, backup_filepath: str) -> ValidationReport:
        """
        Restore configuration from backup file
        
        Args:
            backup_filepath: Path to backup file
        
        Returns:
            ValidationReport: Validation report for restored configuration
        """
        return self.import_config(backup_filepath, validate=True)
    
    def list_available_profiles(self) -> List[Dict[str, str]]:
        """Get list of available configuration profiles"""
        profiles = []
        for profile in ConfigProfile:
            profiles.append({
                'name': profile.value,
                'description': EnvironmentConfig.get_profile_description(profile)
            })
        return profiles
    
    def list_available_environments(self) -> List[Dict[str, str]]:
        """Get list of available environments"""
        environments = []
        for env in EnvironmentConfig.get_environment_list():
            environments.append({
                'name': env,
                'description': EnvironmentConfig.get_environment_description(env)
            })
        return environments
    
    def get_config_suggestions(self) -> List[str]:
        """Get configuration improvement suggestions"""
        if self._current_config is None:
            return ["No configuration loaded"]
        
        return EnvironmentConfig.suggest_optimizations(self._current_config)
    
    def optimize_for_deployment(self, deployment_type: str) -> BaseConfig:
        """
        Optimize current configuration for deployment type
        
        Args:
            deployment_type: Deployment type (docker/kubernetes/serverless)
        
        Returns:
            BaseConfig: Optimized configuration
        """
        if self._current_config is None:
            raise ValueError("No configuration loaded")
        
        optimized_config = copy.deepcopy(self._current_config)
        return EnvironmentConfig.optimize_for_deployment(optimized_config, deployment_type)
    
    def cleanup_old_backups(self, max_backups: int = 10) -> int:
        """
        Clean up old backup files, keeping only the most recent ones
        
        Args:
            max_backups: Maximum number of backups to keep
        
        Returns:
            int: Number of backups deleted
        """
        backup_dir = self.config_dir / "backups"
        if not backup_dir.exists():
            return 0
        
        # Get all backup files
        backup_files = list(backup_dir.glob("config_backup_*.json"))
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Delete old backups
        deleted_count = 0
        for backup_file in backup_files[max_backups:]:
            try:
                backup_file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Warning: Could not delete backup {backup_file}: {e}")
        
        return deleted_count
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime configuration information"""
        return {
            'config_loaded': self._current_config is not None,
            'config_dir': str(self.config_dir),
            'cache_size': len(self._config_cache),
            'callbacks_registered': len(self._change_callbacks),
            'secrets_manager_active': self.secrets_manager is not None,
            'current_environment': self._current_config.ENVIRONMENT if self._current_config else None,
            'load_time': datetime.now().isoformat()
        }

# Configuration manager utilities

class ConfigWatcher:
    """Watch configuration files for changes and auto-reload"""
    
    def __init__(self, config_manager: ConfigManager, watch_interval: float = 5.0):
        self.config_manager = config_manager
        self.watch_interval = watch_interval
        self._watching = False
        self._watch_thread = None
        self._file_timestamps: Dict[str, float] = {}
    
    def start_watching(self) -> None:
        """Start watching configuration files"""
        if self._watching:
            return
        
        self._watching = True
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        print("Configuration file watcher started")
    
    def stop_watching(self) -> None:
        """Stop watching configuration files"""
        self._watching = False
        if self._watch_thread:
            self._watch_thread.join()
        print("Configuration file watcher stopped")
    
    def _watch_loop(self) -> None:
        """Main watch loop"""
        import time
        
        while self._watching:
            try:
                self._check_for_changes()
                time.sleep(self.watch_interval)
            except Exception as e:
                print(f"Error in config file watcher: {e}")
                time.sleep(self.watch_interval)
    
    def _check_for_changes(self) -> None:
        """Check for configuration file changes"""
        config_dir = self.config_manager.config_dir
        if not config_dir.exists():
            return
        
        config_files = list(config_dir.glob("*.json")) + list(config_dir.glob("*.yaml"))
        
        changed_files = []
        for config_file in config_files:
            try:
                current_mtime = config_file.stat().st_mtime
                last_mtime = self._file_timestamps.get(str(config_file), 0)
                
                if current_mtime > last_mtime:
                    changed_files.append(config_file)
                    self._file_timestamps[str(config_file)] = current_mtime
            except Exception:
                continue
        
        if changed_files:
            print(f"Configuration files changed: {[f.name for f in changed_files]}")
            try:
                self.config_manager.reload_config(clear_cache=True)
                print("Configuration reloaded successfully")
            except Exception as e:
                print(f"Error reloading configuration: {e}")

class ConfigBuilder:
    """Builder pattern for creating configurations"""
    
    def __init__(self):
        self._config = BaseConfig()
        self._validator = ConfigValidator()
    
    def with_environment(self, environment: str) -> 'ConfigBuilder':
        """Set environment"""
        self._config.ENVIRONMENT = environment
        return self
    
    def with_profile(self, profile: ConfigProfile) -> 'ConfigBuilder':
        """Apply configuration profile"""
        self._config = EnvironmentConfig.apply_profile_template(self._config, profile)
        return self
    
    def with_debug(self, debug: bool = True) -> 'ConfigBuilder':
        """Set debug mode"""
        self._config.DEBUG = debug
        if debug:
            self._config.LOG_LEVEL = 'DEBUG'
        return self
    
    def with_api_config(self, host: str = None, port: int = None, workers: int = None) -> 'ConfigBuilder':
        """Set API configuration"""
        if host is not None:
            self._config.API_HOST = host
        if port is not None:
            self._config.API_PORT = port
        if workers is not None:
            self._config.API_WORKERS = workers
        return self
    
    def with_database_config(self, host: str = None, port: int = None, name: str = None) -> 'ConfigBuilder':
        """Set database configuration"""
        if host is not None:
            self._config.database.HOST = host
        if port is not None:
            self._config.database.PORT = port
        if name is not None:
            self._config.database.NAME = name
        return self
    
    def with_model_config(self, llm_model: str = None, t2i_model: str = None) -> 'ConfigBuilder':
        """Set model configuration"""
        if llm_model is not None:
            self._config.models.DEFAULT_LLM_MODEL = llm_model
        if t2i_model is not None:
            self._config.models.DEFAULT_T2I_MODEL = t2i_model
        return self
    
    def with_performance_config(self, max_campaigns: int = None, worker_pool_size: int = None) -> 'ConfigBuilder':
        """Set performance configuration"""
        if max_campaigns is not None:
            self._config.performance.MAX_CONCURRENT_CAMPAIGNS = max_campaigns
        if worker_pool_size is not None:
            self._config.performance.WORKER_POOL_SIZE = worker_pool_size
        return self
    
    def with_security_config(self, enable_auth: bool = None, enable_rate_limiting: bool = None) -> 'ConfigBuilder':
        """Set security configuration"""
        if enable_auth is not None:
            self._config.security.ENABLE_AUTHENTICATION = enable_auth
        if enable_rate_limiting is not None:
            self._config.security.ENABLE_RATE_LIMITING = enable_rate_limiting
        return self
    
    def with_custom_setting(self, key: str, value: Any) -> 'ConfigBuilder':
        """Add custom setting"""
        self._config.custom_settings[key] = value
        return self
    
    def validate(self) -> 'ConfigBuilder':
        """Validate current configuration"""
        validation_report = self._validator.validate_full_config(self._config)
        
        if validation_report.has_critical_issues():
            critical_issues = validation_report.get_issues_by_severity(ValidationSeverity.CRITICAL)
            error_messages = [issue.message for issue in critical_issues]
            raise ValueError(f"Critical configuration errors: {'; '.join(error_messages)}")
        
        return self
    
    def build(self) -> BaseConfig:
        """Build final configuration"""
        return copy.deepcopy(self._config)

# Factory functions for common configurations

def create_development_config() -> BaseConfig:
    """Create development configuration"""
    return (ConfigBuilder()
            .with_environment('development')
            .with_profile(ConfigProfile.DEVELOPMENT)
            .with_debug(True)
            .with_api_config(port=8000, workers=1)
            .build())

def create_production_config() -> BaseConfig:
    """Create production configuration"""
    return (ConfigBuilder()
            .with_environment('production')
            .with_profile(ConfigProfile.ENTERPRISE)
            .with_debug(False)
            .with_api_config(workers=4)
            .with_security_config(enable_auth=True, enable_rate_limiting=True)
            .validate()
            .build())

def create_testing_config() -> BaseConfig:
    """Create testing configuration"""
    return (ConfigBuilder()
            .with_environment('testing')
            .with_profile(ConfigProfile.MINIMAL)
            .with_debug(True)
            .with_performance_config(max_campaigns=1, worker_pool_size=1)
            .build())
