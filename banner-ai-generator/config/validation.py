from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import re
import os
from urllib.parse import urlparse

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    field_path: str
    severity: ValidationSeverity
    message: str
    current_value: Any = None
    suggested_value: Any = None
    category: str = "general"

@dataclass
class ValidationReport:
    """Complete validation report"""
    is_valid: bool
    total_issues: int
    issues_by_severity: Dict[ValidationSeverity, int] = field(default_factory=dict)
    issues: List[ValidationIssue] = field(default_factory=list)
    config_summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add validation issue to report"""
        self.issues.append(issue)
        self.total_issues += 1
        
        # Update severity counts
        if issue.severity not in self.issues_by_severity:
            self.issues_by_severity[issue.severity] = 0
        self.issues_by_severity[issue.severity] += 1
        
        # Mark as invalid if critical or error
        if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
            self.is_valid = False
    
    def get_issues_by_category(self, category: str) -> List[ValidationIssue]:
        """Get issues filtered by category"""
        return [issue for issue in self.issues if issue.category == category]
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def has_critical_issues(self) -> bool:
        """Check if report has critical issues"""
        return ValidationSeverity.CRITICAL in self.issues_by_severity
    
    def has_errors(self) -> bool:
        """Check if report has errors"""
        return ValidationSeverity.ERROR in self.issues_by_severity
    
    def format_summary(self) -> str:
        """Format validation summary as string"""
        if self.is_valid:
            return f"✅ Configuration is valid ({self.total_issues} warnings/info)"
        else:
            return f"❌ Configuration has issues: {self.total_issues} total"

class ConfigValidator:
    """Comprehensive configuration validator"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def validate_full_config(self, config: 'BaseConfig') -> ValidationReport:
        """Perform complete configuration validation"""
        report = ValidationReport(is_valid=True, total_issues=0)
        
        # Basic field validation
        self._validate_basic_fields(config, report)
        
        # Database configuration validation
        self._validate_database_config(config.database, report)
        
        # Model configuration validation
        self._validate_model_config(config.models, report)
        
        # Agent configuration validation
        self._validate_agent_config(config.agents, report)
        
        # Security configuration validation
        self._validate_security_config(config.security, report)
        
        # Performance configuration validation
        self._validate_performance_config(config.performance, report)
        
        # Cross-configuration validation
        self._validate_config_consistency(config, report)
        
        # Environment-specific validation
        self._validate_environment_compatibility(config, report)
        
        # Directory validation
        self._validate_directories(config, report)
        
        # Add configuration summary
        report.config_summary = self._generate_config_summary(config)
        
        return report
    
    def _validate_basic_fields(self, config: 'BaseConfig', report: ValidationReport) -> None:
        """Validate basic configuration fields"""
        
        # Environment validation
        valid_environments = ['development', 'staging', 'production', 'testing']
        if config.ENVIRONMENT not in valid_environments:
            report.add_issue(ValidationIssue(
                field_path="ENVIRONMENT",
                severity=ValidationSeverity.ERROR,
                message=f"Invalid environment '{config.ENVIRONMENT}'. Must be one of: {valid_environments}",
                current_value=config.ENVIRONMENT,
                suggested_value="development",
                category="basic"
            ))
        
        # Log level validation
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.LOG_LEVEL not in valid_log_levels:
            report.add_issue(ValidationIssue(
                field_path="LOG_LEVEL",
                severity=ValidationSeverity.ERROR,
                message=f"Invalid log level '{config.LOG_LEVEL}'. Must be one of: {valid_log_levels}",
                current_value=config.LOG_LEVEL,
                suggested_value="INFO",
                category="basic"
            ))
        
        # Numeric validations
        if config.SESSION_TIMEOUT_HOURS <= 0:
            report.add_issue(ValidationIssue(
                field_path="SESSION_TIMEOUT_HOURS",
                severity=ValidationSeverity.ERROR,
                message="Session timeout must be positive",
                current_value=config.SESSION_TIMEOUT_HOURS,
                suggested_value=24,
                category="basic"
            ))
        
        if config.MAX_ITERATIONS <= 0:
            report.add_issue(ValidationIssue(
                field_path="MAX_ITERATIONS",
                severity=ValidationSeverity.ERROR,
                message="Max iterations must be positive",
                current_value=config.MAX_ITERATIONS,
                suggested_value=5,
                category="basic"
            ))
        
        # API validation
        if not (1 <= config.API_PORT <= 65535):
            report.add_issue(ValidationIssue(
                field_path="API_PORT",
                severity=ValidationSeverity.ERROR,
                message="API port must be between 1 and 65535",
                current_value=config.API_PORT,
                suggested_value=8000,
                category="api"
            ))
        
        if config.API_WORKERS <= 0:
            report.add_issue(ValidationIssue(
                field_path="API_WORKERS",
                severity=ValidationSeverity.ERROR,
                message="API workers must be positive",
                current_value=config.API_WORKERS,
                suggested_value=2,
                category="api"
            ))
        
        # Performance warnings
        if config.SESSION_TIMEOUT_HOURS > 72:
            report.add_issue(ValidationIssue(
                field_path="SESSION_TIMEOUT_HOURS",
                severity=ValidationSeverity.WARNING,
                message="Very long session timeout may consume excessive memory",
                current_value=config.SESSION_TIMEOUT_HOURS,
                suggested_value=48,
                category="performance"
            ))
    
    def _validate_database_config(self, db_config: 'DatabaseConfig', report: ValidationReport) -> None:
        """Validate database configuration"""
        
        # Connection validation
        if db_config.PORT <= 0 or db_config.PORT > 65535:
            report.add_issue(ValidationIssue(
                field_path="database.PORT",
                severity=ValidationSeverity.ERROR,
                message="Database port must be between 1 and 65535",
                current_value=db_config.PORT,
                suggested_value=5432,
                category="database"
            ))
        
        if not db_config.HOST or len(db_config.HOST.strip()) == 0:
            report.add_issue(ValidationIssue(
                field_path="database.HOST",
                severity=ValidationSeverity.ERROR,
                message="Database host cannot be empty",
                current_value=db_config.HOST,
                suggested_value="localhost",
                category="database"
            ))
        
        if not db_config.NAME or len(db_config.NAME.strip()) == 0:
            report.add_issue(ValidationIssue(
                field_path="database.NAME",
                severity=ValidationSeverity.ERROR,
                message="Database name cannot be empty",
                current_value=db_config.NAME,
                suggested_value="banner_generator",
                category="database"
            ))
        
        # Connection pool validation
        if db_config.MIN_CONNECTIONS < 0:
            report.add_issue(ValidationIssue(
                field_path="database.MIN_CONNECTIONS",
                severity=ValidationSeverity.ERROR,
                message="Minimum connections cannot be negative",
                current_value=db_config.MIN_CONNECTIONS,
                suggested_value=1,
                category="database"
            ))
        
        if db_config.MAX_CONNECTIONS <= db_config.MIN_CONNECTIONS:
            report.add_issue(ValidationIssue(
                field_path="database.MAX_CONNECTIONS",
                severity=ValidationSeverity.ERROR,
                message="Max connections must be greater than min connections",
                current_value=db_config.MAX_CONNECTIONS,
                suggested_value=db_config.MIN_CONNECTIONS + 5,
                category="database"
            ))
        
        # Redis validation
        if db_config.REDIS_PORT <= 0 or db_config.REDIS_PORT > 65535:
            report.add_issue(ValidationIssue(
                field_path="database.REDIS_PORT",
                severity=ValidationSeverity.ERROR,
                message="Redis port must be between 1 and 65535",
                current_value=db_config.REDIS_PORT,
                suggested_value=6379,
                category="database"
            ))
        
        if db_config.REDIS_DB < 0 or db_config.REDIS_DB > 15:
            report.add_issue(ValidationIssue(
                field_path="database.REDIS_DB",
                severity=ValidationSeverity.WARNING,
                message="Redis DB index should typically be between 0-15",
                current_value=db_config.REDIS_DB,
                suggested_value=0,
                category="database"
            ))
    
    def _validate_model_config(self, model_config: 'ModelConfig', report: ValidationReport) -> None:
        """Validate AI model configuration"""
        
        # API key validation (warn if empty in non-development)
        if not model_config.DEFAULT_LLM_API_KEY:
            report.add_issue(ValidationIssue(
                field_path="models.DEFAULT_LLM_API_KEY",
                severity=ValidationSeverity.WARNING,
                message="LLM API key is not set",
                current_value="",
                category="models"
            ))
        
        # URL validation
        if model_config.DEFAULT_LLM_BASE_URL:
            if not self._is_valid_url(model_config.DEFAULT_LLM_BASE_URL):
                report.add_issue(ValidationIssue(
                    field_path="models.DEFAULT_LLM_BASE_URL",
                    severity=ValidationSeverity.ERROR,
                    message="Invalid LLM base URL format",
                    current_value=model_config.DEFAULT_LLM_BASE_URL,
                    category="models"
                ))
        
        # Parameter validation
        if model_config.LLM_MAX_TOKENS <= 0:
            report.add_issue(ValidationIssue(
                field_path="models.LLM_MAX_TOKENS",
                severity=ValidationSeverity.ERROR,
                message="Max tokens must be positive",
                current_value=model_config.LLM_MAX_TOKENS,
                suggested_value=4096,
                category="models"
            ))
        
        if not (0.0 <= model_config.LLM_TEMPERATURE <= 2.0):
            report.add_issue(ValidationIssue(
                field_path="models.LLM_TEMPERATURE",
                severity=ValidationSeverity.WARNING,
                message="Temperature should be between 0.0 and 2.0",
                current_value=model_config.LLM_TEMPERATURE,
                suggested_value=0.7,
                category="models"
            ))
        
        # Timeout validation
        if model_config.LLM_TIMEOUT <= 0:
            report.add_issue(ValidationIssue(
                field_path="models.LLM_TIMEOUT",
                severity=ValidationSeverity.ERROR,
                message="LLM timeout must be positive",
                current_value=model_config.LLM_TIMEOUT,
                suggested_value=120,
                category="models"
            ))
        
        # T2I validation
        if model_config.T2I_MAX_SIZE <= 0:
            report.add_issue(ValidationIssue(
                field_path="models.T2I_MAX_SIZE",
                severity=ValidationSeverity.ERROR,
                message="T2I max size must be positive",
                current_value=model_config.T2I_MAX_SIZE,
                suggested_value=1024,
                category="models"
            ))
        
        # Rate limiting validation
        if model_config.LLM_REQUESTS_PER_MINUTE <= 0:
            report.add_issue(ValidationIssue(
                field_path="models.LLM_REQUESTS_PER_MINUTE",
                severity=ValidationSeverity.ERROR,
                message="LLM requests per minute must be positive",
                current_value=model_config.LLM_REQUESTS_PER_MINUTE,
                suggested_value=60,
                category="models"
            ))
    
    def _validate_agent_config(self, agent_config: 'AgentConfig', report: ValidationReport) -> None:
        """Validate agent configuration"""
        
        # Retry validation
        if agent_config.MAX_RETRIES < 0:
            report.add_issue(ValidationIssue(
                field_path="agents.MAX_RETRIES",
                severity=ValidationSeverity.ERROR,
                message="Max retries cannot be negative",
                current_value=agent_config.MAX_RETRIES,
                suggested_value=3,
                category="agents"
            ))
        
        if agent_config.MAX_RETRIES > 10:
            report.add_issue(ValidationIssue(
                field_path="agents.MAX_RETRIES",
                severity=ValidationSeverity.WARNING,
                message="High retry count may cause long delays",
                current_value=agent_config.MAX_RETRIES,
                suggested_value=5,
                category="agents"
            ))
        
        # Timeout validation
        if agent_config.TIMEOUT_SECONDS <= 0:
            report.add_issue(ValidationIssue(
                field_path="agents.TIMEOUT_SECONDS",
                severity=ValidationSeverity.ERROR,
                message="Timeout must be positive",
                current_value=agent_config.TIMEOUT_SECONDS,
                suggested_value=300,
                category="agents"
            ))
        
        # Agent-specific validation
        valid_analysis_depths = ['basic', 'detailed', 'comprehensive']
        if agent_config.STRATEGIST_ANALYSIS_DEPTH not in valid_analysis_depths:
            report.add_issue(ValidationIssue(
                field_path="agents.STRATEGIST_ANALYSIS_DEPTH",
                severity=ValidationSeverity.ERROR,
                message=f"Invalid analysis depth. Must be one of: {valid_analysis_depths}",
                current_value=agent_config.STRATEGIST_ANALYSIS_DEPTH,
                suggested_value="detailed",
                category="agents"
            ))
        
        valid_layout_algorithms = ['grid_based', 'free_form', 'template_based']
        if agent_config.FOREGROUND_LAYOUT_ALGORITHM not in valid_layout_algorithms:
            report.add_issue(ValidationIssue(
                field_path="agents.FOREGROUND_LAYOUT_ALGORITHM",
                severity=ValidationSeverity.ERROR,
                message=f"Invalid layout algorithm. Must be one of: {valid_layout_algorithms}",
                current_value=agent_config.FOREGROUND_LAYOUT_ALGORITHM,
                suggested_value="grid_based",
                category="agents"
            ))
        
        # Quality threshold validation
        if not (0.0 <= agent_config.BACKGROUND_QUALITY_THRESHOLD <= 1.0):
            report.add_issue(ValidationIssue(
                field_path="agents.BACKGROUND_QUALITY_THRESHOLD",
                severity=ValidationSeverity.ERROR,
                message="Quality threshold must be between 0.0 and 1.0",
                current_value=agent_config.BACKGROUND_QUALITY_THRESHOLD,
                suggested_value=0.8,
                category="agents"
            ))
        
        # Output formats validation
        valid_formats = ['svg', 'figma', 'png', 'pdf']
        invalid_formats = [fmt for fmt in agent_config.DEVELOPER_OUTPUT_FORMATS if fmt not in valid_formats]
        if invalid_formats:
            report.add_issue(ValidationIssue(
                field_path="agents.DEVELOPER_OUTPUT_FORMATS",
                severity=ValidationSeverity.WARNING,
                message=f"Unknown output formats: {invalid_formats}. Valid formats: {valid_formats}",
                current_value=agent_config.DEVELOPER_OUTPUT_FORMATS,
                category="agents"
            ))
    
    def _validate_security_config(self, security_config: 'SecurityConfig', report: ValidationReport) -> None:
        """Validate security configuration"""
        
        # JWT validation
        if security_config.ENABLE_AUTHENTICATION and not security_config.JWT_SECRET_KEY:
            report.add_issue(ValidationIssue(
                field_path="security.JWT_SECRET_KEY",
                severity=ValidationSeverity.CRITICAL,
                message="JWT secret key is required when authentication is enabled",
                current_value="",
                category="security"
            ))
        
        if security_config.JWT_SECRET_KEY and len(security_config.JWT_SECRET_KEY) < 32:
            report.add_issue(ValidationIssue(
                field_path="security.JWT_SECRET_KEY",
                severity=ValidationSeverity.WARNING,
                message="JWT secret key should be at least 32 characters for security",
                current_value=f"<{len(security_config.JWT_SECRET_KEY)} chars>",
                category="security"
            ))
        
        # File upload validation
        if security_config.MAX_FILE_SIZE_MB <= 0:
            report.add_issue(ValidationIssue(
                field_path="security.MAX_FILE_SIZE_MB",
                severity=ValidationSeverity.ERROR,
                message="Max file size must be positive",
                current_value=security_config.MAX_FILE_SIZE_MB,
                suggested_value=50,
                category="security"
            ))
        
        if security_config.MAX_FILE_SIZE_MB > 500:
            report.add_issue(ValidationIssue(
                field_path="security.MAX_FILE_SIZE_MB",
                severity=ValidationSeverity.WARNING,
                message="Very large file size limit may impact performance",
                current_value=security_config.MAX_FILE_SIZE_MB,
                suggested_value=100,
                category="security"
            ))
        
        # File extension validation
        if not security_config.ALLOWED_FILE_EXTENSIONS:
            report.add_issue(ValidationIssue(
                field_path="security.ALLOWED_FILE_EXTENSIONS",
                severity=ValidationSeverity.ERROR,
                message="At least one file extension must be allowed",
                suggested_value=[".png", ".jpg", ".svg"],
                category="security"
            ))
        
        # Encryption validation
        if security_config.ENCRYPT_SENSITIVE_DATA and not security_config.ENCRYPTION_KEY:
            report.add_issue(ValidationIssue(
                field_path="security.ENCRYPTION_KEY",
                severity=ValidationSeverity.CRITICAL,
                message="Encryption key is required when data encryption is enabled",
                current_value="",
                category="security"
            ))
    
    def _validate_performance_config(self, perf_config: 'PerformanceConfig', report: ValidationReport) -> None:
        """Validate performance configuration"""
        
        # Memory validation
        if perf_config.MAX_MEMORY_SIZE_MB <= 0:
            report.add_issue(ValidationIssue(
                field_path="performance.MAX_MEMORY_SIZE_MB",
                severity=ValidationSeverity.ERROR,
                message="Max memory size must be positive",
                current_value=perf_config.MAX_MEMORY_SIZE_MB,
                suggested_value=1024,
                category="performance"
            ))
        
        if perf_config.MAX_MEMORY_SIZE_MB < 256:
            report.add_issue(ValidationIssue(
                field_path="performance.MAX_MEMORY_SIZE_MB",
                severity=ValidationSeverity.WARNING,
                message="Low memory limit may cause performance issues",
                current_value=perf_config.MAX_MEMORY_SIZE_MB,
                suggested_value=512,
                category="performance"
            ))
        
        # Warning threshold validation
        if not (0.0 < perf_config.MEMORY_WARNING_THRESHOLD < 1.0):
            report.add_issue(ValidationIssue(
                field_path="performance.MEMORY_WARNING_THRESHOLD",
                severity=ValidationSeverity.ERROR,
                message="Memory warning threshold must be between 0.0 and 1.0",
                current_value=perf_config.MEMORY_WARNING_THRESHOLD,
                suggested_value=0.8,
                category="performance"
            ))
        
        # Concurrency validation
        if perf_config.MAX_CONCURRENT_CAMPAIGNS <= 0:
            report.add_issue(ValidationIssue(
                field_path="performance.MAX_CONCURRENT_CAMPAIGNS",
                severity=ValidationSeverity.ERROR,
                message="Max concurrent campaigns must be positive",
                current_value=perf_config.MAX_CONCURRENT_CAMPAIGNS,
                suggested_value=10,
                category="performance"
            ))
        
        if perf_config.WORKER_POOL_SIZE <= 0:
            report.add_issue(ValidationIssue(
                field_path="performance.WORKER_POOL_SIZE",
                severity=ValidationSeverity.ERROR,
                message="Worker pool size must be positive",
                current_value=perf_config.WORKER_POOL_SIZE,
                suggested_value=4,
                category="performance"
            ))
        
        # Cache validation
        if perf_config.ENABLE_CACHING:
            if perf_config.CACHE_TTL_SECONDS <= 0:
                report.add_issue(ValidationIssue(
                    field_path="performance.CACHE_TTL_SECONDS",
                    severity=ValidationSeverity.ERROR,
                    message="Cache TTL must be positive when caching is enabled",
                    current_value=perf_config.CACHE_TTL_SECONDS,
                    suggested_value=3600,
                    category="performance"
                ))
            
            if perf_config.CACHE_MAX_SIZE_MB <= 0:
                report.add_issue(ValidationIssue(
                    field_path="performance.CACHE_MAX_SIZE_MB",
                    severity=ValidationSeverity.ERROR,
                    message="Cache max size must be positive when caching is enabled",
                    current_value=perf_config.CACHE_MAX_SIZE_MB,
                    suggested_value=256,
                    category="performance"
                ))
    
    def _validate_config_consistency(self, config: 'BaseConfig', report: ValidationReport) -> None:
        """Validate cross-configuration consistency"""
        
        # Memory consistency check
        total_expected_memory = (
            config.performance.MAX_MEMORY_SIZE_MB + 
            config.performance.CACHE_MAX_SIZE_MB
        )
        
        if total_expected_memory > 8192:  # 8GB
            report.add_issue(ValidationIssue(
                field_path="performance.memory_total",
                severity=ValidationSeverity.WARNING,
                message=f"Total memory allocation ({total_expected_memory}MB) is very high",
                current_value=total_expected_memory,
                category="consistency"
            ))
        
        # Worker vs concurrent campaigns consistency
        campaigns_per_worker = config.performance.MAX_CONCURRENT_CAMPAIGNS / config.performance.WORKER_POOL_SIZE
        if campaigns_per_worker > 5:
            report.add_issue(ValidationIssue(
                field_path="performance.worker_campaign_ratio",
                severity=ValidationSeverity.WARNING,
                message=f"High campaigns per worker ratio ({campaigns_per_worker:.1f}). Consider increasing worker pool size",
                current_value=campaigns_per_worker,
                category="consistency"
            ))
        
        # Database connections vs workers
        if config.database.MAX_CONNECTIONS < config.performance.WORKER_POOL_SIZE:
            report.add_issue(ValidationIssue(
                field_path="database.connection_worker_mismatch",
                severity=ValidationSeverity.WARNING,
                message="Database connections may be insufficient for worker pool size",
                current_value=f"connections:{config.database.MAX_CONNECTIONS}, workers:{config.performance.WORKER_POOL_SIZE}",
                category="consistency"
            ))
    
    def _validate_environment_compatibility(self, config: 'BaseConfig', report: ValidationReport) -> None:
        """Validate environment-specific requirements"""
        environment = config.ENVIRONMENT.lower()
        
        # Development environment checks
        if environment == 'development':
            if not config.DEBUG:
                report.add_issue(ValidationIssue(
                    field_path="DEBUG",
                    severity=ValidationSeverity.INFO,
                    message="Consider enabling DEBUG in development environment",
                    current_value=config.DEBUG,
                    suggested_value=True,
                    category="environment"
                ))
        
        # Production environment checks
        elif environment == 'production':
            if config.DEBUG:
                report.add_issue(ValidationIssue(
                    field_path="DEBUG",
                    severity=ValidationSeverity.CRITICAL,
                    message="DEBUG must be False in production environment",
                    current_value=config.DEBUG,
                    suggested_value=False,
                    category="environment"
                ))
            
            if config.LOG_LEVEL == 'DEBUG':
                report.add_issue(ValidationIssue(
                    field_path="LOG_LEVEL",
                    severity=ValidationSeverity.WARNING,
                    message="DEBUG log level not recommended for production",
                    current_value=config.LOG_LEVEL,
                    suggested_value="INFO",
                    category="environment"
                ))
            
            if not config.security.ENABLE_AUTHENTICATION:
                report.add_issue(ValidationIssue(
                    field_path="security.ENABLE_AUTHENTICATION",
                    severity=ValidationSeverity.CRITICAL,
                    message="Authentication must be enabled in production",
                    current_value=config.security.ENABLE_AUTHENTICATION,
                    suggested_value=True,
                    category="environment"
                ))
    
    def _validate_directories(self, config: 'BaseConfig', report: ValidationReport) -> None:
        """Validate directory configuration"""
        
        directory_fields = {
            'UPLOAD_DIR': config.UPLOAD_DIR,
            'OUTPUT_DIR': config.OUTPUT_DIR,
            'TEMP_DIR': config.TEMP_DIR,
            'LOGS_DIR': config.LOGS_DIR,
            'EXPORTS_DIR': config.EXPORTS_DIR
        }
        
        for field_name, dir_path in directory_fields.items():
            # Check if directory path is empty
            if not dir_path or len(dir_path.strip()) == 0:
                report.add_issue(ValidationIssue(
                    field_path=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} cannot be empty",
                    current_value=dir_path,
                    category="directories"
                ))
                continue
            
            # Check for path traversal attacks
            if '..' in dir_path or dir_path.startswith('/'):
                report.add_issue(ValidationIssue(
                    field_path=field_name,
                    severity=ValidationSeverity.WARNING,
                    message=f"Potentially unsafe directory path: {dir_path}",
                    current_value=dir_path,
                    category="directories"
                ))
            
            # Check if directory is writable (if it exists)
            path_obj = Path(dir_path)
            try:
                if path_obj.exists():
                    # Try to create a test file
                    test_file = path_obj / '.write_test'
                    test_file.touch()
                    test_file.unlink()
                elif not path_obj.parent.exists():
                    report.add_issue(ValidationIssue(
                        field_path=field_name,
                        severity=ValidationSeverity.WARNING,
                        message=f"Parent directory does not exist: {path_obj.parent}",
                        current_value=dir_path,
                        category="directories"
                    ))
            except PermissionError:
                report.add_issue(ValidationIssue(
                    field_path=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Directory is not writable: {dir_path}",
                    current_value=dir_path,
                    category="directories"
                ))
            except Exception as e:
                report.add_issue(ValidationIssue(
                    field_path=field_name,
                    severity=ValidationSeverity.WARNING,
                    message=f"Could not validate directory access: {str(e)}",
                    current_value=dir_path,
                    category="directories"
                ))
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules configuration"""
        return {
            'required_fields': [
                'ENVIRONMENT', 'LOG_LEVEL', 'API_PORT',
                'database.HOST', 'database.NAME'
            ],
            'numeric_ranges': {
                'API_PORT': (1, 65535),
                'SESSION_TIMEOUT_HOURS': (0.1, 168),  # Max 1 week
                'MAX_ITERATIONS': (1, 50),
                'models.LLM_TEMPERATURE': (0.0, 2.0),
                'performance.MEMORY_WARNING_THRESHOLD': (0.1, 0.99)
            },
            'environment_requirements': {
                'production': {
                    'required_security': ['ENABLE_AUTHENTICATION', 'ENABLE_RATE_LIMITING'],
                    'forbidden_debug': True,
                    'min_workers': 2
                },
                'development': {
                    'recommended_debug': True,
                    'recommended_log_level': 'DEBUG'
                }
            }
        }
    
    def _generate_config_summary(self, config: 'BaseConfig') -> Dict[str, Any]:
        """Generate configuration summary for report"""
        return {
            'environment': config.ENVIRONMENT,
            'debug_mode': config.DEBUG,
            'log_level': config.LOG_LEVEL,
            'api_workers': config.API_WORKERS,
            'max_concurrent_campaigns': config.performance.MAX_CONCURRENT_CAMPAIGNS,
            'authentication_enabled': config.security.ENABLE_AUTHENTICATION,
            'caching_enabled': config.performance.ENABLE_CACHING,
            'database_type': 'postgresql' if config.database.PORT == 5432 else 'custom',
            'total_memory_mb': config.performance.MAX_MEMORY_SIZE_MB + config.performance.CACHE_MAX_SIZE_MB
        }
    
    def validate_field(self, config: 'BaseConfig', field_path: str, expected_type: type = None) -> List[ValidationIssue]:
        """Validate specific configuration field"""
        issues = []
        
        try:
            # Navigate to field using dot notation
            value = config
            path_parts = field_path.split('.')
            
            for part in path_parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    issues.append(ValidationIssue(
                        field_path=field_path,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field {field_path} does not exist",
                        category="field_validation"
                    ))
                    return issues
            
            # Type validation
            if expected_type and not isinstance(value, expected_type):
                issues.append(ValidationIssue(
                    field_path=field_path,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field {field_path} should be of type {expected_type.__name__}, got {type(value).__name__}",
                    current_value=type(value).__name__,
                    suggested_value=expected_type.__name__,
                    category="field_validation"
                ))
            
        except Exception as e:
            issues.append(ValidationIssue(
                field_path=field_path,
                severity=ValidationSeverity.ERROR,
                message=f"Error validating field {field_path}: {str(e)}",
                category="field_validation"
            ))
        
        return issues
    
    def validate_api_keys(self, config: 'BaseConfig') -> List[ValidationIssue]:
        """Validate API key configuration"""
        issues = []
        
        # Check required API keys
        api_key_fields = [
            ('models.DEFAULT_LLM_API_KEY', 'LLM functionality'),
            ('models.T2I_API_KEY', 'Text-to-Image generation'),
            ('models.MLLM_API_KEY', 'Multimodal analysis')
        ]
        
        for field_path, description in api_key_fields:
            key_value = self._get_nested_value(config, field_path)
            
            if not key_value or len(key_value.strip()) == 0:
                severity = ValidationSeverity.WARNING if config.ENVIRONMENT == 'development' else ValidationSeverity.CRITICAL
                issues.append(ValidationIssue(
                    field_path=field_path,
                    severity=severity,
                    message=f"API key for {description} is not configured",
                    current_value="",
                    category="api_keys"
                ))
            elif len(key_value) < 10:
                issues.append(ValidationIssue(
                    field_path=field_path,
                    severity=ValidationSeverity.WARNING,
                    message=f"API key for {description} seems too short",
                    current_value=f"<{len(key_value)} chars>",
                    category="api_keys"
                ))
        
        return issues
    
    def _get_nested_value(self, obj: Any, field_path: str) -> Any:
        """Get nested value using dot notation"""
        try:
            value = obj
            for part in field_path.split('.'):
                value = getattr(value, part)
            return value
        except:
            return None
    
    def quick_validate(self, config: 'BaseConfig') -> Tuple[bool, List[str]]:
        """Quick validation returning simple pass/fail and error messages"""
        report = self.validate_full_config(config)
        
        is_valid = not report.has_critical_issues() and not report.has_errors()
        
        error_messages = []
        for issue in report.issues:
            if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
                error_messages.append(f"{issue.field_path}: {issue.message}")
        
        return is_valid, error_messages
    
    def suggest_fixes(self, config: 'BaseConfig') -> Dict[str, Any]:
        """Suggest automatic fixes for common configuration issues"""
        report = self.validate_full_config(config)
        suggestions = {}
        
        for issue in report.issues:
            if issue.suggested_value is not None and issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                suggestions[issue.field_path] = issue.suggested_value
        
        return suggestions
    
    def auto_fix_config(self, config: 'BaseConfig') -> Tuple['BaseConfig', List[str]]:
        """Automatically fix common configuration issues"""
        fixes_applied = []
        suggestions = self.suggest_fixes(config)
        
        for field_path, suggested_value in suggestions.items():
            try:
                # Apply fix
                self._set_nested_value(config, field_path, suggested_value)
                fixes_applied.append(f"Fixed {field_path}: set to {suggested_value}")
            except Exception as e:
                fixes_applied.append(f"Could not fix {field_path}: {str(e)}")
        
        return config, fixes_applied
    
    def _set_nested_value(self, obj: Any, field_path: str, value: Any) -> None:
        """Set nested value using dot notation"""
        path_parts = field_path.split('.')
        target_obj = obj
        
        # Navigate to parent object
        for part in path_parts[:-1]:
            target_obj = getattr(target_obj, part)
        
        # Set final value
        setattr(target_obj, path_parts[-1], value)

class ValidationError(Exception):
    """Configuration validation error"""
    
    def __init__(self, message: str, issues: List[ValidationIssue] = None):
        super().__init__(message)
        self.issues = issues or []
        self.message = message
    
    def __str__(self) -> str:
        if self.issues:
            issue_summary = f" ({len(self.issues)} validation issues)"
            return self.message + issue_summary
        return self.message

# Additional validation utilities

class ConfigValidator:
    """Comprehensive configuration validator"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def validate_production_readiness(self, config: 'BaseConfig') -> ValidationReport:
        """Validate if configuration is ready for production deployment"""
        report = ValidationReport(is_valid=True, total_issues=0)
        
        # Critical production requirements
        production_requirements = [
            ('DEBUG', False, "DEBUG must be disabled"),
            ('security.ENABLE_AUTHENTICATION', True, "Authentication must be enabled"),
            ('security.ENABLE_RATE_LIMITING', True, "Rate limiting must be enabled"),
            ('security.ENABLE_AUDIT_LOGGING', True, "Audit logging must be enabled"),
            ('models.DEFAULT_LLM_API_KEY', lambda x: x and len(x) > 10, "Valid LLM API key required"),
            ('database.PASSWORD', lambda x: x and len(x) > 8, "Strong database password required")
        ]
        
        for field_path, requirement, message in production_requirements:
            value = self._get_nested_value(config, field_path)
            
            if callable(requirement):
                is_valid = requirement(value)
            else:
                is_valid = value == requirement
            
            if not is_valid:
                report.add_issue(ValidationIssue(
                    field_path=field_path,
                    severity=ValidationSeverity.CRITICAL,
                    message=message,
                    current_value=str(value)[:50] if value else None,
                    category="production_readiness"
                ))
        
        # Performance requirements for production
        perf_requirements = [
            ('API_WORKERS', lambda x: x >= 2, "At least 2 API workers recommended"),
            ('performance.WORKER_POOL_SIZE', lambda x: x >= 4, "At least 4 workers recommended"),
            ('database.MAX_CONNECTIONS', lambda x: x >= 10, "At least 10 DB connections recommended"),
            ('performance.ENABLE_CACHING', True, "Caching should be enabled for performance")
        ]
        
        for field_path, requirement, message in perf_requirements:
            value = self._get_nested_value(config, field_path)
            
            if callable(requirement):
                is_valid = requirement(value)
            else:
                is_valid = value == requirement
            
            if not is_valid:
                report.add_issue(ValidationIssue(
                    field_path=field_path,
                    severity=ValidationSeverity.WARNING,
                    message=message,
                    current_value=value,
                    category="production_readiness"
                ))
        
        return report
    
    def validate_security_configuration(self, config: 'BaseConfig') -> ValidationReport:
        """Perform detailed security configuration validation"""
        report = ValidationReport(is_valid=True, total_issues=0)
        
        # JWT security validation
        if config.security.ENABLE_AUTHENTICATION:
            if not config.security.JWT_SECRET_KEY:
                report.add_issue(ValidationIssue(
                    field_path="security.JWT_SECRET_KEY",
                    severity=ValidationSeverity.CRITICAL,
                    message="JWT secret key is required for authentication",
                    category="security"
                ))
            elif len(config.security.JWT_SECRET_KEY) < 32:
                report.add_issue(ValidationIssue(
                    field_path="security.JWT_SECRET_KEY",
                    severity=ValidationSeverity.ERROR,
                    message="JWT secret key should be at least 32 characters",
                    current_value=f"<{len(config.security.JWT_SECRET_KEY)} chars>",
                    category="security"
                ))
            
            # Check for weak JWT expiration
            if config.security.JWT_EXPIRATION_HOURS > 168:  # 1 week
                report.add_issue(ValidationIssue(
                    field_path="security.JWT_EXPIRATION_HOURS",
                    severity=ValidationSeverity.WARNING,
                    message="Very long JWT expiration may be a security risk",
                    current_value=config.security.JWT_EXPIRATION_HOURS,
                    suggested_value=24,
                    category="security"
                ))
        
        # File upload security
        dangerous_extensions = ['.exe', '.bat', '.sh', '.php', '.jsp', '.asp']
        allowed_extensions = config.security.ALLOWED_FILE_EXTENSIONS
        
        for ext in allowed_extensions:
            if ext.lower() in dangerous_extensions:
                report.add_issue(ValidationIssue(
                    field_path="security.ALLOWED_FILE_EXTENSIONS",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Dangerous file extension allowed: {ext}",
                    current_value=allowed_extensions,
                    category="security"
                ))
        
        # CORS validation
        if config.security.ENABLE_CORS and "*" in config.security.ALLOWED_ORIGINS:
            if config.ENVIRONMENT == 'production':
                report.add_issue(ValidationIssue(
                    field_path="security.ALLOWED_ORIGINS",
                    severity=ValidationSeverity.WARNING,
                    message="Wildcard CORS origin not recommended for production",
                    current_value=config.security.ALLOWED_ORIGINS,
                    category="security"
                ))
        
        return report
    
    def validate_performance_configuration(self, config: 'BaseConfig') -> ValidationReport:
        """Validate performance-related configuration"""
        report = ValidationReport(is_valid=True, total_issues=0)
        
        # Memory allocation validation
        total_memory = config.performance.MAX_MEMORY_SIZE_MB
        cache_memory = config.performance.CACHE_MAX_SIZE_MB
        
        if cache_memory > total_memory * 0.5:
            report.add_issue(ValidationIssue(
                field_path="performance.CACHE_MAX_SIZE_MB",
                severity=ValidationSeverity.WARNING,
                message="Cache memory is more than 50% of total memory allocation",
                current_value=cache_memory,
                suggested_value=int(total_memory * 0.3),
                category="performance"
            ))
        
        # Worker configuration validation
        optimal_workers = min(8, os.cpu_count() or 4)
        if config.performance.WORKER_POOL_SIZE > optimal_workers:
            report.add_issue(ValidationIssue(
                field_path="performance.WORKER_POOL_SIZE",
                severity=ValidationSeverity.WARNING,
                message=f"Worker pool size exceeds optimal CPU count ({optimal_workers})",
                current_value=config.performance.WORKER_POOL_SIZE,
                suggested_value=optimal_workers,
                category="performance"
            ))
        
        # Concurrent campaigns vs resources
        campaigns_per_worker = config.performance.MAX_CONCURRENT_CAMPAIGNS / config.performance.WORKER_POOL_SIZE
        if campaigns_per_worker > 10:
            report.add_issue(ValidationIssue(
                field_path="performance.MAX_CONCURRENT_CAMPAIGNS",
                severity=ValidationSeverity.WARNING,
                message=f"High campaigns per worker ratio ({campaigns_per_worker:.1f})",
                current_value=config.performance.MAX_CONCURRENT_CAMPAIGNS,
                suggested_value=config.performance.WORKER_POOL_SIZE * 5,
                category="performance"
            ))
        
        return report
    
    def validate_model_configuration(self, config: 'BaseConfig') -> ValidationReport:
        """Validate AI model configuration"""
        report = ValidationReport(is_valid=True, total_issues=0)
        
        models = config.models
        
        # API endpoint validation
        if models.DEFAULT_LLM_BASE_URL and not self._is_valid_url(models.DEFAULT_LLM_BASE_URL):
            report.add_issue(ValidationIssue(
                field_path="models.DEFAULT_LLM_BASE_URL",
                severity=ValidationSeverity.ERROR,
                message="Invalid LLM API base URL",
                current_value=models.DEFAULT_LLM_BASE_URL,
                category="models"
            ))
        
        # Rate limiting validation
        rate_limits = [
            ('LLM_REQUESTS_PER_MINUTE', models.LLM_REQUESTS_PER_MINUTE),
            ('T2I_REQUESTS_PER_MINUTE', models.T2I_REQUESTS_PER_MINUTE),
            ('MLLM_REQUESTS_PER_MINUTE', models.MLLM_REQUESTS_PER_MINUTE)
        ]
        
        for field_name, rate_limit in rate_limits:
            if rate_limit <= 0:
                report.add_issue(ValidationIssue(
                    field_path=f"models.{field_name}",
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} must be positive",
                    current_value=rate_limit,
                    suggested_value=60,
                    category="models"
                ))
            elif rate_limit > 1000:
                report.add_issue(ValidationIssue(
                    field_path=f"models.{field_name}",
                    severity=ValidationSeverity.WARNING,
                    message=f"Very high rate limit for {field_name}",
                    current_value=rate_limit,
                    category="models"
                ))
        
        # Model parameter validation
        if models.T2I_STEPS <= 0:
            report.add_issue(ValidationIssue(
                field_path="models.T2I_STEPS",
                severity=ValidationSeverity.ERROR,
                message="T2I steps must be positive",
                current_value=models.T2I_STEPS,
                suggested_value=4,
                category="models"
            ))
        
        if models.T2I_GUIDANCE_SCALE < 1.0 or models.T2I_GUIDANCE_SCALE > 20.0:
            report.add_issue(ValidationIssue(
                field_path="models.T2I_GUIDANCE_SCALE",
                severity=ValidationSeverity.WARNING,
                message="T2I guidance scale should typically be between 1.0 and 20.0",
                current_value=models.T2I_GUIDANCE_SCALE,
                suggested_value=7.5,
                category="models"
            ))
        
        return report
    
    def generate_config_health_report(self, config: 'BaseConfig') -> Dict[str, Any]:
        """Generate comprehensive configuration health report"""
        full_report = self.validate_full_config(config)
        
        health_score = self._calculate_health_score(full_report)
        
        return {
            'overall_health_score': health_score,
            'health_grade': self._get_health_grade(health_score),
            'total_issues': full_report.total_issues,
            'issues_by_severity': full_report.issues_by_severity,
            'critical_issues_count': len(full_report.get_issues_by_severity(ValidationSeverity.CRITICAL)),
            'error_issues_count': len(full_report.get_issues_by_severity(ValidationSeverity.ERROR)),
            'warning_issues_count': len(full_report.get_issues_by_severity(ValidationSeverity.WARNING)),
            'is_production_ready': not full_report.has_critical_issues() and not full_report.has_errors(),
            'environment': config.ENVIRONMENT,
            'recommendations': self._generate_recommendations(config, full_report)
        }
    
    def _calculate_health_score(self, report: ValidationReport) -> float:
        """Calculate configuration health score (0-100)"""
        if report.total_issues == 0:
            return 100.0
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.CRITICAL: -25,
            ValidationSeverity.ERROR: -10,
            ValidationSeverity.WARNING: -3,
            ValidationSeverity.INFO: -1
        }
        
        total_deduction = 0
        for severity, count in report.issues_by_severity.items():
            total_deduction += severity_weights.get(severity, 0) * count
        
        score = max(0, 100 + total_deduction)
        return round(score, 1)
    
    def _get_health_grade(self, score: float) -> str:
        """Convert health score to grade"""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C+"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, config: 'BaseConfig', report: ValidationReport) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Critical issue recommendations
        critical_issues = report.get_issues_by_severity(ValidationSeverity.CRITICAL)
        if critical_issues:
            recommendations.append("🚨 Address critical issues immediately before deployment")
        
        # Environment-specific recommendations
        if config.ENVIRONMENT == 'production':
            error_count = len(report.get_issues_by_severity(ValidationSeverity.ERROR))
            if error_count > 0:
                recommendations.append("🔧 Fix all errors before production deployment")
            
            warning_count = len(report.get_issues_by_severity(ValidationSeverity.WARNING))
            if warning_count > 5:
                recommendations.append("⚠️ Consider addressing warnings for optimal production performance")
        
        # Performance recommendations
        perf_issues = report.get_issues_by_category("performance")
        if len(perf_issues) > 3:
            recommendations.append("🚀 Review performance configuration for better system efficiency")
        
        # Security recommendations
        security_issues = report.get_issues_by_category("security")
        if security_issues:
            recommendations.append("🔒 Review security configuration to ensure proper protection")
        
        # General recommendations
        if report.total_issues == 0:
            recommendations.append("✨ Configuration looks great! Consider periodic reviews as requirements change")
        elif report.total_issues < 5:
            recommendations.append("👍 Configuration is mostly good with minor issues to address")
        else:
            recommendations.append("🔍 Consider using configuration templates or auto-fix for faster setup")
        
        return recommendations

# Factory functions for common validation scenarios

def validate_for_deployment(config: 'BaseConfig', deployment_env: str) -> ValidationReport:
    """Validate configuration for specific deployment environment"""
    validator = ConfigValidator()
    
    if deployment_env.lower() == 'production':
        return validator.validate_production_readiness(config)
    else:
        return validator.validate_full_config(config)

def quick_config_check(config: 'BaseConfig') -> bool:
    """Quick configuration check returning simple pass/fail"""
    validator = ConfigValidator()
    is_valid, _ = validator.quick_validate(config)
    return is_valid

def get_config_issues(config: 'BaseConfig', severity_filter: Optional[ValidationSeverity] = None) -> List[str]:
    """Get list of configuration issues as strings"""
    validator = ConfigValidator()
    report = validator.validate_full_config(config)
    
    issues = report.issues
    if severity_filter:
        issues = [issue for issue in issues if issue.severity == severity_filter]
    
    return [f"{issue.field_path}: {issue.message}" for issue in issues]

def auto_fix_common_issues(config: 'BaseConfig') -> Tuple['BaseConfig', List[str]]:
    """Automatically fix common configuration issues"""
    validator = ConfigValidator()
    return validator.auto_fix_config(config)
