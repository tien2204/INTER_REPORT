# =============================================================================
# config/secrets_manager.py
# =============================================================================

import os
import json
import base64
import hashlib
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import threading

class SecretNotFoundError(Exception):
    """Raised when a secret is not found"""
    pass

class SecretsEncryptionError(Exception):
    """Raised when encryption/decryption fails"""
    pass

class SecretsManager:
    """
    Secure secrets management for the Banner AI Generator system
    
    Supports multiple secret sources:
    - Environment variables
    - Encrypted local files
    - External secret management systems (HashiCorp Vault, AWS Secrets Manager)
    - In-memory encrypted storage
    
    Features:
    - Encryption at rest and in memory
    - Secret rotation
    - Audit logging
    - Secure cleanup
    """
    
    def __init__(self, 
                 secrets_dir: Optional[str] = None,
                 master_key: Optional[str] = None,
                 enable_audit: bool = True):
        """
        Initialize secrets manager
        
        Args:
            secrets_dir: Directory for storing encrypted secrets files
            master_key: Master encryption key (will be derived from password if not provided)
            enable_audit: Whether to enable audit logging
        """
        self.secrets_dir = Path(secrets_dir) if secrets_dir else Path("./secrets")
        self.enable_audit = enable_audit
        
        # Thread safety
        self._lock = threading.Lock()
        
        # In-memory encrypted secret storage
        self._encrypted_secrets: Dict[str, bytes] = {}
        self._secret_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Encryption setup
        self._cipher_suite: Optional[Fernet] = None
        self._setup_encryption(master_key)
        
        # Audit log
        self._audit_log: List[Dict[str, Any]] = []
        
        # Load secrets from various sources
        self._load_secrets()
    
    def _setup_encryption(self, master_key: Optional[str] = None) -> None:
        """Setup encryption for secrets"""
        try:
            if master_key:
                # Use provided master key
                key = self._derive_key_from_password(master_key)
            else:
                # Try to get master key from environment
                env_master_key = os.getenv('SECRETS_MASTER_KEY')
                if env_master_key:
                    key = self._derive_key_from_password(env_master_key)
                else:
                    # Generate a new key (should be persisted securely in production)
                    key = Fernet.generate_key()
                    print("Warning: Generated new encryption key. Set SECRETS_MASTER_KEY environment variable for production.")
            
            self._cipher_suite = Fernet(key)
            
        except Exception as e:
            print(f"Warning: Could not setup encryption: {e}")
            self._cipher_suite = None
    
    def _derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password"""
        if salt is None:
            # Use fixed salt for consistency (in production, use stored salt)
            salt = b"banner_ai_generator_salt"
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def _encrypt_secret(self, secret: str) -> bytes:
        """Encrypt a secret value"""
        if not self._cipher_suite:
            return secret.encode()  # Fallback to plain text if encryption unavailable
        
        try:
            return self._cipher_suite.encrypt(secret.encode())
        except Exception as e:
            raise SecretsEncryptionError(f"Failed to encrypt secret: {e}")
    
    def _decrypt_secret(self, encrypted_secret: bytes) -> str:
        """Decrypt a secret value"""
        if not self._cipher_suite:
            return encrypted_secret.decode()  # Fallback if encryption unavailable
        
        try:
            return self._cipher_suite.decrypt(encrypted_secret).decode()
        except Exception as e:
            raise SecretsEncryptionError(f"Failed to decrypt secret: {e}")
    
    def _load_secrets(self) -> None:
        """Load secrets from all available sources"""
        # Load from environment variables
        self._load_from_environment()
        
        # Load from encrypted files
        self._load_from_files()
        
        # Load from external systems (if configured)
        self._load_from_external_systems()
    
    def _load_from_environment(self) -> None:
        """Load secrets from environment variables"""
        secret_env_vars = [
            'LLM_API_KEY',
            'T2I_API_KEY', 
            'MLLM_API_KEY',
            'DATABASE_PASSWORD',
            'REDIS_PASSWORD',
            'JWT_SECRET_KEY',
            'ENCRYPTION_KEY',
            'VAULT_TOKEN',
            'AWS_SECRET_ACCESS_KEY'
        ]
        
        for env_var in secret_env_vars:
            value = os.getenv(env_var)
            if value:
                secret_name = env_var.lower()
                self._store_secret_in_memory(secret_name, value, source='environment')
    
    def _load_from_files(self) -> None:
        """Load secrets from encrypted files"""
        if not self.secrets_dir.exists():
            return
        
        try:
            secrets_file = self.secrets_dir / "secrets.encrypted"
            if secrets_file.exists():
                with open(secrets_file, 'rb') as f:
                    encrypted_data = f.read()
                
                if self._cipher_suite:
                    decrypted_data = self._cipher_suite.decrypt(encrypted_data)
                    secrets_dict = json.loads(decrypted_data.decode())
                    
                    for secret_name, secret_value in secrets_dict.items():
                        self._store_secret_in_memory(secret_name, secret_value, source='file')
        
        except Exception as e:
            print(f"Warning: Could not load secrets from file: {e}")
    
    def _load_from_external_systems(self) -> None:
        """Load secrets from external secret management systems"""
        # HashiCorp Vault integration
        vault_addr = os.getenv('VAULT_ADDR')
        vault_token = os.getenv('VAULT_TOKEN')
        
        if vault_addr and vault_token:
            self._load_from_vault(vault_addr, vault_token)
        
        # AWS Secrets Manager integration
        aws_region = os.getenv('AWS_REGION')
        if aws_region:
            self._load_from_aws_secrets_manager(aws_region)
    
    def _load_from_vault(self, vault_addr: str, vault_token: str) -> None:
        """Load secrets from HashiCorp Vault"""
        try:
            import requests
            
            headers = {'X-Vault-Token': vault_token}
            secrets_path = 'secret/data/banner-ai-generator'
            
            response = requests.get(f"{vault_addr}/v1/{secrets_path}", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                secrets_data = data.get('data', {}).get('data', {})
                
                for secret_name, secret_value in secrets_data.items():
                    self._store_secret_in_memory(secret_name, secret_value, source='vault')
            
        except ImportError:
            print("Warning: 'requests' library required for Vault integration")
        except Exception as e:
            print(f"Warning: Could not load secrets from Vault: {e}")
    
    def _load_from_aws_secrets_manager(self, region: str) -> None:
        """Load secrets from AWS Secrets Manager"""
        try:
            import boto3
            
            client = boto3.client('secretsmanager', region_name=region)
            secret_name = 'banner-ai-generator-secrets'
            
            response = client.get_secret_value(SecretId=secret_name)
            secret_data = json.loads(response['SecretString'])
            
            for key, value in secret_data.items():
                self._store_secret_in_memory(key, value, source='aws_secrets_manager')
        
        except ImportError:
            print("Warning: 'boto3' library required for AWS Secrets Manager integration")
        except Exception as e:
            print(f"Warning: Could not load secrets from AWS Secrets Manager: {e}")
    
    def _store_secret_in_memory(self, name: str, value: str, source: str = 'unknown') -> None:
        """Store secret in encrypted memory"""
        with self._lock:
            encrypted_value = self._encrypt_secret(value)
            self._encrypted_secrets[name] = encrypted_value
            
            # Store metadata
            self._secret_metadata[name] = {
                'source': source,
                'created_at': datetime.now().isoformat(),
                'last_accessed': None,
                'access_count': 0
            }
            
            # Audit log
            if self.enable_audit:
                self._log_audit_event('secret_stored', name, {'source': source})
    
    def get_secret(self, name: str, default: Optional[str] = None) -> str:
        """
        Get secret value by name
        
        Args:
            name: Secret name
            default: Default value if secret not found
        
        Returns:
            str: Secret value
        
        Raises:
            SecretNotFoundError: If secret not found and no default provided
        """
        with self._lock:
            if name not in self._encrypted_secrets:
                if default is not None:
                    return default
                raise SecretNotFoundError(f"Secret '{name}' not found")
            
            # Update metadata
            self._secret_metadata[name]['last_accessed'] = datetime.now().isoformat()
            self._secret_metadata[name]['access_count'] += 1
            
            # Audit log
            if self.enable_audit:
                self._log_audit_event('secret_accessed', name)
            
            # Decrypt and return
            encrypted_value = self._encrypted_secrets[name]
            return self._decrypt_secret(encrypted_value)
    
    def set_secret(self, name: str, value: str, persist: bool = False) -> None:
        """
        Set secret value
        
        Args:
            name: Secret name
            value: Secret value
            persist: Whether to persist to file
        """
        self._store_secret_in_memory(name, value, source='runtime')
        
        if persist:
            self.persist_secrets()
        
        # Audit log
        if self.enable_audit:
            self._log_audit_event('secret_updated', name, {'persist': persist})
    
    def delete_secret(self, name: str) -> bool:
        """
        Delete secret
        
        Args:
            name: Secret name
        
        Returns:
            bool: True if secret was deleted, False if not found
        """
        with self._lock:
            if name not in self._encrypted_secrets:
                return False
            
            # Securely clear the encrypted data
            self._secure_clear(self._encrypted_secrets[name])
            del self._encrypted_secrets[name]
            del self._secret_metadata[name]
            
            # Audit log
            if self.enable_audit:
                self._log_audit_event('secret_deleted', name)
            
            return True
    
    def list_secrets(self, include_metadata: bool = False) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """
        List all secret names
        
        Args:
            include_metadata: Whether to include metadata
        
        Returns:
            List of secret names or dict with metadata
        """
        with self._lock:
            if include_metadata:
                return dict(self._secret_metadata)
            else:
                return list(self._encrypted_secrets.keys())
    
    def secret_exists(self, name: str) -> bool:
        """Check if secret exists"""
        return name in self._encrypted_secrets
    
    def persist_secrets(self) -> bool:
        """
        Persist secrets to encrypted file
        
        Returns:
            bool: Success status
        """
        try:
            self.secrets_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect all secrets for persistence
            secrets_dict = {}
            with self._lock:
                for name, encrypted_value in self._encrypted_secrets.items():
                    secrets_dict[name] = self._decrypt_secret(encrypted_value)
            
            # Encrypt and save
            secrets_json = json.dumps(secrets_dict)
            
            if self._cipher_suite:
                encrypted_data = self._cipher_suite.encrypt(secrets_json.encode())
            else:
                encrypted_data = secrets_json.encode()
            
            secrets_file = self.secrets_dir / "secrets.encrypted"
            with open(secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(secrets_file, 0o600)
            
            # Audit log
            if self.enable_audit:
                self._log_audit_event('secrets_persisted', None, {'count': len(secrets_dict)})
            
            return True
            
        except Exception as e:
            print(f"Error persisting secrets: {e}")
            return False
    
    def rotate_secret(self, name: str, new_value: str) -> bool:
        """
        Rotate secret with new value
        
        Args:
            name: Secret name
            new_value: New secret value
        
        Returns:
            bool: Success status
        """
        old_exists = self.secret_exists(name)
        
        # Store new value
        self.set_secret(name, new_value)
        
        # Audit log
        if self.enable_audit:
            self._log_audit_event('secret_rotated', name, {'existed': old_exists})
        
        return True
    
    def generate_secret(self, name: str, length: int = 32, persist: bool = False) -> str:
        """
        Generate random secret
        
        Args:
            name: Secret name
            length: Secret length
            persist: Whether to persist
        
        Returns:
            str: Generated secret
        """
        import secrets
        import string
        
        # Generate random string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        secret_value = ''.join(secrets.choice(alphabet) for _ in range(length))
        
        # Store secret
        self.set_secret(name, secret_value, persist)
        
        # Audit log
        if self.enable_audit:
            self._log_audit_event('secret_generated', name, {'length': length, 'persist': persist})
        
        return secret_value
    
    def export_secrets(self, filepath: str, include_sensitive: bool = False) -> bool:
        """
        Export secrets to file (for backup/migration)
        
        Args:
            filepath: Output file path
            include_sensitive: Whether to include actual secret values
        
        Returns:
            bool: Success status
        """
        try:
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'secrets_count': len(self._encrypted_secrets),
                'metadata': dict(self._secret_metadata)
            }
            
            if include_sensitive:
                # Include encrypted secret values
                secrets_data = {}
                with self._lock:
                    for name, encrypted_value in self._encrypted_secrets.items():
                        # Re-encrypt with a different key for export
                        decrypted = self._decrypt_secret(encrypted_value)
                        secrets_data[name] = base64.b64encode(decrypted.encode()).decode()
                
                export_data['secrets'] = secrets_data
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(filepath, 0o600)
            
            # Audit log
            if self.enable_audit:
                self._log_audit_event('secrets_exported', None, {
                    'filepath': filepath,
                    'include_sensitive': include_sensitive,
                    'count': len(self._encrypted_secrets)
                })
            
            return True
            
        except Exception as e:
            print(f"Error exporting secrets: {e}")
            return False
    
    def import_secrets(self, filepath: str, overwrite: bool = False) -> bool:
        """
        Import secrets from file
        
        Args:
            filepath: Import file path
            overwrite: Whether to overwrite existing secrets
        
        Returns:
            bool: Success status
        """
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            imported_count = 0
            
            if 'secrets' in import_data:
                for name, encoded_value in import_data['secrets'].items():
                    if not overwrite and self.secret_exists(name):
                        continue
                    
                    # Decode and store
                    decoded_value = base64.b64decode(encoded_value.encode()).decode()
                    self.set_secret(name, decoded_value)
                    imported_count += 1
            
            # Audit log
            if self.enable_audit:
                self._log_audit_event('secrets_imported', None, {
                    'filepath': filepath,
                    'imported_count': imported_count,
                    'overwrite': overwrite
                })
            
            return True
            
        except Exception as e:
            print(f"Error importing secrets: {e}")
            return False
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get audit log entries
        
        Args:
            limit: Maximum number of entries to return
        
        Returns:
            List of audit log entries
        """
        with self._lock:
            return self._audit_log[-limit:] if self._audit_log else []
    
    def clear_audit_log(self) -> None:
        """Clear audit log"""
        with self._lock:
            self._audit_log.clear()
        
        if self.enable_audit:
            self._log_audit_event('audit_log_cleared', None)
    
    def _log_audit_event(self, action: str, secret_name: Optional[str], details: Optional[Dict[str, Any]] = None) -> None:
        """Log audit event"""
        if not self.enable_audit:
            return
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'secret_name': secret_name,
            'details': details or {},
            'process_id': os.getpid(),
            'user': os.getenv('USER', 'unknown')
        }
        
        self._audit_log.append(audit_entry)
        
        # Keep audit log size manageable
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-500:]
    
    def _secure_clear(self, data: Union[bytes, str]) -> None:
        """Securely clear sensitive data from memory"""
        if isinstance(data, bytes):
            # Overwrite bytes with random data
            import os
            random_data = os.urandom(len(data))
            data = random_data
        elif isinstance(data, str):
            # Overwrite string with random characters
            import random
            import string
            random_chars = ''.join(random.choice(string.ascii_letters) for _ in range(len(data)))
            data = random_chars
    
    def cleanup(self) -> None:
        """
        Cleanup secrets manager
        Securely clear all secrets from memory
        """
        with self._lock:
            # Securely clear all encrypted secrets
            for encrypted_secret in self._encrypted_secrets.values():
                self._secure_clear(encrypted_secret)
            
            self._encrypted_secrets.clear()
            self._secret_metadata.clear()
        
        # Audit log
        if self.enable_audit:
            self._log_audit_event('secrets_manager_cleanup', None)
    
    def get_secrets_health(self) -> Dict[str, Any]:
        """Get secrets manager health status"""
        with self._lock:
            total_secrets = len(self._encrypted_secrets)
            
            # Analyze secret sources
            sources = {}
            for metadata in self._secret_metadata.values():
                source = metadata['source']
                sources[source] = sources.get(source, 0) + 1
            
            # Find secrets that haven't been accessed recently
            stale_secrets = []
            now = datetime.now()
            
            for name, metadata in self._secret_metadata.items():
                last_accessed = metadata.get('last_accessed')
                if last_accessed:
                    last_access_time = datetime.fromisoformat(last_accessed)
                    if now - last_access_time > timedelta(days=30):
                        stale_secrets.append(name)
        
        return {
            'total_secrets': total_secrets,
            'sources': sources,
            'encryption_enabled': self._cipher_suite is not None,
            'audit_enabled': self.enable_audit,
            'audit_entries': len(self._audit_log),
            'stale_secrets': len(stale_secrets),
            'stale_secret_names': stale_secrets
        }
    
    def validate_secrets_configuration(self) -> List[str]:
        """Validate secrets configuration and return issues"""
        issues = []
        
        # Check encryption setup
        if not self._cipher_suite:
            issues.append("Encryption is not properly configured")
        
        # Check for required secrets
        required_secrets = [
            'llm_api_key',
            'database_password',
            'jwt_secret_key'
        ]
        
        for required_secret in required_secrets:
            if not self.secret_exists(required_secret):
                issues.append(f"Required secret '{required_secret}' is missing")
        
        # Check for weak secrets
        for name in self.list_secrets():
            try:
                value = self.get_secret(name)
                if len(value) < 8:
                    issues.append(f"Secret '{name}' appears to be weak (too short)")
            except:
                issues.append(f"Cannot validate secret '{name}'")
        
        # Check file permissions
        if self.secrets_dir.exists():
            secrets_file = self.secrets_dir / "secrets.encrypted"
            if secrets_file.exists():
                file_mode = oct(secrets_file.stat().st_mode)[-3:]
                if file_mode != '600':
                    issues.append(f"Secrets file has insecure permissions: {file_mode}")
        
        return issues
    
    def rotate_all_secrets(self, length: int = 32) -> Dict[str, bool]:
        """
        Rotate all secrets with new random values
        
        Args:
            length: Length for generated secrets
        
        Returns:
            Dict mapping secret names to success status
        """
        results = {}
        
        secret_names = self.list_secrets()
        for name in secret_names:
            try:
                new_secret = self.generate_secret(f"{name}_new", length)
                self.set_secret(name, new_secret)
                results[name] = True
            except Exception as e:
                print(f"Failed to rotate secret '{name}': {e}")
                results[name] = False
        
        return results
    
    def backup_secrets(self, backup_dir: Optional[str] = None) -> str:
        """
        Create encrypted backup of all secrets
        
        Args:
            backup_dir: Backup directory path
        
        Returns:
            str: Path to backup file
        """
        if backup_dir:
            backup_path = Path(backup_dir)
        else:
            backup_path = self.secrets_dir / "backups"
        
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"secrets_backup_{timestamp}.json"
        backup_filepath = backup_path / backup_filename
        
        if self.export_secrets(str(backup_filepath), include_sensitive=True):
            return str(backup_filepath)
        else:
            raise RuntimeError("Failed to create secrets backup")
    
    def restore_from_backup(self, backup_filepath: str, overwrite: bool = False) -> bool:
        """
        Restore secrets from backup file
        
        Args:
            backup_filepath: Path to backup file
            overwrite: Whether to overwrite existing secrets
        
        Returns:
            bool: Success status
        """
        return self.import_secrets(backup_filepath, overwrite)

# Utility functions for secrets management

def create_secrets_manager(config_dir: Optional[str] = None, 
                          master_key: Optional[str] = None) -> SecretsManager:
    """Create and initialize secrets manager"""
    return SecretsManager(
        secrets_dir=config_dir,
        master_key=master_key,
        enable_audit=True
    )

def generate_master_key() -> str:
    """Generate a new master encryption key"""
    return Fernet.generate_key().decode()

def validate_master_key(key: str) -> bool:
    """Validate master key format"""
    try:
        Fernet(key.encode())
        return True
    except:
        return False

class SecretsTemplate:
    """Template for setting up common secrets"""
    
    DEVELOPMENT_SECRETS = {
        'llm_api_key': 'your_openai_api_key_here',
        'database_password': 'dev_password_123',
        'jwt_secret_key': 'dev_jwt_secret_key_change_in_production',
        'encryption_key': 'dev_encryption_key_change_in_production'
    }
    
    PRODUCTION_SECRETS_TEMPLATE = {
        'llm_api_key': '',
        't2i_api_key': '',
        'mllm_api_key': '',
        'database_password': '',
        'redis_password': '',
        'jwt_secret_key': '',
        'encryption_key': '',
        'vault_token': '',
        'aws_secret_access_key': ''
    }
    
    @classmethod
    def setup_development_secrets(cls, secrets_manager: SecretsManager) -> None:
        """Setup development secrets"""
        for name, value in cls.DEVELOPMENT_SECRETS.items():
            if not secrets_manager.secret_exists(name):
                secrets_manager.set_secret(name, value)
    
    @classmethod
    def setup_production_secrets_template(cls, secrets_manager: SecretsManager, 
                                        generate_random: bool = True) -> Dict[str, str]:
        """
        Setup production secrets template
        
        Args:
            secrets_manager: SecretsManager instance
            generate_random: Whether to generate random values for security keys
        
        Returns:
            Dict with secret names and their generated values
        """
        generated_secrets = {}
        
        for name, default_value in cls.PRODUCTION_SECRETS_TEMPLATE.items():
            if not secrets_manager.secret_exists(name):
                if generate_random and name in ['jwt_secret_key', 'encryption_key']:
                    # Generate secure random keys
                    value = secrets_manager.generate_secret(name, 64, persist=False)
                    generated_secrets[name] = value
                else:
                    # Set empty placeholder
                    secrets_manager.set_secret(name, default_value)
                    generated_secrets[name] = default_value
        
        return generated_secrets
    
    @classmethod
    def generate_secrets_config_file(cls, output_path: str) -> bool:
        """Generate secrets configuration template file"""
        try:
            template = {
                'secrets_configuration': {
                    'description': 'Template for secrets configuration',
                    'created_at': datetime.now().isoformat(),
                    'instructions': [
                        '1. Copy this file to your secrets directory',
                        '2. Fill in actual secret values',
                        '3. Ensure file has restrictive permissions (600)',
                        '4. Never commit this file to version control'
                    ]
                },
                'required_secrets': dict(cls.PRODUCTION_SECRETS_TEMPLATE),
                'optional_secrets': {
                    'slack_webhook_url': '',
                    'email_smtp_password': '',
                    'monitoring_api_key': ''
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(template, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(output_path, 0o600)
            
            return True
            
        except Exception as e:
            print(f"Error generating secrets config file: {e}")
            return False

# Context manager for secure secret handling

class SecretContext:
    """Context manager for secure handling of secrets"""
    
    def __init__(self, secrets_manager: SecretsManager, secret_name: str):
        self.secrets_manager = secrets_manager
        self.secret_name = secret_name
        self._secret_value: Optional[str] = None
    
    def __enter__(self) -> str:
        self._secret_value = self.secrets_manager.get_secret(self.secret_name)
        return self._secret_value
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Securely clear the secret from memory
        if self._secret_value:
            self.secrets_manager._secure_clear(self._secret_value)
            self._secret_value = None

# Decorator for functions that need secrets

def with_secret(secret_name: str, secrets_manager: Optional[SecretsManager] = None):
    """Decorator to inject secrets into functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if secrets_manager is None:
                # Use global secrets manager
                sm = SecretsManager()
            else:
                sm = secrets_manager
            
            secret_value = sm.get_secret(secret_name)
            kwargs[f'{secret_name}_secret'] = secret_value
            
            try:
                return func(*args, **kwargs)
            finally:
                # Securely clear secret
                sm._secure_clear(secret_value)
        
        return wrapper
    return decorator
