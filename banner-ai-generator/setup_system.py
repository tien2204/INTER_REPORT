#!/usr/bin/env python3
"""
System Setup and Validation Script

Validates the entire Multi AI Agent Banner Generator system
for consistency, dependencies, and configuration.
"""

import os
import sys
import asyncio
import importlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import traceback


class SystemValidator:
    """System validation and setup utility"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.success_count = 0
        self.total_checks = 0
    
    def validate_dependencies(self) -> bool:
        """Validate all Python dependencies"""
        print("🔍 Checking Python dependencies...")
        
        required_packages = [
            'fastapi', 'uvicorn', 'pydantic', 'sqlalchemy', 'alembic',
            'redis', 'aioredis', 'websockets', 'structlog', 'openai',
            'anthropic', 'pillow', 'numpy', 'transformers', 'torch',
            'diffusers', 'pytest', 'black', 'mypy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                self.success_count += 1
            except ImportError:
                missing_packages.append(package)
                self.errors.append(f"Missing package: {package}")
            
            self.total_checks += 1
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("   Run: pip install -r requirements.txt")
            return False
        
        print("✅ All dependencies installed")
        return True
    
    def validate_file_structure(self) -> bool:
        """Validate project file structure"""
        print("🔍 Checking file structure...")
        
        required_files = [
            'main.py',
            'requirements.txt',
            'env.example',
            'README.md',
            'config/__init__.py',
            'config/system_config.py',
            'agents/__init__.py',
            'agents/strategist/strategist_agent.py',
            'agents/background_designer/background_agent.py',
            'agents/foreground_designer/foreground_agent.py',
            'agents/developer/developer_agent.py',
            'agents/design_reviewer/design_reviewer_agent.py',
            'communication/__init__.py',
            'communication/agent_coordinator.py',
            'communication/message_queue.py',
            'communication/communication_manager.py',
            'memory_manager/__init__.py',
            'memory_manager/shared_memory.py',
            'ai_models/__init__.py',
            'ai_models/llm_interface.py',
            'ai_models/t2i_interface.py',
            'file_processing/__init__.py',
            'file_processing/image_processor.py',
            'database/__init__.py',
            'database/models.py',
            'database/connection.py',
            'api/__init__.py',
            'api/main.py',
            'frontend/package.json',
            'frontend/src/main.ts'
        ]
        
        missing_files = []
        
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
                self.errors.append(f"Missing file: {file_path}")
            else:
                self.success_count += 1
            
            self.total_checks += 1
        
        if missing_files:
            print(f"❌ Missing files: {len(missing_files)}")
            for file in missing_files[:5]:  # Show first 5
                print(f"   - {file}")
            if len(missing_files) > 5:
                print(f"   ... and {len(missing_files) - 5} more")
            return False
        
        print("✅ All required files present")
        return True
    
    def validate_imports(self) -> bool:
        """Validate all module imports"""
        print("🔍 Checking module imports...")
        
        modules_to_test = [
            'config.system_config',
            'memory_manager.shared_memory',
            'communication.agent_coordinator',
            'communication.message_queue',
            'agents.strategist.strategist_agent',
            'agents.background_designer.background_agent',
            'ai_models.llm_interface',
            'database.models',
            'api.main'
        ]
        
        import_errors = []
        
        for module_name in modules_to_test:
            try:
                importlib.import_module(module_name)
                self.success_count += 1
                print(f"   ✅ {module_name}")
            except Exception as e:
                import_errors.append((module_name, str(e)))
                self.errors.append(f"Import error in {module_name}: {e}")
                print(f"   ❌ {module_name}: {e}")
            
            self.total_checks += 1
        
        if import_errors:
            print(f"❌ Import errors found: {len(import_errors)}")
            return False
        
        print("✅ All modules import successfully")
        return True
    
    def validate_configuration(self) -> bool:
        """Validate configuration files"""
        print("🔍 Checking configuration...")
        
        # Check env.example
        env_file = Path('env.example')
        if not env_file.exists():
            self.errors.append("env.example file missing")
            self.total_checks += 1
            return False
        
        # Read and validate env variables
        required_env_vars = [
            'DATABASE_URL',
            'REDIS_URL', 
            'OPENAI_API_KEY',
            'SECRET_KEY',
            'JWT_SECRET'
        ]
        
        env_content = env_file.read_text()
        missing_vars = []
        
        for var in required_env_vars:
            if var not in env_content:
                missing_vars.append(var)
                self.errors.append(f"Missing environment variable: {var}")
            else:
                self.success_count += 1
            
            self.total_checks += 1
        
        if missing_vars:
            print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            return False
        
        print("✅ Configuration validation passed")
        return True
    
    async def validate_async_components(self) -> bool:
        """Validate async components"""
        print("🔍 Checking async components...")
        
        try:
            # Test Redis connection
            import redis.asyncio as redis
            redis_client = redis.from_url("redis://172.26.33.210:6379")
            
            try:
                await redis_client.ping()
                print("   ✅ Redis connection successful")
                self.success_count += 1
            except Exception as e:
                print(f"   ⚠️  Redis connection failed: {e}")
                self.warnings.append("Redis server not running - required for message queue")
            finally:
                await redis_client.close()
            
            self.total_checks += 1
            
            # Test database connection (if configured)
            try:
                from config.system_config import get_system_config
                config = get_system_config()
                db_url = config.get_database_url()
                
                if db_url.startswith('sqlite'):
                    print("   ✅ SQLite database configured")
                    self.success_count += 1
                else:
                    print(f"   ℹ️  Database configured: {db_url.split('@')[1] if '@' in db_url else db_url}")
                    self.success_count += 1
                
            except Exception as e:
                print(f"   ❌ Database configuration error: {e}")
                self.errors.append(f"Database configuration error: {e}")
            
            self.total_checks += 1
            
        except Exception as e:
            print(f"❌ Async validation failed: {e}")
            self.errors.append(f"Async validation failed: {e}")
            return False
        
        print("✅ Async components validation completed")
        return True
    
    def validate_frontend(self) -> bool:
        """Validate frontend setup"""
        print("🔍 Checking frontend setup...")
        
        frontend_path = Path('frontend')
        if not frontend_path.exists():
            self.errors.append("Frontend directory missing")
            self.total_checks += 1
            return False
        
        # Check package.json
        package_json = frontend_path / 'package.json'
        if not package_json.exists():
            self.errors.append("Frontend package.json missing")
            print("❌ Frontend package.json missing")
            self.total_checks += 1
            return False
        
        # Check if node_modules exists (optional warning)
        node_modules = frontend_path / 'node_modules'
        if not node_modules.exists():
            self.warnings.append("Frontend dependencies not installed - run 'npm install' in frontend/")
            print("   ⚠️  Frontend dependencies not installed")
        else:
            print("   ✅ Frontend dependencies installed")
            self.success_count += 1
        
        self.total_checks += 1
        print("✅ Frontend structure validation completed")
        return True
    
    def create_directories(self) -> bool:
        """Create required directories"""
        print("🔍 Creating required directories...")
        
        required_dirs = [
            'uploads',
            'temp', 
            'output',
            'logs',
            'data'
        ]
        
        created_dirs = []
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_name)
                    self.success_count += 1
                except Exception as e:
                    self.errors.append(f"Failed to create directory {dir_name}: {e}")
            else:
                self.success_count += 1
            
            self.total_checks += 1
        
        if created_dirs:
            print(f"   ✅ Created directories: {', '.join(created_dirs)}")
        else:
            print("   ✅ All directories exist")
        
        return True
    
    def generate_report(self) -> None:
        """Generate validation report"""
        print("\n" + "="*60)
        print("📊 SYSTEM VALIDATION REPORT")
        print("="*60)
        
        success_rate = (self.success_count / self.total_checks * 100) if self.total_checks > 0 else 0
        
        print(f"✅ Successful checks: {self.success_count}/{self.total_checks} ({success_rate:.1f}%)")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
            print("\n🔧 To fix these issues:")
            print("   1. Run: pip install -r requirements.txt")
            print("   2. Copy env.example to .env and configure")
            print("   3. Start Redis server: redis-server")
            print("   4. Install frontend deps: cd frontend && npm install")
        else:
            print("\n🎉 All validations passed! System is ready to run.")
            print("\n🚀 To start the system:")
            print("   1. Backend: uvicorn main:app --reload")
            print("   2. Frontend: cd frontend && npm run dev")
        
        print("="*60)
    
    async def run_all_validations(self) -> bool:
        """Run all validation checks"""
        print("🔍 Starting Multi AI Agent Banner Generator System Validation")
        print("="*60)
        
        validations = [
            ("Dependencies", self.validate_dependencies),
            ("File Structure", self.validate_file_structure),
            ("Module Imports", self.validate_imports),
            ("Configuration", self.validate_configuration),
            ("Frontend Setup", self.validate_frontend),
            ("Directory Creation", self.create_directories),
        ]
        
        all_passed = True
        
        for name, validation_func in validations:
            try:
                if not validation_func():
                    all_passed = False
            except Exception as e:
                print(f"❌ {name} validation failed with exception: {e}")
                self.errors.append(f"{name} validation exception: {e}")
                all_passed = False
            
            print()  # Add spacing
        
        # Run async validations
        try:
            await self.validate_async_components()
        except Exception as e:
            print(f"❌ Async validation failed: {e}")
            all_passed = False
        
        self.generate_report()
        return all_passed


async def main():
    """Main validation function"""
    validator = SystemValidator()
    success = await validator.run_all_validations()
    
    if success:
        print("\n✅ System validation completed successfully!")
        return 0
    else:
        print("\n❌ System validation found issues that need attention.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


