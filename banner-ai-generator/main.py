"""
Main Application Entry Point

Starts the Multi AI Agent Banner Generator system.
"""

import asyncio
import signal
import sys
from typing import Optional
import uvicorn
from structlog import get_logger

from config.system_config import init_system_config, get_system_config
from memory_manager.shared_memory import SharedMemory
from memory_manager.memory_store import MemoryStore  
from memory_manager.session_manager import SessionManager
from communication.message_queue import MessageQueue
from agents.strategist.strategist_agent import StrategistAgent
from agents.background_designer.background_agent import BackgroundDesignerAgent
from ai_models.llm_interface import LLMInterface
from ai_models.t2i_interface import TextToImageInterface
from ai_models.mllm_interface import MultimodalLLMInterface
from file_processing.image_processor import ImageProcessor
from file_processing.logo_processor import LogoProcessor
from file_processing.file_validator import FileValidator
from dotenv import load_dotenv

# Load .env file
load_dotenv()

logger = get_logger(__name__)


class BannerGeneratorApp:
    """
    Main application class for the Banner Generator system
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = init_system_config(config_file)
        self.shared_memory: Optional[SharedMemory] = None
        self.memory_store: Optional[MemoryStore] = None
        self.session_manager: Optional[SessionManager] = None
        self.message_queue: Optional[MessageQueue] = None
        self.strategist_agent: Optional[StrategistAgent] = None
        self.background_designer_agent: Optional[BackgroundDesignerAgent] = None
        
        # AI Models
        self.llm_interface: Optional[LLMInterface] = None
        self.t2i_interface: Optional[TextToImageInterface] = None
        self.mllm_interface: Optional[MultimodalLLMInterface] = None
        
        # File Processing
        self.image_processor: Optional[ImageProcessor] = None
        self.logo_processor: Optional[LogoProcessor] = None
        self.file_validator: Optional[FileValidator] = None
        
        self._running = False
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing Banner Generator system...")
            
            # Validate configuration
            if not self.config.validate_configuration():
                raise RuntimeError("Configuration validation failed")
            
            # Initialize memory components
            self.shared_memory = SharedMemory(self.config.get_redis_url())
            await self.shared_memory.initialize()
            
            self.memory_store = MemoryStore(self.config.get_database_url())
            
            self.session_manager = SessionManager(self.shared_memory)
            
            # Initialize communication
            self.message_queue = MessageQueue({"redis_url": self.config.get_redis_url()})
            await self.message_queue.start()
            
            # Initialize AI models
            await self._initialize_ai_models()
            
            # Initialize file processing
            await self._initialize_file_processing()
            
            # Initialize agents
            await self._initialize_agents()
            
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    async def _initialize_agents(self):
        """Initialize AI agents"""
        try:
            # Initialize Strategist Agent
            strategist_config = {
                "brief_analyzer": {
                    "industry_keywords_enabled": True,
                    "mood_detection_threshold": 0.7
                },
                "logo_processor": {
                    "max_size": (300, 300),
                    "color_palette_size": 5
                },
                "brand_analyzer": {
                    "archetype_analysis_enabled": True
                }
            }
            
            self.strategist_agent = StrategistAgent(
                shared_memory=self.shared_memory,
                message_queue=self.message_queue,
                session_manager=self.session_manager,
                config=strategist_config
            )
            
            await self.strategist_agent.start()
            logger.info("Strategist Agent initialized and started")
            
            # Initialize Background Designer Agent
            background_config = {
                "prompt_generator": {
                    "style_emphasis": True,
                    "quality_keywords": True
                },
                "t2i_interface": {
                    "default_model": "flux.1-schnell",
                    "use_local": False
                },
                "refinement_loop": {
                    "max_iterations": 3,
                    "quality_threshold": 0.8
                }
            }
            
            self.background_designer_agent = BackgroundDesignerAgent(
                shared_memory=self.shared_memory,
                message_queue=self.message_queue,
                session_manager=self.session_manager,
                config=background_config
            )
            
            await self.background_designer_agent.start()
            logger.info("Background Designer Agent initialized and started")
            
            # TODO: Initialize remaining agents
            # - Foreground Designer Agent  
            # - Developer Agent
            # - Design Reviewer Agent
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise
    
    async def _initialize_ai_models(self):
        """Initialize AI model interfaces"""
        try:
            # Initialize LLM Interface
            llm_config = {
                "default_provider": "openai",
                "openai": {
                    "enabled": True,
                    "api_key": self.config.get("openai_api_key", ""),
                    "default_model": "gpt-4"
                },
                "anthropic": {
                    "enabled": False,
                    "api_key": self.config.get("anthropic_api_key", "")
                }
            }
            
            self.llm_interface = LLMInterface(llm_config)
            logger.info("LLM Interface initialized")
            
            # Initialize T2I Interface  
            t2i_config = {
                "default_model": "flux.1-schnell",
                "flux_config": {
                    "enabled": True,
                    "use_local": False,
                    "api_url": self.config.get("flux_api_url", ""),
                    "api_key": self.config.get("flux_api_key", "")
                },
                "openai_config": {
                    "enabled": True,
                    "api_key": self.config.get("openai_api_key", "")
                }
            }
            
            self.t2i_interface = TextToImageInterface(t2i_config)
            logger.info("Text-to-Image Interface initialized")
            
            # Initialize MLLM Interface
            mllm_config = {
                "default_provider": "openai_vision",
                "openai_vision": {
                    "enabled": True,
                    "api_key": self.config.get("openai_api_key", ""),
                    "default_model": "gpt-4-vision-preview"
                }
            }
            
            self.mllm_interface = MultimodalLLMInterface(mllm_config)
            logger.info("Multimodal LLM Interface initialized")
            
        except Exception as e:
            logger.error(f"AI models initialization failed: {e}")
            raise
    
    async def _initialize_file_processing(self):
        """Initialize file processing components"""
        try:
            # Initialize Image Processor
            image_config = {
                "jpeg_quality": 90,
                "png_compression": 6,
                "max_file_size_mb": 10,
                "max_dimensions": (4096, 4096),
                "auto_orient": True,
                "strip_metadata": True
            }
            
            self.image_processor = ImageProcessor(image_config)
            logger.info("Image Processor initialized")
            
            # Initialize Logo Processor
            logo_config = {
                "max_logo_size": (500, 500),
                "auto_remove_padding": True,
                "preserve_transparency": True,
                "min_contrast_ratio": 3.0
            }
            
            self.logo_processor = LogoProcessor(logo_config)
            logger.info("Logo Processor initialized")
            
            # Initialize File Validator
            validator_config = {
                "max_file_sizes": {
                    "image": 10 * 1024 * 1024,
                    "svg": 1 * 1024 * 1024,
                    "json": 100 * 1024
                },
                "scan_for_malware": True,
                "validate_image_headers": True,
                "min_image_dimensions": (32, 32),
                "max_image_dimensions": (8192, 8192)
            }
            
            self.file_validator = FileValidator(validator_config)
            logger.info("File Validator initialized")
            
        except Exception as e:
            logger.error(f"File processing initialization failed: {e}")
            raise
    
    async def start(self):
        """Start the application"""
        try:
            await self.initialize()
            self._running = True
            
            logger.info("Banner Generator system is running")
            
            # Keep the application running
            while self._running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the application gracefully"""
        try:
            logger.info("Shutting down Banner Generator system...")
            self._running = False
            
            # Stop agents
            if self.background_designer_agent:
                await self.background_designer_agent.stop()
            
            if self.strategist_agent:
                await self.strategist_agent.stop()
            
            # Close connections
            if self.message_queue:
                await self.message_queue.stop()
            
            if self.shared_memory:
                await self.shared_memory.close()
            
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def run_api_server(self):
        """Run the FastAPI server"""
        config = get_system_config()
        
        uvicorn.run(
            "api.main:app",
            host=config.app["host"],
            port=config.app["port"],
            reload=config.app["debug"],
            workers=config.app["max_workers"] if not config.app["debug"] else 1
        )


async def run_agents():
    """Run agents as background services"""
    app = BannerGeneratorApp()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        asyncio.create_task(app.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    await app.start()


def run_demo():
    """Run a simple demo of the system"""
    async def demo():
        try:
            app = BannerGeneratorApp()
            await app.initialize()
            
            # Demo campaign creation
            demo_brief = {
                "company_name": "TechCorp",
                "product_name": "CloudSync Pro", 
                "primary_message": "Sync your data seamlessly across all devices",
                "cta_text": "Start Free Trial",
                "target_audience": "Tech professionals and small businesses",
                "industry": "technology",
                "mood": "professional",
                "tone": "confident",
                "dimensions": {"width": 728, "height": 90},
                "key_messages": [
                    "Real-time synchronization",
                    "Enterprise-grade security",
                    "Works on all platforms"
                ]
            }
            
            demo_assets = {
                "logo": {
                    "base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                }
            }
            
            logger.info("Creating demo campaign...")
            campaign_id = await app.strategist_agent.create_campaign(demo_brief, demo_assets)
            
            logger.info(f"Demo campaign created successfully! Campaign ID: {campaign_id}")
            
            # Get campaign status
            status = await app.strategist_agent.get_campaign_status(campaign_id)
            logger.info(f"Campaign status: {status}")
            
            await app.shutdown()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    asyncio.run(demo())


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "api":
            # Run API server
            app = BannerGeneratorApp()
            app.run_api_server()
            
        elif command == "agents":
            # Run agents
            asyncio.run(run_agents())
            
        elif command == "demo":
            # Run demo
            run_demo()
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: api, agents, demo")
            sys.exit(1)
    else:
        print("Banner Generator System")
        print("Usage: python main.py [command]")
        print("")
        print("Commands:")
        print("  api    - Start FastAPI server")
        print("  agents - Start agent workers")
        print("  demo   - Run system demo")
        print("")
        print("Examples:")
        print("  python main.py api")
        print("  python main.py agents") 
        print("  python main.py demo")


if __name__ == "__main__":
    main()
