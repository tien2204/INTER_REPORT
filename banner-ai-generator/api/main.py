"""
FastAPI Main Application

Main FastAPI application for the Banner AI Generator system.
Provides REST API endpoints for campaign management, asset handling,
and agent coordination.
"""

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import asyncio
from contextlib import asynccontextmanager
import time
from typing import Dict, Any
from structlog import get_logger

from .routers import campaigns, agents, assets, designs
from .middleware.auth import AuthMiddleware
from .middleware.rate_limiting import RateLimitMiddleware
from .middleware.logging import LoggingMiddleware
from config.system_config import get_system_config
from main import BannerGeneratorApp
from api.dependencies import set_banner_app
from dotenv import load_dotenv

# Load .env file
load_dotenv()

logger = get_logger(__name__)

# Global app instance
banner_app: BannerGeneratorApp = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Banner Generator API...")
    try:
        banner_app = BannerGeneratorApp()
        await banner_app.initialize()
        set_banner_app(banner_app)   # <--- dùng hàm này thay vì global biến trong main.py
        logger.info("Banner Generator system initialized")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise
    finally:
        if banner_app:
            logger.info("Shutting down Banner Generator system...")
            await banner_app.shutdown()
            logger.info("System shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Banner AI Generator API",
    description="Multi-AI Agent system for automated banner generation",
    version="1.0.0",
    docs_url=None,  # Custom docs endpoint
    redoc_url=None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_error"
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "type": "server_error"
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )


# Custom OpenAPI schema
def custom_openapi():
    """Custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Banner AI Generator API",
        version="1.0.0",
        description="""
        ## Multi-AI Agent Banner Generation System
        
        This API provides endpoints for creating professional banner advertisements
        using a multi-agent AI system.
        
        ### Key Features:
        - **Campaign Management**: Create and manage banner campaigns
        - **Asset Upload**: Upload logos, images, and other brand assets
        - **AI Generation**: Automated background and foreground design
        - **Real-time Progress**: Track generation progress in real-time
        - **Quality Control**: Automated design review and optimization
        
        ### Workflow:
        1. Create a campaign with brief and requirements
        2. Upload brand assets (logo, images, etc.)
        3. AI agents generate backgrounds and layouts
        4. Review and iterate on designs
        5. Export final banners in multiple formats
        """,
        routes=app.routes,
    )
    
    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "campaigns",
            "description": "Campaign management operations"
        },
        {
            "name": "assets", 
            "description": "Asset upload and management"
        },
        {
            "name": "designs",
            "description": "Design generation and management"
        },
        {
            "name": "agents",
            "description": "AI agent status and control"
        },
        {
            "name": "system",
            "description": "System health and monitoring"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Custom documentation
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


# Health check endpoints
@app.get("/health", tags=["system"])
async def health_check():
    """System health check"""
    try:
        system_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "components": {}
        }
        
        if banner_app:
            # Check AI models
            if banner_app.llm_interface:
                system_status["components"]["llm"] = "available"
            if banner_app.t2i_interface:
                system_status["components"]["t2i"] = "available"
            if banner_app.mllm_interface:
                system_status["components"]["mllm"] = "available"
            
            # Check agents
            if banner_app.strategist_agent:
                system_status["components"]["strategist_agent"] = "running"
            if banner_app.background_designer_agent:
                system_status["components"]["background_designer_agent"] = "running"
            
            # Check memory and communication
            if banner_app.shared_memory:
                system_status["components"]["shared_memory"] = "connected"
            if banner_app.message_queue:
                system_status["components"]["message_queue"] = "connected"
        
        return system_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@app.get("/", tags=["system"])
async def root():
    """API root endpoint"""
    return {
        "message": "Banner AI Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Include routers
app.include_router(campaigns.router, prefix="/api/v1/campaigns", tags=["campaigns"])
app.include_router(assets.router, prefix="/api/v1/assets", tags=["assets"])
app.include_router(designs.router, prefix="/api/v1/designs", tags=["designs"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])


# Utility function to get the banner app instance
def get_banner_app() -> BannerGeneratorApp:
    """Get the global banner app instance"""
    global banner_app
    if not banner_app:
        raise HTTPException(status_code=503, detail="System not initialized")
    return banner_app


# Export for use in routers
__all__ = ["app", "get_banner_app"]
