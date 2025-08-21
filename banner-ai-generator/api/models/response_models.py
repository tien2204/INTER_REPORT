"""
Response Models

Pydantic models for API response validation and documentation.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class StatusEnum(str, Enum):
    """General status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CampaignStatusEnum(str, Enum):
    """Campaign status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    GENERATING = "generating"
    REVIEW = "review"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class DesignStatusEnum(str, Enum):
    """Design status enumeration"""
    QUEUED = "queued"
    GENERATING_BACKGROUND = "generating_background"
    GENERATING_FOREGROUND = "generating_foreground"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    ITERATING = "iterating"


class AgentStatusEnum(str, Enum):
    """Agent status enumeration"""
    RUNNING = "running"
    STOPPED = "stopped"
    BUSY = "busy"
    ERROR = "error"
    STARTING = "starting"
    STOPPING = "stopping"


class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = Field(..., description="Operation success status")
    message: Optional[str] = Field(None, description="Response message")
    request_id: Optional[str] = Field(None, description="Request tracking ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = Field(False, description="Always false for errors")
    error: Dict[str, Any] = Field(..., description="Error details")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "message": "Validation failed",
                "error": {
                    "code": 400,
                    "type": "validation_error",
                    "details": ["Field 'company_name' is required"]
                },
                "request_id": "req_123456",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class CampaignResponse(BaseResponse):
    """Campaign response model"""
    campaign: Dict[str, Any] = Field(..., description="Campaign data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Campaign created successfully",
                "campaign": {
                    "id": "campaign_123",
                    "company_name": "TechCorp",
                    "status": "draft",
                    "created_at": "2024-01-15T10:30:00Z"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class CampaignListResponse(BaseResponse):
    """Campaign list response model"""
    campaigns: List[Dict[str, Any]] = Field(..., description="List of campaigns")
    total: int = Field(..., description="Total number of campaigns")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")


class AssetResponse(BaseResponse):
    """Asset response model"""
    asset: Dict[str, Any] = Field(..., description="Asset data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Asset uploaded successfully",
                "asset": {
                    "id": "asset_123",
                    "filename": "logo.png",
                    "type": "logo",
                    "size": 15360,
                    "url": "/assets/asset_123"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class AssetListResponse(BaseResponse):
    """Asset list response model"""
    assets: List[Dict[str, Any]] = Field(..., description="List of assets")
    total: int = Field(..., description="Total number of assets")


class DesignResponse(BaseResponse):
    """Design response model"""
    design: Dict[str, Any] = Field(..., description="Design data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Design generated successfully",
                "design": {
                    "id": "design_123",
                    "campaign_id": "campaign_123",
                    "status": "completed",
                    "background_url": "/designs/design_123/background.png",
                    "preview_url": "/designs/design_123/preview.png"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class DesignListResponse(BaseResponse):
    """Design list response model"""
    designs: List[Dict[str, Any]] = Field(..., description="List of designs")
    total: int = Field(..., description="Total number of designs")


class DesignProgressResponse(BaseResponse):
    """Design progress response model"""
    progress: Dict[str, Any] = Field(..., description="Design generation progress")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "progress": {
                    "design_id": "design_123",
                    "status": "generating_background",
                    "percentage": 45,
                    "current_step": "Background generation",
                    "estimated_completion": "2024-01-15T10:35:00Z",
                    "steps_completed": ["Strategy analysis", "Asset processing"],
                    "current_agent": "background_designer"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class AgentStatusResponse(BaseResponse):
    """Agent status response model"""
    agents: Dict[str, Dict[str, Any]] = Field(..., description="Agent status information")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "agents": {
                    "strategist": {
                        "status": "running",
                        "current_sessions": 2,
                        "total_processed": 150,
                        "uptime": "2 days, 3 hours"
                    },
                    "background_designer": {
                        "status": "busy",
                        "current_sessions": 1,
                        "queue_length": 3,
                        "avg_generation_time": "45 seconds"
                    }
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ValidationResponse(BaseResponse):
    """Validation response model"""
    validation_result: Dict[str, Any] = Field(..., description="Validation results")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "File validation completed",
                "validation_result": {
                    "valid": True,
                    "file_info": {
                        "size_kb": 125.4,
                        "format": "PNG",
                        "dimensions": {"width": 500, "height": 200}
                    },
                    "issues": [],
                    "warnings": ["Consider using SVG for better scalability"]
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ProcessingResponse(BaseResponse):
    """Processing response model"""
    processing_result: Dict[str, Any] = Field(..., description="Processing results")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Image processing completed",
                "processing_result": {
                    "original_size": 245760,
                    "processed_size": 158720,
                    "compression_ratio": 35.4,
                    "processed_image": "data:image/png;base64,iVBORw0KGgo...",
                    "metadata": {
                        "width": 400,
                        "height": 300,
                        "format": "PNG"
                    }
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ExportResponse(BaseResponse):
    """Export response model"""
    exports: Dict[str, str] = Field(..., description="Export URLs by format")
    download_links: Dict[str, str] = Field(..., description="Download links")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Design exported successfully",
                "exports": {
                    "svg": "/exports/design_123.svg",
                    "png": "/exports/design_123.png",
                    "figma": "/exports/design_123_figma.json"
                },
                "download_links": {
                    "svg": "/api/v1/designs/design_123/download?format=svg",
                    "png": "/api/v1/designs/design_123/download?format=png"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class AnalyticsResponse(BaseResponse):
    """Analytics response model"""
    analytics: Dict[str, Any] = Field(..., description="Analytics data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "analytics": {
                    "total_campaigns": 156,
                    "total_designs": 342,
                    "success_rate": 94.2,
                    "avg_generation_time": "2.3 minutes",
                    "most_used_industries": ["technology", "healthcare", "finance"],
                    "agent_performance": {
                        "strategist": {"uptime": 99.8, "avg_response_time": "1.2s"},
                        "background_designer": {"uptime": 97.5, "avg_generation_time": "45s"}
                    }
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class SystemHealthResponse(BaseResponse):
    """System health response model"""
    health: Dict[str, Any] = Field(..., description="System health status")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "health": {
                    "status": "healthy",
                    "version": "1.0.0",
                    "uptime": "5 days, 12 hours",
                    "components": {
                        "database": "healthy",
                        "redis": "healthy",
                        "ai_models": "healthy",
                        "agents": "healthy"
                    },
                    "metrics": {
                        "cpu_usage": 45.2,
                        "memory_usage": 67.8,
                        "disk_usage": 23.1
                    }
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class WorkflowResponse(BaseResponse):
    """Workflow response model"""
    workflow: Dict[str, Any] = Field(..., description="Workflow information")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "workflow": {
                    "session_id": "session_123",
                    "status": "in_progress",
                    "current_step": "background_generation",
                    "completed_steps": ["strategy_analysis", "asset_processing"],
                    "remaining_steps": ["foreground_design", "review", "export"],
                    "estimated_completion": "2024-01-15T10:45:00Z"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class SearchResponse(BaseResponse):
    """Search response model"""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total: int = Field(..., description="Total matching results")
    query: str = Field(..., description="Original search query")
    took_ms: int = Field(..., description="Search execution time in milliseconds")


class BulkOperationResponse(BaseResponse):
    """Bulk operation response model"""
    results: List[Dict[str, Any]] = Field(..., description="Individual operation results")
    summary: Dict[str, Any] = Field(..., description="Operation summary")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Bulk operation completed",
                "results": [
                    {"id": "item_1", "success": True},
                    {"id": "item_2", "success": False, "error": "Validation failed"}
                ],
                "summary": {
                    "total": 2,
                    "successful": 1,
                    "failed": 1,
                    "success_rate": 50.0
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class StreamingResponse(BaseModel):
    """Streaming response model for real-time updates"""
    event: str = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "event": "design_progress",
                "data": {
                    "design_id": "design_123",
                    "progress": 65,
                    "current_step": "Foreground generation"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ConfigResponse(BaseResponse):
    """Configuration response model"""
    config: Dict[str, Any] = Field(..., description="Configuration data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "config": {
                    "max_file_size_mb": 10,
                    "supported_formats": ["png", "jpeg", "svg"],
                    "ai_models": {
                        "llm_provider": "openai",
                        "t2i_provider": "flux"
                    }
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
