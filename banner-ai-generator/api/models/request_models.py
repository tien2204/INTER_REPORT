"""
Request Models

Pydantic models for API request validation and documentation.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class IndustryEnum(str, Enum):
    """Supported industries"""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    EDUCATION = "education"
    RETAIL = "retail"
    FOOD = "food"
    TRAVEL = "travel"
    AUTOMOTIVE = "automotive"
    FASHION = "fashion"
    REAL_ESTATE = "real_estate"
    OTHER = "other"


class MoodEnum(str, Enum):
    """Supported moods"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    ENERGETIC = "energetic"
    ELEGANT = "elegant"
    PLAYFUL = "playful"
    TRUSTWORTHY = "trustworthy"
    INNOVATIVE = "innovative"
    CALM = "calm"


class ToneEnum(str, Enum):
    """Supported tones"""
    CONFIDENT = "confident"
    APPROACHABLE = "approachable"
    AUTHORITATIVE = "authoritative"
    CASUAL = "casual"
    FORMAL = "formal"
    INSPIRATIONAL = "inspirational"


class BannerDimensions(BaseModel):
    """Banner dimensions specification"""
    width: int = Field(..., ge=100, le=4096, description="Width in pixels")
    height: int = Field(..., ge=50, le=4096, description="Height in pixels")
    
    @validator('height')
    def validate_aspect_ratio(cls, v, values):
        """Validate reasonable aspect ratio"""
        if 'width' in values:
            ratio = values['width'] / v
            if ratio > 10 or ratio < 0.1:
                raise ValueError("Aspect ratio must be between 0.1 and 10")
        return v


class CampaignBriefRequest(BaseModel):
    """Campaign brief request model"""
    company_name: str = Field(..., min_length=1, max_length=100, description="Company name")
    product_name: Optional[str] = Field(None, max_length=100, description="Product or service name")
    primary_message: str = Field(..., min_length=5, max_length=200, description="Main marketing message")
    cta_text: str = Field(..., min_length=2, max_length=50, description="Call-to-action text")
    
    target_audience: str = Field(..., min_length=5, max_length=300, description="Target audience description")
    industry: IndustryEnum = Field(..., description="Business industry")
    mood: MoodEnum = Field(..., description="Desired mood/emotion")
    tone: ToneEnum = Field(..., description="Communication tone")
    
    dimensions: BannerDimensions = Field(..., description="Banner dimensions")
    
    key_messages: Optional[List[str]] = Field(
        None, 
        max_items=5, 
        description="Additional key messages"
    )
    brand_colors: Optional[List[str]] = Field(
        None,
        max_items=10,
        description="Brand colors (hex codes)"
    )
    brand_guidelines: Optional[str] = Field(
        None,
        max_length=500,
        description="Additional brand guidelines"
    )
    
    @validator('brand_colors')
    def validate_hex_colors(cls, v):
        """Validate hex color format"""
        if v:
            for color in v:
                if not color.startswith('#') or len(color) != 7:
                    raise ValueError(f"Invalid hex color format: {color}")
        return v


class AssetUploadRequest(BaseModel):
    """Asset upload request model"""
    asset_type: str = Field(..., description="Type of asset (logo, image, reference)")
    filename: str = Field(..., description="Original filename")
    file_data: str = Field(..., description="Base64 encoded file data")
    description: Optional[str] = Field(None, max_length=200, description="Asset description")
    tags: Optional[List[str]] = Field(None, max_items=10, description="Asset tags")


class DesignGenerationRequest(BaseModel):
    """Design generation request model"""
    campaign_id: str = Field(..., description="Campaign ID")
    generation_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom generation options"
    )
    priority: Optional[str] = Field("normal", description="Generation priority")
    
    class Config:
        schema_extra = {
            "example": {
                "campaign_id": "campaign_123",
                "generation_options": {
                    "style_emphasis": "modern",
                    "color_preference": "vibrant",
                    "layout_type": "centered"
                },
                "priority": "high"
            }
        }


class DesignIterationRequest(BaseModel):
    """Design iteration request model"""
    design_id: str = Field(..., description="Design ID to iterate on")
    feedback: Optional[str] = Field(None, max_length=500, description="Feedback for iteration")
    changes_requested: Optional[List[str]] = Field(
        None,
        max_items=10,
        description="Specific changes requested"
    )
    iteration_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Iteration-specific options"
    )


class AssetValidationRequest(BaseModel):
    """Asset validation request model"""
    file_data: str = Field(..., description="Base64 encoded file data")
    filename: str = Field(..., description="Filename")
    expected_type: Optional[str] = Field(None, description="Expected file type")
    validation_rules: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom validation rules"
    )


class LogoProcessingRequest(BaseModel):
    """Logo processing request model"""
    image_data: str = Field(..., description="Base64 encoded logo data")
    filename: str = Field(..., description="Logo filename")
    target_sizes: Optional[List[BannerDimensions]] = Field(
        None,
        description="Target sizes to generate"
    )
    background_color: Optional[str] = Field(
        None,
        description="Background color for contrast analysis"
    )
    processing_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Logo processing options"
    )


class BackgroundRemovalRequest(BaseModel):
    """Background removal request model"""
    image_data: str = Field(..., description="Base64 encoded image data")
    method: Optional[str] = Field("auto", description="Removal method")
    tolerance: Optional[int] = Field(30, ge=0, le=100, description="Color tolerance")


class ImageOptimizationRequest(BaseModel):
    """Image optimization request model"""
    image_data: str = Field(..., description="Base64 encoded image data")
    target_size_kb: Optional[int] = Field(200, ge=10, le=5000, description="Target size in KB")
    maintain_quality: Optional[bool] = Field(True, description="Maintain visual quality")
    output_format: Optional[str] = Field("auto", description="Output format")


class AgentControlRequest(BaseModel):
    """Agent control request model"""
    action: str = Field(..., description="Action to perform (start, stop, restart, status)")
    agent_id: Optional[str] = Field(None, description="Specific agent ID")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Action parameters")


class WorkflowControlRequest(BaseModel):
    """Workflow control request model"""
    workflow_action: str = Field(..., description="Workflow action")
    session_id: str = Field(..., description="Session ID")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Workflow parameters")


class SystemConfigRequest(BaseModel):
    """System configuration request model"""
    config_updates: Dict[str, Any] = Field(..., description="Configuration updates")
    apply_immediately: Optional[bool] = Field(False, description="Apply changes immediately")


class BulkAssetUploadRequest(BaseModel):
    """Bulk asset upload request model"""
    assets: List[AssetUploadRequest] = Field(..., max_items=50, description="List of assets to upload")
    campaign_id: Optional[str] = Field(None, description="Associate with campaign")
    validation_strict: Optional[bool] = Field(True, description="Strict validation mode")


class DesignExportRequest(BaseModel):
    """Design export request model"""
    design_id: str = Field(..., description="Design ID to export")
    export_formats: List[str] = Field(..., description="Export formats (svg, png, jpeg, figma)")
    export_options: Optional[Dict[str, Any]] = Field(None, description="Export options")
    
    @validator('export_formats')
    def validate_formats(cls, v):
        """Validate export formats"""
        valid_formats = {'svg', 'png', 'jpeg', 'webp', 'figma', 'json'}
        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid export format: {fmt}")
        return v


class CampaignUpdateRequest(BaseModel):
    """Campaign update request model"""
    campaign_id: str = Field(..., description="Campaign ID")
    updates: Dict[str, Any] = Field(..., description="Fields to update")
    update_reason: Optional[str] = Field(None, description="Reason for update")


class DesignFeedbackRequest(BaseModel):
    """Design feedback request model"""
    design_id: str = Field(..., description="Design ID")
    feedback_type: str = Field(..., description="Type of feedback")
    rating: Optional[int] = Field(None, ge=1, le=10, description="Quality rating")
    comments: Optional[str] = Field(None, max_length=1000, description="Detailed feedback")
    specific_issues: Optional[List[str]] = Field(None, description="Specific issues identified")


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, max_length=200, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    limit: Optional[int] = Field(20, ge=1, le=100, description="Results limit")
    offset: Optional[int] = Field(0, ge=0, description="Results offset")
    sort_by: Optional[str] = Field("created_at", description="Sort field")
    sort_order: Optional[str] = Field("desc", description="Sort order (asc, desc)")
