"""
Campaign Models

Specialized Pydantic models for campaign-related data structures.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class CampaignStatus(str, Enum):
    """Campaign status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    GENERATING = "generating"
    REVIEW = "review"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    CANCELLED = "cancelled"


class AssetType(str, Enum):
    """Asset type enumeration"""
    LOGO = "logo"
    IMAGE = "image"
    REFERENCE = "reference"
    BACKGROUND = "background"
    ICON = "icon"
    FONT = "font"
    OTHER = "other"


class DesignStatus(str, Enum):
    """Design status enumeration"""
    QUEUED = "queued"
    STRATEGY_ANALYSIS = "strategy_analysis"
    ASSET_PROCESSING = "asset_processing"
    BACKGROUND_GENERATION = "background_generation"
    FOREGROUND_DESIGN = "foreground_design"
    DESIGN_REVIEW = "design_review"
    CODE_GENERATION = "code_generation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class CampaignBrief(BaseModel):
    """Campaign brief model"""
    company_name: str = Field(..., description="Company name")
    product_name: Optional[str] = Field(None, description="Product or service name")
    primary_message: str = Field(..., description="Main marketing message")
    cta_text: str = Field(..., description="Call-to-action text")
    
    target_audience: str = Field(..., description="Target audience description")
    industry: str = Field(..., description="Business industry")
    mood: str = Field(..., description="Desired mood/emotion")
    tone: str = Field(..., description="Communication tone")
    
    key_messages: Optional[List[str]] = Field(None, description="Additional key messages")
    brand_colors: Optional[List[str]] = Field(None, description="Brand colors")
    brand_guidelines: Optional[str] = Field(None, description="Brand guidelines")
    
    dimensions: Dict[str, int] = Field(..., description="Banner dimensions")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AssetMetadata(BaseModel):
    """Asset metadata model"""
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="MIME type")
    dimensions: Optional[Dict[str, int]] = Field(None, description="Image dimensions")
    color_palette: Optional[List[str]] = Field(None, description="Extracted colors")
    has_transparency: Optional[bool] = Field(None, description="Has transparency")
    quality_score: Optional[float] = Field(None, description="Quality assessment score")


class Asset(BaseModel):
    """Asset model"""
    id: str = Field(..., description="Asset ID")
    campaign_id: str = Field(..., description="Associated campaign ID")
    asset_type: AssetType = Field(..., description="Type of asset")
    
    # File information
    filename: str = Field(..., description="Original filename")
    storage_path: str = Field(..., description="Storage path")
    public_url: Optional[str] = Field(None, description="Public access URL")
    
    # Asset content
    file_data: Optional[str] = Field(None, description="Base64 encoded data (for small files)")
    
    # Metadata
    metadata: AssetMetadata = Field(..., description="Asset metadata")
    description: Optional[str] = Field(None, description="Asset description")
    tags: Optional[List[str]] = Field(None, description="Asset tags")
    
    # Processing status
    processed: bool = Field(False, description="Processing completed")
    processing_results: Optional[Dict[str, Any]] = Field(None, description="Processing results")
    
    # Timestamps
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = Field(None, description="Processing completion time")


class StrategicDirection(BaseModel):
    """Strategic direction model"""
    mood: str = Field(..., description="Analyzed mood")
    tone: str = Field(..., description="Analyzed tone")
    color_palette: List[str] = Field(..., description="Recommended color palette")
    visual_style: str = Field(..., description="Visual style direction")
    typography_style: str = Field(..., description="Typography recommendations")
    layout_preferences: Dict[str, Any] = Field(..., description="Layout preferences")
    target_audience_insights: Dict[str, Any] = Field(..., description="Audience insights")
    brand_archetype: Optional[str] = Field(None, description="Brand archetype")
    competitive_positioning: Optional[str] = Field(None, description="Competitive positioning")
    
    # Analysis metadata
    confidence_score: float = Field(..., description="Analysis confidence")
    analysis_notes: Optional[str] = Field(None, description="Additional analysis notes")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class DesignBlueprint(BaseModel):
    """Design blueprint model"""
    layout: Dict[str, Any] = Field(..., description="Layout specification")
    components: List[Dict[str, Any]] = Field(..., description="Design components")
    typography: Dict[str, Any] = Field(..., description="Typography settings")
    color_scheme: Dict[str, Any] = Field(..., description="Color scheme")
    spacing: Dict[str, Any] = Field(..., description="Spacing specifications")
    responsive_rules: Optional[Dict[str, Any]] = Field(None, description="Responsive design rules")
    
    # Blueprint metadata
    version: str = Field(default="1.0", description="Blueprint version")
    generated_by: str = Field(..., description="Agent that generated blueprint")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class DesignVariant(BaseModel):
    """Design variant model"""
    id: str = Field(..., description="Variant ID")
    name: str = Field(..., description="Variant name")
    description: Optional[str] = Field(None, description="Variant description")
    
    # Generated assets
    background_url: Optional[str] = Field(None, description="Background image URL")
    preview_url: Optional[str] = Field(None, description="Preview image URL")
    svg_code: Optional[str] = Field(None, description="SVG code")
    figma_code: Optional[str] = Field(None, description="Figma plugin code")
    
    # Quality metrics
    quality_score: Optional[float] = Field(None, description="Overall quality score")
    review_feedback: Optional[Dict[str, Any]] = Field(None, description="Review feedback")
    
    # Generation metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_time: Optional[float] = Field(None, description="Generation time in seconds")
    generation_parameters: Optional[Dict[str, Any]] = Field(None, description="Generation parameters")


class Design(BaseModel):
    """Design model"""
    id: str = Field(..., description="Design ID")
    campaign_id: str = Field(..., description="Associated campaign ID")
    name: str = Field(..., description="Design name")
    description: Optional[str] = Field(None, description="Design description")
    
    # Design data
    strategic_direction: Optional[StrategicDirection] = Field(None, description="Strategic direction")
    blueprint: Optional[DesignBlueprint] = Field(None, description="Design blueprint")
    variants: List[DesignVariant] = Field(default_factory=list, description="Design variants")
    
    # Status and progress
    status: DesignStatus = Field(default=DesignStatus.QUEUED, description="Design status")
    progress_percentage: int = Field(default=0, description="Completion percentage")
    current_step: Optional[str] = Field(None, description="Current processing step")
    
    # Priority and scheduling
    priority: Priority = Field(default=Priority.NORMAL, description="Design priority")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled generation time")
    
    # Iteration tracking
    iteration_count: int = Field(default=0, description="Number of iterations")
    parent_design_id: Optional[str] = Field(None, description="Parent design for iterations")
    
    # Feedback and approval
    feedback_history: List[Dict[str, Any]] = Field(default_factory=list, description="Feedback history")
    approval_status: Optional[str] = Field(None, description="Approval status")
    approved_by: Optional[str] = Field(None, description="Approved by user")
    approved_at: Optional[datetime] = Field(None, description="Approval timestamp")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="Generation start time")
    completed_at: Optional[datetime] = Field(None, description="Generation completion time")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retries")


class Campaign(BaseModel):
    """Campaign model"""
    id: str = Field(..., description="Campaign ID")
    name: str = Field(..., description="Campaign name")
    description: Optional[str] = Field(None, description="Campaign description")
    
    # Campaign brief
    brief: CampaignBrief = Field(..., description="Campaign brief")
    
    # Associated data
    assets: List[Asset] = Field(default_factory=list, description="Campaign assets")
    designs: List[Design] = Field(default_factory=list, description="Campaign designs")
    
    # Status and progress
    status: CampaignStatus = Field(default=CampaignStatus.DRAFT, description="Campaign status")
    progress_percentage: int = Field(default=0, description="Overall progress")
    
    # Collaboration
    owner_id: str = Field(..., description="Campaign owner ID")
    collaborators: List[str] = Field(default_factory=list, description="Collaborator IDs")
    
    # Settings
    settings: Dict[str, Any] = Field(default_factory=dict, description="Campaign settings")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    
    # Analytics
    view_count: int = Field(default=0, description="View count")
    download_count: int = Field(default=0, description="Download count")
    share_count: int = Field(default=0, description="Share count")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Archive and cleanup
    archived_at: Optional[datetime] = Field(None, description="Archive timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")


class WorkflowSession(BaseModel):
    """Workflow session model"""
    id: str = Field(..., description="Session ID")
    campaign_id: str = Field(..., description="Associated campaign ID")
    design_id: Optional[str] = Field(None, description="Associated design ID")
    
    # Workflow state
    current_step: str = Field(..., description="Current workflow step")
    completed_steps: List[str] = Field(default_factory=list, description="Completed steps")
    remaining_steps: List[str] = Field(default_factory=list, description="Remaining steps")
    
    # Agent assignments
    active_agents: List[str] = Field(default_factory=list, description="Currently active agents")
    agent_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Agent states")
    
    # Progress tracking
    progress_percentage: int = Field(default=0, description="Overall progress")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Session data
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context")
    intermediate_results: Dict[str, Any] = Field(default_factory=dict, description="Intermediate results")
    
    # Error handling
    error_count: int = Field(default=0, description="Error count")
    last_error: Optional[str] = Field(None, description="Last error message")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Session expiration")


class GenerationMetrics(BaseModel):
    """Generation metrics model"""
    design_id: str = Field(..., description="Design ID")
    
    # Performance metrics
    total_time: float = Field(..., description="Total generation time in seconds")
    step_times: Dict[str, float] = Field(..., description="Time per step")
    
    # Quality metrics
    quality_scores: Dict[str, float] = Field(..., description="Quality scores by component")
    final_quality_score: float = Field(..., description="Final overall quality score")
    
    # Resource usage
    ai_model_calls: Dict[str, int] = Field(..., description="AI model API calls")
    processing_costs: Dict[str, float] = Field(..., description="Processing costs")
    
    # Agent performance
    agent_performance: Dict[str, Dict[str, Any]] = Field(..., description="Agent performance metrics")
    
    # User interaction
    iteration_count: int = Field(default=0, description="Number of iterations")
    user_feedback_count: int = Field(default=0, description="User feedback instances")
    approval_time: Optional[float] = Field(None, description="Time to approval in seconds")
    
    # Timestamps
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
