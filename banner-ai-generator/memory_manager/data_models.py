from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid

class AgentType(Enum):
    STRATEGIST = "strategist"
    BACKGROUND_DESIGNER = "background_designer"
    FOREGROUND_DESIGNER = "foreground_designer"
    DEVELOPER = "developer"
    DESIGN_REVIEWER = "design_reviewer"

class WorkflowState(Enum):
    INITIALIZED = "initialized"
    STRATEGIST_WORKING = "strategist_working"
    STRATEGIST_COMPLETE = "strategist_complete"
    BACKGROUND_WORKING = "background_working"
    BACKGROUND_COMPLETE = "background_complete"
    FOREGROUND_WORKING = "foreground_working"
    FOREGROUND_COMPLETE = "foreground_complete"
    DEVELOPER_WORKING = "developer_working"
    DEVELOPER_COMPLETE = "developer_complete"
    REVIEWING = "reviewing"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BrandAssets:
    """Brand assets uploaded by advertiser"""
    logo_path: Optional[str] = None
    logo_data: Optional[bytes] = None
    background_image_path: Optional[str] = None
    brand_colors: List[str] = field(default_factory=list)
    additional_assets: Dict[str, str] = field(default_factory=dict)

@dataclass
class CampaignBrief:
    """Campaign brief from advertiser"""
    campaign_name: str
    product_name: str
    target_audience: str
    primary_message: str
    call_to_action: str
    banner_size: tuple  # (width, height)
    campaign_type: str = "general"
    additional_requirements: str = ""
    deadline: Optional[datetime] = None

@dataclass
class StrategicDirection:
    """Output from Strategist Agent"""
    mood_tone: str
    target_audience_analysis: str
    primary_purpose: str
    color_palette: List[str]
    design_style: str
    logo_analysis: Dict[str, Any]
    brand_personality: str
    messaging_strategy: str

@dataclass
class BackgroundAsset:
    """Background image data"""
    image_path: str
    image_data: Optional[bytes] = None
    dimensions: tuple = (0, 0)
    is_generated: bool = False
    generation_prompt: Optional[str] = None
    refinement_iterations: int = 0
    text_free_verified: bool = False

@dataclass
class DesignElement:
    """Individual design element in blueprint"""
    id: str
    element_type: str  # 'logo', 'text', 'cta', 'decorative'
    content: str = ""
    position: Dict[str, Union[int, float]] = field(default_factory=dict)
    size: Dict[str, Union[int, float]] = field(default_factory=dict)
    style: Dict[str, Any] = field(default_factory=dict)
    relative_position: Optional[Dict[str, Any]] = None

@dataclass
class DesignBlueprint:
    """Complete design blueprint (JSON structure)"""
    blueprint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    canvas_size: tuple = (800, 600)
    background_asset: Optional[BackgroundAsset] = None
    elements: List[DesignElement] = field(default_factory=list)
    layout_rationale: str = ""
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DesignFeedback:
    """Feedback from Design Reviewer"""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    blueprint_version: int
    overall_score: float  # 0-10
    visual_hierarchy_score: float
    brand_consistency_score: float
    readability_score: float
    aesthetic_score: float
    specific_issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    is_approved: bool = False
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AgentState:
    """State of individual agent"""
    agent_type: AgentType
    status: str = "idle"
    current_task: Optional[str] = None
    progress: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    output_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CampaignData:
    """Complete campaign data structure"""
    campaign_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    brief: Optional[CampaignBrief] = None
    brand_assets: Optional[BrandAssets] = None
    strategic_direction: Optional[StrategicDirection] = None
    background_asset: Optional[BackgroundAsset] = None
    design_blueprints: List[DesignBlueprint] = field(default_factory=list)
    design_feedbacks: List[DesignFeedback] = field(default_factory=list)
    agent_states: Dict[AgentType, AgentState] = field(default_factory=dict)
    workflow_state: WorkflowState = WorkflowState.INITIALIZED
    current_iteration: int = 0
    max_iterations: int = 5
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class MemoryKey(Enum):
    """Standardized memory keys"""
    CAMPAIGN_DATA = "campaign_data"
    STRATEGIC_DIRECTION = "strategic_direction"
    BACKGROUND_ASSET = "background_asset"
    CURRENT_BLUEPRINT = "current_blueprint"
    DESIGN_HISTORY = "design_history"
    FEEDBACK_HISTORY = "feedback_history"
    AGENT_STATES = "agent_states"
    WORKFLOW_STATE = "workflow_state"
