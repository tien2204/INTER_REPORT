"""
SQLAlchemy Database Models

Defines all database models for the Banner AI Generator system
including campaigns, assets, designs, users, and metrics.
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, 
    ForeignKey, JSON, LargeBinary, Enum as SQLEnum, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum
import uuid


Base = declarative_base()


# Enums
class CampaignStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    GENERATING = "generating"
    REVIEW = "review"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    CANCELLED = "cancelled"


class AssetType(str, Enum):
    LOGO = "logo"
    IMAGE = "image"
    REFERENCE = "reference"
    BACKGROUND = "background"
    ICON = "icon"
    FONT = "font"
    OTHER = "other"


class DesignStatus(str, Enum):
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


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    API_USER = "api_user"
    GUEST = "guest"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# Models
class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Profile information
    full_name = Column(String(100))
    company = Column(String(100))
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # API access
    api_key = Column(String(64), unique=True, index=True)
    api_key_created_at = Column(DateTime(timezone=True))
    
    # Preferences
    preferences = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login_at = Column(DateTime(timezone=True))
    
    # Relationships
    campaigns = relationship("Campaign", back_populates="owner", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"


class Campaign(Base):
    """Campaign model for banner generation projects"""
    __tablename__ = "campaigns"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text)
    
    # Owner
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    
    # Campaign brief
    company_name = Column(String(100), nullable=False)
    product_name = Column(String(100))
    primary_message = Column(String(500), nullable=False)
    cta_text = Column(String(100), nullable=False)
    target_audience = Column(Text, nullable=False)
    
    # Brand information
    industry = Column(String(50), nullable=False, index=True)
    mood = Column(String(50), nullable=False)
    tone = Column(String(50), nullable=False)
    brand_colors = Column(JSON, default=[])
    brand_guidelines = Column(Text)
    key_messages = Column(JSON, default=[])
    
    # Banner specifications
    dimensions = Column(JSON, nullable=False)  # {"width": 800, "height": 600}
    
    # Status and progress
    status = Column(SQLEnum(CampaignStatus), default=CampaignStatus.DRAFT, nullable=False, index=True)
    progress_percentage = Column(Integer, default=0)
    current_step = Column(String(100))
    
    # Collaboration
    collaborators = Column(JSON, default=[])  # List of user IDs
    
    # Settings and preferences
    settings = Column(JSON, default={})
    generation_preferences = Column(JSON, default={})
    
    # Analytics
    view_count = Column(Integer, default=0)
    download_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    archived_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    
    # Relationships
    owner = relationship("User", back_populates="campaigns")
    assets = relationship("Asset", back_populates="campaign", cascade="all, delete-orphan")
    designs = relationship("Design", back_populates="campaign", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="campaign", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('ix_campaigns_owner_status', 'owner_id', 'status'),
        Index('ix_campaigns_industry_status', 'industry', 'status'),
        Index('ix_campaigns_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Campaign(id={self.id}, name={self.name}, status={self.status})>"


class Asset(Base):
    """Asset model for uploaded files and resources"""
    __tablename__ = "assets"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id = Column(String(36), ForeignKey("campaigns.id"), nullable=False, index=True)
    
    # Asset information
    asset_type = Column(SQLEnum(AssetType), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    mime_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # Storage information
    storage_path = Column(String(500), nullable=False)
    storage_backend = Column(String(50), default="local")  # local, s3, gcs, etc.
    public_url = Column(String(500))
    
    # File metadata
    dimensions = Column(JSON)  # {"width": 800, "height": 600}
    color_palette = Column(JSON, default=[])
    has_transparency = Column(Boolean, default=False)
    quality_score = Column(Float)
    
    # Content and description
    description = Column(Text)
    tags = Column(JSON, default=[])
    alt_text = Column(String(500))
    
    # Processing status
    processed = Column(Boolean, default=False, nullable=False)
    processing_status = Column(String(50), default="pending")
    processing_results = Column(JSON, default={})
    processing_error = Column(Text)
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime(timezone=True))
    
    # Timestamps
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    campaign = relationship("Campaign", back_populates="assets")
    
    # Indexes
    __table_args__ = (
        Index('ix_assets_campaign_type', 'campaign_id', 'asset_type'),
        Index('ix_assets_uploaded_at', 'uploaded_at'),
        Index('ix_assets_processed', 'processed'),
    )
    
    def __repr__(self):
        return f"<Asset(id={self.id}, filename={self.filename}, type={self.asset_type})>"


class Design(Base):
    """Design model for generated banner designs"""
    __tablename__ = "designs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id = Column(String(36), ForeignKey("campaigns.id"), nullable=False, index=True)
    
    # Design information
    name = Column(String(200), nullable=False)
    description = Column(Text)
    variant_name = Column(String(100))
    
    # Status and progress
    status = Column(SQLEnum(DesignStatus), default=DesignStatus.QUEUED, nullable=False, index=True)
    progress_percentage = Column(Integer, default=0)
    current_step = Column(String(100))
    current_agent = Column(String(50))
    
    # Priority and scheduling
    priority = Column(SQLEnum(Priority), default=Priority.NORMAL, nullable=False)
    scheduled_at = Column(DateTime(timezone=True))
    
    # Strategic direction
    strategic_direction = Column(JSON, default={})
    analyzed_mood = Column(String(50))
    analyzed_tone = Column(String(50))
    color_palette = Column(JSON, default=[])
    visual_style = Column(String(100))
    
    # Design blueprint
    blueprint = Column(JSON, default={})
    layout_spec = Column(JSON, default={})
    typography_spec = Column(JSON, default={})
    component_specs = Column(JSON, default=[])
    
    # Generated content
    background_url = Column(String(500))
    preview_url = Column(String(500))
    svg_code = Column(Text)
    figma_code = Column(Text)
    
    # Quality metrics
    quality_score = Column(Float)
    design_reviewer_score = Column(Float)
    user_rating = Column(Float)
    
    # Generation metadata
    generation_parameters = Column(JSON, default={})
    generation_time = Column(Float)  # in seconds
    ai_model_versions = Column(JSON, default={})
    
    # Iteration tracking
    iteration_count = Column(Integer, default=0)
    parent_design_id = Column(String(36), ForeignKey("designs.id"), index=True)
    iteration_feedback = Column(Text)
    changes_requested = Column(JSON, default=[])
    
    # Approval workflow
    approval_status = Column(String(50))
    approved_by = Column(String(36), ForeignKey("users.id"))
    approved_at = Column(DateTime(timezone=True))
    rejection_reason = Column(Text)
    
    # Export information
    export_formats = Column(JSON, default=[])
    export_urls = Column(JSON, default={})
    download_count = Column(Integer, default=0)
    
    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    campaign = relationship("Campaign", back_populates="designs")
    approver = relationship("User", foreign_keys=[approved_by])
    parent_design = relationship("Design", remote_side=[id], backref="iterations")
    feedback_entries = relationship("Feedback", back_populates="design", cascade="all, delete-orphan")
    metrics = relationship("GenerationMetrics", back_populates="design", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('ix_designs_campaign_status', 'campaign_id', 'status'),
        Index('ix_designs_status_priority', 'status', 'priority'),
        Index('ix_designs_created_at', 'created_at'),
        Index('ix_designs_parent', 'parent_design_id'),
    )
    
    def __repr__(self):
        return f"<Design(id={self.id}, name={self.name}, status={self.status})>"


class Session(Base):
    """Session model for tracking agent workflows"""
    __tablename__ = "sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id = Column(String(36), ForeignKey("campaigns.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    design_id = Column(String(36), ForeignKey("designs.id"), index=True)
    
    # Session information
    session_type = Column(String(50), nullable=False)  # workflow, analysis, generation, etc.
    workflow_name = Column(String(100))
    
    # Workflow state
    current_step = Column(String(100))
    completed_steps = Column(JSON, default=[])
    remaining_steps = Column(JSON, default=[])
    step_history = Column(JSON, default=[])
    
    # Agent assignments
    active_agents = Column(JSON, default=[])
    agent_states = Column(JSON, default={})
    agent_assignments = Column(JSON, default={})
    
    # Progress tracking
    progress_percentage = Column(Integer, default=0)
    estimated_completion = Column(DateTime(timezone=True))
    
    # Session data and context
    context = Column(JSON, default={})
    intermediate_results = Column(JSON, default={})
    session_metadata = Column(JSON, default={})
    
    # Configuration
    configuration = Column(JSON, default={})
    timeout_settings = Column(JSON, default={})
    retry_settings = Column(JSON, default={})
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_paused = Column(Boolean, default=False, nullable=False)
    pause_reason = Column(String(200))
    
    # Error handling
    error_count = Column(Integer, default=0)
    last_error = Column(Text)
    error_history = Column(JSON, default=[])
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    paused_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    
    # Relationships
    campaign = relationship("Campaign", back_populates="sessions")
    user = relationship("User", back_populates="sessions")
    design = relationship("Design")
    
    # Indexes
    __table_args__ = (
        Index('ix_sessions_user_active', 'user_id', 'is_active'),
        Index('ix_sessions_campaign_active', 'campaign_id', 'is_active'),
        Index('ix_sessions_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Session(id={self.id}, type={self.session_type}, active={self.is_active})>"


class Feedback(Base):
    """Feedback model for design reviews and iterations"""
    __tablename__ = "feedback"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    design_id = Column(String(36), ForeignKey("designs.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), index=True)
    
    # Feedback information
    feedback_type = Column(String(50), nullable=False)  # user, ai_reviewer, system
    feedback_source = Column(String(50))  # human, ai_agent, automated
    
    # Ratings and scores
    overall_rating = Column(Float)
    component_ratings = Column(JSON, default={})  # {"layout": 8.5, "colors": 9.0}
    
    # Textual feedback
    comments = Column(Text)
    suggestions = Column(Text)
    specific_issues = Column(JSON, default=[])
    positive_aspects = Column(JSON, default=[])
    
    # Categories and tags
    feedback_categories = Column(JSON, default=[])  # ["layout", "typography", "colors"]
    severity_level = Column(String(20))  # low, medium, high, critical
    
    # Actionability
    is_actionable = Column(Boolean, default=True)
    action_items = Column(JSON, default=[])
    priority_level = Column(SQLEnum(Priority), default=Priority.NORMAL)
    
    # Resolution tracking
    is_resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text)
    resolved_at = Column(DateTime(timezone=True))
    resolved_by = Column(String(36), ForeignKey("users.id"))
    
    # Metadata
    feedback_context = Column(JSON, default={})
    reviewer_metadata = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    design = relationship("Design", back_populates="feedback_entries")
    user = relationship("User", foreign_keys=[user_id])
    resolver = relationship("User", foreign_keys=[resolved_by])
    
    # Indexes
    __table_args__ = (
        Index('ix_feedback_design_type', 'design_id', 'feedback_type'),
        Index('ix_feedback_created_at', 'created_at'),
        Index('ix_feedback_resolved', 'is_resolved'),
    )
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, type={self.feedback_type}, rating={self.overall_rating})>"


class GenerationMetrics(Base):
    """Metrics model for tracking generation performance"""
    __tablename__ = "generation_metrics"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    design_id = Column(String(36), ForeignKey("designs.id"), nullable=False, index=True)
    session_id = Column(String(36), ForeignKey("sessions.id"), index=True)
    
    # Performance metrics
    total_time = Column(Float, nullable=False)  # Total generation time in seconds
    step_times = Column(JSON, default={})  # Time per step
    queue_time = Column(Float)  # Time spent in queue
    processing_time = Column(Float)  # Actual processing time
    
    # Quality metrics
    quality_scores = Column(JSON, default={})  # Quality scores by component
    final_quality_score = Column(Float)
    ai_confidence_scores = Column(JSON, default={})
    
    # Resource usage
    ai_model_calls = Column(JSON, default={})  # Number of API calls per model
    processing_costs = Column(JSON, default={})  # Costs by service
    token_usage = Column(JSON, default={})  # Token usage for LLMs
    
    # Agent performance
    agent_performance = Column(JSON, default={})  # Performance metrics per agent
    agent_handoff_times = Column(JSON, default={})  # Time for agent handoffs
    react_iterations = Column(JSON, default={})  # ReAct iterations per agent
    
    # User interaction
    iteration_count = Column(Integer, default=0)
    user_feedback_count = Column(Integer, default=0)
    approval_time = Column(Float)  # Time to approval in seconds
    
    # Success metrics
    generation_successful = Column(Boolean, nullable=False)
    user_satisfaction = Column(Float)  # User rating
    business_value_score = Column(Float)  # Estimated business value
    
    # Error tracking
    error_count = Column(Integer, default=0)
    error_types = Column(JSON, default={})
    retry_attempts = Column(Integer, default=0)
    
    # Metadata
    generation_parameters = Column(JSON, default={})
    model_versions = Column(JSON, default={})
    system_metrics = Column(JSON, default={})
    
    # Timestamps
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    generation_started_at = Column(DateTime(timezone=True))
    generation_completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    design = relationship("Design", back_populates="metrics")
    session = relationship("Session")
    
    # Indexes
    __table_args__ = (
        Index('ix_metrics_design', 'design_id'),
        Index('ix_metrics_recorded_at', 'recorded_at'),
        Index('ix_metrics_successful', 'generation_successful'),
    )
    
    def __repr__(self):
        return f"<GenerationMetrics(id={self.id}, total_time={self.total_time}, successful={self.generation_successful})>"
