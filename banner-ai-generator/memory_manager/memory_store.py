"""
Memory Store Module

Provides persistent storage for campaign data, design history,
and system state using SQLAlchemy ORM.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from structlog import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class Campaign(Base):
    """Campaign database model"""
    __tablename__ = "campaigns"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    brief = Column(JSON, nullable=False)
    brand_assets = Column(JSON, default=dict)
    target_audience = Column(JSON, default=dict)
    mood_board = Column(JSON, default=list)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DesignIteration(Base):
    """Design iteration database model"""
    __tablename__ = "design_iterations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id = Column(String, nullable=False)
    iteration_number = Column(Integer, nullable=False)
    background_url = Column(String)
    blueprint = Column(JSON)
    svg_code = Column(Text)
    figma_code = Column(Text)
    feedback = Column(JSON, default=list)
    status = Column(String, default="in_progress")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AssetFile(Base):
    """Asset file database model"""
    __tablename__ = "asset_files"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_metadata = Column(JSON, default=dict)  # Changed from 'metadata' to 'file_metadata'
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemState(Base):
    """System state database model"""
    __tablename__ = "system_state"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String, unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MemoryStore:
    """
    Persistent storage manager using SQLAlchemy
    """
    
    def __init__(self, database_url: str = "sqlite:///./banner_generator.db"):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database initialized")
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def create_campaign(self, name: str, brief: Dict[str, Any], **kwargs) -> str:
        """Create a new campaign"""
        try:
            with self.get_session() as session:
                campaign = Campaign(
                    name=name,
                    brief=brief,
                    brand_assets=kwargs.get("brand_assets", {}),
                    target_audience=kwargs.get("target_audience", {}),
                    mood_board=kwargs.get("mood_board", [])
                )
                session.add(campaign)
                session.commit()
                
                logger.info(f"Campaign created: {campaign.id}")
                return campaign.id
        except Exception as e:
            logger.error(f"Failed to create campaign: {e}")
            raise
    
    def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Get campaign by ID"""
        try:
            with self.get_session() as session:
                campaign = session.query(Campaign).filter_by(id=campaign_id).first()
                if campaign:
                    # Detach from session
                    session.expunge(campaign)
                return campaign
        except Exception as e:
            logger.error(f"Failed to get campaign: {e}")
            return None
    
    def update_campaign(self, campaign_id: str, **updates) -> bool:
        """Update campaign data"""
        try:
            with self.get_session() as session:
                campaign = session.query(Campaign).filter_by(id=campaign_id).first()
                if campaign:
                    for key, value in updates.items():
                        if hasattr(campaign, key):
                            setattr(campaign, key, value)
                    campaign.updated_at = datetime.utcnow()
                    session.commit()
                    logger.info(f"Campaign {campaign_id} updated")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to update campaign: {e}")
            return False
    
    def list_campaigns(self, status: Optional[str] = None) -> List[Campaign]:
        """List all campaigns"""
        try:
            with self.get_session() as session:
                query = session.query(Campaign)
                if status:
                    query = query.filter_by(status=status)
                
                campaigns = query.order_by(Campaign.created_at.desc()).all()
                
                # Detach from session
                for campaign in campaigns:
                    session.expunge(campaign)
                
                return campaigns
        except Exception as e:
            logger.error(f"Failed to list campaigns: {e}")
            return []
    
    def create_design_iteration(self, campaign_id: str, **kwargs) -> str:
        """Create a new design iteration"""
        try:
            with self.get_session() as session:
                # Get next iteration number
                last_iteration = session.query(DesignIteration).filter_by(
                    campaign_id=campaign_id
                ).order_by(DesignIteration.iteration_number.desc()).first()
                
                iteration_number = 1 if not last_iteration else last_iteration.iteration_number + 1
                
                iteration = DesignIteration(
                    campaign_id=campaign_id,
                    iteration_number=iteration_number,
                    background_url=kwargs.get("background_url"),
                    blueprint=kwargs.get("blueprint"),
                    svg_code=kwargs.get("svg_code"),
                    figma_code=kwargs.get("figma_code"),
                    feedback=kwargs.get("feedback", []),
                    status=kwargs.get("status", "in_progress")
                )
                session.add(iteration)
                session.commit()
                
                logger.info(f"Design iteration created: {iteration.id}")
                return iteration.id
        except Exception as e:
            logger.error(f"Failed to create design iteration: {e}")
            raise
    
    def get_design_iteration(self, iteration_id: str) -> Optional[DesignIteration]:
        """Get design iteration by ID"""
        try:
            with self.get_session() as session:
                iteration = session.query(DesignIteration).filter_by(id=iteration_id).first()
                if iteration:
                    session.expunge(iteration)
                return iteration
        except Exception as e:
            logger.error(f"Failed to get design iteration: {e}")
            return None
    
    def get_campaign_iterations(self, campaign_id: str) -> List[DesignIteration]:
        """Get all iterations for a campaign"""
        try:
            with self.get_session() as session:
                iterations = session.query(DesignIteration).filter_by(
                    campaign_id=campaign_id
                ).order_by(DesignIteration.iteration_number.desc()).all()
                
                # Detach from session
                for iteration in iterations:
                    session.expunge(iteration)
                
                return iterations
        except Exception as e:
            logger.error(f"Failed to get campaign iterations: {e}")
            return []
    
    def update_iteration(self, iteration_id: str, **updates) -> bool:
        """Update design iteration"""
        try:
            with self.get_session() as session:
                iteration = session.query(DesignIteration).filter_by(id=iteration_id).first()
                if iteration:
                    for key, value in updates.items():
                        if hasattr(iteration, key):
                            setattr(iteration, key, value)
                    iteration.updated_at = datetime.utcnow()
                    session.commit()
                    logger.info(f"Iteration {iteration_id} updated")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to update iteration: {e}")
            return False
    
    def add_feedback_to_iteration(self, iteration_id: str, feedback: Dict[str, Any]) -> bool:
        """Add feedback to a design iteration"""
        try:
            with self.get_session() as session:
                iteration = session.query(DesignIteration).filter_by(id=iteration_id).first()
                if iteration:
                    current_feedback = iteration.feedback or []
                    current_feedback.append({
                        **feedback,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    iteration.feedback = current_feedback
                    iteration.updated_at = datetime.utcnow()
                    session.commit()
                    logger.info(f"Feedback added to iteration {iteration_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            return False
    
    def store_asset_file(self, campaign_id: str, filename: str, file_path: str, 
                        file_type: str, file_size: int, file_metadata: Dict[str, Any] = None) -> str:
        """Store asset file information"""
        try:
            with self.get_session() as session:
                asset = AssetFile(
                    campaign_id=campaign_id,
                    filename=filename,
                    file_path=file_path,
                    file_type=file_type,
                    file_size=file_size,
                    file_metadata=file_metadata or {}  # Changed parameter name to match column
                )
                session.add(asset)
                session.commit()
                
                logger.info(f"Asset file stored: {asset.id}")
                return asset.id
        except Exception as e:
            logger.error(f"Failed to store asset file: {e}")
            raise
    
    def get_campaign_assets(self, campaign_id: str) -> List[AssetFile]:
        """Get all assets for a campaign"""
        try:
            with self.get_session() as session:
                assets = session.query(AssetFile).filter_by(
                    campaign_id=campaign_id
                ).order_by(AssetFile.created_at.desc()).all()
                
                # Detach from session
                for asset in assets:
                    session.expunge(asset)
                
                return assets
        except Exception as e:
            logger.error(f"Failed to get campaign assets: {e}")
            return []
    
    def set_system_state(self, key: str, value: Any):
        """Set system state value"""
        try:
            with self.get_session() as session:
                state = session.query(SystemState).filter_by(key=key).first()
                if state:
                    state.value = value
                    state.updated_at = datetime.utcnow()
                else:
                    state = SystemState(key=key, value=value)
                    session.add(state)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to set system state: {e}")
    
    def get_system_state(self, key: str) -> Any:
        """Get system state value"""
        try:
            with self.get_session() as session:
                state = session.query(SystemState).filter_by(key=key).first()
                return state.value if state else None
        except Exception as e:
            logger.error(f"Failed to get system state: {e}")
            return None
