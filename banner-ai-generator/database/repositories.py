"""
Repository Pattern for Database Access

Provides data access layer with business logic encapsulation
and database operation abstraction.
"""

from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.future import select
from datetime import datetime, timedelta
from structlog import get_logger

from .models import (
    Campaign, Asset, Design, User, Session as WorkflowSession, 
    Feedback, GenerationMetrics, CampaignStatus, DesignStatus, AssetType
)

logger = get_logger(__name__)


class BaseRepository:
    """Base repository with common database operations"""
    
    def __init__(self, session: Union[Session, AsyncSession], model_class):
        self.session = session
        self.model_class = model_class
        self.is_async = isinstance(session, AsyncSession)
    
    async def get_by_id(self, id: str) -> Optional[Any]:
        """Get entity by ID"""
        try:
            if self.is_async:
                result = await self.session.execute(
                    select(self.model_class).where(self.model_class.id == id)
                )
                return result.scalar_one_or_none()
            else:
                return self.session.query(self.model_class).filter(
                    self.model_class.id == id
                ).first()
        except Exception as e:
            logger.error(f"Error getting {self.model_class.__name__} by ID {id}: {e}")
            return None
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """Get all entities with pagination"""
        try:
            if self.is_async:
                result = await self.session.execute(
                    select(self.model_class)
                    .limit(limit)
                    .offset(offset)
                    .order_by(self.model_class.created_at.desc())
                )
                return result.scalars().all()
            else:
                return (
                    self.session.query(self.model_class)
                    .order_by(self.model_class.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                    .all()
                )
        except Exception as e:
            logger.error(f"Error getting all {self.model_class.__name__}: {e}")
            return []
    
    async def create(self, **kwargs) -> Optional[Any]:
        """Create new entity"""
        try:
            entity = self.model_class(**kwargs)
            self.session.add(entity)
            
            if self.is_async:
                await self.session.commit()
                await self.session.refresh(entity)
            else:
                self.session.commit()
                self.session.refresh(entity)
            
            return entity
        except Exception as e:
            if self.is_async:
                await self.session.rollback()
            else:
                self.session.rollback()
            logger.error(f"Error creating {self.model_class.__name__}: {e}")
            return None
    
    async def update(self, id: str, **kwargs) -> Optional[Any]:
        """Update entity by ID"""
        try:
            entity = await self.get_by_id(id)
            if not entity:
                return None
            
            for key, value in kwargs.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            
            entity.updated_at = datetime.utcnow()
            
            if self.is_async:
                await self.session.commit()
                await self.session.refresh(entity)
            else:
                self.session.commit()
                self.session.refresh(entity)
            
            return entity
        except Exception as e:
            if self.is_async:
                await self.session.rollback()
            else:
                self.session.rollback()
            logger.error(f"Error updating {self.model_class.__name__} {id}: {e}")
            return None
    
    async def delete(self, id: str) -> bool:
        """Delete entity by ID"""
        try:
            entity = await self.get_by_id(id)
            if not entity:
                return False
            
            if self.is_async:
                await self.session.delete(entity)
                await self.session.commit()
            else:
                self.session.delete(entity)
                self.session.commit()
            
            return True
        except Exception as e:
            if self.is_async:
                await self.session.rollback()
            else:
                self.session.rollback()
            logger.error(f"Error deleting {self.model_class.__name__} {id}: {e}")
            return False
    
    async def count(self, **filters) -> int:
        """Count entities with optional filters"""
        try:
            query = select(func.count(self.model_class.id)) if self.is_async else self.session.query(func.count(self.model_class.id))
            
            # Apply filters
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    query = query.where(getattr(self.model_class, key) == value) if self.is_async else query.filter(getattr(self.model_class, key) == value)
            
            if self.is_async:
                result = await self.session.execute(query)
                return result.scalar()
            else:
                return query.scalar()
        except Exception as e:
            logger.error(f"Error counting {self.model_class.__name__}: {e}")
            return 0


class CampaignRepository(BaseRepository):
    """Repository for Campaign operations"""
    
    def __init__(self, session: Union[Session, AsyncSession]):
        super().__init__(session, Campaign)
    
    async def get_by_owner(self, owner_id: str, status: Optional[str] = None, 
                          limit: int = 50, offset: int = 0) -> List[Campaign]:
        """Get campaigns by owner with optional status filter"""
        try:
            if self.is_async:
                query = select(Campaign).where(Campaign.owner_id == owner_id)
                if status:
                    query = query.where(Campaign.status == status)
                query = query.order_by(Campaign.created_at.desc()).limit(limit).offset(offset)
                
                result = await self.session.execute(query)
                return result.scalars().all()
            else:
                query = self.session.query(Campaign).filter(Campaign.owner_id == owner_id)
                if status:
                    query = query.filter(Campaign.status == status)
                return query.order_by(Campaign.created_at.desc()).limit(limit).offset(offset).all()
        except Exception as e:
            logger.error(f"Error getting campaigns by owner {owner_id}: {e}")
            return []
    
    async def get_active_campaigns(self) -> List[Campaign]:
        """Get all active campaigns"""
        try:
            active_statuses = [CampaignStatus.ACTIVE, CampaignStatus.GENERATING]
            
            if self.is_async:
                result = await self.session.execute(
                    select(Campaign).where(Campaign.status.in_(active_statuses))
                )
                return result.scalars().all()
            else:
                return self.session.query(Campaign).filter(Campaign.status.in_(active_statuses)).all()
        except Exception as e:
            logger.error(f"Error getting active campaigns: {e}")
            return []
    
    async def search_campaigns(self, query: str, owner_id: Optional[str] = None,
                              industry: Optional[str] = None, limit: int = 20) -> List[Campaign]:
        """Search campaigns by text query"""
        try:
            search_filter = or_(
                Campaign.name.ilike(f"%{query}%"),
                Campaign.company_name.ilike(f"%{query}%"),
                Campaign.primary_message.ilike(f"%{query}%")
            )
            
            filters = [search_filter]
            if owner_id:
                filters.append(Campaign.owner_id == owner_id)
            if industry:
                filters.append(Campaign.industry == industry)
            
            if self.is_async:
                result = await self.session.execute(
                    select(Campaign)
                    .where(and_(*filters))
                    .order_by(Campaign.created_at.desc())
                    .limit(limit)
                )
                return result.scalars().all()
            else:
                return (
                    self.session.query(Campaign)
                    .filter(and_(*filters))
                    .order_by(Campaign.created_at.desc())
                    .limit(limit)
                    .all()
                )
        except Exception as e:
            logger.error(f"Error searching campaigns: {e}")
            return []
    
    async def get_campaign_stats(self, campaign_id: str) -> Dict[str, Any]:
        """Get campaign statistics"""
        try:
            campaign = await self.get_by_id(campaign_id)
            if not campaign:
                return {}
            
            # Count related entities
            if self.is_async:
                assets_count = await self.session.execute(
                    select(func.count(Asset.id)).where(Asset.campaign_id == campaign_id)
                )
                designs_count = await self.session.execute(
                    select(func.count(Design.id)).where(Design.campaign_id == campaign_id)
                )
                completed_designs = await self.session.execute(
                    select(func.count(Design.id)).where(
                        and_(Design.campaign_id == campaign_id, Design.status == DesignStatus.COMPLETED)
                    )
                )
                
                return {
                    "campaign_id": campaign_id,
                    "total_assets": assets_count.scalar(),
                    "total_designs": designs_count.scalar(),
                    "completed_designs": completed_designs.scalar(),
                    "status": campaign.status,
                    "progress_percentage": campaign.progress_percentage,
                    "created_at": campaign.created_at,
                    "view_count": campaign.view_count,
                    "download_count": campaign.download_count
                }
            else:
                assets_count = self.session.query(func.count(Asset.id)).filter(Asset.campaign_id == campaign_id).scalar()
                designs_count = self.session.query(func.count(Design.id)).filter(Design.campaign_id == campaign_id).scalar()
                completed_designs = self.session.query(func.count(Design.id)).filter(
                    and_(Design.campaign_id == campaign_id, Design.status == DesignStatus.COMPLETED)
                ).scalar()
                
                return {
                    "campaign_id": campaign_id,
                    "total_assets": assets_count,
                    "total_designs": designs_count,
                    "completed_designs": completed_designs,
                    "status": campaign.status,
                    "progress_percentage": campaign.progress_percentage,
                    "created_at": campaign.created_at,
                    "view_count": campaign.view_count,
                    "download_count": campaign.download_count
                }
        except Exception as e:
            logger.error(f"Error getting campaign stats for {campaign_id}: {e}")
            return {}


class AssetRepository(BaseRepository):
    """Repository for Asset operations"""
    
    def __init__(self, session: Union[Session, AsyncSession]):
        super().__init__(session, Asset)
    
    async def get_by_campaign(self, campaign_id: str, asset_type: Optional[str] = None) -> List[Asset]:
        """Get assets by campaign with optional type filter"""
        try:
            if self.is_async:
                query = select(Asset).where(Asset.campaign_id == campaign_id)
                if asset_type:
                    query = query.where(Asset.asset_type == asset_type)
                query = query.order_by(Asset.uploaded_at.desc())
                
                result = await self.session.execute(query)
                return result.scalars().all()
            else:
                query = self.session.query(Asset).filter(Asset.campaign_id == campaign_id)
                if asset_type:
                    query = query.filter(Asset.asset_type == asset_type)
                return query.order_by(Asset.uploaded_at.desc()).all()
        except Exception as e:
            logger.error(f"Error getting assets for campaign {campaign_id}: {e}")
            return []
    
    async def get_unprocessed_assets(self) -> List[Asset]:
        """Get assets that haven't been processed yet"""
        try:
            if self.is_async:
                result = await self.session.execute(
                    select(Asset).where(Asset.processed == False)
                    .order_by(Asset.uploaded_at.asc())
                )
                return result.scalars().all()
            else:
                return self.session.query(Asset).filter(Asset.processed == False).order_by(Asset.uploaded_at.asc()).all()
        except Exception as e:
            logger.error(f"Error getting unprocessed assets: {e}")
            return []
    
    async def mark_as_processed(self, asset_id: str, processing_results: Dict[str, Any]) -> bool:
        """Mark asset as processed with results"""
        try:
            return await self.update(
                asset_id,
                processed=True,
                processing_status="completed",
                processing_results=processing_results,
                processed_at=datetime.utcnow()
            ) is not None
        except Exception as e:
            logger.error(f"Error marking asset {asset_id} as processed: {e}")
            return False
    
    async def increment_usage(self, asset_id: str) -> bool:
        """Increment asset usage count"""
        try:
            asset = await self.get_by_id(asset_id)
            if not asset:
                return False
            
            return await self.update(
                asset_id,
                usage_count=asset.usage_count + 1,
                last_used_at=datetime.utcnow()
            ) is not None
        except Exception as e:
            logger.error(f"Error incrementing usage for asset {asset_id}: {e}")
            return False


class DesignRepository(BaseRepository):
    """Repository for Design operations"""
    
    def __init__(self, session: Union[Session, AsyncSession]):
        super().__init__(session, Design)
    
    async def get_by_campaign(self, campaign_id: str, status: Optional[str] = None) -> List[Design]:
        """Get designs by campaign with optional status filter"""
        try:
            if self.is_async:
                query = select(Design).where(Design.campaign_id == campaign_id)
                if status:
                    query = query.where(Design.status == status)
                query = query.order_by(Design.created_at.desc())
                
                result = await self.session.execute(query)
                return result.scalars().all()
            else:
                query = self.session.query(Design).filter(Design.campaign_id == campaign_id)
                if status:
                    query = query.filter(Design.status == status)
                return query.order_by(Design.created_at.desc()).all()
        except Exception as e:
            logger.error(f"Error getting designs for campaign {campaign_id}: {e}")
            return []
    
    async def get_queued_designs(self) -> List[Design]:
        """Get designs in queue for processing"""
        try:
            if self.is_async:
                result = await self.session.execute(
                    select(Design)
                    .where(Design.status == DesignStatus.QUEUED)
                    .order_by(Design.priority.desc(), Design.created_at.asc())
                )
                return result.scalars().all()
            else:
                return (
                    self.session.query(Design)
                    .filter(Design.status == DesignStatus.QUEUED)
                    .order_by(Design.priority.desc(), Design.created_at.asc())
                    .all()
                )
        except Exception as e:
            logger.error(f"Error getting queued designs: {e}")
            return []
    
    async def get_designs_by_status(self, status: DesignStatus) -> List[Design]:
        """Get designs by status"""
        try:
            if self.is_async:
                result = await self.session.execute(
                    select(Design).where(Design.status == status)
                    .order_by(Design.updated_at.desc())
                )
                return result.scalars().all()
            else:
                return self.session.query(Design).filter(Design.status == status).order_by(Design.updated_at.desc()).all()
        except Exception as e:
            logger.error(f"Error getting designs by status {status}: {e}")
            return []
    
    async def update_progress(self, design_id: str, progress: int, current_step: str, 
                             current_agent: Optional[str] = None) -> bool:
        """Update design progress"""
        try:
            update_data = {
                "progress_percentage": progress,
                "current_step": current_step
            }
            if current_agent:
                update_data["current_agent"] = current_agent
            
            return await self.update(design_id, **update_data) is not None
        except Exception as e:
            logger.error(f"Error updating progress for design {design_id}: {e}")
            return False
    
    async def get_design_iterations(self, parent_design_id: str) -> List[Design]:
        """Get all iterations of a design"""
        try:
            if self.is_async:
                result = await self.session.execute(
                    select(Design).where(Design.parent_design_id == parent_design_id)
                    .order_by(Design.created_at.asc())
                )
                return result.scalars().all()
            else:
                return (
                    self.session.query(Design)
                    .filter(Design.parent_design_id == parent_design_id)
                    .order_by(Design.created_at.asc())
                    .all()
                )
        except Exception as e:
            logger.error(f"Error getting iterations for design {parent_design_id}: {e}")
            return []


class UserRepository(BaseRepository):
    """Repository for User operations"""
    
    def __init__(self, session: Union[Session, AsyncSession]):
        super().__init__(session, User)
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            if self.is_async:
                result = await self.session.execute(
                    select(User).where(User.username == username)
                )
                return result.scalar_one_or_none()
            else:
                return self.session.query(User).filter(User.username == username).first()
        except Exception as e:
            logger.error(f"Error getting user by username {username}: {e}")
            return None
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            if self.is_async:
                result = await self.session.execute(
                    select(User).where(User.email == email)
                )
                return result.scalar_one_or_none()
            else:
                return self.session.query(User).filter(User.email == email).first()
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {e}")
            return None
    
    async def get_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key"""
        try:
            if self.is_async:
                result = await self.session.execute(
                    select(User).where(User.api_key == api_key)
                )
                return result.scalar_one_or_none()
            else:
                return self.session.query(User).filter(User.api_key == api_key).first()
        except Exception as e:
            logger.error(f"Error getting user by API key: {e}")
            return None
    
    async def update_last_login(self, user_id: str) -> bool:
        """Update user's last login timestamp"""
        try:
            return await self.update(user_id, last_login_at=datetime.utcnow()) is not None
        except Exception as e:
            logger.error(f"Error updating last login for user {user_id}: {e}")
            return False


class MetricsRepository(BaseRepository):
    """Repository for GenerationMetrics operations"""
    
    def __init__(self, session: Union[Session, AsyncSession]):
        super().__init__(session, GenerationMetrics)
    
    async def get_by_design(self, design_id: str) -> Optional[GenerationMetrics]:
        """Get metrics for a specific design"""
        try:
            if self.is_async:
                result = await self.session.execute(
                    select(GenerationMetrics).where(GenerationMetrics.design_id == design_id)
                )
                return result.scalar_one_or_none()
            else:
                return self.session.query(GenerationMetrics).filter(GenerationMetrics.design_id == design_id).first()
        except Exception as e:
            logger.error(f"Error getting metrics for design {design_id}: {e}")
            return None
    
    async def get_system_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get system-wide performance metrics"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            if self.is_async:
                # Total generations
                total_result = await self.session.execute(
                    select(func.count(GenerationMetrics.id))
                    .where(GenerationMetrics.recorded_at >= cutoff_date)
                )
                total_generations = total_result.scalar()
                
                # Successful generations
                success_result = await self.session.execute(
                    select(func.count(GenerationMetrics.id))
                    .where(and_(
                        GenerationMetrics.recorded_at >= cutoff_date,
                        GenerationMetrics.generation_successful == True
                    ))
                )
                successful_generations = success_result.scalar()
                
                # Average generation time
                avg_time_result = await self.session.execute(
                    select(func.avg(GenerationMetrics.total_time))
                    .where(and_(
                        GenerationMetrics.recorded_at >= cutoff_date,
                        GenerationMetrics.generation_successful == True
                    ))
                )
                avg_generation_time = avg_time_result.scalar() or 0
                
                # Average quality score
                avg_quality_result = await self.session.execute(
                    select(func.avg(GenerationMetrics.final_quality_score))
                    .where(and_(
                        GenerationMetrics.recorded_at >= cutoff_date,
                        GenerationMetrics.final_quality_score.isnot(None)
                    ))
                )
                avg_quality_score = avg_quality_result.scalar() or 0
                
            else:
                total_generations = (
                    self.session.query(func.count(GenerationMetrics.id))
                    .filter(GenerationMetrics.recorded_at >= cutoff_date)
                    .scalar()
                )
                
                successful_generations = (
                    self.session.query(func.count(GenerationMetrics.id))
                    .filter(and_(
                        GenerationMetrics.recorded_at >= cutoff_date,
                        GenerationMetrics.generation_successful == True
                    ))
                    .scalar()
                )
                
                avg_generation_time = (
                    self.session.query(func.avg(GenerationMetrics.total_time))
                    .filter(and_(
                        GenerationMetrics.recorded_at >= cutoff_date,
                        GenerationMetrics.generation_successful == True
                    ))
                    .scalar() or 0
                )
                
                avg_quality_score = (
                    self.session.query(func.avg(GenerationMetrics.final_quality_score))
                    .filter(and_(
                        GenerationMetrics.recorded_at >= cutoff_date,
                        GenerationMetrics.final_quality_score.isnot(None)
                    ))
                    .scalar() or 0
                )
            
            success_rate = (successful_generations / total_generations * 100) if total_generations > 0 else 0
            
            return {
                "period_days": days,
                "total_generations": total_generations,
                "successful_generations": successful_generations,
                "success_rate": round(success_rate, 2),
                "avg_generation_time": round(avg_generation_time, 2),
                "avg_quality_score": round(avg_quality_score, 3),
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    async def get_agent_performance(self, agent_name: str, days: int = 7) -> Dict[str, Any]:
        """Get performance metrics for a specific agent"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # This would require more complex queries to extract agent-specific data
            # from the agent_performance JSON field
            # For now, return placeholder data
            
            return {
                "agent_name": agent_name,
                "period_days": days,
                "total_tasks": 0,
                "successful_tasks": 0,
                "avg_processing_time": 0,
                "avg_quality_score": 0
            }
            
        except Exception as e:
            logger.error(f"Error getting agent performance for {agent_name}: {e}")
            return {}
