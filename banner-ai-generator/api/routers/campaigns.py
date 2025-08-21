"""
Campaign Management Router

FastAPI router for campaign-related endpoints including
creation, management, and lifecycle operations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
from structlog import get_logger

from ..models.request_models import (
    CampaignBriefRequest, 
    CampaignUpdateRequest,
    SearchRequest
)
from ..models.response_models import (
    CampaignResponse,
    CampaignListResponse, 
    BaseResponse,
    ErrorResponse
)
from ..models.campaign_models import Campaign, CampaignBrief, CampaignStatus
from api.dependencies import get_banner_app

logger = get_logger(__name__)

router = APIRouter()


@router.post("/", response_model=CampaignResponse)
async def create_campaign(
    request: CampaignBriefRequest,
    banner_app = Depends(get_banner_app)
):
    """
    Create a new banner campaign
    
    Creates a new campaign with the provided brief and initializes
    the AI agent workflow for banner generation.
    """
    try:
        logger.info("Creating new campaign")
        
        # Generate campaign ID
        campaign_id = f"campaign_{uuid.uuid4().hex[:12]}"
        
        # Create campaign brief
        brief = CampaignBrief(
            company_name=request.company_name,
            product_name=request.product_name,
            primary_message=request.primary_message,
            cta_text=request.cta_text,
            target_audience=request.target_audience,
            industry=request.industry.value,
            mood=request.mood.value,
            tone=request.tone.value,
            key_messages=request.key_messages,
            brand_colors=request.brand_colors,
            brand_guidelines=request.brand_guidelines,
            dimensions={
                "width": request.dimensions.width,
                "height": request.dimensions.height
            }
        )
        
        # Create campaign
        campaign = Campaign(
            id=campaign_id,
            name=f"{request.company_name} - {request.primary_message[:50]}",
            brief=brief,
            owner_id="user_placeholder",  # TODO: Get from auth
            status=CampaignStatus.DRAFT
        )
        
        # Store campaign in shared memory
        campaign_data = {
            "id": campaign_id,
            "brief": brief.dict(),
            "status": "draft",
            "created_at": datetime.utcnow().isoformat()
        }
        
        await banner_app.shared_memory.set_campaign_data(campaign_id, campaign_data)
        
        # Initialize strategist agent workflow
        if banner_app.strategist_agent:
            await banner_app.strategist_agent.create_campaign(
                brief.dict(), 
                {}  # Empty assets for now
            )
        
        logger.info(f"Campaign created successfully: {campaign_id}")
        
        return CampaignResponse(
            success=True,
            message="Campaign created successfully",
            campaign=campaign.dict()
        )
        
    except Exception as e:
        logger.error(f"Error creating campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=CampaignListResponse)
async def list_campaigns(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    search: Optional[str] = Query(None, description="Search query"),
    banner_app = Depends(get_banner_app)
):
    """
    List campaigns with filtering and pagination
    
    Retrieve a paginated list of campaigns with optional filtering
    by status, industry, or search terms.
    """
    try:
        logger.info(f"Listing campaigns: page={page}, page_size={page_size}")
        
        # TODO: Implement actual database query
        # For now, return mock data
        campaigns = [
            {
                "id": f"campaign_{i}",
                "name": f"Sample Campaign {i}",
                "company_name": f"Company {i}",
                "status": "active" if i % 2 == 0 else "draft",
                "industry": "technology",
                "created_at": datetime.utcnow().isoformat(),
                "progress_percentage": min(100, i * 10)
            }
            for i in range(1, 11)
        ]
        
        # Apply filters
        if status:
            campaigns = [c for c in campaigns if c["status"] == status]
        if industry:
            campaigns = [c for c in campaigns if c["industry"] == industry]
        if search:
            campaigns = [c for c in campaigns if search.lower() in c["name"].lower()]
        
        # Apply pagination
        total = len(campaigns)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_campaigns = campaigns[start:end]
        
        return CampaignListResponse(
            success=True,
            campaigns=paginated_campaigns,
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing campaigns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(
    campaign_id: str = Path(..., description="Campaign ID"),
    banner_app = Depends(get_banner_app)
):
    """
    Get campaign details
    
    Retrieve detailed information about a specific campaign
    including its brief, assets, and designs.
    """
    try:
        logger.info(f"Getting campaign: {campaign_id}")
        
        # Get campaign from shared memory
        campaign_data = await banner_app.shared_memory.get_campaign_data(campaign_id)
        
        if not campaign_data:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        return CampaignResponse(
            success=True,
            message="Campaign retrieved successfully",
            campaign=campaign_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{campaign_id}", response_model=CampaignResponse)
async def update_campaign(
    campaign_id: str = Path(..., description="Campaign ID"),
    request: CampaignUpdateRequest = None,
    banner_app = Depends(get_banner_app)
):
    """
    Update campaign
    
    Update campaign details including brief, status, or settings.
    """
    try:
        logger.info(f"Updating campaign: {campaign_id}")
        
        # Get existing campaign
        campaign_data = await banner_app.shared_memory.get_campaign_data(campaign_id)
        
        if not campaign_data:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Apply updates
        if request and request.updates:
            campaign_data.update(request.updates)
            campaign_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Save updated campaign
        await banner_app.shared_memory.set_campaign_data(campaign_id, campaign_data)
        
        return CampaignResponse(
            success=True,
            message="Campaign updated successfully",
            campaign=campaign_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating campaign {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{campaign_id}", response_model=BaseResponse)
async def delete_campaign(
    campaign_id: str = Path(..., description="Campaign ID"),
    banner_app = Depends(get_banner_app)
):
    """
    Delete campaign
    
    Delete a campaign and all associated assets and designs.
    This action cannot be undone.
    """
    try:
        logger.info(f"Deleting campaign: {campaign_id}")
        
        # Check if campaign exists
        campaign_data = await banner_app.shared_memory.get_campaign_data(campaign_id)
        
        if not campaign_data:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # TODO: Delete associated assets and designs
        # TODO: Stop any running workflows
        
        # Delete campaign from shared memory
        await banner_app.shared_memory.delete_campaign_data(campaign_id)
        
        return BaseResponse(
            success=True,
            message="Campaign deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting campaign {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{campaign_id}/start", response_model=BaseResponse)
async def start_campaign(
    campaign_id: str = Path(..., description="Campaign ID"),
    banner_app = Depends(get_banner_app)
):
    """
    Start campaign generation
    
    Start the AI agent workflow for banner generation.
    """
    try:
        logger.info(f"Starting campaign generation: {campaign_id}")
        
        # Get campaign data
        campaign_data = await banner_app.shared_memory.get_campaign_data(campaign_id)
        
        if not campaign_data:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Update status to active
        campaign_data["status"] = "active"
        campaign_data["started_at"] = datetime.utcnow().isoformat()
        
        await banner_app.shared_memory.set_campaign_data(campaign_id, campaign_data)
        
        # Start workflow with strategist agent
        if banner_app.strategist_agent:
            # TODO: Implement workflow start
            pass
        
        return BaseResponse(
            success=True,
            message="Campaign generation started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting campaign {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{campaign_id}/stop", response_model=BaseResponse)
async def stop_campaign(
    campaign_id: str = Path(..., description="Campaign ID"),
    banner_app = Depends(get_banner_app)
):
    """
    Stop campaign generation
    
    Stop the running AI agent workflow for the campaign.
    """
    try:
        logger.info(f"Stopping campaign generation: {campaign_id}")
        
        # Get campaign data
        campaign_data = await banner_app.shared_memory.get_campaign_data(campaign_id)
        
        if not campaign_data:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Update status
        campaign_data["status"] = "stopped"
        campaign_data["stopped_at"] = datetime.utcnow().isoformat()
        
        await banner_app.shared_memory.set_campaign_data(campaign_id, campaign_data)
        
        # TODO: Stop workflow
        
        return BaseResponse(
            success=True,
            message="Campaign generation stopped"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping campaign {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{campaign_id}/status", response_model=Dict[str, Any])
async def get_campaign_status(
    campaign_id: str = Path(..., description="Campaign ID"),
    banner_app = Depends(get_banner_app)
):
    """
    Get campaign status
    
    Get the current status and progress of campaign generation.
    """
    try:
        logger.info(f"Getting campaign status: {campaign_id}")
        
        # Get campaign data
        campaign_data = await banner_app.shared_memory.get_campaign_data(campaign_id)
        
        if not campaign_data:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Get workflow status if available
        workflow_status = {}
        if banner_app.strategist_agent:
            try:
                workflow_status = await banner_app.strategist_agent.get_campaign_status(campaign_id)
            except Exception as e:
                logger.warning(f"Could not get workflow status: {e}")
        
        status_data = {
            "campaign_id": campaign_id,
            "status": campaign_data.get("status", "unknown"),
            "progress_percentage": campaign_data.get("progress_percentage", 0),
            "current_step": campaign_data.get("current_step"),
            "created_at": campaign_data.get("created_at"),
            "started_at": campaign_data.get("started_at"),
            "estimated_completion": campaign_data.get("estimated_completion"),
            "workflow_status": workflow_status
        }
        
        return status_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign status {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{campaign_id}/duplicate", response_model=CampaignResponse)
async def duplicate_campaign(
    campaign_id: str = Path(..., description="Campaign ID to duplicate"),
    new_name: Optional[str] = Query(None, description="Name for duplicated campaign"),
    banner_app = Depends(get_banner_app)
):
    """
    Duplicate campaign
    
    Create a copy of an existing campaign with the same brief and settings.
    """
    try:
        logger.info(f"Duplicating campaign: {campaign_id}")
        
        # Get original campaign
        campaign_data = await banner_app.shared_memory.get_campaign_data(campaign_id)
        
        if not campaign_data:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Generate new campaign ID
        new_campaign_id = f"campaign_{uuid.uuid4().hex[:12]}"
        
        # Create duplicate
        duplicate_data = campaign_data.copy()
        duplicate_data["id"] = new_campaign_id
        duplicate_data["name"] = new_name or f"Copy of {campaign_data.get('name', 'Campaign')}"
        duplicate_data["status"] = "draft"
        duplicate_data["created_at"] = datetime.utcnow().isoformat()
        duplicate_data["progress_percentage"] = 0
        
        # Remove generated content
        duplicate_data.pop("designs", None)
        duplicate_data.pop("started_at", None)
        duplicate_data.pop("completed_at", None)
        
        # Save duplicate
        await banner_app.shared_memory.set_campaign_data(new_campaign_id, duplicate_data)
        
        return CampaignResponse(
            success=True,
            message="Campaign duplicated successfully",
            campaign=duplicate_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error duplicating campaign {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{campaign_id}/analytics", response_model=Dict[str, Any])
async def get_campaign_analytics(
    campaign_id: str = Path(..., description="Campaign ID"),
    banner_app = Depends(get_banner_app)
):
    """
    Get campaign analytics
    
    Retrieve analytics and performance metrics for the campaign.
    """
    try:
        logger.info(f"Getting campaign analytics: {campaign_id}")
        
        # Get campaign data
        campaign_data = await banner_app.shared_memory.get_campaign_data(campaign_id)
        
        if not campaign_data:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Mock analytics data
        analytics = {
            "campaign_id": campaign_id,
            "total_designs": 3,
            "successful_generations": 2,
            "failed_generations": 1,
            "success_rate": 66.7,
            "average_generation_time": "2.5 minutes",
            "total_iterations": 5,
            "user_satisfaction_score": 8.5,
            "most_used_colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
            "performance_metrics": {
                "strategist_analysis_time": "45 seconds",
                "background_generation_time": "90 seconds",
                "foreground_design_time": "60 seconds"
            },
            "engagement_metrics": {
                "views": campaign_data.get("view_count", 0),
                "downloads": campaign_data.get("download_count", 0),
                "shares": campaign_data.get("share_count", 0)
            }
        }
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign analytics {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
