"""
Design Management Router

FastAPI router for design-related endpoints including
generation, iteration, and export capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Dict, Any
import uuid
import asyncio
import json
from datetime import datetime
from structlog import get_logger

from ..models.request_models import (
    DesignGenerationRequest,
    DesignIterationRequest,
    DesignFeedbackRequest,
    DesignExportRequest
)
from ..models.response_models import (
    DesignResponse,
    DesignListResponse,
    DesignProgressResponse,
    ExportResponse,
    StreamingResponse as StreamingResponseModel
)
from ..models.campaign_models import Design, DesignStatus, Priority
from api.dependencies import get_banner_app

logger = get_logger(__name__)

router = APIRouter()


@router.post("/generate", response_model=DesignResponse)
async def generate_design(
    request: DesignGenerationRequest,
    banner_app = Depends(get_banner_app)
):
    """
    Generate a new design
    
    Start the AI agent workflow to generate a new banner design
    for the specified campaign.
    """
    try:
        logger.info(f"Starting design generation for campaign: {request.campaign_id}")
        
        # Generate design ID
        design_id = f"design_{uuid.uuid4().hex[:12]}"
        
        # Get campaign data
        campaign_data = await banner_app.shared_memory.get_campaign_data(request.campaign_id)
        
        if not campaign_data:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Create design record
        design = Design(
            id=design_id,
            campaign_id=request.campaign_id,
            name=f"Design for {campaign_data.get('name', 'Campaign')}",
            status=DesignStatus.QUEUED,
            priority=Priority(request.priority or "normal")
        )
        
        # Store design data
        design_data = design.dict()
        await banner_app.shared_memory.set_design_data(design_id, design_data)
        
        # Start workflow with background designer
        if banner_app.strategist_agent and banner_app.background_designer_agent:
            # Create workflow session
            session_id = f"session_{uuid.uuid4().hex[:12]}"
            
            # Start strategic analysis
            strategic_result = await banner_app.strategist_agent.analyze_campaign(
                request.campaign_id,
                request.generation_options or {}
            )
            
            if strategic_result:
                # Update design with strategic direction
                design_data["strategic_direction"] = strategic_result
                design_data["status"] = "strategy_analysis"
                design_data["progress_percentage"] = 20
                
                await banner_app.shared_memory.set_design_data(design_id, design_data)
                
                # Start background generation
                background_task = asyncio.create_task(
                    _start_background_generation(design_id, session_id, banner_app)
                )
        
        logger.info(f"Design generation started: {design_id}")
        
        return DesignResponse(
            success=True,
            message="Design generation started",
            design=design_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting design generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=DesignListResponse)
async def list_designs(
    campaign_id: Optional[str] = Query(None, description="Filter by campaign"),
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    banner_app = Depends(get_banner_app)
):
    """
    List designs with filtering and pagination
    """
    try:
        logger.info(f"Listing designs: campaign_id={campaign_id}, status={status}")
        
        # TODO: Implement actual database query
        # Mock data for now
        designs = [
            {
                "id": f"design_{i}",
                "campaign_id": campaign_id or f"campaign_{i % 3}",
                "name": f"Design {i}",
                "status": ["queued", "generating", "completed"][i % 3],
                "progress_percentage": min(100, i * 20),
                "created_at": datetime.utcnow().isoformat(),
                "preview_url": f"/designs/design_{i}/preview.png"
            }
            for i in range(1, 16)
        ]
        
        # Apply filters
        if campaign_id:
            designs = [d for d in designs if d["campaign_id"] == campaign_id]
        if status:
            designs = [d for d in designs if d["status"] == status]
        
        # Apply pagination
        total = len(designs)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_designs = designs[start:end]
        
        return DesignListResponse(
            success=True,
            designs=paginated_designs,
            total=total
        )
        
    except Exception as e:
        logger.error(f"Error listing designs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{design_id}", response_model=DesignResponse)
async def get_design(
    design_id: str = Path(..., description="Design ID"),
    include_variants: bool = Query(True, description="Include design variants"),
    banner_app = Depends(get_banner_app)
):
    """
    Get design details
    """
    try:
        logger.info(f"Getting design: {design_id}")
        
        # Get design data
        design_data = await banner_app.shared_memory.get_design_data(design_id)
        
        if not design_data:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Add runtime information
        design_data["variants_count"] = len(design_data.get("variants", []))
        design_data["last_updated"] = design_data.get("updated_at", design_data.get("created_at"))
        
        return DesignResponse(
            success=True,
            message="Design retrieved successfully",
            design=design_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting design {design_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{design_id}")
async def delete_design(
    design_id: str = Path(..., description="Design ID"),
    banner_app = Depends(get_banner_app)
):
    """
    Delete a design
    """
    try:
        logger.info(f"Deleting design: {design_id}")
        
        # Check if design exists
        design_data = await banner_app.shared_memory.get_design_data(design_id)
        
        if not design_data:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # TODO: Cancel any running generation
        # TODO: Delete associated files
        
        # Delete from shared memory
        await banner_app.shared_memory.delete_design_data(design_id)
        
        return {"success": True, "message": "Design deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting design {design_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{design_id}/iterate", response_model=DesignResponse)
async def iterate_design(
    design_id: str = Path(..., description="Design ID"),
    request: DesignIterationRequest = None,
    banner_app = Depends(get_banner_app)
):
    """
    Create design iteration
    
    Create a new iteration of the design based on feedback
    and requested changes.
    """
    try:
        logger.info(f"Creating iteration for design: {design_id}")
        
        # Get original design
        original_design = await banner_app.shared_memory.get_design_data(design_id)
        
        if not original_design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Create new iteration
        iteration_id = f"design_{uuid.uuid4().hex[:12]}"
        
        iteration_data = original_design.copy()
        iteration_data["id"] = iteration_id
        iteration_data["name"] = f"{original_design['name']} - Iteration {original_design.get('iteration_count', 0) + 1}"
        iteration_data["parent_design_id"] = design_id
        iteration_data["iteration_count"] = original_design.get("iteration_count", 0) + 1
        iteration_data["status"] = "queued"
        iteration_data["progress_percentage"] = 0
        iteration_data["created_at"] = datetime.utcnow().isoformat()
        
        # Add feedback to iteration
        if request and request.feedback:
            iteration_data["iteration_feedback"] = request.feedback
            iteration_data["changes_requested"] = request.changes_requested or []
        
        # Store iteration
        await banner_app.shared_memory.set_design_data(iteration_id, iteration_data)
        
        # Start iteration workflow
        # TODO: Implement iteration workflow
        
        return DesignResponse(
            success=True,
            message="Design iteration created",
            design=iteration_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating iteration for design {design_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{design_id}/progress", response_model=DesignProgressResponse)
async def get_design_progress(
    design_id: str = Path(..., description="Design ID"),
    banner_app = Depends(get_banner_app)
):
    """
    Get design generation progress
    """
    try:
        logger.info(f"Getting progress for design: {design_id}")
        
        # Get design data
        design_data = await banner_app.shared_memory.get_design_data(design_id)
        
        if not design_data:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Estimate completion time
        progress = design_data.get("progress_percentage", 0)
        estimated_completion = None
        
        if progress > 0 and progress < 100:
            # Simple estimation based on current progress
            import datetime as dt
            created_at = dt.datetime.fromisoformat(design_data["created_at"].replace('Z', '+00:00'))
            elapsed = (dt.datetime.utcnow() - created_at).total_seconds()
            
            if progress > 10:  # Avoid division by very small numbers
                estimated_total = elapsed * (100 / progress)
                remaining = estimated_total - elapsed
                estimated_completion = (dt.datetime.utcnow() + dt.timedelta(seconds=remaining)).isoformat()
        
        progress_info = {
            "design_id": design_id,
            "status": design_data.get("status"),
            "percentage": progress,
            "current_step": design_data.get("current_step"),
            "estimated_completion": estimated_completion,
            "steps_completed": design_data.get("steps_completed", []),
            "current_agent": design_data.get("current_agent"),
            "error_message": design_data.get("error_message")
        }
        
        return DesignProgressResponse(
            success=True,
            progress=progress_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting progress for design {design_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{design_id}/feedback")
async def submit_feedback(
    design_id: str = Path(..., description="Design ID"),
    request: DesignFeedbackRequest = None,
    banner_app = Depends(get_banner_app)
):
    """
    Submit feedback for a design
    """
    try:
        logger.info(f"Submitting feedback for design: {design_id}")
        
        # Get design data
        design_data = await banner_app.shared_memory.get_design_data(design_id)
        
        if not design_data:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Add feedback to history
        feedback_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": request.feedback_type if request else "general",
            "rating": request.rating if request else None,
            "comments": request.comments if request else "",
            "specific_issues": request.specific_issues if request else []
        }
        
        feedback_history = design_data.get("feedback_history", [])
        feedback_history.append(feedback_entry)
        
        design_data["feedback_history"] = feedback_history
        design_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Save updated design
        await banner_app.shared_memory.set_design_data(design_id, design_data)
        
        return {"success": True, "message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback for design {design_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{design_id}/export", response_model=ExportResponse)
async def export_design(
    design_id: str = Path(..., description="Design ID"),
    request: DesignExportRequest = None,
    banner_app = Depends(get_banner_app)
):
    """
    Export design in multiple formats
    """
    try:
        logger.info(f"Exporting design: {design_id}")
        
        # Get design data
        design_data = await banner_app.shared_memory.get_design_data(design_id)
        
        if not design_data:
            raise HTTPException(status_code=404, detail="Design not found")
        
        if design_data.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Design not completed yet")
        
        export_formats = request.export_formats if request else ["svg", "png"]
        
        # Generate exports
        exports = {}
        download_links = {}
        
        for format_name in export_formats:
            try:
                if format_name == "svg":
                    # Generate SVG
                    svg_code = design_data.get("svg_code", "<svg>Placeholder SVG</svg>")
                    export_path = f"/exports/{design_id}.svg"
                    exports[format_name] = export_path
                    download_links[format_name] = f"/api/v1/designs/{design_id}/download?format=svg"
                    
                elif format_name == "png":
                    # Generate PNG
                    preview_url = design_data.get("preview_url", f"/designs/{design_id}/preview.png")
                    exports[format_name] = preview_url
                    download_links[format_name] = f"/api/v1/designs/{design_id}/download?format=png"
                    
                elif format_name == "figma":
                    # Generate Figma plugin code
                    figma_code = design_data.get("figma_code", "{}")
                    export_path = f"/exports/{design_id}_figma.json"
                    exports[format_name] = export_path
                    download_links[format_name] = f"/api/v1/designs/{design_id}/download?format=figma"
                    
            except Exception as e:
                logger.error(f"Error exporting format {format_name}: {e}")
                continue
        
        return ExportResponse(
            success=True,
            message="Design exported successfully",
            exports=exports,
            download_links=download_links
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting design {design_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{design_id}/download")
async def download_design(
    design_id: str = Path(..., description="Design ID"),
    format: str = Query(..., description="Download format"),
    banner_app = Depends(get_banner_app)
):
    """
    Download design in specified format
    """
    try:
        logger.info(f"Downloading design {design_id} in format {format}")
        
        # Get design data
        design_data = await banner_app.shared_memory.get_design_data(design_id)
        
        if not design_data:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Generate file content based on format
        if format == "svg":
            content = design_data.get("svg_code", "<svg>Placeholder</svg>")
            media_type = "image/svg+xml"
            filename = f"{design_id}.svg"
            
        elif format == "png":
            # TODO: Generate actual PNG
            content = b"placeholder png content"
            media_type = "image/png"
            filename = f"{design_id}.png"
            
        elif format == "figma":
            content = design_data.get("figma_code", "{}")
            media_type = "application/json"
            filename = f"{design_id}_figma.json"
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
        # Convert to bytes if string
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        return StreamingResponse(
            io.BytesIO(content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading design {design_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/{design_id}/progress-stream")
async def design_progress_stream(
    websocket: WebSocket,
    design_id: str = Path(..., description="Design ID")
):
    """
    WebSocket endpoint for real-time design progress updates
    """
    await websocket.accept()
    
    try:
        logger.info(f"Starting progress stream for design: {design_id}")
        
        while True:
            # TODO: Get actual progress from shared memory or message queue
            # For now, send mock progress updates
            
            progress_data = {
                "design_id": design_id,
                "progress": 50,  # Mock progress
                "current_step": "Background generation",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await websocket.send_json({
                "event": "design_progress",
                "data": progress_data
            })
            
            await asyncio.sleep(5)  # Send updates every 5 seconds
            
    except Exception as e:
        logger.error(f"Error in progress stream: {e}")
    finally:
        await websocket.close()


# Helper function for background generation
async def _start_background_generation(design_id: str, session_id: str, banner_app):
    """Start background generation workflow"""
    try:
        # Send message to background designer agent
        await banner_app.message_queue.publish(
            "agent.background_designer",
            {
                "action": "start_background_design_workflow",
                "session_id": session_id,
                "design_id": design_id
            }
        )
        
        # Update design status
        design_data = await banner_app.shared_memory.get_design_data(design_id)
        if design_data:
            design_data["status"] = "background_generation"
            design_data["progress_percentage"] = 40
            design_data["current_step"] = "Background generation"
            design_data["current_agent"] = "background_designer"
            
            await banner_app.shared_memory.set_design_data(design_id, design_data)
        
    except Exception as e:
        logger.error(f"Error starting background generation: {e}")
        
        # Update design with error
        design_data = await banner_app.shared_memory.get_design_data(design_id)
        if design_data:
            design_data["status"] = "failed"
            design_data["error_message"] = str(e)
            await banner_app.shared_memory.set_design_data(design_id, design_data)
