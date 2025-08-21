"""
Asset Management Router

FastAPI router for asset-related endpoints including
upload, validation, processing, and management.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Dict, Any
import uuid
import base64
import io
from datetime import datetime
from structlog import get_logger

from ..models.request_models import (
    AssetUploadRequest,
    AssetValidationRequest,
    LogoProcessingRequest,
    BackgroundRemovalRequest,
    ImageOptimizationRequest,
    BulkAssetUploadRequest
)
from ..models.response_models import (
    AssetResponse,
    AssetListResponse,
    ValidationResponse,
    ProcessingResponse,
    BulkOperationResponse
)
from ..models.campaign_models import Asset, AssetType, AssetMetadata
from api.dependencies import get_banner_app

logger = get_logger(__name__)

router = APIRouter()


@router.post("/upload", response_model=AssetResponse)
async def upload_asset(
    request: AssetUploadRequest,
    campaign_id: Optional[str] = Query(None, description="Associate with campaign"),
    banner_app = Depends(get_banner_app)
):
    """
    Upload a single asset
    
    Upload and process a single asset (logo, image, etc.)
    with automatic validation and optimization.
    """
    try:
        logger.info(f"Uploading asset: {request.filename}")
        
        # Generate asset ID
        asset_id = f"asset_{uuid.uuid4().hex[:12]}"
        
        # Validate file
        validation_result = await banner_app.file_validator.validate_file(
            request.file_data,
            request.filename,
            request.asset_type
        )
        
        if not validation_result.get("valid", False):
            raise HTTPException(
                status_code=400,
                detail=f"File validation failed: {validation_result.get('issues', [])}"
            )
        
        # Process based on asset type
        processing_result = {}
        
        if request.asset_type == "logo":
            # Use logo processor
            processing_result = await banner_app.logo_processor.process_logo(
                request.file_data,
                request.filename
            )
        elif request.asset_type in ["image", "background"]:
            # Use image processor
            processing_result = await banner_app.image_processor.process_upload(
                request.file_data,
                request.filename
            )
        
        # Create asset metadata
        file_info = validation_result.get("file_info", {})
        metadata = AssetMetadata(
            filename=request.filename,
            file_size=file_info.get("size_bytes", 0),
            mime_type=file_info.get("detected_mime_type", "application/octet-stream"),
            dimensions=file_info.get("dimensions"),
            quality_score=processing_result.get("quality_score")
        )
        
        # Create asset record
        asset = Asset(
            id=asset_id,
            campaign_id=campaign_id or "unassigned",
            asset_type=AssetType(request.asset_type),
            filename=request.filename,
            storage_path=f"/assets/{asset_id}",
            file_data=request.file_data if len(request.file_data) < 100000 else None,  # Store small files inline
            metadata=metadata,
            description=request.description,
            tags=request.tags,
            processed=processing_result.get("success", False),
            processing_results=processing_result
        )
        
        # TODO: Store asset in database and file storage
        
        logger.info(f"Asset uploaded successfully: {asset_id}")
        
        return AssetResponse(
            success=True,
            message="Asset uploaded successfully",
            asset=asset.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-file", response_model=AssetResponse)
async def upload_file(
    file: UploadFile = File(...),
    asset_type: str = Query(..., description="Type of asset"),
    campaign_id: Optional[str] = Query(None, description="Associate with campaign"),
    description: Optional[str] = Query(None, description="Asset description"),
    banner_app = Depends(get_banner_app)
):
    """
    Upload asset file
    
    Upload asset using multipart file upload.
    """
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        
        # Encode to base64
        file_data = base64.b64encode(file_content).decode()
        
        # Create upload request
        request = AssetUploadRequest(
            asset_type=asset_type,
            filename=file.filename,
            file_data=file_data,
            description=description
        )
        
        # Use regular upload endpoint
        return await upload_asset(request, campaign_id, banner_app)
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=AssetListResponse)
async def list_assets(
    campaign_id: Optional[str] = Query(None, description="Filter by campaign"),
    asset_type: Optional[str] = Query(None, description="Filter by asset type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    banner_app = Depends(get_banner_app)
):
    """
    List assets with filtering and pagination
    """
    try:
        logger.info(f"Listing assets: campaign_id={campaign_id}, type={asset_type}")
        
        # TODO: Implement actual database query
        # Mock data for now
        assets = [
            {
                "id": f"asset_{i}",
                "campaign_id": campaign_id or f"campaign_{i % 3}",
                "asset_type": "logo" if i % 3 == 0 else "image",
                "filename": f"asset_{i}.png",
                "file_size": 1024 * (i + 1),
                "uploaded_at": datetime.utcnow().isoformat(),
                "processed": True
            }
            for i in range(1, 21)
        ]
        
        # Apply filters
        if campaign_id:
            assets = [a for a in assets if a["campaign_id"] == campaign_id]
        if asset_type:
            assets = [a for a in assets if a["asset_type"] == asset_type]
        
        # Apply pagination
        total = len(assets)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_assets = assets[start:end]
        
        return AssetListResponse(
            success=True,
            assets=paginated_assets,
            total=total
        )
        
    except Exception as e:
        logger.error(f"Error listing assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{asset_id}", response_model=AssetResponse)
async def get_asset(
    asset_id: str = Path(..., description="Asset ID"),
    banner_app = Depends(get_banner_app)
):
    """
    Get asset details
    """
    try:
        logger.info(f"Getting asset: {asset_id}")
        
        # TODO: Get from database
        # Mock data for now
        asset_data = {
            "id": asset_id,
            "campaign_id": "campaign_123",
            "asset_type": "logo",
            "filename": "logo.png",
            "file_size": 15360,
            "uploaded_at": datetime.utcnow().isoformat(),
            "processed": True,
            "metadata": {
                "dimensions": {"width": 300, "height": 150},
                "quality_score": 0.95
            }
        }
        
        return AssetResponse(
            success=True,
            message="Asset retrieved successfully",
            asset=asset_data
        )
        
    except Exception as e:
        logger.error(f"Error getting asset {asset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{asset_id}")
async def delete_asset(
    asset_id: str = Path(..., description="Asset ID"),
    banner_app = Depends(get_banner_app)
):
    """
    Delete an asset
    """
    try:
        logger.info(f"Deleting asset: {asset_id}")
        
        # TODO: Delete from database and file storage
        
        return {"success": True, "message": "Asset deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting asset {asset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidationResponse)
async def validate_asset(
    request: AssetValidationRequest,
    banner_app = Depends(get_banner_app)
):
    """
    Validate asset file
    
    Validate an asset file without uploading it.
    """
    try:
        logger.info(f"Validating asset: {request.filename}")
        
        # Validate file
        validation_result = await banner_app.file_validator.validate_file(
            request.file_data,
            request.filename,
            request.expected_type,
            request.validation_rules or {}
        )
        
        return ValidationResponse(
            success=True,
            message="File validation completed",
            validation_result=validation_result
        )
        
    except Exception as e:
        logger.error(f"Error validating asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-logo", response_model=ProcessingResponse)
async def process_logo(
    request: LogoProcessingRequest,
    banner_app = Depends(get_banner_app)
):
    """
    Process logo with specialized optimization
    """
    try:
        logger.info(f"Processing logo: {request.filename}")
        
        # Process logo
        processing_result = await banner_app.logo_processor.process_logo(
            request.image_data,
            request.filename,
            request.target_sizes,
            request.background_color
        )
        
        return ProcessingResponse(
            success=True,
            message="Logo processing completed",
            processing_result=processing_result
        )
        
    except Exception as e:
        logger.error(f"Error processing logo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/remove-background", response_model=ProcessingResponse)
async def remove_background(
    request: BackgroundRemovalRequest,
    banner_app = Depends(get_banner_app)
):
    """
    Remove background from image
    """
    try:
        logger.info("Removing background from image")
        
        # Remove background
        processing_result = await banner_app.logo_processor.remove_background(
            request.image_data,
            request.method,
            request.tolerance
        )
        
        return ProcessingResponse(
            success=True,
            message="Background removal completed",
            processing_result=processing_result
        )
        
    except Exception as e:
        logger.error(f"Error removing background: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-image", response_model=ProcessingResponse)
async def optimize_image(
    request: ImageOptimizationRequest,
    banner_app = Depends(get_banner_app)
):
    """
    Optimize image for web delivery
    """
    try:
        logger.info("Optimizing image for web")
        
        # Optimize image
        processing_result = await banner_app.image_processor.optimize_for_web(
            request.image_data,
            request.target_size_kb,
            request.maintain_quality
        )
        
        return ProcessingResponse(
            success=True,
            message="Image optimization completed",
            processing_result=processing_result
        )
        
    except Exception as e:
        logger.error(f"Error optimizing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-upload", response_model=BulkOperationResponse)
async def bulk_upload_assets(
    request: BulkAssetUploadRequest,
    banner_app = Depends(get_banner_app)
):
    """
    Upload multiple assets in bulk
    """
    try:
        logger.info(f"Bulk uploading {len(request.assets)} assets")
        
        results = []
        successful = 0
        failed = 0
        
        for asset_request in request.assets:
            try:
                # Upload individual asset
                result = await upload_asset(asset_request, request.campaign_id, banner_app)
                results.append({
                    "filename": asset_request.filename,
                    "success": True,
                    "asset_id": result.asset["id"]
                })
                successful += 1
                
            except Exception as e:
                results.append({
                    "filename": asset_request.filename,
                    "success": False,
                    "error": str(e)
                })
                failed += 1
        
        summary = {
            "total": len(request.assets),
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / len(request.assets)) * 100 if request.assets else 0
        }
        
        return BulkOperationResponse(
            success=True,
            message="Bulk upload completed",
            results=results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in bulk upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{asset_id}/download")
async def download_asset(
    asset_id: str = Path(..., description="Asset ID"),
    format: Optional[str] = Query(None, description="Download format"),
    banner_app = Depends(get_banner_app)
):
    """
    Download asset file
    """
    try:
        logger.info(f"Downloading asset: {asset_id}")
        
        # TODO: Get asset from storage
        # For now, return a placeholder
        
        # Mock file content
        file_content = b"placeholder file content"
        filename = f"asset_{asset_id}.png"
        
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error downloading asset {asset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{asset_id}/preview")
async def preview_asset(
    asset_id: str = Path(..., description="Asset ID"),
    width: Optional[int] = Query(None, description="Preview width"),
    height: Optional[int] = Query(None, description="Preview height"),
    banner_app = Depends(get_banner_app)
):
    """
    Get asset preview image
    """
    try:
        logger.info(f"Getting preview for asset: {asset_id}")
        
        # TODO: Generate or get cached preview
        # For now, return placeholder
        
        from PIL import Image
        import io
        
        # Create placeholder preview
        preview_size = (width or 300, height or 200)
        preview_image = Image.new('RGB', preview_size, color='lightgray')
        
        # Convert to bytes
        buffer = io.BytesIO()
        preview_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="image/png"
        )
        
    except Exception as e:
        logger.error(f"Error getting preview for asset {asset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
