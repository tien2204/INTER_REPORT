"""
PNG Handler

Specialized handler for PNG files including processing, validation,
optimization, and conversion capabilities.
"""

import io
import base64
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageStat, ImageFilter, ImageEnhance
from PIL.ExifTags import TAGS
from structlog import get_logger

logger = get_logger(__name__)


class PNGHandler:
    """
    Comprehensive PNG file handler
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # PNG processing options
        self.optimize_compression = config.get("optimize_compression", True)
        self.remove_metadata = config.get("remove_metadata", True)
        self.max_width = config.get("max_width", 2048)
        self.max_height = config.get("max_height", 2048)
        
        # Quality settings
        self.jpeg_quality = config.get("jpeg_quality", 85)
        self.webp_quality = config.get("webp_quality", 80)
        
        # Security settings
        self.validate_dimensions = config.get("validate_dimensions", True)
        self.max_file_size = config.get("max_file_size", 10 * 1024 * 1024)  # 10MB
        
        # Supported formats for conversion
        self.supported_formats = ["PNG", "JPEG", "WEBP", "GIF", "BMP"]
    
    async def process_png(self, 
                         png_data: bytes,
                         options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process PNG file with optimization and validation
        
        Args:
            png_data: PNG file data as bytes
            options: Processing options
        
        Returns:
            Processing result with optimized PNG and metadata
        """
        try:
            logger.info("Processing PNG file")
            
            # Validate file size
            if len(png_data) > self.max_file_size:
                raise ValueError(f"File size exceeds maximum allowed size of {self.max_file_size} bytes")
            
            # Load image
            image = Image.open(io.BytesIO(png_data))
            
            # Extract metadata
            metadata = await self._extract_metadata(image, png_data)
            
            # Validate dimensions
            if self.validate_dimensions:
                await self._validate_dimensions(image)
            
            # Create optimized version
            optimized_image = await self._optimize_image(image, options or {})
            
            # Convert back to bytes
            optimized_data = await self._image_to_bytes(optimized_image, "PNG")
            
            # Calculate optimization stats
            original_size = len(png_data)
            optimized_size = len(optimized_data)
            compression_ratio = (original_size - optimized_size) / original_size * 100 if original_size > 0 else 0
            
            result = {
                "success": True,
                "original_data": base64.b64encode(png_data).decode(),
                "optimized_data": base64.b64encode(optimized_data).decode(),
                "metadata": metadata,
                "optimization_stats": {
                    "original_size": original_size,
                    "optimized_size": optimized_size,
                    "compression_ratio": round(compression_ratio, 2),
                    "size_reduction": original_size - optimized_size
                }
            }
            
            logger.info(f"PNG processing completed: {compression_ratio:.1f}% size reduction")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PNG: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_data": base64.b64encode(png_data).decode() if png_data else ""
            }
    
    async def convert_format(self, 
                           png_data: bytes,
                           target_format: str,
                           quality: Optional[int] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Convert PNG to different format
        
        Args:
            png_data: PNG file data
            target_format: Target format (JPEG, WEBP, etc.)
            quality: Quality setting for lossy formats
            **kwargs: Additional conversion options
        
        Returns:
            Conversion result
        """
        try:
            target_format = target_format.upper()
            if target_format not in self.supported_formats:
                raise ValueError(f"Unsupported target format: {target_format}")
            
            logger.info(f"Converting PNG to {target_format}")
            
            # Load original image
            image = Image.open(io.BytesIO(png_data))
            
            # Handle transparency for formats that don't support it
            if target_format in ["JPEG", "BMP"] and image.mode in ["RGBA", "LA"]:
                # Create white background
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
                image = background
            
            # Set quality for lossy formats
            save_kwargs = {}
            if target_format == "JPEG":
                save_kwargs["quality"] = quality or self.jpeg_quality
                save_kwargs["optimize"] = True
            elif target_format == "WEBP":
                save_kwargs["quality"] = quality or self.webp_quality
                save_kwargs["optimize"] = True
            
            # Apply additional options
            save_kwargs.update(kwargs)
            
            # Convert
            converted_data = await self._image_to_bytes(image, target_format, **save_kwargs)
            
            result = {
                "success": True,
                "converted_data": base64.b64encode(converted_data).decode(),
                "format": target_format,
                "original_size": len(png_data),
                "converted_size": len(converted_data),
                "compression_ratio": (len(png_data) - len(converted_data)) / len(png_data) * 100
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error converting PNG: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def resize_image(self, 
                         png_data: bytes,
                         width: int,
                         height: int,
                         method: str = "LANCZOS",
                         maintain_aspect: bool = True) -> Dict[str, Any]:
        """
        Resize PNG image
        
        Args:
            png_data: PNG file data
            width: Target width
            height: Target height
            method: Resampling method
            maintain_aspect: Whether to maintain aspect ratio
        
        Returns:
            Resized image result
        """
        try:
            logger.info(f"Resizing PNG to {width}x{height}")
            
            # Load image
            image = Image.open(io.BytesIO(png_data))
            original_size = image.size
            
            # Calculate target size
            if maintain_aspect:
                image.thumbnail((width, height), getattr(Image, method, Image.LANCZOS))
                target_size = image.size
            else:
                image = image.resize((width, height), getattr(Image, method, Image.LANCZOS))
                target_size = (width, height)
            
            # Convert back to bytes
            resized_data = await self._image_to_bytes(image, "PNG")
            
            result = {
                "success": True,
                "resized_data": base64.b64encode(resized_data).decode(),
                "original_size": original_size,
                "target_size": target_size,
                "file_size": len(resized_data)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error resizing PNG: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def apply_filters(self, 
                          png_data: bytes,
                          filters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply image filters to PNG
        
        Args:
            png_data: PNG file data
            filters: List of filter configurations
        
        Returns:
            Filtered image result
        """
        try:
            logger.info(f"Applying {len(filters)} filters to PNG")
            
            # Load image
            image = Image.open(io.BytesIO(png_data))
            
            # Apply each filter
            for filter_config in filters:
                filter_type = filter_config.get("type", "").lower()
                params = filter_config.get("params", {})
                
                if filter_type == "blur":
                    radius = params.get("radius", 1)
                    image = image.filter(ImageFilter.GaussianBlur(radius))
                
                elif filter_type == "sharpen":
                    image = image.filter(ImageFilter.SHARPEN)
                
                elif filter_type == "brightness":
                    factor = params.get("factor", 1.0)
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(factor)
                
                elif filter_type == "contrast":
                    factor = params.get("factor", 1.0)
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(factor)
                
                elif filter_type == "saturation":
                    factor = params.get("factor", 1.0)
                    enhancer = ImageEnhance.Color(image)
                    image = enhancer.enhance(factor)
                
                elif filter_type == "grayscale":
                    image = image.convert("L").convert("RGB")
                
                else:
                    logger.warning(f"Unknown filter type: {filter_type}")
            
            # Convert back to bytes
            filtered_data = await self._image_to_bytes(image, "PNG")
            
            result = {
                "success": True,
                "filtered_data": base64.b64encode(filtered_data).decode(),
                "filters_applied": len(filters),
                "file_size": len(filtered_data)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def extract_metadata(self, png_data: bytes) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from PNG
        
        Args:
            png_data: PNG file data
        
        Returns:
            Extracted metadata
        """
        try:
            image = Image.open(io.BytesIO(png_data))
            return await self._extract_metadata(image, png_data)
            
        except Exception as e:
            logger.error(f"Error extracting PNG metadata: {e}")
            return {"error": str(e)}
    
    async def validate_png(self, png_data: bytes) -> Dict[str, Any]:
        """
        Validate PNG file
        
        Args:
            png_data: PNG file data
        
        Returns:
            Validation result
        """
        try:
            issues = []
            warnings = []
            
            # Basic file validation
            try:
                image = Image.open(io.BytesIO(png_data))
                image.verify()
            except Exception as e:
                issues.append(f"Invalid PNG file: {e}")
                return {"valid": False, "issues": issues}
            
            # Re-open for further checks (verify() closes the image)
            image = Image.open(io.BytesIO(png_data))
            
            # Check file size
            if len(png_data) > self.max_file_size:
                issues.append(f"File size ({len(png_data)} bytes) exceeds maximum ({self.max_file_size} bytes)")
            
            # Check dimensions
            width, height = image.size
            if width > self.max_width or height > self.max_height:
                warnings.append(f"Image dimensions ({width}x{height}) are very large")
            
            if width < 1 or height < 1:
                issues.append("Invalid image dimensions")
            
            # Check color mode
            if image.mode not in ["RGB", "RGBA", "L", "LA", "P"]:
                warnings.append(f"Unusual color mode: {image.mode}")
            
            # Check for transparency
            has_transparency = image.mode in ["RGBA", "LA"] or "transparency" in image.info
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "file_size": len(png_data),
                "dimensions": {"width": width, "height": height},
                "color_mode": image.mode,
                "has_transparency": has_transparency,
                "format": image.format
            }
            
        except Exception as e:
            logger.error(f"Error validating PNG: {e}")
            return {"valid": False, "error": str(e)}
    
    async def get_color_palette(self, 
                              png_data: bytes,
                              num_colors: int = 16) -> Dict[str, Any]:
        """
        Extract dominant color palette from PNG
        
        Args:
            png_data: PNG file data
            num_colors: Number of colors to extract
        
        Returns:
            Color palette information
        """
        try:
            image = Image.open(io.BytesIO(png_data))
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Get color palette using quantization
            palette_image = image.quantize(colors=num_colors)
            palette = palette_image.getpalette()
            
            # Convert palette to list of RGB tuples
            colors = []
            for i in range(0, len(palette[:num_colors * 3]), 3):
                rgb = tuple(palette[i:i + 3])
                colors.append({
                    "rgb": rgb,
                    "hex": f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                })
            
            # Calculate color statistics
            stats = ImageStat.Stat(image)
            
            result = {
                "success": True,
                "colors": colors,
                "num_colors": len(colors),
                "statistics": {
                    "mean": stats.mean,
                    "median": stats.median,
                    "stddev": stats.stddev
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting color palette: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _extract_metadata(self, image: Image.Image, png_data: bytes) -> Dict[str, Any]:
        """Extract comprehensive metadata from PNG image"""
        try:
            metadata = {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.size[0],
                "height": image.size[1],
                "file_size": len(png_data),
                "has_transparency": image.mode in ["RGBA", "LA"] or "transparency" in image.info
            }
            
            # Extract PNG info
            if hasattr(image, 'info') and image.info:
                metadata["png_info"] = dict(image.info)
            
            # Extract EXIF data if present
            try:
                exif = image._getexif()
                if exif:
                    metadata["exif"] = {}
                    for tag, value in exif.items():
                        tag_name = TAGS.get(tag, tag)
                        metadata["exif"][tag_name] = str(value)
            except:
                pass  # EXIF not available or readable
            
            # Calculate file hash
            metadata["md5_hash"] = hashlib.md5(png_data).hexdigest()
            metadata["sha256_hash"] = hashlib.sha256(png_data).hexdigest()
            
            # Color information
            if image.mode == "P":
                # Palette mode
                palette = image.getpalette()
                if palette:
                    metadata["palette_size"] = len(palette) // 3
            
            # Calculate basic statistics
            try:
                stats = ImageStat.Stat(image)
                metadata["statistics"] = {
                    "mean": stats.mean,
                    "median": stats.median,
                    "stddev": stats.stddev
                }
            except:
                pass
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {"error": str(e)}
    
    async def _validate_dimensions(self, image: Image.Image):
        """Validate image dimensions"""
        width, height = image.size
        
        if width > self.max_width:
            raise ValueError(f"Image width ({width}) exceeds maximum allowed width ({self.max_width})")
        
        if height > self.max_height:
            raise ValueError(f"Image height ({height}) exceeds maximum allowed height ({self.max_height})")
        
        if width < 1 or height < 1:
            raise ValueError("Invalid image dimensions")
    
    async def _optimize_image(self, image: Image.Image, options: Dict[str, Any]) -> Image.Image:
        """Optimize PNG image"""
        try:
            optimized = image.copy()
            
            # Remove metadata if requested
            if self.remove_metadata and hasattr(optimized, 'info'):
                # Keep only essential info
                essential_keys = ['transparency', 'gamma']
                filtered_info = {k: v for k, v in optimized.info.items() if k in essential_keys}
                optimized.info = filtered_info
            
            # Apply resize if requested
            max_size = options.get("max_size")
            if max_size:
                max_width, max_height = max_size
                if optimized.size[0] > max_width or optimized.size[1] > max_height:
                    optimized.thumbnail((max_width, max_height), Image.LANCZOS)
            
            # Apply quality optimization for formats that support it
            quality_level = options.get("quality_level", "medium")
            if quality_level == "high":
                # Minimal optimization
                pass
            elif quality_level == "low":
                # More aggressive optimization
                if optimized.mode == "RGBA":
                    # Quantize alpha channel if possible
                    try:
                        optimized = optimized.quantize(method=Image.MEDIANCUT)
                    except:
                        pass
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing image: {e}")
            return image
    
    async def _image_to_bytes(self, image: Image.Image, format: str, **kwargs) -> bytes:
        """Convert PIL Image to bytes"""
        try:
            buffer = io.BytesIO()
            
            # Set default save parameters
            save_params = {"format": format}
            
            if format == "PNG":
                save_params["optimize"] = self.optimize_compression
                save_params["compress_level"] = kwargs.get("compress_level", 6)
            
            # Update with provided parameters
            save_params.update(kwargs)
            
            image.save(buffer, **save_params)
            buffer.seek(0)
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error converting image to bytes: {e}")
            raise
