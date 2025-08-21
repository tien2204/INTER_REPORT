"""
Image Processor

General-purpose image processing utilities for the banner generation system.
Handles format conversion, optimization, metadata extraction, and basic transformations.
"""

import base64
import io
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ExifTags
import numpy as np
from datetime import datetime
from structlog import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """
    Comprehensive image processing utilities
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Quality settings
        self.jpeg_quality = config.get("jpeg_quality", 90)
        self.png_compression = config.get("png_compression", 6)
        self.webp_quality = config.get("webp_quality", 85)
        
        # Size limits
        self.max_file_size_mb = config.get("max_file_size_mb", 10)
        self.max_dimensions = config.get("max_dimensions", (4096, 4096))
        self.min_dimensions = config.get("min_dimensions", (32, 32))
        
        # Supported formats
        self.input_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}
        self.output_formats = {'.jpg', '.jpeg', '.png', '.webp'}
        
        # Processing options
        self.auto_orient = config.get("auto_orient", True)
        self.strip_metadata = config.get("strip_metadata", True)
        self.progressive_jpeg = config.get("progressive_jpeg", True)
    
    async def process_upload(self, 
                           file_data: Union[str, bytes],
                           filename: str = "image",
                           target_format: str = "png",
                           max_size: Optional[Tuple[int, int]] = None,
                           quality_preset: str = "high") -> Dict[str, Any]:
        """
        Process uploaded image file
        
        Args:
            file_data: Base64 string or raw bytes
            filename: Original filename
            target_format: Target output format
            max_size: Maximum dimensions (width, height)
            quality_preset: Quality preset (high, medium, low, web)
        
        Returns:
            Processing result with image data and metadata
        """
        try:
            logger.info(f"Processing image upload: {filename}")
            
            # Decode image data
            image, original_format = self._decode_image_data(file_data)
            
            # Extract metadata before processing
            metadata = await self._extract_metadata(image, filename)
            
            # Validate image
            validation_result = await self._validate_image(image, filename)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "metadata": metadata
                }
            
            # Apply auto-orientation
            if self.auto_orient:
                image = self._auto_orient_image(image)
            
            # Resize if needed
            if max_size:
                image = await self._smart_resize(image, max_size)
            
            # Apply quality preset
            image = await self._apply_quality_preset(image, quality_preset)
            
            # Convert to target format
            processed_data = await self._convert_format(image, target_format, quality_preset)
            
            # Generate final metadata
            final_metadata = await self._generate_final_metadata(
                image, processed_data, original_format, target_format, metadata
            )
            
            return {
                "success": True,
                "image_data": processed_data,
                "metadata": final_metadata,
                "original_metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing image upload: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {}
            }
    
    async def optimize_for_web(self, 
                             image_data: str,
                             target_size_kb: int = 200,
                             maintain_quality: bool = True) -> Dict[str, Any]:
        """
        Optimize image for web delivery
        
        Args:
            image_data: Base64 image data
            target_size_kb: Target file size in KB
            maintain_quality: Whether to maintain visual quality
        
        Returns:
            Optimized image with metadata
        """
        try:
            image = self._decode_base64_image(image_data)
            original_size = len(base64.b64decode(image_data.split(',')[-1])) / 1024
            
            logger.info(f"Optimizing image: {original_size:.1f}KB -> target {target_size_kb}KB")
            
            # Try different optimization strategies
            strategies = [
                self._optimize_quality,
                self._optimize_progressive,
                self._optimize_dimensions,
                self._optimize_format_conversion
            ]
            
            best_result = None
            best_size = float('inf')
            
            for strategy in strategies:
                try:
                    result = await strategy(image, target_size_kb, maintain_quality)
                    result_size = len(base64.b64decode(result["data"].split(',')[-1])) / 1024
                    
                    if result_size <= target_size_kb and result_size < best_size:
                        best_result = result
                        best_size = result_size
                        
                        # If we're close to target, use this result
                        if result_size >= target_size_kb * 0.8:
                            break
                            
                except Exception as e:
                    logger.warning(f"Optimization strategy failed: {e}")
                    continue
            
            if best_result:
                best_result["optimization"] = {
                    "original_size_kb": original_size,
                    "final_size_kb": best_size,
                    "compression_ratio": original_size / best_size,
                    "size_reduction": (original_size - best_size) / original_size * 100
                }
                return best_result
            else:
                # Fallback: aggressive compression
                return await self._aggressive_optimization(image, target_size_kb)
                
        except Exception as e:
            logger.error(f"Error optimizing image for web: {e}")
            return {"success": False, "error": str(e)}
    
    async def extract_color_palette(self, 
                                  image_data: str,
                                  num_colors: int = 5,
                                  algorithm: str = "kmeans") -> List[Dict[str, Any]]:
        """
        Extract dominant color palette from image
        
        Args:
            image_data: Base64 image data
            num_colors: Number of colors to extract
            algorithm: Algorithm to use (kmeans, quantize)
        
        Returns:
            List of color information
        """
        try:
            image = self._decode_base64_image(image_data)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if algorithm == "kmeans":
                colors = await self._extract_colors_kmeans(image, num_colors)
            elif algorithm == "quantize":
                colors = await self._extract_colors_quantize(image, num_colors)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Convert to color information
            color_palette = []
            for i, color in enumerate(colors):
                rgb = tuple(int(c) for c in color)
                hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
                
                color_info = {
                    "index": i,
                    "rgb": rgb,
                    "hex": hex_color,
                    "hsl": self._rgb_to_hsl(rgb),
                    "name": self._get_color_name(rgb),
                    "luminance": self._calculate_luminance(rgb),
                    "temperature": self._color_temperature(rgb)
                }
                color_palette.append(color_info)
            
            logger.info(f"Extracted {len(color_palette)} colors from image")
            return color_palette
            
        except Exception as e:
            logger.error(f"Error extracting color palette: {e}")
            return []
    
    async def analyze_composition(self, image_data: str) -> Dict[str, Any]:
        """
        Analyze image composition for design purposes
        
        Args:
            image_data: Base64 image data
        
        Returns:
            Composition analysis results
        """
        try:
            image = self._decode_base64_image(image_data)
            
            # Convert to arrays for analysis
            rgb_array = np.array(image)
            gray_array = np.array(image.convert('L'))
            
            analysis = {
                "dimensions": {
                    "width": image.width,
                    "height": image.height,
                    "aspect_ratio": image.width / image.height
                },
                "color_distribution": await self._analyze_color_distribution(rgb_array),
                "contrast": await self._analyze_contrast(gray_array),
                "focus_areas": await self._detect_focus_areas(gray_array),
                "symmetry": await self._analyze_symmetry(gray_array),
                "texture": await self._analyze_texture(gray_array),
                "rule_of_thirds": await self._analyze_rule_of_thirds(gray_array),
                "visual_weight": await self._analyze_visual_weight(gray_array)
            }
            
            # Generate composition score
            analysis["composition_score"] = self._calculate_composition_score(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing composition: {e}")
            return {"error": str(e)}
    
    def _decode_image_data(self, file_data: Union[str, bytes]) -> Tuple[Image.Image, str]:
        """Decode image from various input formats"""
        try:
            if isinstance(file_data, str):
                # Base64 string
                if file_data.startswith('data:image'):
                    # Data URL format
                    header, data = file_data.split(',', 1)
                    format_info = header.split(';')[0].split('/')[1]
                    image_bytes = base64.b64decode(data)
                else:
                    # Plain base64
                    image_bytes = base64.b64decode(file_data)
                    format_info = "unknown"
            else:
                # Raw bytes
                image_bytes = file_data
                format_info = "unknown"
            
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            original_format = image.format or format_info
            
            return image, original_format.lower()
            
        except Exception as e:
            logger.error(f"Error decoding image data: {e}")
            raise
    
    def _decode_base64_image(self, image_data: str) -> Image.Image:
        """Decode base64 image string"""
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))
            
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise
    
    async def _extract_metadata(self, image: Image.Image, filename: str) -> Dict[str, Any]:
        """Extract comprehensive image metadata"""
        try:
            metadata = {
                "filename": filename,
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "has_transparency": image.mode in ('RGBA', 'LA', 'P'),
                "extracted_at": datetime.now().isoformat()
            }
            
            # EXIF data
            if hasattr(image, '_getexif') and image._getexif():
                exif_data = {}
                for tag_id, value in image._getexif().items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)  # Convert to string for JSON serialization
                metadata["exif"] = exif_data
            
            # Color space information
            if hasattr(image, 'info'):
                metadata["info"] = {k: str(v) for k, v in image.info.items()}
            
            # Calculate file hash
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes.seek(0)
            metadata["content_hash"] = hashlib.md5(image_bytes.getvalue()).hexdigest()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {"error": str(e)}
    
    async def _validate_image(self, image: Image.Image, filename: str) -> Dict[str, Any]:
        """Validate image against constraints"""
        try:
            # Check dimensions
            width, height = image.size
            
            if width < self.min_dimensions[0] or height < self.min_dimensions[1]:
                return {
                    "valid": False,
                    "error": f"Image too small: {width}x{height} < {self.min_dimensions}"
                }
            
            if width > self.max_dimensions[0] or height > self.max_dimensions[1]:
                return {
                    "valid": False,
                    "error": f"Image too large: {width}x{height} > {self.max_dimensions}"
                }
            
            # Check file size (rough estimate)
            estimated_size = width * height * 4 / (1024 * 1024)  # Rough RGBA estimate
            if estimated_size > self.max_file_size_mb:
                return {
                    "valid": False,
                    "error": f"Estimated file size too large: {estimated_size:.1f}MB > {self.max_file_size_mb}MB"
                }
            
            # Check for corruption
            try:
                image.verify()
            except Exception:
                return {
                    "valid": False,
                    "error": "Image appears to be corrupted"
                }
            
            return {"valid": True}
            
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return {"valid": False, "error": str(e)}
    
    def _auto_orient_image(self, image: Image.Image) -> Image.Image:
        """Auto-orient image based on EXIF data"""
        try:
            return ImageOps.exif_transpose(image)
        except Exception as e:
            logger.warning(f"Could not auto-orient image: {e}")
            return image
    
    async def _smart_resize(self, 
                          image: Image.Image, 
                          max_size: Tuple[int, int],
                          method: str = "lanczos") -> Image.Image:
        """Smart resize maintaining aspect ratio"""
        try:
            # Calculate new dimensions
            original_width, original_height = image.size
            max_width, max_height = max_size
            
            # Calculate scaling factor
            width_ratio = max_width / original_width
            height_ratio = max_height / original_height
            scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale
            
            if scale_factor < 1.0:
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                
                resample_method = getattr(Image, method.upper(), Image.LANCZOS)
                image = image.resize((new_width, new_height), resample_method)
                
                logger.info(f"Resized image: {original_width}x{original_height} -> {new_width}x{new_height}")
            
            return image
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image
    
    async def _apply_quality_preset(self, image: Image.Image, preset: str) -> Image.Image:
        """Apply quality enhancement preset"""
        try:
            if preset == "high":
                # High quality: slight sharpening, enhanced contrast
                image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.05)
                
            elif preset == "medium":
                # Medium quality: basic enhancement
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)
                
            elif preset == "low":
                # Low quality: minimal processing
                pass
                
            elif preset == "web":
                # Web optimized: slight noise reduction
                image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            logger.error(f"Error applying quality preset: {e}")
            return image
    
    async def _convert_format(self, 
                            image: Image.Image, 
                            target_format: str, 
                            quality_preset: str) -> str:
        """Convert image to target format"""
        try:
            buffer = io.BytesIO()
            
            # Handle transparency
            if target_format.lower() in ['jpg', 'jpeg'] and image.mode in ('RGBA', 'LA', 'P'):
                # Create white background for JPEG
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            
            # Save with appropriate parameters
            save_kwargs = {}
            
            if target_format.lower() in ['jpg', 'jpeg']:
                save_kwargs.update({
                    'format': 'JPEG',
                    'quality': self._get_jpeg_quality(quality_preset),
                    'optimize': True,
                    'progressive': self.progressive_jpeg
                })
            elif target_format.lower() == 'png':
                save_kwargs.update({
                    'format': 'PNG',
                    'optimize': True,
                    'compress_level': self.png_compression
                })
            elif target_format.lower() == 'webp':
                save_kwargs.update({
                    'format': 'WEBP',
                    'quality': self._get_webp_quality(quality_preset),
                    'optimize': True
                })
            
            image.save(buffer, **save_kwargs)
            buffer.seek(0)
            
            # Encode to base64
            image_data = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/{target_format.lower()};base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error converting to {target_format}: {e}")
            raise
    
    def _get_jpeg_quality(self, preset: str) -> int:
        """Get JPEG quality based on preset"""
        quality_map = {
            "high": 95,
            "medium": 85,
            "low": 70,
            "web": 80
        }
        return quality_map.get(preset, self.jpeg_quality)
    
    def _get_webp_quality(self, preset: str) -> int:
        """Get WebP quality based on preset"""
        quality_map = {
            "high": 90,
            "medium": 80,
            "low": 65,
            "web": 75
        }
        return quality_map.get(preset, self.webp_quality)
    
    async def _generate_final_metadata(self, 
                                     image: Image.Image,
                                     processed_data: str,
                                     original_format: str,
                                     target_format: str,
                                     original_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final processing metadata"""
        try:
            # Calculate processed file size
            data_part = processed_data.split(',')[1] if ',' in processed_data else processed_data
            processed_size = len(base64.b64decode(data_part))
            
            metadata = {
                "processing": {
                    "original_format": original_format,
                    "target_format": target_format,
                    "processed_at": datetime.now().isoformat(),
                    "processed_size_bytes": processed_size,
                    "processed_size_kb": round(processed_size / 1024, 2)
                },
                "final_dimensions": {
                    "width": image.width,
                    "height": image.height,
                    "aspect_ratio": round(image.width / image.height, 3)
                },
                "optimization": {
                    "auto_oriented": self.auto_orient,
                    "metadata_stripped": self.strip_metadata,
                    "quality_enhanced": True
                }
            }
            
            # Merge with original metadata
            metadata.update(original_metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating final metadata: {e}")
            return original_metadata
    
    # Additional helper methods for optimization and analysis would go here
    # (Implementation of _optimize_quality, _extract_colors_kmeans, etc.)
    
    async def _optimize_quality(self, image: Image.Image, target_kb: int, maintain_quality: bool) -> Dict[str, Any]:
        """Quality-based optimization strategy"""
        # Implementation would vary quality settings
        return {"data": "", "method": "quality_optimization"}
    
    async def _extract_colors_kmeans(self, image: Image.Image, num_colors: int) -> List[Tuple[int, int, int]]:
        """Extract colors using k-means clustering"""
        try:
            # Convert image to array
            img_array = np.array(image)
            pixels = img_array.reshape(-1, 3)
            
            # Use k-means clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in colors]
            
        except ImportError:
            # Fallback without sklearn
            return await self._extract_colors_quantize(image, num_colors)
        except Exception as e:
            logger.error(f"Error in k-means color extraction: {e}")
            return []
    
    async def _extract_colors_quantize(self, image: Image.Image, num_colors: int) -> List[Tuple[int, int, int]]:
        """Extract colors using PIL quantization"""
        try:
            # Quantize image
            quantized = image.quantize(colors=num_colors)
            palette = quantized.getpalette()
            
            # Convert palette to RGB tuples
            colors = []
            for i in range(num_colors):
                r = palette[i * 3]
                g = palette[i * 3 + 1]
                b = palette[i * 3 + 2]
                colors.append((r, g, b))
            
            return colors
            
        except Exception as e:
            logger.error(f"Error in quantize color extraction: {e}")
            return []
    
    def _rgb_to_hsl(self, rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert RGB to HSL"""
        r, g, b = [x / 255.0 for x in rgb]
        
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Lightness
        l = (max_val + min_val) / 2
        
        if diff == 0:
            h = s = 0
        else:
            # Saturation
            s = diff / (2 - max_val - min_val) if l > 0.5 else diff / (max_val + min_val)
            
            # Hue
            if max_val == r:
                h = (g - b) / diff + (6 if g < b else 0)
            elif max_val == g:
                h = (b - r) / diff + 2
            else:
                h = (r - g) / diff + 4
            h /= 6
        
        return (int(h * 360), int(s * 100), int(l * 100))
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Get approximate color name"""
        # Simplified color naming
        r, g, b = rgb
        
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r > 150 and b > 150 and g < 100:
            return "magenta"
        elif g > 150 and b > 150 and r < 100:
            return "cyan"
        else:
            return "mixed"
    
    def _calculate_luminance(self, rgb: Tuple[int, int, int]) -> float:
        """Calculate relative luminance"""
        r, g, b = [x / 255.0 for x in rgb]
        
        # Apply gamma correction
        def gamma_correct(c):
            return c / 12.92 if c <= 0.03928 else pow((c + 0.055) / 1.055, 2.4)
        
        r = gamma_correct(r)
        g = gamma_correct(g)
        b = gamma_correct(b)
        
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    def _color_temperature(self, rgb: Tuple[int, int, int]) -> str:
        """Determine color temperature"""
        r, g, b = rgb
        
        if b > r and b > g:
            return "cool"
        elif r > g and r > b:
            return "warm"
        else:
            return "neutral"
    
    async def _analyze_color_distribution(self, rgb_array: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution in image"""
        # Implementation for color distribution analysis
        return {"distribution": "balanced"}  # Placeholder
    
    async def _analyze_contrast(self, gray_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image contrast"""
        contrast = np.std(gray_array)
        return {"contrast_value": float(contrast), "level": "high" if contrast > 50 else "low"}
    
    async def _detect_focus_areas(self, gray_array: np.ndarray) -> List[Dict[str, Any]]:
        """Detect areas of focus/interest"""
        # Simplified focus detection
        return [{"x": 0.5, "y": 0.5, "confidence": 0.8}]  # Placeholder
    
    async def _analyze_symmetry(self, gray_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image symmetry"""
        return {"vertical_symmetry": 0.5, "horizontal_symmetry": 0.5}  # Placeholder
    
    async def _analyze_texture(self, gray_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image texture"""
        variance = np.var(gray_array)
        return {"texture_variance": float(variance), "type": "smooth" if variance < 500 else "rough"}
    
    async def _analyze_rule_of_thirds(self, gray_array: np.ndarray) -> Dict[str, Any]:
        """Analyze rule of thirds composition"""
        return {"adherence_score": 0.7}  # Placeholder
    
    async def _analyze_visual_weight(self, gray_array: np.ndarray) -> Dict[str, Any]:
        """Analyze visual weight distribution"""
        return {"balance_score": 0.8}  # Placeholder
    
    def _calculate_composition_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall composition score"""
        # Weighted combination of various factors
        score = 0.7  # Placeholder
        return round(score, 2)
