"""
Logo Processor

Specialized processor for logo files with features like transparent background
detection, padding removal, color extraction, and brand consistency analysis.
"""

import base64
import io
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageFilter, ImageOps, ImageChops
from structlog import get_logger

logger = get_logger(__name__)


class LogoProcessor:
    """
    Specialized processor for logo files and brand assets
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Logo-specific settings
        self.preferred_formats = config.get("preferred_formats", ["png", "svg"])
        self.max_logo_size = config.get("max_logo_size", (500, 500))
        self.min_logo_size = config.get("min_logo_size", (32, 32))
        
        # Processing options
        self.auto_remove_padding = config.get("auto_remove_padding", True)
        self.auto_background_removal = config.get("auto_background_removal", False)
        self.preserve_transparency = config.get("preserve_transparency", True)
        
        # Quality thresholds
        self.min_contrast_ratio = config.get("min_contrast_ratio", 3.0)
        self.transparency_threshold = config.get("transparency_threshold", 240)
    
    async def process_logo(self, 
                          image_data: str,
                          filename: str = "logo",
                          target_sizes: List[Tuple[int, int]] = None,
                          background_color: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """
        Process logo with specialized optimizations
        
        Args:
            image_data: Base64 encoded logo data
            filename: Original filename
            target_sizes: List of target sizes to generate
            background_color: Background color for contrast analysis
        
        Returns:
            Processed logo data and analysis
        """
        try:
            logger.info(f"Processing logo: {filename}")
            
            # Decode image
            image = self._decode_image(image_data)
            
            # Analyze logo properties
            analysis = await self._analyze_logo_properties(image)
            
            # Remove padding if enabled
            if self.auto_remove_padding:
                image = await self._remove_transparent_padding(image)
                analysis["padding_removed"] = True
            
            # Background processing
            if self.auto_background_removal and not analysis["has_transparency"]:
                image = await self._attempt_background_removal(image)
                analysis["background_processed"] = True
            
            # Generate multiple sizes if requested
            variants = {}
            if target_sizes:
                for width, height in target_sizes:
                    variant = await self._create_logo_variant(image, width, height)
                    variants[f"{width}x{height}"] = self._image_to_base64(variant)
            
            # Extract brand colors
            brand_colors = await self._extract_brand_colors(image)
            
            # Validate logo quality
            quality_check = await self._validate_logo_quality(image, background_color)
            
            # Generate optimized formats
            optimized_formats = await self._generate_optimized_formats(image)
            
            result = {
                "success": True,
                "original_logo": self._image_to_base64(image),
                "variants": variants,
                "analysis": analysis,
                "brand_colors": brand_colors,
                "quality_check": quality_check,
                "optimized_formats": optimized_formats,
                "recommendations": await self._generate_recommendations(analysis, quality_check)
            }
            
            logger.info(f"Logo processing completed: {len(variants)} variants, {len(brand_colors)} colors")
            return result
            
        except Exception as e:
            logger.error(f"Error processing logo: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def remove_background(self, 
                              image_data: str,
                              method: str = "auto",
                              tolerance: int = 30) -> Dict[str, Any]:
        """
        Remove background from logo
        
        Args:
            image_data: Base64 encoded image
            method: Background removal method (auto, color_threshold, edge_detection)
            tolerance: Color tolerance for background removal
        
        Returns:
            Logo with background removed
        """
        try:
            image = self._decode_image(image_data)
            
            if method == "auto":
                # Try multiple methods and pick the best
                methods = ["color_threshold", "edge_detection", "flood_fill"]
                best_result = None
                best_score = 0
                
                for m in methods:
                    try:
                        result = await self._remove_background_method(image, m, tolerance)
                        score = await self._evaluate_background_removal(result, image)
                        
                        if score > best_score:
                            best_result = result
                            best_score = score
                    except Exception as e:
                        logger.warning(f"Background removal method {m} failed: {e}")
                        continue
                
                if best_result is not None:
                    return {
                        "success": True,
                        "image_data": self._image_to_base64(best_result),
                        "method_used": "auto",
                        "quality_score": best_score
                    }
                else:
                    raise RuntimeError("All background removal methods failed")
            
            else:
                # Use specific method
                result = await self._remove_background_method(image, method, tolerance)
                score = await self._evaluate_background_removal(result, image)
                
                return {
                    "success": True,
                    "image_data": self._image_to_base64(result),
                    "method_used": method,
                    "quality_score": score
                }
                
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def extract_logo_variations(self, 
                                    image_data: str,
                                    variation_types: List[str] = None) -> Dict[str, str]:
        """
        Extract different logo variations
        
        Args:
            image_data: Base64 encoded logo
            variation_types: Types of variations to create
        
        Returns:
            Dictionary of logo variations
        """
        try:
            image = self._decode_image(image_data)
            
            if variation_types is None:
                variation_types = ["monochrome", "white", "black", "inverted"]
            
            variations = {}
            
            for variation_type in variation_types:
                try:
                    if variation_type == "monochrome":
                        variant = await self._create_monochrome_logo(image)
                    elif variation_type == "white":
                        variant = await self._create_white_logo(image)
                    elif variation_type == "black":
                        variant = await self._create_black_logo(image)
                    elif variation_type == "inverted":
                        variant = await self._create_inverted_logo(image)
                    elif variation_type == "outline":
                        variant = await self._create_outline_logo(image)
                    else:
                        logger.warning(f"Unknown variation type: {variation_type}")
                        continue
                    
                    variations[variation_type] = self._image_to_base64(variant)
                    
                except Exception as e:
                    logger.error(f"Error creating {variation_type} variation: {e}")
                    continue
            
            return variations
            
        except Exception as e:
            logger.error(f"Error extracting logo variations: {e}")
            return {}
    
    async def _analyze_logo_properties(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze logo properties and characteristics"""
        try:
            width, height = image.size
            
            analysis = {
                "dimensions": {"width": width, "height": height},
                "aspect_ratio": round(width / height, 3),
                "has_transparency": image.mode in ('RGBA', 'LA', 'P'),
                "format": image.format,
                "mode": image.mode,
                "is_square": abs(width - height) / max(width, height) < 0.1,
                "is_horizontal": width > height * 1.2,
                "is_vertical": height > width * 1.2
            }
            
            # Analyze transparency
            if analysis["has_transparency"]:
                transparency_analysis = await self._analyze_transparency(image)
                analysis.update(transparency_analysis)
            
            # Analyze complexity
            complexity = await self._analyze_logo_complexity(image)
            analysis["complexity"] = complexity
            
            # Detect logo type
            logo_type = await self._detect_logo_type(image)
            analysis["logo_type"] = logo_type
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing logo properties: {e}")
            return {"error": str(e)}
    
    async def _remove_transparent_padding(self, image: Image.Image) -> Image.Image:
        """Remove transparent padding around logo"""
        try:
            if image.mode != 'RGBA':
                # Convert to RGBA if not already
                image = image.convert('RGBA')
            
            # Get alpha channel
            alpha = image.split()[-1]
            
            # Find bounding box of non-transparent pixels
            bbox = alpha.getbbox()
            
            if bbox:
                # Crop to remove padding
                cropped = image.crop(bbox)
                logger.info(f"Removed padding: {image.size} -> {cropped.size}")
                return cropped
            else:
                logger.warning("No non-transparent pixels found")
                return image
                
        except Exception as e:
            logger.error(f"Error removing transparent padding: {e}")
            return image
    
    async def _attempt_background_removal(self, image: Image.Image) -> Image.Image:
        """Attempt automatic background removal"""
        try:
            # Simple background removal based on corner color detection
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Sample corner pixels to detect background color
            corners = [
                image.getpixel((0, 0)),
                image.getpixel((image.width - 1, 0)),
                image.getpixel((0, image.height - 1)),
                image.getpixel((image.width - 1, image.height - 1))
            ]
            
            # Find most common corner color
            from collections import Counter
            corner_colors = [c[:3] for c in corners]  # Ignore alpha
            background_color = Counter(corner_colors).most_common(1)[0][0]
            
            # Remove background color
            data = np.array(image)
            
            # Create mask for background color (with tolerance)
            tolerance = 30
            mask = np.all(np.abs(data[:, :, :3] - background_color) <= tolerance, axis=2)
            
            # Set background pixels to transparent
            data[mask] = [0, 0, 0, 0]
            
            return Image.fromarray(data, 'RGBA')
            
        except Exception as e:
            logger.error(f"Error in automatic background removal: {e}")
            return image
    
    async def _create_logo_variant(self, 
                                 image: Image.Image, 
                                 width: int, 
                                 height: int) -> Image.Image:
        """Create logo variant with specific dimensions"""
        try:
            # Calculate aspect-ratio preserving resize
            original_width, original_height = image.size
            original_ratio = original_width / original_height
            target_ratio = width / height
            
            # Determine scaling
            if original_ratio > target_ratio:
                # Logo is wider, scale by width
                new_width = width
                new_height = int(width / original_ratio)
            else:
                # Logo is taller, scale by height
                new_height = height
                new_width = int(height * original_ratio)
            
            # Resize with high-quality resampling
            resized = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Center on target canvas if needed
            if new_width != width or new_height != height:
                # Create transparent canvas
                canvas = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                
                # Calculate position to center the logo
                x = (width - new_width) // 2
                y = (height - new_height) // 2
                
                # Paste logo onto canvas
                canvas.paste(resized, (x, y), resized if resized.mode == 'RGBA' else None)
                return canvas
            else:
                return resized
                
        except Exception as e:
            logger.error(f"Error creating logo variant: {e}")
            return image
    
    async def _extract_brand_colors(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Extract brand colors from logo"""
        try:
            # Convert to RGB for color analysis
            if image.mode == 'RGBA':
                # Create white background for transparent areas
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                rgb_image = background
            else:
                rgb_image = image.convert('RGB')
            
            # Extract dominant colors
            colors = []
            
            # Method 1: Color quantization
            quantized = rgb_image.quantize(colors=8)
            palette = quantized.getpalette()
            
            # Get color frequencies
            color_counts = {}
            for pixel in quantized.getdata():
                color_counts[pixel] = color_counts.get(pixel, 0) + 1
            
            # Sort by frequency
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            
            for i, (color_index, count) in enumerate(sorted_colors[:5]):
                if count < 100:  # Skip colors with very low frequency
                    continue
                
                r = palette[color_index * 3]
                g = palette[color_index * 3 + 1]
                b = palette[color_index * 3 + 2]
                
                # Skip white and very light colors (likely background)
                if r > 240 and g > 240 and b > 240:
                    continue
                
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                
                color_info = {
                    "rgb": [r, g, b],
                    "hex": hex_color,
                    "frequency": count,
                    "percentage": round(count / sum(color_counts.values()) * 100, 1),
                    "is_primary": i == 0,
                    "color_name": self._get_color_name(r, g, b)
                }
                colors.append(color_info)
            
            return colors
            
        except Exception as e:
            logger.error(f"Error extracting brand colors: {e}")
            return []
    
    async def _validate_logo_quality(self, 
                                   image: Image.Image,
                                   background_color: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """Validate logo quality for various use cases"""
        try:
            quality_check = {
                "overall_score": 0,
                "issues": [],
                "recommendations": [],
                "contrast_ratios": {},
                "scalability": {}
            }
            
            # Check resolution
            width, height = image.size
            min_size = min(width, height)
            
            if min_size < self.min_logo_size[0]:
                quality_check["issues"].append(f"Logo too small: {min_size}px < {self.min_logo_size[0]}px")
                quality_check["recommendations"].append("Use higher resolution logo")
            
            # Check contrast
            if background_color:
                contrast_ratio = await self._calculate_contrast_ratio(image, background_color)
                quality_check["contrast_ratios"]["background"] = contrast_ratio
                
                if contrast_ratio < self.min_contrast_ratio:
                    quality_check["issues"].append(f"Low contrast ratio: {contrast_ratio:.1f}")
                    quality_check["recommendations"].append("Increase color contrast")
            
            # Check scalability
            scalability_score = await self._assess_scalability(image)
            quality_check["scalability"] = scalability_score
            
            # Calculate overall score
            score = 100
            score -= len(quality_check["issues"]) * 15
            score -= max(0, (self.min_contrast_ratio - quality_check["contrast_ratios"].get("background", 5)) * 10)
            score += scalability_score.get("score", 0) * 10
            
            quality_check["overall_score"] = max(0, min(100, score))
            
            return quality_check
            
        except Exception as e:
            logger.error(f"Error validating logo quality: {e}")
            return {"error": str(e)}
    
    async def _generate_optimized_formats(self, image: Image.Image) -> Dict[str, str]:
        """Generate optimized formats for different use cases"""
        try:
            formats = {}
            
            # PNG (high quality, with transparency)
            png_buffer = io.BytesIO()
            image.save(png_buffer, format='PNG', optimize=True)
            png_data = base64.b64encode(png_buffer.getvalue()).decode()
            formats["png"] = f"data:image/png;base64,{png_data}"
            
            # JPEG (smaller file size, no transparency)
            if image.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                jpeg_image = Image.new('RGB', image.size, (255, 255, 255))
                jpeg_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            else:
                jpeg_image = image.convert('RGB')
            
            jpeg_buffer = io.BytesIO()
            jpeg_image.save(jpeg_buffer, format='JPEG', quality=90, optimize=True)
            jpeg_data = base64.b64encode(jpeg_buffer.getvalue()).decode()
            formats["jpeg"] = f"data:image/jpeg;base64,{jpeg_data}"
            
            # WebP (modern format)
            try:
                webp_buffer = io.BytesIO()
                image.save(webp_buffer, format='WebP', quality=90, optimize=True)
                webp_data = base64.b64encode(webp_buffer.getvalue()).decode()
                formats["webp"] = f"data:image/webp;base64,{webp_data}"
            except Exception:
                # WebP might not be available
                pass
            
            return formats
            
        except Exception as e:
            logger.error(f"Error generating optimized formats: {e}")
            return {}
    
    def _decode_image(self, image_data: str) -> Image.Image:
        """Decode base64 image"""
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))
            
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            raise
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert image to base64 string"""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            image_data = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise
    
    # Additional helper methods for logo processing
    async def _analyze_transparency(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze transparency characteristics"""
        if image.mode not in ('RGBA', 'LA', 'P'):
            return {"transparency_percentage": 0}
        
        # Count transparent pixels
        if image.mode == 'RGBA':
            alpha = image.split()[-1]
            alpha_array = np.array(alpha)
            transparent_pixels = np.sum(alpha_array < self.transparency_threshold)
            total_pixels = alpha_array.size
        else:
            # Simplified for other modes
            transparent_pixels = 0
            total_pixels = image.width * image.height
        
        transparency_percentage = (transparent_pixels / total_pixels) * 100
        
        return {
            "transparency_percentage": round(transparency_percentage, 2),
            "has_partial_transparency": transparency_percentage > 0 and transparency_percentage < 100,
            "is_fully_opaque": transparency_percentage == 0
        }
    
    async def _analyze_logo_complexity(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze logo complexity for scalability assessment"""
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            # Edge detection for complexity
            edges = np.abs(np.gradient(gray_array.astype(float))).sum()
            edge_density = edges / (gray_array.shape[0] * gray_array.shape[1])
            
            # Color count
            if image.mode == 'P':
                unique_colors = len(image.getcolors() or [])
            else:
                unique_colors = len(image.convert('RGB').getcolors(maxcolors=256*256*256) or [])
            
            # Determine complexity level
            if edge_density < 10 and unique_colors < 5:
                complexity_level = "simple"
            elif edge_density < 30 and unique_colors < 15:
                complexity_level = "moderate"
            else:
                complexity_level = "complex"
            
            return {
                "edge_density": round(edge_density, 2),
                "unique_colors": unique_colors,
                "complexity_level": complexity_level,
                "scalability_score": 1.0 if complexity_level == "simple" else 0.7 if complexity_level == "moderate" else 0.4
            }
            
        except Exception as e:
            logger.error(f"Error analyzing logo complexity: {e}")
            return {"complexity_level": "unknown", "scalability_score": 0.5}
    
    async def _detect_logo_type(self, image: Image.Image) -> str:
        """Detect type of logo (text, symbol, combination)"""
        try:
            # Simple heuristics for logo type detection
            width, height = image.size
            aspect_ratio = width / height
            
            # Analyze for text-like characteristics
            if aspect_ratio > 2.5:
                return "wordmark"  # Wide logos are likely text-based
            elif aspect_ratio < 0.4:
                return "vertical_text"
            elif 0.8 <= aspect_ratio <= 1.2:
                return "symbol"  # Square-ish logos are likely symbols
            else:
                return "combination"  # Moderate aspect ratios suggest combination
                
        except Exception:
            return "unknown"
    
    def _get_color_name(self, r: int, g: int, b: int) -> str:
        """Get approximate color name"""
        # Simplified color naming
        if r > 200 and g < 100 and b < 100:
            return "red"
        elif r < 100 and g > 200 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 200:
            return "blue"
        elif r > 200 and g > 200 and b < 100:
            return "yellow"
        elif r > 150 and g < 100 and b > 150:
            return "purple"
        elif r > 200 and g > 100 and b < 100:
            return "orange"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > 200 and g > 200 and b > 200:
            return "white"
        else:
            return "mixed"
    
    # Placeholder methods for additional functionality
    async def _remove_background_method(self, image: Image.Image, method: str, tolerance: int) -> Image.Image:
        """Remove background using specific method"""
        # Implementation would vary based on method
        return image
    
    async def _evaluate_background_removal(self, result: Image.Image, original: Image.Image) -> float:
        """Evaluate quality of background removal"""
        # Implementation would analyze the result quality
        return 0.8
    
    async def _create_monochrome_logo(self, image: Image.Image) -> Image.Image:
        """Create monochrome version of logo"""
        return image.convert('L').convert('RGBA')
    
    async def _create_white_logo(self, image: Image.Image) -> Image.Image:
        """Create white version of logo"""
        # Implementation would create white silhouette
        return image
    
    async def _create_black_logo(self, image: Image.Image) -> Image.Image:
        """Create black version of logo"""
        # Implementation would create black silhouette
        return image
    
    async def _create_inverted_logo(self, image: Image.Image) -> Image.Image:
        """Create inverted version of logo"""
        return ImageOps.invert(image.convert('RGB')).convert('RGBA')
    
    async def _create_outline_logo(self, image: Image.Image) -> Image.Image:
        """Create outline version of logo"""
        # Implementation would create outline effect
        return image
    
    async def _calculate_contrast_ratio(self, image: Image.Image, background_color: Tuple[int, int, int]) -> float:
        """Calculate contrast ratio against background"""
        # Implementation would calculate WCAG contrast ratio
        return 4.5  # Placeholder
    
    async def _assess_scalability(self, image: Image.Image) -> Dict[str, Any]:
        """Assess how well logo scales"""
        # Implementation would test logo at different sizes
        return {"score": 0.8, "details": "Good scalability"}
    
    async def _generate_recommendations(self, analysis: Dict[str, Any], quality_check: Dict[str, Any]) -> List[str]:
        """Generate recommendations for logo improvement"""
        recommendations = []
        
        if quality_check.get("overall_score", 0) < 70:
            recommendations.append("Consider improving logo quality and resolution")
        
        if analysis.get("complexity", {}).get("complexity_level") == "complex":
            recommendations.append("Simplify logo design for better scalability")
        
        if not analysis.get("has_transparency"):
            recommendations.append("Add transparent background for versatile usage")
        
        return recommendations
