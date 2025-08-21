"""
Logo Processor

Processes and analyzes logo files for banner integration.
Handles logo optimization, color extraction, style analysis,
and transparent padding removal.
"""

import asyncio
import io
import base64
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageFilter
import numpy as np
from sklearn.cluster import KMeans
import cv2
from structlog import get_logger

logger = get_logger(__name__)


class LogoProcessor:
    """
    Logo processor for analyzing and optimizing logos for banner integration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_size = self.config.get("max_size", (300, 300))
        self.output_formats = self.config.get("output_formats", ["PNG", "SVG"])
        self.color_palette_size = self.config.get("color_palette_size", 5)
    
    async def process_logo(self, logo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process logo and extract comprehensive information
        
        Args:
            logo_data: Logo data containing file path, base64, or URL
            
        Returns:
            Processed logo information and analysis
        """
        try:
            # Load logo image
            logo_image = await self._load_logo_image(logo_data)
            
            # Basic analysis
            basic_info = await self._analyze_basic_properties(logo_image)
            
            # Remove transparent padding
            trimmed_logo = await self._remove_transparent_padding(logo_image)
            
            # Extract color palette
            color_palette = await self._extract_color_palette(trimmed_logo)
            
            # Analyze logo style
            style_analysis = await self._analyze_logo_style(trimmed_logo)
            
            # Generate optimized versions
            optimized_versions = await self._generate_optimized_versions(trimmed_logo)
            
            # Calculate optimal sizing
            sizing_recommendations = await self._calculate_sizing_recommendations(trimmed_logo)
            
            # Analyze placement suitability
            placement_analysis = await self._analyze_placement_suitability(trimmed_logo)
            
            processed_logo = {
                "original": {
                    "size": basic_info["size"],
                    "format": basic_info["format"],
                    "has_transparency": basic_info["has_transparency"],
                    "aspect_ratio": basic_info["aspect_ratio"]
                },
                "trimmed": {
                    "image_data": await self._image_to_base64(trimmed_logo),
                    "size": trimmed_logo.size,
                    "bounding_box": await self._get_content_bounding_box(logo_image)
                },
                "colors": color_palette,
                "style": style_analysis,
                "optimized_versions": optimized_versions,
                "sizing": sizing_recommendations,
                "placement": placement_analysis,
                "metadata": {
                    "processing_version": "1.0",
                    "quality_score": await self._calculate_quality_score(trimmed_logo),
                    "banner_suitability": await self._assess_banner_suitability(trimmed_logo)
                }
            }
            
            logger.info("Logo processing completed successfully")
            return processed_logo
            
        except Exception as e:
            logger.error(f"Logo processing failed: {e}")
            raise
    
    async def _load_logo_image(self, logo_data: Dict[str, Any]) -> Image.Image:
        """Load logo image from various sources"""
        if "base64" in logo_data:
            # Load from base64
            image_data = base64.b64decode(logo_data["base64"])
            return Image.open(io.BytesIO(image_data)).convert("RGBA")
        
        elif "file_path" in logo_data:
            # Load from file path
            return Image.open(logo_data["file_path"]).convert("RGBA")
        
        elif "url" in logo_data:
            # Load from URL (implement if needed)
            raise NotImplementedError("URL loading not implemented yet")
        
        else:
            raise ValueError("No valid logo source provided")
    
    async def _analyze_basic_properties(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze basic logo properties"""
        return {
            "size": image.size,
            "format": image.format or "PNG",
            "mode": image.mode,
            "has_transparency": image.mode in ("RGBA", "LA") or "transparency" in image.info,
            "aspect_ratio": image.size[0] / image.size[1],
            "is_square": abs(image.size[0] - image.size[1]) / max(image.size) < 0.1,
            "is_horizontal": image.size[0] > image.size[1] * 1.5,
            "is_vertical": image.size[1] > image.size[0] * 1.5
        }
    
    async def _remove_transparent_padding(self, image: Image.Image) -> Image.Image:
        """Remove transparent padding around logo"""
        if image.mode != "RGBA":
            return image
        
        # Get alpha channel
        alpha = image.split()[-1]
        
        # Find bounding box of non-transparent pixels
        bbox = alpha.getbbox()
        
        if bbox:
            # Crop to content
            trimmed = image.crop(bbox)
            
            # Add small padding (5% of smaller dimension)
            padding = min(trimmed.size) * 0.05
            padding = max(2, int(padding))  # Minimum 2px padding
            
            # Create new image with padding
            new_size = (trimmed.size[0] + 2 * padding, trimmed.size[1] + 2 * padding)
            padded_image = Image.new("RGBA", new_size, (0, 0, 0, 0))
            
            # Paste trimmed image in center
            paste_pos = (padding, padding)
            padded_image.paste(trimmed, paste_pos, trimmed)
            
            return padded_image
        
        return image
    
    async def _extract_color_palette(self, image: Image.Image) -> Dict[str, Any]:
        """Extract color palette from logo"""
        try:
            # Convert to RGB for color analysis
            rgb_image = image.convert("RGB")
            
            # Get image data as numpy array
            img_array = np.array(rgb_image)
            
            # Reshape for clustering
            pixels = img_array.reshape(-1, 3)
            
            # Remove similar colors (reduce noise)
            unique_pixels = np.unique(pixels, axis=0)
            
            # Use KMeans to find dominant colors
            n_colors = min(self.color_palette_size, len(unique_pixels))
            if n_colors > 1:
                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                kmeans.fit(unique_pixels)
                colors = kmeans.cluster_centers_.astype(int)
            else:
                colors = unique_pixels
            
            # Convert colors to hex and analyze
            color_palette = []
            for color in colors:
                hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
                color_info = {
                    "hex": hex_color,
                    "rgb": tuple(color),
                    "luminance": self._calculate_luminance(color),
                    "is_dark": self._calculate_luminance(color) < 0.5,
                    "saturation": self._calculate_saturation(color)
                }
                color_palette.append(color_info)
            
            # Sort by luminance (light to dark)
            color_palette.sort(key=lambda x: x["luminance"], reverse=True)
            
            # Determine color scheme
            color_scheme = self._determine_color_scheme(color_palette)
            
            return {
                "palette": color_palette,
                "primary_color": color_palette[0]["hex"] if color_palette else "#000000",
                "secondary_colors": [c["hex"] for c in color_palette[1:3]],
                "dominant_tone": "dark" if sum(c["is_dark"] for c in color_palette) > len(color_palette) / 2 else "light",
                "color_scheme": color_scheme,
                "color_count": len(color_palette),
                "has_vibrant_colors": any(c["saturation"] > 0.6 for c in color_palette)
            }
            
        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return {
                "palette": [],
                "primary_color": "#000000",
                "secondary_colors": [],
                "dominant_tone": "neutral",
                "color_scheme": "monochrome"
            }
    
    def _calculate_luminance(self, rgb: Tuple[int, int, int]) -> float:
        """Calculate relative luminance of RGB color"""
        r, g, b = [c / 255.0 for c in rgb]
        
        # Apply gamma correction
        def gamma_correct(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        
        r, g, b = map(gamma_correct, [r, g, b])
        
        # Calculate luminance
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    def _calculate_saturation(self, rgb: Tuple[int, int, int]) -> float:
        """Calculate saturation of RGB color"""
        r, g, b = [c / 255.0 for c in rgb]
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        
        if max_val == 0:
            return 0
        
        return (max_val - min_val) / max_val
    
    def _determine_color_scheme(self, palette: List[Dict[str, Any]]) -> str:
        """Determine color scheme type"""
        if len(palette) <= 1:
            return "monochrome"
        
        # Analyze color distribution
        saturations = [c["saturation"] for c in palette]
        avg_saturation = sum(saturations) / len(saturations)
        
        if avg_saturation < 0.2:
            return "monochrome"
        elif avg_saturation < 0.5:
            return "muted"
        else:
            return "vibrant"
    
    async def _analyze_logo_style(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze logo style characteristics"""
        try:
            # Convert to grayscale for analysis
            gray_image = image.convert("L")
            img_array = np.array(gray_image)
            
            # Analyze edges and complexity
            edges = cv2.Canny(img_array, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Analyze text vs symbol
            text_likelihood = await self._detect_text_content(image)
            
            # Analyze geometric patterns
            geometric_score = await self._analyze_geometric_patterns(img_array)
            
            # Determine style category
            style_category = self._classify_logo_style(edge_density, text_likelihood, geometric_score)
            
            return {
                "style_category": style_category,
                "complexity": "high" if edge_density > 0.1 else "medium" if edge_density > 0.05 else "low",
                "content_type": "text" if text_likelihood > 0.7 else "symbol" if text_likelihood < 0.3 else "mixed",
                "geometric_elements": geometric_score > 0.6,
                "edge_density": edge_density,
                "text_likelihood": text_likelihood,
                "geometric_score": geometric_score,
                "is_minimal": edge_density < 0.03 and geometric_score > 0.5,
                "is_detailed": edge_density > 0.15,
                "style_confidence": 0.8  # Simplified confidence score
            }
            
        except Exception as e:
            logger.error(f"Style analysis failed: {e}")
            return {
                "style_category": "unknown",
                "complexity": "medium",
                "content_type": "mixed"
            }
    
    async def _detect_text_content(self, image: Image.Image) -> float:
        """Detect likelihood of text content in logo"""
        # Simplified text detection based on aspect ratio and edge patterns
        width, height = image.size
        aspect_ratio = width / height
        
        # Text logos tend to be wider
        text_score = 0.0
        
        if aspect_ratio > 2.0:  # Very wide
            text_score += 0.4
        elif aspect_ratio > 1.5:  # Moderately wide
            text_score += 0.2
        
        # Additional heuristics could be added here
        # For now, return simplified score
        return min(text_score, 1.0)
    
    async def _analyze_geometric_patterns(self, img_array: np.ndarray) -> float:
        """Analyze geometric patterns in logo"""
        # Simplified geometric analysis
        # Look for regular patterns, symmetry, etc.
        
        height, width = img_array.shape
        
        # Check horizontal symmetry
        left_half = img_array[:, :width//2]
        right_half = np.fliplr(img_array[:, width//2:])
        
        if left_half.shape == right_half.shape:
            h_symmetry = np.mean(np.abs(left_half - right_half)) / 255.0
            h_symmetry = 1.0 - h_symmetry  # Convert to similarity score
        else:
            h_symmetry = 0.0
        
        # Check vertical symmetry
        top_half = img_array[:height//2, :]
        bottom_half = np.flipud(img_array[height//2:, :])
        
        if top_half.shape == bottom_half.shape:
            v_symmetry = np.mean(np.abs(top_half - bottom_half)) / 255.0
            v_symmetry = 1.0 - v_symmetry
        else:
            v_symmetry = 0.0
        
        # Combine symmetry scores
        geometric_score = (h_symmetry + v_symmetry) / 2.0
        
        return min(geometric_score, 1.0)
    
    def _classify_logo_style(self, edge_density: float, text_likelihood: float, 
                           geometric_score: float) -> str:
        """Classify logo style based on analysis"""
        if text_likelihood > 0.7:
            return "wordmark"
        elif geometric_score > 0.6 and edge_density < 0.05:
            return "geometric"
        elif edge_density > 0.15:
            return "detailed"
        elif edge_density < 0.03:
            return "minimal"
        else:
            return "modern"
    
    async def _generate_optimized_versions(self, image: Image.Image) -> Dict[str, Any]:
        """Generate optimized versions for different use cases"""
        versions = {}
        
        # High quality version
        versions["high_quality"] = {
            "size": image.size,
            "data": await self._image_to_base64(image),
            "use_case": "large_banners"
        }
        
        # Medium quality version
        medium_size = self._calculate_target_size(image.size, max_dimension=150)
        medium_image = image.resize(medium_size, Image.Resampling.LANCZOS)
        versions["medium_quality"] = {
            "size": medium_size,
            "data": await self._image_to_base64(medium_image),
            "use_case": "standard_banners"
        }
        
        # Small version
        small_size = self._calculate_target_size(image.size, max_dimension=100)
        small_image = image.resize(small_size, Image.Resampling.LANCZOS)
        versions["small"] = {
            "size": small_size,
            "data": await self._image_to_base64(small_image),
            "use_case": "compact_banners"
        }
        
        # Favicon version
        favicon_image = image.resize((32, 32), Image.Resampling.LANCZOS)
        versions["favicon"] = {
            "size": (32, 32),
            "data": await self._image_to_base64(favicon_image),
            "use_case": "small_icons"
        }
        
        return versions
    
    def _calculate_target_size(self, original_size: Tuple[int, int], 
                             max_dimension: int) -> Tuple[int, int]:
        """Calculate target size maintaining aspect ratio"""
        width, height = original_size
        
        if width > height:
            if width > max_dimension:
                ratio = max_dimension / width
                return (max_dimension, int(height * ratio))
        else:
            if height > max_dimension:
                ratio = max_dimension / height
                return (int(width * ratio), max_dimension)
        
        return original_size
    
    async def _calculate_sizing_recommendations(self, image: Image.Image) -> Dict[str, Any]:
        """Calculate sizing recommendations for banner integration"""
        width, height = image.size
        aspect_ratio = width / height
        
        # Banner size recommendations
        recommendations = {
            "leaderboard_728x90": self._calculate_logo_size_for_banner((728, 90), image.size),
            "banner_468x60": self._calculate_logo_size_for_banner((468, 60), image.size),
            "rectangle_300x250": self._calculate_logo_size_for_banner((300, 250), image.size),
            "square_250x250": self._calculate_logo_size_for_banner((250, 250), image.size),
            "skyscraper_160x600": self._calculate_logo_size_for_banner((160, 600), image.size)
        }
        
        # General guidelines
        guidelines = {
            "min_size_ratio": 0.15,  # Logo should be at least 15% of banner height
            "max_size_ratio": 0.4,   # Logo should not exceed 40% of banner height
            "preferred_placement": self._recommend_placement(aspect_ratio),
            "scaling_quality": "excellent" if min(image.size) > 100 else "good" if min(image.size) > 50 else "fair"
        }
        
        return {
            "banner_recommendations": recommendations,
            "guidelines": guidelines,
            "original_size": image.size,
            "aspect_ratio": aspect_ratio
        }
    
    def _calculate_logo_size_for_banner(self, banner_size: Tuple[int, int], 
                                      logo_size: Tuple[int, int]) -> Dict[str, Any]:
        """Calculate optimal logo size for specific banner size"""
        banner_width, banner_height = banner_size
        logo_width, logo_height = logo_size
        logo_aspect = logo_width / logo_height
        
        # Calculate size options
        height_based = int(banner_height * 0.25)  # 25% of banner height
        width_based = int(banner_width * 0.15)    # 15% of banner width
        
        # Choose based on logo aspect ratio
        if logo_aspect > 2.0:  # Wide logo
            recommended_width = min(width_based * 2, banner_width * 0.3)
            recommended_height = int(recommended_width / logo_aspect)
        else:  # Square or tall logo
            recommended_height = height_based
            recommended_width = int(recommended_height * logo_aspect)
        
        return {
            "recommended_size": (recommended_width, recommended_height),
            "scale_factor": recommended_height / logo_height,
            "banner_coverage": (recommended_width / banner_width) * 100,
            "placement_options": ["top-left", "top-right", "center-left"]
        }
    
    def _recommend_placement(self, aspect_ratio: float) -> List[str]:
        """Recommend placement based on logo aspect ratio"""
        if aspect_ratio > 2.0:  # Wide logo
            return ["top-center", "bottom-center", "top-left", "top-right"]
        elif aspect_ratio < 0.8:  # Tall logo
            return ["center-left", "center-right", "top-left", "bottom-left"]
        else:  # Square logo
            return ["top-left", "top-right", "center-left", "bottom-right"]
    
    async def _analyze_placement_suitability(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze logo suitability for different placements"""
        # Analyze logo for placement considerations
        width, height = image.size
        aspect_ratio = width / height
        
        # Convert to grayscale for analysis
        gray = image.convert("L")
        img_array = np.array(gray)
        
        # Analyze content distribution
        content_distribution = self._analyze_content_distribution(img_array)
        
        # Calculate readability on different backgrounds
        bg_compatibility = await self._analyze_background_compatibility(image)
        
        return {
            "aspect_ratio_category": self._categorize_aspect_ratio(aspect_ratio),
            "content_distribution": content_distribution,
            "background_compatibility": bg_compatibility,
            "optimal_placements": self._determine_optimal_placements(aspect_ratio, content_distribution),
            "size_flexibility": self._assess_size_flexibility(image),
            "contrast_requirements": self._analyze_contrast_requirements(image)
        }
    
    def _analyze_content_distribution(self, img_array: np.ndarray) -> Dict[str, float]:
        """Analyze how content is distributed in the logo"""
        height, width = img_array.shape
        
        # Divide into quadrants
        h_mid, w_mid = height // 2, width // 2
        
        quadrants = {
            "top_left": img_array[:h_mid, :w_mid],
            "top_right": img_array[:h_mid, w_mid:],
            "bottom_left": img_array[h_mid:, :w_mid],
            "bottom_right": img_array[h_mid:, w_mid:]
        }
        
        # Calculate content density in each quadrant
        distribution = {}
        for quad_name, quad_data in quadrants.items():
            # Use variance as a measure of content
            distribution[quad_name] = float(np.var(quad_data) / 255.0)
        
        return distribution
    
    async def _analyze_background_compatibility(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze compatibility with different background types"""
        # Extract colors for analysis
        colors = await self._extract_color_palette(image)
        
        # Test compatibility with common background types
        compatibility = {
            "white_background": self._test_bg_compatibility(colors, (255, 255, 255)),
            "black_background": self._test_bg_compatibility(colors, (0, 0, 0)),
            "light_gray": self._test_bg_compatibility(colors, (240, 240, 240)),
            "dark_gray": self._test_bg_compatibility(colors, (64, 64, 64)),
            "colored_backgrounds": self._assess_colored_bg_compatibility(colors)
        }
        
        return compatibility
    
    def _test_bg_compatibility(self, logo_colors: Dict[str, Any], 
                             bg_color: Tuple[int, int, int]) -> Dict[str, Any]:
        """Test logo compatibility with specific background color"""
        bg_luminance = self._calculate_luminance(bg_color)
        
        # Calculate contrast ratios
        contrasts = []
        for color_info in logo_colors["palette"]:
            logo_luminance = color_info["luminance"]
            
            # Calculate contrast ratio
            if bg_luminance > logo_luminance:
                contrast = (bg_luminance + 0.05) / (logo_luminance + 0.05)
            else:
                contrast = (logo_luminance + 0.05) / (bg_luminance + 0.05)
            
            contrasts.append(contrast)
        
        min_contrast = min(contrasts) if contrasts else 1.0
        avg_contrast = sum(contrasts) / len(contrasts) if contrasts else 1.0
        
        return {
            "min_contrast_ratio": min_contrast,
            "avg_contrast_ratio": avg_contrast,
            "readability": "excellent" if min_contrast > 7 else "good" if min_contrast > 4.5 else "poor",
            "wcag_aa_compliant": min_contrast >= 4.5,
            "wcag_aaa_compliant": min_contrast >= 7.0
        }
    
    def _assess_colored_bg_compatibility(self, logo_colors: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compatibility with colored backgrounds"""
        return {
            "works_with_brand_colors": True,  # Simplified
            "requires_outline": logo_colors.get("dominant_tone") == "light",
            "drop_shadow_recommended": logo_colors.get("dominant_tone") == "light",
            "background_blur_recommended": False
        }
    
    def _categorize_aspect_ratio(self, ratio: float) -> str:
        """Categorize logo aspect ratio"""
        if ratio > 3.0:
            return "very_wide"
        elif ratio > 2.0:
            return "wide"
        elif ratio > 1.3:
            return "moderately_wide"
        elif ratio > 0.8:
            return "square"
        elif ratio > 0.5:
            return "tall"
        else:
            return "very_tall"
    
    def _determine_optimal_placements(self, aspect_ratio: float, 
                                    content_distribution: Dict[str, float]) -> List[str]:
        """Determine optimal placement positions"""
        placements = []
        
        # Based on aspect ratio
        if aspect_ratio > 2.0:
            placements.extend(["top-center", "bottom-center"])
        elif aspect_ratio < 0.7:
            placements.extend(["center-left", "center-right"])
        else:
            placements.extend(["top-left", "top-right"])
        
        # Based on content distribution
        max_content_quad = max(content_distribution.items(), key=lambda x: x[1])
        if max_content_quad[0] == "top_left":
            placements.append("top-left")
        elif max_content_quad[0] == "top_right":
            placements.append("top-right")
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(placements))
    
    def _assess_size_flexibility(self, image: Image.Image) -> Dict[str, Any]:
        """Assess how well the logo scales to different sizes"""
        width, height = image.size
        min_dimension = min(width, height)
        
        return {
            "scales_well_small": min_dimension > 100,
            "scales_well_large": True,  # Most logos scale up well
            "min_recommended_size": max(32, min_dimension // 4),
            "max_recommended_size": min_dimension * 3,
            "optimal_size_range": (min_dimension // 2, min_dimension * 2)
        }
    
    def _analyze_contrast_requirements(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze contrast requirements for the logo"""
        # Convert to grayscale
        gray = image.convert("L")
        img_array = np.array(gray)
        
        # Calculate contrast statistics
        std_dev = np.std(img_array)
        contrast_level = std_dev / 128.0  # Normalize to 0-2 range
        
        return {
            "internal_contrast": contrast_level,
            "needs_background_contrast": contrast_level < 0.3,
            "works_on_busy_backgrounds": contrast_level > 0.8,
            "outline_recommended": contrast_level < 0.4,
            "drop_shadow_recommended": contrast_level < 0.5
        }
    
    async def _get_content_bounding_box(self, image: Image.Image) -> Tuple[int, int, int, int]:
        """Get bounding box of actual content (non-transparent area)"""
        if image.mode == "RGBA":
            alpha = image.split()[-1]
            bbox = alpha.getbbox()
            return bbox if bbox else (0, 0, image.size[0], image.size[1])
        else:
            return (0, 0, image.size[0], image.size[1])
    
    async def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    async def _calculate_quality_score(self, image: Image.Image) -> float:
        """Calculate overall quality score for the logo"""
        width, height = image.size
        min_size = min(width, height)
        
        # Size score (prefer larger logos)
        size_score = min(1.0, min_size / 200.0)
        
        # Aspect ratio score (prefer balanced ratios)
        aspect_ratio = width / height
        if 0.5 <= aspect_ratio <= 2.0:
            aspect_score = 1.0
        else:
            aspect_score = 0.7
        
        # Transparency score
        transparency_score = 1.0 if image.mode == "RGBA" else 0.8
        
        # Combine scores
        overall_score = (size_score * 0.4 + aspect_score * 0.3 + transparency_score * 0.3)
        
        return round(overall_score, 2)
    
    async def _assess_banner_suitability(self, image: Image.Image) -> str:
        """Assess overall suitability for banner usage"""
        quality_score = await self._calculate_quality_score(image)
        
        if quality_score >= 0.8:
            return "excellent"
        elif quality_score >= 0.6:
            return "good"
        elif quality_score >= 0.4:
            return "fair"
        else:
            return "poor"
