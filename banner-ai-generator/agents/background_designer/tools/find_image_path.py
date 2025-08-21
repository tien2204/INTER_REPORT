"""
Find Image Path Tool

Tool for finding existing suitable images based on search criteria.
Implements intelligent search through asset libraries and databases.
"""

import os
import asyncio
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
from structlog import get_logger

logger = get_logger(__name__)


class FindImagePathTool:
    """
    Tool for finding existing images that match criteria
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configure search paths
        self.asset_directories = config.get("asset_directories", [
            "assets/backgrounds",
            "assets/stock_images", 
            "assets/generated"
        ])
        
        # Image metadata cache
        self.metadata_cache = {}
        self.cache_file = config.get("cache_file", "assets/image_metadata_cache.json")
        
        # Supported formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        # Load existing cache
        asyncio.create_task(self._load_metadata_cache())
    
    async def find_images(self, search_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find images matching search criteria
        
        Args:
            search_criteria: Dictionary with search parameters:
                - industry: Target industry
                - mood: Desired mood/emotion
                - dimensions: Target dimensions (optional)
                - colors: Preferred colors (optional)
                - style: Visual style (optional)
                - max_results: Maximum number of results (default: 10)
        
        Returns:
            List of matching image records
        """
        try:
            max_results = search_criteria.get("max_results", 10)
            
            # Get all available images
            all_images = await self._scan_asset_directories()
            
            # Filter and score images
            scored_images = []
            for image_path in all_images:
                score = await self._calculate_match_score(image_path, search_criteria)
                if score > 0.3:  # Minimum relevance threshold
                    image_data = await self._get_image_data(image_path, score)
                    if image_data:
                        scored_images.append(image_data)
            
            # Sort by score and return top results
            scored_images.sort(key=lambda x: x["match_score"], reverse=True)
            results = scored_images[:max_results]
            
            logger.info(f"Found {len(results)} images matching criteria (from {len(all_images)} total)")
            return results
            
        except Exception as e:
            logger.error(f"Error finding images: {e}")
            return []
    
    async def _scan_asset_directories(self) -> List[Path]:
        """Scan asset directories for image files"""
        try:
            image_paths = []
            
            for directory in self.asset_directories:
                dir_path = Path(directory)
                if dir_path.exists() and dir_path.is_dir():
                    # Recursively find image files
                    for file_path in dir_path.rglob("*"):
                        if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                            image_paths.append(file_path)
            
            logger.debug(f"Scanned {len(self.asset_directories)} directories, found {len(image_paths)} images")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error scanning asset directories: {e}")
            return []
    
    async def _calculate_match_score(self, image_path: Path, criteria: Dict[str, Any]) -> float:
        """Calculate how well an image matches the search criteria"""
        try:
            score = 0.0
            max_score = 0.0
            
            # Get or create image metadata
            metadata = await self._get_image_metadata(image_path)
            
            # Industry matching
            if criteria.get("industry"):
                max_score += 0.3
                industry_score = self._match_industry(metadata, criteria["industry"])
                score += industry_score * 0.3
            
            # Mood matching
            if criteria.get("mood"):
                max_score += 0.25
                mood_score = self._match_mood(metadata, criteria["mood"])
                score += mood_score * 0.25
            
            # Dimension compatibility
            if criteria.get("dimensions"):
                max_score += 0.2
                dimension_score = self._match_dimensions(metadata, criteria["dimensions"])
                score += dimension_score * 0.2
            
            # Color compatibility
            if criteria.get("colors"):
                max_score += 0.15
                color_score = self._match_colors(metadata, criteria["colors"])
                score += color_score * 0.15
            
            # Style matching
            if criteria.get("style"):
                max_score += 0.1
                style_score = self._match_style(metadata, criteria["style"])
                score += style_score * 0.1
            
            # Normalize score
            final_score = score / max_score if max_score > 0 else 0
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating match score for {image_path}: {e}")
            return 0.0
    
    async def _get_image_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Get or generate metadata for an image"""
        try:
            # Check cache first
            cache_key = str(image_path)
            if cache_key in self.metadata_cache:
                cached_data = self.metadata_cache[cache_key]
                
                # Check if file has been modified since cache
                if cached_data.get("modified_time") == os.path.getmtime(image_path):
                    return cached_data["metadata"]
            
            # Generate new metadata
            metadata = await self._analyze_image(image_path)
            
            # Cache the metadata
            self.metadata_cache[cache_key] = {
                "metadata": metadata,
                "modified_time": os.path.getmtime(image_path)
            }
            
            # Save cache periodically
            await self._save_metadata_cache()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata for {image_path}: {e}")
            return {}
    
    async def _analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """Analyze image to extract metadata"""
        try:
            from PIL import Image
            import numpy as np
            
            metadata = {
                "filename": image_path.name,
                "path": str(image_path),
                "size": os.path.getsize(image_path),
                "format": image_path.suffix.lower(),
                "analyzed_at": asyncio.get_event_loop().time()
            }
            
            # Open and analyze image
            with Image.open(image_path) as img:
                metadata["dimensions"] = {
                    "width": img.width,
                    "height": img.height,
                    "aspect_ratio": img.width / img.height
                }
                
                # Convert to RGB for analysis
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Extract color information
                img_array = np.array(img)
                metadata["colors"] = self._extract_color_info(img_array)
                
                # Analyze brightness and contrast
                gray = np.mean(img_array, axis=2)
                metadata["brightness"] = float(np.mean(gray))
                metadata["contrast"] = float(np.std(gray))
                
                # Simple texture analysis
                metadata["texture"] = self._analyze_texture(gray)
            
            # Extract metadata from filename/path
            metadata.update(self._extract_path_metadata(image_path))
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {"error": str(e)}
    
    def _extract_color_info(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Extract color information from image"""
        try:
            # Calculate average colors
            avg_colors = np.mean(img_array, axis=(0, 1))
            
            # Determine dominant colors (simplified)
            reshaped = img_array.reshape(-1, 3)
            unique_colors, counts = np.unique(reshaped, axis=0, return_counts=True)
            
            # Get top 5 most common colors
            top_indices = np.argsort(counts)[-5:]
            dominant_colors = unique_colors[top_indices].tolist()
            
            # Determine overall color temperature
            r, g, b = avg_colors
            if b > r and b > g:
                temperature = "cool"
            elif r > g and r > b:
                temperature = "warm"
            else:
                temperature = "neutral"
            
            return {
                "average_rgb": avg_colors.tolist(),
                "dominant_colors": dominant_colors,
                "temperature": temperature,
                "saturation": float(np.std(img_array))
            }
            
        except Exception as e:
            logger.error(f"Error extracting color info: {e}")
            return {}
    
    def _analyze_texture(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Analyze image texture"""
        try:
            # Simple texture measures
            variance = float(np.var(gray_image))
            
            # Edge density (simplified)
            edges = np.abs(np.diff(gray_image, axis=0)).sum() + np.abs(np.diff(gray_image, axis=1)).sum()
            edge_density = edges / (gray_image.shape[0] * gray_image.shape[1])
            
            # Classify texture
            if variance < 500:
                texture_type = "smooth"
            elif variance > 2000:
                texture_type = "rough"
            else:
                texture_type = "moderate"
            
            return {
                "variance": variance,
                "edge_density": float(edge_density),
                "type": texture_type
            }
            
        except Exception as e:
            logger.error(f"Error analyzing texture: {e}")
            return {}
    
    def _extract_path_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract metadata from file path and name"""
        try:
            metadata = {}
            
            # Extract from filename
            filename = image_path.stem.lower()
            
            # Industry keywords
            industry_keywords = {
                "tech": "technology", "corp": "corporate", "biz": "business",
                "health": "healthcare", "med": "healthcare", "fin": "finance",
                "edu": "education", "food": "food", "travel": "travel",
                "auto": "automotive", "fashion": "fashion", "real": "real_estate"
            }
            
            for keyword, industry in industry_keywords.items():
                if keyword in filename:
                    metadata["suggested_industry"] = industry
                    break
            
            # Mood keywords
            mood_keywords = {
                "professional": "professional", "clean": "professional",
                "bright": "energetic", "dark": "elegant", "warm": "friendly",
                "cool": "professional", "vibrant": "energetic", "calm": "calm"
            }
            
            for keyword, mood in mood_keywords.items():
                if keyword in filename:
                    metadata["suggested_mood"] = mood
                    break
            
            # Style keywords
            style_keywords = {
                "abstract": "abstract", "geometric": "geometric",
                "gradient": "gradient", "pattern": "pattern",
                "minimal": "minimal", "modern": "modern"
            }
            
            for keyword, style in style_keywords.items():
                if keyword in filename:
                    metadata["suggested_style"] = style
                    break
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting path metadata: {e}")
            return {}
    
    def _match_industry(self, metadata: Dict[str, Any], target_industry: str) -> float:
        """Calculate industry match score"""
        suggested_industry = metadata.get("suggested_industry", "")
        if suggested_industry == target_industry:
            return 1.0
        elif suggested_industry and target_industry in suggested_industry:
            return 0.7
        elif suggested_industry:
            return 0.3
        else:
            return 0.5  # Neutral score for unknown
    
    def _match_mood(self, metadata: Dict[str, Any], target_mood: str) -> float:
        """Calculate mood match score"""
        suggested_mood = metadata.get("suggested_mood", "")
        brightness = metadata.get("brightness", 128)
        contrast = metadata.get("contrast", 64)
        
        # Base score from suggested mood
        mood_score = 0.5
        if suggested_mood == target_mood:
            mood_score = 1.0
        elif suggested_mood:
            mood_score = 0.3
        
        # Adjust based on image characteristics
        if target_mood == "energetic":
            if brightness > 150 and contrast > 70:
                mood_score += 0.3
        elif target_mood == "calm":
            if brightness < 180 and contrast < 50:
                mood_score += 0.3
        elif target_mood == "professional":
            if 100 < brightness < 180 and 30 < contrast < 80:
                mood_score += 0.3
        
        return min(1.0, mood_score)
    
    def _match_dimensions(self, metadata: Dict[str, Any], target_dimensions: Dict[str, Any]) -> float:
        """Calculate dimension compatibility score"""
        try:
            img_dims = metadata.get("dimensions", {})
            if not img_dims:
                return 0.5
            
            img_width = img_dims.get("width", 0)
            img_height = img_dims.get("height", 0)
            target_width = target_dimensions.get("width", 0)
            target_height = target_dimensions.get("height", 0)
            
            if not all([img_width, img_height, target_width, target_height]):
                return 0.5
            
            # Calculate aspect ratio compatibility
            img_ratio = img_width / img_height
            target_ratio = target_width / target_height
            
            ratio_diff = abs(img_ratio - target_ratio) / target_ratio
            ratio_score = max(0, 1 - ratio_diff)
            
            # Resolution score (prefer higher resolution)
            img_pixels = img_width * img_height
            target_pixels = target_width * target_height
            
            if img_pixels >= target_pixels:
                resolution_score = 1.0
            else:
                resolution_score = img_pixels / target_pixels
            
            # Combined score
            return (ratio_score * 0.7 + resolution_score * 0.3)
            
        except (ZeroDivisionError, TypeError):
            return 0.5
    
    def _match_colors(self, metadata: Dict[str, Any], target_colors: List[str]) -> float:
        """Calculate color compatibility score"""
        try:
            if not target_colors:
                return 1.0
            
            color_info = metadata.get("colors", {})
            if not color_info:
                return 0.5
            
            # Simple color matching based on temperature and dominant colors
            temperature = color_info.get("temperature", "neutral")
            
            # Convert target colors to temperature
            target_temp = self._colors_to_temperature(target_colors)
            
            if temperature == target_temp:
                return 1.0
            elif temperature == "neutral" or target_temp == "neutral":
                return 0.7
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"Error matching colors: {e}")
            return 0.5
    
    def _match_style(self, metadata: Dict[str, Any], target_style: str) -> float:
        """Calculate style match score"""
        suggested_style = metadata.get("suggested_style", "")
        texture_info = metadata.get("texture", {})
        
        style_score = 0.5
        if suggested_style == target_style:
            style_score = 1.0
        elif suggested_style:
            style_score = 0.3
        
        # Adjust based on texture for certain styles
        if target_style == "minimal" and texture_info.get("type") == "smooth":
            style_score += 0.3
        elif target_style == "modern" and texture_info.get("variance", 0) > 1000:
            style_score += 0.2
        
        return min(1.0, style_score)
    
    def _colors_to_temperature(self, colors: List[str]) -> str:
        """Convert color list to temperature"""
        warm_colors = {"red", "orange", "yellow", "pink"}
        cool_colors = {"blue", "green", "purple", "cyan"}
        
        warm_count = sum(1 for color in colors if any(w in color.lower() for w in warm_colors))
        cool_count = sum(1 for color in colors if any(c in color.lower() for c in cool_colors))
        
        if warm_count > cool_count:
            return "warm"
        elif cool_count > warm_count:
            return "cool"
        else:
            return "neutral"
    
    async def _get_image_data(self, image_path: Path, match_score: float) -> Optional[Dict[str, Any]]:
        """Get image data for result"""
        try:
            # Read image as base64
            with open(image_path, "rb") as f:
                import base64
                image_data = base64.b64encode(f.read()).decode()
            
            metadata = await self._get_image_metadata(image_path)
            
            return {
                "path": str(image_path),
                "filename": image_path.name,
                "match_score": match_score,
                "data": f"data:image/{image_path.suffix.lstrip('.')};base64,{image_data}",
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting image data for {image_path}: {e}")
            return None
    
    async def _load_metadata_cache(self):
        """Load metadata cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.metadata_cache = json.load(f)
                logger.debug(f"Loaded metadata cache with {len(self.metadata_cache)} entries")
        except Exception as e:
            logger.error(f"Error loading metadata cache: {e}")
    
    async def _save_metadata_cache(self):
        """Save metadata cache to file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.metadata_cache, f)
            logger.debug(f"Saved metadata cache with {len(self.metadata_cache)} entries")
        except Exception as e:
            logger.error(f"Error saving metadata cache: {e}")
