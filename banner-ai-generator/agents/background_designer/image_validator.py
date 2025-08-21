"""
Image Validator for Background Designer

Validates generated background images for quality, composition,
and suitability for banner use.
"""

import base64
import io
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageStat
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


class ImageValidator:
    """
    Validates background images for quality and suitability
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Validation thresholds
        self.min_quality_score = config.get("min_quality_score", 0.7)
        self.min_contrast_ratio = config.get("min_contrast_ratio", 1.5)
        self.max_noise_level = config.get("max_noise_level", 0.3)
        self.min_sharpness = config.get("min_sharpness", 0.5)
    
    async def validate_background(self, image_data: str) -> Dict[str, Any]:
        """
        Comprehensive validation of background image
        
        Args:
            image_data: Base64 encoded image data
        
        Returns:
            Validation result with quality score and details
        """
        try:
            # Decode image
            image = self._decode_image(image_data)
            if image is None:
                return self._create_failure_result("Failed to decode image")
            
            # Run all validations
            validations = [
                self._validate_dimensions(image),
                self._validate_quality(image),
                self._validate_contrast(image),
                self._validate_composition(image),
                self._validate_color_distribution(image),
                self._validate_text_space(image),
                self._validate_noise_level(image),
                self._validate_sharpness(image)
            ]
            
            # Aggregate results
            total_score = 0
            max_score = 0
            issues = []
            details = {}
            
            for validation in validations:
                if validation:
                    score = validation.get("score", 0)
                    weight = validation.get("weight", 1)
                    total_score += score * weight
                    max_score += weight
                    
                    if validation.get("issues"):
                        issues.extend(validation["issues"])
                    
                    details[validation.get("category", "unknown")] = validation
            
            # Calculate final quality score
            quality_score = total_score / max_score if max_score > 0 else 0
            
            result = {
                "quality_score": quality_score,
                "passed": quality_score >= self.min_quality_score,
                "issues": issues,
                "details": details,
                "dimensions": {"width": image.width, "height": image.height},
                "summary": self._generate_summary(quality_score, issues)
            }
            
            logger.info(f"Image validation completed: score={quality_score:.2f}, passed={result['passed']}")
            return result
            
        except Exception as e:
            logger.error(f"Error validating background image: {e}")
            return self._create_failure_result(f"Validation error: {str(e)}")
    
    def _decode_image(self, image_data: str) -> Image.Image:
        """Decode base64 image data to PIL Image"""
        try:
            if image_data.startswith('data:image'):
                # Remove data URL prefix
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None
    
    def _validate_dimensions(self, image: Image.Image) -> Dict[str, Any]:
        """Validate image dimensions"""
        width, height = image.size
        issues = []
        
        # Check minimum dimensions
        min_width = self.config.get("min_width", 300)
        min_height = self.config.get("min_height", 150)
        
        if width < min_width:
            issues.append(f"Width too small: {width} < {min_width}")
        
        if height < min_height:
            issues.append(f"Height too small: {height} < {min_height}")
        
        # Check aspect ratio reasonableness
        ratio = width / height
        if ratio > 10 or ratio < 0.1:
            issues.append(f"Unusual aspect ratio: {ratio:.2f}")
        
        # Score based on size appropriateness
        score = min(1.0, (width * height) / (1024 * 1024))  # Score based on megapixels
        
        return {
            "category": "dimensions",
            "score": score,
            "weight": 1.0,
            "issues": issues,
            "details": {"width": width, "height": height, "ratio": ratio}
        }
    
    def _validate_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Validate overall image quality"""
        issues = []
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Check for artifacts (simplified)
        variance = np.var(img_array)
        if variance < 100:  # Too uniform
            issues.append("Image appears too uniform or flat")
        elif variance > 8000:  # Too noisy
            issues.append("Image appears too noisy or artifacts detected")
        
        # Check color depth
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
        color_richness = min(1.0, unique_colors / 10000)
        
        if color_richness < 0.3:
            issues.append("Limited color palette detected")
        
        # Overall quality score
        quality_factors = [
            min(1.0, variance / 4000),  # Variance score
            color_richness,  # Color richness score
            1.0 - len(issues) * 0.2  # Penalty for issues
        ]
        
        score = np.mean(quality_factors)
        
        return {
            "category": "quality",
            "score": max(0, score),
            "weight": 2.0,  # Higher weight for quality
            "issues": issues,
            "details": {
                "variance": variance,
                "unique_colors": unique_colors,
                "color_richness": color_richness
            }
        }
    
    def _validate_contrast(self, image: Image.Image) -> Dict[str, Any]:
        """Validate image contrast"""
        issues = []
        
        # Convert to grayscale for contrast analysis
        gray = image.convert('L')
        stat = ImageStat.Stat(gray)
        
        # Calculate contrast metrics
        std_dev = stat.stddev[0]
        mean_val = stat.mean[0]
        
        # Normalize contrast score
        contrast_score = min(1.0, std_dev / 64)  # 64 is reasonable std dev for good contrast
        
        if contrast_score < self.min_contrast_ratio / 2:
            issues.append(f"Low contrast detected: {contrast_score:.2f}")
        
        # Check for extreme values
        if mean_val < 50:
            issues.append("Image too dark overall")
        elif mean_val > 200:
            issues.append("Image too bright overall")
        
        return {
            "category": "contrast",
            "score": contrast_score,
            "weight": 1.5,
            "issues": issues,
            "details": {
                "std_dev": std_dev,
                "mean_brightness": mean_val,
                "contrast_score": contrast_score
            }
        }
    
    def _validate_composition(self, image: Image.Image) -> Dict[str, Any]:
        """Validate composition suitability for banner"""
        issues = []
        
        width, height = image.size
        img_array = np.array(image)
        
        # Check for busy areas in center (where text usually goes)
        center_x = width // 2
        center_y = height // 2
        center_region = img_array[
            max(0, center_y - height//4):min(height, center_y + height//4),
            max(0, center_x - width//4):min(width, center_x + width//4)
        ]
        
        center_variance = np.var(center_region)
        if center_variance > 5000:
            issues.append("Center region too busy for text overlay")
        
        # Check edge definition (important for banner boundaries)
        edge_score = self._calculate_edge_score(img_array)
        
        # Composition score
        composition_factors = [
            1.0 - min(1.0, center_variance / 5000),  # Prefer calm center
            edge_score,  # Good edge definition
            1.0 - len(issues) * 0.3  # Penalty for issues
        ]
        
        score = np.mean(composition_factors)
        
        return {
            "category": "composition",
            "score": max(0, score),
            "weight": 1.5,
            "issues": issues,
            "details": {
                "center_variance": center_variance,
                "edge_score": edge_score
            }
        }
    
    def _validate_color_distribution(self, image: Image.Image) -> Dict[str, Any]:
        """Validate color distribution"""
        issues = []
        
        # Analyze color distribution
        stat = ImageStat.Stat(image)
        
        # Check for color balance across RGB channels
        rgb_means = stat.mean
        rgb_stds = stat.stddev
        
        # Color balance score
        mean_diff = max(rgb_means) - min(rgb_means)
        if mean_diff > 100:
            issues.append("Significant color cast detected")
        
        # Color variety score
        color_variety = np.mean(rgb_stds) / 128  # Normalize to 0-1
        if color_variety < 0.2:
            issues.append("Limited color variety")
        
        score = max(0, 1.0 - mean_diff / 128) * min(1.0, color_variety)
        
        return {
            "category": "color_distribution",
            "score": score,
            "weight": 1.0,
            "issues": issues,
            "details": {
                "rgb_means": rgb_means,
                "rgb_stds": rgb_stds,
                "color_variety": color_variety
            }
        }
    
    def _validate_text_space(self, image: Image.Image) -> Dict[str, Any]:
        """Validate suitability for text overlay"""
        issues = []
        
        # Convert to grayscale for text space analysis
        gray = image.convert('L')
        img_array = np.array(gray)
        
        # Find regions suitable for text (relatively uniform areas)
        suitable_areas = []
        
        # Divide image into grid and analyze each section
        grid_size = 8
        h, w = img_array.shape
        step_h, step_w = h // grid_size, w // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                region = img_array[
                    i*step_h:(i+1)*step_h,
                    j*step_w:(j+1)*step_w
                ]
                
                region_std = np.std(region)
                region_mean = np.mean(region)
                
                # Good text area: low variance, not too dark or bright
                if region_std < 30 and 50 < region_mean < 200:
                    suitable_areas.append((i, j, region_std, region_mean))
        
        text_space_score = min(1.0, len(suitable_areas) / 8)  # At least 8 good regions
        
        if text_space_score < 0.3:
            issues.append("Limited suitable space for text overlay")
        
        return {
            "category": "text_space",
            "score": text_space_score,
            "weight": 2.0,  # Very important for banners
            "issues": issues,
            "details": {
                "suitable_regions": len(suitable_areas),
                "text_space_score": text_space_score
            }
        }
    
    def _validate_noise_level(self, image: Image.Image) -> Dict[str, Any]:
        """Validate noise level"""
        issues = []
        
        # Simple noise detection using local variance
        img_array = np.array(image.convert('L'))
        
        # Calculate local variance
        from scipy import ndimage
        local_variance = ndimage.generic_filter(img_array.astype(float), np.var, size=3)
        noise_level = np.mean(local_variance) / 255  # Normalize
        
        if noise_level > self.max_noise_level:
            issues.append(f"High noise level detected: {noise_level:.3f}")
        
        score = max(0, 1.0 - noise_level / self.max_noise_level)
        
        return {
            "category": "noise",
            "score": score,
            "weight": 1.0,
            "issues": issues,
            "details": {"noise_level": noise_level}
        }
    
    def _validate_sharpness(self, image: Image.Image) -> Dict[str, Any]:
        """Validate image sharpness"""
        issues = []
        
        # Simple sharpness detection using Laplacian variance
        img_array = np.array(image.convert('L'))
        
        # Calculate Laplacian variance
        from scipy import ndimage
        laplacian = ndimage.laplace(img_array.astype(float))
        sharpness = np.var(laplacian) / 10000  # Normalize
        
        if sharpness < self.min_sharpness:
            issues.append(f"Low sharpness detected: {sharpness:.3f}")
        
        score = min(1.0, sharpness / self.min_sharpness)
        
        return {
            "category": "sharpness",
            "score": score,
            "weight": 1.0,
            "issues": issues,
            "details": {"sharpness": sharpness}
        }
    
    def _calculate_edge_score(self, img_array: np.ndarray) -> float:
        """Calculate edge definition score"""
        try:
            from scipy import ndimage
            
            # Edge detection using Sobel
            gray = np.mean(img_array, axis=2)
            edges = ndimage.sobel(gray)
            edge_strength = np.mean(np.abs(edges))
            
            return min(1.0, edge_strength / 100)  # Normalize
            
        except ImportError:
            # Fallback without scipy
            return 0.5  # Neutral score
    
    def _create_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create failure result"""
        return {
            "quality_score": 0.0,
            "passed": False,
            "issues": [error_message],
            "details": {},
            "summary": f"Validation failed: {error_message}"
        }
    
    def _generate_summary(self, quality_score: float, issues: List[str]) -> str:
        """Generate human-readable summary"""
        if quality_score >= 0.9:
            quality_desc = "Excellent"
        elif quality_score >= 0.8:
            quality_desc = "Very Good"
        elif quality_score >= 0.7:
            quality_desc = "Good"
        elif quality_score >= 0.5:
            quality_desc = "Fair"
        else:
            quality_desc = "Poor"
        
        summary = f"{quality_desc} quality (score: {quality_score:.2f})"
        
        if issues:
            summary += f". Issues: {len(issues)} found"
        
        return summary
