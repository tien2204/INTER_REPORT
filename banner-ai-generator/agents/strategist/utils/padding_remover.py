"""
Specialized padding removal utilities for logo processing
"""

import logging
from typing import Tuple, Optional, List
from PIL import Image
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PaddingInfo:
    """Information about detected padding"""
    left: int = 0
    top: int = 0  
    right: int = 0
    bottom: int = 0
    padding_color: Optional[Tuple[int, ...]] = None
    confidence: float = 0.0
    
    @property
    def has_padding(self) -> bool:
        """Check if any padding was detected"""
        return any([self.left, self.top, self.right, self.bottom])
    
    @property
    def total_padding(self) -> int:
        """Total padding pixels"""
        return self.left + self.top + self.right + self.bottom

class PaddingRemover:
    """
    Advanced padding removal for logos and images
    Handles transparent padding, solid color padding, and complex backgrounds
    """
    
    def __init__(self, tolerance: int = 10, min_content_ratio: float = 0.1):
        self.tolerance = tolerance  # Color matching tolerance
        self.min_content_ratio = min_content_ratio  # Minimum content to preserve
    
    def remove_padding(self, image: Image.Image, auto_detect: bool = True) -> Tuple[Image.Image, PaddingInfo]:
        """
        Remove padding from image with automatic detection
        
        Args:
            image: PIL Image to process
            auto_detect: Whether to auto-detect padding type
            
        Returns:
            Tuple of (processed_image, padding_info)
        """
        try:
            padding_info = PaddingInfo()
            
            if auto_detect:
                padding_info = self.detect_padding(image)
            
            if not padding_info.has_padding:
                return image, padding_info
            
            # Apply padding removal
            processed_image = self._crop_padding(image, padding_info)
            
            # Validate result
            if self._validate_crop_result(image, processed_image):
                return processed_image, padding_info
            else:
                logger.warning("Padding removal validation failed, returning original")
                return image, PaddingInfo()
                
        except Exception as e:
            logger.error(f"Padding removal failed: {e}")
            return image, PaddingInfo()
    
    def detect_padding(self, image: Image.Image) -> PaddingInfo:
        """
        Detect padding in image using multiple methods
        """
        padding_methods = [
            self._detect_transparent_padding,
            self._detect_solid_color_padding,
            self._detect_edge_similarity_padding
        ]
        
        best_padding = PaddingInfo()
        best_confidence = 0.0
        
        for method in padding_methods:
            try:
                padding_info = method(image)
                if padding_info.confidence > best_confidence:
                    best_padding = padding_info
                    best_confidence = padding_info.confidence
            except Exception as e:
                logger.debug(f"Padding detection method failed: {e}")
                continue
        
        return best_padding
    
    def _detect_transparent_padding(self, image: Image.Image) -> PaddingInfo:
        """Detect transparent padding (alpha channel based)"""
        if image.mode not in ('RGBA', 'LA'):
            return PaddingInfo()
        
        try:
            # Get alpha channel
            if image.mode == 'RGBA':
                alpha = image.split()[3]
            else:  # LA mode
                alpha = image.split()[1]
            
            # Convert to numpy for easier processing
            alpha_array = np.array(alpha)
            
            # Find non-transparent pixels
            non_transparent = alpha_array > self.tolerance
            
            if not np.any(non_transparent):
                # Image is completely transparent
                return PaddingInfo()
            
            # Find bounding box of non-transparent content
            rows = np.any(non_transparent, axis=1)
            cols = np.any(non_transparent, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                return PaddingInfo()
            
            top = np.argmax(rows)
            bottom = len(rows) - 1 - np.argmax(rows[::-1])
            left = np.argmax(cols)
            right = len(cols) - 1 - np.argmax(cols[::-1])
            
            # Calculate padding
            height, width = alpha_array.shape
            padding = PaddingInfo(
                left=left,
                top=top,
                right=width - 1 - right,
                bottom=height - 1 - bottom,
                padding_color=(0, 0, 0, 0),  # Transparent
                confidence=0.9  # High confidence for transparent padding
            )
            
            # Validate content size
            content_width = right - left + 1
            content_height = bottom - top + 1
            content_ratio = (content_width * content_height) / (width * height)
            
            if content_ratio < self.min_content_ratio:
                padding.confidence = 0.0  # Low confidence if content too small
            
            return padding
            
        except Exception as e:
            logger.error(f"Transparent padding detection failed: {e}")
            return PaddingInfo()
    
    def _detect_solid_color_padding(self, image: Image.Image) -> PaddingInfo:
        """Detect solid color padding by analyzing edges"""
        try:
            # Convert to RGB for consistency
            rgb_image = image.convert('RGB')
            width, height = rgb_image.size
            
            # Sample edge pixels
            edge_samples = self._sample_edge_pixels(rgb_image)
            
            # Find most common edge color
            padding_color = self._find_dominant_edge_color(edge_samples)
            
            if not padding_color:
                return PaddingInfo()
            
            # Find content boundaries
            boundaries = self._find_content_boundaries(rgb_image, padding_color)
            
            if not boundaries:
                return PaddingInfo()
            
            left, top, right, bottom = boundaries
            
            # Calculate padding
            padding = PaddingInfo(
                left=left,
                top=top,
                right=width - 1 - right,
                bottom=height - 1 - bottom,
                padding_color=padding_color,
                confidence=self._calculate_solid_padding_confidence(rgb_image, padding_color, boundaries)
            )
            
            return padding
            
        except Exception as e:
            logger.error(f"Solid color padding detection failed: {e}")
            return PaddingInfo()
    
    def _detect_edge_similarity_padding(self, image: Image.Image) -> PaddingInfo:
        """Detect padding based on edge pixel similarity"""
        try:
            rgb_image = image.convert('RGB')
            width, height = rgb_image.size
            
            # Analyze edge strips
            edge_width = max(1, min(width, height) // 20)  # 5% of smaller dimension
            
            # Get edge strips
            top_strip = rgb_image.crop((0, 0, width, edge_width))
            bottom_strip = rgb_image.crop((0, height - edge_width, width, height))
            left_strip = rgb_image.crop((0, 0, edge_width, height))
            right_strip = rgb_image.crop((width - edge_width, 0, width, height))
            
            # Calculate uniformity of each strip
            uniformity_scores = {
                'top': self._calculate_strip_uniformity(top_strip),
                'bottom': self._calculate_strip_uniformity(bottom_strip),
                'left': self._calculate_strip_uniformity(left_strip),
                'right': self._calculate_strip_uniformity(right_strip)
            }
            
            # Find padding based on uniformity
            padding_detected = {}
            for direction, score in uniformity_scores.items():
                padding_detected[direction] = score > 0.8  # High uniformity threshold
            
            if not any(padding_detected.values()):
                return PaddingInfo()
            
            # Calculate padding amounts
            padding_amounts = {}
            for direction in ['top', 'bottom', 'left', 'right']:
                if padding_detected[direction]:
                    padding_amounts[direction] = self._calculate_uniform_padding_depth(
                        rgb_image, direction, edge_width
                    )
                else:
                    padding_amounts[direction] = 0
            
            # Create padding info
            padding = PaddingInfo(
                left=padding_amounts.get('left', 0),
                top=padding_amounts.get('top', 0),
                right=padding_amounts.get('right', 0),
                bottom=padding_amounts.get('bottom', 0),
                confidence=max(uniformity_scores.values()) * 0.7  # Lower confidence than other methods
            )
            
            return padding
            
        except Exception as e:
            logger.error(f"Edge similarity padding detection failed: {e}")
            return PaddingInfo()
    
    def _sample_edge_pixels(self, image: Image.Image, sample_size: int = 50) -> List[Tuple[int, int, int]]:
        """Sample pixels from image edges"""
        width, height = image.size
        samples = []
        
        # Sample from corners and edges
        corner_size = min(10, width // 4, height // 4)
        
        # Corners
        for x in range(corner_size):
            for y in range(corner_size):
                # Top-left
                samples.append(image.getpixel((x, y)))
                # Top-right
                samples.append(image.getpixel((width - 1 - x, y)))
                # Bottom-left  
                samples.append(image.getpixel((x, height - 1 - y)))
                # Bottom-right
                samples.append(image.getpixel((width - 1 - x, height - 1 - y)))
        
        # Edge samples
        step = max(1, max(width, height) // sample_size)
        
        # Top and bottom edges
        for x in range(0, width, step):
            samples.append(image.getpixel((x, 0)))
            samples.append(image.getpixel((x, height - 1)))
        
        # Left and right edges
        for y in range(0, height, step):
            samples.append(image.getpixel((0, y)))
            samples.append(image.getpixel((width - 1, y)))
        
        return samples
    
    def _find_dominant_edge_color(self, edge_samples: List[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
        """Find the most common color in edge samples"""
        if not edge_samples:
            return None
        
        # Count color occurrences with tolerance
        color_groups = {}
        
        for sample in edge_samples:
            # Find existing group within tolerance
            matched_group = None
            for group_color in color_groups.keys():
                if self._colors_similar(sample, group_color, self.tolerance):
                    matched_group = group_color
                    break
            
            if matched_group:
                color_groups[matched_group] += 1
            else:
                color_groups[sample] = 1
        
        if not color_groups:
            return None
        
        # Find most frequent color
        dominant_color = max(color_groups, key=color_groups.get)
        
        # Check if it's significantly more common than others
        total_samples = len(edge_samples)
        dominance_ratio = color_groups[dominant_color] / total_samples
        
        if dominance_ratio < 0.3:  # Less than 30% dominance
            return None
        
        return dominant_color
    
    def _colors_similar(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int], tolerance: int) -> bool:
        """Check if two colors are similar within tolerance"""
        return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color1, color2))
    
    def _find_content_boundaries(self, image: Image.Image, padding_color: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Find content boundaries given padding color"""
        width, height = image.size
        
        # Convert to numpy for efficient processing
        img_array = np.array(image)
        
        # Create mask for non-padding pixels
        padding_mask = np.all(
            np.abs(img_array - np.array(padding_color)) <= self.tolerance,
            axis=2
        )
        content_mask = ~padding_mask
        
        if not np.any(content_mask):
            return None
        
        # Find bounding box
        rows = np.any(content_mask, axis=1)
        cols = np.any(content_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        top = np.argmax(rows)
        bottom = len(rows) - 1 - np.argmax(rows[::-1])
        left = np.argmax(cols)
        right = len(cols) - 1 - np.argmax(cols[::-1])
        
        return (left, top, right, bottom)
    
    def _calculate_solid_padding_confidence(self, image: Image.Image, 
                                          padding_color: Tuple[int, int, int], 
                                          boundaries: Tuple[int, int, int, int]) -> float:
        """Calculate confidence score for solid color padding detection"""
        try:
            left, top, right, bottom = boundaries
            width, height = image.size
            
            # Calculate padding area ratio
            total_area = width * height
            content_area = (right - left + 1) * (bottom - top + 1)
            padding_ratio = 1.0 - (content_area / total_area)
            
            # Calculate color consistency in padding areas
            consistency_score = self._calculate_padding_color_consistency(
                image, padding_color, boundaries
            )
            
            # Combine factors
            confidence = consistency_score * min(1.0, padding_ratio * 2)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def _calculate_strip_uniformity(self, strip: Image.Image) -> float:
        """Calculate uniformity score for an image strip"""
        try:
            # Convert to numpy
            strip_array = np.array(strip)
            
            if strip_array.size == 0:
                return 0.0
            
            # Calculate standard deviation for each channel
            std_devs = np.std(strip_array.reshape(-1, strip_array.shape[-1]), axis=0)
            
            # Average std dev across channels
            avg_std = np.mean(std_devs)
            
            # Convert to uniformity score (lower std = higher uniformity)
            uniformity = max(0.0, 1.0 - (avg_std / 128.0))  # Normalize by half of 255
            
            return uniformity
            
        except Exception as e:
            logger.error(f"Strip uniformity calculation failed: {e}")
            return 0.0
    
    def _calculate_uniform_padding_depth(self, image: Image.Image, direction: str, max_depth: int) -> int:
        """Calculate depth of uniform padding in given direction"""
        try:
            width, height = image.size
            
            if direction == 'top':
                for depth in range(1, min(max_depth, height)):
                    strip = image.crop((0, 0, width, depth))
                    if self._calculate_strip_uniformity(strip) < 0.8:
                        return depth - 1
                return max_depth
            
            elif direction == 'bottom':
                for depth in range(1, min(max_depth, height)):
                    strip = image.crop((0, height - depth, width, height))
                    if self._calculate_strip_uniformity(strip) < 0.8:
                        return depth - 1
                return max_depth
            
            elif direction == 'left':
                for depth in range(1, min(max_depth, width)):
                    strip = image.crop((0, 0, depth, height))
                    if self._calculate_strip_uniformity(strip) < 0.8:
                        return depth - 1
                return max_depth
            
            elif direction == 'right':
                for depth in range(1, min(max_depth, width)):
                    strip = image.crop((width - depth, 0, width, height))
                    if self._calculate_strip_uniformity(strip) < 0.8:
                        return depth - 1
                return max_depth
            
            return 0
            
        except Exception as e:
            logger.error(f"Padding depth calculation failed: {e}")
            return 0
    
    def _calculate_padding_color_consistency(self, image: Image.Image,
                                           padding_color: Tuple[int, int, int],
                                           boundaries: Tuple[int, int, int, int]) -> float:
        """Calculate how consistent the padding color is"""
        try:
            left, top, right, bottom = boundaries
            width, height = image.size
            
            # Sample padding areas
            padding_samples = []
            
            # Top padding
            if top > 0:
                top_strip = image.crop((0, 0, width, top))
                padding_samples.extend(self._sample_strip_colors(top_strip, 20))
            
            # Bottom padding
            if bottom < height - 1:
                bottom_strip = image.crop((0, bottom + 1, width, height))
                padding_samples.extend(self._sample_strip_colors(bottom_strip, 20))
            
            # Left padding
            if left > 0:
                left_strip = image.crop((0, 0, left, height))
                padding_samples.extend(self._sample_strip_colors(left_strip, 20))
            
            # Right padding
            if right < width - 1:
                right_strip = image.crop((right + 1, 0, width, height))
                padding_samples.extend(self._sample_strip_colors(right_strip, 20))
            
            if not padding_samples:
                return 0.0
            
            # Calculate consistency
            consistent_count = sum(
                1 for sample in padding_samples
                if self._colors_similar(sample, padding_color, self.tolerance)
            )
            
            consistency = consistent_count / len(padding_samples)
            return consistency
            
        except Exception as e:
            logger.error(f"Padding color consistency calculation failed: {e}")
            return 0.0
    
    def _sample_strip_colors(self, strip: Image.Image, num_samples: int) -> List[Tuple[int, int, int]]:
        """Sample colors from an image strip"""
        width, height = strip.size
        samples = []
        
        if width == 0 or height == 0:
            return samples
        
        # Sample regularly across the strip
        if width > height:  # Horizontal strip
            step = max(1, width // num_samples)
            y = height // 2
            for x in range(0, width, step):
                samples.append(strip.getpixel((x, y)))
        else:  # Vertical strip
            step = max(1, height // num_samples)
            x = width // 2
            for y in range(0, height, step):
                samples.append(strip.getpixel((x, y)))
        
        return samples
    
    def _crop_padding(self, image: Image.Image, padding_info: PaddingInfo) -> Image.Image:
        """Crop image based on detected padding"""
        try:
            width, height = image.size
            
            left = padding_info.left
            top = padding_info.top
            right = width - padding_info.right
            bottom = height - padding_info.bottom
            
            # Validate crop box
            if left >= right or top >= bottom:
                logger.warning("Invalid crop box, returning original")
                return image
            
            return image.crop((left, top, right, bottom))
            
        except Exception as e:
            logger.error(f"Padding crop failed: {e}")
            return image
    
    def _validate_crop_result(self, original: Image.Image, cropped: Image.Image) -> bool:
        """Validate that cropping result is reasonable"""
        try:
            orig_area = original.size[0] * original.size[1]
            crop_area = cropped.size[0] * cropped.size[1]
            
            # Check if content wasn't removed too aggressively
            content_ratio = crop_area / orig_area
            
            if content_ratio < self.min_content_ratio:
                logger.warning(f"Cropped area too small: {content_ratio:.2%}")
                return False
            
            # Check if dimensions are reasonable
            if cropped.size[0] < 10 or cropped.size[1] < 10:
                logger.warning("Cropped image dimensions too small")
                return False
            
            # Check aspect ratio change isn't too extreme
            orig_ratio = original.size[0] / original.size[1]
            crop_ratio = cropped.size[0] / cropped.size[1]
            ratio_change = abs(orig_ratio - crop_ratio) / orig_ratio
            
            if ratio_change > 2.0:  # More than 200% change
                logger.warning(f"Aspect ratio change too extreme: {ratio_change:.2%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Crop validation failed: {e}")
            return False
    
    def remove_specific_color_padding(self, image: Image.Image, 
                                    padding_color: Tuple[int, int, int],
                                    tolerance: Optional[int] = None) -> Tuple[Image.Image, PaddingInfo]:
        """Remove padding of a specific color"""
        try:
            if tolerance is None:
                tolerance = self.tolerance
            
            boundaries = self._find_content_boundaries(image, padding_color)
            
            if not boundaries:
                return image, PaddingInfo()
            
            left, top, right, bottom = boundaries
            width, height = image.size
            
            padding_info = PaddingInfo(
                left=left,
                top=top,
                right=width - 1 - right,
                bottom=height - 1 - bottom,
                padding_color=padding_color,
                confidence=0.8  # Manual specification gives high confidence
            )
            
            cropped = self._crop_padding(image, padding_info)
            
            if self._validate_crop_result(image, cropped):
                return cropped, padding_info
            else:
                return image, PaddingInfo()
                
        except Exception as e:
            logger.error(f"Specific color padding removal failed: {e}")
            return image, PaddingInfo()
    
    def smart_trim(self, image: Image.Image, aggressive: bool = False) -> Tuple[Image.Image, PaddingInfo]:
        """
        Smart trimming that combines multiple methods for best results
        
        Args:
            image: Input image
            aggressive: If True, uses more aggressive trimming
            
        Returns:
            Tuple of (trimmed_image, padding_info)
        """
        try:
            # Try multiple approaches and pick the best one
            results = []
            
            # Method 1: Auto-detect padding
            result1 = self.remove_padding(image, auto_detect=True)
            results.append(result1)
            
            # Method 2: Edge-based trimming (more conservative)
            if image.mode in ('RGBA', 'LA'):
                result2 = self._trim_by_alpha(image)
                results.append(result2)
            
            # Method 3: Content-based trimming (if aggressive)
            if aggressive:
                result3 = self._aggressive_content_trim(image)
                results.append(result3)
            
            # Select best result based on confidence and content preservation
            best_result = max(results, key=lambda x: x[1].confidence)
            
            # Additional validation for aggressive mode
            if aggressive and best_result[1].confidence < 0.6:
                logger.info("Low confidence in aggressive trim, using conservative approach")
                return results[0]  # Fall back to first method
            
            return best_result
            
        except Exception as e:
            logger.error(f"Smart trim failed: {e}")
            return image, PaddingInfo()
    
    def _trim_by_alpha(self, image: Image.Image) -> Tuple[Image.Image, PaddingInfo]:
        """Trim based on alpha channel with extra precision"""
        if image.mode not in ('RGBA', 'LA'):
            return image, PaddingInfo()
        
        try:
            # Get alpha channel
            alpha = image.split()[-1]
            alpha_array = np.array(alpha)
            
            # Use different thresholds for better detection
            thresholds = [5, 15, 30]  # Multiple thresholds to try
            
            best_padding = PaddingInfo()
            best_area = 0
            
            for threshold in thresholds:
                # Find non-transparent pixels
                content_mask = alpha_array > threshold
                
                if not np.any(content_mask):
                    continue
                
                # Find bounding box
                rows = np.any(content_mask, axis=1)
                cols = np.any(content_mask, axis=0)
                
                if not np.any(rows) or not np.any(cols):
                    continue
                
                top = np.argmax(rows)
                bottom = len(rows) - 1 - np.argmax(rows[::-1])
                left = np.argmax(cols)
                right = len(cols) - 1 - np.argmax(cols[::-1])
                
                # Calculate content area
                content_area = (right - left + 1) * (bottom - top + 1)
                
                # Prefer larger content areas (less aggressive cropping)
                if content_area > best_area:
                    height, width = alpha_array.shape
                    best_padding = PaddingInfo(
                        left=left,
                        top=top,
                        right=width - 1 - right,
                        bottom=height - 1 - bottom,
                        padding_color=(0, 0, 0, 0),
                        confidence=0.85
                    )
                    best_area = content_area
            
            if best_padding.has_padding:
                cropped = self._crop_padding(image, best_padding)
                if self._validate_crop_result(image, cropped):
                    return cropped, best_padding
            
            return image, PaddingInfo()
            
        except Exception as e:
            logger.error(f"Alpha-based trim failed: {e}")
            return image, PaddingInfo()
    
    def _aggressive_content_trim(self, image: Image.Image) -> Tuple[Image.Image, PaddingInfo]:
        """Aggressive content-based trimming"""
        try:
            # Convert to grayscale for content analysis
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            # Calculate gradient to find content edges
            grad_x = np.abs(np.diff(gray_array, axis=1))
            grad_y = np.abs(np.diff(gray_array, axis=0))
            
            # Pad gradients to match original size
            grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='constant')
            grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='constant')
            
            # Combine gradients
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Find content based on gradient threshold
            content_threshold = np.percentile(gradient_magnitude, 70)  # Top 30% of gradients
            content_mask = gradient_magnitude > content_threshold
            
            # Morphological operations to clean up mask
            from scipy import ndimage
            # Dilate to connect nearby content
            content_mask = ndimage.binary_dilation(content_mask, iterations=2)
            # Close small gaps
            content_mask = ndimage.binary_closing(content_mask, iterations=1)
            
            if not np.any(content_mask):
                return image, PaddingInfo()
            
            # Find bounding box
            rows = np.any(content_mask, axis=1)
            cols = np.any(content_mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                return image, PaddingInfo()
            
            top = np.argmax(rows)
            bottom = len(rows) - 1 - np.argmax(rows[::-1])
            left = np.argmax(cols)
            right = len(cols) - 1 - np.argmax(cols[::-1])
            
            height, width = gray_array.shape
            padding_info = PaddingInfo(
                left=left,
                top=top,
                right=width - 1 - right,
                bottom=height - 1 - bottom,
                confidence=0.6  # Lower confidence for aggressive method
            )
            
            cropped = self._crop_padding(image, padding_info)
            
            if self._validate_crop_result(image, cropped):
                return cropped, padding_info
            else:
                return image, PaddingInfo()
                
        except Exception as e:
            logger.error(f"Aggressive content trim failed: {e}")
            return image, PaddingInfo()
    
    def preview_padding_removal(self, image: Image.Image) -> Dict[str, any]:
        """
        Preview what padding would be removed without actually removing it
        
        Returns:
            Dictionary with preview information and visualizations
        """
        try:
            # Detect padding
            padding_info = self.detect_padding(image)
            
            preview_info = {
                'has_padding': padding_info.has_padding,
                'padding_detected': {
                    'left': padding_info.left,
                    'top': padding_info.top,
                    'right': padding_info.right,
                    'bottom': padding_info.bottom
                },
                'confidence': padding_info.confidence,
                'padding_color': padding_info.padding_color,
                'original_size': image.size,
                'would_crop_to': None,
                'content_preservation': 0.0
            }
            
            if padding_info.has_padding:
                # Calculate resulting size
                new_width = image.size[0] - padding_info.left - padding_info.right
                new_height = image.size[1] - padding_info.top - padding_info.bottom
                preview_info['would_crop_to'] = (new_width, new_height)
                
                # Calculate content preservation ratio
                original_area = image.size[0] * image.size[1]
                new_area = new_width * new_height
                preview_info['content_preservation'] = new_area / original_area if original_area > 0 else 0
                
                # Create visualization if possible
                try:
                    preview_info['visualization'] = self._create_padding_visualization(image, padding_info)
                except Exception as viz_e:
                    logger.debug(f"Visualization creation failed: {viz_e}")
            
            return preview_info
            
        except Exception as e:
            logger.error(f"Padding removal preview failed: {e}")
            return {'error': str(e)}
    
    def _create_padding_visualization(self, image: Image.Image, padding_info: PaddingInfo) -> str:
        """Create visualization showing detected padding areas"""
        try:
            from PIL import ImageDraw
            
            # Create copy for visualization
            viz_image = image.convert('RGBA')
            draw = ImageDraw.Draw(viz_image)
            
            width, height = image.size
            
            # Draw semi-transparent overlay on padding areas
            overlay_color = (255, 0, 0, 100)  # Red with transparency
            
            # Top padding
            if padding_info.top > 0:
                draw.rectangle([0, 0, width, padding_info.top], fill=overlay_color)
            
            # Bottom padding
            if padding_info.bottom > 0:
                draw.rectangle([0, height - padding_info.bottom, width, height], fill=overlay_color)
            
            # Left padding
            if padding_info.left > 0:
                draw.rectangle([0, 0, padding_info.left, height], fill=overlay_color)
            
            # Right padding
            if padding_info.right > 0:
                draw.rectangle([width - padding_info.right, 0, width, height], fill=overlay_color)
            
            # Draw content boundary
            content_left = padding_info.left
            content_top = padding_info.top
            content_right = width - padding_info.right
            content_bottom = height - padding_info.bottom
            
            draw.rectangle([content_left, content_top, content_right, content_bottom], 
                         outline=(0, 255, 0, 255), width=2)
            
            # Convert to base64 for easy transmission
            from .image_utils import ImageUtils
            return ImageUtils.image_to_base64(viz_image)
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return ""
    
    def batch_remove_padding(self, images: List[Image.Image], 
                           method: str = 'auto') -> List[Tuple[Image.Image, PaddingInfo]]:
        """
        Remove padding from multiple images using consistent method
        
        Args:
            images: List of PIL Images
            method: 'auto', 'transparent', 'solid', or 'aggressive'
            
        Returns:
            List of (processed_image, padding_info) tuples
        """
        results = []
        
        for image in images:
            try:
                if method == 'auto':
                    result = self.remove_padding(image, auto_detect=True)
                elif method == 'transparent':
                    result = self._detect_transparent_padding(image)
                    if result.has_padding:
                        processed = self._crop_padding(image, result)
                        result = (processed, result)
                    else:
                        result = (image, result)
                elif method == 'aggressive':
                    result = self.smart_trim(image, aggressive=True)
                else:  # Default to auto
                    result = self.remove_padding(image, auto_detect=True)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch padding removal failed for image: {e}")
                results.append((image, PaddingInfo()))  # Return original on error
        
        return results
