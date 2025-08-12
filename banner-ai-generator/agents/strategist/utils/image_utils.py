"""
Image processing utilities for strategist agent
"""

import io
import base64
import logging
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageUtils:
    """
    Utility class for common image processing operations
    """
    
    @staticmethod
    def load_image_from_base64(base64_data: str) -> Optional[Image.Image]:
        """Load PIL Image from base64 string"""
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',')[1]
            
            image_data = base64.b64decode(base64_data)
            return Image.open(io.BytesIO(image_data))
            
        except Exception as e:
            logger.error(f"Failed to load image from base64: {e}")
            return None
    
    @staticmethod
    def image_to_base64(image: Image.Image, format: str = 'PNG', quality: int = 95) -> str:
        """Convert PIL Image to base64 string"""
        try:
            buffer = io.BytesIO()
            
            # Ensure RGB mode for JPEG
            if format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA'):
                # Create white background for transparency
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[3])
                else:
                    background.paste(image, mask=image.split()[1])
                image = background
            
            save_kwargs = {'format': format, 'optimize': True}
            if format.upper() == 'JPEG':
                save_kwargs['quality'] = quality
            
            image.save(buffer, **save_kwargs)
            buffer.seek(0)
            
            return base64.b64encode(buffer.read()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to convert image to base64: {e}")
            return ""
    
    @staticmethod
    def get_image_info(image: Image.Image) -> Dict[str, Any]:
        """Get comprehensive image information"""
        return {
            'size': image.size,
            'mode': image.mode,
            'format': image.format,
            'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
            'aspect_ratio': image.size[0] / image.size[1] if image.size[1] > 0 else 1.0,
            'total_pixels': image.size[0] * image.size[1],
            'color_channels': len(image.getbands()) if hasattr(image, 'getbands') else 1
        }
    
    @staticmethod
    def resize_image(image: Image.Image, 
                    target_size: Tuple[int, int], 
                    maintain_aspect_ratio: bool = True,
                    resample: int = Image.Resampling.LANCZOS) -> Image.Image:
        """Resize image with various options"""
        try:
            if maintain_aspect_ratio:
                # Calculate size maintaining aspect ratio
                image.thumbnail(target_size, resample)
                return image
            else:
                # Resize to exact dimensions
                return image.resize(target_size, resample)
                
        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            return image
    
    @staticmethod
    def crop_to_aspect_ratio(image: Image.Image, aspect_ratio: float) -> Image.Image:
        """Crop image to specific aspect ratio"""
        try:
            width, height = image.size
            current_ratio = width / height
            
            if abs(current_ratio - aspect_ratio) < 0.01:
                return image  # Already close to target ratio
            
            if current_ratio > aspect_ratio:
                # Image is too wide, crop width
                new_width = int(height * aspect_ratio)
                left = (width - new_width) // 2
                return image.crop((left, 0, left + new_width, height))
            else:
                # Image is too tall, crop height
                new_height = int(width / aspect_ratio)
                top = (height - new_height) // 2
                return image.crop((0, top, width, top + new_height))
                
        except Exception as e:
            logger.error(f"Failed to crop to aspect ratio: {e}")
            return image
    
    @staticmethod
    def add_padding(image: Image.Image, 
                   padding: Tuple[int, int, int, int], 
                   fill_color: Tuple[int, int, int, int] = (255, 255, 255, 0)) -> Image.Image:
        """Add padding around image"""
        try:
            left, top, right, bottom = padding
            new_width = image.size[0] + left + right
            new_height = image.size[1] + top + bottom
            
            # Create new image with padding
            new_image = Image.new(image.mode, (new_width, new_height), fill_color)
            new_image.paste(image, (left, top))
            
            return new_image
            
        except Exception as e:
            logger.error(f"Failed to add padding: {e}")
            return image
    
    @staticmethod
    def enhance_image(image: Image.Image, 
                     brightness: float = 1.0,
                     contrast: float = 1.0,
                     saturation: float = 1.0,
                     sharpness: float = 1.0) -> Image.Image:
        """Enhance image properties"""
        try:
            enhanced = image
            
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(brightness)
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(contrast)
            
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(saturation)
            
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(sharpness)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Failed to enhance image: {e}")
            return image
    
    @staticmethod
    def convert_to_format(image: Image.Image, target_format: str) -> Image.Image:
        """Convert image to specific format"""
        try:
            if target_format.upper() == 'RGB' and image.mode != 'RGB':
                if image.mode in ('RGBA', 'LA'):
                    # Handle transparency by adding white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'RGBA':
                        background.paste(image, mask=image.split()[3])
                    else:
                        background.paste(image, mask=image.split()[1])
                    return background
                else:
                    return image.convert('RGB')
            
            elif target_format.upper() == 'RGBA' and image.mode != 'RGBA':
                return image.convert('RGBA')
            
            elif target_format.upper() == 'L' and image.mode != 'L':
                return image.convert('L')
            
            else:
                return image.convert(target_format.upper())
                
        except Exception as e:
            logger.error(f"Failed to convert image format: {e}")
            return image
    
    @staticmethod
    def create_thumbnail_sizes(image: Image.Image) -> Dict[str, str]:
        """Create multiple thumbnail sizes and return as base64"""
        sizes = {
            'small': (150, 150),
            'medium': (300, 300), 
            'large': (600, 600)
        }
        
        thumbnails = {}
        
        for size_name, size_dimensions in sizes.items():
            try:
                thumbnail = image.copy()
                thumbnail.thumbnail(size_dimensions, Image.Resampling.LANCZOS)
                thumbnails[size_name] = ImageUtils.image_to_base64(thumbnail)
            except Exception as e:
                logger.error(f"Failed to create {size_name} thumbnail: {e}")
        
        return thumbnails
    
    @staticmethod
    def detect_edges(image: Image.Image) -> Image.Image:
        """Detect edges in image"""
        try:
            # Convert to grayscale first
            gray_image = image.convert('L')
            
            # Apply edge detection filter
            edges = gray_image.filter(ImageFilter.FIND_EDGES)
            
            return edges
            
        except Exception as e:
            logger.error(f"Edge detection failed: {e}")
            return image
    
    @staticmethod
    def apply_blur(image: Image.Image, radius: float = 1.0) -> Image.Image:
        """Apply blur effect to image"""
        try:
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        except Exception as e:
            logger.error(f"Blur application failed: {e}")
            return image
    
    @staticmethod
    def get_dominant_colors(image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Get dominant colors from image using color quantization"""
        try:
            # Convert to RGB if needed
            rgb_image = image.convert('RGB')
            
            # Resize for faster processing
            rgb_image.thumbnail((150, 150))
            
            # Quantize colors
            quantized = rgb_image.quantize(colors=num_colors)
            
            # Get palette
            palette = quantized.getpalette()
            
            # Extract RGB values
            colors = []
            for i in range(num_colors):
                r = palette[i * 3]
                g = palette[i * 3 + 1] 
                b = palette[i * 3 + 2]
                colors.append((r, g, b))
            
            return colors
            
        except Exception as e:
            logger.error(f"Dominant color extraction failed: {e}")
            return []
    
    @staticmethod
    def calculate_image_quality_score(image: Image.Image) -> float:
        """Calculate quality score for image (0-10)"""
        try:
            score = 0.0
            
            # Resolution score (0-3 points)
            width, height = image.size
            min_dimension = min(width, height)
            
            if min_dimension >= 1000:
                score += 3.0
            elif min_dimension >= 500:
                score += 2.5
            elif min_dimension >= 200:
                score += 2.0
            elif min_dimension >= 100:
                score += 1.0
            
            # Aspect ratio score (0-2 points)
            aspect_ratio = width / height
            if 0.5 <= aspect_ratio <= 2.0:  # Reasonable aspect ratio
                score += 2.0
            elif 0.25 <= aspect_ratio <= 4.0:
                score += 1.0
            
            # Color mode score (0-2 points)
            if image.mode == 'RGBA':
                score += 2.0  # Full color with transparency
            elif image.mode == 'RGB':
                score += 1.5  # Full color
            elif image.mode == 'L':
                score += 0.5  # Grayscale
            
            # File integrity score (0-1 point)
            try:
                image.verify()
                score += 1.0
            except:
                pass  # Verification failed, no points
            
            # Complexity score (0-2 points) - based on color variation
            try:
                colors = ImageUtils.get_dominant_colors(image, 8)
                if len(colors) >= 5:
                    score += 2.0  # Rich color palette
                elif len(colors) >= 3:
                    score += 1.5
                elif len(colors) >= 2:
                    score += 1.0
            except:
                pass
            
            return min(10.0, score)
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 5.0  # Default middle score
    
    @staticmethod
    def validate_image_for_web(image: Image.Image) -> Dict[str, Any]:
        """Validate image for web use"""
        validation = {
            'is_valid': True,
            'issues': [],
            'recommendations': [],
            'optimizations': []
        }
        
        try:
            width, height = image.size
            
            # Check dimensions
            if width < 100 or height < 100:
                validation['issues'].append("Image resolution too low for web use")
                validation['is_valid'] = False
            elif width < 300 or height < 300:
                validation['recommendations'].append("Consider using higher resolution for better quality")
            
            if width > 2000 or height > 2000:
                validation['optimizations'].append("Consider resizing to reduce file size")
            
            # Check aspect ratio
            aspect_ratio = width / height
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                validation['issues'].append("Extreme aspect ratio may cause display issues")
            
            # Check color mode
            if image.mode not in ['RGB', 'RGBA', 'L']:
                validation['issues'].append(f"Color mode '{image.mode}' may not be web-compatible")
                validation['optimizations'].append("Convert to RGB or RGBA")
            
            # Check for transparency usage
            if image.mode == 'RGBA':
                # Check if transparency is actually used
                alpha_channel = image.split()[3]
                alpha_range = alpha_channel.getextrema()
                if alpha_range[0] == alpha_range[1] == 255:
                    validation['optimizations'].append("Convert to RGB - transparency not used")
            
            # File size estimation
            estimated_size = ImageUtils._estimate_file_size(image)
            if estimated_size > 1024 * 1024:  # 1MB
                validation['optimizations'].append("Optimize for smaller file size")
            
            return validation
            
        except Exception as e:
            logger.error(f"Web validation failed: {e}")
            validation['issues'].append("Could not validate image")
            return validation
    
    @staticmethod
    def _estimate_file_size(image: Image.Image) -> int:
        """Estimate file size in bytes"""
        try:
            # Rough estimation based on dimensions and color mode
            width, height = image.size
            pixels = width * height
            
            if image.mode == 'RGBA':
                bytes_per_pixel = 4
            elif image.mode == 'RGB':
                bytes_per_pixel = 3
            elif image.mode == 'L':
                bytes_per_pixel = 1
            else:
                bytes_per_pixel = 3  # Default
            
            # Raw size with compression factor
            raw_size = pixels * bytes_per_pixel
            compressed_size = raw_size * 0.1  # Assume 90% compression
            
            return int(compressed_size)
            
        except:
            return 0
    
    @staticmethod
    def create_placeholder_image(size: Tuple[int, int], 
                                color: Tuple[int, int, int] = (200, 200, 200),
                                text: str = "") -> Image.Image:
        """Create placeholder image with optional text"""
        try:
            from PIL import ImageDraw, ImageFont
            
            # Create image
            image = Image.new('RGB', size, color)
            
            if text:
                draw = ImageDraw.Draw(image)
                
                # Try to use a default font
                try:
                    # Try to load a system font
                    font_size = min(size) // 10
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    # Fallback to default font
                    font = ImageFont.load_default()
                
                # Calculate text position (centered)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                x = (size[0] - text_width) // 2
                y = (size[1] - text_height) // 2
                
                # Draw text
                text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
                draw.text((x, y), text, fill=text_color, font=font)
            
            return image
            
        except Exception as e:
            logger.error(f"Placeholder creation failed: {e}")
            # Return simple colored rectangle
            return Image.new('RGB', size, color)
    
    @staticmethod
    def batch_process_images(images: List[Image.Image], 
                           operations: List[Dict[str, Any]]) -> List[Image.Image]:
        """Apply batch operations to multiple images"""
        processed_images = []
        
        for image in images:
            processed = image.copy()
            
            for operation in operations:
                op_type = operation.get('type')
                params = operation.get('params', {})
                
                try:
                    if op_type == 'resize':
                        processed = ImageUtils.resize_image(processed, **params)
                    elif op_type == 'enhance':
                        processed = ImageUtils.enhance_image(processed, **params)
                    elif op_type == 'crop_aspect':
                        processed = ImageUtils.crop_to_aspect_ratio(processed, **params)
                    elif op_type == 'convert':
                        processed = ImageUtils.convert_to_format(processed, **params)
                    elif op_type == 'blur':
                        processed = ImageUtils.apply_blur(processed, **params)
                    
                except Exception as e:
                    logger.error(f"Batch operation {op_type} failed: {e}")
                    continue
            
            processed_images.append(processed)
        
        return processed_images
    
    @staticmethod
    def compare_images(image1: Image.Image, image2: Image.Image) -> Dict[str, Any]:
        """Compare two images and return similarity metrics"""
        try:
            # Ensure same size for comparison
            size = (100, 100)  # Standard comparison size
            img1_resized = image1.copy()
            img1_resized.thumbnail(size)
            img2_resized = image2.copy()
            img2_resized.thumbnail(size)
            
            # Convert to same mode
            if img1_resized.mode != img2_resized.mode:
                img1_resized = img1_resized.convert('RGB')
                img2_resized = img2_resized.convert('RGB')
            
            # Calculate differences
            diff = ImageUtils._calculate_image_difference(img1_resized, img2_resized)
            
            return {
                'similarity_score': 1.0 - diff,  # 0-1 scale
                'difference_score': diff,
                'are_similar': diff < 0.1,  # 90% similar
                'size_difference': abs(image1.size[0] * image1.size[1] - image2.size[0] * image2.size[1]),
                'aspect_ratio_diff': abs(image1.size[0]/image1.size[1] - image2.size[0]/image2.size[1])
            }
            
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            return {
                'similarity_score': 0.0,
                'difference_score': 1.0,
                'are_similar': False,
                'error': str(e)
            }
    
    @staticmethod
    def _calculate_image_difference(image1: Image.Image, image2: Image.Image) -> float:
        """Calculate normalized difference between two images"""
        try:
            # Convert to numpy arrays
            arr1 = np.array(image1)
            arr2 = np.array(image2)
            
            # Calculate mean squared difference
            diff = np.mean((arr1.astype(float) - arr2.astype(float)) ** 2)
            
            # Normalize to 0-1 scale (assuming max diff is 255^2)
            normalized_diff = diff / (255.0 ** 2)
            
            return min(1.0, normalized_diff)
            
        except Exception as e:
            logger.error(f"Difference calculation failed: {e}")
            return 1.0  # Maximum difference on error
