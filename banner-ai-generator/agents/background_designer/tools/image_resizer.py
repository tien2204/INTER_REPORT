"""
Image Resizer Tool

Handles intelligent resizing and optimization of background images
for different banner dimensions and use cases.
"""

import base64
import io
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


class ImageResizer:
    """
    Intelligent image resizing and optimization tool
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Resize algorithms
        self.default_algorithm = config.get("default_algorithm", "lanczos")
        self.algorithms = {
            "lanczos": Image.LANCZOS,
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST
        }
        
        # Quality settings
        self.jpeg_quality = config.get("jpeg_quality", 95)
        self.png_optimize = config.get("png_optimize", True)
        
        # Enhancement settings
        self.auto_enhance = config.get("auto_enhance", True)
        self.sharpening_factor = config.get("sharpening_factor", 1.1)
        self.contrast_factor = config.get("contrast_factor", 1.05)
    
    async def resize_image(self, 
                          image_data: str, 
                          target_width: int, 
                          target_height: int,
                          maintain_aspect: bool = True,
                          crop_strategy: str = "center",
                          enhance: bool = True) -> str:
        """
        Resize image to target dimensions
        
        Args:
            image_data: Base64 encoded image data
            target_width: Target width in pixels
            target_height: Target height in pixels
            maintain_aspect: Whether to maintain aspect ratio
            crop_strategy: How to handle cropping ("center", "smart", "top", "bottom")
            enhance: Whether to apply enhancement after resizing
        
        Returns:
            Base64 encoded resized image
        """
        try:
            # Decode image
            image = self._decode_image(image_data)
            if not image:
                raise ValueError("Failed to decode image")
            
            original_width, original_height = image.size
            logger.info(f"Resizing image from {original_width}x{original_height} to {target_width}x{target_height}")
            
            # Calculate target dimensions
            if maintain_aspect:
                resized_image = await self._resize_with_aspect_ratio(
                    image, target_width, target_height, crop_strategy
                )
            else:
                resized_image = await self._resize_stretch(
                    image, target_width, target_height
                )
            
            # Apply enhancements if requested
            if enhance and self.auto_enhance:
                resized_image = await self._enhance_image(resized_image)
            
            # Convert back to base64
            return self._encode_image(resized_image)
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            raise
    
    async def smart_crop(self, 
                        image_data: str, 
                        target_width: int, 
                        target_height: int) -> str:
        """
        Intelligent cropping based on content analysis
        
        Args:
            image_data: Base64 encoded image data
            target_width: Target width
            target_height: Target height
        
        Returns:
            Cropped and resized image
        """
        try:
            image = self._decode_image(image_data)
            if not image:
                raise ValueError("Failed to decode image")
            
            # Find the best crop region
            crop_region = await self._find_optimal_crop_region(
                image, target_width, target_height
            )
            
            # Crop and resize
            cropped_image = image.crop(crop_region)
            resized_image = cropped_image.resize(
                (target_width, target_height), 
                self.algorithms[self.default_algorithm]
            )
            
            # Enhance if needed
            if self.auto_enhance:
                resized_image = await self._enhance_image(resized_image)
            
            return self._encode_image(resized_image)
            
        except Exception as e:
            logger.error(f"Error in smart crop: {e}")
            raise
    
    async def create_multiple_sizes(self, 
                                  image_data: str, 
                                  size_variants: List[Dict[str, int]]) -> Dict[str, str]:
        """
        Create multiple size variants of an image
        
        Args:
            image_data: Base64 encoded image data
            size_variants: List of {"width": w, "height": h} dictionaries
        
        Returns:
            Dictionary mapping size descriptions to base64 image data
        """
        try:
            results = {}
            
            for i, size in enumerate(size_variants):
                width = size["width"]
                height = size["height"]
                
                try:
                    resized_data = await self.resize_image(
                        image_data, width, height, maintain_aspect=True
                    )
                    
                    size_key = f"{width}x{height}"
                    results[size_key] = resized_data
                    
                    logger.info(f"Created size variant: {size_key}")
                    
                except Exception as e:
                    logger.error(f"Error creating size {width}x{height}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error creating multiple sizes: {e}")
            return {}
    
    async def optimize_for_web(self, image_data: str, max_file_size_kb: int = 500) -> str:
        """
        Optimize image for web use with file size constraints
        
        Args:
            image_data: Base64 encoded image data
            max_file_size_kb: Maximum file size in KB
        
        Returns:
            Optimized image data
        """
        try:
            image = self._decode_image(image_data)
            if not image:
                raise ValueError("Failed to decode image")
            
            # Start with high quality
            quality = self.jpeg_quality
            optimized_data = None
            
            # Iteratively reduce quality until size target is met
            for attempt in range(5):
                # Convert to JPEG for smaller size
                if image.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparent images
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                    image_to_save = background
                else:
                    image_to_save = image
                
                # Save with current quality
                buffer = io.BytesIO()
                image_to_save.save(buffer, format="JPEG", quality=quality, optimize=True)
                
                # Check file size
                file_size_kb = buffer.tell() / 1024
                
                if file_size_kb <= max_file_size_kb:
                    buffer.seek(0)
                    optimized_data = base64.b64encode(buffer.getvalue()).decode()
                    logger.info(f"Optimized to {file_size_kb:.1f}KB with quality {quality}")
                    break
                
                # Reduce quality for next attempt
                quality = max(50, quality - 15)
            
            if optimized_data:
                return f"data:image/jpeg;base64,{optimized_data}"
            else:
                # If still too large, resize down
                logger.warning("Could not optimize to target size, reducing dimensions")
                return await self._reduce_dimensions_for_size(image, max_file_size_kb)
                
        except Exception as e:
            logger.error(f"Error optimizing for web: {e}")
            return image_data  # Return original if optimization fails
    
    async def _resize_with_aspect_ratio(self, 
                                       image: Image.Image, 
                                       target_width: int, 
                                       target_height: int,
                                       crop_strategy: str) -> Image.Image:
        """Resize maintaining aspect ratio"""
        try:
            original_width, original_height = image.size
            original_ratio = original_width / original_height
            target_ratio = target_width / target_height
            
            if abs(original_ratio - target_ratio) < 0.01:
                # Ratios are very close, just resize
                return image.resize((target_width, target_height), self.algorithms[self.default_algorithm])
            
            # Calculate dimensions for fitting
            if original_ratio > target_ratio:
                # Image is wider, fit by height
                fit_height = target_height
                fit_width = int(fit_height * original_ratio)
            else:
                # Image is taller, fit by width
                fit_width = target_width
                fit_height = int(fit_width / original_ratio)
            
            # Resize to fit dimensions
            fitted_image = image.resize((fit_width, fit_height), self.algorithms[self.default_algorithm])
            
            # Create final image with target dimensions
            final_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))
            
            # Calculate paste position based on crop strategy
            if crop_strategy == "center":
                x = (target_width - fit_width) // 2
                y = (target_height - fit_height) // 2
            elif crop_strategy == "top":
                x = (target_width - fit_width) // 2
                y = 0
            elif crop_strategy == "bottom":
                x = (target_width - fit_width) // 2
                y = target_height - fit_height
            elif crop_strategy == "smart":
                # Use smart positioning based on content
                x, y = await self._calculate_smart_position(
                    fitted_image, target_width, target_height
                )
            else:
                x, y = 0, 0
            
            # Handle cropping if fitted image is larger than target
            if fit_width > target_width or fit_height > target_height:
                # Crop the fitted image
                crop_x = max(0, (fit_width - target_width) // 2)
                crop_y = max(0, (fit_height - target_height) // 2)
                
                fitted_image = fitted_image.crop((
                    crop_x, crop_y,
                    crop_x + target_width, crop_y + target_height
                ))
                
                final_image = fitted_image
            else:
                # Paste fitted image onto background
                final_image.paste(fitted_image, (x, y))
            
            return final_image
            
        except Exception as e:
            logger.error(f"Error in aspect ratio resize: {e}")
            # Fallback to simple resize
            return image.resize((target_width, target_height), self.algorithms[self.default_algorithm])
    
    async def _resize_stretch(self, 
                             image: Image.Image, 
                             target_width: int, 
                             target_height: int) -> Image.Image:
        """Resize without maintaining aspect ratio"""
        return image.resize((target_width, target_height), self.algorithms[self.default_algorithm])
    
    async def _find_optimal_crop_region(self, 
                                       image: Image.Image, 
                                       target_width: int, 
                                       target_height: int) -> Tuple[int, int, int, int]:
        """Find optimal crop region using content analysis"""
        try:
            width, height = image.size
            target_ratio = target_width / target_height
            
            # Calculate crop dimensions
            if width / height > target_ratio:
                # Image is wider, crop width
                crop_height = height
                crop_width = int(height * target_ratio)
            else:
                # Image is taller, crop height
                crop_width = width
                crop_height = int(width / target_ratio)
            
            # Find best position using simple entropy-based analysis
            best_x, best_y = await self._find_best_crop_position(
                image, crop_width, crop_height
            )
            
            return (best_x, best_y, best_x + crop_width, best_y + crop_height)
            
        except Exception as e:
            logger.error(f"Error finding optimal crop: {e}")
            # Fallback to center crop
            width, height = image.size
            target_ratio = target_width / target_height
            
            if width / height > target_ratio:
                crop_width = int(height * target_ratio)
                crop_height = height
                x = (width - crop_width) // 2
                y = 0
            else:
                crop_width = width
                crop_height = int(width / target_ratio)
                x = 0
                y = (height - crop_height) // 2
            
            return (x, y, x + crop_width, y + crop_height)
    
    async def _find_best_crop_position(self, 
                                      image: Image.Image, 
                                      crop_width: int, 
                                      crop_height: int) -> Tuple[int, int]:
        """Find best crop position using content analysis"""
        try:
            width, height = image.size
            
            # Convert to grayscale for analysis
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Calculate entropy for different crop positions
            best_entropy = -1
            best_x, best_y = 0, 0
            
            # Sample positions (to avoid checking every pixel)
            step_x = max(1, (width - crop_width) // 10)
            step_y = max(1, (height - crop_height) // 10)
            
            for y in range(0, height - crop_height + 1, step_y):
                for x in range(0, width - crop_width + 1, step_x):
                    # Extract crop region
                    crop_region = img_array[y:y+crop_height, x:x+crop_width]
                    
                    # Calculate entropy (measure of information content)
                    entropy = self._calculate_entropy(crop_region)
                    
                    if entropy > best_entropy:
                        best_entropy = entropy
                        best_x, best_y = x, y
            
            return best_x, best_y
            
        except Exception as e:
            logger.error(f"Error finding best crop position: {e}")
            # Fallback to center
            width, height = image.size
            return ((width - crop_width) // 2, (height - crop_height) // 2)
    
    def _calculate_entropy(self, img_array: np.ndarray) -> float:
        """Calculate entropy of image region"""
        try:
            # Calculate histogram
            hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
            
            # Normalize histogram
            hist = hist / hist.sum()
            
            # Calculate entropy
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            return entropy
            
        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 0.0
    
    async def _calculate_smart_position(self, 
                                       fitted_image: Image.Image, 
                                       target_width: int, 
                                       target_height: int) -> Tuple[int, int]:
        """Calculate smart positioning for image placement"""
        try:
            fit_width, fit_height = fitted_image.size
            
            # For now, use center positioning
            # In a more sophisticated implementation, this could analyze
            # the image content to find the best positioning
            
            x = max(0, (target_width - fit_width) // 2)
            y = max(0, (target_height - fit_height) // 2)
            
            return x, y
            
        except Exception as e:
            logger.error(f"Error calculating smart position: {e}")
            return 0, 0
    
    async def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply enhancement to resized image"""
        try:
            enhanced = image
            
            # Apply subtle sharpening
            if self.sharpening_factor > 1.0:
                sharpener = ImageEnhance.Sharpness(enhanced)
                enhanced = sharpener.enhance(self.sharpening_factor)
            
            # Apply contrast enhancement
            if self.contrast_factor != 1.0:
                contrast_enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = contrast_enhancer.enhance(self.contrast_factor)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image
    
    async def _reduce_dimensions_for_size(self, 
                                         image: Image.Image, 
                                         max_file_size_kb: int) -> str:
        """Reduce image dimensions to meet file size requirements"""
        try:
            width, height = image.size
            reduction_factor = 0.8
            
            while True:
                new_width = int(width * reduction_factor)
                new_height = int(height * reduction_factor)
                
                if new_width < 200 or new_height < 200:
                    break
                
                resized = image.resize((new_width, new_height), self.algorithms[self.default_algorithm])
                
                # Test file size
                buffer = io.BytesIO()
                if resized.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', resized.size, (255, 255, 255))
                    background.paste(resized, mask=resized.split()[-1] if resized.mode == 'RGBA' else None)
                    background.save(buffer, format="JPEG", quality=80, optimize=True)
                else:
                    resized.save(buffer, format="JPEG", quality=80, optimize=True)
                
                file_size_kb = buffer.tell() / 1024
                
                if file_size_kb <= max_file_size_kb:
                    buffer.seek(0)
                    optimized_data = base64.b64encode(buffer.getvalue()).decode()
                    logger.info(f"Reduced to {new_width}x{new_height}, size: {file_size_kb:.1f}KB")
                    return f"data:image/jpeg;base64,{optimized_data}"
                
                reduction_factor *= 0.9
            
            # If we get here, return heavily compressed version
            buffer = io.BytesIO()
            final_image = image.resize((200, int(200 * height / width)), self.algorithms[self.default_algorithm])
            if final_image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', final_image.size, (255, 255, 255))
                background.paste(final_image, mask=final_image.split()[-1] if final_image.mode == 'RGBA' else None)
                background.save(buffer, format="JPEG", quality=50)
            else:
                final_image.save(buffer, format="JPEG", quality=50)
            
            buffer.seek(0)
            optimized_data = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{optimized_data}"
            
        except Exception as e:
            logger.error(f"Error reducing dimensions: {e}")
            raise
    
    def _decode_image(self, image_data: str) -> Optional[Image.Image]:
        """Decode base64 image to PIL Image"""
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            return image
            
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None
    
    def _encode_image(self, image: Image.Image, format: str = "PNG") -> str:
        """Encode PIL Image to base64"""
        try:
            buffer = io.BytesIO()
            
            if format.upper() == "JPEG" and image.mode in ('RGBA', 'LA', 'P'):
                # Convert to RGB for JPEG
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                background.save(buffer, format=format, quality=self.jpeg_quality, optimize=True)
            else:
                save_kwargs = {"format": format}
                if format.upper() == "PNG" and self.png_optimize:
                    save_kwargs["optimize"] = True
                elif format.upper() == "JPEG":
                    save_kwargs["quality"] = self.jpeg_quality
                    save_kwargs["optimize"] = True
                
                image.save(buffer, **save_kwargs)
            
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/{format.lower()};base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    async def get_image_info(self, image_data: str) -> Dict[str, Any]:
        """Get detailed information about an image"""
        try:
            image = self._decode_image(image_data)
            if not image:
                return {"error": "Failed to decode image"}
            
            # Calculate file size
            if image_data.startswith('data:image'):
                b64_data = image_data.split(',')[1]
            else:
                b64_data = image_data
            
            file_size_bytes = len(base64.b64decode(b64_data))
            
            return {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format or "Unknown",
                "file_size_bytes": file_size_bytes,
                "file_size_kb": round(file_size_bytes / 1024, 2),
                "aspect_ratio": round(image.width / image.height, 3),
                "has_transparency": image.mode in ('RGBA', 'LA', 'P')
            }
            
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {"error": str(e)}
