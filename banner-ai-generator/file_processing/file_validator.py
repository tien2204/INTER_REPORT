"""
File Validator

Comprehensive file validation for security, format compliance,
and business rule enforcement for the banner generation system.
"""

import hashlib
import magic
import base64
import io
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from datetime import datetime
from pathlib import Path
from structlog import get_logger

logger = get_logger(__name__)


class FileValidator:
    """
    Comprehensive file validation and security checking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # File size limits (in bytes)
        self.max_file_sizes = config.get("max_file_sizes", {
            "image": 10 * 1024 * 1024,  # 10MB
            "svg": 1 * 1024 * 1024,     # 1MB
            "json": 100 * 1024,         # 100KB
            "default": 5 * 1024 * 1024   # 5MB
        })
        
        # Allowed file types
        self.allowed_image_types = config.get("allowed_image_types", {
            'image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'
        })
        
        self.allowed_vector_types = config.get("allowed_vector_types", {
            'image/svg+xml', 'application/svg+xml'
        })
        
        self.allowed_data_types = config.get("allowed_data_types", {
            'application/json', 'text/json'
        })
        
        # Security settings
        self.scan_for_malware = config.get("scan_for_malware", True)
        self.validate_image_headers = config.get("validate_image_headers", True)
        self.check_embedded_content = config.get("check_embedded_content", True)
        
        # Business rules
        self.min_image_dimensions = config.get("min_image_dimensions", (32, 32))
        self.max_image_dimensions = config.get("max_image_dimensions", (8192, 8192))
        self.max_color_depth = config.get("max_color_depth", 32)
        
        # Initialize magic for MIME type detection
        try:
            self.magic_mime = magic.Magic(mime=True)
        except Exception as e:
            logger.warning(f"Could not initialize python-magic: {e}")
            self.magic_mime = None
    
    async def validate_file(self, 
                          file_data: Union[str, bytes],
                          filename: str = "unknown",
                          expected_type: Optional[str] = None,
                          additional_checks: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive file validation
        
        Args:
            file_data: File content as bytes or base64 string
            filename: Original filename
            expected_type: Expected file type (image, svg, json)
            additional_checks: Additional validation checks to perform
        
        Returns:
            Validation result with detailed information
        """
        try:
            logger.info(f"Validating file: {filename}")
            
            # Decode file data if base64
            raw_data = self._decode_file_data(file_data)
            
            # Basic file information
            file_info = await self._get_file_info(raw_data, filename)
            
            # Security validation
            security_check = await self._security_validation(raw_data, filename)
            
            # Format validation
            format_check = await self._format_validation(raw_data, filename, expected_type)
            
            # Content validation
            content_check = await self._content_validation(raw_data, filename, expected_type)
            
            # Business rules validation
            business_check = await self._business_rules_validation(raw_data, filename, expected_type)
            
            # Additional checks
            additional_results = {}
            if additional_checks:
                for check in additional_checks:
                    try:
                        result = await self._run_additional_check(check, raw_data, filename)
                        additional_results[check] = result
                    except Exception as e:
                        logger.error(f"Additional check '{check}' failed: {e}")
                        additional_results[check] = {"passed": False, "error": str(e)}
            
            # Compile overall result
            all_checks = [security_check, format_check, content_check, business_check]
            all_passed = all(check.get("passed", False) for check in all_checks)
            
            # Collect all issues
            issues = []
            warnings = []
            
            for check in all_checks:
                issues.extend(check.get("issues", []))
                warnings.extend(check.get("warnings", []))
            
            result = {
                "valid": all_passed,
                "file_info": file_info,
                "security": security_check,
                "format": format_check,
                "content": content_check,
                "business_rules": business_check,
                "additional_checks": additional_results,
                "issues": issues,
                "warnings": warnings,
                "validation_summary": {
                    "total_checks": len(all_checks),
                    "passed_checks": sum(1 for check in all_checks if check.get("passed", False)),
                    "issues_count": len(issues),
                    "warnings_count": len(warnings)
                }
            }
            
            logger.info(f"File validation completed: {'PASSED' if all_passed else 'FAILED'} ({len(issues)} issues)")
            return result
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return {
                "valid": False,
                "error": str(e),
                "file_info": {},
                "issues": [f"Validation error: {str(e)}"]
            }
    
    async def validate_image(self, 
                           image_data: Union[str, bytes],
                           filename: str = "image",
                           requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Specialized image validation
        
        Args:
            image_data: Image data
            filename: Filename
            requirements: Specific image requirements
        
        Returns:
            Image validation result
        """
        try:
            # Decode image
            raw_data = self._decode_file_data(image_data)
            
            # Basic validation
            basic_result = await self.validate_file(raw_data, filename, "image")
            
            if not basic_result["valid"]:
                return basic_result
            
            # Image-specific checks
            image_checks = await self._validate_image_specific(raw_data, requirements or {})
            
            # Merge results
            basic_result["image_specific"] = image_checks
            basic_result["valid"] = basic_result["valid"] and image_checks.get("passed", False)
            
            if not image_checks.get("passed", False):
                basic_result["issues"].extend(image_checks.get("issues", []))
            
            return basic_result
            
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return {"valid": False, "error": str(e)}
    
    async def validate_logo(self, 
                          logo_data: Union[str, bytes],
                          filename: str = "logo",
                          brand_requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Specialized logo validation
        
        Args:
            logo_data: Logo data
            filename: Filename
            brand_requirements: Brand-specific requirements
        
        Returns:
            Logo validation result
        """
        try:
            # Basic image validation
            result = await self.validate_image(logo_data, filename)
            
            if not result["valid"]:
                return result
            
            # Logo-specific validation
            logo_checks = await self._validate_logo_specific(
                self._decode_file_data(logo_data), 
                brand_requirements or {}
            )
            
            result["logo_specific"] = logo_checks
            result["valid"] = result["valid"] and logo_checks.get("passed", False)
            
            if not logo_checks.get("passed", False):
                result["issues"].extend(logo_checks.get("issues", []))
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating logo: {e}")
            return {"valid": False, "error": str(e)}
    
    def _decode_file_data(self, file_data: Union[str, bytes]) -> bytes:
        """Decode file data from various formats"""
        try:
            if isinstance(file_data, bytes):
                return file_data
            elif isinstance(file_data, str):
                if file_data.startswith('data:'):
                    # Data URL format
                    _, data = file_data.split(',', 1)
                    return base64.b64decode(data)
                else:
                    # Assume base64
                    return base64.b64decode(file_data)
            else:
                raise ValueError(f"Unsupported file data type: {type(file_data)}")
                
        except Exception as e:
            logger.error(f"Error decoding file data: {e}")
            raise
    
    async def _get_file_info(self, data: bytes, filename: str) -> Dict[str, Any]:
        """Extract basic file information"""
        try:
            info = {
                "filename": filename,
                "size_bytes": len(data),
                "size_kb": round(len(data) / 1024, 2),
                "size_mb": round(len(data) / (1024 * 1024), 2),
                "file_extension": Path(filename).suffix.lower(),
                "content_hash": hashlib.sha256(data).hexdigest()[:16],
                "analyzed_at": datetime.now().isoformat()
            }
            
            # MIME type detection
            if self.magic_mime:
                try:
                    detected_mime = self.magic_mime.from_buffer(data)
                    info["detected_mime_type"] = detected_mime
                except Exception as e:
                    logger.warning(f"MIME detection failed: {e}")
                    info["detected_mime_type"] = "unknown"
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {"error": str(e)}
    
    async def _security_validation(self, data: bytes, filename: str) -> Dict[str, Any]:
        """Perform security validation"""
        try:
            issues = []
            warnings = []
            
            # File size check
            file_size = len(data)
            max_size = self.max_file_sizes.get("default")
            
            if file_size > max_size:
                issues.append(f"File too large: {file_size} bytes > {max_size} bytes")
            
            # Suspicious filename patterns
            suspicious_patterns = [
                '.exe', '.bat', '.cmd', '.scr', '.pif', '.com',
                '.js', '.vbs', '.php', '.asp', '.jsp'
            ]
            
            filename_lower = filename.lower()
            for pattern in suspicious_patterns:
                if pattern in filename_lower:
                    issues.append(f"Suspicious filename pattern: {pattern}")
            
            # Header validation
            if self.validate_image_headers:
                header_issues = await self._validate_file_headers(data)
                issues.extend(header_issues)
            
            # Embedded content check
            if self.check_embedded_content:
                embedded_issues = await self._check_embedded_content(data)
                warnings.extend(embedded_issues)
            
            # Malware scanning (if enabled)
            if self.scan_for_malware:
                malware_result = await self._scan_for_malware(data)
                if not malware_result.get("clean", True):
                    issues.append("Potential malware detected")
            
            return {
                "passed": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "security_level": "high" if len(issues) == 0 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error in security validation: {e}")
            return {"passed": False, "error": str(e)}
    
    async def _format_validation(self, data: bytes, filename: str, expected_type: Optional[str]) -> Dict[str, Any]:
        """Validate file format compliance"""
        try:
            issues = []
            warnings = []
            
            # Detect actual format
            detected_format = await self._detect_file_format(data)
            
            # Validate against expected type
            if expected_type:
                if not self._is_format_compatible(detected_format, expected_type):
                    issues.append(f"Format mismatch: detected {detected_format}, expected {expected_type}")
            
            # Validate format-specific requirements
            if detected_format == "image":
                format_issues = await self._validate_image_format(data)
                issues.extend(format_issues)
            elif detected_format == "svg":
                format_issues = await self._validate_svg_format(data)
                issues.extend(format_issues)
            elif detected_format == "json":
                format_issues = await self._validate_json_format(data)
                issues.extend(format_issues)
            
            return {
                "passed": len(issues) == 0,
                "detected_format": detected_format,
                "issues": issues,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"Error in format validation: {e}")
            return {"passed": False, "error": str(e)}
    
    async def _content_validation(self, data: bytes, filename: str, expected_type: Optional[str]) -> Dict[str, Any]:
        """Validate file content"""
        try:
            issues = []
            warnings = []
            
            # Content-specific validation
            if expected_type == "image":
                content_issues = await self._validate_image_content(data)
                issues.extend(content_issues)
            elif expected_type == "svg":
                content_issues = await self._validate_svg_content(data)
                issues.extend(content_issues)
            elif expected_type == "json":
                content_issues = await self._validate_json_content(data)
                issues.extend(content_issues)
            
            return {
                "passed": len(issues) == 0,
                "issues": issues,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"Error in content validation: {e}")
            return {"passed": False, "error": str(e)}
    
    async def _business_rules_validation(self, data: bytes, filename: str, expected_type: Optional[str]) -> Dict[str, Any]:
        """Validate against business rules"""
        try:
            issues = []
            warnings = []
            
            # File size business rules
            file_size = len(data)
            
            if expected_type in self.max_file_sizes:
                max_allowed = self.max_file_sizes[expected_type]
                if file_size > max_allowed:
                    issues.append(f"File exceeds business limit: {file_size} > {max_allowed} bytes")
            
            # Image-specific business rules
            if expected_type == "image":
                try:
                    image = Image.open(io.BytesIO(data))
                    width, height = image.size
                    
                    # Dimension checks
                    min_w, min_h = self.min_image_dimensions
                    max_w, max_h = self.max_image_dimensions
                    
                    if width < min_w or height < min_h:
                        issues.append(f"Image too small: {width}x{height} < {min_w}x{min_h}")
                    
                    if width > max_w or height > max_h:
                        issues.append(f"Image too large: {width}x{height} > {max_w}x{max_h}")
                    
                    # Color depth check
                    if hasattr(image, 'bits'):
                        if image.bits > self.max_color_depth:
                            warnings.append(f"High color depth: {image.bits} bits")
                    
                except Exception as e:
                    issues.append(f"Could not analyze image: {e}")
            
            return {
                "passed": len(issues) == 0,
                "issues": issues,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"Error in business rules validation: {e}")
            return {"passed": False, "error": str(e)}
    
    async def _detect_file_format(self, data: bytes) -> str:
        """Detect file format from content"""
        try:
            # Check magic numbers/signatures
            if data.startswith(b'\xFF\xD8\xFF'):
                return "jpeg"
            elif data.startswith(b'\x89PNG\r\n\x1a\n'):
                return "png"
            elif data.startswith(b'RIFF') and b'WEBP' in data[:12]:
                return "webp"
            elif data.startswith(b'BM'):
                return "bmp"
            elif data.startswith(b'<svg') or data.startswith(b'<?xml') and b'<svg' in data[:200]:
                return "svg"
            elif data.startswith(b'{') or data.startswith(b'['):
                return "json"
            elif data.startswith(b'GIF'):
                return "gif"
            else:
                # Try to detect with PIL
                try:
                    image = Image.open(io.BytesIO(data))
                    return image.format.lower() if image.format else "unknown"
                except Exception:
                    return "unknown"
                    
        except Exception as e:
            logger.error(f"Error detecting file format: {e}")
            return "unknown"
    
    def _is_format_compatible(self, detected_format: str, expected_type: str) -> bool:
        """Check if detected format is compatible with expected type"""
        compatibility_map = {
            "image": {"jpeg", "png", "webp", "bmp", "gif", "tiff"},
            "vector": {"svg"},
            "data": {"json"}
        }
        
        return detected_format in compatibility_map.get(expected_type, set())
    
    async def _validate_file_headers(self, data: bytes) -> List[str]:
        """Validate file headers for consistency"""
        issues = []
        
        try:
            # Check for embedded scripts in image files
            if data.startswith(b'\xFF\xD8\xFF') or data.startswith(b'\x89PNG'):
                # Look for script tags
                if b'<script' in data or b'javascript:' in data:
                    issues.append("Suspicious script content detected in image")
            
        except Exception as e:
            logger.error(f"Error validating headers: {e}")
            issues.append(f"Header validation error: {e}")
        
        return issues
    
    async def _check_embedded_content(self, data: bytes) -> List[str]:
        """Check for potentially dangerous embedded content"""
        warnings = []
        
        try:
            # Look for common exploit patterns
            dangerous_patterns = [
                b'javascript:', b'vbscript:', b'data:text/html',
                b'<iframe', b'<object', b'<embed'
            ]
            
            for pattern in dangerous_patterns:
                if pattern in data:
                    warnings.append(f"Potentially dangerous pattern found: {pattern.decode('utf-8', errors='ignore')}")
            
        except Exception as e:
            logger.error(f"Error checking embedded content: {e}")
            warnings.append(f"Embedded content check error: {e}")
        
        return warnings
    
    async def _scan_for_malware(self, data: bytes) -> Dict[str, Any]:
        """Basic malware scanning"""
        try:
            # Simple heuristic-based scanning
            # In production, integrate with proper antivirus API
            
            suspicious_strings = [
                b'eval(', b'exec(', b'system(', b'shell_exec(',
                b'base64_decode', b'gzinflate', b'str_rot13'
            ]
            
            threats_found = []
            for pattern in suspicious_strings:
                if pattern in data:
                    threats_found.append(pattern.decode('utf-8', errors='ignore'))
            
            return {
                "clean": len(threats_found) == 0,
                "threats": threats_found,
                "scan_engine": "basic_heuristic"
            }
            
        except Exception as e:
            logger.error(f"Error in malware scan: {e}")
            return {"clean": True, "error": str(e)}
    
    async def _validate_image_format(self, data: bytes) -> List[str]:
        """Validate image format specifics"""
        issues = []
        
        try:
            image = Image.open(io.BytesIO(data))
            
            # Check for common image issues
            if image.mode not in ['RGB', 'RGBA', 'L', 'LA', 'P']:
                issues.append(f"Unsupported image mode: {image.mode}")
            
            # Check for excessive dimensions
            width, height = image.size
            if width * height > 100_000_000:  # 100MP
                issues.append("Image resolution too high")
            
        except Exception as e:
            issues.append(f"Image format validation error: {e}")
        
        return issues
    
    async def _validate_svg_format(self, data: bytes) -> List[str]:
        """Validate SVG format"""
        issues = []
        
        try:
            content = data.decode('utf-8')
            
            # Basic SVG structure check
            if '<svg' not in content:
                issues.append("Invalid SVG: missing svg element")
            
            # Check for dangerous elements
            dangerous_elements = ['script', 'object', 'embed', 'foreignObject']
            for element in dangerous_elements:
                if f'<{element}' in content:
                    issues.append(f"Dangerous SVG element found: {element}")
            
        except Exception as e:
            issues.append(f"SVG validation error: {e}")
        
        return issues
    
    async def _validate_json_format(self, data: bytes) -> List[str]:
        """Validate JSON format"""
        issues = []
        
        try:
            content = data.decode('utf-8')
            json.loads(content)  # Validate JSON structure
            
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON: {e}")
        except Exception as e:
            issues.append(f"JSON validation error: {e}")
        
        return issues
    
    async def _validate_image_content(self, data: bytes) -> List[str]:
        """Validate image content"""
        issues = []
        
        try:
            image = Image.open(io.BytesIO(data))
            
            # Check for corruption
            image.verify()
            
            # Reopen for further checks (verify() closes the image)
            image = Image.open(io.BytesIO(data))
            
            # Check image properties
            if image.size == (0, 0):
                issues.append("Image has zero dimensions")
            
        except Exception as e:
            issues.append(f"Image content validation error: {e}")
        
        return issues
    
    async def _validate_svg_content(self, data: bytes) -> List[str]:
        """Validate SVG content"""
        issues = []
        
        try:
            content = data.decode('utf-8')
            
            # Check for external references
            if 'xlink:href=' in content and 'http' in content:
                issues.append("SVG contains external references")
            
        except Exception as e:
            issues.append(f"SVG content validation error: {e}")
        
        return issues
    
    async def _validate_json_content(self, data: bytes) -> List[str]:
        """Validate JSON content"""
        issues = []
        
        try:
            content = data.decode('utf-8')
            parsed = json.loads(content)
            
            # Check for reasonable size
            if len(str(parsed)) > 1_000_000:  # 1MB when stringified
                issues.append("JSON content too large")
            
        except Exception as e:
            issues.append(f"JSON content validation error: {e}")
        
        return issues
    
    async def _validate_image_specific(self, data: bytes, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Image-specific validation checks"""
        try:
            issues = []
            
            image = Image.open(io.BytesIO(data))
            width, height = image.size
            
            # Check specific requirements
            if "min_width" in requirements and width < requirements["min_width"]:
                issues.append(f"Width {width} < required {requirements['min_width']}")
            
            if "min_height" in requirements and height < requirements["min_height"]:
                issues.append(f"Height {height} < required {requirements['min_height']}")
            
            if "max_width" in requirements and width > requirements["max_width"]:
                issues.append(f"Width {width} > allowed {requirements['max_width']}")
            
            if "max_height" in requirements and height > requirements["max_height"]:
                issues.append(f"Height {height} > allowed {requirements['max_height']}")
            
            if "aspect_ratio" in requirements:
                actual_ratio = width / height
                required_ratio = requirements["aspect_ratio"]
                tolerance = requirements.get("aspect_ratio_tolerance", 0.1)
                
                if abs(actual_ratio - required_ratio) > tolerance:
                    issues.append(f"Aspect ratio {actual_ratio:.2f} doesn't match required {required_ratio:.2f}")
            
            return {
                "passed": len(issues) == 0,
                "issues": issues,
                "image_properties": {
                    "width": width,
                    "height": height,
                    "mode": image.mode,
                    "format": image.format
                }
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def _validate_logo_specific(self, data: bytes, brand_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Logo-specific validation"""
        try:
            issues = []
            
            image = Image.open(io.BytesIO(data))
            
            # Logo should ideally have transparency
            if "require_transparency" in brand_requirements and brand_requirements["require_transparency"]:
                if image.mode not in ('RGBA', 'LA', 'P'):
                    issues.append("Logo should have transparency support")
            
            # Check for square/reasonable aspect ratio
            width, height = image.size
            aspect_ratio = width / height
            
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                issues.append(f"Unusual logo aspect ratio: {aspect_ratio:.2f}")
            
            # Minimum resolution for logo quality
            min_logo_size = min(width, height)
            if min_logo_size < 64:
                issues.append(f"Logo resolution too low: {min_logo_size}px")
            
            return {
                "passed": len(issues) == 0,
                "issues": issues,
                "logo_properties": {
                    "dimensions": (width, height),
                    "aspect_ratio": aspect_ratio,
                    "has_transparency": image.mode in ('RGBA', 'LA', 'P')
                }
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def _run_additional_check(self, check_name: str, data: bytes, filename: str) -> Dict[str, Any]:
        """Run additional custom validation check"""
        try:
            if check_name == "duplicate_detection":
                return await self._check_for_duplicates(data)
            elif check_name == "metadata_extraction":
                return await self._extract_metadata(data)
            elif check_name == "content_analysis":
                return await self._analyze_content(data)
            else:
                return {"passed": False, "error": f"Unknown check: {check_name}"}
                
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def _check_for_duplicates(self, data: bytes) -> Dict[str, Any]:
        """Check for duplicate content"""
        try:
            content_hash = hashlib.sha256(data).hexdigest()
            
            # In a real implementation, this would check against a database
            # of known file hashes
            
            return {
                "passed": True,
                "content_hash": content_hash,
                "is_duplicate": False  # Placeholder
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def _extract_metadata(self, data: bytes) -> Dict[str, Any]:
        """Extract file metadata"""
        try:
            metadata = {
                "file_size": len(data),
                "content_hash": hashlib.sha256(data).hexdigest(),
                "extracted_at": datetime.now().isoformat()
            }
            
            # Try to extract image metadata
            try:
                image = Image.open(io.BytesIO(data))
                metadata["image_info"] = {
                    "format": image.format,
                    "mode": image.mode,
                    "size": image.size
                }
                
                if hasattr(image, '_getexif') and image._getexif():
                    metadata["exif_present"] = True
                
            except Exception:
                pass  # Not an image or no EXIF data
            
            return {
                "passed": True,
                "metadata": metadata
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def _analyze_content(self, data: bytes) -> Dict[str, Any]:
        """Analyze file content characteristics"""
        try:
            analysis = {
                "entropy": self._calculate_entropy(data),
                "compression_ratio": self._estimate_compression_ratio(data),
                "binary_percentage": self._calculate_binary_percentage(data)
            }
            
            return {
                "passed": True,
                "analysis": analysis
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        try:
            if not data:
                return 0.0
            
            # Count byte frequencies
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1
            
            # Calculate entropy
            entropy = 0.0
            data_len = len(data)
            
            for count in byte_counts:
                if count > 0:
                    probability = count / data_len
                    entropy -= probability * np.log2(probability)
            
            return round(entropy, 3)
            
        except Exception:
            return 0.0
    
    def _estimate_compression_ratio(self, data: bytes) -> float:
        """Estimate how well data compresses"""
        try:
            import zlib
            compressed = zlib.compress(data)
            ratio = len(compressed) / len(data) if len(data) > 0 else 1.0
            return round(ratio, 3)
        except Exception:
            return 1.0
    
    def _calculate_binary_percentage(self, data: bytes) -> float:
        """Calculate percentage of non-printable bytes"""
        try:
            if not data:
                return 0.0
            
            printable_count = sum(1 for byte in data if 32 <= byte <= 126)
            binary_percentage = (1 - printable_count / len(data)) * 100
            return round(binary_percentage, 2)
            
        except Exception:
            return 0.0
