"""
SVG Handler

Specialized handler for SVG files including parsing, validation,
optimization, and conversion capabilities.
"""

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
import base64
from structlog import get_logger

logger = get_logger(__name__)


class SVGHandler:
    """
    Comprehensive SVG file handler
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # SVG processing options
        self.remove_comments = config.get("remove_comments", True)
        self.remove_metadata = config.get("remove_metadata", True)
        self.optimize_paths = config.get("optimize_paths", True)
        self.remove_unused_defs = config.get("remove_unused_defs", True)
        
        # Security settings
        self.remove_scripts = config.get("remove_scripts", True)
        self.remove_external_refs = config.get("remove_external_refs", True)
        self.sanitize_attributes = config.get("sanitize_attributes", True)
        
        # Namespace handling
        self.svg_namespace = "http://www.w3.org/2000/svg"
        self.xlink_namespace = "http://www.w3.org/1999/xlink"
    
    async def process_svg(self, 
                         svg_data: str,
                         options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process SVG file with optimization and validation
        
        Args:
            svg_data: SVG content as string
            options: Processing options
        
        Returns:
            Processing result with optimized SVG and metadata
        """
        try:
            logger.info("Processing SVG file")
            
            # Parse SVG
            root, metadata = await self._parse_svg(svg_data)
            
            # Security sanitization
            if self.remove_scripts:
                root = await self._remove_scripts(root)
            
            if self.remove_external_refs:
                root = await self._remove_external_references(root)
            
            if self.sanitize_attributes:
                root = await self._sanitize_attributes(root)
            
            # Optimization
            if self.remove_comments:
                root = await self._remove_comments(root)
            
            if self.remove_metadata:
                root = await self._remove_metadata_elements(root)
            
            if self.optimize_paths:
                root = await self._optimize_paths(root)
            
            if self.remove_unused_defs:
                root = await self._remove_unused_definitions(root)
            
            # Generate optimized SVG
            optimized_svg = await self._serialize_svg(root)
            
            # Calculate optimization stats
            original_size = len(svg_data.encode('utf-8'))
            optimized_size = len(optimized_svg.encode('utf-8'))
            compression_ratio = (original_size - optimized_size) / original_size * 100
            
            result = {
                "success": True,
                "original_svg": svg_data,
                "optimized_svg": optimized_svg,
                "metadata": metadata,
                "optimization_stats": {
                    "original_size": original_size,
                    "optimized_size": optimized_size,
                    "compression_ratio": round(compression_ratio, 2),
                    "size_reduction": original_size - optimized_size
                }
            }
            
            logger.info(f"SVG processing completed: {compression_ratio:.1f}% size reduction")
            return result
            
        except Exception as e:
            logger.error(f"Error processing SVG: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_svg": svg_data
            }
    
    async def convert_to_png(self, 
                           svg_data: str,
                           width: int = 1024,
                           height: int = 1024,
                           background_color: str = "transparent") -> str:
        """
        Convert SVG to PNG format
        
        Args:
            svg_data: SVG content
            width: Output width
            height: Output height
            background_color: Background color
        
        Returns:
            Base64 encoded PNG data
        """
        try:
            # This would integrate with a library like cairosvg or wand
            # For now, return a placeholder
            logger.warning("SVG to PNG conversion not fully implemented")
            
            # Placeholder implementation
            from PIL import Image
            import io
            
            # Create placeholder image
            image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
            
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            png_data = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{png_data}"
            
        except Exception as e:
            logger.error(f"Error converting SVG to PNG: {e}")
            raise
    
    async def extract_metadata(self, svg_data: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from SVG
        
        Args:
            svg_data: SVG content
        
        Returns:
            Extracted metadata
        """
        try:
            root, metadata = await self._parse_svg(svg_data)
            
            # Extract additional metadata
            metadata.update({
                "elements_count": await self._count_elements(root),
                "text_content": await self._extract_text_content(root),
                "colors_used": await self._extract_colors(root),
                "has_animations": await self._has_animations(root),
                "has_scripts": await self._has_scripts(root),
                "external_references": await self._find_external_references(root)
            })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting SVG metadata: {e}")
            return {"error": str(e)}
    
    async def validate_svg(self, svg_data: str) -> Dict[str, Any]:
        """
        Validate SVG structure and content
        
        Args:
            svg_data: SVG content
        
        Returns:
            Validation result
        """
        try:
            issues = []
            warnings = []
            
            # Basic structure validation
            try:
                root, _ = await self._parse_svg(svg_data)
            except Exception as e:
                issues.append(f"Invalid SVG structure: {e}")
                return {"valid": False, "issues": issues}
            
            # Check for required elements
            if root.tag != f"{{{self.svg_namespace}}}svg":
                issues.append("Root element is not <svg>")
            
            # Check for dangerous elements
            dangerous_elements = await self._find_dangerous_elements(root)
            if dangerous_elements:
                issues.extend([f"Dangerous element found: {elem}" for elem in dangerous_elements])
            
            # Check viewBox and dimensions
            viewbox_warnings = await self._validate_viewbox(root)
            warnings.extend(viewbox_warnings)
            
            # Check for accessibility
            accessibility_warnings = await self._check_accessibility(root)
            warnings.extend(accessibility_warnings)
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "element_count": await self._count_elements(root)
            }
            
        except Exception as e:
            logger.error(f"Error validating SVG: {e}")
            return {"valid": False, "error": str(e)}
    
    async def _parse_svg(self, svg_data: str) -> Tuple[ET.Element, Dict[str, Any]]:
        """Parse SVG and extract basic metadata"""
        try:
            # Register namespaces
            ET.register_namespace('', self.svg_namespace)
            ET.register_namespace('xlink', self.xlink_namespace)
            
            # Parse XML
            root = ET.fromstring(svg_data)
            
            # Extract basic metadata
            metadata = {
                "xmlns": root.get("xmlns", ""),
                "width": root.get("width", ""),
                "height": root.get("height", ""),
                "viewBox": root.get("viewBox", ""),
                "version": root.get("version", ""),
                "xml_lang": root.get("{http://www.w3.org/XML/1998/namespace}lang", "")
            }
            
            # Parse viewBox if present
            if metadata["viewBox"]:
                try:
                    viewbox_values = [float(x) for x in metadata["viewBox"].split()]
                    metadata["viewBox_parsed"] = {
                        "x": viewbox_values[0],
                        "y": viewbox_values[1], 
                        "width": viewbox_values[2],
                        "height": viewbox_values[3]
                    }
                except (ValueError, IndexError):
                    metadata["viewBox_error"] = "Invalid viewBox format"
            
            return root, metadata
            
        except ET.ParseError as e:
            logger.error(f"SVG parsing error: {e}")
            raise ValueError(f"Invalid SVG: {e}")
        except Exception as e:
            logger.error(f"Error parsing SVG: {e}")
            raise
    
    async def _remove_scripts(self, root: ET.Element) -> ET.Element:
        """Remove script elements from SVG"""
        try:
            # Find all script elements
            scripts = root.findall(".//{http://www.w3.org/2000/svg}script")
            
            for script in scripts:
                parent = root.find(f".//*[{script}]/..")
                if parent is not None:
                    parent.remove(script)
            
            return root
            
        except Exception as e:
            logger.error(f"Error removing scripts: {e}")
            return root
    
    async def _remove_external_references(self, root: ET.Element) -> ET.Element:
        """Remove external references from SVG"""
        try:
            # Remove external image references
            for elem in root.iter():
                href = elem.get(f"{{{self.xlink_namespace}}}href")
                if href and (href.startswith("http") or href.startswith("//") or href.startswith("data:")):
                    elem.set(f"{{{self.xlink_namespace}}}href", "")
                
                # Also check 'href' attribute (SVG 2.0)
                href = elem.get("href")
                if href and (href.startswith("http") or href.startswith("//") or href.startswith("data:")):
                    elem.set("href", "")
            
            return root
            
        except Exception as e:
            logger.error(f"Error removing external references: {e}")
            return root
    
    async def _sanitize_attributes(self, root: ET.Element) -> ET.Element:
        """Sanitize SVG attributes"""
        try:
            dangerous_attributes = {
                "onload", "onclick", "onmouseover", "onmouseout", "onfocus", "onblur"
            }
            
            for elem in root.iter():
                # Remove dangerous event attributes
                for attr in list(elem.attrib.keys()):
                    if attr.lower() in dangerous_attributes:
                        del elem.attrib[attr]
                
                # Sanitize style attributes
                style = elem.get("style")
                if style:
                    elem.set("style", await self._sanitize_style(style))
            
            return root
            
        except Exception as e:
            logger.error(f"Error sanitizing attributes: {e}")
            return root
    
    async def _sanitize_style(self, style: str) -> str:
        """Sanitize CSS style string"""
        try:
            # Remove dangerous CSS functions
            dangerous_patterns = [
                r'expression\s*\(',
                r'javascript\s*:',
                r'@import\s+',
                r'url\s*\(\s*["\']?javascript:'
            ]
            
            sanitized = style
            for pattern in dangerous_patterns:
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing style: {e}")
            return style
    
    async def _remove_comments(self, root: ET.Element) -> ET.Element:
        """Remove XML comments"""
        # Note: ET doesn't preserve comments by default, so this is mostly a placeholder
        return root
    
    async def _remove_metadata_elements(self, root: ET.Element) -> ET.Element:
        """Remove metadata elements"""
        try:
            metadata_tags = ["metadata", "title", "desc"]
            
            for tag in metadata_tags:
                elements = root.findall(f".//{{{self.svg_namespace}}}{tag}")
                for elem in elements:
                    parent = elem.getparent()
                    if parent is not None:
                        parent.remove(elem)
            
            return root
            
        except Exception as e:
            logger.error(f"Error removing metadata elements: {e}")
            return root
    
    async def _optimize_paths(self, root: ET.Element) -> ET.Element:
        """Optimize SVG path elements"""
        try:
            # Find all path elements
            paths = root.findall(f".//{{{self.svg_namespace}}}path")
            
            for path in paths:
                d = path.get("d")
                if d:
                    # Simple path optimization (remove redundant spaces)
                    optimized_d = re.sub(r'\s+', ' ', d.strip())
                    path.set("d", optimized_d)
            
            return root
            
        except Exception as e:
            logger.error(f"Error optimizing paths: {e}")
            return root
    
    async def _remove_unused_definitions(self, root: ET.Element) -> ET.Element:
        """Remove unused definitions"""
        try:
            # Find defs element
            defs = root.find(f".//{{{self.svg_namespace}}}defs")
            if defs is None:
                return root
            
            # Get all IDs used in the document
            used_ids = set()
            for elem in root.iter():
                # Check various attributes that reference IDs
                for attr in ["fill", "stroke", "filter", "clip-path", "mask"]:
                    value = elem.get(attr, "")
                    if value.startswith("url(#"):
                        ref_id = value[5:-1]  # Remove "url(#" and ")"
                        used_ids.add(ref_id)
            
            # Remove unused definitions
            for def_elem in list(defs):
                elem_id = def_elem.get("id")
                if elem_id and elem_id not in used_ids:
                    defs.remove(def_elem)
            
            # Remove empty defs element
            if len(defs) == 0:
                parent = defs.getparent()
                if parent is not None:
                    parent.remove(defs)
            
            return root
            
        except Exception as e:
            logger.error(f"Error removing unused definitions: {e}")
            return root
    
    async def _serialize_svg(self, root: ET.Element) -> str:
        """Serialize SVG element back to string"""
        try:
            # Convert back to string
            ET.register_namespace('', self.svg_namespace)
            svg_string = ET.tostring(root, encoding='unicode', method='xml')
            
            # Add XML declaration if not present
            if not svg_string.startswith('<?xml'):
                svg_string = '<?xml version="1.0" encoding="UTF-8"?>\n' + svg_string
            
            return svg_string
            
        except Exception as e:
            logger.error(f"Error serializing SVG: {e}")
            raise
    
    async def _count_elements(self, root: ET.Element) -> Dict[str, int]:
        """Count different types of elements"""
        try:
            counts = {}
            
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                counts[tag] = counts.get(tag, 0) + 1
            
            return counts
            
        except Exception as e:
            logger.error(f"Error counting elements: {e}")
            return {}
    
    async def _extract_text_content(self, root: ET.Element) -> List[str]:
        """Extract all text content from SVG"""
        try:
            text_content = []
            
            # Find text and tspan elements
            for elem in root.iter():
                if elem.tag.endswith('text') or elem.tag.endswith('tspan'):
                    if elem.text:
                        text_content.append(elem.text.strip())
            
            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
            return []
    
    async def _extract_colors(self, root: ET.Element) -> List[str]:
        """Extract colors used in SVG"""
        try:
            colors = set()
            
            for elem in root.iter():
                # Check fill and stroke attributes
                for attr in ["fill", "stroke"]:
                    color = elem.get(attr)
                    if color and not color.startswith("url("):
                        colors.add(color)
                
                # Check style attribute
                style = elem.get("style", "")
                if style:
                    # Simple regex to find color values
                    color_matches = re.findall(r'(?:fill|stroke):\s*([^;]+)', style)
                    colors.update(color_matches)
            
            return list(colors)
            
        except Exception as e:
            logger.error(f"Error extracting colors: {e}")
            return []
    
    async def _has_animations(self, root: ET.Element) -> bool:
        """Check if SVG contains animations"""
        try:
            animation_elements = ["animate", "animateTransform", "animateMotion", "set"]
            
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if tag in animation_elements:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking animations: {e}")
            return False
    
    async def _has_scripts(self, root: ET.Element) -> bool:
        """Check if SVG contains scripts"""
        try:
            scripts = root.findall(f".//{{{self.svg_namespace}}}script")
            return len(scripts) > 0
            
        except Exception as e:
            logger.error(f"Error checking scripts: {e}")
            return False
    
    async def _find_external_references(self, root: ET.Element) -> List[str]:
        """Find external references in SVG"""
        try:
            external_refs = []
            
            for elem in root.iter():
                href = elem.get(f"{{{self.xlink_namespace}}}href") or elem.get("href")
                if href and (href.startswith("http") or href.startswith("//")):
                    external_refs.append(href)
            
            return external_refs
            
        except Exception as e:
            logger.error(f"Error finding external references: {e}")
            return []
    
    async def _find_dangerous_elements(self, root: ET.Element) -> List[str]:
        """Find potentially dangerous elements"""
        try:
            dangerous = []
            dangerous_tags = ["script", "object", "embed", "foreignObject"]
            
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if tag in dangerous_tags:
                    dangerous.append(tag)
            
            return dangerous
            
        except Exception as e:
            logger.error(f"Error finding dangerous elements: {e}")
            return []
    
    async def _validate_viewbox(self, root: ET.Element) -> List[str]:
        """Validate viewBox attribute"""
        warnings = []
        
        try:
            viewbox = root.get("viewBox")
            width = root.get("width")
            height = root.get("height")
            
            if not viewbox and (not width or not height):
                warnings.append("SVG should have either viewBox or width/height attributes")
            
            if viewbox:
                try:
                    values = [float(x) for x in viewbox.split()]
                    if len(values) != 4:
                        warnings.append("viewBox should have exactly 4 values")
                    elif values[2] <= 0 or values[3] <= 0:
                        warnings.append("viewBox width and height should be positive")
                except (ValueError, IndexError):
                    warnings.append("Invalid viewBox format")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Error validating viewBox: {e}")
            return ["viewBox validation error"]
    
    async def _check_accessibility(self, root: ET.Element) -> List[str]:
        """Check SVG accessibility"""
        warnings = []
        
        try:
            # Check for title element
            title = root.find(f".//{{{self.svg_namespace}}}title")
            if title is None:
                warnings.append("Consider adding a <title> element for accessibility")
            
            # Check for desc element
            desc = root.find(f".//{{{self.svg_namespace}}}desc")
            if desc is None:
                warnings.append("Consider adding a <desc> element for accessibility")
            
            # Check for role attribute
            role = root.get("role")
            if not role:
                warnings.append("Consider adding a 'role' attribute for accessibility")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Error checking accessibility: {e}")
            return ["Accessibility check error"]
