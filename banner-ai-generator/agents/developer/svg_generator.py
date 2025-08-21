"""
SVG Generator

Generates optimized SVG code from design blueprints
with full support for interactive elements and animations.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import xml.etree.ElementTree as ET
import re
from structlog import get_logger

logger = get_logger(__name__)


class SVGGenerator:
    """
    SVG code generation system
    
    Capabilities:
    - Scalable vector graphics generation
    - Interactive element support
    - Animation and transition integration
    - Optimized code output
    - Cross-browser compatibility
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # SVG configuration
        self.svg_version = config.get("svg_version", "1.1")
        self.namespace = config.get("namespace", "http://www.w3.org/2000/svg")
        self.optimization_level = config.get("optimization_level", "standard")
        
        # Code generation settings
        self.indent_size = config.get("indent_size", 2)
        self.minify_output = config.get("minify_output", False)
        self.include_comments = config.get("include_comments", True)
        
        # Communication
        self.shared_memory = None
        self.message_queue = None
        
        logger.info("SVG Generator initialized")
    
    async def initialize(self):
        """Initialize the SVG generator"""
        try:
            # Load SVG templates and optimizations
            await self._load_svg_templates()
            await self._initialize_optimizations()
            
            logger.info("SVG Generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SVG Generator: {e}")
            raise
    
    def set_communication(self, shared_memory, message_queue):
        """Set communication interfaces"""
        self.shared_memory = shared_memory
        self.message_queue = message_queue
    
    async def generate_svg(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SVG code from design blueprint
        
        Args:
            blueprint: Complete design blueprint
            
        Returns:
            SVG generation result with code and metadata
        """
        try:
            logger.info("Starting SVG generation")
            
            # Extract blueprint data
            structure = blueprint.get("structure", {})
            components = blueprint.get("components", {})
            styling = blueprint.get("styling", {})
            responsive = blueprint.get("responsive", {})
            
            # Create SVG document structure
            svg_document = await self._create_svg_document(structure)
            
            # Add component definitions
            defs_element = await self._create_definitions(components, styling)
            if defs_element:
                svg_document.append(defs_element)
            
            # Add background elements
            background_elements = await self._create_background_elements(structure, components)
            for element in background_elements:
                svg_document.append(element)
            
            # Add component elements
            component_elements = await self._create_component_elements(components, styling)
            for element in component_elements:
                svg_document.append(element)
            
            # Add responsive and interactive features
            interactive_elements = await self._add_interactive_features(components, svg_document)
            
            # Generate final SVG code
            svg_code = await self._generate_svg_code(svg_document)
            
            # Optimize SVG code
            optimized_svg = await self._optimize_svg_code(svg_code)
            
            # Generate metadata
            metadata = await self._generate_svg_metadata(blueprint, optimized_svg)
            
            result = {
                "success": True,
                "format": "svg",
                "svg_code": optimized_svg,
                "metadata": metadata,
                "file_size": len(optimized_svg.encode('utf-8')),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info("SVG generation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating SVG: {e}")
            return {
                "success": False,
                "error": str(e),
                "format": "svg"
            }
    
    async def _create_svg_document(self, structure: Dict[str, Any]) -> ET.Element:
        """Create the root SVG document element"""
        try:
            document = structure.get("document", {})
            dimensions = document.get("dimensions", {"width": 800, "height": 600})
            
            # Create root SVG element
            svg = ET.Element("svg")
            svg.set("xmlns", self.namespace)
            svg.set("version", self.svg_version)
            svg.set("width", str(dimensions["width"]))
            svg.set("height", str(dimensions["height"]))
            svg.set("viewBox", f"0 0 {dimensions['width']} {dimensions['height']}")
            
            # Add metadata
            title = ET.SubElement(svg, "title")
            title.text = "Generated Banner Advertisement"
            
            desc = ET.SubElement(svg, "desc")
            desc.text = f"AI-generated banner created at {datetime.utcnow().isoformat()}"
            
            return svg
            
        except Exception as e:
            logger.error(f"Error creating SVG document: {e}")
            raise
    
    async def _create_definitions(self, components: Dict[str, Any], 
                                styling: Dict[str, Any]) -> Optional[ET.Element]:
        """Create SVG definitions for reusable elements"""
        try:
            defs = ET.Element("defs")
            has_definitions = False
            
            # Create gradients
            gradients = await self._create_gradients(styling)
            for gradient in gradients:
                defs.append(gradient)
                has_definitions = True
            
            # Create filters
            filters = await self._create_filters(styling)
            for filter_elem in filters:
                defs.append(filter_elem)
                has_definitions = True
            
            # Create patterns
            patterns = await self._create_patterns(components)
            for pattern in patterns:
                defs.append(pattern)
                has_definitions = True
            
            # Create styles
            styles = await self._create_embedded_styles(styling)
            if styles:
                defs.append(styles)
                has_definitions = True
            
            return defs if has_definitions else None
            
        except Exception as e:
            logger.error(f"Error creating definitions: {e}")
            return None
    
    async def _create_gradients(self, styling: Dict[str, Any]) -> List[ET.Element]:
        """Create gradient definitions"""
        gradients = []
        
        try:
            # Example gradient for buttons
            linear_gradient = ET.Element("linearGradient")
            linear_gradient.set("id", "buttonGradient")
            linear_gradient.set("x1", "0%")
            linear_gradient.set("y1", "0%")
            linear_gradient.set("x2", "0%")
            linear_gradient.set("y2", "100%")
            
            stop1 = ET.SubElement(linear_gradient, "stop")
            stop1.set("offset", "0%")
            stop1.set("style", "stop-color:#4A90E2;stop-opacity:1")
            
            stop2 = ET.SubElement(linear_gradient, "stop")
            stop2.set("offset", "100%")
            stop2.set("style", "stop-color:#357ABD;stop-opacity:1")
            
            gradients.append(linear_gradient)
            
        except Exception as e:
            logger.error(f"Error creating gradients: {e}")
        
        return gradients
    
    async def _create_filters(self, styling: Dict[str, Any]) -> List[ET.Element]:
        """Create filter definitions for effects"""
        filters = []
        
        try:
            # Drop shadow filter
            shadow_filter = ET.Element("filter")
            shadow_filter.set("id", "dropShadow")
            shadow_filter.set("x", "-50%")
            shadow_filter.set("y", "-50%")
            shadow_filter.set("width", "200%")
            shadow_filter.set("height", "200%")
            
            # Gaussian blur
            blur = ET.SubElement(shadow_filter, "feGaussianBlur")
            blur.set("in", "SourceAlpha")
            blur.set("stdDeviation", "3")
            blur.set("result", "blur")
            
            # Offset
            offset = ET.SubElement(shadow_filter, "feOffset")
            offset.set("in", "blur")
            offset.set("dx", "2")
            offset.set("dy", "2")
            offset.set("result", "offsetBlur")
            
            # Merge
            merge = ET.SubElement(shadow_filter, "feMerge")
            merge_node1 = ET.SubElement(merge, "feMergeNode")
            merge_node1.set("in", "offsetBlur")
            merge_node2 = ET.SubElement(merge, "feMergeNode")
            merge_node2.set("in", "SourceGraphic")
            
            filters.append(shadow_filter)
            
        except Exception as e:
            logger.error(f"Error creating filters: {e}")
        
        return filters
    
    async def _create_patterns(self, components: Dict[str, Any]) -> List[ET.Element]:
        """Create pattern definitions"""
        patterns = []
        
        # For now, return empty list
        # In production, this would create patterns for textures, etc.
        
        return patterns
    
    async def _create_embedded_styles(self, styling: Dict[str, Any]) -> Optional[ET.Element]:
        """Create embedded CSS styles"""
        try:
            style_element = ET.Element("style")
            style_element.set("type", "text/css")
            
            css_rules = []
            
            # Global styles
            global_styles = styling.get("global_styles", {})
            if global_styles:
                css_rules.append(".svg-text { font-family: 'Inter', sans-serif; }")
                css_rules.append(".svg-button { cursor: pointer; }")
                css_rules.append(".svg-interactive:hover { opacity: 0.8; }")
            
            # Animation styles
            css_rules.extend([
                "@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }",
                ".fade-in { animation: fadeIn 0.6s ease-out; }",
                ".hover-scale:hover { transform: scale(1.05); transition: transform 0.3s ease; }"
            ])
            
            if css_rules:
                style_element.text = "\n".join(css_rules)
                return style_element
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating embedded styles: {e}")
            return None
    
    async def _create_background_elements(self, structure: Dict[str, Any], 
                                        components: Dict[str, Any]) -> List[ET.Element]:
        """Create background elements"""
        background_elements = []
        
        try:
            canvas = structure.get("canvas", {})
            background = canvas.get("background", {})
            
            if background.get("type") == "image" and background.get("source"):
                # Background image
                bg_image = ET.Element("image")
                bg_image.set("href", background["source"])
                bg_image.set("x", "0")
                bg_image.set("y", "0")
                bg_image.set("width", "100%")
                bg_image.set("height", "100%")
                bg_image.set("preserveAspectRatio", "xMidYMid slice")
                background_elements.append(bg_image)
            
            elif background.get("fallback_color"):
                # Background color
                bg_rect = ET.Element("rect")
                bg_rect.set("x", "0")
                bg_rect.set("y", "0")
                bg_rect.set("width", "100%")
                bg_rect.set("height", "100%")
                bg_rect.set("fill", background["fallback_color"])
                background_elements.append(bg_rect)
            
        except Exception as e:
            logger.error(f"Error creating background elements: {e}")
        
        return background_elements
    
    async def _create_component_elements(self, components: Dict[str, Any], 
                                       styling: Dict[str, Any]) -> List[ET.Element]:
        """Create SVG elements for each component"""
        elements = []
        
        try:
            # Sort components by z-index
            sorted_components = sorted(
                components.items(),
                key=lambda x: x[1].get("position", {}).get("z_index", 1)
            )
            
            for comp_id, component in sorted_components:
                element = await self._create_component_element(comp_id, component, styling)
                if element is not None:
                    elements.append(element)
            
        except Exception as e:
            logger.error(f"Error creating component elements: {e}")
        
        return elements
    
    async def _create_component_element(self, comp_id: str, component: Dict[str, Any], 
                                      styling: Dict[str, Any]) -> Optional[ET.Element]:
        """Create SVG element for a single component"""
        try:
            comp_type = component.get("type")
            position = component.get("position", {})
            dimensions = component.get("dimensions", {})
            content = component.get("content", {})
            comp_styling = component.get("styling", {})
            
            if comp_type == "text":
                return await self._create_text_element(comp_id, component)
            elif comp_type == "button":
                return await self._create_button_element(comp_id, component)
            elif comp_type == "logo":
                return await self._create_logo_element(comp_id, component)
            elif comp_type == "background_image":
                return await self._create_image_element(comp_id, component)
            else:
                logger.warning(f"Unknown component type: {comp_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating component element {comp_id}: {e}")
            return None
    
    async def _create_text_element(self, comp_id: str, component: Dict[str, Any]) -> ET.Element:
        """Create SVG text element"""
        position = component.get("position", {})
        dimensions = component.get("dimensions", {})
        content = component.get("content", {})
        styling = component.get("styling", {}).get("base", {})
        
        # Create text element
        text_elem = ET.Element("text")
        text_elem.set("id", comp_id)
        text_elem.set("x", str(position.get("x", 0)))
        text_elem.set("y", str(position.get("y", 0) + dimensions.get("height", 20)))  # Baseline adjustment
        text_elem.set("class", "svg-text")
        
        # Apply styling
        if styling.get("font_family"):
            text_elem.set("font-family", styling["font_family"])
        if styling.get("font_size"):
            text_elem.set("font-size", styling["font_size"])
        if styling.get("font_weight"):
            text_elem.set("font-weight", styling["font_weight"])
        if styling.get("color"):
            text_elem.set("fill", styling["color"])
        if styling.get("text_align"):
            text_anchor = {"left": "start", "center": "middle", "right": "end"}.get(
                styling["text_align"], "start"
            )
            text_elem.set("text-anchor", text_anchor)
            
            # Adjust x position for centering
            if text_anchor == "middle":
                text_elem.set("x", str(position.get("x", 0) + dimensions.get("width", 0) / 2))
            elif text_anchor == "end":
                text_elem.set("x", str(position.get("x", 0) + dimensions.get("width", 0)))
        
        # Set text content
        text_elem.text = content.get("text", "")
        
        return text_elem
    
    async def _create_button_element(self, comp_id: str, component: Dict[str, Any]) -> ET.Element:
        """Create SVG button element (group with rect and text)"""
        position = component.get("position", {})
        dimensions = component.get("dimensions", {})
        content = component.get("content", {})
        styling = component.get("styling", {}).get("base", {})
        
        # Create group element
        group = ET.Element("g")
        group.set("id", comp_id)
        group.set("class", "svg-button svg-interactive hover-scale")
        
        # Create button background
        rect = ET.SubElement(group, "rect")
        rect.set("x", str(position.get("x", 0)))
        rect.set("y", str(position.get("y", 0)))
        rect.set("width", str(dimensions.get("width", 100)))
        rect.set("height", str(dimensions.get("height", 40)))
        rect.set("rx", "4")  # Rounded corners
        
        # Apply button styling
        if styling.get("background_color"):
            rect.set("fill", styling["background_color"])
        else:
            rect.set("fill", "url(#buttonGradient)")
        
        if styling.get("border"):
            rect.set("stroke", "#000000")
            rect.set("stroke-width", "1")
        
        # Add shadow filter
        rect.set("filter", "url(#dropShadow)")
        
        # Create button text
        text = ET.SubElement(group, "text")
        text.set("x", str(position.get("x", 0) + dimensions.get("width", 100) / 2))
        text.set("y", str(position.get("y", 0) + dimensions.get("height", 40) / 2 + 5))  # Center vertically
        text.set("text-anchor", "middle")
        text.set("dominant-baseline", "middle")
        text.set("font-family", styling.get("font_family", "Inter, sans-serif"))
        text.set("font-size", "14")
        text.set("font-weight", styling.get("font_weight", "semibold"))
        text.set("fill", styling.get("color", "#ffffff"))
        text.text = content.get("text", "")
        
        return group
    
    async def _create_logo_element(self, comp_id: str, component: Dict[str, Any]) -> ET.Element:
        """Create SVG logo element"""
        position = component.get("position", {})
        dimensions = component.get("dimensions", {})
        content = component.get("content", {})
        
        # Create image element
        image = ET.Element("image")
        image.set("id", comp_id)
        image.set("x", str(position.get("x", 0)))
        image.set("y", str(position.get("y", 0)))
        image.set("width", str(dimensions.get("width", 100)))
        image.set("height", str(dimensions.get("height", 60)))
        image.set("preserveAspectRatio", "xMidYMid meet")
        image.set("class", "fade-in")
        
        # Set image source
        if content.get("source"):
            image.set("href", content["source"])
        
        return image
    
    async def _create_image_element(self, comp_id: str, component: Dict[str, Any]) -> ET.Element:
        """Create SVG image element"""
        position = component.get("position", {})
        dimensions = component.get("dimensions", {})
        content = component.get("content", {})
        
        # Create image element
        image = ET.Element("image")
        image.set("id", comp_id)
        image.set("x", str(position.get("x", 0)))
        image.set("y", str(position.get("y", 0)))
        image.set("width", str(dimensions.get("width", 200)))
        image.set("height", str(dimensions.get("height", 150)))
        image.set("preserveAspectRatio", "xMidYMid slice")
        
        # Set image source
        if content.get("source"):
            image.set("href", content["source"])
        
        return image
    
    async def _add_interactive_features(self, components: Dict[str, Any], 
                                      svg_document: ET.Element) -> List[ET.Element]:
        """Add interactive features and animations"""
        interactive_elements = []
        
        try:
            # Add JavaScript for interactivity (if needed)
            script_content = []
            
            for comp_id, component in components.items():
                behavior = component.get("behavior", {})
                
                if behavior.get("interactive") and component.get("type") == "button":
                    # Add click handler
                    script_content.append(f"""
                        document.getElementById('{comp_id}').addEventListener('click', function() {{
                            console.log('Button {comp_id} clicked');
                            // Add custom click behavior here
                        }});
                    """)
            
            if script_content:
                script = ET.Element("script")
                script.set("type", "text/javascript")
                script.text = "\n".join(script_content)
                interactive_elements.append(script)
            
        except Exception as e:
            logger.error(f"Error adding interactive features: {e}")
        
        return interactive_elements
    
    async def _generate_svg_code(self, svg_element: ET.Element) -> str:
        """Generate final SVG code string"""
        try:
            # Convert to string
            rough_string = ET.tostring(svg_element, encoding='unicode')
            
            # Add XML declaration
            svg_code = '<?xml version="1.0" encoding="UTF-8"?>\n' + rough_string
            
            # Pretty format if not minifying
            if not self.minify_output:
                svg_code = self._format_svg_code(svg_code)
            
            return svg_code
            
        except Exception as e:
            logger.error(f"Error generating SVG code: {e}")
            return ""
    
    def _format_svg_code(self, svg_code: str) -> str:
        """Format SVG code with proper indentation"""
        try:
            # Basic formatting - in production, use proper XML formatter
            lines = svg_code.split('\n')
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                
                # Decrease indent for closing tags
                if stripped.startswith('</'):
                    indent_level = max(0, indent_level - 1)
                
                # Add indentation
                formatted_lines.append('  ' * indent_level + stripped)
                
                # Increase indent for opening tags (but not self-closing)
                if stripped.startswith('<') and not stripped.startswith('</') and not stripped.endswith('/>'):
                    indent_level += 1
            
            return '\n'.join(formatted_lines)
            
        except Exception as e:
            logger.error(f"Error formatting SVG code: {e}")
            return svg_code
    
    async def _optimize_svg_code(self, svg_code: str) -> str:
        """Optimize SVG code for size and performance"""
        try:
            optimized = svg_code
            
            if self.optimization_level in ["standard", "aggressive"]:
                # Remove unnecessary whitespace
                optimized = re.sub(r'\s+', ' ', optimized)
                
                # Remove comments if not needed
                if not self.include_comments:
                    optimized = re.sub(r'<!--.*?-->', '', optimized, flags=re.DOTALL)
                
                # Round numeric values to reduce precision
                optimized = re.sub(r'(\d+\.\d{3,})', lambda m: f"{float(m.group(1)):.2f}", optimized)
            
            if self.optimization_level == "aggressive":
                # Remove unnecessary attributes
                optimized = re.sub(r'\s+xmlns:xlink="[^"]*"', '', optimized)
                
                # Minify if requested
                if self.minify_output:
                    optimized = optimized.replace('\n', '').replace('  ', ' ')
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing SVG code: {e}")
            return svg_code
    
    async def _generate_svg_metadata(self, blueprint: Dict[str, Any], 
                                   svg_code: str) -> Dict[str, Any]:
        """Generate metadata about the SVG"""
        try:
            structure = blueprint.get("structure", {})
            components = blueprint.get("components", {})
            
            metadata = {
                "format": "svg",
                "version": self.svg_version,
                "dimensions": structure.get("document", {}).get("dimensions", {}),
                "file_size_bytes": len(svg_code.encode('utf-8')),
                "component_count": len(components),
                "interactive_elements": len([
                    c for c in components.values() 
                    if c.get("behavior", {}).get("interactive")
                ]),
                "optimization_level": self.optimization_level,
                "minified": self.minify_output,
                "features": {
                    "animations": "fadeIn" in svg_code,
                    "filters": "dropShadow" in svg_code,
                    "gradients": "buttonGradient" in svg_code,
                    "interactivity": "addEventListener" in svg_code
                }
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating SVG metadata: {e}")
            return {"format": "svg", "error": str(e)}
    
    async def _load_svg_templates(self):
        """Load SVG templates and components"""
        # This would load predefined SVG templates
        pass
    
    async def _initialize_optimizations(self):
        """Initialize SVG optimization tools"""
        # This would initialize SVG optimization libraries
        pass
