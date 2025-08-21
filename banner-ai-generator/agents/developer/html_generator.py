"""
HTML Generator

Generates responsive HTML/CSS code from design blueprints
with modern web standards and accessibility compliance.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
from structlog import get_logger

logger = get_logger(__name__)


class HTMLGenerator:
    """
    HTML/CSS code generation system
    
    Capabilities:
    - Semantic HTML5 generation
    - Responsive CSS with media queries
    - Accessibility compliance (WCAG 2.1)
    - Modern CSS features
    - Cross-browser compatibility
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # HTML configuration
        self.html_version = config.get("html_version", "html5")
        self.css_framework = config.get("css_framework", "custom")
        self.responsive_strategy = config.get("responsive_strategy", "mobile_first")
        
        # Code generation settings
        self.minify_output = config.get("minify_output", False)
        self.include_comments = config.get("include_comments", True)
        self.semantic_markup = config.get("semantic_markup", True)
        self.accessibility_features = config.get("accessibility_features", True)
        
        # Communication
        self.shared_memory = None
        self.message_queue = None
        
        logger.info("HTML Generator initialized")
    
    async def initialize(self):
        """Initialize the HTML generator"""
        try:
            # Load HTML templates and CSS frameworks
            await self._load_html_templates()
            await self._initialize_css_framework()
            
            logger.info("HTML Generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing HTML Generator: {e}")
            raise
    
    def set_communication(self, shared_memory, message_queue):
        """Set communication interfaces"""
        self.shared_memory = shared_memory
        self.message_queue = message_queue
    
    async def generate_html(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate HTML/CSS code from design blueprint
        
        Args:
            blueprint: Complete design blueprint
            
        Returns:
            HTML generation result with code and metadata
        """
        try:
            logger.info("Starting HTML generation")
            
            # Extract blueprint data
            structure = blueprint.get("structure", {})
            components = blueprint.get("components", {})
            styling = blueprint.get("styling", {})
            responsive = blueprint.get("responsive", {})
            interactions = blueprint.get("interactions", {})
            
            # Generate HTML structure
            html_content = await self._generate_html_structure(structure, components)
            
            # Generate CSS styles
            css_content = await self._generate_css_styles(styling, responsive, components)
            
            # Generate JavaScript (if needed)
            js_content = await self._generate_javascript(interactions, components)
            
            # Combine into complete HTML document
            complete_html = await self._create_complete_document(
                html_content, css_content, js_content, structure
            )
            
            # Optimize output
            optimized_html = await self._optimize_html_output(complete_html)
            
            # Generate metadata
            metadata = await self._generate_html_metadata(blueprint, optimized_html)
            
            result = {
                "success": True,
                "format": "html",
                "html_code": optimized_html,
                "html_content": html_content,
                "css_content": css_content,
                "js_content": js_content,
                "metadata": metadata,
                "file_size": len(optimized_html.encode('utf-8')),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info("HTML generation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating HTML: {e}")
            return {
                "success": False,
                "error": str(e),
                "format": "html"
            }
    
    async def _generate_html_structure(self, structure: Dict[str, Any], 
                                     components: Dict[str, Any]) -> str:
        """Generate semantic HTML structure"""
        try:
            dimensions = structure.get("document", {}).get("dimensions", {"width": 800, "height": 600})
            
            html_lines = []
            
            # Container element
            html_lines.extend([
                '<div class="banner-container" role="banner" aria-label="Advertisement Banner">',
                '  <!-- Background Elements -->'
            ])
            
            # Add background elements
            background_html = await self._generate_background_html(structure)
            if background_html:
                html_lines.extend([f"  {line}" for line in background_html.split('\n')])
            
            html_lines.append('  <!-- Content Elements -->')
            
            # Add component elements
            component_html = await self._generate_components_html(components)
            if component_html:
                html_lines.extend([f"  {line}" for line in component_html.split('\n')])
            
            html_lines.append('</div>')
            
            return '\n'.join(html_lines)
            
        except Exception as e:
            logger.error(f"Error generating HTML structure: {e}")
            return '<div class="banner-container">Error generating content</div>'
    
    async def _generate_background_html(self, structure: Dict[str, Any]) -> str:
        """Generate background HTML elements"""
        try:
            canvas = structure.get("canvas", {})
            background = canvas.get("background", {})
            
            if background.get("type") == "image" and background.get("source"):
                return f'''<div class="banner-background">
  <img src="{background['source']}" alt="Background" class="background-image" aria-hidden="true" />
</div>'''
            elif background.get("fallback_color"):
                return f'<div class="banner-background background-color" style="background-color: {background["fallback_color"]};"></div>'
            
            return ""
            
        except Exception as e:
            logger.error(f"Error generating background HTML: {e}")
            return ""
    
    async def _generate_components_html(self, components: Dict[str, Any]) -> str:
        """Generate HTML for all components"""
        try:
            html_lines = []
            
            # Sort components by z-index
            sorted_components = sorted(
                components.items(),
                key=lambda x: x[1].get("position", {}).get("z_index", 1)
            )
            
            for comp_id, component in sorted_components:
                component_html = await self._generate_component_html(comp_id, component)
                if component_html:
                    html_lines.append(component_html)
            
            return '\n'.join(html_lines)
            
        except Exception as e:
            logger.error(f"Error generating components HTML: {e}")
            return ""
    
    async def _generate_component_html(self, comp_id: str, component: Dict[str, Any]) -> str:
        """Generate HTML for a single component"""
        try:
            comp_type = component.get("type")
            content = component.get("content", {})
            accessibility = component.get("accessibility", {})
            
            if comp_type == "text":
                return await self._generate_text_html(comp_id, component)
            elif comp_type == "button":
                return await self._generate_button_html(comp_id, component)
            elif comp_type == "logo":
                return await self._generate_logo_html(comp_id, component)
            elif comp_type == "background_image":
                return await self._generate_image_html(comp_id, component)
            else:
                logger.warning(f"Unknown component type: {comp_type}")
                return f'<div id="{comp_id}" class="component-{comp_type}"><!-- Unknown component --></div>'
                
        except Exception as e:
            logger.error(f"Error generating component HTML {comp_id}: {e}")
            return f'<div id="{comp_id}" class="component-error"><!-- Error generating component --></div>'
    
    async def _generate_text_html(self, comp_id: str, component: Dict[str, Any]) -> str:
        """Generate HTML for text component"""
        content = component.get("content", {})
        accessibility = component.get("accessibility", {})
        
        text = content.get("text", "")
        purpose = content.get("purpose", "general")
        hierarchy_level = content.get("hierarchy_level", "body")
        
        # Determine semantic HTML tag
        if self.semantic_markup:
            if hierarchy_level.startswith("h"):
                tag = hierarchy_level
            elif purpose == "headline":
                tag = "h1"
            elif purpose == "brand":
                tag = "h2"
            elif purpose == "supporting":
                tag = "p"
            else:
                tag = "p"
        else:
            tag = "div"
        
        # Build attributes
        attributes = [f'id="{comp_id}"', f'class="text-component text-{purpose}"']
        
        # Add accessibility attributes
        if self.accessibility_features:
            aria_label = accessibility.get("aria_label")
            if aria_label:
                attributes.append(f'aria-label="{self._escape_html(aria_label)}"')
            
            role = accessibility.get("role")
            if role and role != "text":
                attributes.append(f'role="{role}"')
        
        attributes_str = " ".join(attributes)
        escaped_text = self._escape_html(text)
        
        return f'<{tag} {attributes_str}>{escaped_text}</{tag}>'
    
    async def _generate_button_html(self, comp_id: str, component: Dict[str, Any]) -> str:
        """Generate HTML for button component"""
        content = component.get("content", {})
        accessibility = component.get("accessibility", {})
        
        text = content.get("text", "")
        action = content.get("action", "click")
        target = content.get("target", "#")
        
        # Build attributes
        attributes = [
            f'id="{comp_id}"',
            'class="button-component btn btn-primary"',
            'type="button"'
        ]
        
        # Add accessibility attributes
        if self.accessibility_features:
            attributes.append(f'aria-label="{self._escape_html(accessibility.get("aria_label", text))}"')
            
            tab_index = accessibility.get("tab_index", 0)
            if tab_index >= 0:
                attributes.append(f'tabindex="{tab_index}"')
        
        # Add data attributes for interaction
        if action and target:
            attributes.append(f'data-action="{action}"')
            attributes.append(f'data-target="{target}"')
        
        attributes_str = " ".join(attributes)
        escaped_text = self._escape_html(text)
        
        return f'<button {attributes_str}>{escaped_text}</button>'
    
    async def _generate_logo_html(self, comp_id: str, component: Dict[str, Any]) -> str:
        """Generate HTML for logo component"""
        content = component.get("content", {})
        accessibility = component.get("accessibility", {})
        
        source = content.get("source", "")
        alt_text = content.get("alt_text", accessibility.get("alt_text", "Company Logo"))
        
        # Build attributes
        attributes = [
            f'id="{comp_id}"',
            'class="logo-component"',
            f'src="{source}"',
            f'alt="{self._escape_html(alt_text)}"'
        ]
        
        # Add accessibility attributes
        if self.accessibility_features:
            attributes.append('role="img"')
            
            if accessibility.get("decorative", False):
                attributes.append('aria-hidden="true"')
        
        # Add loading optimization
        attributes.extend([
            'loading="lazy"',
            'decoding="async"'
        ])
        
        attributes_str = " ".join(attributes)
        
        return f'<img {attributes_str} />'
    
    async def _generate_image_html(self, comp_id: str, component: Dict[str, Any]) -> str:
        """Generate HTML for image component"""
        content = component.get("content", {})
        accessibility = component.get("accessibility", {})
        
        source = content.get("source", "")
        alt_text = content.get("alt_text", "")
        
        # Build attributes
        attributes = [
            f'id="{comp_id}"',
            'class="image-component"',
            f'src="{source}"',
            f'alt="{self._escape_html(alt_text)}"'
        ]
        
        # Add accessibility attributes
        if self.accessibility_features and alt_text:
            attributes.append('role="img"')
        elif not alt_text:
            attributes.append('aria-hidden="true"')
        
        # Add loading optimization
        attributes.extend([
            'loading="lazy"',
            'decoding="async"'
        ])
        
        attributes_str = " ".join(attributes)
        
        return f'<img {attributes_str} />'
    
    async def _generate_css_styles(self, styling: Dict[str, Any], 
                                 responsive: Dict[str, Any],
                                 components: Dict[str, Any]) -> str:
        """Generate complete CSS styles"""
        try:
            css_lines = []
            
            # Add CSS reset and base styles
            css_lines.extend(await self._generate_css_reset())
            css_lines.append("")
            
            # Add global styles
            css_lines.extend(await self._generate_global_styles(styling))
            css_lines.append("")
            
            # Add container styles
            css_lines.extend(await self._generate_container_styles(styling))
            css_lines.append("")
            
            # Add component styles
            css_lines.extend(await self._generate_component_styles(components, styling))
            css_lines.append("")
            
            # Add responsive styles
            css_lines.extend(await self._generate_responsive_styles(responsive, components))
            css_lines.append("")
            
            # Add animation styles
            css_lines.extend(await self._generate_animation_styles(styling))
            css_lines.append("")
            
            # Add accessibility styles
            if self.accessibility_features:
                css_lines.extend(await self._generate_accessibility_styles())
                css_lines.append("")
            
            # Add print styles
            css_lines.extend(await self._generate_print_styles())
            
            return '\n'.join(css_lines)
            
        except Exception as e:
            logger.error(f"Error generating CSS styles: {e}")
            return "/* Error generating styles */"
    
    async def _generate_css_reset(self) -> List[str]:
        """Generate CSS reset styles"""
        return [
            "/* CSS Reset and Base Styles */",
            "* {",
            "  box-sizing: border-box;",
            "  margin: 0;",
            "  padding: 0;",
            "}",
            "",
            "img {",
            "  max-width: 100%;",
            "  height: auto;",
            "  display: block;",
            "}",
            "",
            "button {",
            "  font-family: inherit;",
            "  cursor: pointer;",
            "  border: none;",
            "  outline: none;",
            "}"
        ]
    
    async def _generate_global_styles(self, styling: Dict[str, Any]) -> List[str]:
        """Generate global CSS styles"""
        global_styles = styling.get("global_styles", {})
        base_styles = global_styles.get("base", {})
        
        lines = [
            "/* Global Styles */",
            "body {",
            f"  font-family: {base_styles.get('font_family', 'Inter, sans-serif')};",
            f"  font-size: {base_styles.get('font_size', '16px')};",
            f"  line-height: {base_styles.get('line_height', '1.4')};",
            f"  color: {base_styles.get('color', '#000000')};",
            f"  background-color: {base_styles.get('background_color', '#ffffff')};",
            "  -webkit-font-smoothing: antialiased;",
            "  -moz-osx-font-smoothing: grayscale;",
            "  text-rendering: optimizeLegibility;",
            "}"
        ]
        
        return lines
    
    async def _generate_container_styles(self, styling: Dict[str, Any]) -> List[str]:
        """Generate container styles"""
        container_styles = styling.get("global_styles", {}).get("container", {})
        
        lines = [
            "/* Banner Container */",
            ".banner-container {",
            f"  width: {container_styles.get('width', '800px')};",
            f"  height: {container_styles.get('height', '600px')};",
            "  position: relative;",
            "  overflow: hidden;",
            "  border-radius: 8px;",
            "  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);",
            "  background-size: cover;",
            "  background-position: center;",
            "  background-repeat: no-repeat;",
            "}",
            "",
            ".banner-background {",
            "  position: absolute;",
            "  top: 0;",
            "  left: 0;",
            "  width: 100%;",
            "  height: 100%;",
            "  z-index: 1;",
            "}",
            "",
            ".background-image {",
            "  width: 100%;",
            "  height: 100%;",
            "  object-fit: cover;",
            "  object-position: center;",
            "}",
            "",
            ".background-color {",
            "  background-color: inherit;",
            "}"
        ]
        
        return lines
    
    async def _generate_component_styles(self, components: Dict[str, Any], 
                                       styling: Dict[str, Any]) -> List[str]:
        """Generate styles for all components"""
        lines = ["/* Component Styles */"]
        
        for comp_id, component in components.items():
            component_styles = await self._generate_single_component_styles(comp_id, component)
            lines.extend(component_styles)
            lines.append("")
        
        return lines
    
    async def _generate_single_component_styles(self, comp_id: str, 
                                              component: Dict[str, Any]) -> List[str]:
        """Generate CSS for a single component"""
        try:
            position = component.get("position", {})
            dimensions = component.get("dimensions", {})
            styling = component.get("styling", {})
            comp_type = component.get("type")
            
            base_styles = styling.get("base", {})
            hover_styles = styling.get("states", {}).get("hover", {})
            
            lines = [f"/* {comp_id} - {comp_type} */"]
            
            # Base styles
            selector = f"#{comp_id}"
            lines.append(f"{selector} {{")
            
            # Position and dimensions
            lines.append(f"  position: absolute;")
            lines.append(f"  left: {position.get('x', 0)}px;")
            lines.append(f"  top: {position.get('y', 0)}px;")
            lines.append(f"  width: {dimensions.get('width', 'auto')}px;")
            lines.append(f"  height: {dimensions.get('height', 'auto')}px;")
            lines.append(f"  z-index: {position.get('z_index', 1)};")
            
            # Component-specific styles
            for prop, value in base_styles.items():
                css_prop = self._convert_to_css_property(prop)
                lines.append(f"  {css_prop}: {value};")
            
            # Add common component styles
            if comp_type == "button":
                lines.extend([
                    "  border: none;",
                    "  border-radius: 4px;",
                    "  cursor: pointer;",
                    "  transition: all 0.3s ease;",
                    "  user-select: none;",
                    "  touch-action: manipulation;"
                ])
            elif comp_type == "text":
                lines.extend([
                    "  word-wrap: break-word;",
                    "  overflow-wrap: break-word;"
                ])
            elif comp_type == "logo":
                lines.extend([
                    "  object-fit: contain;",
                    "  object-position: center;"
                ])
            
            lines.append("}")
            
            # Hover styles
            if hover_styles and comp_type == "button":
                lines.append(f"{selector}:hover {{")
                for prop, value in hover_styles.items():
                    css_prop = self._convert_to_css_property(prop)
                    lines.append(f"  {css_prop}: {value};")
                lines.append("}")
            
            # Focus styles for accessibility
            if comp_type == "button" and self.accessibility_features:
                lines.extend([
                    f"{selector}:focus {{",
                    "  outline: 2px solid #007bff;",
                    "  outline-offset: 2px;",
                    "}"
                ])
            
            return lines
            
        except Exception as e:
            logger.error(f"Error generating styles for {comp_id}: {e}")
            return [f"/* Error generating styles for {comp_id} */"]
    
    def _convert_to_css_property(self, prop: str) -> str:
        """Convert property name to CSS format"""
        # Convert underscore to hyphen and handle special cases
        css_prop = prop.replace("_", "-")
        
        # Handle specific conversions
        conversions = {
            "font-family": "font-family",
            "font-size": "font-size",
            "font-weight": "font-weight",
            "text-align": "text-align",
            "background-color": "background-color",
            "border-radius": "border-radius"
        }
        
        return conversions.get(css_prop, css_prop)
    
    async def _generate_responsive_styles(self, responsive: Dict[str, Any], 
                                        components: Dict[str, Any]) -> List[str]:
        """Generate responsive CSS media queries"""
        lines = ["/* Responsive Styles */"]
        
        breakpoints = responsive.get("breakpoints", {})
        
        # Mobile styles
        if "mobile" in breakpoints:
            lines.extend([
                f"@media (max-width: {breakpoints['mobile'].get('max_width', 767)}px) {{",
                "  .banner-container {",
                "    width: 100%;",
                "    max-width: 350px;",
                "    height: auto;",
                "    min-height: 200px;",
                "  }",
                ""
            ])
            
            # Mobile component adjustments
            for comp_id, component in components.items():
                comp_type = component.get("type")
                lines.extend([
                    f"  #{comp_id} {{",
                    "    position: relative;",
                    "    width: auto;",
                    "    margin: 8px;",
                    "    font-size: 0.9em;",
                    "  }}"
                ])
            
            lines.append("}")
            lines.append("")
        
        # Tablet styles
        if "tablet" in breakpoints:
            tablet_bp = breakpoints['tablet']
            lines.extend([
                f"@media (min-width: {tablet_bp.get('min_width', 768)}px) and (max-width: {tablet_bp.get('max_width', 1023)}px) {{",
                "  .banner-container {",
                "    width: 100%;",
                "    max-width: 600px;",
                "  }",
                "}"
            ])
            lines.append("")
        
        return lines
    
    async def _generate_animation_styles(self, styling: Dict[str, Any]) -> List[str]:
        """Generate CSS animations and transitions"""
        return [
            "/* Animations */",
            "@keyframes fadeIn {",
            "  from { opacity: 0; }",
            "  to { opacity: 1; }",
            "}",
            "",
            "@keyframes slideInUp {",
            "  from {",
            "    opacity: 0;",
            "    transform: translateY(30px);",
            "  }",
            "  to {",
            "    opacity: 1;",
            "    transform: translateY(0);",
            "  }",
            "}",
            "",
            ".fade-in {",
            "  animation: fadeIn 0.6s ease-out;",
            "}",
            "",
            ".slide-in-up {",
            "  animation: slideInUp 0.6s ease-out;",
            "}"
        ]
    
    async def _generate_accessibility_styles(self) -> List[str]:
        """Generate accessibility-focused CSS"""
        return [
            "/* Accessibility Styles */",
            "@media (prefers-reduced-motion: reduce) {",
            "  * {",
            "    animation-duration: 0.01ms !important;",
            "    animation-iteration-count: 1 !important;",
            "    transition-duration: 0.01ms !important;",
            "  }",
            "}",
            "",
            "@media (prefers-contrast: high) {",
            "  .banner-container {",
            "    border: 2px solid #000000;",
            "  }",
            "  ",
            "  .text-component {",
            "    color: #000000 !important;",
            "    background-color: #ffffff !important;",
            "  }",
            "  ",
            "  .button-component {",
            "    background-color: #000000 !important;",
            "    color: #ffffff !important;",
            "    border: 2px solid #000000 !important;",
            "  }",
            "}",
            "",
            ".sr-only {",
            "  position: absolute;",
            "  width: 1px;",
            "  height: 1px;",
            "  padding: 0;",
            "  margin: -1px;",
            "  overflow: hidden;",
            "  clip: rect(0, 0, 0, 0);",
            "  white-space: nowrap;",
            "  border: 0;",
            "}"
        ]
    
    async def _generate_print_styles(self) -> List[str]:
        """Generate print-specific CSS"""
        return [
            "/* Print Styles */",
            "@media print {",
            "  .banner-container {",
            "    width: 100%;",
            "    height: auto;",
            "    box-shadow: none;",
            "    border: 1px solid #000000;",
            "    page-break-inside: avoid;",
            "  }",
            "  ",
            "  .button-component {",
            "    border: 1px solid #000000;",
            "    background-color: #ffffff !important;",
            "    color: #000000 !important;",
            "  }",
            "  ",
            "  .background-image {",
            "    filter: grayscale(100%);",
            "  }",
            "}"
        ]
    
    async def _generate_javascript(self, interactions: Dict[str, Any], 
                                 components: Dict[str, Any]) -> str:
        """Generate JavaScript for interactions"""
        try:
            if not interactions or not interactions.get("user_interactions"):
                return ""
            
            js_lines = [
                "// Banner Interactions",
                "(function() {",
                "  'use strict';",
                "",
                "  // Wait for DOM to be ready",
                "  document.addEventListener('DOMContentLoaded', function() {",
                "    initializeBannerInteractions();",
                "  });",
                "",
                "  function initializeBannerInteractions() {"
            ]
            
            # Add click handlers for buttons
            user_interactions = interactions.get("user_interactions", {})
            for comp_id, interaction_data in user_interactions.items():
                if "click" in interaction_data:
                    js_lines.extend([
                        f"    // {comp_id} click handler",
                        f"    const {comp_id}Element = document.getElementById('{comp_id}');",
                        f"    if ({comp_id}Element) {{",
                        f"      {comp_id}Element.addEventListener('click', function(e) {{",
                        f"        e.preventDefault();",
                        f"        handleButtonClick('{comp_id}', e);",
                        f"      }});",
                        f"    }}",
                        ""
                    ])
            
            js_lines.extend([
                "  }",
                "",
                "  function handleButtonClick(buttonId, event) {",
                "    console.log('Button clicked:', buttonId);",
                "    ",
                "    // Add your custom click handling here",
                "    const button = event.target;",
                "    const action = button.getAttribute('data-action');",
                "    const target = button.getAttribute('data-target');",
                "    ",
                "    if (action === 'click' && target && target !== '#') {",
                "      window.open(target, '_blank');",
                "    }",
                "  }",
                "",
                "})();"
            ])
            
            return '\n'.join(js_lines)
            
        except Exception as e:
            logger.error(f"Error generating JavaScript: {e}")
            return "// Error generating JavaScript"
    
    async def _create_complete_document(self, html_content: str, css_content: str, 
                                      js_content: str, structure: Dict[str, Any]) -> str:
        """Create complete HTML document"""
        try:
            dimensions = structure.get("document", {}).get("dimensions", {"width": 800, "height": 600})
            
            html_doc = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="AI-generated banner advertisement">
    <title>Generated Banner Advertisement</title>
    
    <!-- Generated Styles -->
    <style>
{css_content}
    </style>
</head>
<body>
    <!-- Generated Banner Content -->
{html_content}
    
    {f'<!-- Generated Interactions --><script>{js_content}</script>' if js_content else ''}
</body>
</html>'''
            
            return html_doc
            
        except Exception as e:
            logger.error(f"Error creating complete document: {e}")
            return "<!DOCTYPE html><html><body>Error creating document</body></html>"
    
    async def _optimize_html_output(self, html_content: str) -> str:
        """Optimize HTML output for size and performance"""
        try:
            optimized = html_content
            
            if not self.include_comments:
                # Remove HTML comments
                optimized = re.sub(r'<!--.*?-->', '', optimized, flags=re.DOTALL)
            
            if self.minify_output:
                # Basic minification
                optimized = re.sub(r'\s+', ' ', optimized)
                optimized = re.sub(r'>\s+<', '><', optimized)
                optimized = optimized.strip()
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing HTML: {e}")
            return html_content
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#x27;"))
    
    async def _generate_html_metadata(self, blueprint: Dict[str, Any], 
                                    html_content: str) -> Dict[str, Any]:
        """Generate metadata about the HTML"""
        try:
            structure = blueprint.get("structure", {})
            components = blueprint.get("components", {})
            
            metadata = {
                "format": "html",
                "html_version": self.html_version,
                "css_framework": self.css_framework,
                "responsive_strategy": self.responsive_strategy,
                "dimensions": structure.get("document", {}).get("dimensions", {}),
                "file_size_bytes": len(html_content.encode('utf-8')),
                "component_count": len(components),
                "semantic_markup": self.semantic_markup,
                "accessibility_features": self.accessibility_features,
                "features": {
                    "responsive": "mobile_first" in html_content,
                    "animations": "@keyframes" in html_content,
                    "accessibility": "aria-label" in html_content,
                    "print_styles": "@media print" in html_content,
                    "interactions": "<script>" in html_content
                }
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating HTML metadata: {e}")
            return {"format": "html", "error": str(e)}
    
    async def _load_html_templates(self):
        """Load HTML templates"""
        # This would load predefined HTML templates
        pass
    
    async def _initialize_css_framework(self):
        """Initialize CSS framework"""
        # This would initialize CSS framework utilities
        pass
