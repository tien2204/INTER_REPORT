"""
Code Optimizer

Optimizes generated code for performance, size, and quality
across different output formats (SVG, HTML, Figma).
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
import json
from structlog import get_logger

logger = get_logger(__name__)


class CodeOptimizer:
    """
    Multi-format code optimization system
    
    Capabilities:
    - SVG optimization and minification
    - CSS optimization and compression
    - HTML structure optimization
    - JavaScript minification
    - Performance optimization
    - Code quality improvements
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Optimization configuration
        self.optimization_level = config.get("optimization_level", "standard")  # minimal, standard, aggressive
        self.preserve_readability = config.get("preserve_readability", True)
        self.remove_comments = config.get("remove_comments", False)
        self.minify_code = config.get("minify_code", False)
        
        # Format-specific settings
        self.svg_optimization = config.get("svg_optimization", True)
        self.css_optimization = config.get("css_optimization", True)
        self.html_optimization = config.get("html_optimization", True)
        self.js_optimization = config.get("js_optimization", True)
        
        # Performance thresholds
        self.max_file_size_kb = config.get("max_file_size_kb", 500)
        self.max_css_selectors = config.get("max_css_selectors", 1000)
        self.max_dom_depth = config.get("max_dom_depth", 10)
        
        # Communication
        self.shared_memory = None
        self.message_queue = None
        
        logger.info("Code Optimizer initialized")
    
    async def initialize(self):
        """Initialize the code optimizer"""
        try:
            # Load optimization rules and patterns
            await self._load_optimization_rules()
            await self._initialize_minifiers()
            
            logger.info("Code Optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Code Optimizer: {e}")
            raise
    
    def set_communication(self, shared_memory, message_queue):
        """Set communication interfaces"""
        self.shared_memory = shared_memory
        self.message_queue = message_queue
    
    async def optimize_code(self, code_result: Dict[str, Any], 
                          format_name: str) -> Dict[str, Any]:
        """
        Optimize generated code based on format
        
        Args:
            code_result: Generated code result from generator
            format_name: Target format (svg, html, figma, etc.)
            
        Returns:
            Optimized code result with improvements and metadata
        """
        try:
            logger.info(f"Starting code optimization for {format_name}")
            
            if not code_result.get("success"):
                return code_result  # Return as-is if generation failed
            
            # Route to format-specific optimizer
            if format_name == "svg":
                optimized_result = await self._optimize_svg(code_result)
            elif format_name == "html":
                optimized_result = await self._optimize_html(code_result)
            elif format_name == "figma":
                optimized_result = await self._optimize_figma(code_result)
            else:
                logger.warning(f"No optimizer available for format: {format_name}")
                optimized_result = code_result
            
            # Add optimization metadata
            optimized_result["optimization"] = {
                "level": self.optimization_level,
                "optimized_at": datetime.utcnow().isoformat(),
                "optimizations_applied": optimized_result.get("optimizations_applied", []),
                "size_reduction": await self._calculate_size_reduction(code_result, optimized_result),
                "quality_score": await self._calculate_quality_score(optimized_result, format_name)
            }
            
            logger.info(f"Code optimization completed for {format_name}")
            return optimized_result
            
        except Exception as e:
            logger.error(f"Error optimizing {format_name} code: {e}")
            # Return original result with error info
            code_result["optimization_error"] = str(e)
            return code_result
    
    async def _optimize_svg(self, svg_result: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize SVG code"""
        try:
            svg_code = svg_result.get("svg_code", "")
            if not svg_code:
                return svg_result
            
            optimizations_applied = []
            optimized_code = svg_code
            
            # 1. Remove unnecessary whitespace
            if self.svg_optimization:
                optimized_code = await self._optimize_svg_whitespace(optimized_code)
                optimizations_applied.append("whitespace_removal")
            
            # 2. Optimize numeric precision
            optimized_code = await self._optimize_svg_numbers(optimized_code)
            optimizations_applied.append("numeric_precision")
            
            # 3. Remove unused definitions
            optimized_code = await self._remove_unused_svg_definitions(optimized_code)
            optimizations_applied.append("unused_definitions_removal")
            
            # 4. Optimize paths
            optimized_code = await self._optimize_svg_paths(optimized_code)
            optimizations_applied.append("path_optimization")
            
            # 5. Compress CSS within SVG
            optimized_code = await self._optimize_svg_css(optimized_code)
            optimizations_applied.append("css_optimization")
            
            # 6. Remove comments if requested
            if self.remove_comments:
                optimized_code = await self._remove_svg_comments(optimized_code)
                optimizations_applied.append("comment_removal")
            
            # 7. Minify if requested
            if self.minify_code:
                optimized_code = await self._minify_svg(optimized_code)
                optimizations_applied.append("minification")
            
            # Update result
            optimized_result = svg_result.copy()
            optimized_result["svg_code"] = optimized_code
            optimized_result["file_size"] = len(optimized_code.encode('utf-8'))
            optimized_result["optimizations_applied"] = optimizations_applied
            
            return optimized_result
            
        except Exception as e:
            logger.error(f"Error optimizing SVG: {e}")
            return svg_result
    
    async def _optimize_svg_whitespace(self, svg_code: str) -> str:
        """Optimize whitespace in SVG"""
        try:
            # Remove excessive whitespace between elements
            optimized = re.sub(r'>\s+<', '><', svg_code)
            
            # Normalize whitespace within attributes
            optimized = re.sub(r'\s+', ' ', optimized)
            
            # Remove leading/trailing whitespace from lines
            lines = optimized.split('\n')
            optimized_lines = [line.strip() for line in lines if line.strip()]
            
            return '\n'.join(optimized_lines) if self.preserve_readability else ''.join(optimized_lines)
            
        except Exception as e:
            logger.error(f"Error optimizing SVG whitespace: {e}")
            return svg_code
    
    async def _optimize_svg_numbers(self, svg_code: str) -> str:
        """Optimize numeric precision in SVG"""
        try:
            # Round decimal numbers to 2 decimal places
            def round_number(match):
                number = float(match.group(0))
                return f"{number:.2f}".rstrip('0').rstrip('.')
            
            # Apply to common numeric patterns
            optimized = re.sub(r'\d+\.\d{3,}', round_number, svg_code)
            
            # Remove unnecessary .0 from integers
            optimized = re.sub(r'(\d+)\.0(?![0-9])', r'\1', optimized)
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing SVG numbers: {e}")
            return svg_code
    
    async def _remove_unused_svg_definitions(self, svg_code: str) -> str:
        """Remove unused definitions from SVG"""
        try:
            # Find all IDs defined in <defs>
            defs_ids = re.findall(r'<defs>.*?id="([^"]+)".*?</defs>', svg_code, re.DOTALL)
            
            # Find all ID references
            used_ids = re.findall(r'url\(#([^)]+)\)', svg_code)
            used_ids.extend(re.findall(r'href="#([^"]+)"', svg_code))
            
            # Remove unused definitions
            for def_id in defs_ids:
                if def_id not in used_ids:
                    # Remove the entire definition element
                    pattern = rf'<[^>]+id="{re.escape(def_id)}"[^>]*>.*?</[^>]+>'
                    svg_code = re.sub(pattern, '', svg_code, flags=re.DOTALL)
            
            return svg_code
            
        except Exception as e:
            logger.error(f"Error removing unused SVG definitions: {e}")
            return svg_code
    
    async def _optimize_svg_paths(self, svg_code: str) -> str:
        """Optimize SVG path data"""
        try:
            # Simplify path commands (basic optimization)
            def optimize_path(match):
                path_data = match.group(1)
                
                # Remove unnecessary spaces
                path_data = re.sub(r'\s+', ' ', path_data)
                
                # Remove spaces around commas
                path_data = re.sub(r'\s*,\s*', ',', path_data)
                
                # Remove spaces after path commands
                path_data = re.sub(r'([MLHVCSQTAZ])\s+', r'\1', path_data, flags=re.IGNORECASE)
                
                return f'd="{path_data.strip()}"'
            
            optimized = re.sub(r'd="([^"]*)"', optimize_path, svg_code)
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing SVG paths: {e}")
            return svg_code
    
    async def _optimize_svg_css(self, svg_code: str) -> str:
        """Optimize CSS within SVG"""
        try:
            # Find and optimize <style> blocks
            def optimize_style_block(match):
                css_content = match.group(1)
                optimized_css = self._optimize_css_content(css_content)
                return f"<style type=\"text/css\">{optimized_css}</style>"
            
            optimized = re.sub(r'<style[^>]*>(.*?)</style>', optimize_style_block, svg_code, flags=re.DOTALL)
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing SVG CSS: {e}")
            return svg_code
    
    async def _remove_svg_comments(self, svg_code: str) -> str:
        """Remove comments from SVG"""
        return re.sub(r'<!--.*?-->', '', svg_code, flags=re.DOTALL)
    
    async def _minify_svg(self, svg_code: str) -> str:
        """Minify SVG code"""
        try:
            minified = svg_code
            
            # Remove all unnecessary whitespace
            minified = re.sub(r'\s+', ' ', minified)
            
            # Remove whitespace around tags
            minified = re.sub(r'>\s+<', '><', minified)
            
            # Remove whitespace at start and end
            minified = minified.strip()
            
            return minified
            
        except Exception as e:
            logger.error(f"Error minifying SVG: {e}")
            return svg_code
    
    async def _optimize_html(self, html_result: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize HTML code"""
        try:
            html_code = html_result.get("html_code", "")
            css_content = html_result.get("css_content", "")
            js_content = html_result.get("js_content", "")
            
            optimizations_applied = []
            
            # 1. Optimize HTML structure
            if self.html_optimization:
                optimized_html = await self._optimize_html_structure(html_code)
                optimizations_applied.append("html_structure")
            else:
                optimized_html = html_code
            
            # 2. Optimize CSS
            if self.css_optimization and css_content:
                optimized_css = await self._optimize_css_content(css_content)
                optimized_html = optimized_html.replace(css_content, optimized_css)
                optimizations_applied.append("css_optimization")
            
            # 3. Optimize JavaScript
            if self.js_optimization and js_content:
                optimized_js = await self._optimize_javascript(js_content)
                optimized_html = optimized_html.replace(js_content, optimized_js)
                optimizations_applied.append("js_optimization")
            
            # 4. Remove comments if requested
            if self.remove_comments:
                optimized_html = await self._remove_html_comments(optimized_html)
                optimizations_applied.append("comment_removal")
            
            # 5. Minify if requested
            if self.minify_code:
                optimized_html = await self._minify_html(optimized_html)
                optimizations_applied.append("minification")
            
            # Update result
            optimized_result = html_result.copy()
            optimized_result["html_code"] = optimized_html
            optimized_result["file_size"] = len(optimized_html.encode('utf-8'))
            optimized_result["optimizations_applied"] = optimizations_applied
            
            return optimized_result
            
        except Exception as e:
            logger.error(f"Error optimizing HTML: {e}")
            return html_result
    
    async def _optimize_html_structure(self, html_code: str) -> str:
        """Optimize HTML structure"""
        try:
            # Remove unnecessary attributes
            optimized = re.sub(r'\s+role="generic"', '', html_code)
            
            # Optimize image attributes
            optimized = re.sub(r'loading="lazy"\s+decoding="async"', 'loading="lazy" decoding="async"', optimized)
            
            # Remove empty class attributes
            optimized = re.sub(r'\s+class=""', '', optimized)
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing HTML structure: {e}")
            return html_code
    
    def _optimize_css_content(self, css_content: str) -> str:
        """Optimize CSS content"""
        try:
            # Remove unnecessary whitespace
            optimized = re.sub(r'\s+', ' ', css_content)
            
            # Remove spaces around certain characters
            optimized = re.sub(r'\s*{\s*', '{', optimized)
            optimized = re.sub(r'\s*}\s*', '}', optimized)
            optimized = re.sub(r'\s*:\s*', ':', optimized)
            optimized = re.sub(r'\s*;\s*', ';', optimized)
            
            # Remove trailing semicolons before closing braces
            optimized = re.sub(r';+}', '}', optimized)
            
            # Optimize color values
            optimized = re.sub(r'#([0-9a-fA-F])\1([0-9a-fA-F])\2([0-9a-fA-F])\3', r'#\1\2\3', optimized)
            
            # Remove unnecessary quotes
            optimized = re.sub(r"font-family:\s*['\"]([^'\"]*)['\"]", r'font-family:\1', optimized)
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing CSS: {e}")
            return css_content
    
    async def _optimize_javascript(self, js_content: str) -> str:
        """Optimize JavaScript content"""
        try:
            # Basic JS optimization
            optimized = js_content
            
            # Remove excessive whitespace
            optimized = re.sub(r'\s+', ' ', optimized)
            
            # Remove spaces around operators (careful with regex)
            optimized = re.sub(r'\s*([=+\-*/])\s*', r'\1', optimized)
            
            # Remove trailing semicolons before closing braces
            optimized = re.sub(r';\s*}', '}', optimized)
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing JavaScript: {e}")
            return js_content
    
    async def _remove_html_comments(self, html_code: str) -> str:
        """Remove HTML comments"""
        return re.sub(r'<!--.*?-->', '', html_code, flags=re.DOTALL)
    
    async def _minify_html(self, html_code: str) -> str:
        """Minify HTML code"""
        try:
            minified = html_code
            
            # Remove whitespace between tags
            minified = re.sub(r'>\s+<', '><', minified)
            
            # Remove multiple spaces
            minified = re.sub(r'\s+', ' ', minified)
            
            # Remove leading/trailing whitespace
            minified = minified.strip()
            
            return minified
            
        except Exception as e:
            logger.error(f"Error minifying HTML: {e}")
            return html_code
    
    async def _optimize_figma(self, figma_result: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize Figma plugin code"""
        try:
            figma_package = figma_result.get("figma_package", {})
            optimizations_applied = []
            
            # 1. Optimize plugin code
            plugin_code = figma_package.get("code", "")
            if plugin_code:
                optimized_code = await self._optimize_figma_code(plugin_code)
                figma_package["code"] = optimized_code
                optimizations_applied.append("code_optimization")
            
            # 2. Optimize UI HTML
            ui_html = figma_package.get("ui", "")
            if ui_html:
                optimized_ui = await self._optimize_figma_ui(ui_html)
                figma_package["ui"] = optimized_ui
                optimizations_applied.append("ui_optimization")
            
            # 3. Optimize node commands
            node_commands = figma_package.get("node_commands", [])
            if node_commands:
                optimized_commands = await self._optimize_figma_commands(node_commands)
                figma_package["node_commands"] = optimized_commands
                optimizations_applied.append("commands_optimization")
            
            # Update result
            optimized_result = figma_result.copy()
            optimized_result["figma_package"] = figma_package
            optimized_result["figma_code"] = json.dumps(figma_package, indent=None if self.minify_code else 2)
            optimized_result["optimizations_applied"] = optimizations_applied
            
            return optimized_result
            
        except Exception as e:
            logger.error(f"Error optimizing Figma: {e}")
            return figma_result
    
    async def _optimize_figma_code(self, code: str) -> str:
        """Optimize Figma plugin JavaScript code"""
        try:
            # Remove unnecessary console.log statements in production
            if self.optimization_level in ["standard", "aggressive"]:
                code = re.sub(r'console\.log\([^)]*\);\s*', '', code)
            
            # Optimize function declarations
            code = re.sub(r'function\s+(\w+)\s*\(', r'function \1(', code)
            
            # Remove excessive comments if requested
            if self.remove_comments:
                code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
                code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
            
            return code
            
        except Exception as e:
            logger.error(f"Error optimizing Figma code: {e}")
            return code
    
    async def _optimize_figma_ui(self, ui_html: str) -> str:
        """Optimize Figma plugin UI HTML"""
        try:
            # Apply HTML optimization
            optimized = await self._optimize_html_structure(ui_html)
            
            # Inline critical CSS if small enough
            if len(ui_html) < 10000:  # Small UI
                optimized = self._inline_critical_css(optimized)
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing Figma UI: {e}")
            return ui_html
    
    def _inline_critical_css(self, html: str) -> str:
        """Inline critical CSS for better performance"""
        # Simple implementation - in production, use proper CSS inlining
        return html
    
    async def _optimize_figma_commands(self, commands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize Figma node commands"""
        try:
            optimized_commands = []
            
            for command in commands:
                # Remove unnecessary properties
                optimized_command = {}
                for key, value in command.items():
                    if value is not None and value != "":
                        optimized_command[key] = value
                
                optimized_commands.append(optimized_command)
            
            return optimized_commands
            
        except Exception as e:
            logger.error(f"Error optimizing Figma commands: {e}")
            return commands
    
    async def _calculate_size_reduction(self, original: Dict[str, Any], 
                                      optimized: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate size reduction from optimization"""
        try:
            original_size = original.get("file_size", 0)
            optimized_size = optimized.get("file_size", 0)
            
            if original_size == 0:
                return {"bytes_saved": 0, "percentage_saved": 0}
            
            bytes_saved = original_size - optimized_size
            percentage_saved = (bytes_saved / original_size) * 100
            
            return {
                "original_size_bytes": original_size,
                "optimized_size_bytes": optimized_size,
                "bytes_saved": bytes_saved,
                "percentage_saved": round(percentage_saved, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating size reduction: {e}")
            return {"bytes_saved": 0, "percentage_saved": 0}
    
    async def _calculate_quality_score(self, optimized_result: Dict[str, Any], 
                                     format_name: str) -> float:
        """Calculate code quality score"""
        try:
            score = 1.0  # Start with perfect score
            
            # File size penalty
            file_size = optimized_result.get("file_size", 0)
            if file_size > self.max_file_size_kb * 1024:
                score -= 0.2
            
            # Format-specific quality checks
            if format_name == "html":
                html_code = optimized_result.get("html_code", "")
                
                # Check for accessibility features
                if "aria-label" in html_code:
                    score += 0.1
                if "alt=" in html_code:
                    score += 0.1
                
                # Check for semantic markup
                if any(tag in html_code for tag in ["<h1", "<h2", "<p", "<button"]):
                    score += 0.1
                
                # Check for responsive design
                if "@media" in html_code:
                    score += 0.1
            
            elif format_name == "svg":
                svg_code = optimized_result.get("svg_code", "")
                
                # Check for accessibility
                if "aria-label" in svg_code or "title>" in svg_code:
                    score += 0.1
                
                # Check for optimization
                if len(svg_code) < 10000:  # Reasonable size
                    score += 0.1
            
            # Optimization applied bonus
            optimizations_count = len(optimized_result.get("optimizations_applied", []))
            score += min(0.2, optimizations_count * 0.05)
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5  # Default score
    
    async def _load_optimization_rules(self):
        """Load optimization rules and patterns"""
        # This would load optimization rules from configuration
        pass
    
    async def _initialize_minifiers(self):
        """Initialize code minification tools"""
        # This would initialize minification libraries
        pass
    
    # Public utility methods
    async def analyze_code_metrics(self, code_result: Dict[str, Any], 
                                 format_name: str) -> Dict[str, Any]:
        """Analyze code metrics and performance"""
        try:
            metrics = {
                "format": format_name,
                "file_size_bytes": code_result.get("file_size", 0),
                "file_size_kb": round(code_result.get("file_size", 0) / 1024, 2),
                "optimization_opportunities": [],
                "performance_score": 0.0,
                "maintainability_score": 0.0
            }
            
            # Format-specific analysis
            if format_name == "svg":
                svg_code = code_result.get("svg_code", "")
                metrics.update(await self._analyze_svg_metrics(svg_code))
            elif format_name == "html":
                html_code = code_result.get("html_code", "")
                metrics.update(await self._analyze_html_metrics(html_code))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing code metrics: {e}")
            return {"error": str(e)}
    
    async def _analyze_svg_metrics(self, svg_code: str) -> Dict[str, Any]:
        """Analyze SVG-specific metrics"""
        return {
            "element_count": len(re.findall(r'<[^/][^>]*>', svg_code)),
            "path_count": len(re.findall(r'<path', svg_code)),
            "text_count": len(re.findall(r'<text', svg_code)),
            "definition_count": len(re.findall(r'<defs>', svg_code)),
            "viewbox_defined": "viewBox" in svg_code
        }
    
    async def _analyze_html_metrics(self, html_code: str) -> Dict[str, Any]:
        """Analyze HTML-specific metrics"""
        return {
            "dom_depth": self._calculate_dom_depth(html_code),
            "css_rules_count": len(re.findall(r'{[^}]*}', html_code)),
            "js_functions_count": len(re.findall(r'function\s+\w+', html_code)),
            "accessibility_features": len(re.findall(r'aria-\w+|alt=|role=', html_code)),
            "responsive_breakpoints": len(re.findall(r'@media', html_code))
        }
    
    def _calculate_dom_depth(self, html_code: str) -> int:
        """Calculate maximum DOM depth"""
        try:
            max_depth = 0
            current_depth = 0
            
            # Simple depth calculation
            for char in html_code:
                if char == '<':
                    # Check if it's a closing tag
                    if html_code[html_code.index(char) + 1:html_code.index(char) + 2] == '/':
                        current_depth -= 1
                    else:
                        current_depth += 1
                        max_depth = max(max_depth, current_depth)
            
            return max_depth
            
        except Exception:
            return 0
