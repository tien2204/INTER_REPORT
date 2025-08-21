"""
Accessibility Auditor

Evaluates design accessibility compliance with WCAG 2.1 guidelines
and ensures inclusive design practices.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
from structlog import get_logger

logger = get_logger(__name__)


class AccessibilityAuditor:
    """
    Accessibility compliance evaluation system
    
    Capabilities:
    - WCAG 2.1 compliance checking
    - Color contrast analysis
    - Text readability assessment
    - Navigation accessibility
    - Alternative text validation
    - Keyboard accessibility evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # WCAG compliance levels
        self.target_level = config.get("target_level", "AA")  # A, AA, AAA
        
        # Contrast requirements
        self.min_contrast_normal = config.get("min_contrast_normal", 4.5)  # WCAG AA
        self.min_contrast_large = config.get("min_contrast_large", 3.0)    # WCAG AA for large text
        self.min_contrast_aaa = config.get("min_contrast_aaa", 7.0)        # WCAG AAA
        
        # Text size thresholds
        self.large_text_size = config.get("large_text_size", 18)  # 18pt or larger
        self.large_text_bold_size = config.get("large_text_bold_size", 14)  # 14pt bold
        
        # Readability settings
        self.min_font_size = config.get("min_font_size", 12)
        self.max_line_length = config.get("max_line_length", 80)  # characters
        
        # Accessibility features
        self.require_alt_text = config.get("require_alt_text", True)
        self.require_aria_labels = config.get("require_aria_labels", True)
        self.check_focus_indicators = config.get("check_focus_indicators", True)
        
        # Communication
        self.shared_memory = None
        self.message_queue = None
        
        logger.info("Accessibility Auditor initialized")
    
    async def initialize(self):
        """Initialize the accessibility auditor"""
        try:
            # Load WCAG guidelines and standards
            await self._load_wcag_guidelines()
            await self._initialize_accessibility_tools()
            
            logger.info("Accessibility Auditor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Accessibility Auditor: {e}")
            raise
    
    def set_communication(self, shared_memory, message_queue):
        """Set communication interfaces"""
        self.shared_memory = shared_memory
        self.message_queue = message_queue
    
    async def audit_design(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive accessibility audit
        
        Args:
            design_data: Complete design data including blueprint and generated code
            
        Returns:
            Accessibility audit result with scores, violations, and recommendations
        """
        try:
            logger.info("Starting accessibility audit")
            
            # Extract design components
            blueprint = design_data.get("blueprint", {})
            components = blueprint.get("components", {})
            generated_code = design_data.get("generated_code", {})
            
            # Perform different accessibility checks
            contrast_analysis = await self._check_color_contrast(components)
            text_analysis = await self._check_text_accessibility(components)
            alternative_text_analysis = await self._check_alternative_text(components)
            navigation_analysis = await self._check_navigation_accessibility(components, generated_code)
            focus_analysis = await self._check_focus_accessibility(components, generated_code)
            semantic_analysis = await self._check_semantic_structure(generated_code)
            
            # Calculate overall accessibility score
            overall_score = await self._calculate_accessibility_score(
                contrast_analysis, text_analysis, alternative_text_analysis,
                navigation_analysis, focus_analysis, semantic_analysis
            )
            
            # Compile violations and recommendations
            violations = await self._compile_accessibility_violations(
                contrast_analysis, text_analysis, alternative_text_analysis,
                navigation_analysis, focus_analysis, semantic_analysis
            )
            
            recommendations = await self._generate_accessibility_recommendations(violations, overall_score)
            
            # Determine WCAG compliance level
            compliance_level = await self._determine_compliance_level(violations, overall_score)
            
            result = {
                "overall_score": overall_score,
                "wcag_compliance_level": compliance_level,
                "detailed_scores": {
                    "color_contrast": contrast_analysis.get("score", 0),
                    "text_accessibility": text_analysis.get("score", 0),
                    "alternative_text": alternative_text_analysis.get("score", 0),
                    "navigation": navigation_analysis.get("score", 0),
                    "focus_management": focus_analysis.get("score", 0),
                    "semantic_structure": semantic_analysis.get("score", 0)
                },
                "violations": violations,
                "recommendations": recommendations,
                "accessibility_details": {
                    "contrast": contrast_analysis,
                    "text": text_analysis,
                    "alt_text": alternative_text_analysis,
                    "navigation": navigation_analysis,
                    "focus": focus_analysis,
                    "semantic": semantic_analysis
                },
                "compliance_summary": await self._generate_compliance_summary(violations)
            }
            
            logger.info("Accessibility audit completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in accessibility audit: {e}")
            return {
                "overall_score": 0.0,
                "error": str(e),
                "violations": [{"type": "error", "description": f"Audit failed: {e}", "severity": "critical"}],
                "recommendations": []
            }
    
    async def _check_color_contrast(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Check color contrast compliance"""
        try:
            contrast_issues = []
            contrast_scores = []
            
            for comp_id, component in components.items():
                if component.get("type") == "text":
                    styling = component.get("styling", {}).get("base", {})
                    text_color = styling.get("color", "#000000")
                    bg_color = styling.get("background_color", "#ffffff")
                    
                    # Calculate contrast ratio
                    contrast_ratio = await self._calculate_contrast_ratio(text_color, bg_color)
                    
                    # Determine if text is large
                    font_size = await self._extract_font_size(styling.get("font_size", "16px"))
                    font_weight = styling.get("font_weight", "normal")
                    is_large_text = await self._is_large_text(font_size, font_weight)
                    
                    # Determine required contrast based on text size
                    if is_large_text:
                        required_contrast = self.min_contrast_large
                        required_contrast_aaa = 4.5
                    else:
                        required_contrast = self.min_contrast_normal
                        required_contrast_aaa = self.min_contrast_aaa
                    
                    # Check compliance
                    if contrast_ratio < required_contrast:
                        severity = "critical" if contrast_ratio < 3.0 else "major"
                        contrast_issues.append({
                            "type": "contrast_insufficient",
                            "component_id": comp_id,
                            "description": f"Insufficient contrast ratio {contrast_ratio:.2f}:1 (required: {required_contrast}:1)",
                            "severity": severity,
                            "guideline": f"WCAG 2.1 {self.target_level}",
                            "current_ratio": contrast_ratio,
                            "required_ratio": required_contrast,
                            "text_color": text_color,
                            "background_color": bg_color
                        })
                        contrast_scores.append(0.0)
                    elif self.target_level == "AAA" and contrast_ratio < required_contrast_aaa:
                        contrast_issues.append({
                            "type": "contrast_aaa_insufficient",
                            "component_id": comp_id,
                            "description": f"Contrast ratio {contrast_ratio:.2f}:1 meets AA but not AAA (required: {required_contrast_aaa}:1)",
                            "severity": "minor",
                            "guideline": "WCAG 2.1 AAA",
                            "current_ratio": contrast_ratio,
                            "required_ratio": required_contrast_aaa
                        })
                        contrast_scores.append(0.7)
                    else:
                        contrast_scores.append(1.0)
            
            # Calculate overall contrast score
            overall_contrast_score = sum(contrast_scores) / len(contrast_scores) if contrast_scores else 1.0
            
            return {
                "score": overall_contrast_score,
                "violations": contrast_issues,
                "total_text_elements": len(contrast_scores),
                "compliant_elements": len([s for s in contrast_scores if s > 0.5]),
                "average_contrast_ratio": await self._calculate_average_contrast(components)
            }
            
        except Exception as e:
            logger.error(f"Error checking color contrast: {e}")
            return {"score": 0.0, "error": str(e), "violations": []}
    
    async def _check_text_accessibility(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Check text accessibility and readability"""
        try:
            text_issues = []
            text_scores = []
            
            for comp_id, component in components.items():
                if component.get("type") == "text":
                    styling = component.get("styling", {}).get("base", {})
                    content = component.get("content", {})
                    
                    # Check font size
                    font_size = await self._extract_font_size(styling.get("font_size", "16px"))
                    if font_size < self.min_font_size:
                        text_issues.append({
                            "type": "font_size_too_small",
                            "component_id": comp_id,
                            "description": f"Font size {font_size}px is too small (minimum: {self.min_font_size}px)",
                            "severity": "major",
                            "guideline": "WCAG 2.1 - Text readability",
                            "current_size": font_size,
                            "minimum_size": self.min_font_size
                        })
                        text_scores.append(0.3)
                    else:
                        text_scores.append(1.0)
                    
                    # Check text content length for readability
                    text_content = content.get("text", "")
                    if len(text_content) > self.max_line_length * 3:  # Multiple lines check
                        text_issues.append({
                            "type": "text_too_long",
                            "component_id": comp_id,
                            "description": f"Text content may be too long for readability ({len(text_content)} characters)",
                            "severity": "minor",
                            "guideline": "Readability best practices"
                        })
                    
                    # Check for proper text hierarchy
                    hierarchy_issues = await self._check_text_hierarchy(comp_id, component, components)
                    text_issues.extend(hierarchy_issues)
                    
                    # Check for proper spacing
                    spacing_issues = await self._check_text_spacing(comp_id, component)
                    text_issues.extend(spacing_issues)
            
            overall_text_score = sum(text_scores) / len(text_scores) if text_scores else 1.0
            
            return {
                "score": max(0.0, overall_text_score - len(text_issues) * 0.05),
                "violations": text_issues,
                "total_text_elements": len(text_scores),
                "readability_score": await self._calculate_readability_score(components)
            }
            
        except Exception as e:
            logger.error(f"Error checking text accessibility: {e}")
            return {"score": 0.0, "error": str(e), "violations": []}
    
    async def _check_alternative_text(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Check alternative text for images and non-text content"""
        try:
            alt_text_issues = []
            alt_text_scores = []
            
            image_components = [
                comp for comp in components.values()
                if comp.get("type") in ["logo", "background_image", "image"]
            ]
            
            for comp_id, component in components.items():
                if component.get("type") in ["logo", "background_image", "image"]:
                    content = component.get("content", {})
                    accessibility = component.get("accessibility", {})
                    
                    # Check for alt text
                    alt_text = content.get("alt_text") or accessibility.get("alt_text")
                    
                    if not alt_text and self.require_alt_text:
                        # Check if image is decorative
                        is_decorative = accessibility.get("decorative", False)
                        
                        if not is_decorative:
                            alt_text_issues.append({
                                "type": "missing_alt_text",
                                "component_id": comp_id,
                                "description": "Image missing alternative text",
                                "severity": "major",
                                "guideline": "WCAG 2.1 - 1.1.1 Non-text Content",
                                "component_type": component.get("type")
                            })
                            alt_text_scores.append(0.0)
                        else:
                            # Decorative images should have empty alt text
                            alt_text_scores.append(1.0)
                    elif alt_text:
                        # Validate alt text quality
                        alt_quality_score = await self._evaluate_alt_text_quality(alt_text, component)
                        alt_text_scores.append(alt_quality_score)
                        
                        if alt_quality_score < 0.7:
                            alt_text_issues.append({
                                "type": "poor_alt_text_quality",
                                "component_id": comp_id,
                                "description": "Alternative text could be more descriptive",
                                "severity": "minor",
                                "guideline": "WCAG 2.1 - Alt text best practices",
                                "current_alt_text": alt_text
                            })
                    else:
                        alt_text_scores.append(1.0)
            
            # Check for ARIA labels on interactive elements
            interactive_components = [
                comp for comp in components.values()
                if comp.get("type") in ["button"]
            ]
            
            for comp_id, component in components.items():
                if component.get("type") == "button":
                    accessibility = component.get("accessibility", {})
                    aria_label = accessibility.get("aria_label")
                    content = component.get("content", {})
                    text_content = content.get("text", "")
                    
                    if not aria_label and not text_content and self.require_aria_labels:
                        alt_text_issues.append({
                            "type": "missing_aria_label",
                            "component_id": comp_id,
                            "description": "Interactive element missing accessible label",
                            "severity": "major",
                            "guideline": "WCAG 2.1 - 4.1.2 Name, Role, Value"
                        })
                        alt_text_scores.append(0.0)
                    else:
                        alt_text_scores.append(1.0)
            
            overall_alt_score = sum(alt_text_scores) / len(alt_text_scores) if alt_text_scores else 1.0
            
            return {
                "score": overall_alt_score,
                "violations": alt_text_issues,
                "total_elements_checked": len(alt_text_scores),
                "images_with_alt_text": len([s for s in alt_text_scores if s > 0]),
                "image_compliance_rate": len([s for s in alt_text_scores if s > 0]) / len(image_components) if image_components else 1.0
            }
            
        except Exception as e:
            logger.error(f"Error checking alternative text: {e}")
            return {"score": 0.0, "error": str(e), "violations": []}
    
    async def _check_navigation_accessibility(self, components: Dict[str, Any], 
                                            generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Check navigation and interaction accessibility"""
        try:
            navigation_issues = []
            navigation_score = 1.0
            
            # Check for proper element roles
            interactive_elements = [
                comp for comp in components.values()
                if comp.get("type") in ["button"]
            ]
            
            for comp_id, component in components.items():
                if component.get("type") == "button":
                    accessibility = component.get("accessibility", {})
                    role = accessibility.get("role")
                    
                    if not role:
                        navigation_issues.append({
                            "type": "missing_role",
                            "component_id": comp_id,
                            "description": "Interactive element missing explicit role",
                            "severity": "minor",
                            "guideline": "WCAG 2.1 - 4.1.2 Name, Role, Value"
                        })
                    
                    # Check for proper tab index
                    tab_index = accessibility.get("tab_index")
                    if tab_index is not None and tab_index < 0 and tab_index != -1:
                        navigation_issues.append({
                            "type": "invalid_tab_index",
                            "component_id": comp_id,
                            "description": "Invalid tabindex value",
                            "severity": "minor",
                            "guideline": "WCAG 2.1 - Keyboard accessibility"
                        })
            
            # Check HTML code for semantic structure
            html_formats = generated_code.get("formats", {})
            html_data = html_formats.get("html", {})
            html_code = html_data.get("html_code", "")
            
            if html_code:
                semantic_issues = await self._check_html_semantic_structure(html_code)
                navigation_issues.extend(semantic_issues)
            
            # Adjust score based on issues
            navigation_score -= len(navigation_issues) * 0.1
            navigation_score = max(0.0, navigation_score)
            
            return {
                "score": navigation_score,
                "violations": navigation_issues,
                "interactive_elements": len(interactive_elements),
                "keyboard_accessible": await self._estimate_keyboard_accessibility(components)
            }
            
        except Exception as e:
            logger.error(f"Error checking navigation accessibility: {e}")
            return {"score": 0.0, "error": str(e), "violations": []}
    
    async def _check_focus_accessibility(self, components: Dict[str, Any], 
                                       generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Check focus management and indicators"""
        try:
            focus_issues = []
            focus_score = 1.0
            
            # Check for focus indicators in CSS
            css_formats = generated_code.get("formats", {})
            html_data = css_formats.get("html", {})
            css_content = html_data.get("css_content", "")
            
            if css_content and self.check_focus_indicators:
                has_focus_styles = ":focus" in css_content
                
                if not has_focus_styles:
                    focus_issues.append({
                        "type": "missing_focus_indicators",
                        "description": "No focus indicators found in CSS",
                        "severity": "major",
                        "guideline": "WCAG 2.1 - 2.4.7 Focus Visible"
                    })
                    focus_score -= 0.5
                
                # Check for focus trap in modal-like elements
                # (This would be more sophisticated in a real implementation)
                
            # Check logical tab order
            tab_order_issues = await self._check_tab_order(components)
            focus_issues.extend(tab_order_issues)
            
            focus_score -= len(tab_order_issues) * 0.1
            focus_score = max(0.0, focus_score)
            
            return {
                "score": focus_score,
                "violations": focus_issues,
                "has_focus_styles": ":focus" in css_content if css_content else False,
                "tab_order_logical": len(tab_order_issues) == 0
            }
            
        except Exception as e:
            logger.error(f"Error checking focus accessibility: {e}")
            return {"score": 0.0, "error": str(e), "violations": []}
    
    async def _check_semantic_structure(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Check semantic HTML structure"""
        try:
            semantic_issues = []
            semantic_score = 1.0
            
            html_formats = generated_code.get("formats", {})
            html_data = html_formats.get("html", {})
            html_code = html_data.get("html_code", "")
            
            if html_code:
                # Check for proper heading structure
                heading_issues = await self._check_heading_structure(html_code)
                semantic_issues.extend(heading_issues)
                
                # Check for semantic elements
                semantic_elements = ["header", "main", "section", "article", "nav", "aside", "footer"]
                has_semantic_elements = any(f"<{elem}" in html_code for elem in semantic_elements)
                
                if not has_semantic_elements:
                    semantic_issues.append({
                        "type": "missing_semantic_elements",
                        "description": "No semantic HTML5 elements found",
                        "severity": "minor",
                        "guideline": "HTML5 semantic structure best practices"
                    })
                
                # Check for proper list structure
                list_issues = await self._check_list_structure(html_code)
                semantic_issues.extend(list_issues)
                
            semantic_score -= len(semantic_issues) * 0.1
            semantic_score = max(0.0, semantic_score)
            
            return {
                "score": semantic_score,
                "violations": semantic_issues,
                "has_semantic_elements": has_semantic_elements if html_code else False,
                "heading_structure_valid": len(heading_issues) == 0
            }
            
        except Exception as e:
            logger.error(f"Error checking semantic structure: {e}")
            return {"score": 0.0, "error": str(e), "violations": []}
    
    # Helper methods
    
    async def _calculate_contrast_ratio(self, color1: str, color2: str) -> float:
        """Calculate WCAG contrast ratio between two colors"""
        try:
            rgb1 = await self._hex_to_rgb(color1)
            rgb2 = await self._hex_to_rgb(color2)
            
            if not rgb1 or not rgb2:
                return 1.0
            
            lum1 = await self._calculate_luminance(rgb1)
            lum2 = await self._calculate_luminance(rgb2)
            
            lighter = max(lum1, lum2)
            darker = min(lum1, lum2)
            
            return (lighter + 0.05) / (darker + 0.05)
            
        except Exception as e:
            logger.error(f"Error calculating contrast ratio: {e}")
            return 1.0
    
    async def _calculate_luminance(self, rgb: Tuple[int, int, int]) -> float:
        """Calculate relative luminance"""
        try:
            def linearize(c):
                c = c / 255.0
                if c <= 0.03928:
                    return c / 12.92
                else:
                    return ((c + 0.055) / 1.055) ** 2.4
            
            r, g, b = rgb
            r_lin = linearize(r)
            g_lin = linearize(g)
            b_lin = linearize(b)
            
            return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
            
        except Exception as e:
            logger.error(f"Error calculating luminance: {e}")
            return 0.5
    
    async def _hex_to_rgb(self, hex_color: str) -> Optional[Tuple[int, int, int]]:
        """Convert hex color to RGB"""
        try:
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            elif len(hex_color) == 3:
                return tuple(int(hex_color[i], 16) * 17 for i in range(3))
            return None
        except Exception:
            return None
    
    async def _extract_font_size(self, font_size_str: str) -> float:
        """Extract numeric font size from string"""
        try:
            # Extract numeric value from font size string
            numbers = re.findall(r'\d+\.?\d*', font_size_str)
            return float(numbers[0]) if numbers else 16.0
        except Exception:
            return 16.0
    
    async def _is_large_text(self, font_size: float, font_weight: str) -> bool:
        """Determine if text qualifies as large text"""
        try:
            # WCAG definition: 18pt+ or 14pt+ bold
            if font_size >= self.large_text_size:
                return True
            
            if font_size >= self.large_text_bold_size:
                bold_weights = ["bold", "600", "700", "800", "900"]
                return font_weight.lower() in bold_weights
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining text size: {e}")
            return False
    
    async def _evaluate_alt_text_quality(self, alt_text: str, component: Dict[str, Any]) -> float:
        """Evaluate quality of alternative text"""
        try:
            score = 1.0
            
            # Check length (not too short or too long)
            if len(alt_text) < 3:
                score -= 0.5  # Too short
            elif len(alt_text) > 125:
                score -= 0.3  # Too long
            
            # Check for redundant phrases
            redundant_phrases = ["image of", "picture of", "graphic of", "logo of"]
            if any(phrase in alt_text.lower() for phrase in redundant_phrases):
                score -= 0.2
            
            # Check for file extensions
            file_extensions = [".jpg", ".png", ".gif", ".svg", ".jpeg"]
            if any(ext in alt_text.lower() for ext in file_extensions):
                score -= 0.3
            
            # Check if it's just a filename
            if re.match(r'^[a-zA-Z0-9_-]+\.(jpg|png|gif|svg|jpeg)$', alt_text.lower()):
                score -= 0.5
            
            return max(0.0, score)
            
        except Exception as e:
            logger.error(f"Error evaluating alt text quality: {e}")
            return 0.5
    
    async def _check_text_hierarchy(self, comp_id: str, component: Dict[str, Any], 
                                  all_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check text hierarchy compliance"""
        issues = []
        
        # This would implement more sophisticated hierarchy checking
        # For now, return empty list
        
        return issues
    
    async def _check_text_spacing(self, comp_id: str, component: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check text spacing compliance"""
        issues = []
        
        # This would check line height, letter spacing, etc.
        # For now, return empty list
        
        return issues
    
    async def _calculate_readability_score(self, components: Dict[str, Any]) -> float:
        """Calculate overall readability score"""
        # Simplified readability calculation
        return 0.8
    
    async def _check_html_semantic_structure(self, html_code: str) -> List[Dict[str, Any]]:
        """Check HTML semantic structure"""
        issues = []
        
        # Check for div soup (too many divs)
        div_count = html_code.count('<div')
        semantic_count = sum(html_code.count(f'<{tag}') for tag in ['header', 'main', 'section', 'article', 'nav', 'aside', 'footer'])
        
        if div_count > 5 and semantic_count == 0:
            issues.append({
                "type": "div_soup",
                "description": "Overuse of div elements without semantic alternatives",
                "severity": "minor",
                "guideline": "HTML5 semantic elements"
            })
        
        return issues
    
    async def _estimate_keyboard_accessibility(self, components: Dict[str, Any]) -> bool:
        """Estimate keyboard accessibility"""
        # Check if interactive elements have proper attributes
        interactive_elements = [comp for comp in components.values() if comp.get("type") == "button"]
        
        for component in interactive_elements:
            accessibility = component.get("accessibility", {})
            tab_index = accessibility.get("tab_index")
            
            # If any interactive element has tabindex="-1", it's not keyboard accessible
            if tab_index == -1:
                return False
        
        return len(interactive_elements) > 0  # If there are interactive elements, assume accessible
    
    async def _check_tab_order(self, components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check logical tab order"""
        issues = []
        
        # Get interactive elements with their positions and tab indices
        interactive_elements = []
        for comp_id, component in components.items():
            if component.get("type") == "button":
                accessibility = component.get("accessibility", {})
                position = component.get("position", {})
                
                interactive_elements.append({
                    "id": comp_id,
                    "x": position.get("x", 0),
                    "y": position.get("y", 0),
                    "tab_index": accessibility.get("tab_index", 0)
                })
        
        # Check if tab order follows visual order (simplified)
        if len(interactive_elements) > 1:
            # Sort by position (top to bottom, left to right)
            visual_order = sorted(interactive_elements, key=lambda x: (x["y"], x["x"]))
            
            # Sort by tab index
            tab_order = sorted([elem for elem in interactive_elements if elem["tab_index"] > 0], 
                             key=lambda x: x["tab_index"])
            
            # If explicit tab indices are used, check if they follow logical order
            if tab_order and len(tab_order) > 1:
                for i in range(len(tab_order) - 1):
                    current_elem = tab_order[i]
                    next_elem = tab_order[i + 1]
                    
                    # Simple check: next element should be after current in visual order
                    current_visual_pos = visual_order.index(next(e for e in visual_order if e["id"] == current_elem["id"]))
                    next_visual_pos = visual_order.index(next(e for e in visual_order if e["id"] == next_elem["id"]))
                    
                    if next_visual_pos < current_visual_pos:
                        issues.append({
                            "type": "illogical_tab_order",
                            "description": f"Tab order doesn't follow visual layout between {current_elem['id']} and {next_elem['id']}",
                            "severity": "minor",
                            "guideline": "WCAG 2.1 - 2.4.3 Focus Order"
                        })
        
        return issues
    
    async def _check_heading_structure(self, html_code: str) -> List[Dict[str, Any]]:
        """Check heading structure compliance"""
        issues = []
        
        # Find all headings
        headings = re.findall(r'<h([1-6])[^>]*>', html_code)
        
        if headings:
            heading_levels = [int(h) for h in headings]
            
            # Check if starts with h1
            if heading_levels and heading_levels[0] != 1:
                issues.append({
                    "type": "heading_structure_invalid",
                    "description": "Heading structure should start with h1",
                    "severity": "minor",
                    "guideline": "WCAG 2.1 - 1.3.1 Info and Relationships"
                })
            
            # Check for skipped levels
            for i in range(len(heading_levels) - 1):
                if heading_levels[i + 1] - heading_levels[i] > 1:
                    issues.append({
                        "type": "heading_level_skipped",
                        "description": f"Heading level skipped from h{heading_levels[i]} to h{heading_levels[i + 1]}",
                        "severity": "minor",
                        "guideline": "WCAG 2.1 - 1.3.1 Info and Relationships"
                    })
        
        return issues
    
    async def _check_list_structure(self, html_code: str) -> List[Dict[str, Any]]:
        """Check list structure compliance"""
        issues = []
        
        # This would check for proper ul/ol/li structure
        # For now, return empty list
        
        return issues
    
    async def _calculate_average_contrast(self, components: Dict[str, Any]) -> float:
        """Calculate average contrast ratio"""
        contrast_ratios = []
        
        for component in components.values():
            if component.get("type") == "text":
                styling = component.get("styling", {}).get("base", {})
                text_color = styling.get("color", "#000000")
                bg_color = styling.get("background_color", "#ffffff")
                
                contrast_ratio = await self._calculate_contrast_ratio(text_color, bg_color)
                contrast_ratios.append(contrast_ratio)
        
        return sum(contrast_ratios) / len(contrast_ratios) if contrast_ratios else 0.0
    
    async def _calculate_accessibility_score(self, *analyses) -> float:
        """Calculate overall accessibility score"""
        scores = [analysis.get("score", 0) for analysis in analyses]
        weights = [0.3, 0.2, 0.2, 0.15, 0.1, 0.05]  # Contrast, text, alt-text, navigation, focus, semantic
        
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        return min(1.0, max(0.0, weighted_score))
    
    async def _compile_accessibility_violations(self, *analyses) -> List[Dict[str, Any]]:
        """Compile all accessibility violations"""
        all_violations = []
        
        for analysis in analyses:
            violations = analysis.get("violations", [])
            all_violations.extend(violations)
        
        # Sort by severity
        severity_order = {"critical": 0, "major": 1, "minor": 2}
        all_violations.sort(key=lambda x: severity_order.get(x.get("severity", "minor"), 2))
        
        return all_violations
    
    async def _generate_accessibility_recommendations(self, violations: List[Dict[str, Any]], 
                                                    overall_score: float) -> List[Dict[str, Any]]:
        """Generate accessibility improvement recommendations"""
        recommendations = []
        
        # Critical violations
        critical_violations = [v for v in violations if v.get("severity") == "critical"]
        if critical_violations:
            recommendations.append({
                "type": "critical_accessibility_fixes",
                "description": "Address critical accessibility barriers immediately",
                "priority": "high",
                "actions": [v.get("description") for v in critical_violations[:3]]
            })
        
        # Major violations
        major_violations = [v for v in violations if v.get("severity") == "major"]
        if major_violations:
            recommendations.append({
                "type": "major_accessibility_improvements",
                "description": "Resolve major accessibility issues",
                "priority": "medium",
                "actions": [v.get("description") for v in major_violations[:3]]
            })
        
        # Overall score recommendations
        if overall_score < 0.5:
            recommendations.append({
                "type": "comprehensive_accessibility_review",
                "description": "Design requires comprehensive accessibility review",
                "priority": "high",
                "actions": ["Conduct full WCAG audit", "Implement accessibility best practices", "Test with assistive technologies"]
            })
        
        return recommendations
    
    async def _determine_compliance_level(self, violations: List[Dict[str, Any]], 
                                        overall_score: float) -> str:
        """Determine WCAG compliance level achieved"""
        critical_violations = [v for v in violations if v.get("severity") == "critical"]
        major_violations = [v for v in violations if v.get("severity") == "major"]
        
        if critical_violations or overall_score < 0.5:
            return "Non-compliant"
        elif major_violations or overall_score < 0.7:
            return "Partial compliance"
        elif overall_score < 0.9:
            return "WCAG A"
        elif overall_score < 0.95:
            return "WCAG AA"
        else:
            return "WCAG AAA"
    
    async def _generate_compliance_summary(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate compliance summary"""
        critical_count = len([v for v in violations if v.get("severity") == "critical"])
        major_count = len([v for v in violations if v.get("severity") == "major"])
        minor_count = len([v for v in violations if v.get("severity") == "minor"])
        
        return {
            "total_violations": len(violations),
            "critical": critical_count,
            "major": major_count,
            "minor": minor_count,
            "most_common_issues": await self._get_most_common_issues(violations)
        }
    
    async def _get_most_common_issues(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Get most common accessibility issues"""
        issue_counts = {}
        
        for violation in violations:
            issue_type = violation.get("type", "unknown")
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Sort by frequency and return top 3
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [issue[0] for issue in sorted_issues[:3]]
    
    async def _load_wcag_guidelines(self):
        """Load WCAG guidelines and standards"""
        # This would load WCAG guidelines from external source
        pass
    
    async def _initialize_accessibility_tools(self):
        """Initialize accessibility evaluation tools"""
        # This would initialize accessibility testing libraries
        pass
