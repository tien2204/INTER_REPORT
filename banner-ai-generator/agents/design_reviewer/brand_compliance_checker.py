"""
Brand Compliance Checker

Evaluates design adherence to brand guidelines, style consistency,
and corporate identity standards.
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import re
from structlog import get_logger

logger = get_logger(__name__)


class BrandComplianceChecker:
    """
    Brand compliance evaluation system
    
    Capabilities:
    - Brand guideline adherence checking
    - Logo usage validation
    - Color palette compliance
    - Typography standards verification
    - Messaging consistency evaluation
    - Style guide enforcement
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Brand guideline configuration
        self.brand_colors = set(config.get("brand_colors", [
            "#FF0000", "#00FF00", "#0000FF"  # Default placeholder colors
        ]))
        
        self.approved_fonts = set(config.get("approved_fonts", [
            "Inter", "Roboto", "Arial", "Helvetica"
        ]))
        
        self.brand_keywords = set(config.get("brand_keywords", []))
        self.prohibited_terms = set(config.get("prohibited_terms", []))
        
        # Logo compliance rules
        self.logo_min_size = config.get("logo_min_size", {"width": 50, "height": 30})
        self.logo_clear_space = config.get("logo_clear_space", 20)
        self.logo_placement_rules = config.get("logo_placement_rules", ["top-left", "top-right"])
        
        # Compliance thresholds
        self.color_tolerance = config.get("color_tolerance", 10)  # RGB tolerance
        self.strict_compliance = config.get("strict_compliance", False)
        
        # Communication
        self.shared_memory = None
        self.message_queue = None
        
        logger.info("Brand Compliance Checker initialized")
    
    async def initialize(self):
        """Initialize the brand compliance checker"""
        try:
            # Load brand guidelines and assets
            await self._load_brand_guidelines()
            await self._load_brand_assets()
            
            logger.info("Brand Compliance Checker initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Brand Compliance Checker: {e}")
            raise
    
    def set_communication(self, shared_memory, message_queue):
        """Set communication interfaces"""
        self.shared_memory = shared_memory
        self.message_queue = message_queue
    
    async def check_compliance(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive brand compliance check
        
        Args:
            design_data: Complete design data including blueprint and assets
            
        Returns:
            Brand compliance analysis with scores, violations, and recommendations
        """
        try:
            logger.info("Starting brand compliance check")
            
            # Extract design components
            blueprint = design_data.get("blueprint", {})
            components = blueprint.get("components", {})
            styling = blueprint.get("styling", {})
            strategy = design_data.get("strategy", {})
            
            # Perform different compliance checks
            logo_compliance = await self._check_logo_compliance(components, strategy)
            color_compliance = await self._check_color_compliance(components, styling)
            typography_compliance = await self._check_typography_compliance(components, styling)
            messaging_compliance = await self._check_messaging_compliance(components, strategy)
            layout_compliance = await self._check_layout_compliance(components, blueprint)
            
            # Calculate overall compliance score
            overall_score = await self._calculate_compliance_score(
                logo_compliance, color_compliance, typography_compliance,
                messaging_compliance, layout_compliance
            )
            
            # Compile violations and recommendations
            violations = await self._compile_violations(
                logo_compliance, color_compliance, typography_compliance,
                messaging_compliance, layout_compliance
            )
            
            recommendations = await self._generate_compliance_recommendations(violations, overall_score)
            
            result = {
                "overall_score": overall_score,
                "compliance_scores": {
                    "logo": logo_compliance.get("score", 0),
                    "color": color_compliance.get("score", 0),
                    "typography": typography_compliance.get("score", 0),
                    "messaging": messaging_compliance.get("score", 0),
                    "layout": layout_compliance.get("score", 0)
                },
                "violations": violations,
                "recommendations": recommendations,
                "compliance_details": {
                    "logo": logo_compliance,
                    "color": color_compliance,
                    "typography": typography_compliance,
                    "messaging": messaging_compliance,
                    "layout": layout_compliance
                },
                "brand_alignment": await self._assess_brand_alignment(strategy, violations)
            }
            
            logger.info("Brand compliance check completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in brand compliance check: {e}")
            return {
                "overall_score": 0.0,
                "error": str(e),
                "violations": [{"type": "error", "description": f"Compliance check failed: {e}", "severity": "critical"}],
                "recommendations": []
            }
    
    async def _check_logo_compliance(self, components: Dict[str, Any], 
                                   strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Check logo usage compliance"""
        try:
            logo_components = [
                comp for comp in components.values() 
                if comp.get("type") == "logo"
            ]
            
            violations = []
            compliance_score = 1.0
            
            if not logo_components:
                violations.append({
                    "type": "logo_missing",
                    "description": "No logo found in design",
                    "severity": "major",
                    "guideline": "Brand logo must be present in all marketing materials"
                })
                compliance_score = 0.3
            else:
                for logo in logo_components:
                    # Check logo size compliance
                    dimensions = logo.get("dimensions", {})
                    width = dimensions.get("width", 0)
                    height = dimensions.get("height", 0)
                    
                    if width < self.logo_min_size["width"] or height < self.logo_min_size["height"]:
                        violations.append({
                            "type": "logo_size",
                            "description": f"Logo too small: {width}x{height}px, minimum: {self.logo_min_size['width']}x{self.logo_min_size['height']}px",
                            "severity": "major",
                            "guideline": "Logo must meet minimum size requirements for legibility"
                        })
                        compliance_score -= 0.3
                    
                    # Check logo placement
                    position = logo.get("position", {})
                    placement = await self._determine_logo_placement(position, dimensions)
                    
                    if placement not in self.logo_placement_rules:
                        violations.append({
                            "type": "logo_placement",
                            "description": f"Logo placed in {placement}, allowed positions: {', '.join(self.logo_placement_rules)}",
                            "severity": "minor",
                            "guideline": "Logo should be positioned according to brand guidelines"
                        })
                        compliance_score -= 0.2
                    
                    # Check clear space around logo
                    clear_space_violations = await self._check_logo_clear_space(logo, components)
                    if clear_space_violations:
                        violations.extend(clear_space_violations)
                        compliance_score -= 0.2
                    
                    # Check logo quality and format
                    content = logo.get("content", {})
                    source = content.get("source", "")
                    
                    if not source:
                        violations.append({
                            "type": "logo_source",
                            "description": "Logo source not specified",
                            "severity": "major",
                            "guideline": "Logo must reference approved brand asset"
                        })
                        compliance_score -= 0.3
                    elif not await self._is_approved_logo_source(source):
                        violations.append({
                            "type": "logo_unauthorized",
                            "description": "Logo source not from approved brand assets",
                            "severity": "major",
                            "guideline": "Only approved logo versions should be used"
                        })
                        compliance_score -= 0.4
            
            return {
                "score": max(0.0, compliance_score),
                "violations": violations,
                "logo_count": len(logo_components),
                "placement_analysis": await self._analyze_logo_placement(logo_components)
            }
            
        except Exception as e:
            logger.error(f"Error checking logo compliance: {e}")
            return {"score": 0.0, "error": str(e), "violations": []}
    
    async def _determine_logo_placement(self, position: Dict[str, Any], 
                                      dimensions: Dict[str, Any]) -> str:
        """Determine logo placement zone"""
        try:
            x = position.get("x", 0)
            y = position.get("y", 0)
            
            # Simple placement determination
            if x < 100 and y < 100:
                return "top-left"
            elif x > 600 and y < 100:
                return "top-right"
            elif x < 100 and y > 400:
                return "bottom-left"
            elif x > 600 and y > 400:
                return "bottom-right"
            elif y < 150:
                return "top-center"
            elif y > 450:
                return "bottom-center"
            elif x < 200:
                return "left-center"
            elif x > 600:
                return "right-center"
            else:
                return "center"
                
        except Exception as e:
            logger.error(f"Error determining logo placement: {e}")
            return "unknown"
    
    async def _check_logo_clear_space(self, logo: Dict[str, Any], 
                                    all_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if logo has adequate clear space"""
        violations = []
        
        try:
            logo_pos = logo.get("position", {})
            logo_dims = logo.get("dimensions", {})
            
            logo_x = logo_pos.get("x", 0)
            logo_y = logo_pos.get("y", 0)
            logo_w = logo_dims.get("width", 0)
            logo_h = logo_dims.get("height", 0)
            
            # Define logo bounding box with clear space
            clear_space = self.logo_clear_space
            logo_clear_box = {
                "x1": logo_x - clear_space,
                "y1": logo_y - clear_space,
                "x2": logo_x + logo_w + clear_space,
                "y2": logo_y + logo_h + clear_space
            }
            
            # Check for overlapping elements
            for comp_id, component in all_components.items():
                if component.get("type") == "logo":
                    continue  # Skip other logos
                
                comp_pos = component.get("position", {})
                comp_dims = component.get("dimensions", {})
                
                comp_x = comp_pos.get("x", 0)
                comp_y = comp_pos.get("y", 0)
                comp_w = comp_dims.get("width", 0)
                comp_h = comp_dims.get("height", 0)
                
                # Check for intersection
                if (comp_x < logo_clear_box["x2"] and comp_x + comp_w > logo_clear_box["x1"] and
                    comp_y < logo_clear_box["y2"] and comp_y + comp_h > logo_clear_box["y1"]):
                    
                    violations.append({
                        "type": "logo_clear_space",
                        "description": f"Element '{comp_id}' violates logo clear space requirement",
                        "severity": "minor",
                        "guideline": f"Maintain {clear_space}px clear space around logo",
                        "violating_element": comp_id
                    })
            
        except Exception as e:
            logger.error(f"Error checking logo clear space: {e}")
        
        return violations
    
    async def _is_approved_logo_source(self, source: str) -> bool:
        """Check if logo source is from approved brand assets"""
        try:
            # Check against approved logo patterns
            approved_patterns = [
                r"brand\.assets/logo",
                r"assets/brand/logo",
                r"cdn\.company\.com/brand",
                r"approved\.logos"
            ]
            
            for pattern in approved_patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking logo source approval: {e}")
            return False
    
    async def _analyze_logo_placement(self, logo_components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze logo placement effectiveness"""
        try:
            if not logo_components:
                return {"effectiveness": 0.0, "issues": ["No logos found"]}
            
            placements = []
            for logo in logo_components:
                position = logo.get("position", {})
                dimensions = logo.get("dimensions", {})
                placement = await self._determine_logo_placement(position, dimensions)
                placements.append(placement)
            
            # Analyze placement effectiveness
            preferred_placements = ["top-left", "top-right", "top-center"]
            effective_placements = [p for p in placements if p in preferred_placements]
            
            effectiveness = len(effective_placements) / len(placements) if placements else 0
            
            return {
                "effectiveness": effectiveness,
                "placements": placements,
                "preferred_count": len(effective_placements),
                "total_count": len(placements)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing logo placement: {e}")
            return {"effectiveness": 0.0, "error": str(e)}
    
    async def _check_color_compliance(self, components: Dict[str, Any], 
                                    styling: Dict[str, Any]) -> Dict[str, Any]:
        """Check color palette compliance"""
        try:
            used_colors = set()
            violations = []
            
            # Extract colors from components
            for component in components.values():
                comp_styling = component.get("styling", {}).get("base", {})
                
                bg_color = comp_styling.get("background_color")
                if bg_color:
                    used_colors.add(bg_color.upper())
                
                text_color = comp_styling.get("color")
                if text_color:
                    used_colors.add(text_color.upper())
            
            # Extract global colors
            global_colors = styling.get("global_styles", {}).get("base", {})
            if global_colors.get("background_color"):
                used_colors.add(global_colors["background_color"].upper())
            if global_colors.get("color"):
                used_colors.add(global_colors["color"].upper())
            
            # Check compliance with brand colors
            non_compliant_colors = []
            compliant_colors = []
            
            for color in used_colors:
                if await self._is_brand_color_compliant(color):
                    compliant_colors.append(color)
                else:
                    non_compliant_colors.append(color)
            
            # Calculate compliance score
            total_colors = len(used_colors)
            if total_colors == 0:
                compliance_score = 1.0
            else:
                compliance_score = len(compliant_colors) / total_colors
            
            # Generate violations for non-compliant colors
            for color in non_compliant_colors:
                suggested_color = await self._find_closest_brand_color(color)
                violations.append({
                    "type": "color_non_compliant",
                    "description": f"Color {color} is not in brand palette",
                    "severity": "major" if self.strict_compliance else "minor",
                    "guideline": "Use only approved brand colors",
                    "suggestion": f"Consider using {suggested_color} instead"
                })
            
            # Check for sufficient brand color usage
            brand_color_usage = len([c for c in used_colors if c.upper() in self.brand_colors])
            if brand_color_usage == 0 and len(used_colors) > 1:
                violations.append({
                    "type": "brand_color_missing",
                    "description": "No brand colors used in design",
                    "severity": "major",
                    "guideline": "Include at least one brand color in the design"
                })
                compliance_score *= 0.5
            
            return {
                "score": compliance_score,
                "violations": violations,
                "colors_used": list(used_colors),
                "compliant_colors": compliant_colors,
                "non_compliant_colors": non_compliant_colors,
                "brand_color_coverage": brand_color_usage / len(self.brand_colors) if self.brand_colors else 0
            }
            
        except Exception as e:
            logger.error(f"Error checking color compliance: {e}")
            return {"score": 0.0, "error": str(e), "violations": []}
    
    async def _is_brand_color_compliant(self, color: str) -> bool:
        """Check if color is compliant with brand palette"""
        try:
            color = color.upper()
            
            # Direct match
            if color in self.brand_colors:
                return True
            
            # Check with tolerance
            color_rgb = await self._hex_to_rgb(color)
            if not color_rgb:
                return False
            
            for brand_color in self.brand_colors:
                brand_rgb = await self._hex_to_rgb(brand_color)
                if brand_rgb and await self._colors_within_tolerance(color_rgb, brand_rgb):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking color compliance: {e}")
            return False
    
    async def _colors_within_tolerance(self, rgb1: tuple, rgb2: tuple) -> bool:
        """Check if two RGB colors are within tolerance"""
        try:
            r_diff = abs(rgb1[0] - rgb2[0])
            g_diff = abs(rgb1[1] - rgb2[1])
            b_diff = abs(rgb1[2] - rgb2[2])
            
            return (r_diff <= self.color_tolerance and 
                   g_diff <= self.color_tolerance and 
                   b_diff <= self.color_tolerance)
                   
        except Exception as e:
            logger.error(f"Error checking color tolerance: {e}")
            return False
    
    async def _find_closest_brand_color(self, color: str) -> str:
        """Find the closest brand color to the given color"""
        try:
            color_rgb = await self._hex_to_rgb(color)
            if not color_rgb:
                return list(self.brand_colors)[0] if self.brand_colors else "#000000"
            
            min_distance = float('inf')
            closest_color = list(self.brand_colors)[0] if self.brand_colors else "#000000"
            
            for brand_color in self.brand_colors:
                brand_rgb = await self._hex_to_rgb(brand_color)
                if brand_rgb:
                    # Calculate Euclidean distance in RGB space
                    distance = sum((c1 - c2) ** 2 for c1, c2 in zip(color_rgb, brand_rgb)) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_color = brand_color
            
            return closest_color
            
        except Exception as e:
            logger.error(f"Error finding closest brand color: {e}")
            return list(self.brand_colors)[0] if self.brand_colors else "#000000"
    
    async def _hex_to_rgb(self, hex_color: str) -> Optional[tuple]:
        """Convert hex color to RGB tuple"""
        try:
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            elif len(hex_color) == 3:
                return tuple(int(hex_color[i], 16) * 17 for i in range(3))
            return None
        except Exception:
            return None
    
    async def _check_typography_compliance(self, components: Dict[str, Any], 
                                         styling: Dict[str, Any]) -> Dict[str, Any]:
        """Check typography compliance with brand guidelines"""
        try:
            used_fonts = set()
            violations = []
            
            # Extract fonts from components
            for component in components.values():
                if component.get("type") == "text":
                    comp_styling = component.get("styling", {}).get("base", {})
                    font_family = comp_styling.get("font_family")
                    if font_family:
                        # Extract primary font name
                        primary_font = font_family.split(',')[0].strip().strip('"\'')
                        used_fonts.add(primary_font)
            
            # Extract global fonts
            global_fonts = styling.get("global_styles", {}).get("base", {})
            if global_fonts.get("font_family"):
                primary_font = global_fonts["font_family"].split(',')[0].strip().strip('"\'')
                used_fonts.add(primary_font)
            
            # Check font compliance
            non_compliant_fonts = []
            compliant_fonts = []
            
            for font in used_fonts:
                if font in self.approved_fonts:
                    compliant_fonts.append(font)
                else:
                    non_compliant_fonts.append(font)
            
            # Calculate compliance score
            total_fonts = len(used_fonts)
            if total_fonts == 0:
                compliance_score = 1.0
            else:
                compliance_score = len(compliant_fonts) / total_fonts
            
            # Generate violations
            for font in non_compliant_fonts:
                violations.append({
                    "type": "font_non_compliant",
                    "description": f"Font '{font}' is not in approved typography list",
                    "severity": "major" if self.strict_compliance else "minor",
                    "guideline": "Use only approved brand fonts",
                    "suggestion": f"Consider using {list(self.approved_fonts)[0] if self.approved_fonts else 'Arial'} instead"
                })
            
            # Check for typography hierarchy compliance
            hierarchy_violations = await self._check_typography_hierarchy(components)
            violations.extend(hierarchy_violations)
            
            return {
                "score": max(0.0, compliance_score - len(hierarchy_violations) * 0.1),
                "violations": violations,
                "fonts_used": list(used_fonts),
                "compliant_fonts": compliant_fonts,
                "non_compliant_fonts": non_compliant_fonts,
                "approved_font_usage": len(compliant_fonts) / len(self.approved_fonts) if self.approved_fonts else 0
            }
            
        except Exception as e:
            logger.error(f"Error checking typography compliance: {e}")
            return {"score": 0.0, "error": str(e), "violations": []}
    
    async def _check_typography_hierarchy(self, components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check typography hierarchy compliance"""
        violations = []
        
        try:
            text_components = [comp for comp in components.values() if comp.get("type") == "text"]
            
            # Check for proper hierarchy
            font_sizes = []
            for component in text_components:
                styling = component.get("styling", {}).get("base", {})
                font_size = styling.get("font_size", "16px")
                
                # Extract numeric value
                size_value = float(''.join(filter(str.isdigit, font_size))) if font_size else 16
                font_sizes.append(size_value)
            
            if len(font_sizes) > 1:
                # Check for sufficient size differentiation
                font_sizes.sort(reverse=True)
                for i in range(len(font_sizes) - 1):
                    if font_sizes[i] - font_sizes[i + 1] < 2:  # Less than 2px difference
                        violations.append({
                            "type": "typography_hierarchy",
                            "description": "Insufficient font size differentiation for clear hierarchy",
                            "severity": "minor",
                            "guideline": "Maintain clear size differences between text elements"
                        })
                        break
            
        except Exception as e:
            logger.error(f"Error checking typography hierarchy: {e}")
        
        return violations
    
    async def _check_messaging_compliance(self, components: Dict[str, Any], 
                                        strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Check messaging and content compliance"""
        try:
            all_text = []
            violations = []
            
            # Extract all text content
            for component in components.values():
                if component.get("type") == "text":
                    content = component.get("content", {})
                    text = content.get("text", "")
                    if text:
                        all_text.append(text.lower())
            
            # Check for prohibited terms
            prohibited_found = []
            for text in all_text:
                for term in self.prohibited_terms:
                    if term.lower() in text:
                        prohibited_found.append(term)
            
            if prohibited_found:
                violations.append({
                    "type": "prohibited_terms",
                    "description": f"Prohibited terms found: {', '.join(set(prohibited_found))}",
                    "severity": "critical",
                    "guideline": "Avoid using prohibited terms in brand communications"
                })
            
            # Check for brand keyword inclusion
            brand_keywords_found = []
            for text in all_text:
                for keyword in self.brand_keywords:
                    if keyword.lower() in text:
                        brand_keywords_found.append(keyword)
            
            keyword_score = len(set(brand_keywords_found)) / len(self.brand_keywords) if self.brand_keywords else 1.0
            
            if keyword_score < 0.3 and self.brand_keywords:  # Less than 30% keyword coverage
                violations.append({
                    "type": "brand_keywords_missing",
                    "description": "Low brand keyword coverage in messaging",
                    "severity": "minor",
                    "guideline": "Include relevant brand keywords for consistency"
                })
            
            # Check message tone and style
            tone_violations = await self._check_message_tone(all_text, strategy)
            violations.extend(tone_violations)
            
            # Calculate overall messaging score
            messaging_score = 1.0
            if prohibited_found:
                messaging_score -= 0.5  # Major penalty for prohibited terms
            messaging_score = messaging_score * keyword_score  # Scale by keyword coverage
            messaging_score -= len(tone_violations) * 0.1  # Penalty for tone violations
            
            return {
                "score": max(0.0, messaging_score),
                "violations": violations,
                "prohibited_terms_found": list(set(prohibited_found)),
                "brand_keywords_found": list(set(brand_keywords_found)),
                "keyword_coverage": keyword_score,
                "text_content": all_text
            }
            
        except Exception as e:
            logger.error(f"Error checking messaging compliance: {e}")
            return {"score": 0.0, "error": str(e), "violations": []}
    
    async def _check_message_tone(self, text_content: List[str], 
                                strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check message tone compliance"""
        violations = []
        
        try:
            # Get target tone from strategy
            target_tone = strategy.get("messaging", {}).get("tone", "professional")
            
            # Simple tone analysis (in production, use NLP)
            all_text = " ".join(text_content)
            
            # Check for tone violations based on simple rules
            if target_tone == "professional":
                informal_indicators = ["hey", "awesome", "cool", "amazing", "wow"]
                if any(indicator in all_text for indicator in informal_indicators):
                    violations.append({
                        "type": "tone_mismatch",
                        "description": "Informal language detected in professional context",
                        "severity": "minor",
                        "guideline": "Maintain professional tone in business communications"
                    })
            
            elif target_tone == "casual":
                formal_indicators = ["furthermore", "nevertheless", "henceforth", "whereby"]
                if any(indicator in all_text for indicator in formal_indicators):
                    violations.append({
                        "type": "tone_mismatch",
                        "description": "Overly formal language for casual tone",
                        "severity": "minor",
                        "guideline": "Use casual, approachable language"
                    })
            
        except Exception as e:
            logger.error(f"Error checking message tone: {e}")
        
        return violations
    
    async def _check_layout_compliance(self, components: Dict[str, Any], 
                                     blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """Check layout compliance with brand standards"""
        try:
            violations = []
            
            # Check for required elements
            required_elements = ["text", "logo"]  # Basic requirements
            present_types = set(comp.get("type") for comp in components.values())
            
            missing_elements = []
            for required in required_elements:
                if required not in present_types:
                    missing_elements.append(required)
            
            for missing in missing_elements:
                violations.append({
                    "type": "missing_element",
                    "description": f"Required element type '{missing}' is missing",
                    "severity": "major",
                    "guideline": f"All designs must include {missing} elements"
                })
            
            # Check layout proportions
            proportion_violations = await self._check_layout_proportions(components)
            violations.extend(proportion_violations)
            
            # Calculate layout score
            layout_score = 1.0 - (len(missing_elements) * 0.3 + len(proportion_violations) * 0.1)
            
            return {
                "score": max(0.0, layout_score),
                "violations": violations,
                "missing_elements": missing_elements,
                "present_elements": list(present_types)
            }
            
        except Exception as e:
            logger.error(f"Error checking layout compliance: {e}")
            return {"score": 0.0, "error": str(e), "violations": []}
    
    async def _check_layout_proportions(self, components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check layout proportion compliance"""
        violations = []
        
        try:
            # Check for oversized elements
            for comp_id, component in components.items():
                dimensions = component.get("dimensions", {})
                width = dimensions.get("width", 0)
                height = dimensions.get("height", 0)
                area = width * height
                
                # Check if any single element takes up too much space
                if area > 200000:  # More than ~50% of typical banner
                    violations.append({
                        "type": "element_oversized",
                        "description": f"Element '{comp_id}' is too large ({width}x{height}px)",
                        "severity": "minor",
                        "guideline": "Maintain balanced element proportions",
                        "element": comp_id
                    })
            
        except Exception as e:
            logger.error(f"Error checking layout proportions: {e}")
        
        return violations
    
    async def _calculate_compliance_score(self, *compliance_checks) -> float:
        """Calculate overall compliance score"""
        try:
            scores = [check.get("score", 0) for check in compliance_checks]
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Logo, color, typography, messaging, layout
            
            weighted_score = sum(score * weight for score, weight in zip(scores, weights))
            return min(1.0, max(0.0, weighted_score))
            
        except Exception as e:
            logger.error(f"Error calculating compliance score: {e}")
            return 0.0
    
    async def _compile_violations(self, *compliance_checks) -> List[Dict[str, Any]]:
        """Compile all violations from compliance checks"""
        all_violations = []
        
        for check in compliance_checks:
            violations = check.get("violations", [])
            all_violations.extend(violations)
        
        # Sort by severity
        severity_order = {"critical": 0, "major": 1, "minor": 2}
        all_violations.sort(key=lambda x: severity_order.get(x.get("severity", "minor"), 2))
        
        return all_violations
    
    async def _generate_compliance_recommendations(self, violations: List[Dict[str, Any]], 
                                                 overall_score: float) -> List[Dict[str, Any]]:
        """Generate recommendations to improve compliance"""
        recommendations = []
        
        try:
            # Critical violations - immediate action needed
            critical_violations = [v for v in violations if v.get("severity") == "critical"]
            if critical_violations:
                recommendations.append({
                    "type": "critical_fix",
                    "description": "Address critical brand violations immediately",
                    "priority": "high",
                    "actions": [v.get("description") for v in critical_violations[:3]]
                })
            
            # Major violations - important improvements
            major_violations = [v for v in violations if v.get("severity") == "major"]
            if major_violations:
                recommendations.append({
                    "type": "major_improvements",
                    "description": "Resolve major brand compliance issues",
                    "priority": "medium",
                    "actions": [v.get("description") for v in major_violations[:3]]
                })
            
            # Overall score recommendations
            if overall_score < 0.5:
                recommendations.append({
                    "type": "comprehensive_review",
                    "description": "Design requires comprehensive brand review",
                    "priority": "high",
                    "actions": ["Review brand guidelines", "Redesign with brand focus", "Get brand approval"]
                })
            elif overall_score < 0.8:
                recommendations.append({
                    "type": "moderate_improvements",
                    "description": "Several brand compliance improvements needed",
                    "priority": "medium",
                    "actions": ["Address color compliance", "Review typography choices", "Verify logo usage"]
                })
            
        except Exception as e:
            logger.error(f"Error generating compliance recommendations: {e}")
        
        return recommendations
    
    async def _assess_brand_alignment(self, strategy: Dict[str, Any], 
                                    violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall brand alignment"""
        try:
            # Calculate alignment score based on violations
            critical_count = len([v for v in violations if v.get("severity") == "critical"])
            major_count = len([v for v in violations if v.get("severity") == "major"])
            minor_count = len([v for v in violations if v.get("severity") == "minor"])
            
            # Weighted penalty system
            penalty = critical_count * 0.4 + major_count * 0.2 + minor_count * 0.1
            alignment_score = max(0.0, 1.0 - penalty)
            
            # Determine alignment level
            if alignment_score >= 0.9:
                alignment_level = "excellent"
            elif alignment_score >= 0.7:
                alignment_level = "good"
            elif alignment_score >= 0.5:
                alignment_level = "moderate"
            elif alignment_score >= 0.3:
                alignment_level = "poor"
            else:
                alignment_level = "critical"
            
            return {
                "score": alignment_score,
                "level": alignment_level,
                "violation_summary": {
                    "critical": critical_count,
                    "major": major_count,
                    "minor": minor_count
                },
                "key_issues": [v.get("type") for v in violations if v.get("severity") in ["critical", "major"]]
            }
            
        except Exception as e:
            logger.error(f"Error assessing brand alignment: {e}")
            return {"score": 0.0, "level": "unknown", "error": str(e)}
    
    async def _load_brand_guidelines(self):
        """Load brand guidelines from configuration"""
        # This would load brand guidelines from external source
        pass
    
    async def _load_brand_assets(self):
        """Load approved brand assets"""
        # This would load approved logos, colors, fonts, etc.
        pass
