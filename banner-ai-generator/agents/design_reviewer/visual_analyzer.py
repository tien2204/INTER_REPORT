"""
Visual Analyzer

Analyzes visual hierarchy, composition, and design principles
to evaluate the effectiveness of banner design elements.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import math
from structlog import get_logger

logger = get_logger(__name__)


class VisualAnalyzer:
    """
    Visual design analysis system
    
    Capabilities:
    - Visual hierarchy evaluation
    - Composition analysis
    - Color harmony assessment
    - Typography evaluation
    - Balance and proportion analysis
    - Eye-tracking simulation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Analysis configuration
        self.hierarchy_weights = config.get("hierarchy_weights", {
            "size": 0.3,
            "position": 0.25,
            "color": 0.2,
            "contrast": 0.15,
            "typography": 0.1
        })
        
        self.composition_rules = config.get("composition_rules", [
            "rule_of_thirds",
            "golden_ratio",
            "visual_balance",
            "focal_point",
            "white_space"
        ])
        
        # Visual analysis thresholds
        self.min_contrast_ratio = config.get("min_contrast_ratio", 4.5)
        self.max_elements_per_quadrant = config.get("max_elements_per_quadrant", 3)
        self.ideal_focal_points = config.get("ideal_focal_points", 1)
        
        # Communication
        self.shared_memory = None
        self.message_queue = None
        
        logger.info("Visual Analyzer initialized")
    
    async def initialize(self):
        """Initialize the visual analyzer"""
        try:
            # Load visual design principles and rules
            await self._load_design_principles()
            await self._initialize_color_analysis()
            
            logger.info("Visual Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Visual Analyzer: {e}")
            raise
    
    def set_communication(self, shared_memory, message_queue):
        """Set communication interfaces"""
        self.shared_memory = shared_memory
        self.message_queue = message_queue
    
    async def analyze_design(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive visual analysis of the design
        
        Args:
            design_data: Complete design data including blueprint and generated code
            
        Returns:
            Visual analysis result with scores, findings, and recommendations
        """
        try:
            logger.info("Starting visual design analysis")
            
            # Extract design components
            blueprint = design_data.get("blueprint", {})
            components = blueprint.get("components", {})
            structure = blueprint.get("structure", {})
            styling = blueprint.get("styling", {})
            
            # Perform different types of visual analysis
            hierarchy_analysis = await self._analyze_visual_hierarchy(components, styling)
            composition_analysis = await self._analyze_composition(components, structure)
            color_analysis = await self._analyze_color_harmony(components, styling)
            typography_analysis = await self._analyze_typography(components, styling)
            balance_analysis = await self._analyze_visual_balance(components, structure)
            
            # Calculate overall visual score
            overall_score = await self._calculate_visual_score(
                hierarchy_analysis, composition_analysis, color_analysis,
                typography_analysis, balance_analysis
            )
            
            # Generate findings and recommendations
            findings = await self._compile_findings(
                hierarchy_analysis, composition_analysis, color_analysis,
                typography_analysis, balance_analysis
            )
            
            recommendations = await self._generate_recommendations(findings, overall_score)
            
            result = {
                "overall_score": overall_score,
                "detailed_scores": {
                    "hierarchy": hierarchy_analysis.get("score", 0),
                    "composition": composition_analysis.get("score", 0),
                    "color_harmony": color_analysis.get("score", 0),
                    "typography": typography_analysis.get("score", 0),
                    "visual_balance": balance_analysis.get("score", 0)
                },
                "findings": findings,
                "recommendations": recommendations,
                "analysis_details": {
                    "hierarchy": hierarchy_analysis,
                    "composition": composition_analysis,
                    "color": color_analysis,
                    "typography": typography_analysis,
                    "balance": balance_analysis
                }
            }
            
            logger.info("Visual analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in visual analysis: {e}")
            return {
                "overall_score": 0.0,
                "error": str(e),
                "findings": [{"type": "error", "description": f"Analysis failed: {e}", "severity": "critical"}],
                "recommendations": []
            }
    
    async def _analyze_visual_hierarchy(self, components: Dict[str, Any], 
                                      styling: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual hierarchy of design elements"""
        try:
            hierarchy_elements = []
            
            # Analyze each component for hierarchy factors
            for comp_id, component in components.items():
                element_analysis = await self._analyze_element_hierarchy(comp_id, component)
                hierarchy_elements.append(element_analysis)
            
            # Sort elements by calculated hierarchy strength
            hierarchy_elements.sort(key=lambda x: x.get("hierarchy_strength", 0), reverse=True)
            
            # Evaluate hierarchy effectiveness
            hierarchy_score = await self._evaluate_hierarchy_effectiveness(hierarchy_elements)
            
            # Check for hierarchy issues
            issues = await self._identify_hierarchy_issues(hierarchy_elements)
            
            return {
                "score": hierarchy_score,
                "hierarchy_order": [elem["component_id"] for elem in hierarchy_elements],
                "hierarchy_elements": hierarchy_elements,
                "issues": issues,
                "primary_focal_point": hierarchy_elements[0]["component_id"] if hierarchy_elements else None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing visual hierarchy: {e}")
            return {"score": 0.0, "error": str(e)}
    
    async def _analyze_element_hierarchy(self, comp_id: str, component: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze hierarchy factors for a single element"""
        try:
            position = component.get("position", {})
            dimensions = component.get("dimensions", {})
            styling = component.get("styling", {}).get("base", {})
            content = component.get("content", {})
            
            # Size factor (larger = higher hierarchy)
            area = dimensions.get("width", 0) * dimensions.get("height", 0)
            size_factor = min(1.0, area / 50000)  # Normalize to canvas size
            
            # Position factor (top-left = higher hierarchy in western reading)
            x_pos = position.get("x", 0)
            y_pos = position.get("y", 0)
            position_factor = max(0, 1.0 - (x_pos + y_pos) / 1000)
            
            # Color factor (high contrast = higher hierarchy)
            color_factor = await self._calculate_color_prominence(styling)
            
            # Typography factor (larger, bolder = higher hierarchy)
            typography_factor = await self._calculate_typography_hierarchy(styling, content)
            
            # Z-index factor
            z_index = position.get("z_index", 1)
            z_factor = min(1.0, z_index / 10)
            
            # Calculate weighted hierarchy strength
            hierarchy_strength = (
                size_factor * self.hierarchy_weights.get("size", 0.3) +
                position_factor * self.hierarchy_weights.get("position", 0.25) +
                color_factor * self.hierarchy_weights.get("color", 0.2) +
                typography_factor * self.hierarchy_weights.get("typography", 0.1) +
                z_factor * 0.15
            )
            
            return {
                "component_id": comp_id,
                "component_type": component.get("type"),
                "hierarchy_strength": hierarchy_strength,
                "factors": {
                    "size": size_factor,
                    "position": position_factor,
                    "color": color_factor,
                    "typography": typography_factor,
                    "z_index": z_factor
                },
                "area": area,
                "position": (x_pos, y_pos)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing element hierarchy for {comp_id}: {e}")
            return {"component_id": comp_id, "hierarchy_strength": 0.0, "error": str(e)}
    
    async def _calculate_color_prominence(self, styling: Dict[str, Any]) -> float:
        """Calculate color prominence factor"""
        try:
            # Get color values
            bg_color = styling.get("background_color", "#ffffff")
            text_color = styling.get("color", "#000000")
            
            # Convert to RGB and calculate contrast
            contrast_ratio = await self._calculate_contrast_ratio(text_color, bg_color)
            
            # Higher contrast = more prominent
            prominence = min(1.0, contrast_ratio / 7.0)  # WCAG AAA level is 7:1
            
            return prominence
            
        except Exception as e:
            logger.error(f"Error calculating color prominence: {e}")
            return 0.5
    
    async def _calculate_typography_hierarchy(self, styling: Dict[str, Any], 
                                            content: Dict[str, Any]) -> float:
        """Calculate typography hierarchy factor"""
        try:
            font_size = styling.get("font_size", "16px")
            font_weight = styling.get("font_weight", "normal")
            text_length = len(content.get("text", ""))
            
            # Extract numeric font size
            size_value = float(''.join(filter(str.isdigit, font_size))) if font_size else 16
            
            # Size factor (larger = higher hierarchy)
            size_factor = min(1.0, size_value / 48)  # 48px as max reference
            
            # Weight factor
            weight_values = {
                "100": 0.1, "200": 0.2, "300": 0.3, "normal": 0.4, "400": 0.4,
                "500": 0.5, "600": 0.6, "bold": 0.7, "700": 0.7, "800": 0.8, "900": 0.9
            }
            weight_factor = weight_values.get(str(font_weight).lower(), 0.4)
            
            # Length factor (shorter text often = higher hierarchy for titles)
            length_factor = max(0.3, 1.0 - text_length / 100) if text_length > 0 else 0.5
            
            return (size_factor * 0.5 + weight_factor * 0.3 + length_factor * 0.2)
            
        except Exception as e:
            logger.error(f"Error calculating typography hierarchy: {e}")
            return 0.5
    
    async def _evaluate_hierarchy_effectiveness(self, hierarchy_elements: List[Dict[str, Any]]) -> float:
        """Evaluate overall hierarchy effectiveness"""
        try:
            if not hierarchy_elements:
                return 0.0
            
            # Check for clear primary element
            if len(hierarchy_elements) > 1:
                primary_strength = hierarchy_elements[0].get("hierarchy_strength", 0)
                secondary_strength = hierarchy_elements[1].get("hierarchy_strength", 0)
                
                # Good hierarchy has clear distinction between levels
                distinction = primary_strength - secondary_strength
                distinction_score = min(1.0, distinction / 0.3)  # 30% difference is ideal
            else:
                distinction_score = 1.0
            
            # Check for reasonable distribution
            strengths = [elem.get("hierarchy_strength", 0) for elem in hierarchy_elements]
            avg_strength = sum(strengths) / len(strengths)
            
            # Avoid all elements having same hierarchy
            variance = sum((s - avg_strength) ** 2 for s in strengths) / len(strengths)
            variance_score = min(1.0, variance * 10)  # Encourage variance
            
            # Overall hierarchy score
            hierarchy_score = (distinction_score * 0.6 + variance_score * 0.4)
            
            return min(1.0, hierarchy_score)
            
        except Exception as e:
            logger.error(f"Error evaluating hierarchy effectiveness: {e}")
            return 0.0
    
    async def _identify_hierarchy_issues(self, hierarchy_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify hierarchy-related issues"""
        issues = []
        
        try:
            if not hierarchy_elements:
                issues.append({
                    "type": "hierarchy",
                    "description": "No elements found for hierarchy analysis",
                    "severity": "major"
                })
                return issues
            
            # Check for weak primary element
            primary_strength = hierarchy_elements[0].get("hierarchy_strength", 0)
            if primary_strength < 0.6:
                issues.append({
                    "type": "hierarchy",
                    "description": "Primary element lacks visual dominance",
                    "severity": "major",
                    "element": hierarchy_elements[0].get("component_id")
                })
            
            # Check for too many competing elements
            strong_elements = [elem for elem in hierarchy_elements 
                             if elem.get("hierarchy_strength", 0) > 0.7]
            if len(strong_elements) > 2:
                issues.append({
                    "type": "hierarchy",
                    "description": "Too many visually dominant elements competing for attention",
                    "severity": "minor",
                    "elements": [elem.get("component_id") for elem in strong_elements]
                })
            
            # Check for insufficient hierarchy differences
            if len(hierarchy_elements) > 1:
                for i in range(len(hierarchy_elements) - 1):
                    current = hierarchy_elements[i].get("hierarchy_strength", 0)
                    next_elem = hierarchy_elements[i + 1].get("hierarchy_strength", 0)
                    
                    if abs(current - next_elem) < 0.1:  # Too similar
                        issues.append({
                            "type": "hierarchy",
                            "description": f"Elements have similar visual weight, unclear hierarchy",
                            "severity": "minor",
                            "elements": [
                                hierarchy_elements[i].get("component_id"),
                                hierarchy_elements[i + 1].get("component_id")
                            ]
                        })
            
        except Exception as e:
            logger.error(f"Error identifying hierarchy issues: {e}")
            issues.append({
                "type": "error",
                "description": f"Error in hierarchy analysis: {e}",
                "severity": "major"
            })
        
        return issues
    
    async def _analyze_composition(self, components: Dict[str, Any], 
                                 structure: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze composition and layout principles"""
        try:
            canvas_dims = structure.get("document", {}).get("dimensions", {"width": 800, "height": 600})
            
            # Analyze different composition rules
            rule_of_thirds_score = await self._evaluate_rule_of_thirds(components, canvas_dims)
            golden_ratio_score = await self._evaluate_golden_ratio(components, canvas_dims)
            focal_point_score = await self._evaluate_focal_points(components, canvas_dims)
            white_space_score = await self._evaluate_white_space(components, canvas_dims)
            balance_score = await self._evaluate_visual_balance_simple(components, canvas_dims)
            
            # Calculate overall composition score
            composition_score = (
                rule_of_thirds_score * 0.25 +
                golden_ratio_score * 0.15 +
                focal_point_score * 0.25 +
                white_space_score * 0.20 +
                balance_score * 0.15
            )
            
            return {
                "score": composition_score,
                "rule_scores": {
                    "rule_of_thirds": rule_of_thirds_score,
                    "golden_ratio": golden_ratio_score,
                    "focal_points": focal_point_score,
                    "white_space": white_space_score,
                    "balance": balance_score
                },
                "canvas_utilization": await self._calculate_canvas_utilization(components, canvas_dims),
                "element_distribution": await self._analyze_element_distribution(components, canvas_dims)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing composition: {e}")
            return {"score": 0.0, "error": str(e)}
    
    async def _evaluate_rule_of_thirds(self, components: Dict[str, Any], 
                                     canvas_dims: Dict[str, Any]) -> float:
        """Evaluate adherence to rule of thirds"""
        try:
            width = canvas_dims.get("width", 800)
            height = canvas_dims.get("height", 600)
            
            # Define thirds lines
            third_x1, third_x2 = width / 3, 2 * width / 3
            third_y1, third_y2 = height / 3, 2 * height / 3
            
            # Key intersection points
            intersections = [
                (third_x1, third_y1), (third_x2, third_y1),
                (third_x1, third_y2), (third_x2, third_y2)
            ]
            
            # Check how many important elements are near intersection points
            important_elements = []
            for comp_id, component in components.items():
                position = component.get("position", {})
                x = position.get("x", 0) + component.get("dimensions", {}).get("width", 0) / 2
                y = position.get("y", 0) + component.get("dimensions", {}).get("height", 0) / 2
                
                # Check if near any intersection (within 50px)
                for ix, iy in intersections:
                    distance = math.sqrt((x - ix) ** 2 + (y - iy) ** 2)
                    if distance < 50:
                        important_elements.append(comp_id)
                        break
            
            # Score based on proportion of elements following rule
            total_elements = len(components)
            if total_elements == 0:
                return 0.0
            
            score = len(important_elements) / total_elements
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error evaluating rule of thirds: {e}")
            return 0.0
    
    async def _evaluate_golden_ratio(self, components: Dict[str, Any], 
                                   canvas_dims: Dict[str, Any]) -> float:
        """Evaluate golden ratio proportions"""
        try:
            # Golden ratio ≈ 1.618
            golden_ratio = 1.618
            width = canvas_dims.get("width", 800)
            height = canvas_dims.get("height", 600)
            
            # Check canvas proportions
            canvas_ratio = width / height if height > 0 else 1
            canvas_score = 1.0 - abs(canvas_ratio - golden_ratio) / golden_ratio
            canvas_score = max(0, canvas_score)
            
            # Check element proportions
            element_scores = []
            for component in components.values():
                dims = component.get("dimensions", {})
                elem_width = dims.get("width", 0)
                elem_height = dims.get("height", 0)
                
                if elem_height > 0:
                    elem_ratio = elem_width / elem_height
                    elem_score = 1.0 - abs(elem_ratio - golden_ratio) / golden_ratio
                    element_scores.append(max(0, elem_score))
            
            avg_element_score = sum(element_scores) / len(element_scores) if element_scores else 0
            
            # Combine canvas and element scores
            overall_score = (canvas_score * 0.3 + avg_element_score * 0.7)
            return min(1.0, overall_score)
            
        except Exception as e:
            logger.error(f"Error evaluating golden ratio: {e}")
            return 0.0
    
    async def _evaluate_focal_points(self, components: Dict[str, Any], 
                                   canvas_dims: Dict[str, Any]) -> float:
        """Evaluate focal point effectiveness"""
        try:
            # Identify potential focal points (high hierarchy elements)
            focal_elements = []
            
            for comp_id, component in components.items():
                # Calculate visual weight
                dims = component.get("dimensions", {})
                area = dims.get("width", 0) * dims.get("height", 0)
                
                # Consider size, position, and type
                position = component.get("position", {})
                comp_type = component.get("type")
                
                weight = area
                if comp_type in ["button", "logo"]:
                    weight *= 1.5  # Interactive/branded elements are more focal
                
                if weight > 5000:  # Threshold for significant elements
                    focal_elements.append({
                        "component_id": comp_id,
                        "weight": weight,
                        "position": position
                    })
            
            # Evaluate number and distribution of focal points
            num_focal = len(focal_elements)
            
            if num_focal == 0:
                return 0.2  # No clear focal point
            elif num_focal == 1:
                return 1.0  # Ideal: single focal point
            elif num_focal == 2:
                return 0.8  # Good: secondary focal point
            else:
                return max(0.3, 1.0 - (num_focal - 2) * 0.2)  # Too many focal points
            
        except Exception as e:
            logger.error(f"Error evaluating focal points: {e}")
            return 0.0
    
    async def _evaluate_white_space(self, components: Dict[str, Any], 
                                  canvas_dims: Dict[str, Any]) -> float:
        """Evaluate white space utilization"""
        try:
            total_area = canvas_dims.get("width", 800) * canvas_dims.get("height", 600)
            occupied_area = 0
            
            # Calculate total occupied area
            for component in components.values():
                dims = component.get("dimensions", {})
                area = dims.get("width", 0) * dims.get("height", 0)
                occupied_area += area
            
            # Calculate white space ratio
            white_space_ratio = (total_area - occupied_area) / total_area if total_area > 0 else 0
            
            # Ideal white space is 40-60%
            ideal_min, ideal_max = 0.4, 0.6
            
            if ideal_min <= white_space_ratio <= ideal_max:
                score = 1.0
            elif white_space_ratio < ideal_min:
                # Too crowded
                score = white_space_ratio / ideal_min
            else:
                # Too empty
                score = 1.0 - (white_space_ratio - ideal_max) / (1.0 - ideal_max)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error evaluating white space: {e}")
            return 0.0
    
    async def _evaluate_visual_balance_simple(self, components: Dict[str, Any], 
                                            canvas_dims: Dict[str, Any]) -> float:
        """Evaluate visual balance (simplified calculation)"""
        try:
            width = canvas_dims.get("width", 800)
            height = canvas_dims.get("height", 600)
            center_x, center_y = width / 2, height / 2
            
            # Calculate visual weight distribution
            left_weight = right_weight = top_weight = bottom_weight = 0
            
            for component in components.values():
                position = component.get("position", {})
                dims = component.get("dimensions", {})
                
                x = position.get("x", 0) + dims.get("width", 0) / 2
                y = position.get("y", 0) + dims.get("height", 0) / 2
                weight = dims.get("width", 0) * dims.get("height", 0)
                
                # Distribute weight based on position
                if x < center_x:
                    left_weight += weight
                else:
                    right_weight += weight
                
                if y < center_y:
                    top_weight += weight
                else:
                    bottom_weight += weight
            
            # Calculate balance ratios
            horizontal_balance = min(left_weight, right_weight) / max(left_weight, right_weight) if max(left_weight, right_weight) > 0 else 1.0
            vertical_balance = min(top_weight, bottom_weight) / max(top_weight, bottom_weight) if max(top_weight, bottom_weight) > 0 else 1.0
            
            # Overall balance score
            balance_score = (horizontal_balance + vertical_balance) / 2
            return balance_score
            
        except Exception as e:
            logger.error(f"Error evaluating visual balance: {e}")
            return 0.0
    
    async def _calculate_canvas_utilization(self, components: Dict[str, Any], 
                                          canvas_dims: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate how well the canvas space is utilized"""
        try:
            total_area = canvas_dims.get("width", 800) * canvas_dims.get("height", 600)
            occupied_area = sum(
                comp.get("dimensions", {}).get("width", 0) * comp.get("dimensions", {}).get("height", 0)
                for comp in components.values()
            )
            
            utilization_ratio = occupied_area / total_area if total_area > 0 else 0
            
            return {
                "total_area": total_area,
                "occupied_area": occupied_area,
                "utilization_ratio": utilization_ratio,
                "efficiency_score": min(1.0, utilization_ratio / 0.6)  # 60% utilization is ideal
            }
            
        except Exception as e:
            logger.error(f"Error calculating canvas utilization: {e}")
            return {"utilization_ratio": 0.0, "efficiency_score": 0.0}
    
    async def _analyze_element_distribution(self, components: Dict[str, Any], 
                                          canvas_dims: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how elements are distributed across the canvas"""
        try:
            width = canvas_dims.get("width", 800)
            height = canvas_dims.get("height", 600)
            
            # Divide canvas into quadrants
            quadrants = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
            
            for component in components.values():
                position = component.get("position", {})
                x = position.get("x", 0)
                y = position.get("y", 0)
                
                if x < width / 2 and y < height / 2:
                    quadrants["top_left"] += 1
                elif x >= width / 2 and y < height / 2:
                    quadrants["top_right"] += 1
                elif x < width / 2 and y >= height / 2:
                    quadrants["bottom_left"] += 1
                else:
                    quadrants["bottom_right"] += 1
            
            # Calculate distribution score
            values = list(quadrants.values())
            max_elements = max(values) if values else 0
            min_elements = min(values) if values else 0
            
            distribution_score = (min_elements / max_elements) if max_elements > 0 else 1.0
            
            return {
                "quadrant_distribution": quadrants,
                "distribution_score": distribution_score,
                "max_elements_per_quadrant": max_elements
            }
            
        except Exception as e:
            logger.error(f"Error analyzing element distribution: {e}")
            return {"distribution_score": 0.0}
    
    async def _analyze_color_harmony(self, components: Dict[str, Any], 
                                   styling: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze color harmony and relationships"""
        try:
            # Extract colors from all components
            colors_used = []
            
            for component in components.values():
                comp_styling = component.get("styling", {}).get("base", {})
                
                # Get background color
                bg_color = comp_styling.get("background_color")
                if bg_color:
                    colors_used.append(bg_color)
                
                # Get text color
                text_color = comp_styling.get("color")
                if text_color:
                    colors_used.append(text_color)
            
            # Get global colors
            global_styling = styling.get("global_styles", {}).get("base", {})
            if global_styling.get("background_color"):
                colors_used.append(global_styling["background_color"])
            
            # Remove duplicates
            unique_colors = list(set(colors_used))
            
            # Analyze color relationships
            harmony_score = await self._evaluate_color_relationships(unique_colors)
            contrast_score = await self._evaluate_color_contrast(components)
            
            return {
                "score": (harmony_score * 0.6 + contrast_score * 0.4),
                "colors_used": unique_colors,
                "color_count": len(unique_colors),
                "harmony_score": harmony_score,
                "contrast_score": contrast_score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing color harmony: {e}")
            return {"score": 0.0, "error": str(e)}
    
    async def _evaluate_color_relationships(self, colors: List[str]) -> float:
        """Evaluate color harmony relationships"""
        try:
            if len(colors) <= 1:
                return 1.0  # Single color is always harmonious
            
            # Convert colors to HSL for analysis
            hsl_colors = []
            for color in colors:
                hsl = await self._hex_to_hsl(color)
                if hsl:
                    hsl_colors.append(hsl)
            
            if len(hsl_colors) < 2:
                return 0.5  # Couldn't parse colors
            
            # Check for common color harmonies
            harmony_scores = []
            
            # Monochromatic: same hue, different saturation/lightness
            hues = [hsl[0] for hsl in hsl_colors]
            hue_variance = max(hues) - min(hues)
            if hue_variance < 30:  # Within 30 degrees
                harmony_scores.append(0.9)
            
            # Complementary: opposite hues (180° apart)
            for i, hsl1 in enumerate(hsl_colors):
                for hsl2 in hsl_colors[i+1:]:
                    hue_diff = abs(hsl1[0] - hsl2[0])
                    if 160 <= hue_diff <= 200:  # Near complementary
                        harmony_scores.append(0.8)
            
            # Analogous: adjacent hues (within 60°)
            for i, hsl1 in enumerate(hsl_colors):
                for hsl2 in hsl_colors[i+1:]:
                    hue_diff = abs(hsl1[0] - hsl2[0])
                    if hue_diff <= 60:
                        harmony_scores.append(0.7)
            
            # Triadic: 120° apart
            if len(hsl_colors) >= 3:
                for i, hsl1 in enumerate(hsl_colors):
                    for j, hsl2 in enumerate(hsl_colors[i+1:], i+1):
                        for hsl3 in hsl_colors[j+1:]:
                            diff1 = abs(hsl1[0] - hsl2[0])
                            diff2 = abs(hsl2[0] - hsl3[0])
                            diff3 = abs(hsl3[0] - hsl1[0])
                            
                            if all(100 <= diff <= 140 for diff in [diff1, diff2, diff3]):
                                harmony_scores.append(0.8)
            
            # Return best harmony score, or penalize for too many colors
            if harmony_scores:
                base_score = max(harmony_scores)
            else:
                base_score = 0.5  # No clear harmony
            
            # Penalize for using too many colors
            color_penalty = max(0, (len(colors) - 5) * 0.1)
            
            return max(0.0, base_score - color_penalty)
            
        except Exception as e:
            logger.error(f"Error evaluating color relationships: {e}")
            return 0.5
    
    async def _evaluate_color_contrast(self, components: Dict[str, Any]) -> float:
        """Evaluate color contrast for readability"""
        try:
            contrast_scores = []
            
            for component in components.values():
                styling = component.get("styling", {}).get("base", {})
                bg_color = styling.get("background_color")
                text_color = styling.get("color")
                
                if bg_color and text_color:
                    contrast_ratio = await self._calculate_contrast_ratio(text_color, bg_color)
                    
                    # WCAG standards: 4.5:1 for normal text, 3:1 for large text
                    if contrast_ratio >= 4.5:
                        contrast_scores.append(1.0)
                    elif contrast_ratio >= 3.0:
                        contrast_scores.append(0.7)
                    elif contrast_ratio >= 2.0:
                        contrast_scores.append(0.4)
                    else:
                        contrast_scores.append(0.1)
            
            return sum(contrast_scores) / len(contrast_scores) if contrast_scores else 1.0
            
        except Exception as e:
            logger.error(f"Error evaluating color contrast: {e}")
            return 0.5
    
    async def _calculate_contrast_ratio(self, color1: str, color2: str) -> float:
        """Calculate WCAG contrast ratio between two colors"""
        try:
            # Convert to RGB
            rgb1 = await self._hex_to_rgb(color1)
            rgb2 = await self._hex_to_rgb(color2)
            
            if not rgb1 or not rgb2:
                return 1.0  # Default if conversion fails
            
            # Calculate relative luminance
            lum1 = await self._calculate_luminance(rgb1)
            lum2 = await self._calculate_luminance(rgb2)
            
            # Calculate contrast ratio
            lighter = max(lum1, lum2)
            darker = min(lum1, lum2)
            
            contrast_ratio = (lighter + 0.05) / (darker + 0.05)
            return contrast_ratio
            
        except Exception as e:
            logger.error(f"Error calculating contrast ratio: {e}")
            return 1.0
    
    async def _calculate_luminance(self, rgb: Tuple[int, int, int]) -> float:
        """Calculate relative luminance of RGB color"""
        try:
            r, g, b = rgb
            
            # Convert to sRGB
            def linearize(c):
                c = c / 255.0
                if c <= 0.03928:
                    return c / 12.92
                else:
                    return ((c + 0.055) / 1.055) ** 2.4
            
            r_lin = linearize(r)
            g_lin = linearize(g)
            b_lin = linearize(b)
            
            # Calculate luminance
            luminance = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
            return luminance
            
        except Exception as e:
            logger.error(f"Error calculating luminance: {e}")
            return 0.5
    
    async def _hex_to_rgb(self, hex_color: str) -> Optional[Tuple[int, int, int]]:
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
    
    async def _hex_to_hsl(self, hex_color: str) -> Optional[Tuple[float, float, float]]:
        """Convert hex color to HSL tuple"""
        try:
            rgb = await self._hex_to_rgb(hex_color)
            if not rgb:
                return None
            
            r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
            
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            diff = max_val - min_val
            
            # Lightness
            l = (max_val + min_val) / 2
            
            if diff == 0:
                h = s = 0  # Achromatic
            else:
                # Saturation
                s = diff / (2 - max_val - min_val) if l > 0.5 else diff / (max_val + min_val)
                
                # Hue
                if max_val == r:
                    h = ((g - b) / diff + (6 if g < b else 0)) / 6
                elif max_val == g:
                    h = ((b - r) / diff + 2) / 6
                else:
                    h = ((r - g) / diff + 4) / 6
                
                h *= 360  # Convert to degrees
            
            return (h, s, l)
            
        except Exception as e:
            logger.error(f"Error converting hex to HSL: {e}")
            return None
    
    # Continue with more analysis methods...
    async def _analyze_typography(self, components: Dict[str, Any], 
                                styling: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze typography effectiveness"""
        # Implementation for typography analysis
        return {"score": 0.8}  # Placeholder
    
    async def _analyze_visual_balance(self, components: Dict[str, Any], 
                                    structure: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual balance and weight distribution"""
        # Implementation for visual balance analysis
        return {"score": 0.7}  # Placeholder
    
    async def _calculate_visual_score(self, *analyses) -> float:
        """Calculate overall visual score from all analyses"""
        scores = [analysis.get("score", 0) for analysis in analyses]
        return sum(scores) / len(scores) if scores else 0.0
    
    async def _compile_findings(self, *analyses) -> List[Dict[str, Any]]:
        """Compile findings from all analyses"""
        findings = []
        for analysis in analyses:
            if "issues" in analysis:
                findings.extend(analysis["issues"])
        return findings
    
    async def _generate_recommendations(self, findings: List[Dict[str, Any]], 
                                      overall_score: float) -> List[Dict[str, Any]]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        # Generate recommendations based on score
        if overall_score < 0.6:
            recommendations.append({
                "type": "general",
                "description": "Overall visual design needs significant improvement",
                "priority": "high"
            })
        
        return recommendations
    
    async def _load_design_principles(self):
        """Load design principles and rules"""
        pass
    
    async def _initialize_color_analysis(self):
        """Initialize color analysis tools"""
        pass
