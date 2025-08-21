"""
Typography Manager

Handles font selection, sizing, hierarchy, and text optimization
for banner advertisements with AI-driven typography intelligence.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
from structlog import get_logger

logger = get_logger(__name__)


class TypographyManager:
    """
    AI-powered typography optimization system
    
    Capabilities:
    - Font pairing and selection
    - Hierarchy and sizing optimization
    - Readability analysis
    - Brand-consistent typography
    - Multi-language support
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Typography databases
        self.font_library = self._load_font_library()
        self.font_pairings = self._load_font_pairings()
        self.typography_rules = self._load_typography_rules()
        
        # AI integration
        self.llm_interface = None
        self.mllm_interface = None
        
        # Communication
        self.shared_memory = None
        self.message_queue = None
        
        logger.info("Typography Manager initialized")
    
    async def initialize(self):
        """Initialize the typography manager"""
        try:
            # Load typography models
            await self._load_typography_models()
            
            # Initialize readability analyzers
            await self._initialize_readability_analyzers()
            
            logger.info("Typography Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Typography Manager: {e}")
            raise
    
    def set_communication(self, shared_memory, message_queue):
        """Set communication interfaces"""
        self.shared_memory = shared_memory
        self.message_queue = message_queue
    
    async def optimize_typography(self, strategic_direction: Dict[str, Any],
                                layout_data: Dict[str, Any],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize typography for the banner design
        
        Args:
            strategic_direction: Brand strategy and design direction
            layout_data: Layout specification with zones and hierarchy
            context: Design context and requirements
            
        Returns:
            Typography specification with fonts, sizes, and spacing
        """
        try:
            logger.info("Starting typography optimization")
            
            # Extract requirements
            brand_guidelines = strategic_direction.get("brand_guidelines", {})
            visual_style = strategic_direction.get("visual_style", "modern")
            mood = strategic_direction.get("mood", "professional")
            
            # Analyze text content
            text_content = await self._analyze_text_content(context)
            
            # Select primary and secondary fonts
            font_selection = await self._select_optimal_fonts(
                brand_guidelines, visual_style, mood, text_content
            )
            
            # Calculate optimal sizes and hierarchy
            size_hierarchy = await self._calculate_size_hierarchy(
                layout_data, text_content, context
            )
            
            # Optimize spacing and line heights
            spacing_optimization = await self._optimize_spacing(
                font_selection, size_hierarchy, layout_data
            )
            
            # Ensure readability
            readability_adjustments = await self._ensure_readability(
                font_selection, size_hierarchy, spacing_optimization, layout_data
            )
            
            # Create typography specification
            typography_spec = {
                "success": True,
                "typography_id": f"typo_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "fonts": font_selection,
                "hierarchy": size_hierarchy,
                "spacing": spacing_optimization,
                "readability": readability_adjustments,
                "text_content": text_content,
                "metadata": {
                    "style_influence": visual_style,
                    "mood_influence": mood,
                    "optimization_method": "ai_driven",
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            
            logger.info("Typography optimization completed successfully")
            return typography_spec
            
        except Exception as e:
            logger.error(f"Error optimizing typography: {e}")
            return {
                "success": False,
                "error": str(e),
                "typography_id": None
            }
    
    async def _analyze_text_content(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text content and requirements"""
        try:
            # Extract text elements from context
            primary_message = context.get("primary_message", "")
            company_name = context.get("company_name", "")
            cta_text = context.get("cta_text", "")
            key_messages = context.get("key_messages", [])
            
            # Analyze text characteristics
            analysis = {
                "elements": {
                    "primary_message": {
                        "text": primary_message,
                        "length": len(primary_message),
                        "word_count": len(primary_message.split()),
                        "priority": "high",
                        "purpose": "headline"
                    },
                    "company_name": {
                        "text": company_name,
                        "length": len(company_name),
                        "word_count": len(company_name.split()),
                        "priority": "high",
                        "purpose": "brand"
                    },
                    "cta_text": {
                        "text": cta_text,
                        "length": len(cta_text),
                        "word_count": len(cta_text.split()),
                        "priority": "high",
                        "purpose": "action"
                    }
                },
                "characteristics": {
                    "total_text_volume": "medium",
                    "language": self._detect_language(primary_message),
                    "tone": self._analyze_tone(primary_message),
                    "complexity": self._analyze_complexity(primary_message)
                },
                "requirements": {
                    "hierarchy_levels": self._determine_hierarchy_levels(context),
                    "emphasis_points": self._identify_emphasis_points(context),
                    "readability_priority": "high"
                }
            }
            
            # Add key messages if present
            if key_messages:
                for i, message in enumerate(key_messages):
                    analysis["elements"][f"key_message_{i+1}"] = {
                        "text": message,
                        "length": len(message),
                        "word_count": len(message.split()),
                        "priority": "medium",
                        "purpose": "supporting"
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing text content: {e}")
            return {"elements": {}, "characteristics": {}, "requirements": {}}
    
    def _detect_language(self, text: str) -> str:
        """Detect text language"""
        # Simple detection - in production, use proper language detection
        if re.search(r'[àáâãäåăạảấầẩẫậắằẳẵặèéêëếềểễệìíîïịỉđòóôõöốồổỗộớờởỡợùúûüụủứừửữựỳýỵỷỹ]', text.lower()):
            return "vietnamese"
        elif re.search(r'[a-zA-Z]', text):
            return "english"
        else:
            return "unknown"
    
    def _analyze_tone(self, text: str) -> str:
        """Analyze text tone"""
        text_lower = text.lower()
        
        # Simple tone analysis
        if any(word in text_lower for word in ["exciting", "amazing", "fantastic", "wow", "!"]):
            return "energetic"
        elif any(word in text_lower for word in ["professional", "quality", "trust", "expert"]):
            return "professional"
        elif any(word in text_lower for word in ["friendly", "welcome", "enjoy", "happy"]):
            return "friendly"
        else:
            return "neutral"
    
    def _analyze_complexity(self, text: str) -> str:
        """Analyze text complexity"""
        words = text.split()
        if len(words) <= 3:
            return "simple"
        elif len(words) <= 7:
            return "medium"
        else:
            return "complex"
    
    def _determine_hierarchy_levels(self, context: Dict[str, Any]) -> int:
        """Determine number of hierarchy levels needed"""
        elements = 0
        
        if context.get("primary_message"):
            elements += 1
        if context.get("company_name"):
            elements += 1
        if context.get("cta_text"):
            elements += 1
        if context.get("key_messages"):
            elements += len(context["key_messages"])
        
        # Map elements to hierarchy levels
        if elements <= 2:
            return 2
        elif elements <= 4:
            return 3
        else:
            return 4
    
    def _identify_emphasis_points(self, context: Dict[str, Any]) -> List[str]:
        """Identify text elements that need emphasis"""
        emphasis_points = []
        
        if context.get("primary_message"):
            emphasis_points.append("primary_message")
        if context.get("cta_text"):
            emphasis_points.append("cta_text")
        
        return emphasis_points
    
    async def _select_optimal_fonts(self, brand_guidelines: Dict[str, Any],
                                  visual_style: str, mood: str,
                                  text_content: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal font combinations"""
        try:
            # Get brand font preferences
            brand_fonts = brand_guidelines.get("fonts", {})
            
            # Select primary font (for headlines)
            primary_font = await self._select_primary_font(
                brand_fonts, visual_style, mood, text_content
            )
            
            # Select secondary font (for body text)
            secondary_font = await self._select_secondary_font(
                primary_font, brand_fonts, visual_style, text_content
            )
            
            # Select accent font (for CTA and special elements)
            accent_font = await self._select_accent_font(
                primary_font, secondary_font, visual_style, mood
            )
            
            font_selection = {
                "primary": primary_font,
                "secondary": secondary_font,
                "accent": accent_font,
                "fallbacks": self._get_font_fallbacks(),
                "web_safe_alternatives": self._get_web_safe_alternatives(
                    [primary_font, secondary_font, accent_font]
                )
            }
            
            return font_selection
            
        except Exception as e:
            logger.error(f"Error selecting fonts: {e}")
            return self._get_default_font_selection()
    
    async def _select_primary_font(self, brand_fonts: Dict[str, Any],
                                 visual_style: str, mood: str,
                                 text_content: Dict[str, Any]) -> Dict[str, Any]:
        """Select primary font for headlines"""
        # Check if brand specifies a primary font
        if brand_fonts.get("primary"):
            return {
                "family": brand_fonts["primary"],
                "weight": brand_fonts.get("primary_weight", "bold"),
                "style": "normal",
                "source": "brand_guidelines"
            }
        
        # Select based on visual style and mood
        style_mood_fonts = {
            "modern_professional": {"family": "Inter", "weight": "bold"},
            "modern_friendly": {"family": "Poppins", "weight": "semibold"},
            "modern_energetic": {"family": "Montserrat", "weight": "bold"},
            "classic_professional": {"family": "Playfair Display", "weight": "bold"},
            "classic_friendly": {"family": "Merriweather", "weight": "bold"},
            "minimalist_professional": {"family": "Roboto", "weight": "medium"},
            "minimalist_friendly": {"family": "Open Sans", "weight": "semibold"},
            "bold_energetic": {"family": "Oswald", "weight": "bold"},
            "bold_professional": {"family": "Roboto Condensed", "weight": "bold"}
        }
        
        key = f"{visual_style}_{mood}"
        selected = style_mood_fonts.get(key, {"family": "Inter", "weight": "bold"})
        
        return {
            "family": selected["family"],
            "weight": selected["weight"],
            "style": "normal",
            "source": "ai_selected"
        }
    
    async def _select_secondary_font(self, primary_font: Dict[str, Any],
                                   brand_fonts: Dict[str, Any],
                                   visual_style: str,
                                   text_content: Dict[str, Any]) -> Dict[str, Any]:
        """Select secondary font for body text"""
        # Check brand guidelines
        if brand_fonts.get("secondary"):
            return {
                "family": brand_fonts["secondary"],
                "weight": brand_fonts.get("secondary_weight", "normal"),
                "style": "normal",
                "source": "brand_guidelines"
            }
        
        # Select complementary font
        primary_family = primary_font["family"]
        
        # Font pairing rules
        complementary_fonts = {
            "Inter": "Source Sans Pro",
            "Poppins": "Open Sans",
            "Montserrat": "Lato",
            "Playfair Display": "Source Sans Pro",
            "Merriweather": "Open Sans",
            "Roboto": "Roboto",
            "Open Sans": "Open Sans",
            "Oswald": "Open Sans",
            "Roboto Condensed": "Roboto"
        }
        
        secondary_family = complementary_fonts.get(primary_family, "Open Sans")
        
        return {
            "family": secondary_family,
            "weight": "normal",
            "style": "normal",
            "source": "ai_paired"
        }
    
    async def _select_accent_font(self, primary_font: Dict[str, Any],
                                secondary_font: Dict[str, Any],
                                visual_style: str, mood: str) -> Dict[str, Any]:
        """Select accent font for special elements"""
        # For CTA buttons and special emphasis
        if mood == "energetic":
            return {
                "family": primary_font["family"],
                "weight": "bold",
                "style": "normal",
                "source": "primary_variant"
            }
        else:
            return {
                "family": secondary_font["family"],
                "weight": "semibold",
                "style": "normal",
                "source": "secondary_variant"
            }
    
    def _get_font_fallbacks(self) -> Dict[str, List[str]]:
        """Get font fallback chains"""
        return {
            "sans_serif": ["Arial", "Helvetica", "sans-serif"],
            "serif": ["Times New Roman", "serif"],
            "monospace": ["Courier New", "monospace"]
        }
    
    def _get_web_safe_alternatives(self, fonts: List[Dict[str, Any]]) -> Dict[str, str]:
        """Get web-safe alternatives for fonts"""
        alternatives = {}
        
        for font in fonts:
            family = font["family"]
            # Map to web-safe alternatives
            if family in ["Inter", "Roboto", "Open Sans", "Lato"]:
                alternatives[family] = "Arial, sans-serif"
            elif family in ["Playfair Display", "Merriweather"]:
                alternatives[family] = "Times New Roman, serif"
            else:
                alternatives[family] = "Arial, sans-serif"
        
        return alternatives
    
    def _get_default_font_selection(self) -> Dict[str, Any]:
        """Get default font selection as fallback"""
        return {
            "primary": {"family": "Inter", "weight": "bold", "style": "normal", "source": "default"},
            "secondary": {"family": "Open Sans", "weight": "normal", "style": "normal", "source": "default"},
            "accent": {"family": "Inter", "weight": "semibold", "style": "normal", "source": "default"},
            "fallbacks": self._get_font_fallbacks(),
            "web_safe_alternatives": {"Inter": "Arial, sans-serif", "Open Sans": "Arial, sans-serif"}
        }
    
    async def _calculate_size_hierarchy(self, layout_data: Dict[str, Any],
                                      text_content: Dict[str, Any],
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal font sizes and hierarchy"""
        try:
            # Get layout dimensions
            dimensions = context.get("dimensions", {"width": 800, "height": 600})
            base_size = self._calculate_base_font_size(dimensions)
            
            # Define hierarchy multipliers
            hierarchy_multipliers = {
                "h1": 2.5,    # Primary headline
                "h2": 2.0,    # Secondary headline
                "h3": 1.5,    # Subheadline
                "body": 1.0,  # Body text
                "small": 0.875, # Small text
                "cta": 1.25   # Call-to-action
            }
            
            # Calculate sizes based on content requirements
            hierarchy = {}
            
            # Map content elements to hierarchy levels
            text_elements = text_content.get("elements", {})
            
            for element_name, element_data in text_elements.items():
                purpose = element_data.get("purpose", "body")
                priority = element_data.get("priority", "medium")
                
                # Determine hierarchy level
                if purpose == "headline" and priority == "high":
                    level = "h1"
                elif purpose == "brand" and priority == "high":
                    level = "h2"
                elif purpose == "action" and priority == "high":
                    level = "cta"
                elif purpose == "supporting":
                    level = "h3"
                else:
                    level = "body"
                
                hierarchy[element_name] = {
                    "level": level,
                    "size": round(base_size * hierarchy_multipliers[level]),
                    "line_height": self._calculate_line_height(level),
                    "letter_spacing": self._calculate_letter_spacing(level),
                    "purpose": purpose,
                    "priority": priority
                }
            
            return {
                "base_size": base_size,
                "scale_ratio": 1.25,  # Modular scale ratio
                "hierarchy": hierarchy,
                "responsive_scaling": self._calculate_responsive_scaling(base_size)
            }
            
        except Exception as e:
            logger.error(f"Error calculating size hierarchy: {e}")
            return {"base_size": 16, "hierarchy": {}}
    
    def _calculate_base_font_size(self, dimensions: Dict[str, int]) -> int:
        """Calculate base font size based on banner dimensions"""
        width = dimensions.get("width", 800)
        height = dimensions.get("height", 600)
        
        # Calculate base size relative to banner dimensions
        area = width * height
        
        if area < 200000:  # Small banners
            return 14
        elif area < 500000:  # Medium banners
            return 16
        else:  # Large banners
            return 18
    
    def _calculate_line_height(self, hierarchy_level: str) -> float:
        """Calculate line height for hierarchy level"""
        line_heights = {
            "h1": 1.1,
            "h2": 1.2,
            "h3": 1.3,
            "body": 1.4,
            "small": 1.4,
            "cta": 1.2
        }
        
        return line_heights.get(hierarchy_level, 1.4)
    
    def _calculate_letter_spacing(self, hierarchy_level: str) -> str:
        """Calculate letter spacing for hierarchy level"""
        letter_spacings = {
            "h1": "-0.02em",
            "h2": "-0.01em",
            "h3": "0em",
            "body": "0em",
            "small": "0.01em",
            "cta": "0.05em"
        }
        
        return letter_spacings.get(hierarchy_level, "0em")
    
    def _calculate_responsive_scaling(self, base_size: int) -> Dict[str, Dict[str, float]]:
        """Calculate responsive scaling factors"""
        return {
            "mobile": {
                "scale_factor": 0.8,
                "min_size": 12,
                "max_size": base_size * 2
            },
            "tablet": {
                "scale_factor": 0.9,
                "min_size": 14,
                "max_size": base_size * 2.5
            },
            "desktop": {
                "scale_factor": 1.0,
                "min_size": base_size,
                "max_size": base_size * 3
            }
        }
    
    async def _optimize_spacing(self, font_selection: Dict[str, Any],
                              size_hierarchy: Dict[str, Any],
                              layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize text spacing and layout"""
        try:
            base_size = size_hierarchy.get("base_size", 16)
            
            spacing = {
                "paragraph_spacing": base_size * 0.75,
                "section_spacing": base_size * 1.5,
                "word_spacing": "normal",
                "character_spacing": self._calculate_optimal_character_spacing(font_selection),
                "margins": {
                    "top": base_size * 0.5,
                    "bottom": base_size * 0.5,
                    "left": base_size * 0.25,
                    "right": base_size * 0.25
                },
                "padding": {
                    "text_blocks": base_size * 0.5,
                    "buttons": {
                        "horizontal": base_size * 1.5,
                        "vertical": base_size * 0.75
                    }
                }
            }
            
            return spacing
            
        except Exception as e:
            logger.error(f"Error optimizing spacing: {e}")
            return {"paragraph_spacing": 12, "section_spacing": 24}
    
    def _calculate_optimal_character_spacing(self, font_selection: Dict[str, Any]) -> Dict[str, str]:
        """Calculate optimal character spacing for different fonts"""
        character_spacing = {}
        
        for font_type, font_data in font_selection.items():
            if font_type in ["primary", "secondary", "accent"]:
                family = font_data.get("family", "")
                weight = font_data.get("weight", "normal")
                
                # Adjust spacing based on font characteristics
                if weight in ["bold", "black"]:
                    spacing = "-0.01em"
                elif "Condensed" in family or "Narrow" in family:
                    spacing = "0.02em"
                else:
                    spacing = "0em"
                
                character_spacing[font_type] = spacing
        
        return character_spacing
    
    async def _ensure_readability(self, font_selection: Dict[str, Any],
                                size_hierarchy: Dict[str, Any],
                                spacing_optimization: Dict[str, Any],
                                layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure text readability and accessibility"""
        try:
            readability = {
                "contrast_requirements": {
                    "minimum_ratio": 4.5,
                    "preferred_ratio": 7.0,
                    "large_text_minimum": 3.0
                },
                "size_requirements": {
                    "minimum_body_size": 14,
                    "minimum_button_size": 16,
                    "maximum_line_length": 75  # characters
                },
                "accessibility_features": {
                    "high_contrast_mode": True,
                    "dyslexia_friendly": True,
                    "screen_reader_optimized": True
                },
                "recommendations": await self._generate_readability_recommendations(
                    font_selection, size_hierarchy, spacing_optimization
                )
            }
            
            return readability
            
        except Exception as e:
            logger.error(f"Error ensuring readability: {e}")
            return {"contrast_requirements": {"minimum_ratio": 4.5}}
    
    async def _generate_readability_recommendations(self, font_selection: Dict[str, Any],
                                                  size_hierarchy: Dict[str, Any],
                                                  spacing_optimization: Dict[str, Any]) -> List[str]:
        """Generate readability improvement recommendations"""
        recommendations = []
        
        # Check font sizes
        hierarchy = size_hierarchy.get("hierarchy", {})
        for element, specs in hierarchy.items():
            size = specs.get("size", 16)
            purpose = specs.get("purpose", "")
            
            if purpose == "body" and size < 14:
                recommendations.append(f"Increase {element} font size to at least 14px for better readability")
            
            if purpose == "action" and size < 16:
                recommendations.append(f"Increase {element} font size to at least 16px for better accessibility")
        
        # Check font choices
        primary_font = font_selection.get("primary", {}).get("family", "")
        if "Condensed" in primary_font:
            recommendations.append("Consider using a regular width font for better readability")
        
        # Check spacing
        paragraph_spacing = spacing_optimization.get("paragraph_spacing", 0)
        if paragraph_spacing < 12:
            recommendations.append("Increase paragraph spacing for better text flow")
        
        return recommendations
    
    def _load_font_library(self) -> Dict[str, Any]:
        """Load font library and metadata"""
        return {
            "google_fonts": [
                "Inter", "Open Sans", "Roboto", "Poppins", "Montserrat",
                "Lato", "Source Sans Pro", "Oswald", "Playfair Display",
                "Merriweather", "Roboto Condensed"
            ],
            "system_fonts": [
                "Arial", "Helvetica", "Times New Roman", "Georgia",
                "Verdana", "Tahoma", "Trebuchet MS"
            ],
            "categories": {
                "sans_serif": ["Inter", "Open Sans", "Roboto", "Poppins"],
                "serif": ["Playfair Display", "Merriweather", "Georgia"],
                "display": ["Oswald", "Montserrat", "Roboto Condensed"]
            }
        }
    
    def _load_font_pairings(self) -> Dict[str, List[str]]:
        """Load optimal font pairing combinations"""
        return {
            "Inter": ["Source Sans Pro", "Open Sans", "Roboto"],
            "Poppins": ["Open Sans", "Lato", "Source Sans Pro"],
            "Montserrat": ["Lato", "Open Sans", "Source Sans Pro"],
            "Playfair Display": ["Source Sans Pro", "Open Sans", "Lato"],
            "Oswald": ["Open Sans", "Lato", "Source Sans Pro"]
        }
    
    def _load_typography_rules(self) -> Dict[str, Any]:
        """Load typography design rules and principles"""
        return {
            "hierarchy_rules": {
                "max_levels": 4,
                "size_ratio_min": 1.125,
                "size_ratio_max": 1.618
            },
            "readability_rules": {
                "min_font_size": 12,
                "max_line_length": 75,
                "min_line_height": 1.2,
                "max_line_height": 1.8
            },
            "contrast_rules": {
                "min_contrast_ratio": 4.5,
                "preferred_contrast_ratio": 7.0,
                "large_text_min_ratio": 3.0
            }
        }
    
    async def _load_typography_models(self):
        """Load AI models for typography optimization"""
        # This would load trained models for typography optimization
        pass
    
    async def _initialize_readability_analyzers(self):
        """Initialize readability analysis tools"""
        # This would initialize readability analysis algorithms
        pass
