"""
Prompt Generator for Background Designer

Generates optimized prompts for Text-to-Image models to create
high-quality, text-free backgrounds.
"""

from typing import Any, Dict, List, Optional
from structlog import get_logger

logger = get_logger(__name__)


class PromptGenerator:
    """
    Generates optimized prompts for T2I background generation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Style mappings
        self.mood_styles = {
            "professional": "clean, modern, corporate, sophisticated",
            "friendly": "warm, approachable, welcoming, bright",
            "energetic": "dynamic, vibrant, exciting, bold",
            "elegant": "refined, luxurious, premium, polished",
            "playful": "fun, colorful, creative, lively",
            "trustworthy": "stable, reliable, secure, professional",
            "innovative": "cutting-edge, futuristic, tech-forward, sleek",
            "calm": "peaceful, serene, gentle, soft"
        }
        
        self.industry_styles = {
            "technology": "digital, futuristic, clean lines, gradients",
            "healthcare": "clean, sterile, calming blues and whites",
            "finance": "professional, stable, navy blues, grays",
            "education": "bright, inspiring, growth-oriented",
            "retail": "attractive, product-focused, lifestyle",
            "food": "appetizing, warm, natural textures",
            "travel": "inspiring, adventure, landscapes",
            "automotive": "dynamic, powerful, metallic"
        }
        
        # Negative prompts to avoid text
        self.text_avoidance_negatives = [
            "text", "letters", "words", "writing", "typography",
            "alphabet", "numbers", "symbols", "signs", "labels",
            "captions", "titles", "headlines", "logos with text",
            "readable text", "font", "script", "watermark"
        ]
        
        # Quality enhancement keywords
        self.quality_enhancers = [
            "high quality", "4k", "8k", "ultra detailed",
            "professional photography", "studio lighting",
            "sharp focus", "crisp", "vibrant colors",
            "perfect composition", "artistic"
        ]
    
    async def generate_background_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Generate optimized prompt for background generation
        
        Args:
            prompt_data: Dictionary containing:
                - mood: Target mood/emotion
                - style: Visual style preference
                - colors: Brand color palette
                - industry: Business industry
                - thought: Agent's current thought (optional)
                - ensure_no_text: Whether to emphasize text-free requirement
        
        Returns:
            Optimized prompt string
        """
        try:
            mood = prompt_data.get("mood", "")
            style = prompt_data.get("style", "")
            colors = prompt_data.get("colors", [])
            industry = prompt_data.get("industry", "")
            ensure_no_text = prompt_data.get("ensure_no_text", True)
            
            # Build prompt components
            prompt_parts = []
            
            # Base description
            if industry and industry in self.industry_styles:
                prompt_parts.append(f"Abstract {industry} background,")
                prompt_parts.append(self.industry_styles[industry])
            else:
                prompt_parts.append("Abstract professional background,")
            
            # Mood styling
            if mood and mood in self.mood_styles:
                prompt_parts.append(f"with {self.mood_styles[mood]} aesthetic,")
            
            # Color information
            if colors:
                color_desc = self._generate_color_description(colors)
                prompt_parts.append(f"using {color_desc},")
            
            # Style specifications
            if style:
                prompt_parts.append(f"{style} style,")
            
            # Composition guidelines
            prompt_parts.extend([
                "smooth gradients",
                "subtle patterns",
                "modern design",
                "banner-friendly composition",
                "plenty of space for text overlay"
            ])
            
            # Quality enhancers
            prompt_parts.extend(self.quality_enhancers[:3])  # Use first 3
            
            # Join main prompt
            main_prompt = " ".join(prompt_parts)
            
            # Add negative prompt for text avoidance
            if ensure_no_text:
                negative_prompt = ", ".join(self.text_avoidance_negatives)
                full_prompt = f"{main_prompt} --negative {negative_prompt}"
            else:
                full_prompt = main_prompt
            
            logger.info(f"Generated background prompt: {full_prompt[:100]}...")
            return full_prompt
            
        except Exception as e:
            logger.error(f"Error generating background prompt: {e}")
            return self._get_fallback_prompt()
    
    def _generate_color_description(self, colors: List[str]) -> str:
        """Generate natural color description from hex/color list"""
        try:
            if not colors:
                return "balanced color palette"
            
            # Convert hex colors to color names (simplified)
            color_names = []
            for color in colors[:3]:  # Use max 3 colors
                if isinstance(color, str):
                    if color.startswith('#'):
                        color_name = self._hex_to_color_name(color)
                    else:
                        color_name = color
                    color_names.append(color_name)
            
            if len(color_names) == 1:
                return f"{color_names[0]} tones"
            elif len(color_names) == 2:
                return f"{color_names[0]} and {color_names[1]} palette"
            else:
                return f"{', '.join(color_names[:-1])} and {color_names[-1]} color scheme"
                
        except Exception as e:
            logger.error(f"Error generating color description: {e}")
            return "balanced color palette"
    
    def _hex_to_color_name(self, hex_color: str) -> str:
        """Convert hex color to approximate color name"""
        # Simplified color mapping
        color_mapping = {
            '#FF0000': 'red', '#00FF00': 'green', '#0000FF': 'blue',
            '#FFFF00': 'yellow', '#FF00FF': 'magenta', '#00FFFF': 'cyan',
            '#000000': 'black', '#FFFFFF': 'white', '#808080': 'gray',
            '#FFA500': 'orange', '#800080': 'purple', '#FFC0CB': 'pink',
            '#A52A2A': 'brown', '#008000': 'dark green', '#000080': 'navy'
        }
        
        # Try exact match first
        if hex_color.upper() in color_mapping:
            return color_mapping[hex_color.upper()]
        
        # Simplified analysis based on RGB values
        try:
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Determine dominant color
            if r > g and r > b:
                if r > 200:
                    return "bright red"
                elif r > 100:
                    return "red"
                else:
                    return "dark red"
            elif g > r and g > b:
                if g > 200:
                    return "bright green"
                elif g > 100:
                    return "green"
                else:
                    return "dark green"
            elif b > r and b > g:
                if b > 200:
                    return "bright blue"
                elif b > 100:
                    return "blue"
                else:
                    return "dark blue"
            else:
                # Mixed colors
                if r > 150 and g > 150 and b < 100:
                    return "yellow"
                elif r > 150 and b > 150 and g < 100:
                    return "purple"
                elif g > 150 and b > 150 and r < 100:
                    return "cyan"
                else:
                    return "neutral"
                    
        except ValueError:
            return "custom color"
    
    def _get_fallback_prompt(self) -> str:
        """Get fallback prompt when generation fails"""
        return (
            "Abstract professional background, clean modern design, "
            "smooth gradients, subtle patterns, high quality, "
            "banner-friendly composition, plenty of space for text overlay "
            "--negative text, letters, words, writing, typography"
        )
    
    async def refine_prompt_based_on_feedback(self, 
                                             original_prompt: str,
                                             feedback: Dict[str, Any]) -> str:
        """
        Refine prompt based on validation feedback
        
        Args:
            original_prompt: The original prompt that was used
            feedback: Validation feedback with issues found
        
        Returns:
            Refined prompt
        """
        try:
            issues = feedback.get("issues", [])
            quality_score = feedback.get("quality_score", 0)
            
            refinements = []
            
            # Address specific issues
            for issue in issues:
                if "text detected" in issue.lower():
                    refinements.append("completely text-free")
                    refinements.append("no typography elements")
                elif "low contrast" in issue.lower():
                    refinements.append("high contrast")
                    refinements.append("vibrant colors")
                elif "busy" in issue.lower() or "cluttered" in issue.lower():
                    refinements.append("minimalist")
                    refinements.append("clean composition")
                elif "blurry" in issue.lower():
                    refinements.append("sharp focus")
                    refinements.append("crisp details")
            
            if refinements:
                # Add refinements to original prompt
                refined_prompt = f"{original_prompt}, {', '.join(refinements)}"
            else:
                # General quality improvement
                refined_prompt = f"{original_prompt}, enhanced quality, perfect composition"
            
            logger.info(f"Refined prompt based on feedback: {refined_prompt[:100]}...")
            return refined_prompt
            
        except Exception as e:
            logger.error(f"Error refining prompt: {e}")
            return original_prompt
    
    def get_aspect_ratio_prompt_modifier(self, width: int, height: int) -> str:
        """Get prompt modifier based on aspect ratio"""
        try:
            ratio = width / height
            
            if ratio > 2.0:
                return "wide banner format, horizontal composition"
            elif ratio > 1.5:
                return "landscape format, balanced horizontal layout"
            elif ratio < 0.5:
                return "vertical banner format, tall composition"
            elif ratio < 0.7:
                return "portrait format, vertical layout"
            else:
                return "square format, centered composition"
                
        except (ZeroDivisionError, TypeError):
            return "balanced composition"
