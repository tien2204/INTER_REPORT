"""
Brand Analyzer

Analyzes brand information from brief and assets to understand
brand personality, positioning, and creative direction.
"""

from typing import Any, Dict, List, Optional
from structlog import get_logger

logger = get_logger(__name__)


class BrandAnalyzer:
    """
    Brand analyzer that extracts brand personality and positioning
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.brand_archetypes = self._load_brand_archetypes()
        self.industry_traits = self._load_industry_traits()
    
    async def analyze_brand(self, brief_analysis: Dict[str, Any], 
                          brand_assets: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze brand characteristics and define brand strategy
        
        Args:
            brief_analysis: Analyzed campaign brief
            brand_assets: Processed brand assets (logo, images, etc.)
            
        Returns:
            Comprehensive brand analysis
        """
        try:
            brand_analysis = {
                "brand_personality": {},
                "brand_positioning": {},
                "visual_identity": {},
                "tone_of_voice": {},
                "brand_archetype": {},
                "competitive_positioning": {},
                "mood_board": [],
                "brand_guidelines": {},
                "creative_direction": {}
            }
            
            # Extract brand info from brief
            extracted_info = brief_analysis.get("extracted_info", {})
            creative_direction = brief_analysis.get("creative_direction", {})
            
            # Analyze brand personality
            brand_analysis["brand_personality"] = await self._analyze_brand_personality(
                extracted_info, creative_direction
            )
            
            # Determine brand archetype
            brand_analysis["brand_archetype"] = await self._determine_brand_archetype(
                extracted_info, brand_analysis["brand_personality"]
            )
            
            # Analyze visual identity
            if brand_assets:
                brand_analysis["visual_identity"] = await self._analyze_visual_identity(brand_assets)
            
            # Define tone of voice
            brand_analysis["tone_of_voice"] = await self._define_tone_of_voice(
                extracted_info, creative_direction
            )
            
            # Analyze brand positioning
            brand_analysis["brand_positioning"] = await self._analyze_brand_positioning(
                extracted_info, brand_analysis["brand_archetype"]
            )
            
            # Generate mood board
            brand_analysis["mood_board"] = await self._generate_mood_board(
                brand_analysis["brand_personality"],
                brand_analysis["visual_identity"],
                creative_direction
            )
            
            # Create brand guidelines
            brand_analysis["brand_guidelines"] = await self._create_brand_guidelines(
                brand_analysis
            )
            
            # Define creative direction
            brand_analysis["creative_direction"] = await self._define_creative_direction(
                brand_analysis, extracted_info
            )
            
            logger.info("Brand analysis completed successfully")
            return brand_analysis
            
        except Exception as e:
            logger.error(f"Brand analysis failed: {e}")
            raise
    
    async def _analyze_brand_personality(self, extracted_info: Dict[str, Any],
                                       creative_direction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze brand personality traits"""
        
        # Big Five personality dimensions for brands
        personality_traits = {
            "sincerity": 0.0,      # Down-to-earth, honest, wholesome, cheerful
            "excitement": 0.0,      # Daring, spirited, imaginative, up-to-date
            "competence": 0.0,      # Reliable, intelligent, successful
            "sophistication": 0.0,  # Upper class, charming
            "ruggedness": 0.0       # Outdoorsy, tough
        }
        
        # Analyze from industry
        industry = extracted_info.get("industry") or extracted_info.get("detected_industry", "general")
        if industry in self.industry_traits:
            industry_personality = self.industry_traits[industry]["personality"]
            for trait, weight in industry_personality.items():
                personality_traits[trait] += weight * 0.3
        
        # Analyze from mood/tone
        mood = creative_direction.get("mood", {}).get("primary_mood", "professional")
        tone = creative_direction.get("tone", {}).get("primary", "professional")
        
        # Map mood to personality traits
        mood_personality_mapping = {
            "professional": {"competence": 0.8, "sincerity": 0.6},
            "playful": {"excitement": 0.9, "sincerity": 0.7},
            "elegant": {"sophistication": 0.9, "competence": 0.6},
            "trustworthy": {"sincerity": 0.9, "competence": 0.8},
            "innovative": {"excitement": 0.8, "competence": 0.7},
            "warm": {"sincerity": 0.8, "excitement": 0.5}
        }
        
        if mood in mood_personality_mapping:
            for trait, weight in mood_personality_mapping[mood].items():
                personality_traits[trait] += weight * 0.4
        
        # Map tone to personality traits
        tone_personality_mapping = {
            "professional": {"competence": 0.7, "sincerity": 0.5},
            "friendly": {"sincerity": 0.8, "excitement": 0.6},
            "confident": {"competence": 0.8, "excitement": 0.6},
            "playful": {"excitement": 0.9, "sincerity": 0.6},
            "sophisticated": {"sophistication": 0.9, "competence": 0.7}
        }
        
        if tone in tone_personality_mapping:
            for trait, weight in tone_personality_mapping[tone].items():
                personality_traits[trait] += weight * 0.3
        
        # Normalize scores
        max_score = max(personality_traits.values()) if personality_traits.values() else 1.0
        if max_score > 0:
            personality_traits = {k: min(v / max_score, 1.0) for k, v in personality_traits.items()}
        
        # Determine dominant traits
        sorted_traits = sorted(personality_traits.items(), key=lambda x: x[1], reverse=True)
        dominant_traits = [trait for trait, score in sorted_traits[:2] if score > 0.5]
        
        return {
            "traits": personality_traits,
            "dominant_traits": dominant_traits,
            "personality_summary": self._generate_personality_summary(dominant_traits),
            "brand_character": self._determine_brand_character(personality_traits)
        }
    
    async def _determine_brand_archetype(self, extracted_info: Dict[str, Any],
                                       personality: Dict[str, Any]) -> Dict[str, Any]:
        """Determine brand archetype based on personality and context"""
        
        # Score each archetype based on personality traits and context
        archetype_scores = {}
        
        for archetype_name, archetype_data in self.brand_archetypes.items():
            score = 0.0
            
            # Score based on personality alignment
            for trait, trait_score in personality["traits"].items():
                archetype_trait_weight = archetype_data["personality_weights"].get(trait, 0)
                score += trait_score * archetype_trait_weight
            
            # Score based on industry fit
            industry = extracted_info.get("industry") or extracted_info.get("detected_industry")
            if industry in archetype_data.get("common_industries", []):
                score += 0.2
            
            # Score based on company size/type (if available)
            company_type = extracted_info.get("company_type", "").lower()
            if company_type in archetype_data.get("company_types", []):
                score += 0.1
            
            archetype_scores[archetype_name] = score
        
        # Get top archetype
        primary_archetype = max(archetype_scores, key=archetype_scores.get)
        
        # Get secondary archetypes
        sorted_archetypes = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
        secondary_archetypes = [name for name, score in sorted_archetypes[1:3] if score > 0.3]
        
        return {
            "primary": primary_archetype,
            "secondary": secondary_archetypes,
            "scores": archetype_scores,
            "archetype_description": self.brand_archetypes[primary_archetype]["description"],
            "brand_promise": self.brand_archetypes[primary_archetype]["brand_promise"],
            "communication_style": self.brand_archetypes[primary_archetype]["communication_style"]
        }
    
    async def _analyze_visual_identity(self, brand_assets: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual identity from brand assets"""
        visual_identity = {
            "color_palette": {},
            "typography_style": {},
            "visual_style": {},
            "logo_characteristics": {},
            "design_principles": {}
        }
        
        # Analyze logo if available
        if "logo" in brand_assets:
            logo_data = brand_assets["logo"]
            
            # Extract color information
            if "colors" in logo_data:
                visual_identity["color_palette"] = logo_data["colors"]
            
            # Extract style information
            if "style" in logo_data:
                visual_identity["logo_characteristics"] = logo_data["style"]
                visual_identity["visual_style"] = self._infer_visual_style_from_logo(logo_data["style"])
            
            # Extract design principles
            visual_identity["design_principles"] = self._extract_design_principles(logo_data)
        
        # Analyze other visual assets
        for asset_type, asset_data in brand_assets.items():
            if asset_type != "logo" and isinstance(asset_data, dict):
                if "colors" in asset_data:
                    # Merge color palettes
                    self._merge_color_palettes(visual_identity["color_palette"], asset_data["colors"])
        
        return visual_identity
    
    def _infer_visual_style_from_logo(self, logo_style: Dict[str, Any]) -> Dict[str, Any]:
        """Infer overall visual style from logo characteristics"""
        style_category = logo_style.get("style_category", "modern")
        complexity = logo_style.get("complexity", "medium")
        content_type = logo_style.get("content_type", "mixed")
        
        # Map logo characteristics to visual style
        style_mapping = {
            "minimal": {
                "aesthetic": "clean_modern",
                "complexity_preference": "simple",
                "visual_hierarchy": "clear",
                "spacing": "generous"
            },
            "geometric": {
                "aesthetic": "structured_modern",
                "complexity_preference": "balanced",
                "visual_hierarchy": "geometric",
                "spacing": "structured"
            },
            "detailed": {
                "aesthetic": "rich_detailed",
                "complexity_preference": "complex",
                "visual_hierarchy": "layered",
                "spacing": "compact"
            },
            "wordmark": {
                "aesthetic": "typographic",
                "complexity_preference": "simple",
                "visual_hierarchy": "text_focused",
                "spacing": "text_optimized"
            }
        }
        
        return style_mapping.get(style_category, style_mapping["minimal"])
    
    def _extract_design_principles(self, logo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract design principles from logo analysis"""
        style = logo_data.get("style", {})
        placement = logo_data.get("placement", {})
        
        return {
            "minimalism": style.get("is_minimal", False),
            "geometric_precision": style.get("geometric_elements", False),
            "scalability": placement.get("size_flexibility", {}).get("scales_well_small", True),
            "versatility": len(placement.get("optimal_placements", [])) > 3,
            "contrast_focus": placement.get("contrast_requirements", {}).get("needs_background_contrast", False)
        }
    
    def _merge_color_palettes(self, main_palette: Dict[str, Any], 
                            secondary_palette: Dict[str, Any]):
        """Merge color palettes from multiple assets"""
        if not main_palette:
            main_palette.update(secondary_palette)
            return
        
        # Add secondary colors if not already present
        main_colors = set(main_palette.get("palette", []))
        for color in secondary_palette.get("palette", []):
            if color not in main_colors:
                main_palette.setdefault("secondary_colors", []).append(color)
    
    async def _define_tone_of_voice(self, extracted_info: Dict[str, Any],
                                  creative_direction: Dict[str, Any]) -> Dict[str, Any]:
        """Define brand tone of voice"""
        
        # Base tone from brief analysis
        primary_tone = creative_direction.get("tone", {}).get("primary", "professional")
        
        # Enhance based on industry and brand context
        industry = extracted_info.get("industry") or extracted_info.get("detected_industry")
        
        # Define tone characteristics
        tone_characteristics = {
            "formality": self._determine_formality_level(primary_tone, industry),
            "enthusiasm": self._determine_enthusiasm_level(primary_tone),
            "empathy": self._determine_empathy_level(primary_tone),
            "authority": self._determine_authority_level(primary_tone, industry),
            "humor": self._determine_humor_level(primary_tone, industry)
        }
        
        # Define communication guidelines
        communication_guidelines = {
            "voice_adjectives": self._get_voice_adjectives(primary_tone),
            "do_say": self._get_positive_language_examples(primary_tone),
            "dont_say": self._get_negative_language_examples(primary_tone),
            "writing_style": self._get_writing_style_guidelines(primary_tone),
            "message_structure": self._get_message_structure_guidelines(primary_tone)
        }
        
        return {
            "primary_tone": primary_tone,
            "characteristics": tone_characteristics,
            "communication_guidelines": communication_guidelines,
            "tone_consistency_score": 0.8  # Placeholder for consistency scoring
        }
    
    def _determine_formality_level(self, tone: str, industry: str = None) -> str:
        """Determine appropriate formality level"""
        formal_tones = ["professional", "sophisticated", "trustworthy"]
        informal_tones = ["playful", "friendly", "energetic"]
        
        formal_industries = ["finance", "healthcare", "legal", "consulting"]
        
        if tone in formal_tones or (industry and industry in formal_industries):
            return "formal"
        elif tone in informal_tones:
            return "casual"
        else:
            return "semi_formal"
    
    def _determine_enthusiasm_level(self, tone: str) -> str:
        """Determine enthusiasm level"""
        high_energy_tones = ["energetic", "playful", "confident", "innovative"]
        low_energy_tones = ["sophisticated", "trustworthy", "professional"]
        
        if tone in high_energy_tones:
            return "high"
        elif tone in low_energy_tones:
            return "moderate"
        else:
            return "balanced"
    
    def _determine_empathy_level(self, tone: str) -> str:
        """Determine empathy level"""
        empathetic_tones = ["warm", "friendly", "trustworthy", "caring"]
        
        if tone in empathetic_tones:
            return "high"
        else:
            return "moderate"
    
    def _determine_authority_level(self, tone: str, industry: str = None) -> str:
        """Determine authority level"""
        authoritative_tones = ["confident", "professional", "sophisticated"]
        authority_industries = ["finance", "healthcare", "consulting", "technology"]
        
        if tone in authoritative_tones or (industry and industry in authority_industries):
            return "high"
        else:
            return "moderate"
    
    def _determine_humor_level(self, tone: str, industry: str = None) -> str:
        """Determine appropriate humor level"""
        humorous_tones = ["playful", "friendly", "energetic"]
        serious_industries = ["healthcare", "finance", "legal"]
        
        if tone in humorous_tones and (not industry or industry not in serious_industries):
            return "light"
        else:
            return "none"
    
    def _get_voice_adjectives(self, tone: str) -> List[str]:
        """Get adjectives that describe the voice"""
        adjective_mapping = {
            "professional": ["reliable", "knowledgeable", "clear", "respectful"],
            "friendly": ["approachable", "warm", "helpful", "conversational"],
            "confident": ["assertive", "strong", "decisive", "bold"],
            "playful": ["fun", "creative", "energetic", "lighthearted"],
            "sophisticated": ["refined", "elegant", "cultured", "polished"],
            "trustworthy": ["honest", "transparent", "dependable", "sincere"]
        }
        
        return adjective_mapping.get(tone, ["clear", "helpful", "professional"])
    
    def _get_positive_language_examples(self, tone: str) -> List[str]:
        """Get examples of positive language for the tone"""
        positive_language = {
            "professional": ["We recommend", "Our expertise", "Proven results", "Industry-leading"],
            "friendly": ["We're here to help", "Let's work together", "Happy to assist", "We understand"],
            "confident": ["We deliver", "Guaranteed", "The best choice", "Leading solution"],
            "playful": ["Let's have fun", "Exciting opportunity", "Amazing results", "Love what we do"],
            "sophisticated": ["Exclusive", "Premium quality", "Refined approach", "Elegant solution"],
            "trustworthy": ["Transparent pricing", "Honest advice", "Your peace of mind", "Reliable partner"]
        }
        
        return positive_language.get(tone, ["We help", "Quality service", "Customer focused"])
    
    def _get_negative_language_examples(self, tone: str) -> List[str]:
        """Get examples of language to avoid"""
        negative_language = {
            "professional": ["Cheap", "Quick fix", "Maybe", "Sort of"],
            "friendly": ["You must", "Obviously", "You should know", "That's wrong"],
            "confident": ["We think", "Hopefully", "We'll try", "Might work"],
            "playful": ["Boring", "Serious business only", "No fun allowed", "Strictly professional"],
            "sophisticated": ["Cheap", "Basic", "Low-end", "Mass market"],
            "trustworthy": ["Trust us", "Secret", "Hidden fees", "Fine print"]
        }
        
        return negative_language.get(tone, ["Cheap", "Maybe", "We think"])
    
    def _get_writing_style_guidelines(self, tone: str) -> Dict[str, str]:
        """Get writing style guidelines"""
        style_guidelines = {
            "professional": {
                "sentence_length": "Medium to long",
                "vocabulary": "Industry-appropriate, clear",
                "structure": "Logical, well-organized",
                "punctuation": "Proper, formal"
            },
            "friendly": {
                "sentence_length": "Short to medium",
                "vocabulary": "Conversational, accessible",
                "structure": "Natural flow, personal",
                "punctuation": "Relaxed, expressive"
            },
            "confident": {
                "sentence_length": "Short, punchy",
                "vocabulary": "Strong, active voice",
                "structure": "Direct, action-oriented",
                "punctuation": "Decisive, clear"
            }
        }
        
        return style_guidelines.get(tone, style_guidelines["professional"])
    
    def _get_message_structure_guidelines(self, tone: str) -> Dict[str, str]:
        """Get message structure guidelines"""
        structure_guidelines = {
            "professional": {
                "opening": "Formal greeting, establish credibility",
                "body": "Logical argument, evidence-based",
                "closing": "Clear next steps, professional sign-off"
            },
            "friendly": {
                "opening": "Warm greeting, personal connection",
                "body": "Conversational, story-driven",
                "closing": "Friendly invitation, approachable sign-off"
            },
            "confident": {
                "opening": "Bold statement, value proposition",
                "body": "Strong benefits, social proof",
                "closing": "Clear call-to-action, confident close"
            }
        }
        
        return structure_guidelines.get(tone, structure_guidelines["professional"])
    
    async def _analyze_brand_positioning(self, extracted_info: Dict[str, Any],
                                       brand_archetype: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze brand positioning in the market"""
        
        # Extract positioning elements
        value_proposition = extracted_info.get("value_proposition", "")
        target_audience = extracted_info.get("target_audience", {})
        industry = extracted_info.get("industry") or extracted_info.get("detected_industry")
        
        # Define positioning strategy
        positioning_strategy = self._define_positioning_strategy(
            brand_archetype["primary"], industry, value_proposition
        )
        
        # Analyze competitive differentiation
        competitive_differentiation = self._analyze_competitive_differentiation(
            extracted_info, brand_archetype
        )
        
        return {
            "positioning_strategy": positioning_strategy,
            "value_proposition": value_proposition,
            "target_positioning": self._define_target_positioning(target_audience),
            "competitive_differentiation": competitive_differentiation,
            "brand_promise": brand_archetype.get("brand_promise", ""),
            "unique_selling_points": extracted_info.get("key_messages", [])
        }
    
    def _define_positioning_strategy(self, archetype: str, industry: str, 
                                   value_prop: str) -> Dict[str, Any]:
        """Define positioning strategy based on archetype and context"""
        
        archetype_positioning = {
            "the_innocent": {
                "position": "Pure, simple, trustworthy",
                "market_approach": "Safety and reliability",
                "emotional_appeal": "Peace of mind"
            },
            "the_sage": {
                "position": "Expert, knowledgeable, wise",
                "market_approach": "Thought leadership",
                "emotional_appeal": "Understanding and truth"
            },
            "the_hero": {
                "position": "Champion, problem-solver",
                "market_approach": "Performance and achievement",
                "emotional_appeal": "Triumph and success"
            },
            "the_outlaw": {
                "position": "Revolutionary, disruptive",
                "market_approach": "Change and innovation",
                "emotional_appeal": "Freedom and rebellion"
            },
            "the_explorer": {
                "position": "Adventurous, pioneering",
                "market_approach": "New frontiers",
                "emotional_appeal": "Discovery and freedom"
            }
        }
        
        return archetype_positioning.get(archetype, archetype_positioning["the_sage"])
    
    def _define_target_positioning(self, target_audience: Dict[str, Any]) -> Dict[str, Any]:
        """Define how to position for target audience"""
        return {
            "audience_segment": target_audience.get("primary_segment", "general"),
            "positioning_message": "Tailored solution for your needs",
            "emotional_connection": "Understanding and empathy",
            "rational_benefits": ["Quality", "Value", "Reliability"]
        }
    
    def _analyze_competitive_differentiation(self, extracted_info: Dict[str, Any],
                                           brand_archetype: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitive differentiation"""
        competitors = extracted_info.get("mentioned_competitors", [])
        
        return {
            "key_differentiators": extracted_info.get("key_messages", []),
            "competitive_advantages": ["Quality", "Service", "Innovation"],  # Simplified
            "market_position": "Premium" if "premium" in str(extracted_info).lower() else "Value",
            "differentiation_strategy": brand_archetype.get("communication_style", {}).get("differentiation", "Quality focused")
        }
    
    async def _generate_mood_board(self, personality: Dict[str, Any],
                                 visual_identity: Dict[str, Any],
                                 creative_direction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mood board elements"""
        mood_board = []
        
        # Color mood
        if visual_identity.get("color_palette"):
            mood_board.append({
                "type": "color_palette",
                "data": visual_identity["color_palette"],
                "description": "Brand color palette"
            })
        
        # Style mood
        dominant_traits = personality.get("dominant_traits", [])
        for trait in dominant_traits:
            mood_board.append({
                "type": "personality_trait",
                "data": {"trait": trait},
                "description": f"Brand personality: {trait}"
            })
        
        # Visual style mood
        if visual_identity.get("visual_style"):
            mood_board.append({
                "type": "visual_style",
                "data": visual_identity["visual_style"],
                "description": "Visual style direction"
            })
        
        # Creative mood
        mood = creative_direction.get("mood", {}).get("primary_mood")
        if mood:
            mood_board.append({
                "type": "mood",
                "data": {"mood": mood},
                "description": f"Overall mood: {mood}"
            })
        
        return mood_board
    
    async def _create_brand_guidelines(self, brand_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive brand guidelines"""
        return {
            "visual_guidelines": {
                "logo_usage": brand_analysis.get("visual_identity", {}).get("logo_characteristics", {}),
                "color_palette": brand_analysis.get("visual_identity", {}).get("color_palette", {}),
                "typography": brand_analysis.get("visual_identity", {}).get("typography_style", {}),
                "spacing_rules": {"minimum_clearspace": "1x logo height"}
            },
            "communication_guidelines": brand_analysis.get("tone_of_voice", {}).get("communication_guidelines", {}),
            "brand_voice": {
                "personality": brand_analysis.get("brand_personality", {}),
                "archetype": brand_analysis.get("brand_archetype", {}),
                "tone": brand_analysis.get("tone_of_voice", {})
            },
            "usage_guidelines": {
                "do": ["Maintain consistency", "Use approved colors", "Follow spacing rules"],
                "dont": ["Distort logo", "Use unapproved colors", "Overcrowd design"]
            }
        }
    
    async def _define_creative_direction(self, brand_analysis: Dict[str, Any],
                                       extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Define creative direction for banner generation"""
        
        personality = brand_analysis.get("brand_personality", {})
        visual_identity = brand_analysis.get("visual_identity", {})
        archetype = brand_analysis.get("brand_archetype", {})
        
        return {
            "design_principles": self._extract_design_principles_from_analysis(brand_analysis),
            "visual_hierarchy": self._define_visual_hierarchy(personality, archetype),
            "layout_approach": self._define_layout_approach(visual_identity),
            "color_strategy": self._define_color_strategy(visual_identity),
            "typography_direction": self._define_typography_direction(personality, archetype),
            "imagery_style": self._define_imagery_style(personality, archetype),
            "messaging_approach": self._define_messaging_approach(brand_analysis)
        }
    
    def _extract_design_principles_from_analysis(self, brand_analysis: Dict[str, Any]) -> List[str]:
        """Extract design principles from brand analysis"""
        principles = []
        
        personality = brand_analysis.get("brand_personality", {})
        dominant_traits = personality.get("dominant_traits", [])
        
        if "sincerity" in dominant_traits:
            principles.extend(["Clarity", "Honesty", "Simplicity"])
        if "excitement" in dominant_traits:
            principles.extend(["Dynamism", "Innovation", "Energy"])
        if "competence" in dominant_traits:
            principles.extend(["Professionalism", "Precision", "Reliability"])
        if "sophistication" in dominant_traits:
            principles.extend(["Elegance", "Refinement", "Quality"])
        if "ruggedness" in dominant_traits:
            principles.extend(["Strength", "Authenticity", "Boldness"])
        
        # Remove duplicates
        return list(set(principles))
    
    def _define_visual_hierarchy(self, personality: Dict[str, Any], 
                               archetype: Dict[str, Any]) -> Dict[str, str]:
        """Define visual hierarchy approach"""
        dominant_traits = personality.get("dominant_traits", [])
        primary_archetype = archetype.get("primary", "the_sage")
        
        if "competence" in dominant_traits or primary_archetype in ["the_sage", "the_ruler"]:
            return {
                "approach": "structured",
                "emphasis": "logical_flow",
                "balance": "asymmetrical_professional"
            }
        elif "excitement" in dominant_traits or primary_archetype in ["the_outlaw", "the_explorer"]:
            return {
                "approach": "dynamic",
                "emphasis": "visual_impact",
                "balance": "creative_asymmetrical"
            }
        else:
            return {
                "approach": "balanced",
                "emphasis": "clear_communication",
                "balance": "symmetrical"
            }
    
    def _define_layout_approach(self, visual_identity: Dict[str, Any]) -> Dict[str, str]:
        """Define layout approach"""
        visual_style = visual_identity.get("visual_style", {})
        aesthetic = visual_style.get("aesthetic", "clean_modern")
        
        layout_mapping = {
            "clean_modern": {"grid": "simple", "spacing": "generous", "alignment": "left"},
            "structured_modern": {"grid": "modular", "spacing": "structured", "alignment": "grid_based"},
            "rich_detailed": {"grid": "complex", "spacing": "compact", "alignment": "flexible"},
            "typographic": {"grid": "text_focused", "spacing": "text_optimized", "alignment": "centered"}
        }
        
        return layout_mapping.get(aesthetic, layout_mapping["clean_modern"])
    
    def _define_color_strategy(self, visual_identity: Dict[str, Any]) -> Dict[str, Any]:
        """Define color usage strategy"""
        color_palette = visual_identity.get("color_palette", {})
        
        return {
            "primary_usage": "Brand identity and accents",
            "secondary_usage": "Supporting elements",
            "background_strategy": "Neutral with brand color accents",
            "contrast_approach": "High contrast for readability",
            "color_dominance": color_palette.get("dominant_tone", "balanced")
        }
    
    def _define_typography_direction(self, personality: Dict[str, Any],
                                   archetype: Dict[str, Any]) -> Dict[str, str]:
        """Define typography direction"""
        dominant_traits = personality.get("dominant_traits", [])
        
        if "sophistication" in dominant_traits:
            return {"style": "elegant_serif", "weight": "refined", "spacing": "generous"}
        elif "excitement" in dominant_traits:
            return {"style": "modern_sans", "weight": "bold", "spacing": "dynamic"}
        elif "competence" in dominant_traits:
            return {"style": "professional_sans", "weight": "medium", "spacing": "structured"}
        else:
            return {"style": "versatile_sans", "weight": "regular", "spacing": "balanced"}
    
    def _define_imagery_style(self, personality: Dict[str, Any],
                            archetype: Dict[str, Any]) -> Dict[str, str]:
        """Define imagery style direction"""
        dominant_traits = personality.get("dominant_traits", [])
        
        if "sincerity" in dominant_traits:
            return {"style": "authentic", "treatment": "natural", "mood": "warm"}
        elif "excitement" in dominant_traits:
            return {"style": "dynamic", "treatment": "vibrant", "mood": "energetic"}
        elif "sophistication" in dominant_traits:
            return {"style": "refined", "treatment": "polished", "mood": "elegant"}
        else:
            return {"style": "clean", "treatment": "professional", "mood": "balanced"}
    
    def _define_messaging_approach(self, brand_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Define messaging approach"""
        tone_of_voice = brand_analysis.get("tone_of_voice", {})
        primary_tone = tone_of_voice.get("primary_tone", "professional")
        
        return {
            "message_style": primary_tone,
            "hierarchy": "benefit_focused",
            "call_to_action": "clear_directive",
            "length": "concise" if primary_tone in ["confident", "playful"] else "detailed"
        }
    
    def _generate_personality_summary(self, dominant_traits: List[str]) -> str:
        """Generate a personality summary"""
        if not dominant_traits:
            return "Professional and balanced brand personality"
        
        trait_descriptions = {
            "sincerity": "honest and down-to-earth",
            "excitement": "dynamic and innovative",
            "competence": "reliable and professional",
            "sophistication": "elegant and refined",
            "ruggedness": "strong and authentic"
        }
        
        descriptions = [trait_descriptions.get(trait, trait) for trait in dominant_traits]
        
        if len(descriptions) == 1:
            return f"A {descriptions[0]} brand personality"
        elif len(descriptions) == 2:
            return f"A {descriptions[0]} and {descriptions[1]} brand personality"
        else:
            return f"A {', '.join(descriptions[:-1])}, and {descriptions[-1]} brand personality"
    
    def _determine_brand_character(self, personality_traits: Dict[str, float]) -> str:
        """Determine overall brand character"""
        # Find the highest scoring trait
        max_trait = max(personality_traits, key=personality_traits.get)
        max_score = personality_traits[max_trait]
        
        if max_score < 0.3:
            return "neutral"
        elif max_trait == "sincerity":
            return "trustworthy_reliable"
        elif max_trait == "excitement":
            return "innovative_dynamic"
        elif max_trait == "competence":
            return "professional_expert"
        elif max_trait == "sophistication":
            return "premium_refined"
        elif max_trait == "ruggedness":
            return "strong_authentic"
        else:
            return "balanced_versatile"
    
    def _load_brand_archetypes(self) -> Dict[str, Dict[str, Any]]:
        """Load brand archetype definitions"""
        return {
            "the_innocent": {
                "description": "Pure, virtuous, simple, optimistic",
                "brand_promise": "Simple solutions that work",
                "personality_weights": {"sincerity": 0.9, "competence": 0.6},
                "communication_style": {"tone": "simple", "approach": "straightforward"},
                "common_industries": ["food", "health", "family"],
                "company_types": ["b2c", "family"]
            },
            "the_sage": {
                "description": "Wise, knowledgeable, truth-seeking",
                "brand_promise": "Expert guidance and wisdom",
                "personality_weights": {"competence": 0.9, "sincerity": 0.7},
                "communication_style": {"tone": "educational", "approach": "informative"},
                "common_industries": ["education", "consulting", "technology"],
                "company_types": ["b2b", "professional"]
            },
            "the_hero": {
                "description": "Brave, determined, honorable",
                "brand_promise": "Triumph through perseverance",
                "personality_weights": {"competence": 0.8, "excitement": 0.7},
                "communication_style": {"tone": "inspiring", "approach": "motivational"},
                "common_industries": ["sports", "automotive", "military"],
                "company_types": ["b2c", "performance"]
            },
            "the_outlaw": {
                "description": "Revolutionary, wild, disruptive",
                "brand_promise": "Break the rules and change the world",
                "personality_weights": {"excitement": 0.9, "ruggedness": 0.7},
                "communication_style": {"tone": "rebellious", "approach": "provocative"},
                "common_industries": ["fashion", "technology", "entertainment"],
                "company_types": ["startup", "disruptive"]
            },
            "the_explorer": {
                "description": "Free, adventurous, pioneering",
                "brand_promise": "Find yourself through exploration",
                "personality_weights": {"excitement": 0.8, "ruggedness": 0.6},
                "communication_style": {"tone": "adventurous", "approach": "inspiring"},
                "common_industries": ["travel", "outdoor", "technology"],
                "company_types": ["b2c", "lifestyle"]
            }
        }
    
    def _load_industry_traits(self) -> Dict[str, Dict[str, Any]]:
        """Load industry-specific brand traits"""
        return {
            "technology": {
                "personality": {"competence": 0.8, "excitement": 0.7},
                "common_archetypes": ["the_sage", "the_explorer", "the_outlaw"],
                "visual_preferences": ["modern", "clean", "innovative"]
            },
            "finance": {
                "personality": {"competence": 0.9, "sincerity": 0.8},
                "common_archetypes": ["the_sage", "the_ruler", "the_caregiver"],
                "visual_preferences": ["professional", "trustworthy", "stable"]
            },
            "healthcare": {
                "personality": {"sincerity": 0.9, "competence": 0.8},
                "common_archetypes": ["the_caregiver", "the_sage", "the_innocent"],
                "visual_preferences": ["clean", "trustworthy", "calming"]
            },
            "retail": {
                "personality": {"excitement": 0.7, "sincerity": 0.6},
                "common_archetypes": ["the_innocent", "the_explorer", "the_lover"],
                "visual_preferences": ["appealing", "accessible", "friendly"]
            }
        }
