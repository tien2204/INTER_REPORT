"""
Target Audience Analyzer

Analyzes target audience characteristics from campaign briefs
to inform design decisions and messaging strategy.
"""

from typing import Dict, Any, List
from structlog import get_logger

logger = get_logger(__name__)


class TargetAudienceAnalyzer:
    """
    Analyzes target audience characteristics and preferences
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Age group mappings
        self.age_groups = {
            "gen_z": {"min": 18, "max": 27, "characteristics": ["digital_native", "authentic", "visual"]},
            "millennial": {"min": 28, "max": 43, "characteristics": ["tech_savvy", "value_driven", "social"]},
            "gen_x": {"min": 44, "max": 59, "characteristics": ["practical", "skeptical", "direct"]},
            "boomer": {"min": 60, "max": 78, "characteristics": ["traditional", "quality_focused", "detailed"]}
        }
        
        # Industry-audience mappings
        self.industry_audiences = {
            "tech": ["professionals", "early_adopters", "innovators"],
            "finance": ["professionals", "executives", "investors"],
            "healthcare": ["patients", "caregivers", "professionals"],
            "retail": ["consumers", "shoppers", "bargain_hunters"],
            "education": ["students", "parents", "educators"],
            "real_estate": ["homebuyers", "investors", "renters"]
        }
        
    async def analyze_target_audience(self, brief_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze target audience from brief analysis
        
        Args:
            brief_analysis: Analyzed brief data
            
        Returns:
            Target audience analysis
        """
        try:
            logger.info("Analyzing target audience")
            
            # Extract audience data from brief
            audience_data = brief_analysis.get("target_audience", {})
            industry = brief_analysis.get("industry", "general")
            
            analysis = {
                "demographics": self._analyze_demographics(audience_data),
                "psychographics": self._analyze_psychographics(audience_data, industry),
                "behavioral_traits": self._analyze_behavioral_traits(audience_data),
                "communication_preferences": self._determine_communication_preferences(audience_data),
                "design_preferences": self._determine_design_preferences(audience_data),
                "media_consumption": self._analyze_media_consumption(audience_data),
                "decision_factors": self._analyze_decision_factors(audience_data, industry),
                "persona_summary": self._create_persona_summary(audience_data)
            }
            
            logger.info("Target audience analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Target audience analysis failed: {e}")
            raise
    
    def _analyze_demographics(self, audience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze demographic characteristics"""
        
        # Extract or infer demographics
        age_range = audience_data.get("age_range", "25-45")
        gender = audience_data.get("gender", "mixed")
        income_level = audience_data.get("income_level", "middle")
        education = audience_data.get("education", "college")
        location = audience_data.get("location", "urban")
        
        # Determine primary age group
        age_group = self._determine_age_group(age_range)
        
        return {
            "age_range": age_range,
            "primary_age_group": age_group,
            "gender_distribution": gender,
            "income_level": income_level,
            "education_level": education,
            "geographic_focus": location,
            "lifestyle_segment": self._determine_lifestyle_segment(age_group, income_level)
        }
    
    def _analyze_psychographics(self, audience_data: Dict[str, Any], 
                               industry: str) -> Dict[str, Any]:
        """Analyze psychographic characteristics"""
        
        interests = audience_data.get("interests", [])
        values = audience_data.get("values", [])
        lifestyle = audience_data.get("lifestyle", "balanced")
        
        # Infer psychographics from industry if not provided
        if not interests and industry in self.industry_audiences:
            interests = self.industry_audiences[industry]
        
        return {
            "primary_interests": interests,
            "core_values": values or self._infer_values_from_industry(industry),
            "lifestyle_type": lifestyle,
            "motivation_drivers": self._determine_motivation_drivers(values, lifestyle),
            "brand_affinity": self._determine_brand_affinity(interests, values),
            "social_influence": self._assess_social_influence(audience_data)
        }
    
    def _analyze_behavioral_traits(self, audience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral traits"""
        
        buying_behavior = audience_data.get("buying_behavior", "considered")
        tech_adoption = audience_data.get("tech_adoption", "mainstream")
        social_activity = audience_data.get("social_activity", "moderate")
        
        return {
            "buying_behavior": buying_behavior,
            "purchase_decision_speed": self._map_buying_behavior_to_speed(buying_behavior),
            "technology_adoption": tech_adoption,
            "social_media_engagement": social_activity,
            "information_seeking": self._determine_info_seeking_behavior(buying_behavior),
            "brand_loyalty": self._assess_brand_loyalty(buying_behavior)
        }
    
    def _determine_communication_preferences(self, audience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine communication preferences"""
        
        age_range = audience_data.get("age_range", "25-45")
        education = audience_data.get("education", "college")
        
        # Map age to communication style
        age_group = self._determine_age_group(age_range)
        
        if age_group == "gen_z":
            style = "casual_authentic"
            channels = ["social_media", "video", "mobile"]
        elif age_group == "millennial":
            style = "conversational_informative"
            channels = ["social_media", "email", "content_marketing"]
        elif age_group == "gen_x":
            style = "direct_practical"
            channels = ["email", "websites", "traditional_media"]
        else:  # boomer
            style = "formal_detailed"
            channels = ["traditional_media", "email", "phone"]
        
        return {
            "communication_style": style,
            "preferred_channels": channels,
            "message_complexity": "simple" if age_group in ["gen_z"] else "detailed",
            "tone_preference": self._determine_tone_preference(age_group, education),
            "content_format": self._determine_content_format_preference(age_group)
        }
    
    def _determine_design_preferences(self, audience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine design preferences based on audience"""
        
        age_range = audience_data.get("age_range", "25-45")
        income_level = audience_data.get("income_level", "middle")
        lifestyle = audience_data.get("lifestyle", "balanced")
        
        age_group = self._determine_age_group(age_range)
        
        # Map characteristics to design preferences
        if age_group == "gen_z":
            aesthetic = "bold_vibrant"
            layout = "dynamic_mobile_first"
            colors = "bright_contrasting"
        elif age_group == "millennial":
            aesthetic = "modern_clean"
            layout = "structured_responsive"
            colors = "contemporary_palette"
        elif age_group == "gen_x":
            aesthetic = "professional_trustworthy"
            layout = "traditional_clear"
            colors = "conservative_reliable"
        else:  # boomer
            aesthetic = "classic_elegant"
            layout = "simple_readable"
            colors = "traditional_muted"
        
        # Adjust for income level
        if income_level in ["high", "premium"]:
            aesthetic = f"luxury_{aesthetic}"
            colors = f"sophisticated_{colors}"
        
        return {
            "aesthetic_preference": aesthetic,
            "layout_style": layout,
            "color_preference": colors,
            "typography_style": self._map_age_to_typography(age_group),
            "image_style": self._determine_image_style(age_group, lifestyle),
            "interactive_elements": age_group in ["gen_z", "millennial"]
        }
    
    def _analyze_media_consumption(self, audience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze media consumption patterns"""
        
        age_range = audience_data.get("age_range", "25-45")
        tech_adoption = audience_data.get("tech_adoption", "mainstream")
        
        age_group = self._determine_age_group(age_range)
        
        # Map to media preferences
        media_mapping = {
            "gen_z": {
                "primary": ["tiktok", "instagram", "youtube_shorts"],
                "secondary": ["snapchat", "twitter", "twitch"],
                "screen_time": "high",
                "attention_span": "short"
            },
            "millennial": {
                "primary": ["facebook", "instagram", "linkedin"],
                "secondary": ["twitter", "youtube", "podcasts"],
                "screen_time": "high",
                "attention_span": "medium"
            },
            "gen_x": {
                "primary": ["facebook", "linkedin", "email"],
                "secondary": ["websites", "youtube", "traditional_media"],
                "screen_time": "medium",
                "attention_span": "medium"
            },
            "boomer": {
                "primary": ["facebook", "email", "traditional_media"],
                "secondary": ["websites", "print", "radio"],
                "screen_time": "low",
                "attention_span": "long"
            }
        }
        
        return media_mapping.get(age_group, media_mapping["millennial"])
    
    def _analyze_decision_factors(self, audience_data: Dict[str, Any], 
                                industry: str) -> Dict[str, Any]:
        """Analyze key decision factors"""
        
        buying_behavior = audience_data.get("buying_behavior", "considered")
        income_level = audience_data.get("income_level", "middle")
        
        # Base decision factors
        if buying_behavior == "impulsive":
            primary_factors = ["emotion", "convenience", "price"]
            decision_speed = "fast"
        elif buying_behavior == "considered":
            primary_factors = ["value", "quality", "reviews"]
            decision_speed = "medium"
        else:  # analytical
            primary_factors = ["features", "comparison", "long_term_value"]
            decision_speed = "slow"
        
        # Industry-specific adjustments
        industry_factors = {
            "tech": ["innovation", "features", "compatibility"],
            "finance": ["security", "reputation", "returns"],
            "healthcare": ["trust", "safety", "effectiveness"],
            "retail": ["price", "convenience", "variety"],
            "education": ["quality", "accreditation", "outcomes"],
            "real_estate": ["location", "value", "investment_potential"]
        }
        
        if industry in industry_factors:
            primary_factors.extend(industry_factors[industry])
        
        return {
            "primary_decision_factors": primary_factors[:5],  # Top 5
            "decision_making_speed": decision_speed,
            "research_intensity": self._map_buying_behavior_to_research(buying_behavior),
            "influence_sources": self._determine_influence_sources(audience_data),
            "risk_tolerance": self._assess_risk_tolerance(buying_behavior, income_level)
        }
    
    def _create_persona_summary(self, audience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a persona summary"""
        
        age_range = audience_data.get("age_range", "25-45")
        gender = audience_data.get("gender", "mixed")
        income_level = audience_data.get("income_level", "middle")
        interests = audience_data.get("interests", [])
        
        # Create persona name based on characteristics
        age_group = self._determine_age_group(age_range)
        persona_name = self._generate_persona_name(age_group, gender)
        
        return {
            "persona_name": persona_name,
            "age_group": age_group,
            "key_characteristics": self.age_groups.get(age_group, {}).get("characteristics", []),
            "primary_motivations": interests[:3] if interests else ["success", "convenience", "value"],
            "communication_style": "authentic" if age_group == "gen_z" else "professional",
            "decision_style": audience_data.get("buying_behavior", "considered"),
            "summary": self._generate_persona_summary(persona_name, age_group, interests)
        }
    
    # Helper methods
    
    def _determine_age_group(self, age_range: str) -> str:
        """Determine primary age group from age range"""
        try:
            if "-" in age_range:
                min_age = int(age_range.split("-")[0])
                max_age = int(age_range.split("-")[1])
                avg_age = (min_age + max_age) / 2
            else:
                avg_age = int(age_range)
            
            for group, data in self.age_groups.items():
                if data["min"] <= avg_age <= data["max"]:
                    return group
            
            return "millennial"  # default
        except:
            return "millennial"
    
    def _determine_lifestyle_segment(self, age_group: str, income_level: str) -> str:
        """Determine lifestyle segment"""
        segments = {
            ("gen_z", "low"): "budget_conscious_digital",
            ("gen_z", "middle"): "social_experience_seekers",
            ("gen_z", "high"): "premium_early_adopters",
            ("millennial", "low"): "value_conscious_families",
            ("millennial", "middle"): "established_professionals",
            ("millennial", "high"): "affluent_achievers",
            ("gen_x", "low"): "practical_savers",
            ("gen_x", "middle"): "stable_traditionalists",
            ("gen_x", "high"): "successful_executives",
            ("boomer", "low"): "careful_planners",
            ("boomer", "middle"): "comfortable_retirees",
            ("boomer", "high"): "wealthy_conservatives"
        }
        
        return segments.get((age_group, income_level), "mainstream_consumers")
    
    def _infer_values_from_industry(self, industry: str) -> List[str]:
        """Infer values from industry context"""
        industry_values = {
            "tech": ["innovation", "efficiency", "progress"],
            "finance": ["security", "growth", "stability"],
            "healthcare": ["wellness", "trust", "care"],
            "retail": ["value", "convenience", "choice"],
            "education": ["knowledge", "achievement", "future"],
            "real_estate": ["security", "investment", "home"]
        }
        
        return industry_values.get(industry, ["quality", "value", "reliability"])
    
    def _determine_motivation_drivers(self, values: List[str], lifestyle: str) -> List[str]:
        """Determine key motivation drivers"""
        base_drivers = ["achievement", "security", "belonging", "autonomy", "purpose"]
        
        # Adjust based on values and lifestyle
        if "innovation" in values:
            base_drivers.insert(0, "innovation")
        if lifestyle == "luxury":
            base_drivers.insert(0, "status")
        if lifestyle == "family":
            base_drivers.insert(0, "family_welfare")
        
        return base_drivers[:5]
    
    def _determine_brand_affinity(self, interests: List[str], values: List[str]) -> str:
        """Determine brand affinity type"""
        if "luxury" in interests or "premium" in values:
            return "premium_brands"
        elif "sustainability" in interests or "environment" in values:
            return "conscious_brands"
        elif "innovation" in interests or "technology" in values:
            return "innovative_brands"
        else:
            return "trusted_brands"
    
    def _assess_social_influence(self, audience_data: Dict[str, Any]) -> str:
        """Assess social influence level"""
        social_activity = audience_data.get("social_activity", "moderate")
        age_range = audience_data.get("age_range", "25-45")
        
        age_group = self._determine_age_group(age_range)
        
        if age_group in ["gen_z", "millennial"] and social_activity == "high":
            return "high_social_influence"
        elif social_activity == "low":
            return "low_social_influence"
        else:
            return "moderate_social_influence"
    
    def _map_buying_behavior_to_speed(self, buying_behavior: str) -> str:
        """Map buying behavior to decision speed"""
        mapping = {
            "impulsive": "immediate",
            "considered": "days_weeks",
            "analytical": "weeks_months"
        }
        return mapping.get(buying_behavior, "days_weeks")
    
    def _determine_info_seeking_behavior(self, buying_behavior: str) -> str:
        """Determine information seeking behavior"""
        mapping = {
            "impulsive": "minimal_research",
            "considered": "moderate_research",
            "analytical": "extensive_research"
        }
        return mapping.get(buying_behavior, "moderate_research")
    
    def _assess_brand_loyalty(self, buying_behavior: str) -> str:
        """Assess brand loyalty tendency"""
        mapping = {
            "impulsive": "low_loyalty",
            "considered": "moderate_loyalty",
            "analytical": "high_loyalty"
        }
        return mapping.get(buying_behavior, "moderate_loyalty")
    
    def _determine_tone_preference(self, age_group: str, education: str) -> str:
        """Determine preferred communication tone"""
        if age_group == "gen_z":
            return "casual_authentic"
        elif age_group == "millennial":
            return "friendly_professional"
        elif education in ["graduate", "professional"]:
            return "sophisticated_informative"
        else:
            return "respectful_clear"
    
    def _determine_content_format_preference(self, age_group: str) -> List[str]:
        """Determine preferred content formats"""
        format_mapping = {
            "gen_z": ["video", "images", "stories", "interactive"],
            "millennial": ["articles", "infographics", "videos", "social_posts"],
            "gen_x": ["articles", "emails", "websites", "pdfs"],
            "boomer": ["articles", "emails", "print", "phone_calls"]
        }
        return format_mapping.get(age_group, ["articles", "images"])
    
    def _map_age_to_typography(self, age_group: str) -> str:
        """Map age group to typography preferences"""
        mapping = {
            "gen_z": "bold_creative",
            "millennial": "modern_clean",
            "gen_x": "professional_readable",
            "boomer": "traditional_large"
        }
        return mapping.get(age_group, "modern_clean")
    
    def _determine_image_style(self, age_group: str, lifestyle: str) -> str:
        """Determine preferred image style"""
        if age_group == "gen_z":
            return "authentic_diverse"
        elif lifestyle == "luxury":
            return "premium_polished"
        elif age_group == "boomer":
            return "traditional_professional"
        else:
            return "modern_relatable"
    
    def _map_buying_behavior_to_research(self, buying_behavior: str) -> str:
        """Map buying behavior to research intensity"""
        mapping = {
            "impulsive": "light",
            "considered": "moderate",
            "analytical": "extensive"
        }
        return mapping.get(buying_behavior, "moderate")
    
    def _determine_influence_sources(self, audience_data: Dict[str, Any]) -> List[str]:
        """Determine key influence sources"""
        age_range = audience_data.get("age_range", "25-45")
        age_group = self._determine_age_group(age_range)
        
        influence_mapping = {
            "gen_z": ["social_media_influencers", "peer_reviews", "video_content"],
            "millennial": ["online_reviews", "social_networks", "expert_opinions"],
            "gen_x": ["professional_networks", "traditional_media", "word_of_mouth"],
            "boomer": ["traditional_media", "professional_advice", "established_brands"]
        }
        
        return influence_mapping.get(age_group, ["online_reviews", "word_of_mouth"])
    
    def _assess_risk_tolerance(self, buying_behavior: str, income_level: str) -> str:
        """Assess risk tolerance"""
        if buying_behavior == "impulsive":
            base_tolerance = "medium"
        elif buying_behavior == "analytical":
            base_tolerance = "low"
        else:
            base_tolerance = "medium"
        
        # Adjust for income
        if income_level == "high":
            if base_tolerance == "low":
                return "medium"
            else:
                return "high"
        elif income_level == "low":
            return "low"
        
        return base_tolerance
    
    def _generate_persona_name(self, age_group: str, gender: str) -> str:
        """Generate a persona name"""
        names = {
            "gen_z": {"male": "Digital Dylan", "female": "Social Sophia", "mixed": "Connected Casey"},
            "millennial": {"male": "Professional Marcus", "female": "Ambitious Amanda", "mixed": "Driven Drew"},
            "gen_x": {"male": "Practical Paul", "female": "Experienced Emma", "mixed": "Seasoned Sam"},
            "boomer": {"male": "Traditional Tom", "female": "Reliable Ruth", "mixed": "Established Alex"}
        }
        
        return names.get(age_group, {}).get(gender, f"Target {age_group.title()}")
    
    def _generate_persona_summary(self, persona_name: str, age_group: str, 
                                 interests: List[str]) -> str:
        """Generate a persona summary"""
        interests_str = ", ".join(interests[:3]) if interests else "various interests"
        
        summaries = {
            "gen_z": f"{persona_name} is a digital native who values authenticity and visual content. Interested in {interests_str}.",
            "millennial": f"{persona_name} is a tech-savvy professional balancing career and personal life. Focuses on {interests_str}.",
            "gen_x": f"{persona_name} is a practical decision-maker who values quality and reliability. Interested in {interests_str}.",
            "boomer": f"{persona_name} is a traditional consumer who prefers detailed information and established brands. Values {interests_str}."
        }
        
        return summaries.get(age_group, f"{persona_name} represents the target audience with interests in {interests_str}.")
