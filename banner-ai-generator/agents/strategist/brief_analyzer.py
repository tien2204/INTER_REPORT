"""
Brief Analyzer

Analyzes campaign briefs to extract key information,
understand requirements, and structure data for downstream agents.
"""

import re
from typing import Any, Dict, List, Optional
from structlog import get_logger

logger = get_logger(__name__)


class BriefAnalyzer:
    """
    Campaign brief analyzer that extracts and structures key information
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.industry_keywords = self._load_industry_keywords()
        self.mood_keywords = self._load_mood_keywords()
        self.urgency_keywords = self._load_urgency_keywords()
    
    async def analyze_brief(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze campaign brief and extract structured information
        
        Args:
            brief: Raw campaign brief from advertiser
            
        Returns:
            Structured analysis of the brief
        """
        try:
            analysis = {
                "original_brief": brief,
                "extracted_info": {},
                "structured_requirements": {},
                "creative_direction": {},
                "constraints": {},
                "metadata": {}
            }
            
            # Extract basic information
            analysis["extracted_info"] = await self._extract_basic_info(brief)
            
            # Structure requirements
            analysis["structured_requirements"] = await self._structure_requirements(brief)
            
            # Analyze creative direction
            analysis["creative_direction"] = await self._analyze_creative_direction(brief)
            
            # Extract constraints
            analysis["constraints"] = await self._extract_constraints(brief)
            
            # Add metadata
            analysis["metadata"] = await self._generate_metadata(brief, analysis)
            
            logger.info("Brief analysis completed successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"Brief analysis failed: {e}")
            raise
    
    async def _extract_basic_info(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic information from brief"""
        info = {}
        
        # Direct field extraction
        direct_fields = [
            "company_name", "product_name", "campaign_name", "industry",
            "target_audience", "primary_message", "cta_text", "budget",
            "timeline", "dimensions", "formats"
        ]
        
        for field in direct_fields:
            if field in brief:
                info[field] = brief[field]
        
        # Extract from description/text fields
        description = brief.get("description", "")
        if description:
            info.update(await self._extract_from_text(description))
        
        # Extract key messages
        info["key_messages"] = self._extract_key_messages(brief)
        
        # Extract contact information
        info["contact_info"] = self._extract_contact_info(brief)
        
        return info
    
    async def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract information from free text description"""
        extracted = {}
        
        # Industry detection
        industry = self._detect_industry(text)
        if industry:
            extracted["detected_industry"] = industry
        
        # Mood detection
        mood = self._detect_mood(text)
        if mood:
            extracted["detected_mood"] = mood
        
        # Urgency detection
        urgency = self._detect_urgency(text)
        if urgency:
            extracted["detected_urgency"] = urgency
        
        # Product/service detection
        products = self._extract_products_services(text)
        if products:
            extracted["products_services"] = products
        
        # Competitor mentions
        competitors = self._extract_competitors(text)
        if competitors:
            extracted["mentioned_competitors"] = competitors
        
        return extracted
    
    def _detect_industry(self, text: str) -> Optional[str]:
        """Detect industry from text"""
        text_lower = text.lower()
        
        for industry, keywords in self.industry_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return industry
        
        return None
    
    def _detect_mood(self, text: str) -> Optional[str]:
        """Detect mood/tone from text"""
        text_lower = text.lower()
        
        mood_scores = {}
        for mood, keywords in self.mood_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                mood_scores[mood] = score
        
        if mood_scores:
            return max(mood_scores, key=mood_scores.get)
        
        return None
    
    def _detect_urgency(self, text: str) -> str:
        """Detect urgency level from text"""
        text_lower = text.lower()
        
        urgency_scores = {"high": 0, "medium": 0, "low": 0}
        
        for urgency, keywords in self.urgency_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            urgency_scores[urgency] = score
        
        return max(urgency_scores, key=urgency_scores.get)
    
    def _extract_products_services(self, text: str) -> List[str]:
        """Extract product/service names from text"""
        # Simple pattern matching for product/service mentions
        patterns = [
            r"our (\w+(?:\s+\w+){0,2})",
            r"the (\w+(?:\s+\w+){0,2}) (?:product|service|solution)",
            r"(\w+(?:\s+\w+){0,2}) (?:software|platform|app|tool)",
        ]
        
        products = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            products.extend(matches)
        
        # Clean and deduplicate
        products = [p.strip() for p in products if len(p.strip()) > 2]
        return list(set(products))
    
    def _extract_competitors(self, text: str) -> List[str]:
        """Extract competitor mentions from text"""
        # Pattern for competitor mentions
        patterns = [
            r"(?:compete with|vs|versus|compared to|unlike) (\w+)",
            r"competitor (\w+)",
            r"(\w+) is our main competitor"
        ]
        
        competitors = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            competitors.extend(matches)
        
        return list(set(competitors))
    
    def _extract_key_messages(self, brief: Dict[str, Any]) -> List[str]:
        """Extract key messages from brief"""
        messages = []
        
        # Direct key messages
        if "key_messages" in brief:
            if isinstance(brief["key_messages"], list):
                messages.extend(brief["key_messages"])
            elif isinstance(brief["key_messages"], str):
                messages.append(brief["key_messages"])
        
        # Extract from other fields
        message_fields = ["primary_message", "value_proposition", "unique_selling_point"]
        for field in message_fields:
            if field in brief and brief[field]:
                messages.append(brief[field])
        
        # Extract from bullet points in description
        description = brief.get("description", "")
        bullet_points = re.findall(r"[•\-\*]\s*(.+)", description)
        messages.extend(bullet_points)
        
        # Clean and deduplicate
        messages = [msg.strip() for msg in messages if msg.strip()]
        return list(dict.fromkeys(messages))  # Remove duplicates while preserving order
    
    def _extract_contact_info(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contact information"""
        contact = {}
        
        contact_fields = [
            "contact_name", "contact_email", "contact_phone",
            "company_contact", "project_manager", "stakeholder"
        ]
        
        for field in contact_fields:
            if field in brief:
                contact[field] = brief[field]
        
        return contact
    
    async def _structure_requirements(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Structure campaign requirements"""
        requirements = {
            "technical": {},
            "creative": {},
            "business": {},
            "deliverables": {}
        }
        
        # Technical requirements
        requirements["technical"] = {
            "dimensions": brief.get("dimensions", {}),
            "formats": brief.get("formats", ["SVG", "PNG"]),
            "file_size_limit": brief.get("file_size_limit"),
            "color_space": brief.get("color_space", "RGB"),
            "resolution": brief.get("resolution", "72dpi"),
            "platforms": brief.get("platforms", ["web"])
        }
        
        # Creative requirements
        requirements["creative"] = {
            "style_preferences": brief.get("style_preferences", []),
            "color_preferences": brief.get("color_preferences", {}),
            "font_preferences": brief.get("font_preferences", {}),
            "imagery_style": brief.get("imagery_style"),
            "brand_guidelines": brief.get("brand_guidelines", {}),
            "do_not_use": brief.get("do_not_use", [])
        }
        
        # Business requirements
        requirements["business"] = {
            "campaign_goals": brief.get("campaign_goals", []),
            "success_metrics": brief.get("success_metrics", []),
            "target_ctr": brief.get("target_ctr"),
            "conversion_goals": brief.get("conversion_goals", []),
            "budget_constraints": brief.get("budget_constraints", {})
        }
        
        # Deliverables
        requirements["deliverables"] = {
            "variations": brief.get("variations", 1),
            "sizes": brief.get("sizes", []),
            "timeline": brief.get("timeline", {}),
            "review_rounds": brief.get("review_rounds", 2),
            "approval_process": brief.get("approval_process", {})
        }
        
        return requirements
    
    async def _analyze_creative_direction(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze creative direction from brief"""
        direction = {}
        
        # Tone analysis
        direction["tone"] = self._analyze_tone(brief)
        
        # Style analysis  
        direction["style"] = self._analyze_style(brief)
        
        # Mood analysis
        direction["mood"] = self._analyze_mood(brief)
        
        # Message hierarchy
        direction["message_hierarchy"] = self._analyze_message_hierarchy(brief)
        
        # Visual preferences
        direction["visual_preferences"] = self._analyze_visual_preferences(brief)
        
        return direction
    
    def _analyze_tone(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze communication tone"""
        tone_indicators = {
            "professional": ["professional", "corporate", "business", "formal"],
            "friendly": ["friendly", "approachable", "warm", "welcoming"],
            "confident": ["confident", "strong", "bold", "assertive"],
            "playful": ["playful", "fun", "creative", "energetic"],
            "trustworthy": ["trustworthy", "reliable", "credible", "honest"],
            "innovative": ["innovative", "cutting-edge", "modern", "advanced"]
        }
        
        text = " ".join([
            str(v) for v in brief.values() 
            if isinstance(v, str)
        ]).lower()
        
        tone_scores = {}
        for tone, keywords in tone_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                tone_scores[tone] = score
        
        primary_tone = max(tone_scores, key=tone_scores.get) if tone_scores else "professional"
        
        return {
            "primary": primary_tone,
            "secondary": sorted(tone_scores.items(), key=lambda x: x[1], reverse=True)[1:3],
            "scores": tone_scores
        }
    
    def _analyze_style(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual style preferences"""
        style_keywords = {
            "minimal": ["minimal", "clean", "simple", "uncluttered"],
            "modern": ["modern", "contemporary", "current", "fresh"],
            "classic": ["classic", "traditional", "timeless", "elegant"],
            "bold": ["bold", "striking", "dramatic", "impactful"],
            "sophisticated": ["sophisticated", "refined", "premium", "luxury"]
        }
        
        text = " ".join([
            str(v) for v in brief.values() 
            if isinstance(v, str)
        ]).lower()
        
        style_scores = {}
        for style, keywords in style_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                style_scores[style] = score
        
        return {
            "detected_styles": style_scores,
            "primary_style": max(style_scores, key=style_scores.get) if style_scores else "modern"
        }
    
    def _analyze_mood(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze desired mood"""
        mood = self._detect_mood(" ".join([
            str(v) for v in brief.values() 
            if isinstance(v, str)
        ]))
        
        return {
            "primary_mood": mood or "professional",
            "emotional_direction": self._get_emotional_direction(mood or "professional")
        }
    
    def _get_emotional_direction(self, mood: str) -> Dict[str, Any]:
        """Get emotional direction for mood"""
        emotional_mapping = {
            "professional": {"energy": "calm", "warmth": "neutral", "sophistication": "high"},
            "playful": {"energy": "high", "warmth": "high", "sophistication": "medium"},
            "elegant": {"energy": "low", "warmth": "low", "sophistication": "high"},
            "energetic": {"energy": "high", "warmth": "medium", "sophistication": "medium"},
            "trustworthy": {"energy": "calm", "warmth": "medium", "sophistication": "high"}
        }
        
        return emotional_mapping.get(mood, {"energy": "medium", "warmth": "medium", "sophistication": "medium"})
    
    def _analyze_message_hierarchy(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze message hierarchy and importance"""
        messages = self._extract_key_messages(brief)
        
        # Determine primary message
        primary = brief.get("primary_message") or (messages[0] if messages else "")
        
        # Secondary messages
        secondary = [msg for msg in messages if msg != primary][:2]
        
        # CTA analysis
        cta = brief.get("cta_text", "Learn More")
        cta_importance = brief.get("cta_importance", "high")
        
        return {
            "primary_message": primary,
            "secondary_messages": secondary,
            "cta": {
                "text": cta,
                "importance": cta_importance,
                "placement": brief.get("cta_placement", "prominent")
            },
            "message_count": len([primary] + secondary),
            "complexity": "simple" if len(messages) <= 2 else "complex"
        }
    
    def _analyze_visual_preferences(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual preferences"""
        return {
            "color_preferences": brief.get("color_preferences", {}),
            "imagery_style": brief.get("imagery_style", "photography"),
            "layout_preference": brief.get("layout_preference", "balanced"),
            "typography_preference": brief.get("typography_preference", {}),
            "logo_prominence": brief.get("logo_prominence", "medium"),
            "visual_hierarchy": brief.get("visual_hierarchy", "clear")
        }
    
    async def _extract_constraints(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Extract design and business constraints"""
        constraints = {
            "technical_constraints": {},
            "brand_constraints": {},
            "legal_constraints": {},
            "budget_constraints": {},
            "timeline_constraints": {}
        }
        
        # Technical constraints
        constraints["technical_constraints"] = {
            "max_file_size": brief.get("max_file_size"),
            "supported_formats": brief.get("supported_formats", []),
            "color_limitations": brief.get("color_limitations", {}),
            "animation_allowed": brief.get("animation_allowed", False),
            "accessibility_requirements": brief.get("accessibility_requirements", {})
        }
        
        # Brand constraints
        constraints["brand_constraints"] = {
            "brand_guidelines": brief.get("brand_guidelines", {}),
            "logo_usage": brief.get("logo_usage", {}),
            "approved_colors": brief.get("approved_colors", []),
            "approved_fonts": brief.get("approved_fonts", []),
            "brand_voice": brief.get("brand_voice", {})
        }
        
        # Legal constraints
        constraints["legal_constraints"] = {
            "required_disclaimers": brief.get("required_disclaimers", []),
            "copyright_notices": brief.get("copyright_notices", []),
            "regulatory_requirements": brief.get("regulatory_requirements", []),
            "prohibited_claims": brief.get("prohibited_claims", [])
        }
        
        return constraints
    
    async def _generate_metadata(self, brief: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata about the analysis"""
        return {
            "analysis_version": "1.0",
            "confidence_scores": self._calculate_confidence_scores(analysis),
            "complexity_assessment": self._assess_complexity(brief, analysis),
            "estimated_effort": self._estimate_effort(analysis),
            "risk_factors": self._identify_risk_factors(brief, analysis)
        }
    
    def _calculate_confidence_scores(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for different aspects"""
        # Simplified confidence calculation
        return {
            "industry_detection": 0.8,
            "mood_detection": 0.7,
            "requirements_extraction": 0.9,
            "creative_direction": 0.8
        }
    
    def _assess_complexity(self, brief: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Assess overall complexity of the brief"""
        complexity_factors = [
            len(analysis["extracted_info"].get("key_messages", [])),
            len(analysis["structured_requirements"].get("technical", {})),
            len(analysis["constraints"].get("brand_constraints", {}))
        ]
        
        total_complexity = sum(complexity_factors)
        
        if total_complexity <= 5:
            return "low"
        elif total_complexity <= 10:
            return "medium"
        else:
            return "high"
    
    def _estimate_effort(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate effort required for campaign"""
        complexity = analysis["metadata"]["complexity_assessment"]
        
        effort_mapping = {
            "low": {"hours": "2-4", "iterations": "1-2", "review_rounds": "1"},
            "medium": {"hours": "4-8", "iterations": "2-3", "review_rounds": "2"},
            "high": {"hours": "8-16", "iterations": "3-5", "review_rounds": "3+"}
        }
        
        return effort_mapping.get(complexity, effort_mapping["medium"])
    
    def _identify_risk_factors(self, brief: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        # Missing information risks
        if not analysis["extracted_info"].get("target_audience"):
            risks.append("missing_target_audience")
        
        if not analysis["extracted_info"].get("primary_message"):
            risks.append("unclear_primary_message")
        
        # Constraint risks
        constraints = analysis["constraints"]
        if constraints.get("technical_constraints", {}).get("max_file_size"):
            risks.append("strict_file_size_limits")
        
        # Timeline risks
        timeline = brief.get("timeline", {})
        if timeline.get("deadline") and timeline.get("rush_job"):
            risks.append("tight_timeline")
        
        return risks
    
    def _load_industry_keywords(self) -> Dict[str, List[str]]:
        """Load industry classification keywords"""
        return {
            "technology": ["tech", "software", "app", "platform", "digital", "ai", "ml"],
            "finance": ["finance", "bank", "investment", "insurance", "fintech", "trading"],
            "healthcare": ["health", "medical", "pharma", "wellness", "clinic", "hospital"],
            "retail": ["retail", "ecommerce", "shop", "store", "fashion", "consumer"],
            "education": ["education", "learning", "course", "training", "school", "university"],
            "real_estate": ["real estate", "property", "housing", "mortgage", "rent"],
            "automotive": ["auto", "car", "vehicle", "automotive", "transport"],
            "travel": ["travel", "hotel", "tourism", "flight", "vacation", "booking"]
        }
    
    def _load_mood_keywords(self) -> Dict[str, List[str]]:
        """Load mood classification keywords"""
        return {
            "professional": ["professional", "corporate", "business", "formal", "serious"],
            "playful": ["playful", "fun", "creative", "colorful", "energetic", "vibrant"],
            "elegant": ["elegant", "sophisticated", "refined", "premium", "luxury", "classy"],
            "trustworthy": ["trustworthy", "reliable", "credible", "honest", "secure", "stable"],
            "innovative": ["innovative", "cutting-edge", "modern", "advanced", "revolutionary"],
            "warm": ["warm", "friendly", "approachable", "welcoming", "personal", "human"]
        }
    
    def _load_urgency_keywords(self) -> Dict[str, List[str]]:
        """Load urgency classification keywords"""
        return {
            "high": ["urgent", "asap", "immediate", "rush", "critical", "emergency", "now"],
            "medium": ["soon", "timely", "important", "priority", "needed"],
            "low": ["when possible", "eventually", "flexible", "no rush", "standard"]
        }
