"""
Strategist Agent

Main agent responsible for campaign analysis, brand processing,
and strategic direction for banner generation.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
from structlog import get_logger

from .brief_analyzer import BriefAnalyzer
from .logo_processor import LogoProcessor
from .brand_analyzer import BrandAnalyzer
from .target_analyzer import TargetAudienceAnalyzer
from communication.protocol import MessageProtocol, AgentIdentifiers, WorkflowProtocol
from communication.message_queue import MessageQueue
from memory_manager.shared_memory import SharedMemory, CampaignData
from memory_manager.session_manager import SessionManager

logger = get_logger(__name__)


class StrategistAgent:
    """
    Strategist Agent - Interface with advertisers and campaign strategy
    """
    
    def __init__(self, shared_memory: SharedMemory, message_queue: MessageQueue,
                 session_manager: SessionManager, config: Dict[str, Any] = None):
        self.agent_id = AgentIdentifiers.STRATEGIST
        self.shared_memory = shared_memory
        self.message_queue = message_queue
        self.session_manager = session_manager
        self.config = config or {}
        
        # Initialize sub-components
        self.brief_analyzer = BriefAnalyzer(config.get("brief_analyzer", {}))
        self.logo_processor = LogoProcessor(config.get("logo_processor", {}))
        self.brand_analyzer = BrandAnalyzer(config.get("brand_analyzer", {}))
        self.target_analyzer = TargetAudienceAnalyzer(config.get("target_analyzer", {}))
        
        self._running = False
        self._session_id = None
    
    async def start(self):
        """Start the strategist agent"""
        try:
            self._running = True
            
            # Subscribe to messages using the correct method name
            await self.message_queue.subscribe(
                self.agent_id, 
                self._handle_message
            )
            
            logger.info(f"Strategist Agent {self.agent_id} started")
            
        except Exception as e:
            logger.error(f"Failed to start Strategist Agent: {e}")
            raise
    
    async def stop(self):
        """Stop the strategist agent"""
        self._running = False
        # Use the correct method name
        await self.message_queue.unsubscribe(self.agent_id)
        logger.info(f"Strategist Agent {self.agent_id} stopped")
    
    async def _handle_message(self, message):
        """Handle incoming messages"""
        try:
            action = message.payload.get("action")
            
            if action == WorkflowProtocol.ANALYZE_BRIEF:
                await self._handle_analyze_brief(message)
            elif action == WorkflowProtocol.PROCESS_LOGO:
                await self._handle_process_logo(message)
            elif action == WorkflowProtocol.EXTRACT_BRAND_INFO:
                await self._handle_extract_brand_info(message)
            elif action == WorkflowProtocol.DEFINE_STRATEGY:
                await self._handle_define_strategy(message)
            else:
                logger.warning(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            
            # Send error response
            error_response = MessageProtocol.create_response(
                message, self.agent_id, False, error=str(e)
            )
            await self.message_queue.publish(message.reply_to or self.agent_id, error_response)
    
    async def create_campaign(self, brief: Dict[str, Any], brand_assets: Dict[str, Any] = None) -> str:
        """
        Create a new campaign and analyze it
        
        Args:
            brief: Campaign brief from advertiser
            brand_assets: Brand assets (logo, images, etc.)
            
        Returns:
            Campaign ID
        """
        try:
            campaign_id = str(uuid.uuid4())
            
            # Create session
            self._session_id = await self.session_manager.create_agent_session(
                self.agent_id, campaign_id
            )
            
            logger.info(f"Created campaign {campaign_id} with session {self._session_id}")
            
            # Analyze brief
            brief_analysis = await self.brief_analyzer.analyze_brief(brief)
            
            # Process brand assets if provided
            processed_assets = {}
            if brand_assets:
                if "logo" in brand_assets:
                    processed_logo = await self.logo_processor.process_logo(
                        brand_assets["logo"]
                    )
                    processed_assets["logo"] = processed_logo
                
                # Process other assets
                for asset_type, asset_data in brand_assets.items():
                    if asset_type != "logo":
                        processed_assets[asset_type] = asset_data
            
            # Analyze brand
            brand_analysis = await self.brand_analyzer.analyze_brand(
                brief_analysis, processed_assets
            )
            
            # Analyze target audience
            target_analysis = await self.target_analyzer.analyze_target_audience(
                brief_analysis
            )
            
            # Create campaign data
            campaign_data = CampaignData(
                campaign_id=campaign_id,
                brief=brief_analysis,
                brand_assets=processed_assets,
                target_audience=target_analysis,
                mood_board=brand_analysis.get("mood_board", []),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Store in shared memory
            await self.shared_memory.set_campaign_data(campaign_id, campaign_data)
            
            logger.info(f"Campaign {campaign_id} created and analyzed successfully")
            return campaign_id
            
        except Exception as e:
            logger.error(f"Failed to create campaign: {e}")
            raise
    
    async def _handle_analyze_brief(self, message):
        """Handle brief analysis request"""
        try:
            data = message.payload.get("data", {})
            brief = data.get("brief")
            
            if not brief:
                raise ValueError("Brief is required for analysis")
            
            # Analyze brief
            analysis = await self.brief_analyzer.analyze_brief(brief)
            
            # Send response
            response = MessageProtocol.create_response(
                message, self.agent_id, True, {"analysis": analysis}
            )
            await self.message_queue.publish(message.reply_to or message.sender, response)
            
        except Exception as e:
            logger.error(f"Brief analysis failed: {e}")
            response = MessageProtocol.create_response(
                message, self.agent_id, False, error=str(e)
            )
            await self.message_queue.publish(message.reply_to or message.sender, response)
    
    async def _handle_process_logo(self, message):
        """Handle logo processing request"""
        try:
            data = message.payload.get("data", {})
            logo_data = data.get("logo")
            
            if not logo_data:
                raise ValueError("Logo data is required for processing")
            
            # Process logo
            processed_logo = await self.logo_processor.process_logo(logo_data)
            
            # Send response
            response = MessageProtocol.create_response(
                message, self.agent_id, True, {"processed_logo": processed_logo}
            )
            await self.message_queue.publish(message.reply_to or message.sender, response)
            
        except Exception as e:
            logger.error(f"Logo processing failed: {e}")
            response = MessageProtocol.create_response(
                message, self.agent_id, False, error=str(e)
            )
            await self.message_queue.publish(message.reply_to or message.sender, response)
    
    async def _handle_extract_brand_info(self, message):
        """Handle brand information extraction request"""
        try:
            data = message.payload.get("data", {})
            brief_analysis = data.get("brief_analysis")
            brand_assets = data.get("brand_assets", {})
            
            if not brief_analysis:
                raise ValueError("Brief analysis is required for brand extraction")
            
            # Analyze brand
            brand_analysis = await self.brand_analyzer.analyze_brand(
                brief_analysis, brand_assets
            )
            
            # Send response
            response = MessageProtocol.create_response(
                message, self.agent_id, True, {"brand_analysis": brand_analysis}
            )
            await self.message_queue.publish(message.reply_to or message.sender, response)
            
        except Exception as e:
            logger.error(f"Brand analysis failed: {e}")
            response = MessageProtocol.create_response(
                message, self.agent_id, False, error=str(e)
            )
            await self.message_queue.publish(message.reply_to or message.sender, response)
    
    async def _handle_define_strategy(self, message):
        """Handle strategy definition request"""
        try:
            campaign_id = message.payload.get("campaign_id")
            
            if not campaign_id:
                raise ValueError("Campaign ID is required for strategy definition")
            
            # Get campaign data
            campaign_data = await self.shared_memory.get_campaign_data(campaign_id)
            if not campaign_data:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            # Define strategy
            strategy = await self._define_campaign_strategy(campaign_data)
            
            # Update campaign data with strategy
            campaign_data.brief["strategy"] = strategy
            campaign_data.updated_at = datetime.utcnow()
            await self.shared_memory.set_campaign_data(campaign_id, campaign_data)
            
            # Send response
            response = MessageProtocol.create_response(
                message, self.agent_id, True, {"strategy": strategy}
            )
            await self.message_queue.publish(message.reply_to or message.sender, response)
            
            # Notify next agent (Background Designer)
            background_request = MessageProtocol.create_request(
                self.agent_id,
                AgentIdentifiers.BACKGROUND_DESIGNER,
                WorkflowProtocol.GENERATE_BACKGROUND,
                {"campaign_id": campaign_id}
            )
            await self.message_queue.publish(AgentIdentifiers.BACKGROUND_DESIGNER, background_request)
            
        except Exception as e:
            logger.error(f"Strategy definition failed: {e}")
            response = MessageProtocol.create_response(
                message, self.agent_id, False, error=str(e)
            )
            await self.message_queue.publish(message.reply_to or message.sender, response)
    
    async def _define_campaign_strategy(self, campaign_data: CampaignData) -> Dict[str, Any]:
        """Define comprehensive campaign strategy"""
        try:
            brief = campaign_data.brief
            brand_assets = campaign_data.brand_assets
            target_audience = campaign_data.target_audience
            
            # Extract key strategy elements
            strategy = {
                "visual_style": self._determine_visual_style(brief, brand_assets),
                "color_palette": self._extract_color_palette(brand_assets),
                "typography_style": self._determine_typography_style(brief, target_audience),
                "layout_preferences": self._determine_layout_preferences(brief),
                "messaging_strategy": self._define_messaging_strategy(brief, target_audience),
                "creative_direction": self._define_creative_direction(brief, brand_assets),
                "constraints": self._extract_constraints(brief),
                "success_metrics": self._define_success_metrics(brief)
            }
            
            logger.info(f"Strategy defined for campaign {campaign_data.campaign_id}")
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to define strategy: {e}")
            raise
    
    def _determine_visual_style(self, brief: Dict[str, Any], 
                               brand_assets: Dict[str, Any]) -> Dict[str, Any]:
        """Determine visual style based on brief and brand assets"""
        # Extract mood and tone from brief
        mood = brief.get("mood", "professional")
        tone = brief.get("tone", "neutral")
        industry = brief.get("industry", "general")
        
        # Analyze logo if available
        logo_style = "modern"
        if "logo" in brand_assets and "analysis" in brand_assets["logo"]:
            logo_analysis = brand_assets["logo"]["analysis"]
            logo_style = logo_analysis.get("style", "modern")
        
        return {
            "mood": mood,
            "tone": tone,
            "style": logo_style,
            "industry": industry,
            "aesthetic": self._map_mood_to_aesthetic(mood),
            "visual_hierarchy": "clear" if tone == "professional" else "dynamic"
        }
    
    def _extract_color_palette(self, brand_assets: Dict[str, Any]) -> Dict[str, Any]:
        """Extract color palette from brand assets"""
        if "logo" in brand_assets and "colors" in brand_assets["logo"]:
            return brand_assets["logo"]["colors"]
        
        # Default color palette
        return {
            "primary": "#2563eb",
            "secondary": "#64748b", 
            "accent": "#f59e0b",
            "neutral": "#f8fafc",
            "text": "#1e293b"
        }
    
    def _determine_typography_style(self, brief: Dict[str, Any], 
                                  target_audience: Dict[str, Any]) -> Dict[str, Any]:
        """Determine typography style"""
        industry = brief.get("industry", "general")
        audience_age = target_audience.get("demographics", {}).get("age_range", "25-45")
        tone = brief.get("tone", "neutral")
        
        # Map characteristics to font choices
        if industry in ["tech", "startup", "software"]:
            font_family = "modern_sans"
        elif industry in ["finance", "law", "consulting"]:
            font_family = "traditional_serif"
        elif industry in ["creative", "design", "arts"]:
            font_family = "creative_display"
        else:
            font_family = "versatile_sans"
        
        return {
            "primary_font": font_family,
            "font_weight": "bold" if tone == "confident" else "medium",
            "font_size_scale": "large" if "young" in audience_age else "medium",
            "letter_spacing": "normal",
            "line_height": "comfortable"
        }
    
    def _determine_layout_preferences(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Determine layout preferences"""
        message_complexity = len(brief.get("key_messages", []))
        cta_importance = brief.get("cta_importance", "medium")
        
        return {
            "layout_style": "minimal" if message_complexity <= 2 else "structured",
            "cta_prominence": cta_importance,
            "logo_placement": "top-left",
            "text_alignment": "left",
            "spacing": "generous" if message_complexity <= 2 else "compact"
        }
    
    def _define_messaging_strategy(self, brief: Dict[str, Any], 
                                 target_audience: Dict[str, Any]) -> Dict[str, Any]:
        """Define messaging strategy"""
        return {
            "primary_message": brief.get("primary_message", ""),
            "secondary_messages": brief.get("key_messages", []),
            "cta_text": brief.get("cta_text", "Learn More"),
            "tone_of_voice": target_audience.get("communication_style", "professional"),
            "value_proposition": brief.get("value_proposition", ""),
            "urgency_level": brief.get("urgency", "medium")
        }
    
    def _define_creative_direction(self, brief: Dict[str, Any], 
                                 brand_assets: Dict[str, Any]) -> Dict[str, Any]:
        """Define creative direction"""
        return {
            "concept": brief.get("concept", "product_focused"),
            "imagery_style": "clean_modern",
            "composition": "balanced",
            "contrast_level": "medium",
            "texture_usage": "minimal",
            "effects": "subtle"
        }
    
    def _extract_constraints(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Extract design constraints"""
        return {
            "dimensions": brief.get("dimensions", {"width": 728, "height": 90}),
            "file_size_limit": brief.get("file_size_limit", "150KB"),
            "format_requirements": brief.get("formats", ["SVG", "PNG"]),
            "brand_guidelines": brief.get("brand_guidelines", {}),
            "legal_requirements": brief.get("legal_text", []),
            "accessibility": brief.get("accessibility_requirements", {})
        }
    
    def _define_success_metrics(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Define success metrics"""
        return {
            "primary_kpi": brief.get("primary_kpi", "click_through_rate"),
            "target_ctr": brief.get("target_ctr", 0.02),
            "conversion_goal": brief.get("conversion_goal", "lead_generation"),
            "brand_recall": brief.get("brand_recall_target", 0.25),
            "engagement_metrics": brief.get("engagement_metrics", ["clicks", "views"])
        }
    
    def _map_mood_to_aesthetic(self, mood: str) -> str:
        """Map mood to aesthetic style"""
        mood_mapping = {
            "professional": "clean_corporate",
            "playful": "vibrant_friendly",
            "elegant": "sophisticated_minimal",
            "energetic": "dynamic_bold",
            "trustworthy": "stable_reliable",
            "innovative": "modern_cutting_edge",
            "warm": "approachable_human",
            "premium": "luxury_refined"
        }
        return mood_mapping.get(mood, "balanced_versatile")
    
    async def get_campaign_status(self, campaign_id: str) -> Dict[str, Any]:
        """Get current campaign status"""
        try:
            campaign_data = await self.shared_memory.get_campaign_data(campaign_id)
            if not campaign_data:
                return {"status": "not_found"}
            
            return {
                "status": "active",
                "campaign_id": campaign_id,
                "created_at": campaign_data.created_at.isoformat(),
                "updated_at": campaign_data.updated_at.isoformat(),
                "has_strategy": "strategy" in campaign_data.brief,
                "brief_analyzed": bool(campaign_data.brief),
                "assets_processed": bool(campaign_data.brand_assets),
                "target_analyzed": bool(campaign_data.target_audience)
            }
            
        except Exception as e:
            logger.error(f"Failed to get campaign status: {e}")
            return {"status": "error", "error": str(e)}
