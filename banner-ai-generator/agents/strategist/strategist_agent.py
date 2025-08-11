"""
Strategist Agent - Campaign analysis and strategic direction
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from communication.protocol import AgentMessage, AgentResponse, MessageType, ResponseStatus, AgentType
from communication.message_queue import MessageQueue, MessagePriority
from communication.event_dispatcher import EventDispatcher, EventType
from memory_manager.shared_memory import SharedMemoryManager
from config.environments import get_agent_config, get_model_config

from .brief_analyzer import BriefAnalyzer, BriefAnalysisResult
from .logo_processor import LogoProcessor, LogoProcessingResult
from .brand_analyzer import BrandAnalyzer, BrandAnalysisResult
from .target_analyzer import TargetAnalyzer, TargetAnalysisResult

logger = logging.getLogger(__name__)

@dataclass
class StrategicDirection:
    """Strategic direction output from Strategist"""
    session_id: str
    campaign_id: str
    
    # Analysis results
    brief_analysis: BriefAnalysisResult
    brand_analysis: BrandAnalysisResult
    target_analysis: TargetAnalysisResult
    logo_processing: Optional[LogoProcessingResult] = None
    
    # Strategic recommendations
    mood_board: Dict[str, Any] = field(default_factory=dict)
    color_palette: List[str] = field(default_factory=list)
    design_direction: Dict[str, Any] = field(default_factory=dict)
    messaging_strategy: Dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    brand_consistency_score: float = 0.0
    target_alignment_score: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    confidence_score: float = 0.0

class StrategistAgent:
    """
    Strategist Agent - Analyzes campaign briefs and provides strategic direction
    Main entry point for the multi-agent workflow
    """
    
    def __init__(self,
                 message_queue: MessageQueue,
                 event_dispatcher: EventDispatcher,
                 shared_memory: SharedMemoryManager,
                 agent_id: str = "strategist_001"):
        
        self.agent_id = agent_id
        self.agent_type = AgentType.STRATEGIST
        self.message_queue = message_queue
        self.event_dispatcher = event_dispatcher
        self.shared_memory = shared_memory
        
        # Load configuration
        self.config = get_agent_config("strategist")
        if not self.config:
            raise ValueError("Strategist configuration not found")
        
        # Initialize analyzers
        self.brief_analyzer = BriefAnalyzer()
        self.logo_processor = LogoProcessor()
        self.brand_analyzer = BrandAnalyzer()
        self.target_analyzer = TargetAnalyzer()
        
        # State management
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
        # Statistics
        self._stats = {
            'campaigns_analyzed': 0,
            'logos_processed': 0,
            'processing_errors': 0,
            'average_processing_time': 0.0
        }
        
        # Setup message handling
        self.message_queue.subscribe(self.agent_id, self._handle_message)
        
        # Setup event handling
        self.event_dispatcher.subscribe(EventType.SYSTEM_ERROR, self._handle_system_error)
        
        logger.info(f"Strategist Agent initialized: {self.agent_id}")
    
    async def start(self) -> None:
        """Start the strategist agent"""
        if self._running:
            return
        
        self._running = True
        
        # Start processing loop
        asyncio.create_task(self._process_requests())
        
        # Register with coordinator
        self.event_dispatcher.dispatch_event(
            EventType.AGENT_STARTED,
            source=self.agent_id,
            data={
                'agent_type': self.agent_type.value,
                'capabilities': self.config.capabilities,
                'status': 'active'
            }
        )
        
        logger.info(f"Strategist Agent started: {self.agent_id}")
    
    async def stop(self) -> None:
        """Stop the strategist agent"""
        self._running = False
        
        # Dispatch stop event
        self.event_dispatcher.dispatch_event(
            EventType.AGENT_STOPPED,
            source=self.agent_id,
            data={'agent_type': self.agent_type.value}
        )
        
        logger.info(f"Strategist Agent stopped: {self.agent_id}")
    
    def _handle_message(self, message) -> None:
        """Handle incoming messages"""
        try:
            agent_message = message.content
            
            # Add to processing queue
            asyncio.create_task(self._queue_request(agent_message))
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self._send_error_response(agent_message, str(e))
    
    async def _queue_request(self, message: AgentMessage) -> None:
        """Queue request for processing"""
        await self._processing_queue.put(message)
    
    async def _process_requests(self) -> None:
        """Process requests from queue"""
        while self._running:
            try:
                # Get request from queue with timeout
                message = await asyncio.wait_for(
                    self._processing_queue.get(), 
                    timeout=1.0
                )
                
                # Process request
                await self._process_request(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing requests: {e}")
    
    async def _process_request(self, message: AgentMessage) -> None:
        """Process individual request"""
        start_time = datetime.now()
        
        try:
            # Route based on action
            if message.action == "analyze_brief":
                response = await self._analyze_brief(message)
            elif message.action == "process_logo":
                response = await self._process_logo(message)
            elif message.action == "analyze_brand":
                response = await self._analyze_brand(message)
            elif message.action == "analyze_target":
                response = await self._analyze_target(message)
            elif message.action == "create_strategic_direction":
                response = await self._create_strategic_direction(message)
            elif message.action == "validate_brand_assets":
                response = await self._validate_brand_assets(message)
            elif message.action == "ping":
                response = self._handle_ping(message)
            else:
                response = AgentResponse(
                    request_id=message.message_id,
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    status=ResponseStatus.ERROR,
                    error=f"Unknown action: {message.action}"
                )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            response.processing_time = processing_time
            
            # Update statistics
            self._update_stats(message.action, processing_time, response.status == ResponseStatus.SUCCESS)
            
            # Send response
            self.message_queue.send_response(response)
            
        except Exception as e:
            logger.error(f"Error processing request {message.action}: {e}")
            self._send_error_response(message, str(e))
    
    async def _analyze_brief(self, message: AgentMessage) -> AgentResponse:
        """Analyze campaign brief"""
        try:
            brief_data = message.payload.get('brief', {})
            session_id = message.session_id
            
            if not brief_data:
                raise ValueError("No brief data provided")
            
            # Analyze brief
            analysis_result = await self.brief_analyzer.analyze(brief_data)
            
            # Store in shared memory
            if session_id:
                self.shared_memory.set_data(
                    session_id, 
                    'brief_analysis', 
                    analysis_result.to_dict(),
                    agent_id=self.agent_id
                )
            
            return AgentResponse(
                request_id=message.message_id,
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                status=ResponseStatus.SUCCESS,
                result={
                    'analysis': analysis_result.to_dict(),
                    'session_id': session_id
                }
            )
            
        except Exception as e:
            raise Exception(f"Brief analysis failed: {e}")
    
    async def _process_logo(self, message: AgentMessage) -> AgentResponse:
        """Process logo file"""
        try:
            logo_data = message.payload.get('logo', {})
            session_id = message.session_id
            
            if not logo_data:
                raise ValueError("No logo data provided")
            
            # Process logo
            processing_result = await self.logo_processor.process(logo_data)
            
            # Store in shared memory
            if session_id:
                self.shared_memory.set_data(
                    session_id,
                    'logo_processing',
                    processing_result.to_dict(),
                    agent_id=self.agent_id
                )
                
                self._stats['logos_processed'] += 1
            
            return AgentResponse(
                request_id=message.message_id,
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                status=ResponseStatus.SUCCESS,
                result={
                    'processing_result': processing_result.to_dict(),
                    'session_id': session_id
                }
            )
            
        except Exception as e:
            raise Exception(f"Logo processing failed: {e}")
    
    async def _analyze_brand(self, message: AgentMessage) -> AgentResponse:
        """Analyze brand assets and identity"""
        try:
            brand_data = message.payload.get('brand_assets', {})
            brief_context = message.payload.get('brief_context', {})
            session_id = message.session_id
            
            if not brand_data:
                raise ValueError("No brand data provided")
            
            # Analyze brand
            analysis_result = await self.brand_analyzer.analyze(brand_data, brief_context)
            
            # Store in shared memory
            if session_id:
                self.shared_memory.set_data(
                    session_id,
                    'brand_analysis', 
                    analysis_result.to_dict(),
                    agent_id=self.agent_id
                )
            
            return AgentResponse(
                request_id=message.message_id,
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                status=ResponseStatus.SUCCESS,
                result={
                    'analysis': analysis_result.to_dict(),
                    'session_id': session_id
                }
            )
            
        except Exception as e:
            raise Exception(f"Brand analysis failed: {e}")
    
    async def _analyze_target(self, message: AgentMessage) -> AgentResponse:
        """Analyze target audience"""
        try:
            target_data = message.payload.get('target_audience', {})
            brief_context = message.payload.get('brief_context', {})
            session_id = message.session_id
            
            if not target_data:
                raise ValueError("No target audience data provided")
            
            # Analyze target audience
            analysis_result = await self.target_analyzer.analyze(target_data, brief_context)
            
            # Store in shared memory
            if session_id:
                self.shared_memory.set_data(
                    session_id,
                    'target_analysis',
                    analysis_result.to_dict(),
                    agent_id=self.agent_id
                )
            
            return AgentResponse(
                request_id=message.message_id,
                from_agent=self.agent_id, 
                to_agent=message.from_agent,
                status=ResponseStatus.SUCCESS,
                result={
                    'analysis': analysis_result.to_dict(),
                    'session_id': session_id
                }
            )
            
        except Exception as e:
            raise Exception(f"Target analysis failed: {e}")
    
    async def _create_strategic_direction(self, message: AgentMessage) -> AgentResponse:
        """Create comprehensive strategic direction"""
        try:
            session_id = message.session_id
            campaign_id = message.payload.get('campaign_id', str(uuid.uuid4()))
            
            if not session_id:
                raise ValueError("No session ID provided")
            
            # Get all analysis results from shared memory
            brief_analysis_data = self.shared_memory.get_data(session_id, 'brief_analysis')
            brand_analysis_data = self.shared_memory.get_data(session_id, 'brand_analysis')
            target_analysis_data = self.shared_memory.get_data(session_id, 'target_analysis')
            logo_processing_data = self.shared_memory.get_data(session_id, 'logo_processing')
            
            if not all([brief_analysis_data, brand_analysis_data, target_analysis_data]):
                raise ValueError("Missing analysis data. Please complete all analysis steps first.")
            
            # Recreate analysis objects
            brief_analysis = BriefAnalysisResult.from_dict(brief_analysis_data)
            brand_analysis = BrandAnalysisResult.from_dict(brand_analysis_data)
            target_analysis = TargetAnalysisResult.from_dict(target_analysis_data)
            logo_processing = LogoProcessingResult.from_dict(logo_processing_data) if logo_processing_data else None
            
            # Create strategic direction
            strategic_direction = await self._synthesize_strategic_direction(
                session_id, campaign_id, brief_analysis, brand_analysis, target_analysis, logo_processing
            )
            
            # Store in shared memory
            self.shared_memory.set_data(
                session_id,
                'strategic_direction',
                strategic_direction.to_dict(),
                agent_id=self.agent_id
            )
            
            # Dispatch event for workflow continuation
            self.event_dispatcher.dispatch_event(
                EventType.DESIGN_CREATED,
                source=self.agent_id,
                data={
                    'session_id': session_id,
                    'campaign_id': campaign_id,
                    'strategic_direction_ready': True
                },
                session_id=session_id
            )
            
            self._stats['campaigns_analyzed'] += 1
            
            return AgentResponse(
                request_id=message.message_id,
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                status=ResponseStatus.SUCCESS,
                result={
                    'strategic_direction': strategic_direction.to_dict(),
                    'session_id': session_id,
                    'campaign_id': campaign_id
                }
            )
            
        except Exception as e:
            raise Exception(f"Strategic direction creation failed: {e}")
    
    async def _synthesize_strategic_direction(self,
                                            session_id: str,
                                            campaign_id: str,
                                            brief_analysis: BriefAnalysisResult,
                                            brand_analysis: BrandAnalysisResult,
                                            target_analysis: TargetAnalysisResult,
                                            logo_processing: Optional[LogoProcessingResult]) -> StrategicDirection:
        """Synthesize all analysis into strategic direction"""
        
        # Create mood board based on brief and brand
        mood_board = {
            'style': brief_analysis.design_requirements.get('style', 'modern'),
            'tone': brief_analysis.tone_of_voice,
            'emotions': target_analysis.emotional_triggers,
            'visual_themes': brand_analysis.visual_themes
        }
        
        # Create color palette
        color_palette = brand_analysis.color_palette
        if not color_palette and logo_processing:
            color_palette = logo_processing.extracted_colors
        
        # Design direction
        design_direction = {
            'primary_message': brief_analysis.key_messages[0] if brief_analysis.key_messages else "",
            'visual_hierarchy': brief_analysis.design_requirements.get('hierarchy', 'message-first'),
            'layout_style': self._determine_layout_style(brief_analysis, target_analysis),
            'imagery_style': brand_analysis.imagery_style,
            'typography_direction': self._determine_typography_direction(brand_analysis, target_analysis)
        }
        
        # Messaging strategy
        messaging_strategy = {
            'primary_message': brief_analysis.key_messages[0] if brief_analysis.key_messages else "",
            'secondary_messages': brief_analysis.key_messages[1:3],
            'cta_text': brief_analysis.call_to_action,
            'value_proposition': brief_analysis.value_proposition,
            'tone_guidelines': brief_analysis.tone_of_voice
        }
        
        # Calculate validation scores
        brand_consistency_score = self._calculate_brand_consistency(brand_analysis, brief_analysis)
        target_alignment_score = self._calculate_target_alignment(target_analysis, brief_analysis)
        
        # Overall confidence score
        confidence_score = (brand_consistency_score + target_alignment_score) / 2
        
        return StrategicDirection(
            session_id=session_id,
            campaign_id=campaign_id,
            brief_analysis=brief_analysis,
            brand_analysis=brand_analysis,
            target_analysis=target_analysis,
            logo_processing=logo_processing,
            mood_board=mood_board,
            color_palette=color_palette,
            design_direction=design_direction,
            messaging_strategy=messaging_strategy,
            brand_consistency_score=brand_consistency_score,
            target_alignment_score=target_alignment_score,
            confidence_score=confidence_score
        )
    
    def _determine_layout_style(self, brief_analysis: BriefAnalysisResult, target_analysis: TargetAnalysisResult) -> str:
        """Determine optimal layout style"""
        # Logic to determine layout based on brief and target
        if 'professional' in target_analysis.psychographics:
            return 'clean-professional'
        elif 'young' in target_analysis.demographics.get('age', ''):
            return 'dynamic-modern'
        elif brief_analysis.campaign_type == 'product_launch':
            return 'hero-focused'
        else:
            return 'balanced-hierarchy'
    
    def _determine_typography_direction(self, brand_analysis: BrandAnalysisResult, target_analysis: TargetAnalysisResult) -> Dict[str, str]:
        """Determine typography direction"""
        return {
            'primary_font_style': brand_analysis.brand_personality.get('font_style', 'modern-sans'),
            'hierarchy_approach': 'clear-contrast',
            'readability_priority': 'high' if 'accessibility' in target_analysis.preferences else 'medium'
        }
    
    def _calculate_brand_consistency(self, brand_analysis: BrandAnalysisResult, brief_analysis: BriefAnalysisResult) -> float:
        """Calculate brand consistency score"""
        score = 0.0
        factors = 0
        
        # Check tone alignment
        if brand_analysis.brand_personality.get('tone') == brief_analysis.tone_of_voice:
            score += 1.0
        factors += 1
        
        # Check message alignment
        brand_values = brand_analysis.brand_values
        brief_messages = brief_analysis.key_messages
        if any(value.lower() in ' '.join(brief_messages).lower() for value in brand_values):
            score += 1.0
        factors += 1
        
        # Check visual alignment
        if brand_analysis.visual_themes:
            score += 1.0
        factors += 1
        
        return (score / factors) * 10 if factors > 0 else 5.0
    
    def _calculate_target_alignment(self, target_analysis: TargetAnalysisResult, brief_analysis: BriefAnalysisResult) -> float:
        """Calculate target audience alignment score"""
        score = 0.0
        factors = 0
        
        # Check emotional trigger alignment
        if target_analysis.emotional_triggers and brief_analysis.emotional_appeal:
            if any(trigger in brief_analysis.emotional_appeal for trigger in target_analysis.emotional_triggers):
                score += 1.0
        factors += 1
        
        # Check communication style alignment
        if target_analysis.communication_preferences.get('style') == brief_analysis.tone_of_voice:
            score += 1.0
        factors += 1
        
        # Check platform alignment
        brief_platforms = brief_analysis.platforms
        target_platforms = target_analysis.preferred_platforms
        if any(platform in brief_platforms for platform in target_platforms):
            score += 1.0
        factors += 1
        
        return (score / factors) * 10 if factors > 0 else 5.0
    
    async def _validate_brand_assets(self, message: AgentMessage) -> AgentResponse:
        """Validate brand assets"""
        try:
            assets = message.payload.get('assets', {})
            
            validation_results = {
                'valid_assets': [],
                'invalid_assets': [],
                'warnings': [],
                'recommendations': []
            }
            
            # Validate each asset
            for asset_name, asset_data in assets.items():
                is_valid, issues = await self._validate_single_asset(asset_name, asset_data)
                
                if is_valid:
                    validation_results['valid_assets'].append(asset_name)
                else:
                    validation_results['invalid_assets'].append({
                        'asset': asset_name,
                        'issues': issues
                    })
            
            return AgentResponse(
                request_id=message.message_id,
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                status=ResponseStatus.SUCCESS,
                result=validation_results
            )
            
        except Exception as e:
            raise Exception(f"Asset validation failed: {e}")
    
    async def _validate_single_asset(self, asset_name: str, asset_data: Dict[str, Any]) -> tuple:
        """Validate individual asset"""
        issues = []
        
        # Check file format
        if 'format' in asset_data:
            if asset_data['format'].lower() not in self.config.supported_logo_formats:
                issues.append(f"Unsupported format: {asset_data['format']}")
        
        # Check file size
        if 'size_mb' in asset_data:
            if asset_data['size_mb'] > self.config.logo_max_size_mb:
                issues.append(f"File too large: {asset_data['size_mb']}MB > {self.config.logo_max_size_mb}MB")
        
        # Check dimensions if available
        if 'dimensions' in asset_data:
            width, height = asset_data['dimensions']
            if width < 100 or height < 100:
                issues.append("Resolution too low for quality output")
        
        return len(issues) == 0, issues
    
    def _handle_ping(self, message: AgentMessage) -> AgentResponse:
        """Handle ping message"""
        return AgentResponse(
            request_id=message.message_id,
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            status=ResponseStatus.SUCCESS,
            result={
                'pong': True,
                'agent_id': self.agent_id,
                'agent_type': self.agent_type.value,
                'status': 'active',
                'stats': self.get_stats()
            }
        )
    
    def _handle_system_error(self, event) -> None:
        """Handle system error events"""
        logger.error(f"System error received: {event.data}")
        self._stats['processing_errors'] += 1
    
    def _send_error_response(self, message: AgentMessage, error: str) -> None:
        """Send error response"""
        response = AgentResponse(
            request_id=message.message_id,
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            status=ResponseStatus.ERROR,
            error=error
        )
        
        self.message_queue.send_response(response)
        self._stats['processing_errors'] += 1
    
    def _update_stats(self, action: str, processing_time: float, success: bool) -> None:
        """Update agent statistics"""
        if success:
            # Update average processing time
            current_avg = self._stats['average_processing_time']
            total_processed = sum([
                self._stats['campaigns_analyzed'],
                self._stats['logos_processed']
            ])
            
            if total_processed > 0:
                self._stats['average_processing_time'] = (
                    (current_avg * (total_processed - 1) + processing_time) / total_processed
                )
            else:
                self._stats['average_processing_time'] = processing_time
        else:
            self._stats['processing_errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self._stats,
            'active_sessions': len(self._active_sessions),
            'queue_size': self._processing_queue.qsize(),
            'uptime': (datetime.now() - datetime.now()).total_seconds(),  # This would track actual start time
            'status': 'running' if self._running else 'stopped'
        }
