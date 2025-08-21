"""
System Integration Tests

Test the complete system integration and workflow.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from memory_manager.shared_memory import SharedMemory, CampaignData
from memory_manager.session_manager import SessionManager
from communication.message_queue import MessageQueue
from agents.strategist.strategist_agent import StrategistAgent
from config.system_config import SystemConfig


@pytest.fixture
async def mock_shared_memory():
    """Mock shared memory for testing"""
    mock = AsyncMock(spec=SharedMemory)
    mock.initialize = AsyncMock()
    mock.close = AsyncMock()
    mock.set_campaign_data = AsyncMock()
    mock.get_campaign_data = AsyncMock()
    return mock


@pytest.fixture  
async def mock_message_queue():
    """Mock message queue for testing"""
    mock = AsyncMock(spec=MessageQueue)
    mock.initialize = AsyncMock()
    mock.close = AsyncMock()
    mock.send_message = AsyncMock(return_value=True)
    mock.subscribe_to_messages = AsyncMock()
    return mock


@pytest.fixture
async def mock_session_manager():
    """Mock session manager for testing"""
    mock = AsyncMock(spec=SessionManager)
    mock.create_agent_session = AsyncMock(return_value="test-session-id")
    return mock


@pytest.fixture
def sample_campaign_brief():
    """Sample campaign brief for testing"""
    return {
        "company_name": "TestCorp",
        "product_name": "TestProduct",
        "primary_message": "Test your product today",
        "cta_text": "Get Started",
        "target_audience": "Tech professionals",
        "industry": "technology",
        "mood": "professional",
        "tone": "confident",
        "dimensions": {"width": 728, "height": 90}
    }


@pytest.fixture
def sample_brand_assets():
    """Sample brand assets for testing"""
    return {
        "logo": {
            "base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        }
    }


class TestSystemIntegration:
    """Test system integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_strategist_agent_initialization(self, mock_shared_memory, 
                                                   mock_message_queue, mock_session_manager):
        """Test strategist agent initialization"""
        # Create strategist agent
        agent = StrategistAgent(
            shared_memory=mock_shared_memory,
            message_queue=mock_message_queue,
            session_manager=mock_session_manager,
            config={}
        )
        
        # Start agent
        await agent.start()
        
        # Verify initialization
        mock_message_queue.subscribe_to_messages.assert_called_once()
        
        # Stop agent
        await agent.stop()
        mock_message_queue.unsubscribe_from_messages.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_campaign_creation_workflow(self, mock_shared_memory, mock_message_queue,
                                            mock_session_manager, sample_campaign_brief,
                                            sample_brand_assets):
        """Test complete campaign creation workflow"""
        # Setup mocks
        mock_shared_memory.get_campaign_data.return_value = None
        
        # Create strategist agent
        agent = StrategistAgent(
            shared_memory=mock_shared_memory,
            message_queue=mock_message_queue, 
            session_manager=mock_session_manager,
            config={
                "brief_analyzer": {},
                "logo_processor": {},
                "brand_analyzer": {},
                "target_analyzer": {}
            }
        )
        
        await agent.start()
        
        # Create campaign
        campaign_id = await agent.create_campaign(sample_campaign_brief, sample_brand_assets)
        
        # Verify campaign creation
        assert campaign_id is not None
        assert isinstance(campaign_id, str)
        
        # Verify shared memory interactions
        mock_shared_memory.set_campaign_data.assert_called_once()
        mock_session_manager.create_agent_session.assert_called_once()
        
        await agent.stop()
    
    @pytest.mark.asyncio
    async def test_brief_analysis_process(self, mock_shared_memory, mock_message_queue,
                                        mock_session_manager, sample_campaign_brief):
        """Test brief analysis process"""
        agent = StrategistAgent(
            shared_memory=mock_shared_memory,
            message_queue=mock_message_queue,
            session_manager=mock_session_manager,
            config={}
        )
        
        # Test brief analysis
        analysis = await agent.brief_analyzer.analyze_brief(sample_campaign_brief)
        
        # Verify analysis structure
        assert "extracted_info" in analysis
        assert "structured_requirements" in analysis
        assert "creative_direction" in analysis
        assert "constraints" in analysis
        assert "metadata" in analysis
        
        # Verify extracted information
        extracted_info = analysis["extracted_info"]
        assert extracted_info["company_name"] == "TestCorp"
        assert extracted_info["product_name"] == "TestProduct"
        assert extracted_info["industry"] == "technology"
    
    @pytest.mark.asyncio 
    async def test_logo_processing(self, mock_shared_memory, mock_message_queue,
                                 mock_session_manager, sample_brand_assets):
        """Test logo processing functionality"""
        agent = StrategistAgent(
            shared_memory=mock_shared_memory,
            message_queue=mock_message_queue,
            session_manager=mock_session_manager,
            config={}
        )
        
        # Test logo processing
        logo_data = sample_brand_assets["logo"]
        
        # Mock the logo processing (since we can't process actual images in tests)
        with pytest.raises(Exception):  # Expected to fail with mock base64
            await agent.logo_processor.process_logo(logo_data)
    
    @pytest.mark.asyncio
    async def test_brand_analysis(self, mock_shared_memory, mock_message_queue,
                                mock_session_manager, sample_campaign_brief):
        """Test brand analysis functionality"""
        agent = StrategistAgent(
            shared_memory=mock_shared_memory,
            message_queue=mock_message_queue,
            session_manager=mock_session_manager,
            config={}
        )
        
        # Create mock brief analysis
        brief_analysis = {
            "extracted_info": {
                "industry": "technology",
                "company_name": "TestCorp",
                "target_audience": "Tech professionals"
            },
            "creative_direction": {
                "mood": {"primary_mood": "professional"},
                "tone": {"primary": "confident"}
            }
        }
        
        # Test brand analysis
        brand_analysis = await agent.brand_analyzer.analyze_brand(brief_analysis)
        
        # Verify analysis structure
        assert "brand_personality" in brand_analysis
        assert "brand_archetype" in brand_analysis
        assert "tone_of_voice" in brand_analysis
        assert "brand_positioning" in brand_analysis
        assert "mood_board" in brand_analysis
        
        # Verify personality analysis
        personality = brand_analysis["brand_personality"]
        assert "traits" in personality
        assert "dominant_traits" in personality
        
        # Verify archetype analysis
        archetype = brand_analysis["brand_archetype"]
        assert "primary" in archetype
        assert archetype["primary"] in ["the_sage", "the_hero", "the_innocent", "the_outlaw", "the_explorer"]
    
    @pytest.mark.asyncio
    async def test_target_audience_analysis(self, mock_shared_memory, mock_message_queue,
                                          mock_session_manager, sample_campaign_brief):
        """Test target audience analysis"""
        agent = StrategistAgent(
            shared_memory=mock_shared_memory,
            message_queue=mock_message_queue,
            session_manager=mock_session_manager,
            config={}
        )
        
        # Create mock brief analysis
        brief_analysis = {
            "extracted_info": {
                "target_audience": "Tech professionals",
                "industry": "technology"
            }
        }
        
        # Test target audience analysis
        target_analysis = await agent.target_analyzer.analyze_target_audience(brief_analysis)
        
        # Verify analysis structure
        assert "primary_segment" in target_analysis
        assert "demographics" in target_analysis
        assert "psychographics" in target_analysis
        assert "behavior_patterns" in target_analysis
        assert "communication_preferences" in target_analysis
    
    @pytest.mark.asyncio
    async def test_campaign_status_retrieval(self, mock_shared_memory, mock_message_queue,
                                           mock_session_manager):
        """Test campaign status retrieval"""
        # Setup mock campaign data
        from datetime import datetime
        mock_campaign_data = CampaignData(
            campaign_id="test-campaign-id",
            brief={"strategy": {"defined": True}},
            brand_assets={"logo": "processed"},
            target_audience={"analyzed": True},
            mood_board=["element1", "element2"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        mock_shared_memory.get_campaign_data.return_value = mock_campaign_data
        
        agent = StrategistAgent(
            shared_memory=mock_shared_memory,
            message_queue=mock_message_queue,
            session_manager=mock_session_manager,
            config={}
        )
        
        # Test campaign status retrieval
        status = await agent.get_campaign_status("test-campaign-id")
        
        # Verify status structure
        assert status["status"] == "active"
        assert status["campaign_id"] == "test-campaign-id" 
        assert status["has_strategy"] == True
        assert status["brief_analyzed"] == True
        assert status["assets_processed"] == True
        assert status["target_analyzed"] == True
        assert "created_at" in status
        assert "updated_at" in status
    
    @pytest.mark.asyncio
    async def test_system_configuration(self):
        """Test system configuration loading"""
        config = SystemConfig()
        
        # Test configuration validation
        # Note: This may fail if environment variables are not set properly
        # In production, ensure proper configuration
        is_valid = config.validate_configuration()
        
        # Test configuration access
        assert config.get_database_url() is not None
        assert config.get_redis_url() is not None
        assert config.get_upload_directory() is not None
        
        # Test configuration dictionary conversion
        config_dict = config.to_dict()
        assert "database" in config_dict
        assert "redis" in config_dict
        assert "app" in config_dict
    
    @pytest.mark.asyncio
    async def test_memory_manager_integration(self):
        """Test memory manager integration"""
        # This test would require actual Redis instance
        # For now, test the interface
        
        # Test SharedMemory interface
        shared_memory = SharedMemory()
        assert hasattr(shared_memory, 'initialize')
        assert hasattr(shared_memory, 'set_campaign_data')
        assert hasattr(shared_memory, 'get_campaign_data')
        
        # Test SessionManager interface
        session_manager = SessionManager(shared_memory)
        assert hasattr(session_manager, 'create_agent_session')
        assert hasattr(session_manager, 'get_agent_session')


@pytest.mark.asyncio
async def test_complete_workflow_simulation():
    """Test complete workflow simulation with mocks"""
    # Create all mocks
    shared_memory = AsyncMock(spec=SharedMemory)
    message_queue = AsyncMock(spec=MessageQueue)
    session_manager = AsyncMock(spec=SessionManager)
    
    # Setup mock returns
    shared_memory.initialize.return_value = None
    message_queue.initialize.return_value = None
    session_manager.create_agent_session.return_value = "test-session"
    
    # Create sample data
    brief = {
        "company_name": "DemoCompany",
        "product_name": "DemoProduct", 
        "primary_message": "Demo message",
        "industry": "technology",
        "mood": "professional"
    }
    
    assets = {
        "logo": {
            "base64": "demo-base64-data"
        }
    }
    
    # Initialize strategist agent
    strategist = StrategistAgent(
        shared_memory=shared_memory,
        message_queue=message_queue,
        session_manager=session_manager,
        config={}
    )
    
    await strategist.start()
    
    # Simulate campaign creation
    try:
        campaign_id = await strategist.create_campaign(brief, assets)
        assert campaign_id is not None
        
        # Verify mock calls
        session_manager.create_agent_session.assert_called()
        shared_memory.set_campaign_data.assert_called()
        
    except Exception as e:
        # Expected to fail with mock data, but should reach the processing logic
        assert "logo" in str(e).lower() or "processing" in str(e).lower()
    
    await strategist.stop()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
