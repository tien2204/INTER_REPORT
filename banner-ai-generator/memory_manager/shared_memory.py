"""
Shared Memory Module

Manages shared memory between all AI agents in the system.
Provides thread-safe access to shared data structures.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import redis.asyncio as redis
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class CampaignData:
    """Campaign data structure"""
    campaign_id: str
    brief: Dict[str, Any]
    brand_assets: Dict[str, Any]
    target_audience: Dict[str, Any]
    mood_board: List[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class DesignIteration:
    """Design iteration data structure"""
    iteration_id: str
    campaign_id: str
    background_url: Optional[str]
    blueprint: Optional[Dict[str, Any]]
    svg_code: Optional[str]
    figma_code: Optional[str]
    feedback: List[Dict[str, Any]]
    status: str  # "in_progress", "completed", "failed"
    created_at: datetime


class SharedMemory:
    """
    Shared memory manager using Redis for distributed access
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self._local_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self._redis.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
    
    async def set_campaign_data(self, campaign_id: str, data: CampaignData):
        """Store campaign data in shared memory"""
        try:
            serialized_data = json.dumps(asdict(data), default=str)
            await self._redis.hset(
                f"campaign:{campaign_id}",
                mapping={"data": serialized_data}
            )
            
            # Cache locally
            with self._cache_lock:
                self._local_cache[f"campaign:{campaign_id}"] = data
                
            logger.info(f"Campaign data stored for {campaign_id}")
        except Exception as e:
            logger.error(f"Failed to store campaign data: {e}")
            raise
    
    async def get_campaign_data(self, campaign_id: str) -> Optional[CampaignData]:
        """Retrieve campaign data from shared memory"""
        try:
            # Check local cache first
            cache_key = f"campaign:{campaign_id}"
            with self._cache_lock:
                if cache_key in self._local_cache:
                    return self._local_cache[cache_key]
            
            # Get from Redis
            data = await self._redis.hget(f"campaign:{campaign_id}", "data")
            if data:
                parsed_data = json.loads(data)
                campaign_data = CampaignData(**parsed_data)
                
                # Update local cache
                with self._cache_lock:
                    self._local_cache[cache_key] = campaign_data
                
                return campaign_data
            
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve campaign data: {e}")
            return None
    
    async def add_design_iteration(self, iteration: DesignIteration):
        """Add a new design iteration"""
        try:
            serialized_iteration = json.dumps(asdict(iteration), default=str)
            await self._redis.hset(
                f"iteration:{iteration.iteration_id}",
                mapping={"data": serialized_iteration}
            )
            
            # Add to campaign's iteration list
            await self._redis.lpush(
                f"campaign:{iteration.campaign_id}:iterations",
                iteration.iteration_id
            )
            
            logger.info(f"Design iteration {iteration.iteration_id} added")
        except Exception as e:
            logger.error(f"Failed to add design iteration: {e}")
            raise
    
    async def get_design_iteration(self, iteration_id: str) -> Optional[DesignIteration]:
        """Get a specific design iteration"""
        try:
            data = await self._redis.hget(f"iteration:{iteration_id}", "data")
            if data:
                parsed_data = json.loads(data)
                return DesignIteration(**parsed_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get design iteration: {e}")
            return None
    
    async def get_campaign_iterations(self, campaign_id: str) -> List[DesignIteration]:
        """Get all iterations for a campaign"""
        try:
            iteration_ids = await self._redis.lrange(
                f"campaign:{campaign_id}:iterations", 0, -1
            )
            
            iterations = []
            for iteration_id in iteration_ids:
                iteration = await self.get_design_iteration(iteration_id)
                if iteration:
                    iterations.append(iteration)
            
            return iterations
        except Exception as e:
            logger.error(f"Failed to get campaign iterations: {e}")
            return []
    
    async def update_iteration_status(self, iteration_id: str, status: str):
        """Update iteration status"""
        try:
            iteration = await self.get_design_iteration(iteration_id)
            if iteration:
                iteration.status = status
                await self.add_design_iteration(iteration)
                logger.info(f"Iteration {iteration_id} status updated to {status}")
        except Exception as e:
            logger.error(f"Failed to update iteration status: {e}")
    
    async def add_feedback(self, iteration_id: str, feedback: Dict[str, Any]):
        """Add feedback to a design iteration"""
        try:
            iteration = await self.get_design_iteration(iteration_id)
            if iteration:
                iteration.feedback.append({
                    **feedback,
                    "timestamp": datetime.now().isoformat()
                })
                await self.add_design_iteration(iteration)
                logger.info(f"Feedback added to iteration {iteration_id}")
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
    
    async def set_agent_state(self, agent_id: str, state: Dict[str, Any]):
        """Set agent state"""
        try:
            serialized_state = json.dumps(state, default=str)
            await self._redis.hset(
                f"agent:{agent_id}",
                mapping={"state": serialized_state}
            )
        except Exception as e:
            logger.error(f"Failed to set agent state: {e}")
    
    async def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get agent state"""
        try:
            state = await self._redis.hget(f"agent:{agent_id}", "state")
            if state:
                return json.loads(state)
            return {}
        except Exception as e:
            logger.error(f"Failed to get agent state: {e}")
            return {}
    
    async def publish_event(self, channel: str, message: Dict[str, Any]):
        """Publish event to a channel"""
        try:
            serialized_message = json.dumps(message, default=str)
            await self._redis.publish(channel, serialized_message)
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
    
    async def subscribe_to_events(self, channels: List[str]):
        """Subscribe to event channels"""
        try:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(*channels)
            return pubsub
        except Exception as e:
            logger.error(f"Failed to subscribe to events: {e}")
            return None
