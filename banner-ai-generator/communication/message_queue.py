"""
Message Queue Module

Handles message queuing and reliable delivery between agents
using Redis as the backend message broker.
"""

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta
import redis.asyncio as redis
from structlog import get_logger

from .protocol import Message, MessageType  # Fixed import: AgentMessage -> Message
from memory_manager.serializers import MessageSerializer

logger = get_logger(__name__)


class MessageQueue:
    """
    Message queue implementation using Redis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        self.config = config
        self.redis_url = config.get("redis_url", "redis://172.26.33.210:6379")
        self._redis: Optional[redis.Redis] = None
        self._subscribers: Dict[str, Callable] = {}
        self._running = False
        self._consumer_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Initialize Redis connection"""
        try:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self._redis.ping()
            logger.info("Message queue initialized")
        except Exception as e:
            logger.error(f"Failed to initialize message queue: {e}")
            raise
    
    async def stop(self):
        """Close message queue"""
        self._running = False
        
        # Cancel consumer tasks
        for task in self._consumer_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        
        if self._redis:
            await self._redis.close()
    
    async def publish(self, channel: str, message_data: Dict[str, Any]) -> bool:
        """
        Publish a message to a channel
        
        Args:
            channel: Channel name
            message_data: Message data to publish
            
        Returns:
            bool: True if successful
        """
        try:
            if not self._redis:
                await self.start()
            
            # Serialize message
            serialized_message = MessageSerializer.serialize(message_data)
            
            # Publish to Redis
            await self._redis.publish(channel, serialized_message)
            
            return True
        except Exception as e:
            logger.error(f"Failed to publish message to {channel}: {e}")
            return False
    
    async def subscribe(self, channel: str, handler: Callable) -> bool:
        """
        Subscribe to a channel
        
        Args:
            channel: Channel name
            handler: Message handler function
            
        Returns:
            bool: True if successful
        """
        try:
            self._subscribers[channel] = handler
            
            if not self._running:
                self._running = True
                task = asyncio.create_task(self._consume_messages(channel))
                self._consumer_tasks.append(task)
            
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")
            return False
    
    async def unsubscribe(self, channel: str) -> bool:
        """
        Unsubscribe from a channel
        
        Args:
            channel: Channel name
            
        Returns:
            bool: True if successful
        """
        try:
            if channel in self._subscribers:
                del self._subscribers[channel]
            
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {channel}: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get message queue status
        
        Returns:
            Dict containing status information
        """
        try:
            status = {
                "running": self._running,
                "connected": self._redis is not None,
                "subscribers": len(self._subscribers),
                "active_tasks": len(self._consumer_tasks),
                "channels": list(self._subscribers.keys())
            }
            
            if self._redis:
                # Get Redis info
                try:
                    await self._redis.ping()
                    status["redis_connected"] = True
                except:
                    status["redis_connected"] = False
            
            return status
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}
    
    async def _consume_messages(self, channel: str):
        """
        Consume messages from a Redis channel
        
        Args:
            channel: Channel name
        """
        try:
            if not self._redis:
                await self.start()
            
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(channel)
            
            while self._running:
                try:
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        # Deserialize message
                        message_data = MessageSerializer.deserialize(message['data'])
                        
                        # Call handler
                        handler = self._subscribers.get(channel)
                        if handler:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(message_data)
                            else:
                                handler(message_data)
                                
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing message on {channel}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in message consumer for {channel}: {e}")
        finally:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except:
                pass
