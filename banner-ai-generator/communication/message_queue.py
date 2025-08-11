"""
Message queue system for asynchronous inter-agent communication
"""

import asyncio
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import PriorityQueue, Empty
import logging

from .protocol import AgentMessage, AgentResponse

logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0

@dataclass
class Message:
    """Message wrapper with priority and metadata"""
    content: AgentMessage
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    delivery_attempts: int = 0
    max_attempts: int = 3
    
    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.delivery_attempts < self.max_attempts

class MessageQueue:
    """
    Thread-safe message queue for inter-agent communication
    Supports priority-based message delivery and retries
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queue: PriorityQueue = PriorityQueue(maxsize=max_size)
        self._pending_responses: Dict[str, AgentResponse] = {}
        self._subscribers: Dict[str, List[Callable[[Message], None]]] = {}
        self._dead_letter_queue: List[Message] = []
        self._stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_failed': 0,
            'messages_expired': 0
        }
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
    
    def start(self) -> None:
        """Start message queue processor"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._processor_thread = threading.Thread(
                target=self._process_messages,
                daemon=True,
                name="MessageQueueProcessor"
            )
            self._processor_thread.start()
            logger.info("Message queue processor started")
    
    def stop(self) -> None:
        """Stop message queue processor"""
        with self._lock:
            self._running = False
            if self._processor_thread and self._processor_thread.is_alive():
                self._processor_thread.join(timeout=5.0)
            logger.info("Message queue processor stopped")
    
    def send_message(self, 
                    message: AgentMessage, 
                    priority: MessagePriority = MessagePriority.NORMAL,
                    expires_in: Optional[int] = None) -> bool:
        """Send message to queue"""
        try:
            expires_at = None
            if expires_in:
                expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            msg = Message(
                content=message,
                priority=priority,
                expires_at=expires_at
            )
            
            self._queue.put(msg, block=False)
            self._stats['messages_sent'] += 1
            logger.debug(f"Message queued: {message.message_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue message: {e}")
            return False
    
    def subscribe(self, agent_id: str, handler: Callable[[Message], None]) -> None:
        """Subscribe agent to receive messages"""
        with self._lock:
            if agent_id not in self._subscribers:
                self._subscribers[agent_id] = []
            self._subscribers[agent_id].append(handler)
            logger.debug(f"Agent {agent_id} subscribed to messages")
    
    def unsubscribe(self, agent_id: str, handler: Optional[Callable[[Message], None]] = None) -> None:
        """Unsubscribe agent from messages"""
        with self._lock:
            if agent_id in self._subscribers:
                if handler:
                    if handler in self._subscribers[agent_id]:
                        self._subscribers[agent_id].remove(handler)
                else:
                    del self._subscribers[agent_id]
                logger.debug(f"Agent {agent_id} unsubscribed from messages")
    
    def send_response(self, response: AgentResponse) -> None:
        """Send response for a previous request"""
        with self._lock:
            self._pending_responses[response.request_id] = response
            logger.debug(f"Response stored for request: {response.request_id}")
    
    def wait_for_response(self, request_id: str, timeout: int = 30) -> Optional[AgentResponse]:
        """Wait for response to a request"""
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < timeout:
            with self._lock:
                if request_id in self._pending_responses:
                    response = self._pending_responses.pop(request_id)
                    return response
            
            threading.Event().wait(0.1)  # Small delay
        
        logger.warning(f"Timeout waiting for response to request: {request_id}")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message queue statistics"""
        with self._lock:
            return {
                **self._stats,
                'queue_size': self._queue.qsize(),
                'pending_responses': len(self._pending_responses),
                'dead_letter_count': len(self._dead_letter_queue),
                'active_subscribers': len(self._subscribers)
            }
    
    def get_dead_letter_messages(self) -> List[Message]:
        """Get messages that failed delivery"""
        with self._lock:
            return self._dead_letter_queue.copy()
    
    def clear_dead_letter_queue(self) -> None:
        """Clear dead letter queue"""
        with self._lock:
            self._dead_letter_queue.clear()
            logger.info("Dead letter queue cleared")
    
    def _process_messages(self) -> None:
        """Process messages in queue"""
        while self._running:
            try:
                # Get message with timeout
                message = self._queue.get(timeout=1.0)
                
                # Check if message expired
                if message.is_expired():
                    self._stats['messages_expired'] += 1
                    logger.warning(f"Message expired: {message.content.message_id}")
                    continue
                
                # Deliver message
                delivered = self._deliver_message(message)
                
                if delivered:
                    self._stats['messages_received'] += 1
                else:
                    message.delivery_attempts += 1
                    
                    if message.can_retry():
                        # Requeue for retry
                        self._queue.put(message)
                        logger.debug(f"Message requeued for retry: {message.content.message_id}")
                    else:
                        # Move to dead letter queue
                        self._dead_letter_queue.append(message)
                        self._stats['messages_failed'] += 1
                        logger.error(f"Message failed delivery: {message.content.message_id}")
                
                self._queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _deliver_message(self, message: Message) -> bool:
        """Deliver message to subscribers"""
        target_agent = message.content.to_agent
        
        with self._lock:
            handlers = self._subscribers.get(target_agent, [])
        
        if not handlers:
            logger.warning(f"No handlers for agent: {target_agent}")
            return False
        
        delivered = False
        for handler in handlers:
            try:
                handler(message)
                delivered = True
                logger.debug(f"Message delivered to {target_agent}: {message.content.message_id}")
            except Exception as e:
                logger.error(f"Error delivering message to {target_agent}: {e}")
        
        return delivered
