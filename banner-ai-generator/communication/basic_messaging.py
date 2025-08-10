from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass
from datetime import datetime
import traceback

@dataclass
class Message:
    """Basic message structure for agent communication"""
    sender: str
    receiver: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = datetime.now()
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class BasicMessaging:
    """Basic direct messaging system for agent communication"""
    
    def __init__(self):
        self._handlers: Dict[str, Dict[str, Callable]] = {}
        self._message_history: List[Message] = []
    
    def register_handler(self, agent_name: str, message_type: str, handler: Callable) -> None:
        """Register message handler for agent"""
        if agent_name not in self._handlers:
            self._handlers[agent_name] = {}
        self._handlers[agent_name][message_type] = handler
    
    def send_message(self, message: Message) -> bool:
        """Send message to target agent"""
        try:
            self._message_history.append(message)
            
            # Find handler for receiver
            if message.receiver in self._handlers:
                if message.message_type in self._handlers[message.receiver]:
                    handler = self._handlers[message.receiver][message.message_type]
                    # Execute handler
                    result = handler(message)
                    return result if isinstance(result, bool) else True
                else:
                    print(f"No handler for message type {message.message_type} in agent {message.receiver}")
                    return False
            else:
                print(f"No handlers registered for agent {message.receiver}")
                return False
                
        except Exception as e:
            print(f"Error sending message: {e}")
            print(traceback.format_exc())
            return False
    
    def broadcast_message(self, sender: str, message_type: str, payload: Dict[str, Any]) -> List[bool]:
        """Broadcast message to all registered agents"""
        results = []
        for agent_name in self._handlers.keys():
            if agent_name != sender:
                message = Message(
                    sender=sender,
                    receiver=agent_name,
                    message_type=message_type,
                    payload=payload
                )
                results.append(self.send_message(message))
        return results
    
    def get_message_history(self, limit: int = 100) -> List[Message]:
        """Get recent message history"""
        return self._message_history[-limit:] if self._message_history else []
    
    def clear_history(self) -> None:
        """Clear message history"""
        self._message_history.clear()
