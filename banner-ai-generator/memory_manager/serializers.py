"""
Serializers Module

Handles serialization and deserialization of data structures
for the multi-agent system.
"""

import json
import pickle
import base64
from typing import Any, Dict, List, Union
from datetime import datetime
from dataclasses import is_dataclass, asdict
from enum import Enum
import numpy as np
from PIL import Image
import io
from structlog import get_logger

logger = get_logger(__name__)


class SerializationFormat(Enum):
    """Supported serialization formats"""
    JSON = "json"
    PICKLE = "pickle"
    BASE64 = "base64"


class AgentDataSerializer:
    """
    Serializer for agent data structures
    """
    
    @staticmethod
    def _json_serializer(obj) -> Any:
        """Custom JSON serializer for complex objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif is_dataclass(obj):
            return asdict(obj)
        elif isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": base64.b64encode(obj.tobytes()).decode('utf-8')
            }
        elif isinstance(obj, Image.Image):
            buffer = io.BytesIO()
            obj.save(buffer, format='PNG')
            return {
                "__pil_image__": True,
                "format": "PNG",
                "data": base64.b64encode(buffer.getvalue()).decode('utf-8')
            }
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    @staticmethod
    def _json_deserializer(obj) -> Any:
        """Custom JSON deserializer for complex objects"""
        if isinstance(obj, dict):
            if "__numpy_array__" in obj:
                data = base64.b64decode(obj["data"])
                return np.frombuffer(data, dtype=obj["dtype"]).reshape(obj["shape"])
            elif "__pil_image__" in obj:
                data = base64.b64decode(obj["data"])
                return Image.open(io.BytesIO(data))
        return obj
    
    @classmethod
    def serialize(cls, data: Any, format: SerializationFormat = SerializationFormat.JSON) -> str:
        """
        Serialize data to string format
        
        Args:
            data: Data to serialize
            format: Serialization format
            
        Returns:
            Serialized string
        """
        try:
            if format == SerializationFormat.JSON:
                return json.dumps(data, default=cls._json_serializer, ensure_ascii=False)
            elif format == SerializationFormat.PICKLE:
                pickled_data = pickle.dumps(data)
                return base64.b64encode(pickled_data).decode('utf-8')
            elif format == SerializationFormat.BASE64:
                if isinstance(data, str):
                    return base64.b64encode(data.encode('utf-8')).decode('utf-8')
                else:
                    json_data = json.dumps(data, default=cls._json_serializer)
                    return base64.b64encode(json_data.encode('utf-8')).decode('utf-8')
            else:
                raise ValueError(f"Unsupported serialization format: {format}")
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise
    
    @classmethod
    def deserialize(cls, data: str, format: SerializationFormat = SerializationFormat.JSON) -> Any:
        """
        Deserialize string data back to original format
        
        Args:
            data: Serialized string data
            format: Serialization format used
            
        Returns:
            Deserialized data
        """
        try:
            if format == SerializationFormat.JSON:
                return json.loads(data, object_hook=cls._json_deserializer)
            elif format == SerializationFormat.PICKLE:
                pickled_data = base64.b64decode(data.encode('utf-8'))
                return pickle.loads(pickled_data)
            elif format == SerializationFormat.BASE64:
                decoded_data = base64.b64decode(data.encode('utf-8')).decode('utf-8')
                try:
                    return json.loads(decoded_data, object_hook=cls._json_deserializer)
                except json.JSONDecodeError:
                    return decoded_data
            else:
                raise ValueError(f"Unsupported serialization format: {format}")
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise
    
    @classmethod
    def serialize_campaign_brief(cls, brief: Dict[str, Any]) -> str:
        """Serialize campaign brief"""
        return cls.serialize(brief, SerializationFormat.JSON)
    
    @classmethod
    def deserialize_campaign_brief(cls, brief_data: str) -> Dict[str, Any]:
        """Deserialize campaign brief"""
        return cls.deserialize(brief_data, SerializationFormat.JSON)
    
    @classmethod
    def serialize_design_blueprint(cls, blueprint: Dict[str, Any]) -> str:
        """Serialize design blueprint"""
        return cls.serialize(blueprint, SerializationFormat.JSON)
    
    @classmethod
    def deserialize_design_blueprint(cls, blueprint_data: str) -> Dict[str, Any]:
        """Deserialize design blueprint"""
        return cls.deserialize(blueprint_data, SerializationFormat.JSON)
    
    @classmethod
    def serialize_agent_state(cls, state: Dict[str, Any]) -> str:
        """Serialize agent state"""
        return cls.serialize(state, SerializationFormat.JSON)
    
    @classmethod
    def deserialize_agent_state(cls, state_data: str) -> Dict[str, Any]:
        """Deserialize agent state"""
        return cls.deserialize(state_data, SerializationFormat.JSON)
    
    @classmethod
    def serialize_feedback(cls, feedback: List[Dict[str, Any]]) -> str:
        """Serialize feedback data"""
        return cls.serialize(feedback, SerializationFormat.JSON)
    
    @classmethod
    def deserialize_feedback(cls, feedback_data: str) -> List[Dict[str, Any]]:
        """Deserialize feedback data"""
        return cls.deserialize(feedback_data, SerializationFormat.JSON)
    
    @classmethod
    def serialize_image_data(cls, image: Image.Image) -> str:
        """Serialize PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    @classmethod
    def deserialize_image_data(cls, image_data: str) -> Image.Image:
        """Deserialize base64 string to PIL Image"""
        data = base64.b64decode(image_data)
        return Image.open(io.BytesIO(data))
    
    @classmethod
    def serialize_numpy_array(cls, array: np.ndarray) -> str:
        """Serialize numpy array to base64 string"""
        return base64.b64encode(array.tobytes()).decode('utf-8')
    
    @classmethod
    def deserialize_numpy_array(cls, array_data: str, dtype: str, shape: tuple) -> np.ndarray:
        """Deserialize base64 string to numpy array"""
        data = base64.b64decode(array_data)
        return np.frombuffer(data, dtype=dtype).reshape(shape)


class MessageSerializer:
    """
    Serializer for inter-agent messages
    """
    
    @classmethod
    def serialize_message(cls, message: Dict[str, Any]) -> str:
        """Serialize inter-agent message"""
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
        
        return AgentDataSerializer.serialize(message, SerializationFormat.JSON)
    
    @classmethod
    def deserialize_message(cls, message_data: str) -> Dict[str, Any]:
        """Deserialize inter-agent message"""
        message = AgentDataSerializer.deserialize(message_data, SerializationFormat.JSON)
        
        # Convert timestamp back to datetime object
        if "timestamp" in message:
            message["timestamp"] = datetime.fromisoformat(message["timestamp"])
        
        return message


class ConfigSerializer:
    """
    Serializer for configuration data
    """
    
    @classmethod
    def serialize_config(cls, config: Dict[str, Any]) -> str:
        """Serialize configuration data"""
        return AgentDataSerializer.serialize(config, SerializationFormat.JSON)
    
    @classmethod
    def deserialize_config(cls, config_data: str) -> Dict[str, Any]:
        """Deserialize configuration data"""
        return AgentDataSerializer.deserialize(config_data, SerializationFormat.JSON)
    
    @classmethod
    def serialize_agent_config(cls, agent_id: str, config: Dict[str, Any]) -> str:
        """Serialize agent-specific configuration"""
        agent_config = {
            "agent_id": agent_id,
            "config": config,
            "timestamp": datetime.utcnow().isoformat()
        }
        return cls.serialize_config(agent_config)
    
    @classmethod
    def deserialize_agent_config(cls, config_data: str) -> tuple:
        """Deserialize agent-specific configuration"""
        data = cls.deserialize_config(config_data)
        return data["agent_id"], data["config"]


class LogSerializer:
    """
    Serializer for logging data
    """
    
    @classmethod
    def serialize_log_entry(cls, level: str, message: str, 
                          context: Dict[str, Any] = None) -> str:
        """Serialize log entry"""
        log_entry = {
            "level": level,
            "message": message,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        return AgentDataSerializer.serialize(log_entry, SerializationFormat.JSON)
    
    @classmethod
    def deserialize_log_entry(cls, log_data: str) -> Dict[str, Any]:
        """Deserialize log entry"""
        log_entry = AgentDataSerializer.deserialize(log_data, SerializationFormat.JSON)
        
        # Convert timestamp back to datetime object
        if "timestamp" in log_entry:
            log_entry["timestamp"] = datetime.fromisoformat(log_entry["timestamp"])
        
        return log_entry


def serialize_for_storage(data: Any, format: SerializationFormat = SerializationFormat.JSON) -> str:
    """Convenience function for serializing data for storage"""
    return AgentDataSerializer.serialize(data, format)


def deserialize_from_storage(data: str, format: SerializationFormat = SerializationFormat.JSON) -> Any:
    """Convenience function for deserializing data from storage"""
    return AgentDataSerializer.deserialize(data, format)
