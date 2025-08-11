"""
Data serialization utilities for different data types
"""

import json
import pickle
import base64
from typing import Any, Dict, List, Union, Optional
from datetime import datetime
from dataclasses import asdict, is_dataclass
import logging

logger = logging.getLogger(__name__)

class DataSerializer:
    """
    Generic data serializer with support for various data types
    Handles JSON serialization with custom type support
    """
    
    @staticmethod
    def serialize(data: Any, format: str = "json") -> Union[str, bytes]:
        """Serialize data to specified format"""
        if format == "json":
            return DataSerializer._serialize_json(data)
        elif format == "pickle":
            return DataSerializer._serialize_pickle(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def deserialize(data: Union[str, bytes], format: str = "json") -> Any:
        """Deserialize data from specified format"""
        if format == "json":
            return DataSerializer._deserialize_json(data)
        elif format == "pickle":
            return DataSerializer._deserialize_pickle(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _serialize_json(data: Any) -> str:
        """JSON serialization with custom encoder"""
        return json.dumps(data, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)
    
    @staticmethod
    def _deserialize_json(data: str) -> Any:
        """JSON deserialization with custom decoder"""
        return json.loads(data, object_hook=custom_json_decoder)
    
    @staticmethod
    def _serialize_pickle(data: Any) -> bytes:
        """Pickle serialization"""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def _deserialize_pickle(data: bytes) -> Any:
        """Pickle deserialization"""
        return pickle.loads(data)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for special data types"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return {
                '__type__': 'datetime',
                'value': obj.isoformat()
            }
        elif is_dataclass(obj):
            return {
                '__type__': 'dataclass',
                'class_name': obj.__class__.__name__,
                'value': asdict(obj)
            }
        elif isinstance(obj, bytes):
            return {
                '__type__': 'bytes',
                'value': base64.b64encode(obj).decode('utf-8')
            }
        elif isinstance(obj, set):
            return {
                '__type__': 'set',
                'value': list(obj)
            }
        elif hasattr(obj, '__dict__'):
            return {
                '__type__': 'object',
                'class_name': obj.__class__.__name__,
                'value': obj.__dict__
            }
        
        return super().default(obj)

def custom_json_decoder(dct):
    """Custom JSON decoder for special data types"""
    if '__type__' in dct:
        obj_type = dct['__type__']
        value = dct['value']
        
        if obj_type == 'datetime':
            return datetime.fromisoformat(value)
        elif obj_type == 'bytes':
            return base64.b64decode(value.encode('utf-8'))
        elif obj_type == 'set':
            return set(value)
        # Note: dataclass and object reconstruction would need class imports
        # which is complex for a generic deserializer
        
    return dct

class DesignSerializer:
    """
    Specialized serializer for design data structures
    Handles blueprints, design versions, and related metadata
    """
    
    @staticmethod
    def serialize_blueprint(blueprint: Dict[str, Any]) -> str:
        """Serialize design blueprint"""
        try:
            # Validate blueprint structure
            DesignSerializer._validate_blueprint(blueprint)
            return DataSerializer.serialize(blueprint, "json")
        except Exception as e:
            logger.error(f"Error serializing blueprint: {e}")
            raise
    
    @staticmethod
    def deserialize_blueprint(data: str) -> Dict[str, Any]:
        """Deserialize design blueprint"""
        try:
            blueprint = DataSerializer.deserialize(data, "json")
            DesignSerializer._validate_blueprint(blueprint)
            return blueprint
        except Exception as e:
            logger.error(f"Error deserializing blueprint: {e}")
            raise
    
    @staticmethod
    def _validate_blueprint(blueprint: Dict[str, Any]) -> None:
        """Validate blueprint structure"""
        required_fields = ['version', 'canvas', 'elements']
        for field in required_fields:
            if field not in blueprint:
                raise ValueError(f"Missing required field in blueprint: {field}")
        
        # Validate canvas
        canvas = blueprint['canvas']
        canvas_required = ['width', 'height']
        for field in canvas_required:
            if field not in canvas:
                raise ValueError(f"Missing required canvas field: {field}")
        
        # Validate elements
        if not isinstance(blueprint['elements'], list):
            raise ValueError("Blueprint elements must be a list")
    
    @staticmethod
    def serialize_design_version(version_data: Dict[str, Any]) -> str:
        """Serialize design version data"""
        return DataSerializer.serialize(version_data, "json")
    
    @staticmethod
    def deserialize_design_version(data: str) -> Dict[str, Any]:
        """Deserialize design version data"""
        return DataSerializer.deserialize(data, "json")

class FeedbackSerializer:
    """
    Specialized serializer for feedback data
    Handles user feedback, design critique, and improvement suggestions
    """
    
    @staticmethod
    def serialize_feedback(feedback: Dict[str, Any]) -> str:
        """Serialize feedback data"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in feedback:
                feedback['timestamp'] = datetime.now()
            
            # Validate feedback structure
            FeedbackSerializer._validate_feedback(feedback)
            return DataSerializer.serialize(feedback, "json")
        except Exception as e:
            logger.error(f"Error serializing feedback: {e}")
            raise
    
    @staticmethod
    def deserialize_feedback(data: str) -> Dict[str, Any]:
        """Deserialize feedback data"""
        try:
            feedback = DataSerializer.deserialize(data, "json")
            FeedbackSerializer._validate_feedback(feedback)
            return feedback
        except Exception as e:
            logger.error(f"Error deserializing feedback: {e}")
            raise
    
    @staticmethod
    def _validate_feedback(feedback: Dict[str, Any]) -> None:
        """Validate feedback structure"""
        required_fields = ['type', 'content']
        for field in required_fields:
            if field not in feedback:
                raise ValueError(f"Missing required field in feedback: {field}")
        
        # Validate feedback type
        valid_types = ['user', 'agent', 'system', 'automated']
        if feedback['type'] not in valid_types:
            raise ValueError(f"Invalid feedback type: {feedback['type']}")
    
    @staticmethod
    def serialize_feedback_batch(feedback_list: List[Dict[str, Any]]) -> str:
        """Serialize multiple feedback items"""
        serialized_feedback = []
        for feedback in feedback_list:
            try:
                serialized_feedback.append(DataSerializer.deserialize(
                    FeedbackSerializer.serialize_feedback(feedback), "json"
                ))
            except Exception as e:
                logger.error(f"Error serializing feedback item: {e}")
                continue
        
        return DataSerializer.serialize(serialized_feedback, "json")
    
    @staticmethod
    def deserialize_feedback_batch(data: str) -> List[Dict[str, Any]]:
        """Deserialize multiple feedback items"""
        try:
            feedback_list = DataSerializer.deserialize(data, "json")
            validated_feedback = []
            
            for feedback in feedback_list:
                try:
                    FeedbackSerializer._validate_feedback(feedback)
                    validated_feedback.append(feedback)
                except Exception as e:
                    logger.error(f"Error validating feedback item: {e}")
                    continue
            
            return validated_feedback
        except Exception as e:
            logger.error(f"Error deserializing feedback batch: {e}")
            raise
