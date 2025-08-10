from typing import Dict, Any, Optional, List
from threading import Lock
import json
import pickle
from pathlib import Path

class BasicMemoryManager:
    """Basic in-memory storage for agent communication"""
    
    def __init__(self):
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        
    def create_session(self, session_id: str) -> None:
        """Create new session memory space"""
        with self._lock:
            if session_id not in self._memory:
                self._memory[session_id] = {}
    
    def store(self, session_id: str, key: Union[str, MemoryKey], data: Any) -> bool:
        """Store data in session memory"""
        try:
            with self._lock:
                if session_id not in self._memory:
                    self._memory[session_id] = {}
                
                key_str = key.value if isinstance(key, MemoryKey) else key
                self._memory[session_id][key_str] = data
                return True
        except Exception as e:
            print(f"Error storing data: {e}")
            return False
    
    def retrieve(self, session_id: str, key: Union[str, MemoryKey]) -> Optional[Any]:
        """Retrieve data from session memory"""
        try:
            with self._lock:
                if session_id not in self._memory:
                    return None
                
                key_str = key.value if isinstance(key, MemoryKey) else key
                return self._memory[session_id].get(key_str)
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None
    
    def update(self, session_id: str, key: Union[str, MemoryKey], data: Any) -> bool:
        """Update existing data in session memory"""
        return self.store(session_id, key, data)  # Same as store for basic implementation
    
    def delete(self, session_id: str, key: Union[str, MemoryKey]) -> bool:
        """Delete data from session memory"""
        try:
            with self._lock:
                if session_id not in self._memory:
                    return False
                
                key_str = key.value if isinstance(key, MemoryKey) else key
                if key_str in self._memory[session_id]:
                    del self._memory[session_id][key_str]
                    return True
                return False
        except Exception as e:
            print(f"Error deleting data: {e}")
            return False
    
    def list_keys(self, session_id: str) -> List[str]:
        """List all keys in session memory"""
        with self._lock:
            if session_id not in self._memory:
                return []
            return list(self._memory[session_id].keys())
    
    def clear_session(self, session_id: str) -> bool:
        """Clear all data for a session"""
        try:
            with self._lock:
                if session_id in self._memory:
                    del self._memory[session_id]
                return True
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all active sessions"""
        with self._lock:
            return list(self._memory.keys())
    
    def export_session(self, session_id: str, filepath: str) -> bool:
        """Export session data to file (JSON format)"""
        try:
            data = self.retrieve_all(session_id)
            if data is None:
                return False
            
            # Convert dataclasses to dict for JSON serialization
            serializable_data = self._make_serializable(data)
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            print(f"Error exporting session: {e}")
            return False
    
    def import_session(self, session_id: str, filepath: str) -> bool:
        """Import session data from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with self._lock:
                self._memory[session_id] = data
            return True
        except Exception as e:
            print(f"Error importing session: {e}")
            return False
    
    def retrieve_all(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve all data for a session"""
        with self._lock:
            return self._memory.get(session_id, {}).copy()
    
    def _make_serializable(self, data: Any) -> Any:
        """Convert complex objects to JSON-serializable format"""
        if hasattr(data, '__dict__'):
            # Handle dataclass objects
            result = {}
            for key, value in data.__dict__.items():
                result[key] = self._make_serializable(value)
            return result
        elif isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, (datetime, Enum)):
            return str(data)
        elif isinstance(data, bytes):
            return f"<bytes:{len(data)}>"
        else:
            return data
