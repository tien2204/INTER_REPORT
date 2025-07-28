import json
import os
from datetime import datetime
from typing import Dict, List, Any
import logging
from ..config.settings import settings

logger = logging.getLogger(__name__)

class HistoryManager:
    """
    Quản lý lịch sử cuộc trò chuyện của Voice Agent.
    
    Attributes:
        history_dir (str): Thư mục lưu trữ lịch sử
        max_history_size (int): Số lượng cuộc trò chuyện tối đa
    """
    
    def __init__(self):
        """
        Khởi tạo HistoryManager.
        """
        self.history_dir = os.path.join(settings.DATA_DIR, "history")
        self.max_history_size = 100  # Số lượng cuộc trò chuyện tối đa
        
        # Tạo thư mục lịch sử nếu chưa tồn tại
        os.makedirs(self.history_dir, exist_ok=True)
        
    def _get_history_file(self, conversation_id: str) -> str:
        """
        Lấy đường dẫn file lịch sử cho cuộc trò chuyện.
        
        Args:
            conversation_id (str): ID của cuộc trò chuyện
            
        Returns:
            str: Đường dẫn file lịch sử
        """
        return os.path.join(self.history_dir, f"{conversation_id}.json")
    
    def save_conversation(self, conversation_id: str, conversation: List[Dict[str, Any]]) -> None:
        """
        Lưu cuộc trò chuyện vào file.
        
        Args:
            conversation_id (str): ID của cuộc trò chuyện
            conversation (List[Dict[str, Any]]): Danh sách các tin nhắn trong cuộc trò chuyện
        """
        try:
            file_path = self._get_history_file(conversation_id)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "messages": conversation
                }, f, ensure_ascii=False, indent=2)
            
            self._cleanup_old_conversations()
            logger.info(f"Lưu cuộc trò chuyện {conversation_id} thành công")
        except Exception as e:
            logger.error(f"Lỗi khi lưu cuộc trò chuyện: {e}")
            raise
    
    def load_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Tải cuộc trò chuyện từ file.
        
        Args:
            conversation_id (str): ID của cuộc trò chuyện
            
        Returns:
            List[Dict[str, Any]]: Danh sách các tin nhắn trong cuộc trò chuyện
        """
        try:
            file_path = self._get_history_file(conversation_id)
            if not os.path.exists(file_path):
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data["messages"]
        except Exception as e:
            logger.error(f"Lỗi khi tải cuộc trò chuyện: {e}")
            return []
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các cuộc trò chuyện gần đây.
        
        Args:
            limit (int): Số lượng cuộc trò chuyện cần lấy
            
        Returns:
            List[Dict[str, Any]]: Danh sách các cuộc trò chuyện gần đây
        """
        try:
            history_files = [f for f in os.listdir(self.history_dir) if f.endswith('.json')]
            history_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.history_dir, x)), reverse=True)
            
            recent_conversations = []
            for file_name in history_files[:limit]:
                with open(os.path.join(self.history_dir, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    recent_conversations.append(data)
            
            return recent_conversations
        except Exception as e:
            logger.error(f"Lỗi khi lấy cuộc trò chuyện gần đây: {e}")
            return []
    
    def _cleanup_old_conversations(self) -> None:
        """
        Xóa các cuộc trò chuyện cũ để duy trì kích thước tối đa.
        """
        try:
            history_files = [f for f in os.listdir(self.history_dir) if f.endswith('.json')]
            if len(history_files) > self.max_history_size:
                history_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.history_dir, x)))
                files_to_delete = history_files[:len(history_files) - self.max_history_size]
                
                for file_name in files_to_delete:
                    os.remove(os.path.join(self.history_dir, file_name))
        except Exception as e:
            logger.error(f"Lỗi khi làm sạch cuộc trò chuyện cũ: {e}")
