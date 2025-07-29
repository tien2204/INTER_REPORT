from typing import Optional
import logging
from ..core.voice.stt import SpeechToText
from ..core.voice.tts import TextToSpeech
from ..core.text.question_rewriter import QuestionRewriter
from ..services.document.processor import DocumentProcessor
from ..utils.history_manager import HistoryManager
from ..config.settings import settings
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VoiceAssistant:
    """
    Main Voice Assistant class that orchestrates all components.
    
    Attributes:
        stt (SpeechToText): Speech-to-text processor
        tts (TextToSpeech): Text-to-speech processor
        question_rewriter (QuestionRewriter): Question rewriting processor
        doc_processor (DocumentProcessor): Document processing component
        history_manager (HistoryManager): History management component
        context_model (SentenceTransformer): Model for context similarity
    """
    
    def __init__(self):
        """
        Initialize the Voice Assistant with all required components.
        """
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.question_rewriter = QuestionRewriter()
        self.doc_processor = DocumentProcessor()
        self.history_manager = HistoryManager()
        self.context_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Set Vietnamese voice
        self.tts.set_voice("vi")

    async def process_voice_query(self) -> None:
        """
        Main loop for processing voice queries.
        
        This method continuously listens for wake word and processes user queries.
        """
        while True:
            # Listen for wake word
            if not self.stt.listen_for_wake_word():
                continue

            # Listen for user query
            query = self.stt.listen_for_query()
            if not query:
                self.tts.speak("Xin lỗi, tôi không thể hiểu câu hỏi của bạn.")
                continue

            logger.info(f"Received query: {query}")
            
            # Process query
            try:
                # Get context from history
                context = self._get_context_from_history(query)
                logger.info(f"Context from history: {context}")
                
                # Rewrite question with context
                rewritten_questions = self.question_rewriter.rewrite_question(query, context)
                logger.info(f"Rewritten questions: {rewritten_questions}")
                
                # Process document chunks
                self.doc_processor.process_documents()
                
                # Get answer from document chunks
                chunks = self.doc_processor.get_document_chunks()
                answer = self._generate_answer(query, chunks)
                
                # Save conversation to history
                self._save_conversation(query, answer, "conversation_id")
                
                # Speak the answer
                self.tts.speak(answer)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                self.tts.speak("Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi.")

    def _get_context_from_history(self, query: str) -> str:
        """
        Get context from conversation history.
        
        Args:
            query (str): Current user query
            
        Returns:
            str: Context from relevant previous conversations
        """
        try:
            # Get recent conversations
            recent_conversations = self.history_manager.get_recent_conversations(limit=5)
            
            # Find relevant messages
            relevant_messages = []
            for conv in recent_conversations:
                for msg in conv["messages"]:
                    if self._is_relevant(msg["content"], query):
                        relevant_messages.append(msg)
            
            # Create context from relevant messages
            context = "Lịch sử cuộc trò chuyện:\n"
            for msg in relevant_messages:
                context += f"{msg['role']}: {msg['content']}\n"
            
            return context
        except Exception as e:
            logger.error(f"Error getting context from history: {e}")
            return ""

    def _is_relevant(self, message: str, query: str) -> bool:
        """
        Check if a message is relevant to the query.
        
        Args:
            message (str): Message content
            query (str): User's query
            
        Returns:
            bool: True if relevant, False otherwise
        """
        try:
            # Get embeddings
            embeddings = self.context_model.encode([message, query])
            
            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return similarity > 0.5  # Similarity threshold
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            return False

    def _save_conversation(self, query: str, answer: str, conversation_id: str) -> None:
        """
        Save conversation to history.
        
        Args:
            query (str): User's query
            answer (str): Assistant's answer
            conversation_id (str): Conversation ID
        """
        try:
            conversation = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer}
            ]
            self.history_manager.save_conversation(conversation_id, conversation)
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")

    def _generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Generate answer based on query and document chunks.
        
        Args:
            query (str): User's query
            chunks (List[Dict[str, Any]]): Document chunks with embeddings
            
        Returns:
            str: Generated answer
        """
        # TODO: Implement answer generation logic
        return "Xin lỗi, chức năng này chưa được triển khai."

async def main():
    """
    Main entry point of the application.
    """
    assistant = VoiceAssistant()
    await assistant.process_voice_query()

if __name__ == "__main__":
    asyncio.run(main())
