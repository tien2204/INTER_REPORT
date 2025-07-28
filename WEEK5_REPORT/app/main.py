from typing import Optional
import logging
from ..core.voice.stt import SpeechToText
from ..core.voice.tts import TextToSpeech
from ..core.text.question_rewriter import QuestionRewriter
from ..services.document.processor import DocumentProcessor
from ..config.settings import settings
import asyncio

logger = logging.getLogger(__name__)

class VoiceAssistant:
    """
    Main Voice Assistant class that orchestrates all components.
    
    Attributes:
        stt (SpeechToText): Speech-to-text processor
        tts (TextToSpeech): Text-to-speech processor
        question_rewriter (QuestionRewriter): Question rewriting processor
        doc_processor (DocumentProcessor): Document processing component
    """
    
    def __init__(self):
        """
        Initialize the Voice Assistant with all required components.
        """
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.question_rewriter = QuestionRewriter()
        self.doc_processor = DocumentProcessor()
        
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
                # Rewrite question for better context matching
                rewritten_questions = self.question_rewriter.rewrite_question(query)
                logger.info(f"Rewritten questions: {rewritten_questions}")
                
                # Process document chunks
                self.doc_processor.process_documents()
                
                # Get answer from document chunks
                chunks = self.doc_processor.get_document_chunks()
                answer = self._generate_answer(query, chunks)
                
                # Speak the answer
                self.tts.speak(answer)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                self.tts.speak("Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi.")

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
