import os
import time
import json
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from speech_processor import SpeechProcessor
from rag_processor import RAGProcessor
from tts_processor import TTSProcessor

# Load environment variables
load_dotenv()

# Initialize components
document_processor = DocumentProcessor(
    documents_path="documents",
    database_path="chroma_db"
)

speech_processor = SpeechProcessor()
tts_processor = TTSProcessor()
rag_processor = RAGProcessor()

def main():
    try:
        print("Processing documents...")
        document_processor.process_documents()
        print("Documents processed successfully!")

        print("Voice Assistant is ready! Say 'Hey Assistant' to start.")

        while True:
            # Listen for wake word
            wake_word = speech_processor.listen_for_wake_word()
            if wake_word:
                print("Wake word detected!")
                tts_processor.speak("I'm here. How can I help you?")

                # Listen for query
                query = speech_processor.listen_for_query()
                if query:
                    print(f"Query received: {query}")
                    tts_processor.speak("I'm on it!")

                    # Process query
                    response = rag_processor.process_query(query)
                    print(f"Response: {response}")

                    # Speak response
                    tts_processor.speak(response)

            elif wake_word == "stop":
                tts_processor.speak("Goodbye!")
                break

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")
        tts_processor.speak("I encountered an error. Please try again.")

if __name__ == "__main__":
    main()
