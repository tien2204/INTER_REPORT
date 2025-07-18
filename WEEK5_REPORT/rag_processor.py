from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

class RAGProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )

    def process_query(self, query: str) -> str:
        """Process a user query using RAG"""
        try:
            # Initialize ChromaDB
            db = Chroma(
                persist_directory="chroma_db",
                embedding_function=self.embeddings
            )

            # Create retrieval chain
            qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 3})
            )

            # Get response
            response = qa.run(query)
            return response

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return "I encountered an error while processing your query. Please try again."
