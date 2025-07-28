from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any
from scipy.spatial.distance import cosine
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)

class DocumentReranker:
    """
    A class for reranking documents based on relevance to a query using specialized reranking model.
    
    Attributes:
        encoder (SentenceTransformer): Model for initial embedding
        reranker (CrossEncoder): Model specialized for reranking
    """
    
    def __init__(self):
        """
        Initialize the DocumentReranker with specialized reranking models.
        """
        # Use MiniLM for initial embedding
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        
        # Use specialized reranking model
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        
        logger.info("Using specialized reranking models:")
        logger.info("- Embedding model: all-MiniLM-L6-v2")
        logger.info("- Reranking model: cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for texts using MiniLM.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[np.ndarray]: Embeddings
        """
        try:
            embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

    def rerank_documents(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query using specialized reranking model.
        
        Args:
            query (str): User's query
            documents (List[str]): List of document texts
            
        Returns:
            List[Dict[str, Any]]: List of documents with reranking scores
        """
        try:
            # First do initial filtering with embeddings
            query_embedding = self._get_embeddings([query])[0]
            doc_embeddings = self._get_embeddings(documents)
            
            # Calculate cosine similarities for initial filtering
            similarities = []
            for i, doc_emb in enumerate(doc_embeddings):
                similarity = 1 - cosine(query_embedding, doc_emb)
                similarities.append({
                    "document": documents[i],
                    "similarity": float(similarity)
                })
            
            # Sort and take top N for reranking
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            top_n = min(20, len(similarities))  # Take top 20 for reranking
            top_documents = [doc["document"] for doc in similarities[:top_n]]
            
            # Use cross-encoder for reranking
            rerank_scores = []
            for doc in top_documents:
                score = self.reranker.predict([(query, doc)])[0]
                rerank_scores.append({
                    "document": doc,
                    "score": float(score)
                })
            
            # Sort by reranking score
            rerank_scores.sort(key=lambda x: x["score"], reverse=True)
            
            return rerank_scores
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            raise

    def get_top_k_documents(self, query: str, documents: List[str], k: int = 5) -> List[str]:
        """
        Get top k most relevant documents using MiniLM reranking.
        
        Args:
            query (str): User's query
            documents (List[str]): List of document texts
            k (int): Number of documents to return (default: 5)
            
        Returns:
            List[str]: List of top k most relevant documents
        """
        try:
            ranked_docs = self.rerank_documents(query, documents)
            return [doc["document"] for doc in ranked_docs[:k]]
        except Exception as e:
            logger.error(f"Error getting top k documents: {e}")
            raise
