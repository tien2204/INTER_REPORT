import networkx as nx
import numpy as np
from typing import List, Dict, Any
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import torch
from transformers import AutoTokenizer, AutoModel
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class GraphRAG:
    """
    Graph-based Retrieval-Augmented Generation system.
    
    Attributes:
        embeddings (OpenAIEmbeddings): Embedding model for text
        graph (nx.Graph): NetworkX graph for document relationships
        tokenizer (AutoTokenizer): BERT tokenizer
        model (AutoModel): BERT model
        device (torch.device): CUDA or CPU device
    """
    
    def __init__(self):
        """
        Initialize the GraphRAG system.
        """
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        self.graph = nx.Graph()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _create_graph_from_documents(self, documents: List[str]) -> None:
        """
        Create graph structure from documents.
        
        Args:
            documents (List[str]): List of document texts
        """
        try:
            # Create nodes from documents
            for i, doc in enumerate(documents):
                self.graph.add_node(i, text=doc)

            # Create edges based on document similarity
            embeddings = self.embeddings.embed_documents(documents)
            n = len(documents)
            
            for i in range(n):
                for j in range(i + 1, n):
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    if similarity > 0.3:  # Similarity threshold
                        self.graph.add_edge(i, j, weight=similarity)
                        logger.debug(f"Added edge between nodes {i} and {j} with weight {similarity}")
        except Exception as e:
            logger.error(f"Error creating graph: {e}")
            raise

    def _get_node_embeddings(self, node_texts: List[str]) -> np.ndarray:
        """
        Get embeddings for graph nodes.
        
        Args:
            node_texts (List[str]): List of node texts
            
        Returns:
            np.ndarray: Node embeddings
        """
        try:
            inputs = self.tokenizer(
                node_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings
        except Exception as e:
            logger.error(f"Error getting node embeddings: {e}")
            raise

    def query(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Process query using GraphRAG.
        
        Args:
            query (str): User's query
            documents (List[str]): List of document texts
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents with similarity scores and context
        """
        try:
            # Create graph from documents
            self._create_graph_from_documents(documents)

            # Get embeddings for query and nodes
            query_embedding = self.embeddings.embed_query(query)
            node_texts = [self.graph.nodes[node]["text"] for node in self.graph.nodes]
            node_embeddings = self._get_node_embeddings(node_texts)

            # Find related nodes
            similarities = np.dot(node_embeddings, query_embedding) / (
                np.linalg.norm(node_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            # Get top 5 related nodes
            top_indices = np.argsort(similarities)[-5:]
            
            # Use graph to find related nodes
            related_nodes = set()
            for idx in top_indices:
                node = list(self.graph.nodes)[idx]
                related_nodes.add(node)
                
                # Add neighboring nodes
                neighbors = list(self.graph.neighbors(node))
                for neighbor in neighbors:
                    if similarities[list(self.graph.nodes).index(neighbor)] > 0.1:
                        related_nodes.add(neighbor)

            # Get information about related nodes
            results = []
            for node in related_nodes:
                results.append({
                    "text": self.graph.nodes[node]["text"],
                    "similarity": similarities[list(self.graph.nodes).index(node)],
                    "context": self._get_context(node)
                })

            # Sort by similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            return results

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def _get_context(self, node: int) -> str:
        """
        Get context for a node by combining its text with related nodes' text.
        
        Args:
            node (int): Node index
            
        Returns:
            str: Combined context text
        """
        try:
            context = self.graph.nodes[node]["text"]
            neighbors = list(self.graph.neighbors(node))
            
            if neighbors:
                neighbor_texts = [self.graph.nodes[n]["text"] for n in neighbors]
                context += "\n\nRelated information:\n" + "\n\n".join(neighbor_texts)
            
            return context
        except Exception as e:
            logger.error(f"Error getting context for node {node}: {e}")
            return self.graph.nodes[node]["text"]
