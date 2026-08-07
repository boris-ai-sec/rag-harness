"""Service adapters for the local-first RAG Harness."""

from rag_harness.services.embeddings import OllamaEmbeddingService
from rag_harness.services.vector_store import QdrantVectorStore

__all__ = ["OllamaEmbeddingService", "QdrantVectorStore"]
