"""
RAG Adapters
============
RAG component implementations.

For backward compatibility, RAG components are currently in src/rag/.
This module re-exports them.
"""

# Re-export from original location for backward compatibility
try:
    from src.rag.context_builder import ContextBuilder
    from src.rag.hybrid_retriever import HybridRetriever
    from src.rag.retriever import DocumentRetriever
    from src.rag.router import RAGRouter
except ImportError:
    pass  # Original RAG may not exist yet

__all__ = [
    "HybridRetriever",
    "ContextBuilder",
    "DocumentRetriever",
    "RAGRouter",
]
