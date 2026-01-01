"""
RAG (Retrieval-Augmented Generation) modules
Includes Hybrid Retrieval combining Ontology reasoning with traditional RAG
"""

from .router import RAGRouter
from .retriever import DocumentRetriever
from .templates import ResponseTemplates
from .hybrid_retriever import HybridRetriever, EntityExtractor, HybridContext
from .context_builder import ContextBuilder, CompactContextBuilder, ContextSection

__all__ = [
    # Legacy RAG
    "RAGRouter",
    "DocumentRetriever",
    "ResponseTemplates",
    # Hybrid RAG
    "HybridRetriever",
    "EntityExtractor",
    "HybridContext",
    "ContextBuilder",
    "CompactContextBuilder",
    "ContextSection"
]
