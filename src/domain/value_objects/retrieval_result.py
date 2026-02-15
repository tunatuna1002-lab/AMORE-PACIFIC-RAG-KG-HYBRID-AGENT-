"""
Retrieval Result Value Objects
==============================
RAG + Ontology 하이브리드 검색 결과를 나타내는 불변 값 객체.

이 모듈은 모든 Retriever 백엔드의 표준 반환 타입을 정의합니다.
"""

from dataclasses import dataclass, field


@dataclass
class UnifiedRetrievalResult:
    """Unified result type for all retriever backends.

    This dataclass provides a common interface for both legacy HybridRetriever
    and OWL-based retrieval strategies.

    Attributes:
        query: Original user query
        entities: Extracted entities by type (brands, categories, indicators, products)
        ontology_facts: Facts retrieved from knowledge graph
        inferences: Reasoning inferences from ontology
        rag_chunks: RAG document chunks
        combined_context: Merged context string for LLM prompts
        confidence: Overall confidence score (0.0-1.0)
        entity_links: Linked entities (from OWL strategy)
        metadata: Additional metadata
        retriever_type: Backend type ("legacy" | "owl")
    """

    query: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    ontology_facts: list[dict] = field(default_factory=list)
    inferences: list[dict] = field(default_factory=list)
    rag_chunks: list[dict] = field(default_factory=list)
    combined_context: str = ""
    confidence: float = 0.0
    entity_links: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    retriever_type: str = "unknown"  # "legacy" | "owl"
