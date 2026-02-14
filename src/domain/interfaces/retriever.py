"""
Retriever Protocol
==================
RAG Retriever에 대한 추상 인터페이스

구현체:
- HybridRetriever (src/adapters/rag/hybrid_retriever.py)
- DocumentRetriever (src/adapters/rag/retriever.py)
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RetrieverProtocol(Protocol):
    """
    Retriever Protocol

    질문에 대한 관련 컨텍스트를 검색합니다.
    KG, RAG, 또는 둘의 조합을 통해 검색할 수 있습니다.

    Methods:
        initialize: 리트리버 초기화
        retrieve: 컨텍스트 검색
        search: 문서 검색
    """

    async def initialize(self) -> None:
        """
        리트리버를 초기화합니다.

        문서 로드, 인덱스 생성 등을 수행합니다.
        """
        ...

    async def retrieve(
        self, query: str, current_metrics: dict[str, Any] | None = None, top_k: int = 5
    ) -> Any:
        """
        질문에 대한 컨텍스트를 검색합니다.

        Args:
            query: 사용자 질문
            current_metrics: 현재 메트릭 데이터 (선택)
            top_k: 반환할 최대 결과 수

        Returns:
            HybridContext 또는 검색 결과
        """
        ...

    async def search(
        self, query: str, top_k: int = 5, doc_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """
        문서를 검색합니다.

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            doc_filter: 문서 필터 (선택)

        Returns:
            검색된 문서 목록 [{"content": str, "metadata": {...}, "score": float}]
        """
        ...


@runtime_checkable
class DocumentRetrieverProtocol(Protocol):
    """
    문서 검색 Protocol

    RAG 문서 저장소에서 관련 문서를 검색합니다.

    Methods:
        initialize: 문서 로드 및 인덱스 생성
        search: 키워드/벡터 검색
        get_relevant_context: 토큰 제한 내 컨텍스트 반환
    """

    async def initialize(self) -> None:
        """문서를 로드하고 인덱스를 생성합니다."""
        ...

    async def search(
        self, query: str, top_k: int = 5, doc_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """
        문서를 검색합니다.

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            doc_filter: 문서 ID 필터 (선택)

        Returns:
            검색된 문서 목록
        """
        ...

    def get_relevant_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        토큰 제한 내에서 관련 컨텍스트를 반환합니다.

        Args:
            query: 검색 쿼리
            max_tokens: 최대 토큰 수

        Returns:
            결합된 컨텍스트 문자열
        """
        ...


from dataclasses import dataclass, field


@dataclass
class UnifiedRetrievalResult:
    """Unified result type for all retriever backends.

    This dataclass provides a common interface for both HybridRetriever
    (returns HybridContext) and TrueHybridRetriever (returns HybridResult).

    Attributes:
        query: Original user query
        entities: Extracted entities by type (brands, categories, indicators, products)
        ontology_facts: Facts retrieved from knowledge graph
        inferences: Reasoning inferences from ontology
        rag_chunks: RAG document chunks
        combined_context: Merged context string for LLM prompts
        confidence: Overall confidence score (0.0-1.0)
        entity_links: Linked entities (from TrueHybridRetriever)
        metadata: Additional metadata
        retriever_type: Backend type ("hybrid" | "true_hybrid")
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
    retriever_type: str = "unknown"  # "hybrid" | "true_hybrid"
