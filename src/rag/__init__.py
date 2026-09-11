"""
RAG (Retrieval-Augmented Generation) Package
=============================================

Search pipeline (composition, not inheritance):

    Query
     ├─ QueryRewriter        – 한국어 질의 정규화 + LLM 재작성
     ├─ RAGRouter             – 질의 유형 분류 (keyword scoring)
     ├─ EntityExtractor       – 브랜드/카테고리/지표 추출 (config/entities.json)
     ├─ DocumentRetriever     – ChromaDB 벡터 검색
     ├─ HybridRetriever       – DocumentRetriever + KG 통합 오케스트레이터
     ├─ OWLRetrievalStrategy  – EntityLinker + OWLReasoner + ConfidenceFusion
     ├─ CrossEncoderReranker  – 교차 인코더 재순위
     ├─ ConfidenceFusion      – 다중 소스 신뢰도 융합
     ├─ ContextBuilder        – LLM 프롬프트 조립 (토큰 예산 관리)
     └─ ResponseTemplates     – 응답 포맷팅

패키지 최상위 이름은 지연 로딩된다 (chromadb 등 무거운 의존성은 접근 시에만 import).
"""

import importlib

_LAZY: dict[str, str] = {
    "QueryRewriter": ".query_rewriter",
    "RewriteResult": ".query_rewriter",
    "create_rewrite_result_no_change": ".query_rewriter",
    "RAGRouter": ".router",
    "EntityExtractor": ".hybrid_retriever",
    "HybridContext": ".hybrid_retriever",
    "HybridRetriever": ".hybrid_retriever",
    "EntityLinker": ".entity_linker",
    "DocumentRetriever": ".retriever",
    "RetrievalStrategy": ".retrieval_strategy",
    "OWLRetrievalStrategy": ".retrieval_strategy",
    "IntentRetrievalConfig": ".retrieval_strategy",
    "get_intent_retrieval_config": ".retrieval_strategy",
    "SemanticChunker": ".chunker",
    "CrossEncoderReranker": ".reranker",
    "ConfidenceFusion": ".confidence_fusion",
    "FusedResult": ".confidence_fusion",
    "FusionStrategy": ".confidence_fusion",
    "ScoreNormalizationMethod": ".confidence_fusion",
    "SearchResult": ".confidence_fusion",
    "FusionInferenceResult": ".confidence_fusion",
    "FusedEntity": ".confidence_fusion",
    "SourceScore": ".confidence_fusion",
    "ContextBuilder": ".context_builder",
    "CompactContextBuilder": ".context_builder",
    "ContextSection": ".context_builder",
    "ContextPriority": ".context_builder",
    "OutputFormat": ".context_builder",
    "ResponseTemplates": ".templates",
    "RAGKGExtractor": ".rag_kg_extractor",
}
_OPTIONAL = frozenset(())

__all__ = list(_LAZY)


def __getattr__(name: str):
    """지연 로딩: 하위 모듈은 실제로 접근할 때만 import한다 (무거운 의존성 로딩 방지)."""
    if name in _LAZY:
        try:
            module = importlib.import_module(_LAZY[name], __name__)
        except ImportError:
            if name in _OPTIONAL:
                globals()[name] = None
                return None
            raise
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
