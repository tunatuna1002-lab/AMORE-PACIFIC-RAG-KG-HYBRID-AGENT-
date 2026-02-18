"""
RAG (Retrieval-Augmented Generation) Package
=============================================

Search pipeline (composition, not inheritance):

    Query
     ├─ QueryRewriter        – 한국어 질의 정규화 + LLM 재작성
     ├─ RAGRouter             – 질의 유형 분류 (keyword scoring)
     ├─ EntityExtractor       – 브랜드/카테고리/지표 추출 (config/entities.json)
     │
     ├─ DocumentRetriever     – ChromaDB 벡터 검색 (순수 RAG, Layer 3-B)
     ├─ HybridRetriever       – DocumentRetriever + KG 통합 오케스트레이터
     ├─ OWLRetrievalStrategy  – EntityLinker + OWLReasoner + ConfidenceFusion
     │
     ├─ CrossEncoderReranker  – 교차 인코더 재순위
     ├─ ConfidenceFusion      – 다중 소스 신뢰도 융합
     ├─ ContextBuilder        – LLM 프롬프트 조립 (토큰 예산 관리)
     └─ ResponseTemplates     – 응답 포맷팅 (guardrails, insight, metrics)

Note: EntityLinker (entity_linker.py) is the canonical entity extractor.
EntityExtractor (hybrid_retriever) and RAGRouter.extract_entities both delegate
to EntityLinker.extract_entities() for unified extraction.
"""

# --- Stage 1: Query preprocessing ---
from .chunker import SemanticChunker
from .confidence_fusion import (
    ConfidenceFusion,
    FusedResult,
    FusionStrategy,
    InferenceResult,
    LinkedEntity,
    ScoreNormalizationMethod,
    SearchResult,
    SourceScore,
)

# --- Stage 5: Context assembly & response ---
from .context_builder import (
    CompactContextBuilder,
    ContextBuilder,
    ContextPriority,
    ContextSection,
    OutputFormat,
)
from .entity_linker import EntityLinker

# --- Stage 2: Entity extraction ---
from .hybrid_retriever import EntityExtractor, HybridContext, HybridRetriever
from .query_rewriter import QueryRewriter, RewriteResult, create_rewrite_result_no_change

# --- KG extraction (RAG → KG feedback) ---
from .rag_kg_extractor import RAGKGExtractor

# --- Stage 4: Reranking & fusion ---
from .reranker import CrossEncoderReranker
from .retrieval_strategy import (
    IntentRetrievalConfig,
    OWLRetrievalStrategy,
    RetrievalStrategy,
    get_intent_retrieval_config,
)

# --- Stage 3: Retrieval ---
from .retriever import DocumentRetriever
from .router import RAGRouter
from .templates import ResponseTemplates

__all__ = [
    # Stage 1: Query preprocessing
    "QueryRewriter",
    "RewriteResult",
    "create_rewrite_result_no_change",
    "RAGRouter",
    # Stage 2: Entity extraction
    "EntityExtractor",
    "HybridContext",
    "EntityLinker",
    # Stage 3: Retrieval
    "DocumentRetriever",
    "HybridRetriever",
    "RetrievalStrategy",
    "OWLRetrievalStrategy",
    "IntentRetrievalConfig",
    "get_intent_retrieval_config",
    "SemanticChunker",
    # Stage 4: Reranking & fusion
    "CrossEncoderReranker",
    "ConfidenceFusion",
    "FusedResult",
    "FusionStrategy",
    "ScoreNormalizationMethod",
    "SearchResult",
    "InferenceResult",
    "LinkedEntity",
    "SourceScore",
    # Stage 5: Context assembly & response
    "ContextBuilder",
    "CompactContextBuilder",
    "ContextSection",
    "ContextPriority",
    "OutputFormat",
    "ResponseTemplates",
    # KG extraction (RAG → KG feedback)
    "RAGKGExtractor",
]
