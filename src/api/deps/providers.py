"""
Providers
=========
Lazily-constructed singletons shared by the routes. Nothing here is built at
import time: the RAG router/retriever, StateManager, SheetsWriter and the market
intelligence engine are created on first use.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from src.api.deps.data import DOCS_PATH
from src.rag.router import QueryType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.core.state_manager import StateManager
    from src.rag.retriever import DocumentRetriever
    from src.rag.router import RAGRouter
    from src.tools.intelligence.market_intelligence import MarketIntelligenceEngine
    from src.tools.storage.sheets_writer import SheetsWriter


# ============= URL Helper =============


def get_base_url() -> str:
    """배포 환경에 맞는 Base URL 반환"""
    if dashboard_url := os.getenv("DASHBOARD_URL"):
        return dashboard_url.rstrip("/")

    if railway_domain := os.getenv("RAILWAY_PUBLIC_DOMAIN"):
        return f"https://{railway_domain}"

    port = os.getenv("PORT", "8001")
    return f"http://localhost:{port}"


# ============= RAG System (lazy) =============

_rag_router: RAGRouter | None = None
_doc_retriever: DocumentRetriever | None = None


def get_rag_router() -> RAGRouter:
    global _rag_router
    if _rag_router is None:
        from src.rag.router import RAGRouter

        _rag_router = RAGRouter()
    return _rag_router


def get_doc_retriever() -> DocumentRetriever:
    global _doc_retriever
    if _doc_retriever is None:
        from src.rag.retriever import DocumentRetriever

        _doc_retriever = DocumentRetriever(DOCS_PATH)
    return _doc_retriever


_DOC_NAME_MAP = {
    "strategic_indicators": "Strategic Indicators Definition",
    "metric_interpretation": "Metric Interpretation Guide",
    "indicator_combination": "Indicator Combination Playbook",
    "home_insight_rules": "Home Page Insight Rules",
}


async def get_rag_context(query: str, query_type: QueryType) -> tuple[str, list[str]]:
    """
    RAG 컨텍스트 검색

    Returns:
        (컨텍스트 문자열, 출처 목록)
    """
    doc_retriever = get_doc_retriever()
    # DocumentRetriever 초기화 (처음 호출 시)
    if not doc_retriever._initialized:
        await doc_retriever.initialize()

    # 질문 유형에 맞는 문서 검색
    target_doc = get_rag_router().get_target_document(query_type)

    # 검색 실행
    results = await doc_retriever.search(query, top_k=3, doc_filter=target_doc)

    if not results:
        return "", []

    # 컨텍스트 구성
    context_parts = []
    sources = []

    for result in results:
        metadata = result.get("metadata", {})
        content = result.get("content", "")
        title = metadata.get("title", "Unknown")
        doc_id = metadata.get("doc_id", "")

        context_parts.append(f"[{title}]\n{content}")

        # 출처 추가
        if doc_id in _DOC_NAME_MAP and _DOC_NAME_MAP[doc_id] not in sources:
            sources.append(_DOC_NAME_MAP[doc_id])

    return "\n\n---\n\n".join(context_parts), sources


# ============= State Manager =============

_state_manager: StateManager | None = None


def get_app_state_manager() -> StateManager:
    """앱 레벨 State Manager 반환"""
    global _state_manager
    if _state_manager is None:
        from src.core.state_manager import get_state_manager

        _state_manager = get_state_manager()
    return _state_manager


# ============= SheetsWriter Singleton =============

_sheets_writer: SheetsWriter | None = None


def get_sheets_writer() -> SheetsWriter:
    """SheetsWriter 싱글톤 인스턴스 반환"""
    global _sheets_writer
    if _sheets_writer is None:
        from src.tools.storage.sheets_writer import SheetsWriter

        _sheets_writer = SheetsWriter()
    return _sheets_writer


# ============= Market Intelligence Singleton =============

_market_intelligence_engine: MarketIntelligenceEngine | None = None


async def get_market_intelligence() -> MarketIntelligenceEngine:
    """Market Intelligence Engine 싱글톤 반환"""
    global _market_intelligence_engine
    if _market_intelligence_engine is None:
        from src.tools.intelligence.market_intelligence import MarketIntelligenceEngine

        _market_intelligence_engine = MarketIntelligenceEngine()
        await _market_intelligence_engine.initialize()
    return _market_intelligence_engine


def _reset_providers() -> None:
    """테스트용: 싱글톤 초기화"""
    global _rag_router, _doc_retriever, _state_manager, _sheets_writer
    global _market_intelligence_engine
    _rag_router = None
    _doc_retriever = None
    _state_manager = None
    _sheets_writer = None
    _market_intelligence_engine = None


def __getattr__(name: str) -> Any:
    # Legacy module attributes, resolved lazily so importing does not build them.
    if name == "rag_router":
        return get_rag_router()
    if name == "doc_retriever":
        return get_doc_retriever()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
