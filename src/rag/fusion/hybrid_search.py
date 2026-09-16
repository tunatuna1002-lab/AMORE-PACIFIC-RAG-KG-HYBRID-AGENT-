"""Dense + BM25 hybrid search
==========================

The sparse leg ``HybridRetriever`` adds on top of ``DocumentRetriever.search``.

Known duplication (unchanged by the F3 split, deliberately): ``DocumentRetriever.search``
*already* fuses its own BM25 leg via RRF, so a query that comes through
``HybridRetriever`` is BM25-scored and RRF-fused twice, in two different score
spaces. Removing the second pass changes retrieval results and is pinned by
``tests/characterization/test_hybrid_retriever_char.py``
(``metadata["search_method"] == "hybrid_rrf"``); it needs a golden-set run, not
a refactor. See the F3 report.

Moved verbatim out of ``hybrid_retriever.py`` (F3 split).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

RRF_K = 60


def bm25_actually_available(doc_retriever: Any) -> bool:
    """BM25 sparse 검색 가용 여부 — 메서드 존재가 아니라 rank_bm25 설치 여부까지 확인"""
    if not hasattr(doc_retriever, "search_bm25"):
        return False
    try:
        from src.rag.retriever import BM25_AVAILABLE

        return bool(BM25_AVAILABLE)
    except ImportError:
        return False


async def hybrid_search(
    doc_retriever: Any,
    query: str,
    top_k: int = 5,
    doc_type_filter: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Dense + BM25 hybrid search with RRF fusion.

    Args:
        doc_retriever: the DocumentRetriever (or a fake exposing the same methods)
        query: Search query
        top_k: Number of results to return
        doc_type_filter: Optional document type filter

    Returns:
        ``(results, search_method)`` where search_method is
        ``"hybrid_rrf"`` or ``"dense_only"``.
    """
    # 1. Dense search via doc_retriever.search()
    dense_results = await doc_retriever.search(query, top_k=top_k, doc_type_filter=doc_type_filter)

    # 2. BM25 search (if available)
    bm25_results: list[dict[str, Any]] = []
    if hasattr(doc_retriever, "search_bm25"):
        try:
            bm25_results = doc_retriever.search_bm25(query, top_k=top_k)
        except Exception as e:
            logger.debug(f"BM25 search failed in _hybrid_search: {e}")

    # 3. RRF fusion
    if bm25_results:
        if hasattr(doc_retriever, "reciprocal_rank_fusion"):
            fused = doc_retriever.reciprocal_rank_fusion(
                dense_results, bm25_results, k=RRF_K, top_k=top_k
            )
            return fused, "hybrid_rrf"
        # Fallback: try confidence_fusion.fuse_documents_rrf
        try:
            from src.rag.confidence_fusion import ConfidenceFusion

            fused = ConfidenceFusion().fuse_documents_rrf(
                {"dense": dense_results, "bm25": bm25_results},
                k=RRF_K,
                top_n=top_k,
            )
            return fused, "hybrid_rrf"
        except (ImportError, Exception) as e:
            logger.debug(f"Confidence fusion RRF fallback failed: {e}")

    return dense_results, "dense_only"
