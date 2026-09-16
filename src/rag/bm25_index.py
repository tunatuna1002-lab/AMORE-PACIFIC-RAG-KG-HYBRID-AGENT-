"""BM25 sparse index — one search, two result shapes
==================================================

Before F3 the sparse leg was scored twice by two near-identical methods on
``DocumentRetriever``. They tokenised, scored, filtered on ``score > 0``, sorted
and truncated in exactly the same way; what differed was only how the surviving
chunks were packaged:

============================  ==================  ==========================
consumer                      score               metadata
============================  ==================  ==========================
``_bm25_search`` (dense RRF)  raw Okapi           curated, matches the vector
                                                  search's metadata keys
``search_bm25`` (public API)  divided by the      the whole chunk minus
                              corpus max (0..1)   ``content``, plus ``source``
============================  ==================  ==========================

:func:`rank` is now the only scorer; the two shapes are two formatters over its
output. The score/metadata difference is a *contract* of two different callers
and is pinned by ``tests/unit/rag/test_fusion_equivalence.py``.

``BM25_AVAILABLE`` guards stay in ``retriever.py`` so that tests patching
``src.rag.retriever.BM25_AVAILABLE`` keep working.

Moved out of ``retriever.py`` (F3 split).
"""

from __future__ import annotations

import re
from typing import Any

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    BM25Okapi = None  # type: ignore[assignment]
    BM25_AVAILABLE = False


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for Korean+English"""
    tokens = re.findall(r"[가-힣]+|[a-zA-Z]+|[0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1]


def build(chunks: list[dict[str, Any]]) -> tuple[Any | None, list[str]]:
    """Build a BM25 index from chunks.

    Returns ``(index, corpus_ids)``; ``(None, [])`` when there is nothing to
    index. Title and keywords are prepended to the body so that a chunk is
    findable by its heading.
    """
    tokenized_corpus: list[list[str]] = []
    corpus_ids: list[str] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        title = chunk.get("title", "")
        keywords = " ".join(chunk.get("keywords", []))
        tokenized_corpus.append(tokenize(f"{title} {keywords} {content}"))
        corpus_ids.append(chunk["id"])

    if not tokenized_corpus:
        return None, []
    return BM25Okapi(tokenized_corpus), corpus_ids


def rank(
    index: Any,
    corpus_ids: list[str],
    chunk_index: dict[str, dict[str, Any]],
    query: str,
    top_k: int = 10,
    doc_filter: str | None = None,
    doc_type_filter: list[str] | None = None,
) -> tuple[list[tuple[str, dict[str, Any], float]], float]:
    """Score, filter, sort and truncate — the single BM25 search.

    Returns ``(hits, max_score)`` where each hit is
    ``(chunk_id, chunk, raw_score)`` and ``max_score`` is the maximum over the
    *whole* corpus (not just the surviving hits), which is what the normalised
    formatter divides by.
    """
    if index is None or not corpus_ids:
        return [], 1.0

    query_tokens = tokenize(query)
    if not query_tokens:
        return [], 1.0

    scores = index.get_scores(query_tokens)
    max_score = float(max(scores)) if len(scores) > 0 and max(scores) > 0 else 1.0

    hits: list[tuple[str, dict[str, Any], float]] = []
    for chunk_id, score in zip(corpus_ids, scores, strict=False):
        if score <= 0:
            continue
        chunk = chunk_index.get(chunk_id)
        if chunk is None:
            continue
        if doc_filter and chunk.get("doc_id") != doc_filter:
            continue
        if doc_type_filter and chunk.get("doc_type") not in doc_type_filter:
            continue
        hits.append((chunk_id, chunk, float(score)))

    hits.sort(key=lambda hit: hit[2], reverse=True)
    return hits[:top_k], max_score


def as_detailed_results(
    hits: list[tuple[str, dict[str, Any], float]],
) -> list[dict[str, Any]]:
    """Raw scores + the curated metadata dict the vector search also emits."""
    return [
        {
            "id": chunk_id,
            "content": chunk["content"],
            "metadata": {
                "doc_id": chunk["doc_id"],
                "doc_type": chunk.get("doc_type", "metric_guide"),
                "title": chunk.get("title", ""),
                "description": chunk.get("description", ""),
                "keywords": chunk.get("keywords", []),
                "content_type": chunk.get("content_type", "text"),
                "chunk_id": chunk_id,
                "source_filename": chunk.get("source_filename", ""),
                "target_brand": chunk.get("target_brand"),
                "brands_covered": chunk.get("brands_covered", []),
            },
            "score": score,
        }
        for chunk_id, chunk, score in hits
    ]


def as_normalized_results(
    hits: list[tuple[str, dict[str, Any], float]], max_score: float
) -> list[dict[str, Any]]:
    """Scores normalised to 0-1, the whole chunk as metadata, ``source='bm25'``."""
    return [
        {
            "id": chunk_id,
            "content": chunk.get("content", ""),
            "score": score / max_score,
            "metadata": {k: v for k, v in chunk.items() if k not in ("content",)},
            "source": "bm25",
        }
        for chunk_id, chunk, score in hits
    ]
