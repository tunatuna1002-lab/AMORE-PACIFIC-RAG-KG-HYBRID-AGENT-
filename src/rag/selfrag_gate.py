"""Self-RAG gate
==============

Decides whether a query needs retrieval at all, so that greetings, thanks and
system commands do not pay for a KG query + vector search + LLM reasoning pass.

Two gates live here because the codebase has two, at different depths:

``should_retrieve``
    The gate ``HybridRetriever`` runs. Returns a 3-tuple
    ``(should, reason, confidence)``; the confidence also tunes ``top_k``
    downstream (< 0.5 halves it).

``needs_retrieval``
    The narrower gate ``DocumentRetriever.search`` runs on its own. It is
    regex-anchored (``re.match``) against the lower-cased query and only
    answers yes/no.

The two do NOT agree on every query, and that is deliberate for now: the
``DocumentRetriever`` gate also guards direct ``search()`` callers that never
pass through ``HybridRetriever``. Their pinned behaviour lives in
``tests/characterization/test_hybrid_retriever_char.py`` and
``tests/unit/rag/test_retriever.py``.

Moved verbatim out of ``hybrid_retriever.py`` / ``retriever.py`` (F3 split).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# HybridRetriever gate
# ---------------------------------------------------------------------------

# Patterns that indicate retrieval is NOT needed.
SKIP_PATTERNS: list[str] = [
    # Greetings (no \b for Korean; Korean chars are not word-boundary friendly)
    r"^(안녕|하이|헬로)",
    r"^(hi|hello|hey)\b",
    # Thanks
    r"^(고마워|감사|thanks|thank you)",
    # System commands
    r"^(도움말|설정|help|config)",
]

# Patterns that indicate retrieval IS needed.
RETRIEVE_PATTERNS: list[str] = [
    # Brand names
    r"(?i)(laneige|cosrx|anua|tirtir|round\s*lab|innisfree|sulwhasoo)",
    # Metrics
    r"(?i)(sos|hhi|cpi|share\s*of\s*shelf|순위|rank|점유율)",
    # Analysis keywords
    r"(분석|비교|전략|경쟁|트렌드|시장|매출|성장)",
    # Question words
    r"(왜|어떻게|뭐|몇|어디|언제|무엇|how|what|why|which)",
]


def should_retrieve(
    query: str,
    skip_patterns: list[str] | None = None,
    retrieve_patterns: list[str] | None = None,
) -> tuple[bool, str, float]:
    """Self-RAG gate: determine if retrieval is needed.

    Args:
        query: user query.
        skip_patterns: override for :data:`SKIP_PATTERNS` (the retriever passes
            its own class attribute so subclasses can still re-tune the gate).
        retrieve_patterns: override for :data:`RETRIEVE_PATTERNS`.

    Returns:
        ``(should_retrieve, reason, confidence)`` where confidence is 1.0 for
        strong domain queries, 0.8 for the conservative default and 0.0 on skip.
    """
    skip = SKIP_PATTERNS if skip_patterns is None else skip_patterns
    retrieve = RETRIEVE_PATTERNS if retrieve_patterns is None else retrieve_patterns

    if not query or len(query.strip()) <= 2:
        return False, "query_too_short", 0.0

    query_stripped = query.strip()

    # Check skip patterns first
    for pattern in skip:
        if re.search(pattern, query_stripped, re.IGNORECASE):
            return False, "greeting_or_command", 0.0

    # Check retrieve patterns
    for pattern in retrieve:
        if re.search(pattern, query_stripped):
            return True, "domain_query_detected", 1.0

    # Default: retrieve (conservative)
    if len(query_stripped) > 5:
        return True, "default_retrieve", 0.8

    return False, "short_non_domain_query", 0.0


# ---------------------------------------------------------------------------
# DocumentRetriever gate
# ---------------------------------------------------------------------------

_NO_RETRIEVAL_PATTERNS: list[str] = [
    r"^(안녕|hello|hi|hey|감사|고마워|thank)",
    r"^(네|예|응|ok|okay|맞아|그래)$",
    r"^(도움|help|뭐 할 수|무엇을 할)",
]


def needs_retrieval(query: str) -> bool:
    """Self-RAG: does this document search need to run at all?"""
    query_lower = query.lower().strip()
    for pattern in _NO_RETRIEVAL_PATTERNS:
        if re.match(pattern, query_lower):
            return False
    if len(query_lower) < 3:
        return False
    return True
