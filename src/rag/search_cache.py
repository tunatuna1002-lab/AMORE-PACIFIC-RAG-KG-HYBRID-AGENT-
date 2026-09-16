"""Search result cache
===================

TTL cache for ``DocumentRetriever.search`` results. The cache dicts themselves
stay on ``DocumentRetriever`` as class attributes (they are shared across every
instance, and tests reach into them), so this module holds only the key
derivation and the TTL arithmetic.

Known gap (not changed by the F3 split): the cache key does not include the
embedding model name, so switching ``OPENAI_EMBEDDING_MODEL`` keeps serving
results computed with the previous model until the TTL expires.

Moved verbatim out of ``retriever.py`` (F3 split).
"""

from __future__ import annotations

import time
from typing import Any


def make_key(
    query: str,
    top_k: int,
    doc_filter: str | None,
    doc_type_filter: list[str] | None = None,
) -> str:
    """캐시 키 생성"""
    type_key = ",".join(doc_type_filter) if doc_type_filter else "all_types"
    return f"{query}:{top_k}:{doc_filter or 'all'}:{type_key}"


def is_valid(timestamps: dict[str, float], cache_key: str, ttl: int) -> bool:
    """캐시 유효성 확인 (TTL 체크)"""
    if cache_key not in timestamps:
        return False
    return time.time() - timestamps[cache_key] < ttl


def clean_expired(cache: dict[str, Any], timestamps: dict[str, float], ttl: int) -> None:
    """만료된 캐시 항목 정리"""
    current_time = time.time()
    expired_keys = [key for key, ts in timestamps.items() if current_time - ts >= ttl]
    for key in expired_keys:
        cache.pop(key, None)
        timestamps.pop(key, None)
