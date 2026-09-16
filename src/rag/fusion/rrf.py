"""Reciprocal Rank Fusion — the single implementation
===================================================

``RRF_score(d) = Σ_i 1 / (k + rank_i(d))``

Before F3 the codebase carried three copies of that loop inside
``DocumentRetriever`` (``_fuse_query_rankings``, ``_rrf_merge``,
``reciprocal_rank_fusion``) plus a fourth in ``ConfidenceFusion``. The scoring
and the sort were identical in all of them; what genuinely differed was
peripheral and is expressed here as parameters:

=========================  ====================  =========  =========
call site                  dedup key             truncates  annotates
=========================  ====================  =========  =========
``_fuse_query_rankings``   ``id`` (``""`` miss)  no         no
``_rrf_merge``             ``id`` (rank miss)    no         no
``reciprocal_rank_fusion`` content hash          ``top_k``  yes
=========================  ====================  =========  =========

Those three keying rules are NOT interchangeable — they disagree on documents
with a missing or duplicated id, and ``tests/unit/rag/test_fusion_equivalence.py``
pins each disagreement. They stay selectable so that the split is provably
behaviour-preserving; unifying them changes retrieval results and belongs to a
defect commit with its own failing-test-first evidence (see the F3 report).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

DEFAULT_K = 60

# A key function receives (doc, rank_within_its_list) and returns the identity
# under which the document's RRF contributions are accumulated.
KeyFn = Callable[[dict, int], str]


def key_by_id(doc: dict, rank: int) -> str:
    """``id``, with the empty string for id-less documents.

    All id-less documents therefore collapse into one entry. Used by the
    expanded-query fusion, whose inputs always carry chunk ids.
    """
    return doc.get("id", "")


def key_by_id_or_rank(doc: dict, rank: int) -> str:
    """``id``, falling back to the document's rank as a string.

    Note the fallback is per-list, so the rank-0 documents of two different
    lists share the key ``"0"``. Used by the dense+sparse merge.
    """
    return doc.get("id", str(rank))


def key_by_content_hash(doc: dict, rank: int) -> str:
    """SHA-256 of ``content``, truncated to 16 hex chars.

    Identifies documents by what they say rather than by which index produced
    them. Used by the public ``reciprocal_rank_fusion``.
    """
    return hashlib.sha256(doc.get("content", "").encode()).hexdigest()[:16]


def fuse(
    ranked_lists: Iterable[Sequence[dict]],
    *,
    k: int = DEFAULT_K,
    top_k: int | None = None,
    key: KeyFn = key_by_id,
    annotate: bool = False,
    source_label: str = "hybrid_rrf",
) -> list[dict[str, Any]]:
    """Merge ranked lists by reciprocal rank fusion.

    Args:
        ranked_lists: the ranked result lists to merge.
        k: RRF constant. Higher values flatten the rank discount.
        top_k: keep only this many results (``None`` keeps all).
        key: how two entries are recognised as the same document.
        annotate: when true each result is a *copy* carrying ``rrf_score`` and
            ``source``; when false the caller's own dict objects are returned.
        source_label: the value written to ``source`` when annotating.

    Returns:
        The merged list, highest RRF score first.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            doc_key = key(doc, rank)
            scores[doc_key] = scores.get(doc_key, 0.0) + 1.0 / (k + rank + 1)
            docs.setdefault(doc_key, doc)

    ordered = sorted(scores, key=lambda key_: scores[key_], reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]

    if not annotate:
        return [docs[key_] for key_ in ordered]

    results: list[dict[str, Any]] = []
    for key_ in ordered:
        doc = docs[key_].copy()
        doc["rrf_score"] = scores[key_]
        doc["source"] = source_label
        results.append(doc)
    return results


def fuse_named(
    ranked_lists: dict[str, list[dict]],
    *,
    k: int = DEFAULT_K,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Document-level RRF over *named* lists, recording which sources voted.

    The variant ``ConfidenceFusion.fuse_documents_rrf`` exposes: it identifies
    documents by ``content`` (or ``insight``), falls back to
    ``"<source>_<rank>"`` for empty bodies, and stamps ``rrf_score`` (rounded to
    6 dp) plus ``rrf_sources``.
    """
    if not ranked_lists:
        return []

    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    sources: dict[str, list[str]] = {}

    for source_name, ranked in ranked_lists.items():
        for rank, doc in enumerate(ranked):
            content = doc.get("content", doc.get("insight", ""))
            doc_key = str(hash(content))[:16] if content else f"{source_name}_{rank}"
            scores[doc_key] = scores.get(doc_key, 0.0) + 1.0 / (k + rank + 1)
            docs.setdefault(doc_key, doc.copy())
            sources.setdefault(doc_key, []).append(source_name)

    ordered = sorted(scores, key=scores.get, reverse=True)

    results: list[dict[str, Any]] = []
    for doc_key in ordered[:top_n]:
        doc = docs[doc_key]
        doc["rrf_score"] = round(scores[doc_key], 6)
        doc["rrf_sources"] = sources.get(doc_key, [])
        results.append(doc)
    return results
