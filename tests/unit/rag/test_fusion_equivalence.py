"""Equivalence pins for the duplicated RAG algorithms (F3 convergence guard).

The refactor plan (docs/plans/folder-structure-refactor-plan-2026-09-01.md, §F3)
records three duplications inside ``src/rag``:

* **3 RRF implementations** — ``DocumentRetriever._fuse_query_rankings``,
  ``DocumentRetriever._rrf_merge`` and ``DocumentRetriever.reciprocal_rank_fusion``
  (plus a fourth, document-level one in ``ConfidenceFusion.fuse_documents_rrf``).
* **2 BM25 searches** — ``DocumentRetriever._bm25_search`` (raw scores) and
  ``DocumentRetriever.search_bm25`` (max-normalised scores).
* **3 KG-fact/document → prompt renderers** — ``HybridRetriever._combine_contexts``
  (500-char truncation), ``ContextBuilder`` sections (400-char truncation) and
  ``OWLRetrievalStrategy._build_combined_context`` (500-char truncation).

These tests pin the **current** behaviour of every copy on identical input,
including the points where the copies *disagree*. They are written before any
code is moved, so that the convergence commit which collapses the copies onto a
single implementation is provably behaviour-preserving.

Unlike the rest of the new F3 tests, this module addresses each duplicate copy
directly: an equivalence proof is only meaningful if it names the copies it
compares. Everything reachable from a public entry point is additionally covered
by ``tests/characterization/test_hybrid_retriever_char.py``.
"""

from __future__ import annotations

import pytest

from src.rag.confidence_fusion import ConfidenceFusion
from src.rag.context_builder import ContextBuilder
from src.rag.hybrid_retriever import HybridContext, HybridRetriever
from src.rag.retrieval_strategy import OWLRetrievalStrategy
from src.rag.retriever import DocumentRetriever

try:
    import rank_bm25  # noqa: F401

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Two ranked lists whose ids and contents are in 1:1 correspondence, so that the
# id-keyed copies and the content-hash-keyed copy have to agree.
DENSE: list[dict] = [
    {"id": "a", "content": "alpha"},
    {"id": "b", "content": "bravo"},
    {"id": "c", "content": "charlie"},
]
SPARSE: list[dict] = [
    {"id": "c", "content": "charlie"},
    {"id": "a", "content": "alpha"},
    {"id": "d", "content": "delta"},
]

# RRF(k=60) scores for the pair above:
#   a = 1/61 + 1/62 = 0.0325222...   c = 1/63 + 1/61 = 0.0322662...
#   b = 1/62        = 0.0161290...   d = 1/63        = 0.0158730...
EXPECTED_ORDER = ["a", "c", "b", "d"]


@pytest.fixture
def retriever() -> DocumentRetriever:
    return DocumentRetriever(docs_path="/tmp/fusion-equivalence")


def _ids(results: list[dict]) -> list[str]:
    return [r.get("id") for r in results]


# ===========================================================================
# 1. RRF — where the three copies AGREE
# ===========================================================================


class TestRRFCopiesAgree:
    """On id/content-aligned input all three copies produce the same order."""

    def test_rrf_merge_order(self, retriever: DocumentRetriever) -> None:
        assert _ids(retriever._rrf_merge(DENSE, SPARSE, k=60)) == EXPECTED_ORDER

    def test_fuse_query_rankings_order(self, retriever: DocumentRetriever) -> None:
        assert _ids(retriever._fuse_query_rankings([DENSE, SPARSE], k=60)) == EXPECTED_ORDER

    def test_reciprocal_rank_fusion_order(self) -> None:
        fused = DocumentRetriever.reciprocal_rank_fusion(DENSE, SPARSE, k=60, top_k=10)
        assert _ids(fused) == EXPECTED_ORDER

    def test_three_copies_produce_identical_order(self, retriever: DocumentRetriever) -> None:
        """The convergence guard: one input, three implementations, one order."""
        a = _ids(retriever._rrf_merge(DENSE, SPARSE, k=60))
        b = _ids(retriever._fuse_query_rankings([DENSE, SPARSE], k=60))
        c = _ids(DocumentRetriever.reciprocal_rank_fusion(DENSE, SPARSE, k=60, top_k=10))
        assert a == b == c == EXPECTED_ORDER

    def test_rrf_scores_are_the_textbook_formula(self) -> None:
        fused = DocumentRetriever.reciprocal_rank_fusion(DENSE, SPARSE, k=60, top_k=10)
        by_id = {d["id"]: d["rrf_score"] for d in fused}
        assert by_id["a"] == pytest.approx(1 / 61 + 1 / 62)
        assert by_id["c"] == pytest.approx(1 / 63 + 1 / 61)
        assert by_id["b"] == pytest.approx(1 / 62)
        assert by_id["d"] == pytest.approx(1 / 63)

    @pytest.mark.parametrize("k", [1, 10, 60, 1000])
    def test_copies_agree_for_every_k(self, retriever: DocumentRetriever, k: int) -> None:
        a = _ids(retriever._rrf_merge(DENSE, SPARSE, k=k))
        b = _ids(retriever._fuse_query_rankings([DENSE, SPARSE], k=k))
        c = _ids(DocumentRetriever.reciprocal_rank_fusion(DENSE, SPARSE, k=k, top_k=10))
        assert a == b == c

    def test_all_copies_return_empty_for_empty_input(self, retriever: DocumentRetriever) -> None:
        assert retriever._rrf_merge([], []) == []
        assert retriever._fuse_query_rankings([]) == []
        assert DocumentRetriever.reciprocal_rank_fusion() == []
        assert ConfidenceFusion().fuse_documents_rrf({}) == []


# ===========================================================================
# 2. RRF — where the three copies DIFFER (pinned, not fixed, by this commit)
# ===========================================================================


class TestRRFCopiesDiffer:
    """Documented divergences. Each is a knob of the converged implementation."""

    def test_only_reciprocal_rank_fusion_truncates(self, retriever: DocumentRetriever) -> None:
        """``top_k`` exists on one copy only; the other two return everything."""
        assert len(retriever._rrf_merge(DENSE, SPARSE)) == 4
        assert len(retriever._fuse_query_rankings([DENSE, SPARSE])) == 4
        assert len(DocumentRetriever.reciprocal_rank_fusion(DENSE, SPARSE, top_k=2)) == 2

    def test_only_reciprocal_rank_fusion_annotates_results(
        self, retriever: DocumentRetriever
    ) -> None:
        """One copy copies each doc and stamps ``rrf_score``/``source``;
        the other two hand back the caller's own dict objects untouched."""
        merged = retriever._rrf_merge(DENSE, SPARSE)
        assert merged[0] is DENSE[0]
        assert "rrf_score" not in merged[0] and "source" not in merged[0]

        fused_q = retriever._fuse_query_rankings([DENSE, SPARSE])
        assert fused_q[0] is DENSE[0]
        assert "rrf_score" not in fused_q[0]

        annotated = DocumentRetriever.reciprocal_rank_fusion(DENSE, SPARSE)
        assert annotated[0] is not DENSE[0]
        assert annotated[0]["source"] == "hybrid_rrf"
        assert annotated[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
        # ... and the originals are left clean
        assert "rrf_score" not in DENSE[0]

    def test_single_ranking_short_circuits_only_in_fuse_query_rankings(
        self, retriever: DocumentRetriever
    ) -> None:
        """``_fuse_query_rankings`` returns the *same list object* for one input;
        the other copies always run the full fusion."""
        only = retriever._fuse_query_rankings([DENSE])
        assert only is DENSE

        assert retriever._rrf_merge(DENSE, []) is not DENSE
        assert _ids(retriever._rrf_merge(DENSE, [])) == ["a", "b", "c"]
        assert _ids(DocumentRetriever.reciprocal_rank_fusion(DENSE)) == ["a", "b", "c"]

    def test_dedup_key_differs_same_id_different_content(
        self, retriever: DocumentRetriever
    ) -> None:
        """Same id + different content: the id-keyed copies merge, the
        content-hash-keyed copy keeps both."""
        left = [{"id": "x", "content": "first"}]
        right = [{"id": "x", "content": "second"}]

        assert len(retriever._rrf_merge(left, right)) == 1
        assert len(retriever._fuse_query_rankings([left, right])) == 1
        assert len(DocumentRetriever.reciprocal_rank_fusion(left, right)) == 2

    def test_dedup_key_differs_same_content_different_id(
        self, retriever: DocumentRetriever
    ) -> None:
        """Same content + different id: the mirror image of the case above."""
        left = [{"id": "x", "content": "same"}]
        right = [{"id": "y", "content": "same"}]

        assert len(retriever._rrf_merge(left, right)) == 2
        assert len(retriever._fuse_query_rankings([left, right])) == 2
        assert len(DocumentRetriever.reciprocal_rank_fusion(left, right)) == 1

    def test_missing_id_fallback_differs_across_the_three_copies(
        self, retriever: DocumentRetriever
    ) -> None:
        """PINS CURRENT BEHAVIOUR (two latent bugs).

        With id-less documents:
        * ``_fuse_query_rankings`` keys everything on ``""`` → every id-less doc
          in every ranking collapses into a single entry.
        * ``_rrf_merge`` keys on ``str(rank)`` → the *positionally* matching docs
          of the two lists falsely merge (dense[0] and sparse[0] are both "0").
        * ``reciprocal_rank_fusion`` keys on the content hash → distinct
          contents stay distinct, which is the only correct answer here.
        """
        left = [{"content": "p"}, {"content": "q"}]
        right = [{"content": "r"}, {"content": "s"}]

        assert len(retriever._fuse_query_rankings([left, right])) == 1
        assert len(retriever._rrf_merge(left, right)) == 2
        assert len(DocumentRetriever.reciprocal_rank_fusion(left, right)) == 4

    def test_duplicate_id_within_one_list_keeps_the_first_occurrence(
        self, retriever: DocumentRetriever
    ) -> None:
        """A ranked list that repeats an id keeps the *first* entry's payload.

        CHANGED (F3): ``_rrf_merge`` used to keep the **last** occurrence for
        its first (dense) argument and the first for its second (sparse) one —
        an accident of ``result_map[rid] = result`` vs ``if rid not in
        result_map``. The converged implementation uses ``setdefault``
        throughout, matching what the other two copies already did. This is
        unreachable from production (both indexes emit unique chunk ids per
        list); it is normalised rather than reproduced so the single
        implementation has one rule instead of three.
        """
        dupes = [{"id": "x", "content": "first"}, {"id": "x", "content": "second"}]

        assert retriever._rrf_merge(dupes, [])[0]["content"] == "first"
        assert retriever._fuse_query_rankings([dupes, []])[0]["content"] == "first"
        assert DocumentRetriever.reciprocal_rank_fusion(dupes)[0]["content"] == "first"

    def test_confidence_fusion_rrf_is_a_fourth_copy_with_its_own_contract(self) -> None:
        """``ConfidenceFusion.fuse_documents_rrf`` takes *named* lists, rounds the
        score to 6 dp and records which sources contributed."""
        fused = ConfidenceFusion().fuse_documents_rrf(
            {"dense": DENSE, "bm25": SPARSE}, k=60, top_n=10
        )
        assert _ids(fused) == EXPECTED_ORDER
        assert fused[0]["rrf_sources"] == ["dense", "bm25"]
        assert fused[0]["rrf_score"] == pytest.approx(round(1 / 61 + 1 / 62, 6))
        assert fused[2]["rrf_sources"] == ["dense"]
        # It keys on content too, so the ids survive only because they align.
        assert len(ConfidenceFusion().fuse_documents_rrf({"a": DENSE, "b": SPARSE}, top_n=2)) == 2


# ===========================================================================
# 3. BM25 — the two searches
# ===========================================================================


def _bm25_chunks() -> list[dict]:
    """Five chunks — a smaller corpus gives rank_bm25 negative IDFs."""
    return [
        {
            "id": "c1",
            "doc_id": "d1",
            "doc_type": "metric_guide",
            "title": "LANEIGE Lip",
            "content_type": "text",
            "content": "LANEIGE Lip Sleeping Mask Berry flavor product",
            "keywords": ["laneige", "lip"],
            "description": "lip care",
            "source_filename": "t.md",
            "target_brand": "laneige",
            "brands_covered": [],
        },
        {
            "id": "c2",
            "doc_id": "d1",
            "doc_type": "metric_guide",
            "title": "COSRX",
            "content_type": "text",
            "content": "COSRX Snail 96 Mucin Power Essence review skincare",
            "keywords": ["cosrx"],
            "description": "skincare",
            "source_filename": "t.md",
            "target_brand": None,
            "brands_covered": [],
        },
        {
            "id": "c3",
            "doc_id": "d2",
            "doc_type": "playbook",
            "title": "LANEIGE Water",
            "content_type": "text",
            "content": "LANEIGE Water Sleeping Mask overnight treatment care",
            "keywords": ["laneige", "water"],
            "description": "sleeping mask",
            "source_filename": "t.md",
            "target_brand": "laneige",
            "brands_covered": [],
        },
        {
            "id": "c4",
            "doc_id": "d3",
            "doc_type": "playbook",
            "title": "Tirtir",
            "content_type": "text",
            "content": "TIRTIR Mask Fit Red Cushion foundation makeup product",
            "keywords": ["tirtir"],
            "description": "makeup",
            "source_filename": "t.md",
            "target_brand": None,
            "brands_covered": [],
        },
        {
            "id": "c5",
            "doc_id": "d3",
            "doc_type": "intelligence",
            "title": "Anua",
            "content_type": "text",
            "content": "ANUA Heartleaf Pore Control Cleansing Oil gentle toner",
            "keywords": ["anua"],
            "description": "cleansing",
            "source_filename": "t.md",
            "target_brand": None,
            "brands_covered": [],
        },
    ]


@pytest.fixture
def bm25_retriever() -> DocumentRetriever:
    r = DocumentRetriever(docs_path="/tmp/fusion-equivalence")
    r.chunks = _bm25_chunks()
    r._chunk_index = {c["id"]: c for c in r.chunks}
    return r


@pytest.mark.skipif(not BM25_AVAILABLE, reason="rank_bm25 not installed")
class TestBM25CopiesAgree:
    """Both searches score, filter and order identically."""

    async def test_same_ranking(self, bm25_retriever: DocumentRetriever) -> None:
        sync_ids = _ids(bm25_retriever.search_bm25("LANEIGE sleeping mask", top_k=5))
        async_ids = _ids(await bm25_retriever._bm25_search("LANEIGE sleeping mask", top_k=5))
        assert sync_ids == async_ids
        assert sync_ids  # non-empty, otherwise the comparison is vacuous

    async def test_same_zero_score_filtering(self, bm25_retriever: DocumentRetriever) -> None:
        """Both drop every chunk whose BM25 score is not strictly positive."""
        sync_res = bm25_retriever.search_bm25("LANEIGE", top_k=10)
        async_res = await bm25_retriever._bm25_search("LANEIGE", top_k=10)
        assert _ids(sync_res) == _ids(async_res) == ["c1", "c3"]

    async def test_same_top_k_truncation(self, bm25_retriever: DocumentRetriever) -> None:
        assert len(bm25_retriever.search_bm25("mask product", top_k=2)) == 2
        assert len(await bm25_retriever._bm25_search("mask product", top_k=2)) == 2

    async def test_both_return_empty_for_untokenizable_query(
        self, bm25_retriever: DocumentRetriever
    ) -> None:
        assert bm25_retriever.search_bm25("!", top_k=5) == []
        assert await bm25_retriever._bm25_search("!", top_k=5) == []


@pytest.mark.skipif(not BM25_AVAILABLE, reason="rank_bm25 not installed")
class TestBM25CopiesDiffer:
    """Same ranking, different result *shape*. Pinned, not fixed."""

    async def test_score_normalisation_differs(self, bm25_retriever: DocumentRetriever) -> None:
        """``search_bm25`` divides by the corpus max (top hit == 1.0);
        ``_bm25_search`` returns the raw Okapi score (> 1.0 here)."""
        sync_res = bm25_retriever.search_bm25("LANEIGE sleeping mask", top_k=5)
        async_res = await bm25_retriever._bm25_search("LANEIGE sleeping mask", top_k=5)

        assert sync_res[0]["score"] == pytest.approx(1.0)
        assert all(0.0 < r["score"] <= 1.0 for r in sync_res)
        assert async_res[0]["score"] > 1.0
        # Same ranking though: the sync scores are the async ones over their max.
        top = async_res[0]["score"]
        assert [r["score"] for r in sync_res] == pytest.approx(
            [r["score"] / top for r in async_res]
        )

    async def test_metadata_shape_differs(self, bm25_retriever: DocumentRetriever) -> None:
        """``_bm25_search`` builds the curated metadata dict the vector search
        also emits; ``search_bm25`` dumps the whole chunk minus ``content``."""
        sync_meta = bm25_retriever.search_bm25("LANEIGE", top_k=1)[0]["metadata"]
        async_meta = (await bm25_retriever._bm25_search("LANEIGE", top_k=1))[0]["metadata"]

        assert set(async_meta) == {
            "doc_id",
            "doc_type",
            "title",
            "description",
            "keywords",
            "content_type",
            "chunk_id",
            "source_filename",
            "target_brand",
            "brands_covered",
        }
        assert "chunk_id" in async_meta and "chunk_id" not in sync_meta
        assert "id" in sync_meta  # the raw chunk's own id field leaks into metadata

    async def test_source_marker_only_on_the_sync_copy(
        self, bm25_retriever: DocumentRetriever
    ) -> None:
        assert bm25_retriever.search_bm25("LANEIGE", top_k=1)[0]["source"] == "bm25"
        assert "source" not in (await bm25_retriever._bm25_search("LANEIGE", top_k=1))[0]

    async def test_doc_filters_only_on_the_async_copy(
        self, bm25_retriever: DocumentRetriever
    ) -> None:
        """``_bm25_search`` honours doc_filter/doc_type_filter; ``search_bm25``
        has no filter parameters at all, which is why ``HybridRetriever``'s
        doc-type filter silently does not apply to its BM25 leg."""
        filtered = await bm25_retriever._bm25_search(
            "LANEIGE", top_k=10, doc_type_filter=["playbook"]
        )
        assert _ids(filtered) == ["c3"]

        by_doc = await bm25_retriever._bm25_search("LANEIGE", top_k=10, doc_filter="d1")
        assert _ids(by_doc) == ["c1"]

        assert _ids(bm25_retriever.search_bm25("LANEIGE", top_k=10)) == ["c1", "c3"]


# ===========================================================================
# 4. KG-fact / document → prompt renderers
# ===========================================================================

RENDER_FACTS: list[dict] = [
    {"type": "brand_info", "entity": "laneige", "data": {"sos": 0.123, "avg_rank": 8.5}},
    {"type": "brand_products", "entity": "laneige", "data": {"product_count": 3}},
    {
        "type": "competitors",
        "entity": "laneige",
        "data": [{"brand": "COSRX"}, {"brand": "ANUA"}],
    },
    {
        "type": "category_brands",
        "entity": "lip_care",
        "data": {"top_brands": [{"brand": "LANEIGE"}, {"brand": "TIRTIR"}]},
    },
]

LONG = "가" * 600
RENDER_CHUNKS: list[dict] = [{"content": LONG, "metadata": {"title": "긴 문서", "doc_id": "d1"}}]


class TestRendererCopiesAgree:
    """The facts every renderer draws from the same KG payload."""

    def test_both_kg_renderers_report_the_same_numbers(self) -> None:
        legacy = HybridRetriever._combine_contexts.__get__(_stub_retriever(), HybridRetriever)(
            HybridContext(query="q", ontology_facts=RENDER_FACTS)
        )
        builder = ContextBuilder()._build_facts_section(RENDER_FACTS).content

        for text in (legacy, builder):
            assert "12.3%" in text  # sos 0.123 rendered as a percentage
            assert "8.5" in text  # avg_rank
            assert "COSRX" in text and "ANUA" in text
            assert "LANEIGE" in text and "TIRTIR" in text

    def test_all_three_renderers_cap_the_fact_list_or_chunk_list(self) -> None:
        """5 facts / 3 chunks are the shared caps."""
        many_facts = [
            {"type": "brand_products", "entity": f"b{i}", "data": {"product_count": i}}
            for i in range(9)
        ]
        legacy = HybridRetriever._combine_contexts.__get__(_stub_retriever(), HybridRetriever)(
            HybridContext(query="q", ontology_facts=many_facts)
        )
        builder = ContextBuilder()._build_facts_section(many_facts).content
        assert legacy.count("제품 수") == 5
        assert builder.count("제품") == 5

        many_chunks = [{"content": f"chunk-{i}", "metadata": {}} for i in range(9)]
        legacy_chunks = HybridRetriever._combine_contexts.__get__(
            _stub_retriever(), HybridRetriever
        )(HybridContext(query="q", rag_chunks=many_chunks))
        builder_chunks = ContextBuilder()._build_rag_section(many_chunks).content
        owl_chunks = OWLRetrievalStrategy._build_combined_context([], {}, many_chunks)
        assert legacy_chunks.count("chunk-") == 3
        assert builder_chunks.count("chunk-") == 3
        assert owl_chunks.count("chunk-") == 3


class TestRendererCopiesDiffer:
    """PINS CURRENT BEHAVIOUR: three renderers, three output formats."""

    def test_truncation_limit_differs(self) -> None:
        legacy = HybridRetriever._combine_contexts.__get__(_stub_retriever(), HybridRetriever)(
            HybridContext(query="q", rag_chunks=RENDER_CHUNKS)
        )
        builder = ContextBuilder()._build_rag_section(RENDER_CHUNKS).content
        owl = OWLRetrievalStrategy._build_combined_context([], {}, RENDER_CHUNKS)

        assert "가" * 500 + "..." in legacy
        assert "가" * 501 not in legacy
        assert "가" * 400 + "..." in builder
        assert "가" * 401 not in builder
        assert "가" * 500 + "..." in owl
        assert "가" * 501 not in owl

    def test_only_context_builder_emits_citation_markers(self) -> None:
        legacy = HybridRetriever._combine_contexts.__get__(_stub_retriever(), HybridRetriever)(
            HybridContext(query="q", rag_chunks=RENDER_CHUNKS)
        )
        builder = ContextBuilder()._build_rag_section(RENDER_CHUNKS).content

        assert "### 긴 문서 [1]" in builder
        assert "### 긴 문서" in legacy and "[1]" not in legacy

    def test_kg_fact_line_format_differs(self) -> None:
        """Legacy renders one bullet per fact; ContextBuilder renders a bold
        entity header followed by indented attribute lines."""
        legacy = HybridRetriever._combine_contexts.__get__(_stub_retriever(), HybridRetriever)(
            HybridContext(query="q", ontology_facts=RENDER_FACTS)
        )
        builder = ContextBuilder()._build_facts_section(RENDER_FACTS).content

        assert "- **laneige** SoS: 12.3%" in legacy
        assert "  - 평균 순위: 8.5" in legacy
        assert "- **laneige** 주요 경쟁사: COSRX, ANUA" in legacy

        assert "**laneige**:" in builder
        assert "  - 점유율: 12.3%" in builder
        assert "**laneige** 경쟁사: COSRX, ANUA" in builder

    def test_owl_renderer_does_not_render_kg_facts_at_all(self) -> None:
        """The OWL renderer is not a third copy of the KG-fact renderer: it
        renders entity links and inference dicts, and shares only the document
        section with the other two."""
        owl = OWLRetrievalStrategy._build_combined_context(
            [],
            {
                "inferences": [
                    {
                        "type": "market_position",
                        "brand": "LANEIGE",
                        "position": "리더",
                        "sos": 0.123,
                    }
                ]
            },
            [],
        )
        assert "## 온톨로지 추론 결과" in owl
        assert "- LANEIGE의 시장 포지션: 리더 (SoS: 12.30%)" in owl
        assert "Knowledge Graph" not in owl

    def test_section_headings_differ(self) -> None:
        legacy = HybridRetriever._combine_contexts.__get__(_stub_retriever(), HybridRetriever)(
            HybridContext(query="q", ontology_facts=RENDER_FACTS, rag_chunks=RENDER_CHUNKS)
        )
        owl = OWLRetrievalStrategy._build_combined_context([], {}, RENDER_CHUNKS)

        assert "## 관련 정보 (Knowledge Graph)" in legacy
        assert "## 참고 가이드라인 (RAG)" in legacy
        # ContextBuilder keeps the heading out of the body: it is section metadata.
        assert ContextBuilder()._build_rag_section(RENDER_CHUNKS).title == "참고 가이드라인 (RAG)"
        assert ContextBuilder()._build_facts_section(RENDER_FACTS).title == (
            "관련 정보 (Knowledge Graph)"
        )
        assert "## 관련 문서" in owl


def _stub_retriever() -> object:
    """Bare object good enough for ``_combine_contexts``, which touches no state.

    Building a real ``HybridRetriever`` here would pull in the KnowledgeGraph and
    the rule registry for a pure string-formatting assertion.
    """

    class _Stub:
        pass

    return _Stub()
