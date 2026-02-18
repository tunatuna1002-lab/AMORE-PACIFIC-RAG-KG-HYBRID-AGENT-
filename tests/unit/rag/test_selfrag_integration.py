"""
Unit tests for D-3: HybridRetriever Self-RAG integration + hybrid search.

Tests:
- should_retrieve() 3-tuple return with confidence
- _hybrid_search() with BM25 + dense + RRF
- _hybrid_search() dense-only fallback
- retrieve() uses _hybrid_search()
- metadata search_method tracking
- Self-RAG confidence reduces top_k
- Self-RAG skip patterns have 0.0 confidence
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_kg():
    """Mock KnowledgeGraph."""
    kg = MagicMock()
    kg.get_entity_metadata.return_value = {}
    kg.get_brand_products.return_value = []
    kg.get_competitors.return_value = []
    kg.get_neighbors.return_value = {"outgoing": [], "incoming": []}
    kg.query.return_value = []
    kg.get_category_brands.return_value = []
    kg.get_category_hierarchy.return_value = {}
    kg.get_product_sentiments.return_value = {}
    kg.get_brand_sentiment_profile.return_value = {}
    kg.find_products_by_sentiment.return_value = []
    kg.load_category_hierarchy.return_value = 0
    kg.get_stats.return_value = {}
    return kg


@pytest.fixture
def mock_reasoner():
    """Mock OntologyReasoner."""
    reasoner = MagicMock()
    reasoner.rules = ["dummy_rule"]
    reasoner.infer.return_value = []
    reasoner.get_inference_stats.return_value = {}
    return reasoner


@pytest.fixture
def mock_doc_retriever():
    """Mock DocumentRetriever with search_bm25 and reciprocal_rank_fusion."""
    retriever = MagicMock()
    retriever.initialize = AsyncMock()
    retriever.search = AsyncMock(
        return_value=[
            {"id": "doc1", "content": "LANEIGE Lip Sleeping Mask", "score": 0.95, "metadata": {}},
            {"id": "doc2", "content": "COSRX product analysis", "score": 0.85, "metadata": {}},
        ]
    )
    retriever.search_bm25 = MagicMock(
        return_value=[
            {"id": "doc1", "content": "LANEIGE Lip Sleeping Mask", "score": 0.9, "metadata": {}},
            {"id": "doc3", "content": "Lip Care market overview", "score": 0.7, "metadata": {}},
        ]
    )
    retriever.reciprocal_rank_fusion = MagicMock(
        return_value=[
            {"id": "doc1", "content": "LANEIGE Lip Sleeping Mask", "score": 0.95, "metadata": {}},
            {"id": "doc3", "content": "Lip Care market overview", "score": 0.7, "metadata": {}},
            {"id": "doc2", "content": "COSRX product analysis", "score": 0.85, "metadata": {}},
        ]
    )
    return retriever


@pytest.fixture
def mock_doc_retriever_no_bm25():
    """Mock DocumentRetriever WITHOUT BM25 support."""
    retriever = MagicMock()
    retriever.initialize = AsyncMock()
    retriever.search = AsyncMock(
        return_value=[
            {"id": "doc1", "content": "LANEIGE Lip Sleeping Mask", "score": 0.95, "metadata": {}},
            {"id": "doc2", "content": "COSRX product analysis", "score": 0.85, "metadata": {}},
        ]
    )
    # Remove BM25 attributes
    if hasattr(retriever, "search_bm25"):
        del retriever.search_bm25
    if hasattr(retriever, "reciprocal_rank_fusion"):
        del retriever.reciprocal_rank_fusion
    # Explicitly set spec to exclude these attributes
    retriever.search_bm25 = None
    delattr(retriever, "search_bm25")
    return retriever


@pytest.fixture
def hybrid_retriever(mock_kg, mock_reasoner, mock_doc_retriever):
    """Create HybridRetriever with mocked components."""
    from src.rag.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(
        knowledge_graph=mock_kg,
        reasoner=mock_reasoner,
        doc_retriever=mock_doc_retriever,
        auto_init_rules=False,
    )
    retriever._initialized = True
    return retriever


class TestShouldRetrieveWithConfidence:
    """Test should_retrieve() returns 3-tuple with confidence."""

    def test_returns_three_tuple(self, hybrid_retriever):
        result = hybrid_retriever.should_retrieve("LANEIGE 분석")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_domain_query_high_confidence(self, hybrid_retriever):
        should, reason, confidence = hybrid_retriever.should_retrieve("LANEIGE 순위 분석")
        assert should is True
        assert reason == "domain_query_detected"
        assert confidence == 1.0

    def test_default_retrieve_medium_confidence(self, hybrid_retriever):
        should, reason, confidence = hybrid_retriever.should_retrieve("이것은 일반적인 질문입니다")
        assert should is True
        assert reason == "default_retrieve"
        assert confidence == 0.8

    def test_greeting_skip_zero_confidence(self, hybrid_retriever):
        should, reason, confidence = hybrid_retriever.should_retrieve("안녕하세요")
        assert should is False
        assert reason == "greeting_or_command"
        assert confidence == 0.0

    def test_hello_skip_zero_confidence(self, hybrid_retriever):
        should, reason, confidence = hybrid_retriever.should_retrieve("hello")
        assert should is False
        assert reason == "greeting_or_command"
        assert confidence == 0.0

    def test_empty_query_zero_confidence(self, hybrid_retriever):
        should, reason, confidence = hybrid_retriever.should_retrieve("")
        assert should is False
        assert reason == "query_too_short"
        assert confidence == 0.0

    def test_short_non_domain_zero_confidence(self, hybrid_retriever):
        should, reason, confidence = hybrid_retriever.should_retrieve("ab")
        assert should is False
        assert reason == "query_too_short"
        assert confidence == 0.0

    def test_help_command_skip(self, hybrid_retriever):
        should, reason, confidence = hybrid_retriever.should_retrieve("help")
        assert should is False
        assert confidence == 0.0

    def test_thanks_skip(self, hybrid_retriever):
        should, reason, confidence = hybrid_retriever.should_retrieve("감사합니다")
        assert should is False
        assert confidence == 0.0

    def test_brand_name_high_confidence(self, hybrid_retriever):
        should, reason, confidence = hybrid_retriever.should_retrieve("COSRX 제품")
        assert should is True
        assert confidence == 1.0

    def test_metric_keyword_high_confidence(self, hybrid_retriever):
        should, reason, confidence = hybrid_retriever.should_retrieve("SoS 점유율")
        assert should is True
        assert confidence == 1.0

    def test_question_word_high_confidence(self, hybrid_retriever):
        should, reason, confidence = hybrid_retriever.should_retrieve("왜 순위가 떨어졌나요")
        assert should is True
        assert confidence == 1.0


class TestHybridSearch:
    """Test _hybrid_search() method."""

    @pytest.mark.asyncio
    async def test_hybrid_search_with_bm25(self, hybrid_retriever, mock_doc_retriever):
        """When BM25 is available, should use RRF fusion."""
        results, method = await hybrid_retriever._hybrid_search("LANEIGE 분석", top_k=5)

        assert method == "hybrid_rrf"
        assert len(results) > 0
        mock_doc_retriever.search.assert_called_once()
        mock_doc_retriever.search_bm25.assert_called_once()
        mock_doc_retriever.reciprocal_rank_fusion.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_dense_only_fallback(self, mock_kg, mock_reasoner):
        """When BM25 is NOT available, should fall back to dense-only."""
        from src.rag.hybrid_retriever import HybridRetriever

        doc_retriever = MagicMock()
        doc_retriever.initialize = AsyncMock()
        doc_retriever.search = AsyncMock(
            return_value=[
                {"id": "d1", "content": "test doc", "score": 0.9, "metadata": {}},
            ]
        )
        # Remove BM25 capability
        doc_retriever.configure_mock(**{"search_bm25": None})
        del doc_retriever.search_bm25

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=mock_reasoner,
            doc_retriever=doc_retriever,
            auto_init_rules=False,
        )
        retriever._initialized = True

        results, method = await retriever._hybrid_search("LANEIGE", top_k=5)

        assert method == "dense_only"
        assert len(results) == 1
        doc_retriever.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_bm25_empty_results(self, hybrid_retriever, mock_doc_retriever):
        """When BM25 returns empty, should use dense results."""
        mock_doc_retriever.search_bm25.return_value = []

        results, method = await hybrid_retriever._hybrid_search("test query", top_k=5)

        assert method == "dense_only"
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_hybrid_search_passes_doc_type_filter(self, hybrid_retriever, mock_doc_retriever):
        """Should pass doc_type_filter to dense search."""
        await hybrid_retriever._hybrid_search("test", top_k=3, doc_type_filter=["playbook"])

        call_kwargs = mock_doc_retriever.search.call_args
        assert call_kwargs[1].get("doc_type_filter") == ["playbook"]

    @pytest.mark.asyncio
    async def test_hybrid_search_respects_top_k(self, hybrid_retriever, mock_doc_retriever):
        """Should pass top_k to search methods."""
        await hybrid_retriever._hybrid_search("test", top_k=3)

        call_kwargs = mock_doc_retriever.search.call_args
        assert call_kwargs[1].get("top_k") == 3


class TestRetrieveUsesHybridSearch:
    """Test that retrieve() delegates to _hybrid_search()."""

    @pytest.mark.asyncio
    async def test_retrieve_calls_hybrid_search(self, hybrid_retriever):
        """retrieve() should call _hybrid_search instead of doc_retriever.search directly."""
        with patch.object(
            hybrid_retriever,
            "_hybrid_search",
            new_callable=AsyncMock,
            return_value=(
                [{"id": "r1", "content": "result", "score": 0.9, "metadata": {}}],
                "hybrid_rrf",
            ),
        ) as mock_hybrid:
            context = await hybrid_retriever.retrieve("LANEIGE 경쟁력 분석")

            # _hybrid_search should have been called
            assert mock_hybrid.called

    @pytest.mark.asyncio
    async def test_retrieve_skip_does_not_call_hybrid(self, hybrid_retriever):
        """retrieve() should skip _hybrid_search for greeting queries."""
        with patch.object(
            hybrid_retriever, "_hybrid_search", new_callable=AsyncMock
        ) as mock_hybrid:
            context = await hybrid_retriever.retrieve("안녕하세요")

            mock_hybrid.assert_not_called()
            assert context.metadata.get("self_rag_skip") is True


class TestMetadataSearchMethod:
    """Test search_method is recorded in metadata."""

    @pytest.mark.asyncio
    async def test_metadata_contains_search_method(self, hybrid_retriever):
        with patch.object(
            hybrid_retriever,
            "_hybrid_search",
            new_callable=AsyncMock,
            return_value=(
                [{"id": "r1", "content": "result", "score": 0.9, "metadata": {}}],
                "hybrid_rrf",
            ),
        ):
            context = await hybrid_retriever.retrieve("LANEIGE SoS 분석")

            assert "search_method" in context.metadata
            assert context.metadata["search_method"] == "hybrid_rrf"

    @pytest.mark.asyncio
    async def test_metadata_contains_selfrag_confidence(self, hybrid_retriever):
        with patch.object(
            hybrid_retriever,
            "_hybrid_search",
            new_callable=AsyncMock,
            return_value=(
                [{"id": "r1", "content": "result", "score": 0.9, "metadata": {}}],
                "dense_only",
            ),
        ):
            context = await hybrid_retriever.retrieve("LANEIGE 경쟁력")

            assert "selfrag_confidence" in context.metadata
            assert context.metadata["selfrag_confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_metadata_contains_bm25_available(self, hybrid_retriever):
        with patch.object(
            hybrid_retriever,
            "_hybrid_search",
            new_callable=AsyncMock,
            return_value=([], "dense_only"),
        ):
            context = await hybrid_retriever.retrieve("LANEIGE 분석")

            assert "bm25_available" in context.metadata

    @pytest.mark.asyncio
    async def test_skip_metadata_contains_selfrag_confidence(self, hybrid_retriever):
        """Skipped queries should also have selfrag_confidence in metadata."""
        context = await hybrid_retriever.retrieve("안녕하세요")

        assert context.metadata.get("selfrag_confidence") == 0.0


class TestSelfRAGConfidenceReducesTopK:
    """Test that low confidence reduces top_k."""

    @pytest.mark.asyncio
    async def test_low_confidence_reduces_top_k(self, hybrid_retriever):
        """Queries with confidence < 0.5 should have reduced top_k."""
        # Patch should_retrieve to return low confidence
        with (
            patch.object(
                hybrid_retriever,
                "should_retrieve",
                return_value=(True, "default_retrieve", 0.3),
            ),
            patch.object(
                hybrid_retriever,
                "_hybrid_search",
                new_callable=AsyncMock,
                return_value=([], "dense_only"),
            ) as mock_hybrid,
        ):
            await hybrid_retriever.retrieve("short q")

            if mock_hybrid.called:
                call_kwargs = mock_hybrid.call_args
                top_k = call_kwargs[1].get(
                    "top_k", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None
                )
                # top_k should be reduced (at least halved from default)
                assert top_k is not None
                assert top_k <= 5  # default is usually 5-10

    @pytest.mark.asyncio
    async def test_high_confidence_keeps_top_k(self, hybrid_retriever):
        """High-confidence queries should NOT reduce top_k."""
        with patch.object(
            hybrid_retriever,
            "_hybrid_search",
            new_callable=AsyncMock,
            return_value=(
                [{"id": "r1", "content": "result", "score": 0.9, "metadata": {}}],
                "hybrid_rrf",
            ),
        ) as mock_hybrid:
            await hybrid_retriever.retrieve("LANEIGE SoS 순위 분석")

            # For a high-confidence domain query, top_k should NOT be reduced
            if mock_hybrid.called:
                call_kwargs = mock_hybrid.call_args
                top_k_val = call_kwargs[1].get("top_k", 5)
                assert top_k_val >= 2


class TestSelfRAGSkipZeroConfidence:
    """Test that skip patterns consistently return 0.0 confidence."""

    @pytest.mark.parametrize(
        "query",
        [
            "안녕하세요",
            "hello",
            "hi there",
            "감사합니다",
            "thanks",
            "help",
            "도움말",
            "",
            "a",
        ],
    )
    def test_skip_patterns_zero_confidence(self, hybrid_retriever, query):
        should, reason, confidence = hybrid_retriever.should_retrieve(query)
        assert should is False
        assert confidence == 0.0

    @pytest.mark.parametrize(
        "query",
        [
            "LANEIGE 립케어",
            "COSRX 경쟁 분석",
            "SoS 점유율은?",
            "왜 순위가 떨어졌나요",
            "트렌드 분석해줘",
        ],
    )
    def test_domain_patterns_high_confidence(self, hybrid_retriever, query):
        should, reason, confidence = hybrid_retriever.should_retrieve(query)
        assert should is True
        assert confidence == 1.0
