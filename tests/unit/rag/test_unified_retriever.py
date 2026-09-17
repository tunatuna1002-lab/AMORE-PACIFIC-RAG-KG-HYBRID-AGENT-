"""
Unit tests for HybridRetriever.retrieve_unified()
===================================================
Tests the unified retrieval path that returns UnifiedRetrievalResult
(HybridContext → UnifiedRetrievalResult 변환).

2026-09: OWL 전략 위임 경로는 삭제됐다 (검색 경로는 HybridRetriever 하나).
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.domain.value_objects.retrieval_result import UnifiedRetrievalResult


@pytest.fixture
def mock_kg():
    """Mock KnowledgeGraph."""
    kg = MagicMock()
    kg.get_entity_metadata = MagicMock(return_value=None)
    kg.get_brand_products = MagicMock(return_value=[])
    kg.get_competitors = MagicMock(return_value=[])
    kg.get_category_hierarchy = MagicMock(return_value=[])
    kg.query_triples = MagicMock(return_value=[])
    return kg


@pytest.fixture
def mock_reasoner():
    """Mock OntologyReasoner."""
    reasoner = MagicMock()
    reasoner.rules = []
    return reasoner


@pytest.fixture
def mock_doc_retriever():
    """Mock DocumentRetriever."""
    doc = AsyncMock()
    doc.initialize = AsyncMock()
    doc.search = AsyncMock(return_value=[])
    return doc


class TestRetrieveUnifiedLegacyFallback:
    """Test retrieve_unified() without OWL strategy (legacy path)."""

    @pytest.mark.asyncio
    async def test_converts_hybrid_context_to_unified_result(
        self, mock_kg, mock_reasoner, mock_doc_retriever
    ):
        """Should convert HybridContext to UnifiedRetrievalResult."""
        with patch("src.rag.hybrid_retriever.register_all_rules"):
            from src.rag.hybrid_retriever import HybridContext, HybridRetriever

            # Create a mock HybridContext
            mock_inference = Mock()
            mock_inference.to_dict = Mock(return_value={"type": "growth", "content": "Growing"})

            mock_context = HybridContext(
                query="test query",
                entities={"brands": ["LANEIGE"]},
                ontology_facts=[{"subject": "LANEIGE"}],
                inferences=[mock_inference],
                rag_chunks=[{"content": "chunk", "score": 0.8}],
                combined_context="Legacy context",
                metadata={"source": "legacy"},
            )

            retriever = HybridRetriever(
                knowledge_graph=mock_kg,
                reasoner=mock_reasoner,
                doc_retriever=mock_doc_retriever,
            )

            # Patch retrieve() to return mock context
            retriever.retrieve = AsyncMock(return_value=mock_context)

            result = await retriever.retrieve_unified("test query")

            assert isinstance(result, UnifiedRetrievalResult)
            assert result.retriever_type == "legacy"
            assert result.query == "test query"
            assert result.entities == {"brands": ["LANEIGE"]}
            assert len(result.inferences) == 1
            assert result.inferences[0] == {"type": "growth", "content": "Growing"}
            assert result.combined_context == "Legacy context"


class TestRetrieveUnifiedSearch:
    """Test search() method delegation."""

    @pytest.mark.asyncio
    async def test_search_delegates_to_doc_retriever(
        self, mock_kg, mock_reasoner, mock_doc_retriever
    ):
        """search()는 DocumentRetriever로 위임한다."""
        mock_doc_retriever.search = AsyncMock(return_value=[{"content": "fallback", "score": 0.7}])

        with patch("src.rag.hybrid_retriever.register_all_rules"):
            from src.rag.hybrid_retriever import HybridRetriever

            retriever = HybridRetriever(
                knowledge_graph=mock_kg,
                reasoner=mock_reasoner,
                doc_retriever=mock_doc_retriever,
            )

            results = await retriever.search("test", top_k=5, doc_filter="brand:LANEIGE")

            assert len(results) == 1
            assert results[0]["content"] == "fallback"
            mock_doc_retriever.search.assert_awaited_once_with(
                query="test", top_k=5, doc_filter="brand:LANEIGE"
            )


class TestUnifiedRetrievalResultValidation:
    """Test UnifiedRetrievalResult data structure."""

    def test_has_correct_field_types(self):
        """UnifiedRetrievalResult should have all expected fields with correct types."""
        result = UnifiedRetrievalResult(
            query="test",
            entities={"brands": ["LANEIGE"]},
            ontology_facts=[{"subject": "LANEIGE"}],
            inferences=[{"type": "growth"}],
            rag_chunks=[{"content": "doc"}],
            combined_context="context",
            confidence=0.8,
            entity_links=[],
            metadata={"key": "value"},
            retriever_type="legacy",
        )

        assert isinstance(result.query, str)
        assert isinstance(result.entities, dict)
        assert isinstance(result.ontology_facts, list)
        assert isinstance(result.inferences, list)
        assert isinstance(result.rag_chunks, list)
        assert isinstance(result.combined_context, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.entity_links, list)
        assert isinstance(result.metadata, dict)
        assert isinstance(result.retriever_type, str)

    def test_defaults(self):
        """Test default values."""
        result = UnifiedRetrievalResult(query="test")

        assert result.entities == {}
        assert result.ontology_facts == []
        assert result.inferences == []
        assert result.rag_chunks == []
        assert result.combined_context == ""
        assert result.confidence == 0.0
        assert result.entity_links == []
        assert result.metadata == {}
        assert result.retriever_type == "unknown"
