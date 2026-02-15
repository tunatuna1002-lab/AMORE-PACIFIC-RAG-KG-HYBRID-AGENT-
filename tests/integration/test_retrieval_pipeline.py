"""
Integration tests for the Retrieval Pipeline

Tests: HybridRetriever.retrieve_unified() → ContextGatherer → Context output
Uses mocks for external services (LLM, ChromaDB) but real domain logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.value_objects.retrieval_result import UnifiedRetrievalResult


class TestRetrievalPipelineBasic:
    """Test basic retrieval pipeline integration"""

    @pytest.mark.asyncio
    async def test_retrieve_unified_legacy_path(self):
        """Test HybridRetriever.retrieve_unified() via legacy path"""
        from src.rag.hybrid_retriever import HybridContext

        mock_context = HybridContext(
            query="test query",
            entities={"brands": ["LANEIGE"], "categories": [], "indicators": [], "products": []},
            ontology_facts=[],
            inferences=[],
            rag_chunks=[{"content": "test chunk", "score": 0.8, "metadata": {}}],
            combined_context="Test context",
            metadata={"retrieval_time_ms": 100},
        )

        mock_kg = MagicMock()
        mock_kg.get_entity_metadata = MagicMock(return_value=None)
        mock_kg.get_brand_products = MagicMock(return_value=[])
        mock_kg.get_competitors = MagicMock(return_value=[])
        mock_kg.get_category_hierarchy = MagicMock(return_value=[])
        mock_kg.query_triples = MagicMock(return_value=[])

        mock_reasoner = MagicMock()
        mock_reasoner.rules = []

        mock_doc = AsyncMock()

        with patch("src.rag.hybrid_retriever.register_all_rules"):
            from src.rag.hybrid_retriever import HybridRetriever

            retriever = HybridRetriever(
                knowledge_graph=mock_kg,
                reasoner=mock_reasoner,
                doc_retriever=mock_doc,
                owl_strategy=None,
            )
            # Patch legacy retrieve to return mock context
            retriever.retrieve = AsyncMock(return_value=mock_context)

            result = await retriever.retrieve_unified("test query")

            assert isinstance(result, UnifiedRetrievalResult)
            assert result.query == "test query"
            assert "LANEIGE" in result.entities["brands"]
            assert len(result.rag_chunks) == 1
            assert result.retriever_type == "legacy"

    @pytest.mark.asyncio
    async def test_retrieve_unified_owl_path(self):
        """Test HybridRetriever.retrieve_unified() via OWL strategy"""
        owl_result = UnifiedRetrievalResult(
            query="test query",
            entities={"brands": ["LANEIGE"]},
            ontology_facts=[
                {"subject": "LANEIGE", "predicate": "hasCategory", "object": "lip_care"}
            ],
            inferences=[{"type": "growth", "content": "Growing trend"}],
            rag_chunks=[{"content": "test", "score": 0.8, "metadata": {}}],
            combined_context="OWL context",
            confidence=0.85,
            entity_links=[{"text": "LANEIGE", "type": "brand"}],
            metadata={"retrieval_time_ms": 150},
            retriever_type="owl",
        )

        mock_strategy = AsyncMock()
        mock_strategy.retrieve = AsyncMock(return_value=owl_result)

        mock_kg = MagicMock()
        mock_kg.get_entity_metadata = MagicMock(return_value=None)
        mock_kg.get_brand_products = MagicMock(return_value=[])
        mock_kg.get_competitors = MagicMock(return_value=[])
        mock_kg.get_category_hierarchy = MagicMock(return_value=[])
        mock_kg.query_triples = MagicMock(return_value=[])

        mock_reasoner = MagicMock()
        mock_reasoner.rules = []

        mock_doc = AsyncMock()

        with patch("src.rag.hybrid_retriever.register_all_rules"):
            from src.rag.hybrid_retriever import HybridRetriever

            retriever = HybridRetriever(
                knowledge_graph=mock_kg,
                reasoner=mock_reasoner,
                doc_retriever=mock_doc,
                owl_strategy=mock_strategy,
            )

            result = await retriever.retrieve_unified("test query")

            assert isinstance(result, UnifiedRetrievalResult)
            assert result.query == "test query"
            assert result.confidence == 0.85
            assert result.retriever_type == "owl"


class TestRetrievalPipelineContextOutput:
    """Test final context output structure validation"""

    @pytest.mark.asyncio
    async def test_unified_retrieval_result_structure(self):
        """Test UnifiedRetrievalResult has all required fields"""
        result = UnifiedRetrievalResult(
            query="test",
            entities={"brands": ["LANEIGE"]},
            ontology_facts=["fact1"],
            inferences=[{"insight": "insight1"}],
            rag_chunks=[{"content": "chunk1", "score": 0.8, "metadata": {}}],
            combined_context="context",
            confidence=0.8,
            metadata={"retrieval_time_ms": 100, "entity_count": 1},
            retriever_type="owl",
        )

        assert hasattr(result, "query")
        assert hasattr(result, "entities")
        assert hasattr(result, "ontology_facts")
        assert hasattr(result, "inferences")
        assert hasattr(result, "rag_chunks")
        assert hasattr(result, "combined_context")
        assert hasattr(result, "metadata")
        assert hasattr(result, "confidence")
        assert hasattr(result, "retriever_type")


class TestRetrievalPipelineMultiQuery:
    """Test pipeline with multiple queries"""

    @pytest.mark.asyncio
    async def test_multiple_queries_maintain_isolation(self):
        """Test multiple queries don't interfere with each other"""

        async def make_result(query, **kwargs):
            return UnifiedRetrievalResult(
                query=query,
                combined_context=f"Context for {query}",
                retriever_type="owl",
            )

        mock_strategy = AsyncMock()
        mock_strategy.retrieve = AsyncMock(side_effect=make_result)

        mock_kg = MagicMock()
        mock_kg.get_entity_metadata = MagicMock(return_value=None)
        mock_kg.get_brand_products = MagicMock(return_value=[])
        mock_kg.get_competitors = MagicMock(return_value=[])
        mock_kg.get_category_hierarchy = MagicMock(return_value=[])
        mock_kg.query_triples = MagicMock(return_value=[])

        mock_reasoner = MagicMock()
        mock_reasoner.rules = []

        mock_doc = AsyncMock()

        with patch("src.rag.hybrid_retriever.register_all_rules"):
            from src.rag.hybrid_retriever import HybridRetriever

            retriever = HybridRetriever(
                knowledge_graph=mock_kg,
                reasoner=mock_reasoner,
                doc_retriever=mock_doc,
                owl_strategy=mock_strategy,
            )

            result1 = await retriever.retrieve_unified("query 1")
            result2 = await retriever.retrieve_unified("query 2")

            assert result1.query == "query 1"
            assert result2.query == "query 2"
            assert result1.combined_context == "Context for query 1"
            assert result2.combined_context == "Context for query 2"
