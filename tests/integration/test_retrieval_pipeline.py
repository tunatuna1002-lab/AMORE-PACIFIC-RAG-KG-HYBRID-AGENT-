"""
Integration tests for the Retrieval Pipeline

Tests: UnifiedRetriever → ContextGatherer → Context output
Uses mocks for external services (LLM, ChromaDB) but real domain logic.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.domain.interfaces.retriever import UnifiedRetrievalResult
from src.rag.unified_retriever import UnifiedRetriever


class TestRetrievalPipelineBasic:
    """Test basic retrieval pipeline integration"""

    @pytest.mark.asyncio
    async def test_unified_retriever_with_hybrid_backend(self):
        """Test UnifiedRetriever using hybrid backend"""
        # Mock HybridRetriever
        mock_hybrid = AsyncMock()
        mock_hybrid.retrieve = AsyncMock(
            return_value={
                "query": "test query",
                "entities": {
                    "brands": ["LANEIGE"],
                    "categories": [],
                    "indicators": [],
                    "products": [],
                },
                "ontology_facts": [],
                "inferences": [],
                "rag_chunks": [{"content": "test chunk", "score": 0.8, "metadata": {}}],
                "combined_context": "Test context",
                "metadata": {"retrieval_time_ms": 100},
            }
        )

        with patch("src.rag.unified_retriever.HybridRetriever", return_value=mock_hybrid):
            retriever = UnifiedRetriever(backend="hybrid")
            await retriever.initialize()

            result = await retriever.retrieve("test query")

            assert isinstance(result, UnifiedRetrievalResult)
            assert result.query == "test query"
            assert "LANEIGE" in result.entities["brands"]
            assert len(result.documents) > 0

    @pytest.mark.asyncio
    async def test_unified_retriever_with_true_hybrid_backend(self):
        """Test UnifiedRetriever using true_hybrid backend"""
        # Mock TrueHybridRetriever with HybridResult
        from src.rag.entity_linker import LinkedEntity
        from src.rag.true_hybrid_retriever import HybridResult

        mock_entity = Mock(spec=LinkedEntity)
        mock_entity.entity_type = "brand"
        mock_entity.ontology_id = "LANEIGE"
        mock_entity.text = "LANEIGE"
        mock_entity.confidence = 0.9

        mock_result = HybridResult(
            query="test query",
            documents=[{"content": "test", "score": 0.8, "metadata": {}}],
            entity_links=[mock_entity],
            ontology_context={"facts": [], "inferences": []},
            confidence=0.85,
            combined_context="Test context",
            metadata={"retrieval_time_ms": 150},
        )

        mock_true_hybrid = AsyncMock()
        mock_true_hybrid.retrieve = AsyncMock(return_value=mock_result)

        with patch("src.rag.unified_retriever.TrueHybridRetriever", return_value=mock_true_hybrid):
            retriever = UnifiedRetriever(backend="true_hybrid")
            await retriever.initialize()

            result = await retriever.retrieve("test query")

            assert isinstance(result, UnifiedRetrievalResult)
            assert result.query == "test query"
            assert result.confidence == 0.85


class TestRetrievalPipelineFeatureFlags:
    """Test feature flag switching between backends"""

    @pytest.mark.asyncio
    async def test_feature_flag_switches_backend(self):
        """Test feature flags can switch retrieval backend"""
        mock_flags = Mock()
        mock_flags.get = Mock(return_value="true_hybrid")

        mock_true_hybrid = AsyncMock()
        from src.rag.true_hybrid_retriever import HybridResult

        mock_result = HybridResult(query="test", documents=[], entity_links=[])
        mock_true_hybrid.retrieve = AsyncMock(return_value=mock_result)

        with patch("src.rag.unified_retriever.get_feature_flags", return_value=mock_flags):
            with patch(
                "src.rag.unified_retriever.TrueHybridRetriever", return_value=mock_true_hybrid
            ):
                retriever = UnifiedRetriever()  # Should use feature flag
                await retriever.initialize()

                await retriever.retrieve("test")

                # Should have used true_hybrid
                mock_true_hybrid.retrieve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_feature_flag_fallback_to_default(self):
        """Test fallback to default backend when feature flag missing"""
        mock_flags = Mock()
        mock_flags.get = Mock(return_value=None)  # No override

        mock_hybrid = AsyncMock()
        mock_hybrid.retrieve = AsyncMock(
            return_value={
                "query": "test",
                "entities": {"brands": [], "categories": [], "indicators": [], "products": []},
                "ontology_facts": [],
                "inferences": [],
                "rag_chunks": [],
                "combined_context": "",
                "metadata": {},
            }
        )

        with patch("src.rag.unified_retriever.get_feature_flags", return_value=mock_flags):
            with patch("src.rag.unified_retriever.HybridRetriever", return_value=mock_hybrid):
                retriever = UnifiedRetriever()  # Should fallback to hybrid
                await retriever.initialize()

                await retriever.retrieve("test")

                mock_hybrid.retrieve.assert_awaited_once()


class TestRetrievalPipelineContextGathering:
    """Test context gathering from retrieval results"""

    @pytest.mark.asyncio
    async def test_context_builder_integration(self):
        """Test context building from retrieval results"""
        # This would test ContextGatherer if it exists
        # For now, test that UnifiedRetriever provides proper context

        mock_hybrid = AsyncMock()
        mock_hybrid.retrieve = AsyncMock(
            return_value={
                "query": "LANEIGE Lip Care analysis",
                "entities": {
                    "brands": ["LANEIGE"],
                    "categories": ["lip_care"],
                    "indicators": ["SoS"],
                    "products": [],
                },
                "ontology_facts": ["LANEIGE is a K-beauty brand"],
                "inferences": [
                    {"type": "market_position", "brand": "LANEIGE", "position": "Strong"}
                ],
                "rag_chunks": [
                    {"content": "LANEIGE Lip Sleeping Mask is popular", "score": 0.9},
                    {"content": "Lip Care category is competitive", "score": 0.85},
                ],
                "combined_context": "Combined context here",
                "metadata": {"retrieval_time_ms": 200},
            }
        )

        with patch("src.rag.unified_retriever.HybridRetriever", return_value=mock_hybrid):
            retriever = UnifiedRetriever(backend="hybrid")
            await retriever.initialize()

            result = await retriever.retrieve("LANEIGE Lip Care analysis")

            # Verify context structure
            assert result.query == "LANEIGE Lip Care analysis"
            assert "LANEIGE" in result.entities["brands"]
            assert "lip_care" in result.entities["categories"]
            assert len(result.ontology_facts) > 0
            assert len(result.inferences) > 0
            assert len(result.documents) == 2


class TestRetrievalPipelineContextOutput:
    """Test final context output structure validation"""

    @pytest.mark.asyncio
    async def test_unified_retrieval_result_structure(self):
        """Test UnifiedRetrievalResult has all required fields"""
        mock_hybrid = AsyncMock()
        mock_hybrid.retrieve = AsyncMock(
            return_value={
                "query": "test",
                "entities": {"brands": [], "categories": [], "indicators": [], "products": []},
                "ontology_facts": ["fact1"],
                "inferences": [{"insight": "insight1"}],
                "rag_chunks": [{"content": "chunk1", "score": 0.8, "metadata": {}}],
                "combined_context": "context",
                "metadata": {"retrieval_time_ms": 100, "entity_count": 0},
            }
        )

        with patch("src.rag.unified_retriever.HybridRetriever", return_value=mock_hybrid):
            retriever = UnifiedRetriever(backend="hybrid")
            await retriever.initialize()

            result = await retriever.retrieve("test")

            # Verify all expected fields
            assert hasattr(result, "query")
            assert hasattr(result, "entities")
            assert hasattr(result, "ontology_facts")
            assert hasattr(result, "inferences")
            assert hasattr(result, "documents")
            assert hasattr(result, "combined_context")
            assert hasattr(result, "metadata")
            assert hasattr(result, "confidence")

    @pytest.mark.asyncio
    async def test_context_output_with_metrics(self):
        """Test context output includes retrieval metrics"""
        mock_hybrid = AsyncMock()
        mock_hybrid.retrieve = AsyncMock(
            return_value={
                "query": "test",
                "entities": {
                    "brands": ["LANEIGE"],
                    "categories": [],
                    "indicators": [],
                    "products": [],
                },
                "ontology_facts": [],
                "inferences": [],
                "rag_chunks": [],
                "combined_context": "",
                "metadata": {
                    "retrieval_time_ms": 250,
                    "entity_count": 1,
                    "vector_results_count": 5,
                    "final_results_count": 3,
                },
            }
        )

        with patch("src.rag.unified_retriever.HybridRetriever", return_value=mock_hybrid):
            retriever = UnifiedRetriever(backend="hybrid")
            await retriever.initialize()

            result = await retriever.retrieve("test")

            # Verify metrics are preserved
            assert "retrieval_time_ms" in result.metadata
            assert result.metadata["retrieval_time_ms"] == 250
            assert "entity_count" in result.metadata


class TestRetrievalPipelineErrorHandling:
    """Test error handling in the retrieval pipeline"""

    @pytest.mark.asyncio
    async def test_retrieval_handles_backend_errors(self):
        """Test pipeline handles backend errors gracefully"""
        mock_hybrid = AsyncMock()
        mock_hybrid.retrieve = AsyncMock(side_effect=Exception("Backend failed"))

        with patch("src.rag.unified_retriever.HybridRetriever", return_value=mock_hybrid):
            retriever = UnifiedRetriever(backend="hybrid")
            await retriever.initialize()

            # Should not raise, but return error result
            result = await retriever.retrieve("test")

            # Depending on implementation, could return empty or error result
            assert result is not None

    @pytest.mark.asyncio
    async def test_retrieval_handles_missing_fields(self):
        """Test pipeline handles missing optional fields"""
        mock_hybrid = AsyncMock()
        mock_hybrid.retrieve = AsyncMock(
            return_value={
                "query": "test",
                # Missing some optional fields
                "entities": {"brands": [], "categories": [], "indicators": [], "products": []},
                "rag_chunks": [],
            }
        )

        with patch("src.rag.unified_retriever.HybridRetriever", return_value=mock_hybrid):
            retriever = UnifiedRetriever(backend="hybrid")
            await retriever.initialize()

            result = await retriever.retrieve("test")

            # Should handle missing fields gracefully
            assert result.query == "test"
            assert result.ontology_facts == [] or result.ontology_facts is not None


class TestRetrievalPipelinePerformance:
    """Test pipeline performance characteristics"""

    @pytest.mark.asyncio
    async def test_retrieval_completes_within_timeout(self):
        """Test retrieval completes in reasonable time"""
        import asyncio

        mock_hybrid = AsyncMock()

        async def slow_retrieve(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate work
            return {
                "query": "test",
                "entities": {"brands": [], "categories": [], "indicators": [], "products": []},
                "ontology_facts": [],
                "inferences": [],
                "rag_chunks": [],
                "combined_context": "",
                "metadata": {},
            }

        mock_hybrid.retrieve = slow_retrieve

        with patch("src.rag.unified_retriever.HybridRetriever", return_value=mock_hybrid):
            retriever = UnifiedRetriever(backend="hybrid")
            await retriever.initialize()

            # Should complete within timeout
            result = await asyncio.wait_for(retriever.retrieve("test"), timeout=1.0)

            assert result is not None


class TestRetrievalPipelineMultiQuery:
    """Test pipeline with multiple queries"""

    @pytest.mark.asyncio
    async def test_multiple_queries_maintain_isolation(self):
        """Test multiple queries don't interfere with each other"""
        mock_hybrid = AsyncMock()

        def create_result(query):
            return {
                "query": query,
                "entities": {"brands": [], "categories": [], "indicators": [], "products": []},
                "ontology_facts": [],
                "inferences": [],
                "rag_chunks": [],
                "combined_context": f"Context for {query}",
                "metadata": {},
            }

        mock_hybrid.retrieve = AsyncMock(side_effect=lambda q, **kw: create_result(q))

        with patch("src.rag.unified_retriever.HybridRetriever", return_value=mock_hybrid):
            retriever = UnifiedRetriever(backend="hybrid")
            await retriever.initialize()

            result1 = await retriever.retrieve("query 1")
            result2 = await retriever.retrieve("query 2")

            assert result1.query == "query 1"
            assert result2.query == "query 2"
            assert result1.combined_context == "Context for query 1"
            assert result2.combined_context == "Context for query 2"


class TestRetrievalPipelineConfidence:
    """Test confidence scoring through the pipeline"""

    @pytest.mark.asyncio
    async def test_confidence_preserved_from_backend(self):
        """Test confidence scores are preserved from backend"""
        from src.rag.true_hybrid_retriever import HybridResult

        mock_result = HybridResult(
            query="test", documents=[], entity_links=[], confidence=0.92, metadata={}
        )

        mock_true_hybrid = AsyncMock()
        mock_true_hybrid.retrieve = AsyncMock(return_value=mock_result)

        with patch("src.rag.unified_retriever.TrueHybridRetriever", return_value=mock_true_hybrid):
            retriever = UnifiedRetriever(backend="true_hybrid")
            await retriever.initialize()

            result = await retriever.retrieve("test")

            assert result.confidence == 0.92

    @pytest.mark.asyncio
    async def test_low_confidence_handled_appropriately(self):
        """Test low confidence results are still valid"""
        mock_hybrid = AsyncMock()
        mock_hybrid.retrieve = AsyncMock(
            return_value={
                "query": "obscure query",
                "entities": {"brands": [], "categories": [], "indicators": [], "products": []},
                "ontology_facts": [],
                "inferences": [],
                "rag_chunks": [],
                "combined_context": "",
                "metadata": {"confidence": 0.3},
            }
        )

        with patch("src.rag.unified_retriever.HybridRetriever", return_value=mock_hybrid):
            retriever = UnifiedRetriever(backend="hybrid")
            await retriever.initialize()

            result = await retriever.retrieve("obscure query")

            # Should still return valid result even with low confidence
            assert result is not None
            assert result.query == "obscure query"
