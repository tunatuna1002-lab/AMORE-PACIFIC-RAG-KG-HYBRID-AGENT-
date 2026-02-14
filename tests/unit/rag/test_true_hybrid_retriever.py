"""
Unit tests for TrueHybridRetriever

Tests the complete hybrid retrieval pipeline with mocked dependencies.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.rag.confidence_fusion import FusedResult
from src.rag.entity_linker import LinkedEntity
from src.rag.true_hybrid_retriever import (
    HybridResult,
    TrueHybridRetriever,
)


class TestHybridResult:
    """Test HybridResult dataclass"""

    def test_initialization(self):
        """Test HybridResult initialization with defaults"""
        result = HybridResult(query="test query")

        assert result.query == "test query"
        assert result.documents == []
        assert result.ontology_context == {}
        assert result.entity_links == []
        assert result.confidence == 0.0
        assert result.combined_context == ""
        assert result.metadata == {}

    def test_to_dict_empty(self):
        """Test to_dict with empty result"""
        result = HybridResult(query="test")
        data = result.to_dict()

        assert data["query"] == "test"
        assert data["entities"]["brands"] == []
        assert data["entities"]["categories"] == []
        assert data["entities"]["indicators"] == []
        assert data["entities"]["products"] == []
        assert data["ontology_facts"] == []
        assert data["inferences"] == []
        assert data["rag_chunks"] == []

    def test_to_dict_with_entities(self):
        """Test to_dict with entity links"""
        entity1 = Mock()
        entity1.entity_type = "brand"
        entity1.ontology_id = "LANEIGE"
        entity1.text = "LANEIGE"

        entity2 = Mock()
        entity2.entity_type = "category"
        entity2.ontology_id = "lip_care"
        entity2.text = "Lip Care"

        result = HybridResult(query="test", entity_links=[entity1, entity2])
        data = result.to_dict()

        assert "LANEIGE" in data["entities"]["brands"]
        assert "lip_care" in data["entities"]["categories"]

    def test_to_dict_with_ontology_context(self):
        """Test to_dict with ontology facts and inferences"""
        result = HybridResult(
            query="test",
            ontology_context={
                "facts": ["fact1", "fact2"],
                "inferences": [{"type": "insight", "content": "test insight"}],
            },
        )
        data = result.to_dict()

        assert data["ontology_facts"] == ["fact1", "fact2"]
        assert len(data["inferences"]) == 1


class TestTrueHybridRetrieverInit:
    """Test TrueHybridRetriever initialization"""

    def test_default_initialization(self):
        """Test initialization with defaults"""
        retriever = TrueHybridRetriever()

        assert retriever.kg is None
        assert retriever.owl_reasoner is not None
        assert retriever.doc_retriever is not None
        assert retriever.entity_linker is not None
        assert retriever.confidence_fusion is not None
        assert retriever._initialized is False
        assert retriever.use_reranking is True

    def test_initialization_with_dependencies(self):
        """Test initialization with provided dependencies"""
        mock_kg = Mock()
        mock_owl = Mock()
        mock_doc = Mock()

        retriever = TrueHybridRetriever(
            knowledge_graph=mock_kg,
            owl_reasoner=mock_owl,
            doc_retriever=mock_doc,
            use_reranking=False,
        )

        assert retriever.kg is mock_kg
        assert retriever.owl_reasoner is mock_owl
        assert retriever.doc_retriever is mock_doc
        assert retriever.use_reranking is False

    def test_initialization_with_unified_reasoner(self):
        """Test initialization with unified reasoner"""
        mock_unified = Mock()

        retriever = TrueHybridRetriever(unified_reasoner=mock_unified)

        assert retriever.unified_reasoner is mock_unified


class TestTrueHybridRetrieverInitialize:
    """Test async initialize method"""

    @pytest.mark.asyncio
    async def test_initialize_basic(self):
        """Test basic initialization"""
        mock_doc = AsyncMock()
        mock_owl = AsyncMock()

        retriever = TrueHybridRetriever(doc_retriever=mock_doc, owl_reasoner=mock_owl)

        await retriever.initialize()

        assert retriever._initialized is True
        mock_doc.initialize.assert_awaited_once()
        mock_owl.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_with_ontology_kg(self):
        """Test initialization with OntologyKnowledgeGraph"""
        mock_doc = AsyncMock()
        mock_owl = AsyncMock()
        mock_ontology_kg = AsyncMock()
        mock_ontology_kg.initialize = AsyncMock()

        retriever = TrueHybridRetriever(
            doc_retriever=mock_doc, owl_reasoner=mock_owl, ontology_kg=mock_ontology_kg
        )

        await retriever.initialize()

        mock_ontology_kg.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_with_kg_migration(self):
        """Test OWL migration from KnowledgeGraph"""
        mock_doc = AsyncMock()
        mock_owl = AsyncMock()
        mock_owl.import_from_knowledge_graph = Mock(return_value=10)
        mock_owl.run_reasoner = Mock()
        mock_owl.infer_market_positions = Mock()
        mock_kg = Mock()

        retriever = TrueHybridRetriever(
            doc_retriever=mock_doc, owl_reasoner=mock_owl, knowledge_graph=mock_kg
        )

        await retriever.initialize()

        mock_owl.import_from_knowledge_graph.assert_called_once_with(mock_kg)
        mock_owl.run_reasoner.assert_called_once()
        mock_owl.infer_market_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_handles_errors(self):
        """Test initialization handles errors gracefully"""
        mock_doc = AsyncMock()
        mock_owl = AsyncMock()
        mock_ontology_kg = AsyncMock()
        mock_ontology_kg.initialize = AsyncMock(side_effect=Exception("Init failed"))

        retriever = TrueHybridRetriever(
            doc_retriever=mock_doc, owl_reasoner=mock_owl, ontology_kg=mock_ontology_kg
        )

        # Should not raise, just warn
        await retriever.initialize()
        assert retriever._initialized is True


class TestTrueHybridRetrieverEntityLinking:
    """Test entity linking methods"""

    def test_link_entities(self):
        """Test _link_entities calls EntityLinker"""
        mock_entity = Mock(spec=LinkedEntity)
        mock_linker = Mock()
        mock_linker.link = Mock(return_value=[mock_entity])

        retriever = TrueHybridRetriever()
        retriever.entity_linker = mock_linker

        result = retriever._link_entities("LANEIGE Lip Care")

        mock_linker.link.assert_called_once_with("LANEIGE Lip Care")
        assert result == [mock_entity]

    def test_build_ontology_filters(self):
        """Test _build_ontology_filters"""
        mock_entity = Mock(spec=LinkedEntity)
        mock_linker = Mock()
        mock_linker.get_ontology_filters = Mock(return_value={"brand": {"$in": ["LANEIGE"]}})

        retriever = TrueHybridRetriever()
        retriever.entity_linker = mock_linker

        filters = retriever._build_ontology_filters([mock_entity])

        mock_linker.get_ontology_filters.assert_called_once_with([mock_entity])
        assert filters == {"brand": {"$in": ["LANEIGE"]}}


class TestTrueHybridRetrieverSearch:
    """Test search pipeline methods"""

    @pytest.mark.asyncio
    async def test_expand_query(self):
        """Test query expansion"""
        mock_doc = AsyncMock()
        mock_doc.expand_query = AsyncMock(return_value=["query1", "query2"])

        retriever = TrueHybridRetriever(doc_retriever=mock_doc)

        result = await retriever._expand_query("test query")

        mock_doc.expand_query.assert_awaited_once_with("test query")
        assert result == ["query1", "query2"]

    @pytest.mark.asyncio
    async def test_ontology_guided_search(self):
        """Test ontology-guided vector search"""
        mock_doc = AsyncMock()
        mock_doc.search = AsyncMock(
            return_value=[
                {"id": "doc1", "content": "test", "metadata": {"brand": "LANEIGE"}},
                {"id": "doc2", "content": "test2", "metadata": {"brand": "COSRX"}},
            ]
        )

        retriever = TrueHybridRetriever(doc_retriever=mock_doc)

        filters = {"brand": {"$in": ["LANEIGE"]}}
        results = await retriever._ontology_guided_search(["query1"], filters, top_k=5)

        # Should filter to only LANEIGE
        assert len(results) == 1
        assert results[0]["id"] == "doc1"

    @pytest.mark.asyncio
    async def test_ontology_guided_search_no_filters(self):
        """Test search with no filters passes all results"""
        mock_doc = AsyncMock()
        mock_doc.search = AsyncMock(
            return_value=[
                {"id": "doc1", "content": "test", "metadata": {}},
                {"id": "doc2", "content": "test2", "metadata": {}},
            ]
        )

        retriever = TrueHybridRetriever(doc_retriever=mock_doc)

        results = await retriever._ontology_guided_search(["query1"], {}, top_k=5)

        assert len(results) == 2

    def test_matches_filters(self):
        """Test metadata filter matching"""
        retriever = TrueHybridRetriever()

        # Exact match
        assert retriever._matches_filters({"brand": "LANEIGE"}, {"brand": "LANEIGE"})

        # $in operator
        assert retriever._matches_filters(
            {"brand": "LANEIGE"}, {"brand": {"$in": ["LANEIGE", "COSRX"]}}
        )

        # No match
        assert not retriever._matches_filters({"brand": "TIRTIR"}, {"brand": {"$in": ["LANEIGE"]}})

        # Empty filters
        assert retriever._matches_filters({"brand": "any"}, {})


class TestTrueHybridRetrieverOntologyReasoning:
    """Test ontology reasoning integration"""

    @pytest.mark.asyncio
    async def test_infer_with_unified_reasoner(self):
        """Test inference with UnifiedReasoner"""
        mock_result = Mock()
        mock_result.to_dict = Mock(return_value={"insight": "test", "confidence": 0.8})

        mock_unified = Mock()
        mock_unified.infer = Mock(return_value=mock_result)

        mock_entity = Mock()
        mock_entity.entity_type = "brand"
        mock_entity.ontology_id = "LANEIGE"
        mock_entity.text = "LANEIGE"

        retriever = TrueHybridRetriever(unified_reasoner=mock_unified)

        context = await retriever._infer_with_ontology([mock_entity], {"sos": 0.15})

        assert "inferences" in context
        assert len(context["inferences"]) > 0

    @pytest.mark.asyncio
    async def test_infer_with_owl_fallback(self):
        """Test OWL-only inference when no UnifiedReasoner"""
        mock_owl = AsyncMock()
        mock_owl.get_inferred_facts = Mock(return_value=[{"type": "fact", "subject": "test"}])
        mock_owl.get_brand_info = Mock(
            return_value={
                "market_position": "StrongBrand",
                "sos": 0.15,
                "competitors": ["COSRX", "TIRTIR"],
            }
        )

        mock_entity = Mock()
        mock_entity.entity_type = "brand"
        mock_entity.ontology_id = "LANEIGE"
        mock_entity.text = "LANEIGE"

        retriever = TrueHybridRetriever(owl_reasoner=mock_owl)

        context = await retriever._infer_with_ontology([mock_entity], {})

        assert "facts" in context
        assert "inferences" in context

    @pytest.mark.asyncio
    async def test_infer_handles_exceptions(self):
        """Test inference handles exceptions gracefully"""
        mock_unified = Mock()
        mock_unified.infer = Mock(side_effect=Exception("Inference failed"))

        mock_entity = Mock()
        mock_entity.entity_type = "brand"
        mock_entity.ontology_id = "LANEIGE"

        retriever = TrueHybridRetriever(unified_reasoner=mock_unified)

        context = await retriever._infer_with_ontology([mock_entity], {})

        # Should return empty context, not raise
        assert context == {"inferences": [], "facts": [], "related_docs": []}


class TestTrueHybridRetrieverReranking:
    """Test reranking methods"""

    @pytest.mark.asyncio
    async def test_rerank_success(self):
        """Test successful reranking"""
        mock_reranker = Mock()
        mock_doc1 = Mock(content="test1", score=0.9, metadata={"chunk_id": "1"})
        mock_doc2 = Mock(content="test2", score=0.7, metadata={"chunk_id": "2"})
        mock_reranker.rerank = Mock(return_value=[mock_doc1, mock_doc2])

        with patch("src.rag.true_hybrid_retriever.get_reranker", return_value=mock_reranker):
            retriever = TrueHybridRetriever(use_reranking=True)

            docs = [
                {"id": "1", "content": "test1", "score": 0.5},
                {"id": "2", "content": "test2", "score": 0.6},
            ]

            result = await retriever._rerank("query", docs, top_k=2)

            assert len(result) == 2
            assert result[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_rerank_failure_fallback(self):
        """Test reranking fallback on error"""
        mock_reranker = Mock()
        mock_reranker.rerank = Mock(side_effect=Exception("Rerank failed"))

        with patch("src.rag.true_hybrid_retriever.get_reranker", return_value=mock_reranker):
            retriever = TrueHybridRetriever(use_reranking=True)

            docs = [
                {"id": "1", "content": "test1", "score": 0.5},
                {"id": "2", "content": "test2", "score": 0.6},
            ]

            result = await retriever._rerank("query", docs, top_k=2)

            # Should fallback to original top_k
            assert len(result) == 2


class TestTrueHybridRetrieverFusion:
    """Test confidence fusion"""

    def test_fuse_results(self):
        """Test result fusion"""
        mock_fusion = Mock()
        mock_fused = FusedResult(
            documents=[{"content": "test", "score": 0.8}],
            confidence=0.85,
            explanation="test",
            source_scores=[],
            fusion_strategy="weighted",
        )
        mock_fusion.fuse = Mock(return_value=mock_fused)

        retriever = TrueHybridRetriever()
        retriever.confidence_fusion = mock_fusion

        vector_results = [{"content": "test", "score": 0.7, "metadata": {}}]

        result = retriever._fuse_results(
            vector_results=vector_results,
            ontology_results=[],
            reranked_results=[],
            entity_confidence=0.9,
        )

        mock_fusion.fuse.assert_called_once()
        assert result == mock_fused


class TestTrueHybridRetrieverCombinedContext:
    """Test combined context building"""

    def test_build_combined_context_with_entities(self):
        """Test context building with entities"""
        mock_entity = Mock()
        mock_entity.entity_type = "brand"
        mock_entity.text = "LANEIGE"
        mock_entity.confidence = 0.95

        result = HybridResult(query="test", entity_links=[mock_entity])
        retriever = TrueHybridRetriever()

        context = retriever._build_combined_context(result)

        assert "추출된 엔티티" in context
        assert "LANEIGE" in context

    def test_build_combined_context_with_ontology(self):
        """Test context building with ontology inferences"""
        result = HybridResult(
            query="test",
            ontology_context={
                "inferences": [
                    {
                        "type": "market_position",
                        "brand": "LANEIGE",
                        "position": "StrongBrand",
                        "sos": 0.15,
                    }
                ]
            },
        )
        retriever = TrueHybridRetriever()

        context = retriever._build_combined_context(result)

        assert "온톨로지 추론 결과" in context
        assert "LANEIGE" in context

    def test_build_combined_context_with_documents(self):
        """Test context building with documents"""
        result = HybridResult(
            query="test",
            documents=[
                {"content": "Long content " * 100, "metadata": {"title": "Test Doc"}, "score": 0.9}
            ],
        )
        retriever = TrueHybridRetriever()

        context = retriever._build_combined_context(result)

        assert "관련 문서" in context
        assert "Test Doc" in context
        # Should truncate long content
        assert len(context) < 1000


class TestTrueHybridRetrieverConfidenceCalculation:
    """Test overall confidence calculation"""

    def test_calculate_overall_confidence(self):
        """Test confidence calculation formula"""
        retriever = TrueHybridRetriever()

        # High confidence across all sources
        conf = retriever._calculate_overall_confidence(
            entity_confidence=0.9, avg_doc_score=0.85, ontology_coverage=1.0
        )
        assert conf > 0.8

        # Low confidence
        conf = retriever._calculate_overall_confidence(
            entity_confidence=0.3, avg_doc_score=0.4, ontology_coverage=0.0
        )
        assert conf < 0.5

        # Bounds check
        conf = retriever._calculate_overall_confidence(
            entity_confidence=1.5, avg_doc_score=1.2, ontology_coverage=2.0
        )
        assert conf == 1.0


class TestTrueHybridRetrieverRetrieve:
    """Test the main retrieve method"""

    @pytest.mark.asyncio
    async def test_retrieve_full_pipeline(self):
        """Test complete retrieval pipeline"""
        # Setup mocks
        mock_entity = Mock()
        mock_entity.entity_type = "brand"
        mock_entity.ontology_id = "LANEIGE"
        mock_entity.text = "LANEIGE"
        mock_entity.confidence = 0.9

        mock_linker = Mock()
        mock_linker.link = Mock(return_value=[mock_entity])
        mock_linker.get_ontology_filters = Mock(return_value={})

        mock_doc = AsyncMock()
        mock_doc.initialize = AsyncMock()
        mock_doc.expand_query = AsyncMock(return_value=["query1"])
        mock_doc.search = AsyncMock(
            return_value=[{"id": "1", "content": "test", "metadata": {}, "score": 0.8}]
        )

        mock_owl = AsyncMock()
        mock_owl.initialize = AsyncMock()
        mock_owl.get_inferred_facts = Mock(return_value=[])
        mock_owl.get_brand_info = Mock(return_value=None)

        mock_fusion = Mock()
        mock_fused = FusedResult(
            documents=[{"id": "1", "content": "test", "score": 0.8, "metadata": {}}],
            confidence=0.85,
            explanation="test",
            source_scores=[],
            fusion_strategy="weighted",
        )
        mock_fusion.fuse = Mock(return_value=mock_fused)

        # Create retriever
        retriever = TrueHybridRetriever(
            doc_retriever=mock_doc,
            owl_reasoner=mock_owl,
            use_reranking=False,
        )
        retriever.entity_linker = mock_linker
        retriever.confidence_fusion = mock_fusion

        # Execute
        result = await retriever.retrieve("LANEIGE analysis", top_k=5)

        # Verify
        assert result.query == "LANEIGE analysis"
        assert len(result.entity_links) == 1
        assert len(result.documents) > 0
        assert result.confidence > 0
        assert "retrieval_time_ms" in result.metadata

    @pytest.mark.asyncio
    async def test_retrieve_auto_initializes(self):
        """Test retrieve auto-initializes if needed"""
        mock_doc = AsyncMock()
        mock_doc.initialize = AsyncMock()
        mock_doc.expand_query = AsyncMock(return_value=["query"])
        mock_doc.search = AsyncMock(return_value=[])

        mock_owl = AsyncMock()
        mock_owl.initialize = AsyncMock()
        mock_owl.get_inferred_facts = Mock(return_value=[])

        retriever = TrueHybridRetriever(doc_retriever=mock_doc, owl_reasoner=mock_owl)
        retriever.entity_linker.link = Mock(return_value=[])

        assert retriever._initialized is False

        await retriever.retrieve("test")

        assert retriever._initialized is True
        mock_doc.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieve_handles_errors(self):
        """Test retrieve handles errors gracefully"""
        mock_doc = AsyncMock()
        mock_doc.initialize = AsyncMock()
        mock_doc.expand_query = AsyncMock(side_effect=Exception("Search failed"))

        retriever = TrueHybridRetriever(doc_retriever=mock_doc)
        retriever.entity_linker.link = Mock(return_value=[])

        result = await retriever.retrieve("test")

        assert result.confidence == 0.0
        assert "error" in result.metadata


class TestTrueHybridRetrieverStats:
    """Test statistics methods"""

    def test_get_stats(self):
        """Test get_stats returns expected data"""
        mock_owl = Mock()
        mock_owl.get_stats = Mock(return_value={"entities": 100})

        mock_doc = Mock()
        mock_doc.documents = ["doc1", "doc2"]
        mock_doc.chunks = ["chunk1", "chunk2", "chunk3"]

        retriever = TrueHybridRetriever(owl_reasoner=mock_owl, doc_retriever=mock_doc)

        stats = retriever.get_stats()

        assert stats["owl_reasoner"]["entities"] == 100
        assert stats["doc_retriever"]["documents_count"] == 2
        assert stats["doc_retriever"]["chunks_count"] == 3
        assert stats["initialized"] is False


class TestGetTrueHybridRetriever:
    """Test singleton factory function"""

    def test_get_singleton(self):
        """Test singleton pattern"""
        with patch("src.rag.true_hybrid_retriever._retriever_instance", None):
            retriever1 = Mock(spec=TrueHybridRetriever)

            with patch(
                "src.rag.true_hybrid_retriever.TrueHybridRetriever", return_value=retriever1
            ):
                from src.rag.true_hybrid_retriever import get_true_hybrid_retriever

                result1 = get_true_hybrid_retriever()
                result2 = get_true_hybrid_retriever()

                # Should return same instance
                assert result1 is result2
