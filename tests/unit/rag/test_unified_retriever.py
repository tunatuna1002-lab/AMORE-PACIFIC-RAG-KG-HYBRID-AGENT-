"""
Unit tests for UnifiedRetriever facade
========================================
Tests backend selection, result conversion, and feature flag integration.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.domain.interfaces.retriever import UnifiedRetrievalResult
from src.rag.unified_retriever import UnifiedRetriever, get_unified_retriever


@pytest.fixture
def mock_knowledge_graph():
    """Mock KnowledgeGraph instance."""
    kg = MagicMock()
    kg.get_product_facts = MagicMock(return_value=[])
    return kg


@pytest.fixture
def mock_hybrid_context():
    """Mock HybridContext from legacy HybridRetriever."""
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class MockInferenceResult:
        insight_type: str
        content: str
        confidence: float = 0.8

        def to_dict(self):
            return {
                "insight_type": self.insight_type,
                "content": self.content,
                "confidence": self.confidence,
            }

    @dataclass
    class MockHybridContext:
        query: str
        entities: dict[str, list[str]] = field(default_factory=dict)
        ontology_facts: list[dict[str, Any]] = field(default_factory=list)
        inferences: list = field(default_factory=list)
        rag_chunks: list[dict[str, Any]] = field(default_factory=list)
        combined_context: str = ""
        metadata: dict[str, Any] = field(default_factory=dict)

    ctx = MockHybridContext(
        query="test query",
        entities={"brands": ["LANEIGE"], "categories": ["lip_care"]},
        ontology_facts=[{"subject": "LANEIGE", "predicate": "hasProduct", "object": "Lip Mask"}],
        inferences=[
            MockInferenceResult(insight_type="market_leader", content="LANEIGE leads in lip care")
        ],
        rag_chunks=[{"content": "LANEIGE is popular", "score": 0.9}],
        combined_context="LANEIGE context",
        metadata={"source": "hybrid"},
    )
    return ctx


@pytest.fixture
def mock_hybrid_result():
    """Mock HybridResult from TrueHybridRetriever."""
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class MockLinkedEntity:
        text: str
        entity_type: str
        ontology_id: str = ""

        def __post_init__(self):
            if not self.ontology_id:
                self.ontology_id = self.text.lower()

    @dataclass
    class MockHybridResult:
        query: str
        documents: list[dict[str, Any]] = field(default_factory=list)
        ontology_context: dict[str, Any] = field(default_factory=dict)
        entity_links: list = field(default_factory=list)
        confidence: float = 0.0
        combined_context: str = ""
        metadata: dict[str, Any] = field(default_factory=dict)

    result = MockHybridResult(
        query="test query",
        documents=[{"content": "LANEIGE doc", "score": 0.95}],
        ontology_context={
            "facts": [{"subject": "LANEIGE", "predicate": "hasCategory", "object": "lip_care"}],
            "inferences": [{"type": "growth", "content": "Growing trend"}],
        },
        entity_links=[
            MockLinkedEntity(text="LANEIGE", entity_type="brand", ontology_id="laneige"),
            MockLinkedEntity(text="Lip Care", entity_type="category", ontology_id="lip_care"),
        ],
        confidence=0.92,
        combined_context="True hybrid context",
        metadata={"source": "true_hybrid"},
    )
    return result


class TestUnifiedRetrieverBackendSelection:
    """Test backend selection logic based on feature flags and owlready2 availability."""

    def test_selects_true_hybrid_when_flag_enabled_and_owlready2_available(
        self, mock_knowledge_graph
    ):
        """Should select TrueHybridRetriever when flag=True and owlready2 available."""
        with patch("src.rag.unified_retriever.FeatureFlags") as mock_flags_class:
            mock_flags = Mock()
            mock_flags.use_true_hybrid_retriever.return_value = True
            mock_flags_class.get_instance.return_value = mock_flags

            with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", True):
                with patch(
                    "src.rag.true_hybrid_retriever.get_true_hybrid_retriever"
                ) as mock_get_true:
                    mock_get_true.return_value = MagicMock()

                    retriever = UnifiedRetriever(knowledge_graph=mock_knowledge_graph)

                    assert retriever.backend_type == "true_hybrid"
                    mock_get_true.assert_called_once()

    def test_falls_back_to_hybrid_when_owlready2_unavailable(self, mock_knowledge_graph):
        """Should fall back to HybridRetriever when owlready2 not available."""
        with patch("src.rag.unified_retriever.FeatureFlags") as mock_flags_class:
            mock_flags = Mock()
            mock_flags.use_true_hybrid_retriever.return_value = True
            mock_flags_class.get_instance.return_value = mock_flags

            with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
                with patch("src.rag.hybrid_retriever.HybridRetriever") as mock_hybrid:
                    with patch("src.ontology.reasoner.OntologyReasoner"):
                        with patch("src.rag.retriever.DocumentRetriever"):
                            mock_hybrid.return_value = MagicMock()

                            retriever = UnifiedRetriever(knowledge_graph=mock_knowledge_graph)

                            assert retriever.backend_type == "hybrid"
                            mock_hybrid.assert_called_once()

    def test_uses_hybrid_when_flag_disabled(self, mock_knowledge_graph):
        """Should use HybridRetriever when use_true_hybrid_retriever=False."""
        with patch("src.rag.unified_retriever.FeatureFlags") as mock_flags_class:
            mock_flags = Mock()
            mock_flags.use_true_hybrid_retriever.return_value = False
            mock_flags_class.get_instance.return_value = mock_flags

            with patch("src.rag.hybrid_retriever.HybridRetriever") as mock_hybrid:
                with patch("src.ontology.reasoner.OntologyReasoner"):
                    with patch("src.rag.retriever.DocumentRetriever"):
                        mock_hybrid.return_value = MagicMock()

                        retriever = UnifiedRetriever(knowledge_graph=mock_knowledge_graph)

                        assert retriever.backend_type == "hybrid"
                        mock_hybrid.assert_called_once()


class TestUnifiedRetrieverConversion:
    """Test conversion from backend-specific results to UnifiedRetrievalResult."""

    def test_convert_hybrid_context_to_unified_result(
        self, mock_knowledge_graph, mock_hybrid_context
    ):
        """Should correctly convert HybridContext to UnifiedRetrievalResult."""
        with patch("src.rag.unified_retriever.FeatureFlags") as mock_flags_class:
            mock_flags = Mock()
            mock_flags.use_true_hybrid_retriever.return_value = False
            mock_flags_class.get_instance.return_value = mock_flags

            with patch("src.rag.hybrid_retriever.HybridRetriever"):
                with patch("src.ontology.reasoner.OntologyReasoner"):
                    with patch("src.rag.retriever.DocumentRetriever"):
                        retriever = UnifiedRetriever(knowledge_graph=mock_knowledge_graph)

                        result = retriever._convert_hybrid_context(mock_hybrid_context, "test")

                        assert isinstance(result, UnifiedRetrievalResult)
                        assert result.query == "test"
                        assert result.entities == {
                            "brands": ["LANEIGE"],
                            "categories": ["lip_care"],
                        }
                        assert len(result.ontology_facts) == 1
                        assert len(result.inferences) == 1
                        assert len(result.rag_chunks) == 1
                        assert result.combined_context == "LANEIGE context"
                        assert result.retriever_type == "hybrid"
                        assert result.confidence == 0.0  # HybridContext doesn't have confidence

    def test_convert_hybrid_result_to_unified_result(
        self, mock_knowledge_graph, mock_hybrid_result
    ):
        """Should correctly convert HybridResult to UnifiedRetrievalResult."""
        with patch("src.rag.unified_retriever.FeatureFlags") as mock_flags_class:
            mock_flags = Mock()
            mock_flags.use_true_hybrid_retriever.return_value = False
            mock_flags_class.get_instance.return_value = mock_flags

            with patch("src.rag.hybrid_retriever.HybridRetriever"):
                with patch("src.ontology.reasoner.OntologyReasoner"):
                    with patch("src.rag.retriever.DocumentRetriever"):
                        retriever = UnifiedRetriever(knowledge_graph=mock_knowledge_graph)

                        result = retriever._convert_hybrid_result(mock_hybrid_result, "test")

                        assert isinstance(result, UnifiedRetrievalResult)
                        assert result.query == "test"
                        assert "laneige" in result.entities.get("brands", [])
                        assert "lip_care" in result.entities.get("categories", [])
                        assert len(result.ontology_facts) == 1
                        assert len(result.inferences) == 1
                        assert len(result.rag_chunks) == 1
                        assert result.combined_context == "True hybrid context"
                        assert result.confidence == 0.92
                        assert result.retriever_type == "true_hybrid"
                        assert len(result.entity_links) == 2


class TestUnifiedRetrieverRetrieve:
    """Test retrieve() method with different backends."""

    @pytest.mark.asyncio
    async def test_retrieve_with_hybrid_backend(self, mock_knowledge_graph, mock_hybrid_context):
        """Should call HybridRetriever.retrieve() and convert result."""
        with patch("src.rag.unified_retriever.FeatureFlags") as mock_flags_class:
            mock_flags = Mock()
            mock_flags.use_true_hybrid_retriever.return_value = False
            mock_flags_class.get_instance.return_value = mock_flags

            mock_backend = AsyncMock()
            mock_backend.retrieve.return_value = mock_hybrid_context

            with patch("src.rag.hybrid_retriever.HybridRetriever", return_value=mock_backend):
                with patch("src.ontology.reasoner.OntologyReasoner"):
                    with patch("src.rag.retriever.DocumentRetriever"):
                        retriever = UnifiedRetriever(knowledge_graph=mock_knowledge_graph)

                        result = await retriever.retrieve(query="test query", top_k=5)

                        assert isinstance(result, UnifiedRetrievalResult)
                        assert result.retriever_type == "hybrid"
                        mock_backend.retrieve.assert_called_once_with(
                            query="test query",
                            current_metrics=None,
                            include_explanations=True,
                        )

    @pytest.mark.asyncio
    async def test_retrieve_with_true_hybrid_backend(
        self, mock_knowledge_graph, mock_hybrid_result
    ):
        """Should call TrueHybridRetriever.retrieve() and convert result."""
        with patch("src.rag.unified_retriever.FeatureFlags") as mock_flags_class:
            mock_flags = Mock()
            mock_flags.use_true_hybrid_retriever.return_value = True
            mock_flags_class.get_instance.return_value = mock_flags

            mock_backend = AsyncMock()
            mock_backend.retrieve.return_value = mock_hybrid_result

            with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", True):
                with patch(
                    "src.rag.true_hybrid_retriever.get_true_hybrid_retriever",
                    return_value=mock_backend,
                ):
                    retriever = UnifiedRetriever(knowledge_graph=mock_knowledge_graph)

                    result = await retriever.retrieve(
                        query="test query", current_metrics={"sos": 0.15}, top_k=3
                    )

                    assert isinstance(result, UnifiedRetrievalResult)
                    assert result.retriever_type == "true_hybrid"
                    assert result.confidence == 0.92
                    mock_backend.retrieve.assert_called_once_with(
                        query="test query",
                        current_metrics={"sos": 0.15},
                        top_k=3,
                    )

    @pytest.mark.asyncio
    async def test_retrieve_raises_when_backend_not_initialized(self):
        """Should raise ValueError if backend is None."""
        retriever = UnifiedRetriever.__new__(UnifiedRetriever)
        retriever.backend = None

        with pytest.raises(ValueError, match="backend not initialized"):
            await retriever.retrieve(query="test")


class TestUnifiedRetrieverTypeValidation:
    """Test type validation of result fields."""

    def test_unified_result_has_correct_field_types(self):
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
            retriever_type="hybrid",
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


class TestUnifiedRetrieverFactoryFunction:
    """Test get_unified_retriever() factory function."""

    def test_factory_creates_unified_retriever(self, mock_knowledge_graph):
        """Factory function should create UnifiedRetriever instance."""
        with patch("src.rag.unified_retriever.FeatureFlags") as mock_flags_class:
            mock_flags = Mock()
            mock_flags.use_true_hybrid_retriever.return_value = False
            mock_flags_class.get_instance.return_value = mock_flags

            with patch("src.rag.hybrid_retriever.HybridRetriever"):
                with patch("src.ontology.reasoner.OntologyReasoner"):
                    with patch("src.rag.retriever.DocumentRetriever"):
                        retriever = get_unified_retriever(
                            knowledge_graph=mock_knowledge_graph,
                            config={"docs_path": "./custom/docs"},
                        )

                        assert isinstance(retriever, UnifiedRetriever)
                        assert retriever.config["docs_path"] == "./custom/docs"


class TestUnifiedRetrieverInitialize:
    """Test initialize() method."""

    @pytest.mark.asyncio
    async def test_initialize_calls_backend_initialize(self, mock_knowledge_graph):
        """Should call backend.initialize() if it exists."""
        with patch("src.rag.unified_retriever.FeatureFlags") as mock_flags_class:
            mock_flags = Mock()
            mock_flags.use_true_hybrid_retriever.return_value = False
            mock_flags_class.get_instance.return_value = mock_flags

            mock_backend = AsyncMock()

            with patch("src.rag.hybrid_retriever.HybridRetriever", return_value=mock_backend):
                with patch("src.ontology.reasoner.OntologyReasoner"):
                    with patch("src.rag.retriever.DocumentRetriever"):
                        retriever = UnifiedRetriever(knowledge_graph=mock_knowledge_graph)

                        await retriever.initialize()

                        mock_backend.initialize.assert_called_once()


class TestUnifiedRetrieverSearch:
    """Test search() method delegation."""

    @pytest.mark.asyncio
    async def test_search_delegates_to_backend(self, mock_knowledge_graph):
        """Should delegate search() to backend if available."""
        with patch("src.rag.unified_retriever.FeatureFlags") as mock_flags_class:
            mock_flags = Mock()
            mock_flags.use_true_hybrid_retriever.return_value = False
            mock_flags_class.get_instance.return_value = mock_flags

            mock_backend = AsyncMock()
            mock_backend.search.return_value = [{"content": "doc1"}, {"content": "doc2"}]

            with patch("src.rag.hybrid_retriever.HybridRetriever", return_value=mock_backend):
                with patch("src.ontology.reasoner.OntologyReasoner"):
                    with patch("src.rag.retriever.DocumentRetriever"):
                        retriever = UnifiedRetriever(knowledge_graph=mock_knowledge_graph)

                        results = await retriever.search(query="test", top_k=2)

                        assert len(results) == 2
                        mock_backend.search.assert_called_once_with(
                            query="test", top_k=2, doc_filter=None
                        )
