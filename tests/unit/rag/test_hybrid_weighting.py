"""Tests for hybrid retrieval weighted merge (2.4)"""

from unittest.mock import MagicMock, patch

from src.rag.hybrid_retriever import HybridContext, HybridRetriever


class TestWeightedMerge:
    """_weighted_merge() tests"""

    def _make_retriever(self):
        """Create HybridRetriever with mocked dependencies"""
        with (
            patch("src.rag.hybrid_retriever.KnowledgeGraph"),
            patch("src.rag.hybrid_retriever.OntologyReasoner"),
            patch("src.rag.hybrid_retriever.DocumentRetriever"),
            patch("src.rag.hybrid_retriever.register_all_rules"),
        ):
            return HybridRetriever(auto_init_rules=False)

    def test_weighted_merge_sorts_rag_by_score(self):
        retriever = self._make_retriever()
        context = HybridContext(query="test")
        context.rag_chunks = [
            {"content": "low", "score": 0.3, "metadata": {}},
            {"content": "high", "score": 0.9, "metadata": {}},
            {"content": "mid", "score": 0.6, "metadata": {}},
        ]
        result = retriever._weighted_merge(context)
        scores = [c.get("_weighted_score", 0) for c in result.rag_chunks]
        assert scores == sorted(scores, reverse=True)

    def test_weighted_merge_limits_items(self):
        retriever = self._make_retriever()
        retriever._retrieval_weights = {
            "weights": {"kg": 0.4, "rag": 0.4, "inference": 0.2},
            "freshness": {"weekly": 1.0, "quarterly": 0.9, "static": 0.8},
            "max_context_items": {"ontology_facts": 2, "inferences": 2, "rag_chunks": 1},
        }
        context = HybridContext(query="test")
        context.ontology_facts = [
            {"type": "brand_info", "entity": "a", "data": {}},
            {"type": "brand_info", "entity": "b", "data": {}},
            {"type": "brand_info", "entity": "c", "data": {}},
        ]
        context.rag_chunks = [
            {"content": "a", "score": 0.9, "metadata": {}},
            {"content": "b", "score": 0.5, "metadata": {}},
        ]
        result = retriever._weighted_merge(context)
        assert len(result.ontology_facts) <= 2
        assert len(result.rag_chunks) <= 1

    def test_weighted_merge_empty_context(self):
        retriever = self._make_retriever()
        context = HybridContext(query="test")
        result = retriever._weighted_merge(context)
        assert result.rag_chunks == []
        assert result.ontology_facts == []

    def test_weighted_merge_preserves_metadata(self):
        retriever = self._make_retriever()
        context = HybridContext(query="test")
        context.metadata = {"existing_key": "value"}
        context.rag_chunks = [{"content": "x", "score": 0.8, "metadata": {}}]
        result = retriever._weighted_merge(context)
        assert "existing_key" in result.metadata
        assert "weighted_scores" in result.metadata

    def test_freshness_factor_applied(self):
        retriever = self._make_retriever()
        context = HybridContext(query="test")
        context.rag_chunks = [
            {"content": "static", "score": 0.8, "metadata": {"doc_type": "metric_guide"}},
            {"content": "weekly", "score": 0.8, "metadata": {"doc_type": "intelligence"}},
        ]
        result = retriever._weighted_merge(context)
        # intelligence (weekly freshness=1.0) should score higher than metric_guide (static freshness=0.8)
        assert result.rag_chunks[0]["content"] == "weekly"

    def test_inference_sorted_by_confidence(self):
        retriever = self._make_retriever()
        context = HybridContext(query="test")
        # Create mock inferences
        inf_low = MagicMock()
        inf_low.confidence = 0.3
        inf_high = MagicMock()
        inf_high.confidence = 0.9
        context.inferences = [inf_low, inf_high]
        result = retriever._weighted_merge(context)
        assert result.inferences[0].confidence >= result.inferences[1].confidence

    def test_load_retrieval_weights_defaults(self):
        retriever = self._make_retriever()
        weights = retriever._load_retrieval_weights()
        assert "weights" in weights
        assert weights["weights"]["kg"] == 0.4
        assert weights["weights"]["rag"] == 0.4
        assert weights["weights"]["inference"] == 0.2
