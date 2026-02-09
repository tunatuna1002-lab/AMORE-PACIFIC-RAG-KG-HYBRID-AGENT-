"""Tests for L2 retrieval metrics."""

import pytest

from eval.metrics.l2_retrieval import (
    L2RetrievalMetrics,
    context_precision_at_k,
    context_recall_at_k,
    mrr,
)
from eval.schemas import DocRetrievalTrace, GoldEvidence


class TestL2RetrievalMetrics:
    """Tests for L2 retrieval metrics."""

    @pytest.fixture
    def calculator(self):
        """Create L2 metrics calculator."""
        return L2RetrievalMetrics(default_k=5)

    def test_recall_at_k_all_hits(self, calculator):
        """Test recall@k with all gold chunks retrieved."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3", "c4", "c5"],
            snippets=[""] * 5,
            scores=[0.9, 0.8, 0.7, 0.6, 0.5],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1", "c2"])

        recall = calculator._compute_recall_at_k(trace, gold, k=5)
        assert recall == 1.0

    def test_recall_at_k_partial_hits(self, calculator):
        """Test recall@k with partial hits."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c3", "c5"],
            snippets=[""] * 3,
            scores=[0.9, 0.8, 0.7],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1", "c2"])

        recall = calculator._compute_recall_at_k(trace, gold, k=3)
        assert recall == 0.5  # 1 of 2 gold chunks found

    def test_recall_at_k_no_hits(self, calculator):
        """Test recall@k with no hits."""
        trace = DocRetrievalTrace(
            chunk_ids=["c3", "c4", "c5"],
            snippets=[""] * 3,
            scores=[0.9, 0.8, 0.7],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1", "c2"])

        recall = calculator._compute_recall_at_k(trace, gold, k=3)
        assert recall == 0.0

    def test_recall_at_k_empty_gold(self, calculator):
        """Test recall@k with empty gold."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2"],
            snippets=[""] * 2,
            scores=[0.9, 0.8],
        )
        gold = GoldEvidence(doc_chunk_ids=[])

        recall = calculator._compute_recall_at_k(trace, gold, k=5)
        assert recall == 1.0  # No gold = perfect recall

    def test_precision_at_k_all_relevant(self, calculator):
        """Test precision@k when all retrieved are relevant."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2"],
            snippets=[""] * 2,
            scores=[0.9, 0.8],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1", "c2", "c3"])

        precision = calculator._compute_precision_at_k(trace, gold, k=2)
        assert precision == 1.0

    def test_precision_at_k_partial_relevant(self, calculator):
        """Test precision@k with partial relevance."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3", "c4"],
            snippets=[""] * 4,
            scores=[0.9, 0.8, 0.7, 0.6],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1", "c3"])

        precision = calculator._compute_precision_at_k(trace, gold, k=4)
        assert precision == 0.5  # 2 of 4 are relevant

    def test_mrr_first_position(self, calculator):
        """Test MRR when first result is relevant."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3"],
            snippets=[""] * 3,
            scores=[0.9, 0.8, 0.7],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1"])

        mrr_score = calculator._compute_mrr(trace, gold)
        assert mrr_score == 1.0  # 1/(1) = 1.0

    def test_mrr_second_position(self, calculator):
        """Test MRR when second result is relevant."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3"],
            snippets=[""] * 3,
            scores=[0.9, 0.8, 0.7],
        )
        gold = GoldEvidence(doc_chunk_ids=["c2"])

        mrr_score = calculator._compute_mrr(trace, gold)
        assert mrr_score == 0.5  # 1/(2) = 0.5

    def test_mrr_third_position(self, calculator):
        """Test MRR when third result is relevant."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3"],
            snippets=[""] * 3,
            scores=[0.9, 0.8, 0.7],
        )
        gold = GoldEvidence(doc_chunk_ids=["c3"])

        mrr_score = calculator._compute_mrr(trace, gold)
        assert abs(mrr_score - 0.333) < 0.01  # 1/(3) ≈ 0.333

    def test_mrr_no_relevant(self, calculator):
        """Test MRR when no relevant results."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3"],
            snippets=[""] * 3,
            scores=[0.9, 0.8, 0.7],
        )
        gold = GoldEvidence(doc_chunk_ids=["c4", "c5"])

        mrr_score = calculator._compute_mrr(trace, gold)
        assert mrr_score == 0.0

    def test_compute_full_metrics(self, calculator):
        """Test full L2 metrics computation."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3", "c4", "c5"],
            snippets=[""] * 5,
            scores=[0.9, 0.8, 0.7, 0.6, 0.5],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1", "c3"])

        metrics = calculator.compute(trace, gold, k=5)

        assert metrics.context_recall_at_k == 1.0  # Both found in top 5
        assert metrics.context_precision_at_k == 0.4  # 2/5 relevant
        assert metrics.mrr == 1.0  # c1 is first

    def test_case_insensitive(self, calculator):
        """Test metrics are case insensitive."""
        trace = DocRetrievalTrace(
            chunk_ids=["CHUNK1", "chunk2"],
            snippets=[""] * 2,
            scores=[0.9, 0.8],
        )
        gold = GoldEvidence(doc_chunk_ids=["chunk1", "CHUNK2"])

        metrics = calculator.compute(trace, gold, k=2)
        assert metrics.context_recall_at_k == 1.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_context_recall_at_k_function(self):
        """Test context_recall_at_k convenience function."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2"],
            snippets=[""] * 2,
            scores=[0.9, 0.8],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1"])

        recall = context_recall_at_k(trace, gold, k=2)
        assert recall == 1.0

    def test_context_precision_at_k_function(self):
        """Test context_precision_at_k convenience function."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2"],
            snippets=[""] * 2,
            scores=[0.9, 0.8],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1"])

        precision = context_precision_at_k(trace, gold, k=2)
        assert precision == 0.5

    def test_mrr_function(self):
        """Test mrr convenience function."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2"],
            snippets=[""] * 2,
            scores=[0.9, 0.8],
        )
        gold = GoldEvidence(doc_chunk_ids=["c2"])

        mrr_score = mrr(trace, gold)
        assert mrr_score == 0.5
