"""Tests for NDCG metrics."""

import math

import pytest

from eval.metrics.ndcg import NDCGCalculator, ndcg_at_k


class TestNDCGCalculator:
    """Tests for NDCG calculator."""

    @pytest.fixture
    def calculator(self):
        """Create NDCG calculator."""
        return NDCGCalculator()

    def test_ndcg_perfect_ranking(self, calculator):
        """Test NDCG with perfect ranking (all gold items first)."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        gold = ["doc1", "doc2"]

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=4)
        assert ndcg == 1.0  # Perfect ranking

    def test_ndcg_worst_ranking(self, calculator):
        """Test NDCG with worst ranking (all gold items last)."""
        retrieved = ["doc3", "doc4", "doc1", "doc2"]
        gold = ["doc1", "doc2"]

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=4)
        assert ndcg < 0.7  # Poor ranking

    def test_ndcg_partial_ranking(self, calculator):
        """Test NDCG with partial ranking (one gold first, one last)."""
        retrieved = ["doc1", "doc3", "doc4", "doc2"]
        gold = ["doc1", "doc2"]

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=4)
        assert 0.7 < ndcg < 1.0  # Medium ranking

    def test_ndcg_empty_gold(self, calculator):
        """Test NDCG with empty gold (no relevant docs)."""
        retrieved = ["doc1", "doc2", "doc3"]
        gold = []

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=3)
        assert ndcg == 1.0  # No gold = perfect score

    def test_ndcg_empty_retrieved(self, calculator):
        """Test NDCG with empty retrieved but non-empty gold."""
        retrieved = []
        gold = ["doc1", "doc2"]

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=10)
        assert ndcg == 0.0  # No retrieved but gold exists = 0.0

    def test_ndcg_both_empty(self, calculator):
        """Test NDCG with both empty."""
        retrieved = []
        gold = []

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=10)
        assert ndcg == 1.0  # Both empty = perfect

    def test_ndcg_at_k_cutoff(self, calculator):
        """Test NDCG@k with different cutoff values."""
        retrieved = ["doc3", "doc1", "doc2", "doc4"]
        gold = ["doc1", "doc2"]

        # At k=1, only doc3 considered (no gold)
        ndcg_1 = calculator.compute_ndcg_at_k(retrieved, gold, k=1)
        assert ndcg_1 == 0.0

        # At k=2, doc3 and doc1 considered (1 gold found)
        ndcg_2 = calculator.compute_ndcg_at_k(retrieved, gold, k=2)
        assert 0.0 < ndcg_2 < 1.0

        # At k=3, doc3, doc1, doc2 considered (both gold found)
        ndcg_3 = calculator.compute_ndcg_at_k(retrieved, gold, k=3)
        assert ndcg_3 > ndcg_2

    def test_ndcg_with_relevance_scores(self, calculator):
        """Test NDCG with custom relevance scores."""
        retrieved = ["doc1", "doc2", "doc3"]
        gold = ["doc1", "doc2", "doc3"]
        relevance = {
            "doc1": 1.0,  # Highly relevant
            "doc2": 0.5,  # Moderately relevant
            "doc3": 0.1,  # Slightly relevant
        }

        # Perfect ranking (highest relevance first)
        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, relevance, k=3)
        assert ndcg == 1.0

    def test_ndcg_with_relevance_scores_bad_ranking(self, calculator):
        """Test NDCG with custom relevance scores and bad ranking."""
        retrieved = ["doc3", "doc2", "doc1"]  # Reversed (worst first)
        gold = ["doc1", "doc2", "doc3"]
        relevance = {
            "doc1": 1.0,  # Should be first
            "doc2": 0.5,
            "doc3": 0.1,  # Should be last
        }

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, relevance, k=3)
        assert ndcg < 0.8  # Penalty for bad ranking

    def test_ndcg_binary_relevance_default(self, calculator):
        """Test NDCG uses binary relevance by default."""
        retrieved = ["doc1", "doc2", "doc3"]
        gold = ["doc1", "doc3"]

        # Without relevance_scores, should use binary (1.0 for gold, 0.0 otherwise)
        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=3)
        assert 0.0 < ndcg <= 1.0

    def test_ndcg_no_relevant_in_top_k(self, calculator):
        """Test NDCG when no relevant docs in top-k."""
        retrieved = ["doc3", "doc4", "doc1", "doc2"]
        gold = ["doc1", "doc2"]

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=2)
        assert ndcg == 0.0  # No gold in top-2

    def test_dcg_calculation(self, calculator):
        """Test DCG calculation directly."""
        doc_ids = ["doc1", "doc2", "doc3"]
        relevance = {"doc1": 1.0, "doc2": 0.5, "doc3": 0.0}

        dcg = calculator._compute_dcg(doc_ids, relevance, k=3)

        # Expected: 1.0/log2(2) + 0.5/log2(3) + 0.0/log2(4)
        expected = 1.0 / math.log2(2) + 0.5 / math.log2(3)
        assert abs(dcg - expected) < 0.01

    def test_dcg_with_k_cutoff(self, calculator):
        """Test DCG respects k cutoff."""
        doc_ids = ["doc1", "doc2", "doc3", "doc4"]
        relevance = {"doc1": 1.0, "doc2": 1.0, "doc3": 1.0, "doc4": 1.0}

        dcg_2 = calculator._compute_dcg(doc_ids, relevance, k=2)
        dcg_4 = calculator._compute_dcg(doc_ids, relevance, k=4)

        assert dcg_4 > dcg_2  # More docs considered = higher DCG

    def test_case_insensitive_matching(self, calculator):
        """Test NDCG is case insensitive."""
        retrieved = ["DOC1", "doc2", "Doc3"]
        gold = ["doc1", "DOC2"]

        # Should match despite case differences (both found in top 2)
        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=3)
        assert ndcg == 1.0  # Perfect ranking: both gold docs in positions 1 and 2


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_ndcg_at_k_function(self):
        """Test ndcg_at_k convenience function."""
        retrieved = ["doc1", "doc2", "doc3"]
        gold = ["doc1", "doc2"]

        ndcg = ndcg_at_k(retrieved, gold, k=3)
        assert ndcg == 1.0

    def test_ndcg_at_k_with_relevance(self):
        """Test ndcg_at_k with custom relevance."""
        retrieved = ["doc1", "doc2", "doc3"]
        gold = ["doc1", "doc2", "doc3"]
        relevance = {"doc1": 1.0, "doc2": 0.5, "doc3": 0.1}

        ndcg = ndcg_at_k(retrieved, gold, relevance, k=3)
        assert ndcg == 1.0

    def test_ndcg_at_k_default_k(self):
        """Test ndcg_at_k uses default k=10."""
        retrieved = ["doc1"] + [f"doc{i}" for i in range(2, 20)]
        gold = ["doc1"]

        ndcg = ndcg_at_k(retrieved, gold)  # Uses default k=10
        assert ndcg == 1.0


class TestNDCGEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def calculator(self):
        """Create NDCG calculator."""
        return NDCGCalculator()

    def test_single_doc_perfect(self, calculator):
        """Test NDCG with single document (perfect)."""
        retrieved = ["doc1"]
        gold = ["doc1"]

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=1)
        assert ndcg == 1.0

    def test_single_doc_wrong(self, calculator):
        """Test NDCG with single document (wrong)."""
        retrieved = ["doc2"]
        gold = ["doc1"]

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=1)
        assert ndcg == 0.0

    def test_k_larger_than_retrieved(self, calculator):
        """Test NDCG when k > len(retrieved)."""
        retrieved = ["doc1", "doc2"]
        gold = ["doc1", "doc2"]

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=10)
        assert ndcg == 1.0  # Should handle gracefully

    def test_k_zero(self, calculator):
        """Test NDCG with k=0."""
        retrieved = ["doc1", "doc2"]
        gold = ["doc1"]

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=0)
        # DCG and IDCG both 0, should return 0.0
        assert ndcg == 0.0

    def test_all_zero_relevance(self, calculator):
        """Test NDCG when all relevance scores are 0."""
        retrieved = ["doc1", "doc2"]
        gold = ["doc1", "doc2"]
        relevance = {"doc1": 0.0, "doc2": 0.0}

        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, relevance, k=2)
        # IDCG is 0, should return 0.0 (avoid division by zero)
        assert ndcg == 0.0

    def test_duplicate_docs_in_retrieved(self, calculator):
        """Test NDCG with duplicate docs in retrieved list."""
        retrieved = ["doc1", "doc1", "doc2"]
        gold = ["doc1", "doc2"]

        # Duplicates cause DCG to exceed IDCG, so we clamp to 1.0
        ndcg = calculator.compute_ndcg_at_k(retrieved, gold, k=3)
        assert ndcg == 1.0  # Clamped to max 1.0
