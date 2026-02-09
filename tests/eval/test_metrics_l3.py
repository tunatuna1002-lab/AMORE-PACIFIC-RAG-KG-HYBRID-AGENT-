"""Tests for L3 Knowledge Graph metrics."""

import pytest

from eval.metrics.l3_kg import (
    L3KGMetrics,
    hits_at_k,
    kg_edge_f1,
    kg_entity_precision,
    kg_entity_recall,
)
from eval.schemas import GoldEvidence, KGQueryTrace


class TestL3KGMetrics:
    """Tests for L3 KG metrics."""

    @pytest.fixture
    def calculator(self):
        """Create L3 metrics calculator."""
        return L3KGMetrics(default_k=10)

    def test_hits_at_k_hit(self, calculator):
        """Test hits@k when there's a hit."""
        trace = KGQueryTrace(
            kg_entities_found=["laneige", "cosrx", "innisfree"],
            kg_edges_found=[],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(kg_entities=["laneige"])

        hits = calculator._compute_hits_at_k(trace, gold, k=3)
        assert hits == 1.0

    def test_hits_at_k_miss(self, calculator):
        """Test hits@k when there's no hit."""
        trace = KGQueryTrace(
            kg_entities_found=["cosrx", "innisfree", "tirtir"],
            kg_edges_found=[],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(kg_entities=["laneige"])

        hits = calculator._compute_hits_at_k(trace, gold, k=3)
        assert hits == 0.0

    def test_hits_at_k_outside_k(self, calculator):
        """Test hits@k when hit is outside k."""
        trace = KGQueryTrace(
            kg_entities_found=["a", "b", "c", "d", "laneige"],
            kg_edges_found=[],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(kg_entities=["laneige"])

        hits = calculator._compute_hits_at_k(trace, gold, k=3)
        assert hits == 0.0  # laneige is at position 5, outside k=3

    def test_hits_at_k_empty_gold(self, calculator):
        """Test hits@k with empty gold."""
        trace = KGQueryTrace(
            kg_entities_found=["laneige"],
            kg_edges_found=[],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(kg_entities=[])

        hits = calculator._compute_hits_at_k(trace, gold, k=10)
        assert hits == 1.0  # Empty gold = automatic pass

    def test_kg_edge_f1_exact_match(self, calculator):
        """Test KG edge F1 with exact match."""
        trace = KGQueryTrace(
            kg_entities_found=[],
            kg_edges_found=[
                "laneige -hasProduct-> B08XYZ",
                "laneige -competesWith-> cosrx",
            ],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(
            kg_edges=["laneige -hasProduct-> B08XYZ", "laneige -competesWith-> cosrx"]
        )

        f1 = calculator._compute_kg_edge_f1(trace, gold)
        assert f1 == 1.0

    def test_kg_edge_f1_partial_match(self, calculator):
        """Test KG edge F1 with partial match."""
        trace = KGQueryTrace(
            kg_entities_found=[],
            kg_edges_found=["laneige -hasProduct-> B08XYZ"],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(
            kg_edges=["laneige -hasProduct-> B08XYZ", "laneige -competesWith-> cosrx"]
        )

        f1 = calculator._compute_kg_edge_f1(trace, gold)
        # Precision: 1/1 = 1.0, Recall: 1/2 = 0.5
        # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) ≈ 0.667
        assert 0.66 < f1 < 0.67

    def test_kg_edge_f1_no_match(self, calculator):
        """Test KG edge F1 with no match."""
        trace = KGQueryTrace(
            kg_entities_found=[],
            kg_edges_found=["brand1 -relation-> brand2"],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(kg_edges=["laneige -competesWith-> cosrx"])

        f1 = calculator._compute_kg_edge_f1(trace, gold)
        assert f1 == 0.0

    def test_kg_edge_f1_empty_both(self, calculator):
        """Test KG edge F1 with both empty."""
        trace = KGQueryTrace(
            kg_entities_found=[],
            kg_edges_found=[],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(kg_edges=[])

        f1 = calculator._compute_kg_edge_f1(trace, gold)
        assert f1 == 1.0  # Both empty = perfect match

    def test_compute_full_metrics(self, calculator):
        """Test full L3 metrics computation."""
        trace = KGQueryTrace(
            kg_entities_found=["laneige", "cosrx"],
            kg_edges_found=["laneige -competesWith-> cosrx"],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(
            kg_entities=["laneige", "cosrx"],
            kg_edges=["laneige -competesWith-> cosrx"],
        )

        metrics = calculator.compute(trace, gold, k=10)

        assert metrics.hits_at_k == 1.0
        assert metrics.kg_edge_f1 == 1.0

    def test_case_insensitive(self, calculator):
        """Test metrics are case insensitive."""
        trace = KGQueryTrace(
            kg_entities_found=["LANEIGE"],
            kg_edges_found=["LANEIGE -competesWith-> COSRX"],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(
            kg_entities=["laneige"],
            kg_edges=["laneige -competesWith-> cosrx"],
        )

        metrics = calculator.compute(trace, gold, k=10)
        assert metrics.hits_at_k == 1.0
        assert metrics.kg_edge_f1 == 1.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_hits_at_k_function(self):
        """Test hits_at_k convenience function."""
        trace = KGQueryTrace(
            kg_entities_found=["laneige"],
            kg_edges_found=[],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(kg_entities=["laneige"])

        result = hits_at_k(trace, gold, k=10)
        assert result == 1.0

    def test_kg_edge_f1_function(self):
        """Test kg_edge_f1 convenience function."""
        trace = KGQueryTrace(
            kg_entities_found=[],
            kg_edges_found=["edge1"],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(kg_edges=["edge1"])

        result = kg_edge_f1(trace, gold)
        assert result == 1.0

    def test_kg_entity_recall_function(self):
        """Test kg_entity_recall convenience function."""
        trace = KGQueryTrace(
            kg_entities_found=["laneige"],
            kg_edges_found=[],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(kg_entities=["laneige", "cosrx"])

        recall = kg_entity_recall(trace, gold)
        assert recall == 0.5

    def test_kg_entity_precision_function(self):
        """Test kg_entity_precision convenience function."""
        trace = KGQueryTrace(
            kg_entities_found=["laneige", "innisfree"],
            kg_edges_found=[],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence(kg_entities=["laneige"])

        precision = kg_entity_precision(trace, gold)
        assert precision == 0.5
