"""
L3 Knowledge Graph Metrics
==========================
KG traversal and query quality metrics.

Measures how well the system retrieves KG information:
- Hits@k: Binary indicator if any gold entity in top-k
- KG Edge F1: F1 between retrieved and gold edges
"""

from eval.metrics.base import MetricCalculator
from eval.schemas import GoldEvidence, KGQueryTrace, L3Metrics


class L3KGMetrics(MetricCalculator):
    """
    L3 metrics for Knowledge Graph retrieval.

    Evaluates KG query quality against gold entities and edges.
    """

    def __init__(self, default_k: int = 10):
        """
        Initialize L3 metrics calculator.

        Args:
            default_k: Default cutoff for Hits@k
        """
        self.default_k = default_k

    def compute(
        self,
        trace: KGQueryTrace,
        gold: GoldEvidence,
        k: int | None = None,
    ) -> L3Metrics:
        """
        Compute L3 metrics.

        Args:
            trace: KG query trace
            gold: Gold standard evidence
            k: Cutoff for Hits@k (defaults to self.default_k)

        Returns:
            L3Metrics with hits_at_k and kg_edge_f1
        """
        k = k or self.default_k

        hits = self._compute_hits_at_k(trace, gold, k)
        edge_f1 = self._compute_kg_edge_f1(trace, gold)

        return L3Metrics(
            hits_at_k=hits,
            kg_edge_f1=edge_f1,
        )

    def _compute_hits_at_k(self, trace: KGQueryTrace, gold: GoldEvidence, k: int) -> float:
        """
        Compute Hits@k.

        Binary metric: 1 if any gold entity appears in top-k KG results.
        """
        gold_entities = set(gold.kg_entities)

        if not gold_entities:
            return 1.0  # No gold entities to find

        return self.hits_at_k(trace.kg_entities_found, gold_entities, k)

    def _compute_kg_edge_f1(self, trace: KGQueryTrace, gold: GoldEvidence) -> float:
        """
        Compute KG edge F1.

        F1 between retrieved edges and gold edges.
        Edges are normalized for comparison (lowercased, stripped).
        """
        retrieved_edges = set(trace.kg_edges_found)
        gold_edges = set(gold.kg_edges)

        return self.set_f1(retrieved_edges, gold_edges)


def hits_at_k(trace: KGQueryTrace, gold: GoldEvidence, k: int = 10) -> float:
    """
    Convenience function for Hits@k.

    Args:
        trace: KG query trace
        gold: Gold evidence
        k: Cutoff position

    Returns:
        1.0 if any gold entity in top-k, else 0.0
    """
    calc = L3KGMetrics(default_k=k)
    return calc._compute_hits_at_k(trace, gold, k)


def kg_edge_f1(trace: KGQueryTrace, gold: GoldEvidence) -> float:
    """
    Convenience function for KG edge F1.

    Args:
        trace: KG query trace
        gold: Gold evidence

    Returns:
        F1 score between retrieved and gold edges
    """
    calc = L3KGMetrics()
    return calc._compute_kg_edge_f1(trace, gold)


def kg_entity_recall(trace: KGQueryTrace, gold: GoldEvidence) -> float:
    """
    Compute entity recall (proportion of gold entities found).

    Args:
        trace: KG query trace
        gold: Gold evidence

    Returns:
        Recall of gold entities
    """
    gold_entities = set(gold.kg_entities)

    if not gold_entities:
        return 1.0

    return MetricCalculator.set_recall(set(trace.kg_entities_found), gold_entities)


def kg_entity_precision(trace: KGQueryTrace, gold: GoldEvidence) -> float:
    """
    Compute entity precision (proportion of found entities that are gold).

    Args:
        trace: KG query trace
        gold: Gold evidence

    Returns:
        Precision of found entities
    """
    gold_entities = set(gold.kg_entities)
    found_entities = set(trace.kg_entities_found)

    if not found_entities:
        return 1.0 if not gold_entities else 0.0

    return MetricCalculator.set_precision(found_entities, gold_entities)
