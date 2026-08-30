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
        edge_recall = self._compute_kg_edge_recall(trace, gold)
        edge_precision = self._compute_kg_edge_precision(trace, gold)

        return L3Metrics(
            hits_at_k=hits,
            kg_edge_f1=edge_f1,
            kg_edge_recall=edge_recall,
            kg_edge_precision=edge_precision,
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

    @staticmethod
    def _norm_edges(edges) -> set[str]:
        return {str(e).lower().strip() for e in edges}

    def _compute_kg_edge_recall(self, trace: KGQueryTrace, gold: GoldEvidence) -> float:
        """
        Compute KG edge recall — 골드 엣지 중 검색된 비율.

        `kg_edge_f1`은 이 데이터셋에서 구조적으로 판별력이 없다: 골드는 문항당
        1~3개(중앙값 1)를 열거하는 반면 KG 컨텍스트는 문항당 최대 12개를
        방출하므로, **골드를 100% 회수해도 F1은 약 0.18**에 그친다. F1 0.5
        게이트는 총 방출 엣지가 3개 이하일 때만 도달 가능해 사실상 상시 fail
        이었다 (v4.1: requires_kg 130문항 중 125문항 fail). 그래서 게이트는
        recall로 옮기고 F1은 연속성을 위해 계속 보고한다. recall만 보면
        "엣지를 전부 쏟아내기"로 점수를 올릴 수 있으므로 방출 상한(12개)을
        유지하고 `kg_edge_precision`을 함께 보고해 남용을 감시한다.
        골드 엣지가 없으면 hits@k와 동일하게 1.0(찾을 것이 없음)으로 둔다.
        """
        gold_edges = self._norm_edges(gold.kg_edges)
        if not gold_edges:
            return 1.0
        retrieved = self._norm_edges(trace.kg_edges_found)
        return len(gold_edges & retrieved) / len(gold_edges)

    def _compute_kg_edge_precision(self, trace: KGQueryTrace, gold: GoldEvidence) -> float:
        """Compute KG edge precision — 방출 엣지 중 골드에 있는 비율."""
        retrieved = self._norm_edges(trace.kg_edges_found)
        if not retrieved:
            return 1.0
        gold_edges = self._norm_edges(gold.kg_edges)
        return len(gold_edges & retrieved) / len(retrieved)


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
