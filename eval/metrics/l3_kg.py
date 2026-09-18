"""
L3 Knowledge Graph Metrics
==========================
KG traversal and query quality metrics.

Measures how well the system retrieves KG information:
- Hits@k: Binary indicator if any gold entity in top-k
- KG Edge F1: F1 between retrieved and gold edges
"""

from typing import Any

from eval.metrics.base import MetricCalculator
from eval.schemas import GoldEvidence, KGQueryTrace, L3Metrics
from eval.validators.ontology_validator import parse_edge


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

        gold_only = self.compute_gold_edge_breakdown(trace, gold)

        return L3Metrics(
            hits_at_k=hits,
            kg_edge_f1=edge_f1,
            kg_edge_recall=edge_recall,
            kg_edge_precision=edge_precision,
            **gold_only,
        )

    def compute_gold_edge_breakdown(
        self, trace: KGQueryTrace, gold: GoldEvidence
    ) -> dict[str, Any]:
        """[2026-09 사후] O0-A: 골드 엣지가 있는 문항만의 recall과 술어별 일치 수.

        - ``kg_edge_recall_gold_only``: 골드 엣지가 1개 이상이면 ``kg_edge_recall``과 같은 값,
          없으면 None (1.0으로 채우지 않는다 — 집계에서 빠진다).
        - ``edge_recall_by_predicate``: 골드 술어(골드 표기 그대로)별 {matched, total}.
          일치 판정은 ``kg_edge_recall``과 같은 정규화(소문자·앞뒤 공백 제거) 문자열 일치다.
          별칭(ownedBy ↔ ownedByGroup)은 맞춰 주지 않는다 — 이름이 달라 놓치는 것도
          지금 시스템의 실제 결과로 센다.
        """
        gold_edges = self._norm_edges(gold.kg_edges)
        retrieved = self._norm_edges(trace.kg_edges_found)
        by_predicate: dict[str, dict[str, int]] = {}
        for edge in sorted(gold_edges):
            parsed = parse_edge(edge)
            # 정규화로 소문자가 됐으므로 술어는 원 골드 표기에서 다시 찾는다
            predicate = self._gold_predicate(gold.kg_edges, edge) if parsed else "(unparsed)"
            bucket = by_predicate.setdefault(predicate, {"matched": 0, "total": 0})
            bucket["total"] += 1
            if edge in retrieved:
                bucket["matched"] += 1
        matched = len(gold_edges & retrieved)
        return {
            "kg_edge_recall_gold_only": (matched / len(gold_edges)) if gold_edges else None,
            "gold_edge_count": len(gold_edges),
            "gold_edge_matched": matched,
            "edge_recall_by_predicate": by_predicate,
        }

    @staticmethod
    def _gold_predicate(raw_edges: list[str], normalized_edge: str) -> str:
        for raw in raw_edges:
            if str(raw).lower().strip() == normalized_edge:
                parsed = parse_edge(raw)
                if parsed:
                    return parsed[1]
        parsed = parse_edge(normalized_edge)
        return parsed[1] if parsed else "(unparsed)"

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


def aggregate_l3_extended(metrics: list[L3Metrics]) -> dict[str, Any]:
    """[2026-09 사후] O0-A: 문항별 L3Metrics를 리포트 요약용으로 집계한다.

    - ``kg_edge_recall_all``: 기존 ``kg_edge_recall`` 평균(골드 엣지 없는 문항 = 1.0 포함)
    - ``kg_edge_recall_gold_only``: 골드 엣지 있는 문항만의 문항 평균(macro). 해당 문항 0이면 None
    - ``kg_edge_recall_micro``: 전체 골드 엣지 중 일치 비율(엣지 단위)
    - ``recall_by_predicate``: 술어별 {matched, total, recall}
    """
    n = len(metrics)
    gold_only = [
        m.kg_edge_recall_gold_only for m in metrics if m.kg_edge_recall_gold_only is not None
    ]
    total_edges = sum(m.gold_edge_count for m in metrics)
    total_matched = sum(m.gold_edge_matched for m in metrics)
    by_predicate: dict[str, dict[str, float]] = {}
    for m in metrics:
        for predicate, counts in m.edge_recall_by_predicate.items():
            bucket = by_predicate.setdefault(predicate, {"matched": 0, "total": 0})
            bucket["matched"] += counts.get("matched", 0)
            bucket["total"] += counts.get("total", 0)
    for bucket in by_predicate.values():
        bucket["recall"] = bucket["matched"] / bucket["total"] if bucket["total"] else None
    return {
        "items": n,
        "kg_edge_recall_all": (sum(m.kg_edge_recall for m in metrics) / n) if n else None,
        "gold_edge_items": len(gold_only),
        "kg_edge_recall_gold_only": (sum(gold_only) / len(gold_only)) if gold_only else None,
        "gold_edges_total": total_edges,
        "gold_edges_matched": total_matched,
        "kg_edge_recall_micro": (total_matched / total_edges) if total_edges else None,
        "recall_by_predicate": dict(sorted(by_predicate.items())),
    }
