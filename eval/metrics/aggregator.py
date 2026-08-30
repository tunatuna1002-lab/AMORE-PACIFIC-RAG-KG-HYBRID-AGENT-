"""
Metric Aggregator
=================
Combines L1-L5 metrics into overall scores with gating logic.

Features:
- Weighted overall score computation
- Gating threshold checks
- Fail reason tagging
"""

from eval.schemas import (
    ItemMetadata,
    ItemResult,
    L1Metrics,
    L2Metrics,
    L3Metrics,
    L4Metrics,
    L5Metrics,
)

# Default weights for overall score calculation
DEFAULT_WEIGHTS = {
    "l5": 0.45,  # Answer quality is most important
    "l2_l3": 0.35,  # Retrieval quality
    "l1": 0.10,  # Query understanding
    "l4": 0.10,  # Ontology compliance
}

# Default gating thresholds
DEFAULT_THRESHOLDS = {
    # L5 gating
    "groundedness_min": 0.70,
    "answer_f1_min": 0.50,
    # token-F1은 골드(단문 정의)와 에이전트(장문 마크다운)의 스타일 차이에
    # 구조적으로 취약하므로, semantic similarity가 계산된 경우 이 값 이상이면
    # answer_f1 미달이어도 정답으로 인정한다
    "semantic_similarity_min": 0.65,
    # L4 gating
    "constraint_violation_max": 0.05,
    "type_consistency_min": 0.90,
    # L2/L3 gating (conditional on requires_kg)
    "context_recall_min": 0.80,  # For requires_kg=False (청크 단위, 참고 보고용)
    # L2 게이트는 출처 문서 단위 recall 기준 (2026-08-30 사이클 5):
    # 골드 doc_chunk_ids는 문서 수준 가상 ID를 대표 청크 1개로 치환한 것이라
    # 청크 단위 판정은 라벨 입도를 측정한다. 근거는 eval/metrics/l2_retrieval.py.
    "context_recall_doc_min": 0.80,
    "hits_at_k_min": 0.80,  # For requires_kg=True
    # L3 엣지 게이트는 recall 기준 (2026-08-30 사이클 4):
    # set-F1은 골드(1~3개)와 방출(최대 12개)의 규모 비대칭 때문에 100% 회수해도
    # ~0.18에 그쳐 판별력이 없었다. 상세 근거는 eval/metrics/l3_kg.py 참조.
    "kg_edge_recall_min": 0.50,
    # L1 gating
    "entity_link_f1_min": 0.50,
    # 게이트 재보정 (2026-08-30, 오염 수정 후 v4.0 분포 기준):
    # concept F1 중앙값 0.40 — 0.5는 상위 문항까지 fail 처리해 판별력이 없었음.
    # 0.3은 하위 40%를 플래그하는 판별 임계.
    "concept_map_f1_min": 0.30,
}

# Fail reason taxonomy
FAIL_REASONS = {
    "L1_mapping_fail": "Entity linking F1 below threshold",
    "L1_concept_fail": "Concept mapping F1 below threshold",
    "L2_doc_retrieval_fail": "Document-level context recall below threshold",
    "L3_kg_fail": "KG hits@k below threshold",
    "L3_edge_fail": "KG edge recall below threshold",
    "L4_constraint_violation": "Constraint violation rate above threshold",
    "L4_type_inconsistency": "Type consistency rate below threshold",
    "L5_grounding_fail": "Groundedness score below threshold",
    "L5_wrong_answer": "Answer F1 below threshold",
    "L5_relevance_fail": "Relevance score below threshold",
}


class MetricAggregator:
    """
    Aggregates L1-L5 metrics into overall scores.

    Handles:
    - Weighted score computation
    - Gating threshold checks
    - Pass/fail determination
    - Fail reason tagging
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        thresholds: dict[str, float] | None = None,
    ):
        """
        Initialize aggregator.

        Args:
            weights: Override default weights
            thresholds: Override default thresholds
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()

    def compute_overall_score(
        self,
        l1: L1Metrics,
        l2: L2Metrics,
        l3: L3Metrics,
        l4: L4Metrics,
        l5: L5Metrics,
        metadata: ItemMetadata | None = None,
    ) -> float:
        """
        Compute weighted overall score.

        Args:
            l1: L1 metrics
            l2: L2 metrics
            l3: L3 metrics
            l4: L4 metrics
            l5: L5 metrics
            metadata: Item metadata (for requires_kg flag)

        Returns:
            Overall score (0.0-1.0)
        """
        # L1 score: average of entity, concept, constraint F1
        l1_score = (l1.entity_link_f1 + l1.concept_map_f1 + l1.constraint_extraction_f1) / 3

        # L2/L3 score: depends on requires_kg
        requires_kg = metadata.requires_kg if metadata else True

        if requires_kg:
            # Weight KG metrics more
            l2_l3_score = (
                l2.context_recall_at_k * 0.3
                + l2.mrr * 0.1
                + l3.hits_at_k * 0.4
                + l3.kg_edge_f1 * 0.2
            )
        else:
            # Weight document retrieval more
            l2_l3_score = (
                l2.context_recall_at_k * 0.5 + l2.context_precision_at_k * 0.2 + l2.mrr * 0.3
            )

        # L4 score: penalize violations
        l4_score = (1.0 - l4.constraint_violation_rate) * 0.5 + l4.type_consistency_rate * 0.5

        # L5 score: answer quality
        l5_components = [l5.answer_f1]
        if l5.groundedness_score is not None:
            l5_components.append(l5.groundedness_score)
        if l5.answer_relevance_score is not None:
            l5_components.append(l5.answer_relevance_score)
        if l5.semantic_similarity is not None:
            l5_components.append(l5.semantic_similarity)
        if l5.factuality_score is not None:
            l5_components.append(l5.factuality_score)

        l5_score = sum(l5_components) / len(l5_components)

        # Weighted combination
        overall = (
            self.weights["l1"] * l1_score
            + self.weights["l2_l3"] * l2_l3_score
            + self.weights["l4"] * l4_score
            + self.weights["l5"] * l5_score
        )

        return min(1.0, max(0.0, overall))

    def check_gating(
        self,
        l1: L1Metrics,
        l2: L2Metrics,
        l3: L3Metrics,
        l4: L4Metrics,
        l5: L5Metrics,
        metadata: ItemMetadata | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Check gating thresholds.

        Args:
            l1: L1 metrics
            l2: L2 metrics
            l3: L3 metrics
            l4: L4 metrics
            l5: L5 metrics
            metadata: Item metadata

        Returns:
            Tuple of (passed, list of fail_reason_tags)
        """
        fail_reasons: list[str] = []
        requires_kg = metadata.requires_kg if metadata else True

        # L1 checks
        if l1.entity_link_f1 < self.thresholds.get("entity_link_f1_min", 0.5):
            fail_reasons.append("L1_mapping_fail")

        if l1.concept_map_f1 < self.thresholds.get("concept_map_f1_min", 0.5):
            fail_reasons.append("L1_concept_fail")

        # L2 checks (for non-KG queries)
        if not requires_kg:
            if l2.context_recall_at_k_doc < self.thresholds.get("context_recall_doc_min", 0.8):
                fail_reasons.append("L2_doc_retrieval_fail")

        # L3 checks (for KG queries)
        if requires_kg:
            if l3.hits_at_k < self.thresholds.get("hits_at_k_min", 0.8):
                fail_reasons.append("L3_kg_fail")

            if l3.kg_edge_recall < self.thresholds.get("kg_edge_recall_min", 0.5):
                fail_reasons.append("L3_edge_fail")

        # L4 checks
        if l4.constraint_violation_rate > self.thresholds.get("constraint_violation_max", 0.05):
            fail_reasons.append("L4_constraint_violation")

        if l4.type_consistency_rate < self.thresholds.get("type_consistency_min", 0.9):
            fail_reasons.append("L4_type_inconsistency")

        # L5 checks — token-F1 미달이어도 semantic similarity가 충분하면 정답 인정
        answer_ok = l5.answer_f1 >= self.thresholds.get("answer_f1_min", 0.5)
        if not answer_ok and l5.semantic_similarity is not None:
            answer_ok = l5.semantic_similarity >= self.thresholds.get(
                "semantic_similarity_min", 0.65
            )
        if not answer_ok:
            fail_reasons.append("L5_wrong_answer")

        if l5.groundedness_score is not None:
            if l5.groundedness_score < self.thresholds.get("groundedness_min", 0.7):
                fail_reasons.append("L5_grounding_fail")

        if l5.answer_relevance_score is not None:
            if l5.answer_relevance_score < self.thresholds.get("relevance_min", 0.7):
                fail_reasons.append("L5_relevance_fail")

        passed = len(fail_reasons) == 0
        return passed, fail_reasons

    def aggregate(
        self,
        item_id: str,
        l1: L1Metrics,
        l2: L2Metrics,
        l3: L3Metrics,
        l4: L4Metrics,
        l5: L5Metrics,
        trace: "EvalTrace",  # type: ignore  # noqa: F821
        metadata: ItemMetadata | None = None,
    ) -> ItemResult:
        """
        Aggregate all metrics into ItemResult.

        Args:
            item_id: Evaluation item ID
            l1-l5: Layer metrics
            trace: Full evaluation trace
            metadata: Item metadata

        Returns:
            ItemResult with all metrics, overall score, and pass/fail
        """
        overall_score = self.compute_overall_score(l1, l2, l3, l4, l5, metadata)
        passed, fail_reasons = self.check_gating(l1, l2, l3, l4, l5, metadata)

        return ItemResult(
            item_id=item_id,
            passed=passed,
            l1=l1,
            l2=l2,
            l3=l3,
            l4=l4,
            l5=l5,
            overall_score=overall_score,
            fail_reason_tags=fail_reasons,
            trace=trace,
        )


def compute_overall_score(
    l1: L1Metrics,
    l2: L2Metrics,
    l3: L3Metrics,
    l4: L4Metrics,
    l5: L5Metrics,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Convenience function for overall score.

    Args:
        l1-l5: Layer metrics
        weights: Override default weights

    Returns:
        Overall score (0.0-1.0)
    """
    aggregator = MetricAggregator(weights=weights)
    return aggregator.compute_overall_score(l1, l2, l3, l4, l5)


def check_gating(
    l1: L1Metrics,
    l2: L2Metrics,
    l3: L3Metrics,
    l4: L4Metrics,
    l5: L5Metrics,
    thresholds: dict[str, float] | None = None,
    metadata: ItemMetadata | None = None,
) -> tuple[bool, list[str]]:
    """
    Convenience function for gating check.

    Args:
        l1-l5: Layer metrics
        thresholds: Override default thresholds
        metadata: Item metadata

    Returns:
        Tuple of (passed, fail_reason_tags)
    """
    aggregator = MetricAggregator(thresholds=thresholds)
    return aggregator.check_gating(l1, l2, l3, l4, l5, metadata)


def get_fail_reason_description(tag: str) -> str:
    """
    Get human-readable description of fail reason.

    Args:
        tag: Fail reason tag

    Returns:
        Description string
    """
    return FAIL_REASONS.get(tag, f"Unknown fail reason: {tag}")
