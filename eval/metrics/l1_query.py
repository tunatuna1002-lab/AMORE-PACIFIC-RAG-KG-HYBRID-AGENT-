"""
L1 Query Metrics
================
Entity linking, concept mapping, and constraint extraction metrics.

Measures how well the system understands the query:
- Entity link F1: Extracted entities vs gold entities
- Concept map F1: Extracted categories/concepts vs gold concepts
- Constraint extraction F1: Applied rules vs gold constraints

Supports both exact matching and fuzzy matching with alias resolution:
- Exact: "laneige" matches "laneige" only
- Fuzzy: "라네즈" matches "laneige" via alias map
"""

from eval.metrics.base import (
    BRAND_ALIASES,
    CATEGORY_ALIASES,
    MetricCalculator,
)
from eval.schemas import EntityLinkingTrace, GoldEvidence, L1Metrics, OntologyReasoningTrace


def normalize_constraints(constraints: list) -> set[str]:
    """골드 constraint를 비교 가능한 문자열 집합으로 정규화.

    골든셋 스키마(`GoldEvidence.constraints`)는 `list[str | dict]`을 허용하는데
    dict 형태(`{"field": "period", "operator": "=", "value": "last_3_months"}`)가
    그대로 `set()`에 들어가 TypeError로 **문항 전체 채점이 크래시**하고 있었다
    (v4.1 기준 160문항 중 7문항, time 도메인 15문항 중 6문항). dict는
    `field:operator:value` 형태로 직렬화한다 — 2026-08-30 사이클 4 수정.
    """
    normalized: set[str] = set()
    for c in constraints or []:
        if isinstance(c, dict):
            field = str(c.get("field", "")).strip().lower()
            operator = str(c.get("operator", c.get("op", "="))).strip()
            value = str(c.get("value", "")).strip().lower()
            normalized.add(f"{field}:{operator}:{value}")
        else:
            normalized.add(str(c).strip().lower())
    return normalized


class L1QueryMetrics(MetricCalculator):
    """
    L1 metrics for query understanding.

    Evaluates:
    - Entity extraction (brands, products)
    - Concept/category mapping
    - Constraint identification

    Supports fuzzy matching for cross-language entity aliases.
    """

    def __init__(self, use_fuzzy: bool = False, fuzzy_threshold: float = 0.8):
        """
        Initialize L1 metrics calculator.

        Args:
            use_fuzzy: Enable fuzzy matching with alias resolution
            fuzzy_threshold: Minimum similarity for fuzzy matches (default 0.8)
        """
        self.use_fuzzy = use_fuzzy
        self.fuzzy_threshold = fuzzy_threshold

    def compute(
        self,
        entity_trace: EntityLinkingTrace,
        ontology_trace: OntologyReasoningTrace,
        gold: GoldEvidence,
    ) -> L1Metrics:
        """
        Compute L1 metrics.

        Args:
            entity_trace: Entity linking trace from evaluation
            ontology_trace: Ontology reasoning trace for constraints
            gold: Gold standard evidence

        Returns:
            L1Metrics with entity_link_f1, concept_map_f1, constraint_extraction_f1
        """
        entity_f1 = self._compute_entity_link_f1(entity_trace, gold)
        concept_f1 = self._compute_concept_map_f1(entity_trace, gold)
        constraint_f1 = self._compute_constraint_extraction_f1(ontology_trace, gold)

        return L1Metrics(
            entity_link_f1=entity_f1,
            concept_map_f1=concept_f1,
            constraint_extraction_f1=constraint_f1,
        )

    def _compute_entity_link_f1(self, trace: EntityLinkingTrace, gold: GoldEvidence) -> float:
        """
        Compute entity linking F1.

        Combines brands, products, indicators, and categories into a single
        entity set. 카테고리를 포함하는 이유: 골드 `kg_entities`가 브랜드·제품·
        지표뿐 아니라 카테고리 ID(lip_care, face_powder 등)도 함께 열거한다
        (160문항 중 93문항, 엔티티 언급 518건 중 104건). 추출기는 카테고리를
        별도 필드로 내보내므로 이를 제외하면 해당 골드 항목은 구조적으로
        매칭 불가였다 — 필드 매핑 결함 수정 (2026-08-30 사이클 4).
        Uses fuzzy matching with alias resolution if enabled.
        """
        # Collect all extracted entities
        extracted = set()
        extracted.update(trace.extracted_brands)
        extracted.update(trace.extracted_products)
        extracted.update(trace.extracted_indicators)
        extracted.update(trace.extracted_categories)

        # Gold entities
        gold_entities = set(gold.kg_entities)

        if self.use_fuzzy:
            return self.set_f1_fuzzy(
                extracted, gold_entities, threshold=self.fuzzy_threshold, alias_map=BRAND_ALIASES
            )
        return self.set_f1(extracted, gold_entities)

    def _compute_concept_map_f1(self, trace: EntityLinkingTrace, gold: GoldEvidence) -> float:
        """
        Compute concept mapping F1.

        Combines extracted categories + indicators to match gold concepts.
        Gold concepts include both metric types (sos, hhi) and query types
        (definition, data_query), so we union categories and indicators.
        Uses fuzzy matching with alias resolution if enabled.
        """
        # 카테고리 + 지표 + 감성 + 분석 개념을 합쳐 개념 집합 구성
        extracted_concepts = (
            set(trace.extracted_categories)
            | set(trace.extracted_indicators)
            | set(trace.extracted_sentiments)
            | set(trace.extracted_concepts)
        )
        gold_concepts = set(gold.concepts)

        if self.use_fuzzy:
            return self.set_f1_fuzzy(
                extracted_concepts,
                gold_concepts,
                threshold=self.fuzzy_threshold,
                alias_map=CATEGORY_ALIASES,
            )
        return self.set_f1(extracted_concepts, gold_concepts)

    def _compute_constraint_extraction_f1(
        self, trace: OntologyReasoningTrace, gold: GoldEvidence
    ) -> float:
        """
        Compute constraint extraction F1.

        Compares applied rules to gold constraints.
        """
        applied_rules = set(trace.applied_rules)
        gold_constraints = normalize_constraints(gold.constraints)

        return self.set_f1(applied_rules, gold_constraints)


def entity_link_f1(
    trace: EntityLinkingTrace,
    gold: GoldEvidence,
    use_fuzzy: bool = False,
    fuzzy_threshold: float = 0.8,
) -> float:
    """
    Convenience function for entity linking F1.

    Args:
        trace: Entity linking trace
        gold: Gold evidence
        use_fuzzy: Enable fuzzy matching with alias resolution
        fuzzy_threshold: Minimum similarity for fuzzy matches

    Returns:
        F1 score between extracted entities and gold.kg_entities
    """
    calc = L1QueryMetrics(use_fuzzy=use_fuzzy, fuzzy_threshold=fuzzy_threshold)
    return calc._compute_entity_link_f1(trace, gold)


def concept_map_f1(
    trace: EntityLinkingTrace,
    gold: GoldEvidence,
    use_fuzzy: bool = False,
    fuzzy_threshold: float = 0.8,
) -> float:
    """
    Convenience function for concept mapping F1.

    Args:
        trace: Entity linking trace
        gold: Gold evidence
        use_fuzzy: Enable fuzzy matching with alias resolution
        fuzzy_threshold: Minimum similarity for fuzzy matches

    Returns:
        F1 score between extracted categories and gold.concepts
    """
    calc = L1QueryMetrics(use_fuzzy=use_fuzzy, fuzzy_threshold=fuzzy_threshold)
    return calc._compute_concept_map_f1(trace, gold)


def constraint_extraction_f1(trace: OntologyReasoningTrace, gold: GoldEvidence) -> float:
    """
    Convenience function for constraint extraction F1.

    Args:
        trace: Ontology reasoning trace
        gold: Gold evidence

    Returns:
        F1 score between applied rules and gold.constraints
    """
    extracted = set(trace.applied_rules)
    gold_set = normalize_constraints(gold.constraints)
    return MetricCalculator.set_f1(extracted, gold_set)
