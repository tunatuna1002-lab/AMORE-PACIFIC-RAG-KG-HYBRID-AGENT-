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

        Combines brands, products, and indicators into a single entity set.
        Uses fuzzy matching with alias resolution if enabled.
        """
        # Collect all extracted entities
        extracted = set()
        extracted.update(trace.extracted_brands)
        extracted.update(trace.extracted_products)
        extracted.update(trace.extracted_indicators)

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

        Maps extracted categories to gold concepts.
        Uses fuzzy matching with alias resolution if enabled.
        """
        extracted_concepts = set(trace.extracted_categories)
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
        gold_constraints = set(gold.constraints)

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
    gold_set = set(gold.constraints)
    return MetricCalculator.set_f1(extracted, gold_set)
