"""
Evaluation Schemas
==================
Pydantic models for dataset items, traces, and metrics.

Layers:
- L1: Query interpretation (entity linking, concept mapping, constraint extraction)
- L2: Document retrieval quality
- L3: Knowledge Graph retrieval/traversal quality
- L4: Ontology constraint compliance
- L5: Final answer quality (groundedness, relevance, correctness)
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# Dataset Schemas
# =============================================================================


class GoldEvidence(BaseModel):
    """
    Gold standard evidence for evaluation.

    Not all fields are required - use what's available for your test cases.
    """

    answer: str | None = Field(
        default=None, description="Expected answer text (for exact/F1 matching)"
    )
    doc_chunk_ids: list[str] = Field(
        default_factory=list, description="Expected document chunk IDs for retrieval"
    )
    kg_entities: list[str] = Field(
        default_factory=list,
        description="Expected KG entities (brands, categories, products)",
    )
    kg_edges: list[str] = Field(
        default_factory=list,
        description="Expected KG edges in format 'subject -predicate-> object'",
    )
    concepts: list[str] = Field(
        default_factory=list,
        description="Expected ontology concepts/categories",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Expected ontology rules/constraints to be applied",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "LANEIGE의 Lip Care 카테고리 SoS는 5.2%입니다.",
                "doc_chunk_ids": ["metric_guide_sos_01", "playbook_ranking_02"],
                "kg_entities": ["laneige", "lip_care"],
                "kg_edges": ["laneige -hasProduct-> B08XYZ123"],
                "concepts": ["lip_care", "market_position"],
                "constraints": ["low_sos_warning", "competitive_threat"],
            }
        }
    }


class ItemMetadata(BaseModel):
    """Metadata for evaluation item to guide metric calculation."""

    requires_kg: bool = Field(default=True, description="Whether this query requires KG lookup")
    domain: Literal["market", "brand", "product", "metric", "general"] = Field(
        default="general", description="Domain category of the query"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium", description="Difficulty level for stratified analysis"
    )


class EvalItem(BaseModel):
    """
    Single evaluation item (test case).

    Follows the spec:
    - id: unique identifier
    - question: user query
    - gold: expected evidence/answers
    - metadata: evaluation configuration
    """

    id: str = Field(..., description="Unique identifier for the test case")
    question: str = Field(..., description="User query to evaluate")
    gold: GoldEvidence = Field(default_factory=GoldEvidence, description="Gold standard evidence")
    metadata: ItemMetadata = Field(default_factory=ItemMetadata, description="Evaluation metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "q001",
                "question": "LANEIGE Lip Care SoS는?",
                "gold": {
                    "answer": None,
                    "doc_chunk_ids": [],
                    "kg_entities": ["laneige", "lip_care"],
                    "kg_edges": ["laneige -hasProduct-> B08XYZ"],
                    "concepts": ["lip_care"],
                    "constraints": [],
                },
                "metadata": {
                    "requires_kg": True,
                    "domain": "metric",
                    "difficulty": "easy",
                },
            }
        }
    }


# =============================================================================
# Trace Schemas (Intermediate Artifacts)
# =============================================================================


class EntityLinkingTrace(BaseModel):
    """L1: Entity linking/extraction results."""

    extracted_brands: list[str] = Field(
        default_factory=list, description="Brands extracted from query"
    )
    extracted_categories: list[str] = Field(
        default_factory=list, description="Categories extracted from query"
    )
    extracted_indicators: list[str] = Field(
        default_factory=list, description="Metrics/indicators extracted (sos, hhi, cpi)"
    )
    extracted_products: list[str] = Field(
        default_factory=list, description="Product ASINs extracted"
    )
    extracted_sentiments: list[str] = Field(
        default_factory=list, description="Sentiment keywords extracted"
    )
    time_range: list[str] = Field(default_factory=list, description="Time ranges extracted")


class DocRetrievalTrace(BaseModel):
    """L2: Document retrieval results."""

    chunk_ids: list[str] = Field(default_factory=list, description="IDs of retrieved chunks")
    snippets: list[str] = Field(
        default_factory=list, description="Content snippets of retrieved chunks"
    )
    scores: list[float] = Field(
        default_factory=list, description="Relevance scores of retrieved chunks"
    )
    doc_types: list[str] = Field(
        default_factory=list,
        description="Document types (metric_guide, playbook, intelligence)",
    )


class KGQueryTrace(BaseModel):
    """L3: Knowledge Graph query results."""

    kg_entities_found: list[str] = Field(default_factory=list, description="Entities found in KG")
    kg_edges_found: list[str] = Field(
        default_factory=list,
        description="Edges found (format: 'subj -pred-> obj')",
    )
    ontology_facts: list[dict[str, Any]] = Field(
        default_factory=list, description="Raw ontology facts from KG"
    )
    competitor_network: list[dict[str, Any]] = Field(
        default_factory=list, description="Competitor relationships found"
    )
    category_hierarchy: list[dict[str, Any]] = Field(
        default_factory=list, description="Category hierarchy info"
    )


class OntologyReasoningTrace(BaseModel):
    """L4: Ontology reasoning results."""

    inferences: list[dict[str, Any]] = Field(
        default_factory=list, description="Generated inferences"
    )
    applied_rules: list[str] = Field(
        default_factory=list, description="Names of applied inference rules"
    )
    insight_types: list[str] = Field(
        default_factory=list, description="Types of generated insights"
    )
    constraint_violations: list[str] = Field(
        default_factory=list, description="Detected constraint violations"
    )


class AnswerTrace(BaseModel):
    """L5: Final answer generation results."""

    final_answer: str = Field(default="", description="Generated answer text")
    citations: list[str] = Field(default_factory=list, description="Source citations in answer")
    confidence: float | None = Field(default=None, description="Answer confidence score")
    query_type: str = Field(default="unknown", description="Detected query type")
    was_rewritten: bool = Field(default=False, description="Whether query was rewritten")
    rewritten_query: str | None = Field(default=None, description="Rewritten query if applicable")


class CostTrace(BaseModel):
    """Cost tracking for evaluation run."""

    # Token counts per layer
    l1_tokens: int = Field(default=0, description="Tokens used in L1 (entity extraction)")
    l2_tokens: int = Field(default=0, description="Tokens used in L2 (embedding)")
    l3_tokens: int = Field(default=0, description="Tokens used in L3 (KG queries)")
    l4_tokens: int = Field(default=0, description="Tokens used in L4 (reasoning)")
    l5_tokens: int = Field(default=0, description="Tokens used in L5 (answer generation)")
    judge_tokens: int = Field(default=0, description="Tokens used in judge scoring")

    # Cost estimates (USD)
    l1_cost_usd: float = Field(default=0.0, description="Cost for L1 in USD")
    l2_cost_usd: float = Field(default=0.0, description="Cost for L2 in USD")
    l3_cost_usd: float = Field(default=0.0, description="Cost for L3 in USD")
    l4_cost_usd: float = Field(default=0.0, description="Cost for L4 in USD")
    l5_cost_usd: float = Field(default=0.0, description="Cost for L5 in USD")
    judge_cost_usd: float = Field(default=0.0, description="Cost for judge in USD")

    @property
    def total_tokens(self) -> int:
        """Total tokens across all layers."""
        return (
            self.l1_tokens
            + self.l2_tokens
            + self.l3_tokens
            + self.l4_tokens
            + self.l5_tokens
            + self.judge_tokens
        )

    @property
    def total_cost_usd(self) -> float:
        """Total cost in USD."""
        return (
            self.l1_cost_usd
            + self.l2_cost_usd
            + self.l3_cost_usd
            + self.l4_cost_usd
            + self.l5_cost_usd
            + self.judge_cost_usd
        )


class EvalTrace(BaseModel):
    """Complete evaluation trace for a single item."""

    item_id: str = Field(..., description="ID of the evaluated item")
    timestamp: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")
    l1_entity_linking: EntityLinkingTrace = Field(default_factory=EntityLinkingTrace)
    l2_doc_retrieval: DocRetrievalTrace = Field(default_factory=DocRetrievalTrace)
    l3_kg_query: KGQueryTrace = Field(default_factory=KGQueryTrace)
    l4_ontology: OntologyReasoningTrace = Field(default_factory=OntologyReasoningTrace)
    l5_answer: AnswerTrace = Field(default_factory=AnswerTrace)
    cost: CostTrace = Field(default_factory=CostTrace, description="Cost tracking")
    latency_ms: float = Field(default=0.0, description="Total latency in milliseconds")
    error: str | None = Field(default=None, description="Error message if any")


# =============================================================================
# Metrics Schemas
# =============================================================================


class L1Metrics(BaseModel):
    """L1: Query interpretation metrics."""

    entity_link_f1: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Set-F1 for entity linking"
    )
    concept_map_f1: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Set-F1 for concept mapping"
    )
    constraint_extraction_f1: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Set-F1 for constraint extraction",
    )


class L2Metrics(BaseModel):
    """L2: Document retrieval metrics."""

    context_recall_at_k: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Recall of gold docs in top-k"
    )
    context_precision_at_k: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Precision of top-k retrieval"
    )
    mrr: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean Reciprocal Rank")


class L3Metrics(BaseModel):
    """L3: Knowledge Graph retrieval metrics."""

    hits_at_k: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Gold entities in top-k KG results"
    )
    kg_edge_f1: float = Field(default=0.0, ge=0.0, le=1.0, description="Set-F1 for KG edges")


class L4Metrics(BaseModel):
    """L4: Ontology constraint compliance metrics."""

    constraint_violation_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Violations / total constraints checked",
    )
    type_consistency_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="1 - inconsistent_types / total_entities",
    )


class L5Metrics(BaseModel):
    """L5: Final answer quality metrics."""

    answer_exact_match: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Exact match (normalized)"
    )
    answer_f1: float = Field(default=0.0, ge=0.0, le=1.0, description="Token-level F1")
    semantic_similarity: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Semantic similarity score (0-1)"
    )
    groundedness_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Judge-based groundedness (0-1)"
    )
    answer_relevance_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Judge-based relevance (0-1)"
    )


# =============================================================================
# Result Schemas
# =============================================================================


class ItemResult(BaseModel):
    """Complete evaluation result for a single item."""

    item_id: str = Field(..., description="ID of the evaluated item")
    question: str = Field(default="", description="Original question")
    passed: bool = Field(default=False, description="Whether item passed all gates")
    l1: L1Metrics = Field(default_factory=L1Metrics)
    l2: L2Metrics = Field(default_factory=L2Metrics)
    l3: L3Metrics = Field(default_factory=L3Metrics)
    l4: L4Metrics = Field(default_factory=L4Metrics)
    l5: L5Metrics = Field(default_factory=L5Metrics)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Weighted overall score")
    fail_reason_tags: list[str] = Field(default_factory=list, description="Failure reason tags")
    trace: EvalTrace | None = Field(default=None, description="Full evaluation trace")
    metadata: ItemMetadata = Field(default_factory=ItemMetadata)


class AggregateMetrics(BaseModel):
    """Aggregate metrics across all evaluated items."""

    total: int = Field(default=0, description="Total items evaluated")
    passed: int = Field(default=0, description="Items that passed all gates")
    failed: int = Field(default=0, description="Items that failed")
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_latency_ms: float = Field(default=0.0, ge=0.0)

    # Cost tracking aggregates
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_cost_usd: float = Field(default=0.0, description="Total cost in USD")
    avg_tokens_per_item: float = Field(default=0.0, description="Average tokens per item")
    avg_cost_per_item_usd: float = Field(default=0.0, description="Average cost per item in USD")
    cost_by_layer: dict[str, float] = Field(
        default_factory=dict, description="Cost breakdown by layer (USD)"
    )

    # By layer averages (flexible dict format for report generator)
    by_layer: dict[str, float] = Field(
        default_factory=dict, description="Per-layer metric averages"
    )

    # Failure breakdown
    top_fail_reasons: dict[str, int] = Field(
        default_factory=dict, description="Count by fail_reason_tag"
    )

    # By difficulty
    by_difficulty: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Metrics by difficulty level"
    )

    # By domain
    by_domain: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Metrics by domain"
    )


class EvalConfig(BaseModel):
    """Configuration for evaluation run."""

    top_k: int = Field(default=8, description="Top-k for retrieval metrics")
    use_judge: bool = Field(default=False, description="Whether to use LLM judge")
    judge_model: str = Field(default="gpt-4.1-mini", description="Judge model name")
    save_traces: bool = Field(default=False, description="Save individual traces to files")
    weights: dict[str, float] = Field(
        default_factory=lambda: {"l5": 0.45, "l2_l3": 0.35, "l1": 0.10, "l4": 0.10},
        description="Layer weights for overall score",
    )
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "groundedness_min": 0.70,
            "constraint_violation_max": 0.05,
            "context_recall_min": 0.80,
            "hits_at_k_min": 0.80,
            "entity_link_min": 0.50,
            "answer_f1_min": 0.50,
        },
        description="Gating thresholds",
    )


class EvalReport(BaseModel):
    """Complete evaluation report."""

    timestamp: datetime = Field(default_factory=datetime.now)
    config: EvalConfig = Field(default_factory=EvalConfig)
    aggregates: AggregateMetrics = Field(default_factory=AggregateMetrics)
    items: list[ItemResult] = Field(default_factory=list)
    recommendations: list[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )
