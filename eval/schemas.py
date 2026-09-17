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
        default_factory=list,
        description="Expected document chunk IDs (doc_chunk_groups의 평면 합집합)",
    )
    doc_chunk_groups: list[list[str]] = Field(
        default_factory=list,
        description=(
            "개념(근거 단위)별 청크 ID 집합. 한 개념의 근거는 문단 하나가 아니라 "
            "그 개념이 서술된 절 전체이므로, 집합 중 하나라도 검색되면 그 개념을 "
            "찾은 것으로 본다 (scripts/remap_golden_chunk_groups.py)."
        ),
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
    constraints: list[str | dict] = Field(
        default_factory=list,
        description="Expected ontology rules/constraints to be applied",
    )
    expected_values: dict[str, float] = Field(
        default_factory=dict,
        description="Expected numerical KPI values for accuracy checking",
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
    domain: Literal[
        "market", "brand", "product", "metric", "general", "multi_hop", "edge", "time", "ir"
    ] = Field(default="general", description="Domain category of the query")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium", description="Difficulty level for stratified analysis"
    )
    # 골드 수치의 출처. 검증 가능한 근거가 무엇이냐에 따라 채점 방식이 갈린다
    # (scripts/classify_golden_sources.py, 2026-09-06):
    #   document          — 코퍼스 문서에서 나오고 시간에 따라 변하지 않는 답
    #   snapshot          — 크롤 DB의 특정 시점 수치 (as_of 필수)
    #   domain_expectation— DB에도 문서에도 없는 도메인 추정치
    # 원자료로 검증할 수 없는 골드에 정답 일치를 요구하면 지표가 문체 유사도를 잰다.
    gold_source: Literal["document", "snapshot", "domain_expectation"] = Field(
        default="document", description="골드 수치의 검증 근거"
    )
    as_of: str | None = Field(default=None, description="snapshot 문항의 기준 시점 (YYYY-MM-DD)")
    # 유형별 시험지·규칙 관측 관련 필드 (트랙 3-C, 2026-09-17). 로더가 기본적으로
    # 버리던 키들 중 리포트만으로 분석하는 데 필요한 것만 보존한다. 없으면 None —
    # 구형 골든셋/리포트와 하위 호환.
    question_type: str | None = Field(
        default=None, description="유형별 시험지 분류 (numeric/relation/rule/multihop 등)"
    )
    generated: bool | None = Field(default=None, description="생성 문항 여부 (스크립트 생성)")
    rule_gold: dict[str, Any] | None = Field(
        default=None,
        description=(
            "rule 유형 문항의 규칙 정답: {'rule_ids': [...], 'expected_conclusion': "
            "{'fires': bool, ...}, ...} (scripts/generate_rule_questions.py). "
            "리포트의 ItemResult.rule_agreement 계산에 쓴다. 없으면 규칙 정답 판정 불가."
        ),
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
    extracted_concepts: list[str] = Field(
        default_factory=list, description="Analytic concepts extracted (sos, time_series, ...)"
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

    # Granular per-layer token breakdown (prompt/completion for LLM layers,
    # embedding for L2). Added 2026-09 (F4 cost pricing fix). All default to 0
    # so older report.json/baseline files without these fields still load.
    l1_prompt_tokens: int = Field(default=0, description="L1 prompt tokens")
    l1_completion_tokens: int = Field(default=0, description="L1 completion tokens")
    l2_embedding_tokens: int = Field(default=0, description="L2 embedding tokens")
    l3_prompt_tokens: int = Field(default=0, description="L3 prompt tokens")
    l3_completion_tokens: int = Field(default=0, description="L3 completion tokens")
    l4_prompt_tokens: int = Field(default=0, description="L4 prompt tokens")
    l4_completion_tokens: int = Field(default=0, description="L4 completion tokens")
    l5_prompt_tokens: int = Field(default=0, description="L5 prompt tokens")
    l5_completion_tokens: int = Field(default=0, description="L5 completion tokens")
    judge_prompt_tokens: int = Field(default=0, description="Judge prompt tokens")
    judge_completion_tokens: int = Field(default=0, description="Judge completion tokens")

    # Cost estimates (USD)
    l1_cost_usd: float = Field(default=0.0, description="Cost for L1 in USD")
    l2_cost_usd: float = Field(default=0.0, description="Cost for L2 in USD")
    l3_cost_usd: float = Field(default=0.0, description="Cost for L3 in USD")
    l4_cost_usd: float = Field(default=0.0, description="Cost for L4 in USD")
    l5_cost_usd: float = Field(default=0.0, description="Cost for L5 in USD")
    judge_cost_usd: float = Field(default=0.0, description="Cost for judge in USD")

    # Unit prices actually used to compute the costs above, keyed by model name,
    # e.g. {"gpt-4.1-mini": {"input_per_1m_usd": 0.40, "output_per_1m_usd": 1.60,
    # "source": "litellm" | "fallback_table"}}. Added 2026-09 (F4). Defaults to
    # {} so older report.json/baseline files without this field still load.
    pricing: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Unit prices used per model, with their source"
    )

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
    data_facts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="검색이 컨텍스트에 실은 크롤 DB 수치 사실 (judge 근거성 컨텍스트에 포함)",
    )
    cost: CostTrace = Field(default_factory=CostTrace, description="Cost tracking")
    latency_ms: float = Field(default=0.0, description="Total latency in milliseconds")
    error: str | None = Field(default=None, description="Error message if any")
    retrieval_error: str | None = Field(
        default=None,
        description=(
            "핵심 검색 실패 원인 (HybridContext.metadata['retrieval_error'] 또는 "
            "V4RetrievalTrace 동등 필드). 채워지면 run_item이 인프라 실패로 분리한다."
        ),
    )
    degraded: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "선택 기능(비핵심) 실패 목록: [{'component':..., 'error':...}, ...]. "
            "채점은 계속하되 어떤 하위 조회가 저하됐는지 노출한다."
        ),
    )
    route_trace: dict[str, Any] | None = Field(
        default=None,
        description=(
            "문항별 질의 경로 관측 (QueryGraph._finalize_route_trace, 커밋 31040bf). "
            "route/confidence_level/confidence_score/confidence_components/tools_used/"
            "decision_tool/is_complex를 담는다. v1(HybridChatbotAgent) 경로나 구형 "
            "report.json에는 없으므로 기본값 None으로 하위 호환한다."
        ),
    )
    evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "답변 프롬프트에 실제로 실린 증거 카드(prompt_evidence)를 "
            "Evidence.model_dump(mode='json')한 목록 — judge 컨텍스트를 그대로 재구성하는 데 "
            "쓴다 (트랙 2-C). 검색기가 evidence/prompt_evidence 속성을 아직 주지 않는 "
            "경로(2-B 미병합, v1 구형)나 구형 report.json에는 없으므로 기본값 빈 리스트로 "
            "하위 호환한다 — 그때는 judge_context_source가 'legacy'가 된다."
        ),
    )
    evidence_all_count: int = Field(
        default=0,
        description=(
            "검색이 만든 전체 증거 카드 수(prompt_evidence로 선별되기 전). evidence(선별 후) "
            "길이와 비교하면 judge가 보지 못한 채 버려진 근거가 얼마나 되는지 드러난다."
        ),
    )
    judge_context_source: str = Field(
        default="legacy",
        description=(
            "judge 근거성 컨텍스트를 만든 방식. 'evidence' = evidence 필드의 카드를 "
            "render_for_judge로 렌더 (답변 프롬프트와 같은 집합·같은 내용). "
            "'legacy' = 구형 로직(문서 스니펫 + KG 사실 + data_facts 전부) — 카드가 없는 "
            "경로에서만 쓴다."
        ),
    )
    rule_evaluation: dict[str, Any] | None = Field(
        default=None,
        description=(
            "규칙 엔진 추론 관측 (트랙 3-B, HybridContext.metadata['rule_evaluation']을 "
            "그대로 복사): {'combinations': [...], 'evaluated': int, 'fired': [rule_name,...], "
            "'non_fire_top': [[label, count], ...], 'non_fire_counts_by_kind': {...}}. "
            "metadata에 키가 없으면(3-B 미병합, v1 구형, 구형 report.json) None."
        ),
    )


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
    context_recall_at_k_concept: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="개념 단위 recall@k — 근거를 찾은 개념 비율 (게이트 기준)",
    )
    context_recall_at_k_doc: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="출처 문서 단위 recall@k (골드 라벨의 실제 입도, 게이트 기준)",
    )


class L3Metrics(BaseModel):
    """L3: Knowledge Graph retrieval metrics."""

    hits_at_k: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Gold entities in top-k KG results"
    )
    kg_edge_f1: float = Field(default=0.0, ge=0.0, le=1.0, description="Set-F1 for KG edges")
    kg_edge_recall: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Recall of gold KG edges (게이트 기준 지표)"
    )
    kg_edge_precision: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Precision of emitted KG edges (남용 감시용)"
    )


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
    factuality_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Judge-based factuality (0-1)"
    )
    numeric_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "gold.expected_values의 수치를 답변이 맞힌 비율. "
            "expected_values가 없으면 None. gold_source=snapshot 문항에만 게이트로 쓴다."
        ),
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
    rule_agreement: bool | None = Field(
        default=None,
        description=(
            "규칙 정답 일치 여부 (트랙 3-C). metadata.rule_gold가 있고 "
            "rule_ids·expected_conclusion.fires가 모두 있을 때만 계산: "
            "bool(applied_rules ∩ rule_ids) == fires. 판정 불가하면 None "
            "(rule_gold 없음, 구형 리포트 등)."
        ),
    )


class AggregateMetrics(BaseModel):
    """Aggregate metrics across all evaluated items."""

    total: int = Field(default=0, description="Items actually scored (인프라 실패 제외)")
    passed: int = Field(default=0, description="Items that passed all gates")
    failed: int = Field(default=0, description="Items that failed")
    # 인프라 실패(타임아웃·API 오류)는 모델 실패와 같은 열에 섞지 않는다.
    # 이 문항들은 total/passed/failed·평균 지표 어디에도 들어가지 않는다.
    errored: int = Field(default=0, description="답변을 얻지 못해 채점에서 분리된 문항 수")
    error_item_ids: list[str] = Field(default_factory=list, description="채점에서 분리된 문항 ID")
    # 선택 기능(비핵심) 실패는 채점을 막지 않는다 — 몇 문항이 저하된 채로
    # 채점됐는지만 드러낸다 (F3). errored와 달리 total/평균 지표에서 빠지지 않는다.
    degraded_items: int = Field(
        default=0, description="선택 기능이 하나 이상 저하된 채로 채점된 문항 수"
    )
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

    # Route / confidence distribution (trace.route_trace 기반, v4 전용 — 커밋 31040bf).
    # 구형 report.json에는 이 필드들이 없으므로 기본값(빈 dict/0)으로 하위 호환한다.
    route_counts: dict[str, int] = Field(
        default_factory=dict,
        description="문항별 경로(route) 분포: direct/clarify/decide/react/blocked/cache별 개수",
    )
    confidence_level_counts: dict[str, int] = Field(
        default_factory=dict, description="문항별 신뢰도 레벨(confidence_level) 분포"
    )
    react_items: int = Field(
        default=0,
        description="route_trace.route == 'react'로 관측된 문항 수 (route_counts 동일 값)",
    )
    rule_fired_items: int = Field(
        default=0,
        description="trace.l4_ontology.inferences가 비어있지 않은 문항 수 (규칙 추론 발동)",
    )
    rule_inference_total: int = Field(
        default=0, description="채점된 전체 문항의 inferences 총 개수 합"
    )
    # 규칙 정답 일치 관측 (트랙 3-C). rule_gold가 있는 문항(judged)만 대상 —
    # rule_fired_items/rule_inference_total(0-D2, 발화 여부만 봄)과는 다른 지표다.
    rule_agreement_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="rule_agreement 판정 가능한 문항 중 일치 비율. 판정 가능 문항 0이면 None",
    )
    rule_agreement_items: int = Field(
        default=0, description="rule_agreement가 판정된(None이 아닌) 문항 수"
    )
    non_fire_reason_top: list[tuple[str, int]] = Field(
        default_factory=list,
        description=(
            "채점된 문항의 rule_evaluation.non_fire_top을 라벨별로 합산한 상위 15개 "
            "[(label, count), ...] — 예: missing_input:sos, conditions_not_met:hhi_below_0.15"
        ),
    )


class EvalConfig(BaseModel):
    """Configuration for evaluation run."""

    top_k: int = Field(default=8, description="Top-k for retrieval metrics")
    item_timeout_seconds: float = Field(
        default=120.0,
        description=(
            "문항당 에이전트 호출 상한(초). 초과하면 그 문항은 0점이 아니라 "
            "인프라 실패로 분리된다 (trace.error, AggregateMetrics.errored)."
        ),
    )
    data_as_of: str | None = Field(
        default=None,
        description=(
            "시스템이 읽을 크롤 DB 시점 상한(YYYY-MM-DD). 골든셋 snapshot 문항의 as_of에서 "
            "정해진다 — 골드와 같은 날짜의 데이터를 읽어야 수치 비교가 성립한다."
        ),
    )
    target: str = Field(
        default="v1",
        description=(
            "평가 대상 경로: v1 = HybridChatbotAgent(/api/chat), v4 = UnifiedBrain.process_query"
            "(대시보드와 같은 Brain 경로의 비스트림 판). 두 경로의 기준선은 직접 비교하지 않는다."
        ),
    )
    git_commit: str | None = Field(default=None, description="평가 대상 코드의 git HEAD")
    use_judge: bool = Field(default=False, description="Whether to use LLM judge")
    judge_model: str | None = Field(default="gpt-4.1-mini", description="Judge model name")
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


# =============================================================================
# Cost & Regression Schemas
# =============================================================================


class LayerCost(BaseModel):
    """Cost breakdown for a single evaluation layer."""

    tokens: int = Field(default=0, description="Tokens used")
    cost_usd: float = Field(default=0.0, description="Cost in USD")
    avg_latency_ms: float = Field(default=0.0, description="Average latency in ms")


class CostBreakdown(BaseModel):
    """Detailed cost analysis for an evaluation run."""

    run_id: str = Field(..., description="Evaluation run identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    total_tokens: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)
    by_layer: dict[str, LayerCost] = Field(default_factory=dict, description="Cost per layer")
    prev_run_cost_usd: float | None = Field(
        default=None, description="Previous run cost for comparison"
    )

    @property
    def cost_delta_pct(self) -> float | None:
        """Percentage change from previous run."""
        if self.prev_run_cost_usd is None or self.prev_run_cost_usd == 0:
            return None
        return ((self.total_cost_usd - self.prev_run_cost_usd) / self.prev_run_cost_usd) * 100

    def summary(self) -> dict:
        """Generate summary dict."""
        return {
            "run_id": self.run_id,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "cost_delta_pct": self.cost_delta_pct,
            "layers": {
                k: {"tokens": v.tokens, "cost_usd": v.cost_usd} for k, v in self.by_layer.items()
            },
        }


class RegressionItem(BaseModel):
    """Single metric regression or improvement."""

    metric: str = Field(..., description="Metric name")
    baseline_value: float = Field(..., description="Baseline value")
    current_value: float = Field(..., description="Current value")
    delta: float = Field(..., description="Absolute change")
    severity: Literal["low", "medium", "high"] = Field(default="low")


class RegressionComparison(BaseModel):
    """Compare two evaluation runs for regression detection."""

    baseline_run_id: str = Field(...)
    current_run_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    metric_deltas: dict[str, float] = Field(default_factory=dict)
    regressions: list[RegressionItem] = Field(default_factory=list)
    improvements: list[RegressionItem] = Field(default_factory=list)

    def has_regressions(self, threshold: float = 0.05) -> bool:
        """Check if any regression exceeds threshold."""
        return any(abs(r.delta) >= threshold for r in self.regressions)

    def summary(self) -> dict:
        """Generate comparison summary."""
        return {
            "baseline": self.baseline_run_id,
            "current": self.current_run_id,
            "total_regressions": len(self.regressions),
            "total_improvements": len(self.improvements),
            "has_significant_regressions": self.has_regressions(),
            "worst_regression": max((r.delta for r in self.regressions), default=0.0),
        }
