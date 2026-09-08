# Evaluation Harness

Offline evaluation pipeline for the AMORE RAG + KG + Ontology Hybrid Agent.

## Overview

This evaluation harness measures the quality of the hybrid chatbot agent across 5 layers:

| Layer | Name | Metrics |
|-------|------|---------|
| L1 | Query Understanding | Entity Link F1, Concept Map F1, Constraint Extraction F1 |
| L2 | Document Retrieval | Context Recall@k, Context Precision@k, MRR |
| L3 | Knowledge Graph | Hits@k, KG Edge F1 |
| L4 | Ontology Compliance | Constraint Violation Rate, Type Consistency Rate |
| L5 | Answer Quality | Exact Match, Token F1, Groundedness*, Relevance* |

*Optional judge-based metrics

## Quick Start

```bash
# Basic evaluation with stub judge
python -m eval.cli --dataset eval/data/examples/chatbot_eval.jsonl

# With LLM judge enabled
python -m eval.cli --dataset data.jsonl --use-judge --judge-model gpt-4.1-mini

# Dry run (validate dataset only)
python -m eval.cli --dataset data.jsonl --dry-run
```

## Dataset Format

Evaluation datasets are JSONL files with the following schema:

```jsonl
{
  "id": "q001",
  "question": "LANEIGE Lip Care SoS는?",
  "gold": {
    "answer": "optional gold answer",
    "doc_chunk_ids": ["chunk_1", "chunk_2"],
    "kg_entities": ["laneige", "lip_care"],
    "kg_edges": ["laneige -hasProduct-> B08XYZ"],
    "concepts": ["lip_care"],
    "constraints": []
  },
  "metadata": {
    "requires_kg": true,
    "domain": "metric",
    "difficulty": "easy"
  }
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique identifier |
| `question` | Yes | User query |
| `gold.answer` | No | Expected answer text (for exact match/F1) |
| `gold.doc_chunk_ids` | No | Expected document chunks |
| `gold.kg_entities` | No | Expected KG entities |
| `gold.kg_edges` | No | Expected KG edges |
| `gold.concepts` | No | Expected concepts/categories |
| `gold.constraints` | No | Expected ontology constraints |
| `metadata.requires_kg` | No | Whether query needs KG (default: true) |
| `metadata.domain` | No | Query domain (market/brand/product/metric/general) |
| `metadata.difficulty` | No | Difficulty level (easy/medium/hard) |

## Output

The evaluation generates two reports:

### 1. `report.json`

Full structured results including:
- Configuration
- Aggregate metrics
- Per-item results with traces

### 2. `summary.md`

Human-readable summary including:
- Pass/fail statistics
- Layer-by-layer metrics
- Top failure reasons
- Recommendations

## Metrics

### L1: Query Understanding

- **Entity Link F1**: Set-F1 between extracted entities and gold entities
- **Concept Map F1**: Set-F1 between extracted categories and gold concepts
- **Constraint Extraction F1**: Set-F1 between applied rules and gold constraints

### L2: Document Retrieval

- **Context Recall@k (concept)**: 근거를 하나라도 찾은 **개념**의 비율 — 게이트 기준.
  골드 `doc_chunk_groups`는 개념마다 그 개념이 서술된 절의 청크 집합을 담으며,
  집합 중 하나라도 top-k에 들면 그 개념의 근거를 찾은 것으로 본다
  (`scripts/remap_golden_chunk_groups.py`, 2026-08-30 사이클 6).
- **Context Recall@k (chunk)**: 골드 청크 전체 중 top-k에 든 비율 (엄격한 커버리지 지표)
- **Context Recall@k (doc)**: 출처 문서 단위 recall — 사이클 5에서 라벨이 깨져 있던
  동안의 임시 대체 지표. 보고만 유지한다.
- **Context Precision@k**: Proportion of top-k that are gold chunks
- **MRR**: Mean Reciprocal Rank of first relevant document

### L3: Knowledge Graph

- **Hits@k**: Binary indicator if any gold entity in top-k
- **KG Edge Recall**: 골드 엣지 중 검색된 비율 — 게이트 기준 (사이클 4)
- **KG Edge Precision**: 방출 엣지 중 골드에 있는 비율 (남용 감시용)
- **KG Edge F1**: F1 between retrieved and gold edges (연속성 보고용)

### L4: Ontology Compliance

- **Constraint Violation Rate**: Proportion of inferences violating constraints
- **Type Consistency Rate**: Proportion of entities with consistent types

### L5: Answer Quality

- **Exact Match**: Normalized exact match against gold answer
- **Token F1**: Token-level F1 score
- **Groundedness Score**: LLM judge score for context grounding (optional)
- **Answer Relevance Score**: LLM judge score for question relevance (optional)
- **Numeric Accuracy**: `gold.expected_values`의 수치를 답변이 맞힌 비율.
  상대 오차 10% 이내면 정답, `x_low`/`x_high` 쌍은 구간 포함으로 판정한다.
  `expected_values`가 없는 문항은 None(측정 대상 아님).

## 골드 층(gold_source)과 채점

`metadata.gold_source`는 골드 수치를 무엇으로 검증할 수 있는지를 말한다
(`scripts/classify_golden_sources.py`). 층에 따라 게이트 적용 범위가 다르다.

| gold_source | 뜻 | Numeric Accuracy | L5_wrong_answer |
|---|---|---|---|
| `document` | 코퍼스 문서 기반, 시간 불변 | 보고만 | 적용 |
| `snapshot` | `as_of` 시점 크롤 DB 수치 | **게이트 (< 0.50 실패)** | 적용 |
| `domain_expectation` | DB에도 문서에도 없는 추정치 | 보고만 | **제외** |

`domain_expectation` 문항을 정답 일치 게이트에서 빼는 이유: 원자료로 검증할 수
없는 골드에 정답 일치를 요구하면 지표가 문체 유사도를 재게 된다. 이 문항들은
groundedness·relevance로만 판정한다.

Numeric Accuracy는 **종합 점수 공식에는 넣지 않는다.** 적용 대상이 일부 문항이라
분모가 달라져 문항 간 비교가 깨지기 때문이다. 게이트와 보고 전용이다.

snapshot 문항의 `expected_values`는 `scripts/refresh_golden_snapshot_values.py`가
`as_of` 시점 DB에서 생성한다. 문항별 조회 SQL이 그 스크립트에 있다.

## Overall Score

가중 합: L5 45% / L2·L3 35% / L1 10% / L4 10% (`eval/metrics/aggregator.py`).

L2·L3 성분은 **게이트와 같은 지표**를 쓴다 (2026-09-06 변경).

| | 이전 공식 (~v8.1) | 현재 공식 |
|---|---|---|
| L2 | `context_recall_at_k` (청크 단위) | `context_recall_at_k_concept` (개념 단위) |
| L3 | `kg_edge_f1` | `kg_edge_recall` |

이전에는 공식과 게이트가 다른 지표를 봐서 게이트를 개선해도 종합 점수가 움직이지
않았다. **v8.1 이전 baseline과 종합 점수를 직접 비교하지 말 것** — 정의가 다르다.
재집계 값과 연속성 표: `docs/eval/overall-score-formula-2026-09-06.md`
(`python3 scripts/reaggregate_baseline_scores.py`로 재현).

## 인프라 실패는 채점하지 않는다

문항당 에이전트 호출 상한은 `EvalConfig.item_timeout_seconds`(기본 120초), judge
호출 상한은 `LLMJudge(timeout=...)`(기본 60초)다. 타임아웃·API 오류로 답변을 얻지
못한 문항은 **0점으로 채점하지 않고** `trace.error`에 사유를 남긴 뒤 집계에서
분리한다.

- `aggregates.total`은 실제로 채점된 문항 수다 (분리된 문항 제외).
- 분리된 문항은 `aggregates.errored`와 `error_item_ids`로만 보고된다.
- 평균 지표·pass_rate·실패 사유 집계 어디에도 들어가지 않는다.
- 비용은 예외다 — 토큰을 실제로 썼으므로 실패 문항도 합산한다.

## 비용 기록

`report.json`의 `total_tokens`·`total_cost_usd`는 **API 응답의 usage 필드**에서
온다(추정치 아님). 현재 집계 범위는 **답변 생성 호출(L5)과 judge 호출**이다.
질의 재구성·질의 확장 호출은 아직 배선되지 않아 실제 지출은 기록값보다 조금 크다.
usage가 없는 응답은 추정으로 채우지 않고 0으로 남긴다 — 미계측임이 드러나야 한다.

## Gating Thresholds

Items are marked as failed if any of these thresholds are violated:

| Metric | Threshold | Fail Tag |
|--------|-----------|----------|
| Entity Link F1 | < 0.50 | `L1_mapping_fail` |
| Concept Map F1 | < 0.30 | `L1_concept_fail` |
| Context Recall — concept (requires_kg=false) | < 0.80 | `L2_doc_retrieval_fail` |
| Hits@k (requires_kg=true) | < 0.80 | `L3_kg_fail` |
| KG Edge Recall | < 0.50 | `L3_edge_fail` |
| Constraint Violation Rate | > 0.05 | `L4_constraint_violation` |
| Type Consistency Rate | < 0.90 | `L4_type_inconsistency` |
| Answer F1 (domain_expectation 제외) | < 0.50 | `L5_wrong_answer` |
| Numeric Accuracy (snapshot만) | < 0.50 | `L5_numeric_mismatch` |
| Groundedness | < 0.70 | `L5_grounding_fail` |
| Relevance | < 0.70 | `L5_relevance_fail` |

## Architecture

```
eval/
├── __init__.py
├── schemas.py           # Pydantic models
├── loader.py            # Dataset loading
├── runner.py            # Agent invocation + trace capture
├── report.py            # Report generation
├── cli.py               # CLI entrypoint
├── metrics/
│   ├── base.py          # Base metric utilities
│   ├── l1_query.py      # L1 metrics
│   ├── l2_retrieval.py  # L2 metrics
│   ├── l3_kg.py         # L3 metrics
│   ├── l4_ontology.py   # L4 metrics
│   ├── l5_answer.py     # L5 metrics
│   └── aggregator.py    # Weighted scoring + gating
├── validators/
│   └── ontology_validator.py  # Constraint validation
├── judge/
│   ├── interface.py     # LLM-as-judge interface
│   └── stub.py          # Stub scorer (no network)
└── data/
    └── examples/
        └── chatbot_eval.jsonl  # Example dataset
```

## Configuration

Create a custom configuration:

```python
from eval.schemas import EvalConfig

config = EvalConfig(
    top_k=8,                    # Top-k for retrieval metrics
    item_timeout_seconds=120.0, # 문항당 에이전트 호출 상한(초)
    use_judge=False,            # Enable LLM judge
    judge_model=None,           # Model for judge
    save_traces=True,           # Save individual traces
)
```

## Extending

### Adding New Metrics

1. Create a new metric calculator in `eval/metrics/`
2. Implement the `MetricCalculator` interface
3. Update `aggregator.py` to include the new metric

### Custom Judge Implementation

Implement the `JudgeInterface` protocol:

```python
from eval.judge.interface import JudgeInterface

class MyJudge(JudgeInterface):
    async def score_groundedness(self, answer: str, context: str) -> float:
        # Your implementation
        pass

    async def score_relevance(self, answer: str, question: str) -> float:
        # Your implementation
        pass
```

## Testing

```bash
# Run all eval tests
python -m pytest tests/eval/ -v

# Run specific test
python -m pytest tests/eval/test_metrics_l1.py -v
```

## License

Internal use only - AMOREPACIFIC
