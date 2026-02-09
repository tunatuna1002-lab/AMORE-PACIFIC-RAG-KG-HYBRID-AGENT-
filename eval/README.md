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

- **Context Recall@k**: Proportion of gold chunks in top-k retrieved
- **Context Precision@k**: Proportion of top-k that are gold chunks
- **MRR**: Mean Reciprocal Rank of first relevant document

### L3: Knowledge Graph

- **Hits@k**: Binary indicator if any gold entity in top-k
- **KG Edge F1**: F1 between retrieved and gold edges

### L4: Ontology Compliance

- **Constraint Violation Rate**: Proportion of inferences violating constraints
- **Type Consistency Rate**: Proportion of entities with consistent types

### L5: Answer Quality

- **Exact Match**: Normalized exact match against gold answer
- **Token F1**: Token-level F1 score
- **Groundedness Score**: LLM judge score for context grounding (optional)
- **Answer Relevance Score**: LLM judge score for question relevance (optional)

## Gating Thresholds

Items are marked as failed if any of these thresholds are violated:

| Metric | Threshold | Fail Tag |
|--------|-----------|----------|
| Entity Link F1 | < 0.50 | `L1_mapping_fail` |
| Concept Map F1 | < 0.50 | `L1_concept_fail` |
| Context Recall (requires_kg=false) | < 0.80 | `L2_doc_retrieval_fail` |
| Hits@k (requires_kg=true) | < 0.80 | `L3_kg_fail` |
| KG Edge F1 | < 0.50 | `L3_edge_fail` |
| Constraint Violation Rate | > 0.05 | `L4_constraint_violation` |
| Type Consistency Rate | < 0.90 | `L4_type_inconsistency` |
| Answer F1 | < 0.50 | `L5_wrong_answer` |
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
    top_k=8,              # Top-k for retrieval metrics
    use_judge=False,      # Enable LLM judge
    judge_model=None,     # Model for judge
    save_traces=True,     # Save individual traces
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
