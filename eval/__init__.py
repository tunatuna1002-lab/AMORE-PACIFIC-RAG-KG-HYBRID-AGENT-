"""
Evaluation Harness for AMORE RAG-KG Hybrid Agent
=================================================

Comprehensive offline evaluation pipeline supporting:
- Layered metrics (L1-L5) for different pipeline stages
- Trace capture for all intermediate artifacts
- Label-based and judge-based metrics (LLM & NLI)
- Fuzzy entity matching with alias resolution
- Cost tracking per layer
- Regression testing on golden sets
- JSON + Markdown report generation

Usage:
    python -m eval.cli --dataset eval/data/examples/chatbot_eval.jsonl --out ./eval_output

Version 2.0 Features:
- Fuzzy matching for Korean ↔ English entity aliases
- LLMJudge for real groundedness/relevance scoring
- NLIJudge for local inference-based scoring
- CostTracker for token/USD tracking
- RegressionTester for baseline comparisons
"""

from eval.cost_tracker import CostTracker, estimate_llm_cost, estimate_tokens
from eval.loader import load_dataset
from eval.regression import RegressionTester, RegressionThresholds, check_regression
from eval.report import generate_json_report, generate_markdown_summary
from eval.runner import EvalRunner
from eval.schemas import (
    AnswerTrace,
    CostTrace,
    DocRetrievalTrace,
    EntityLinkingTrace,
    EvalItem,
    EvalReport,
    EvalTrace,
    GoldEvidence,
    ItemMetadata,
    ItemResult,
    KGQueryTrace,
    L1Metrics,
    L2Metrics,
    L3Metrics,
    L4Metrics,
    L5Metrics,
    OntologyReasoningTrace,
)

__version__ = "2.0.0"

__all__ = [
    # Schemas
    "EvalItem",
    "GoldEvidence",
    "ItemMetadata",
    "EvalTrace",
    "CostTrace",
    "EntityLinkingTrace",
    "DocRetrievalTrace",
    "KGQueryTrace",
    "OntologyReasoningTrace",
    "AnswerTrace",
    "L1Metrics",
    "L2Metrics",
    "L3Metrics",
    "L4Metrics",
    "L5Metrics",
    "ItemResult",
    "EvalReport",
    # Core functions
    "load_dataset",
    "EvalRunner",
    "generate_json_report",
    "generate_markdown_summary",
    # Cost tracking
    "CostTracker",
    "estimate_tokens",
    "estimate_llm_cost",
    # Regression testing
    "RegressionTester",
    "RegressionThresholds",
    "check_regression",
]
