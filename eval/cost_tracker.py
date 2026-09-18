"""
Cost Tracker
============
Track token usage and estimated costs across evaluation layers.

Provides per-layer cost breakdown for:
- L1: Entity extraction (LLM calls if any)
- L2: Embedding API calls
- L3: KG queries (typically free, but tracked)
- L4: Reasoning LLM calls
- L5: Answer generation LLM calls
- Judge: LLM/NLI judge calls

Usage:
    tracker = CostTracker(model="gpt-4.1-mini", embedding_model="text-embedding-3-small")
    tracker.track_l5_tokens(prompt_tokens=500, completion_tokens=200)
    print(tracker.get_summary())
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Pricing Configuration (per 1M tokens, USD)
# =============================================================================
#
# These tables are a FALLBACK only. Live pricing is looked up from litellm's
# `model_cost` table first (see resolve_llm_pricing / resolve_embedding_pricing
# below); this table is used only when litellm is unavailable or doesn't know
# the model. Keep it corrected so the fallback path is never wrong on its own.
#
# gpt-4.1-mini corrected 2026-09 (F4): was $0.15/$0.60, published price is
# $0.40/$1.60 per 1M tokens (input/output). Source: OpenAI pricing page, as
# mirrored in litellm.model_cost["gpt-4.1-mini"]["source"]
# (https://developers.openai.com/api/docs/pricing).

LLM_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic (2026-09 공식 요금, per 1M tokens)
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    # Defaults
    "default": {"input": 1.00, "output": 3.00},
}

EMBEDDING_PRICING: dict[str, float] = {
    # OpenAI (per 1M tokens)
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
    # Defaults
    "default": 0.10,
}


def _litellm_model_info(model: str) -> dict[str, Any] | None:
    """Look up a model's cost entry from litellm, or None if unavailable/unmapped."""
    try:
        import litellm
    except ImportError:
        return None
    try:
        return litellm.get_model_info(model)
    except Exception:
        # litellm raises a plain Exception (not a specific subclass) for models
        # it doesn't recognize ("This model isn't mapped yet...").
        return None


def resolve_llm_pricing(model: str) -> tuple[float, float, str]:
    """
    Resolve (input_usd_per_1m, output_usd_per_1m, source) for an LLM model.

    Prefers litellm's live `model_cost` table (handles provider prefixes like
    "openai/gpt-4.1-mini" and dated snapshots like "gpt-4.1-mini-2025-04-14").
    Falls back to the internal LLM_PRICING table when litellm is unavailable
    or doesn't know the model.
    """
    info = _litellm_model_info(model)
    if info is not None:
        input_per_token = info.get("input_cost_per_token")
        if input_per_token is not None:
            output_per_token = info.get("output_cost_per_token") or 0.0
            return input_per_token * 1_000_000, output_per_token * 1_000_000, "litellm"

    pricing = LLM_PRICING.get(model, LLM_PRICING["default"])
    return pricing["input"], pricing["output"], "fallback_table"


def resolve_embedding_pricing(model: str) -> tuple[float, str]:
    """
    Resolve (usd_per_1m_tokens, source) for an embedding model.

    Prefers litellm's live `model_cost` table; falls back to EMBEDDING_PRICING.
    """
    info = _litellm_model_info(model)
    if info is not None:
        input_per_token = info.get("input_cost_per_token")
        if input_per_token is not None:
            return input_per_token * 1_000_000, "litellm"

    return EMBEDDING_PRICING.get(model, EMBEDDING_PRICING["default"]), "fallback_table"


@dataclass
class LayerCost:
    """Cost tracking for a single layer."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.embedding_tokens

    def add_llm_call(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Track an LLM call."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += 1

    def add_embedding_call(self, tokens: int) -> None:
        """Track an embedding call."""
        self.embedding_tokens += tokens
        self.calls += 1


@dataclass
class CostTracker:
    """
    Track token usage and costs across evaluation layers.

    Supports multiple pricing models and provides detailed breakdowns.
    """

    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    judge_model: str = "gpt-4.1-mini"

    # Per-layer tracking
    l1: LayerCost = field(default_factory=LayerCost)
    l2: LayerCost = field(default_factory=LayerCost)
    l3: LayerCost = field(default_factory=LayerCost)
    l4: LayerCost = field(default_factory=LayerCost)
    l5: LayerCost = field(default_factory=LayerCost)
    judge: LayerCost = field(default_factory=LayerCost)

    # Session tracking
    start_time: datetime = field(default_factory=datetime.now)
    items_evaluated: int = 0

    # =========================================================================
    # Tracking Methods
    # =========================================================================

    def track_l1_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Track L1 entity extraction tokens."""
        self.l1.add_llm_call(prompt_tokens, completion_tokens)

    def track_l2_tokens(self, embedding_tokens: int = 0) -> None:
        """Track L2 embedding tokens."""
        self.l2.add_embedding_call(embedding_tokens)

    def track_l3_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Track L3 KG query tokens (if using LLM for query generation)."""
        self.l3.add_llm_call(prompt_tokens, completion_tokens)

    def track_l4_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Track L4 reasoning tokens."""
        self.l4.add_llm_call(prompt_tokens, completion_tokens)

    def track_l5_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Track L5 answer generation tokens."""
        self.l5.add_llm_call(prompt_tokens, completion_tokens)

    def track_judge_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Track judge evaluation tokens."""
        self.judge.add_llm_call(prompt_tokens, completion_tokens)

    def track_item_completed(self) -> None:
        """Mark an evaluation item as completed."""
        self.items_evaluated += 1

    # =========================================================================
    # Cost Calculation
    # =========================================================================

    def _get_llm_pricing(self, model: str) -> dict[str, float]:
        """Get pricing for an LLM model (litellm live table, fallback to internal table)."""
        input_per_m, output_per_m, _source = resolve_llm_pricing(model)
        return {"input": input_per_m, "output": output_per_m}

    def _get_embedding_pricing(self, model: str) -> float:
        """Get pricing for an embedding model (litellm live table, fallback to internal table)."""
        price_per_m, _source = resolve_embedding_pricing(model)
        return price_per_m

    def get_pricing_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Unit prices actually used for this run's cost calculations, keyed by model
        name, with the source ("litellm" or "fallback_table") each came from.
        """
        metadata: dict[str, dict[str, Any]] = {}

        for model in (self.llm_model, self.judge_model):
            input_per_m, output_per_m, source = resolve_llm_pricing(model)
            metadata[model] = {
                "input_per_1m_usd": input_per_m,
                "output_per_1m_usd": output_per_m,
                "source": source,
            }

        embed_price, embed_source = resolve_embedding_pricing(self.embedding_model)
        metadata[self.embedding_model] = {
            "input_per_1m_usd": embed_price,
            "output_per_1m_usd": 0.0,
            "source": embed_source,
        }

        return metadata

    def _compute_layer_cost(self, layer: LayerCost, is_embedding: bool = False) -> float:
        """Compute cost for a layer in USD."""
        if is_embedding:
            price_per_m = self._get_embedding_pricing(self.embedding_model)
            return (layer.embedding_tokens / 1_000_000) * price_per_m
        else:
            pricing = self._get_llm_pricing(self.llm_model)
            input_cost = (layer.prompt_tokens / 1_000_000) * pricing["input"]
            output_cost = (layer.completion_tokens / 1_000_000) * pricing["output"]
            return input_cost + output_cost

    def get_l1_cost(self) -> float:
        """Get L1 cost in USD."""
        return self._compute_layer_cost(self.l1)

    def get_l2_cost(self) -> float:
        """Get L2 cost in USD (embeddings)."""
        return self._compute_layer_cost(self.l2, is_embedding=True)

    def get_l3_cost(self) -> float:
        """Get L3 cost in USD."""
        return self._compute_layer_cost(self.l3)

    def get_l4_cost(self) -> float:
        """Get L4 cost in USD."""
        return self._compute_layer_cost(self.l4)

    def get_l5_cost(self) -> float:
        """Get L5 cost in USD."""
        return self._compute_layer_cost(self.l5)

    def get_judge_cost(self) -> float:
        """Get judge cost in USD."""
        # Judge may use a different model
        pricing = self._get_llm_pricing(self.judge_model)
        input_cost = (self.judge.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.judge.completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def get_total_cost(self) -> float:
        """Get total cost in USD."""
        return (
            self.get_l1_cost()
            + self.get_l2_cost()
            + self.get_l3_cost()
            + self.get_l4_cost()
            + self.get_l5_cost()
            + self.get_judge_cost()
        )

    def get_total_tokens(self) -> int:
        """Get total tokens across all layers."""
        return (
            self.l1.total_tokens
            + self.l2.total_tokens
            + self.l3.total_tokens
            + self.l4.total_tokens
            + self.l5.total_tokens
            + self.judge.total_tokens
        )

    # =========================================================================
    # Summary & Export
    # =========================================================================

    def get_summary(self) -> dict[str, Any]:
        """
        Get a complete cost summary.

        Returns:
            Dictionary with all cost metrics
        """
        duration = (datetime.now() - self.start_time).total_seconds()

        return {
            "duration_seconds": duration,
            "items_evaluated": self.items_evaluated,
            "models": {
                "llm": self.llm_model,
                "embedding": self.embedding_model,
                "judge": self.judge_model,
            },
            "tokens": {
                "l1": self.l1.total_tokens,
                "l2": self.l2.total_tokens,
                "l3": self.l3.total_tokens,
                "l4": self.l4.total_tokens,
                "l5": self.l5.total_tokens,
                "judge": self.judge.total_tokens,
                "total": self.get_total_tokens(),
            },
            "calls": {
                "l1": self.l1.calls,
                "l2": self.l2.calls,
                "l3": self.l3.calls,
                "l4": self.l4.calls,
                "l5": self.l5.calls,
                "judge": self.judge.calls,
            },
            "cost_usd": {
                "l1": round(self.get_l1_cost(), 6),
                "l2": round(self.get_l2_cost(), 6),
                "l3": round(self.get_l3_cost(), 6),
                "l4": round(self.get_l4_cost(), 6),
                "l5": round(self.get_l5_cost(), 6),
                "judge": round(self.get_judge_cost(), 6),
                "total": round(self.get_total_cost(), 6),
            },
            "averages": {
                "tokens_per_item": (
                    self.get_total_tokens() / self.items_evaluated
                    if self.items_evaluated > 0
                    else 0
                ),
                "cost_per_item_usd": (
                    self.get_total_cost() / self.items_evaluated if self.items_evaluated > 0 else 0
                ),
            },
        }

    def get_cost_breakdown_by_layer(self) -> dict[str, float]:
        """Get cost breakdown by layer for reports."""
        return {
            "l1": self.get_l1_cost(),
            "l2": self.get_l2_cost(),
            "l3": self.get_l3_cost(),
            "l4": self.get_l4_cost(),
            "l5": self.get_l5_cost(),
            "judge": self.get_judge_cost(),
        }

    def to_cost_trace(self) -> "CostTrace":  # noqa: F821
        """Convert to CostTrace schema for storage."""
        from eval.schemas import CostTrace

        return CostTrace(
            l1_tokens=self.l1.total_tokens,
            l2_tokens=self.l2.total_tokens,
            l3_tokens=self.l3.total_tokens,
            l4_tokens=self.l4.total_tokens,
            l5_tokens=self.l5.total_tokens,
            judge_tokens=self.judge.total_tokens,
            l1_prompt_tokens=self.l1.prompt_tokens,
            l1_completion_tokens=self.l1.completion_tokens,
            l2_embedding_tokens=self.l2.embedding_tokens,
            l3_prompt_tokens=self.l3.prompt_tokens,
            l3_completion_tokens=self.l3.completion_tokens,
            l4_prompt_tokens=self.l4.prompt_tokens,
            l4_completion_tokens=self.l4.completion_tokens,
            l5_prompt_tokens=self.l5.prompt_tokens,
            l5_completion_tokens=self.l5.completion_tokens,
            judge_prompt_tokens=self.judge.prompt_tokens,
            judge_completion_tokens=self.judge.completion_tokens,
            l1_cost_usd=self.get_l1_cost(),
            l2_cost_usd=self.get_l2_cost(),
            l3_cost_usd=self.get_l3_cost(),
            l4_cost_usd=self.get_l4_cost(),
            l5_cost_usd=self.get_l5_cost(),
            judge_cost_usd=self.get_judge_cost(),
            pricing=self.get_pricing_metadata(),
        )

    def reset(self) -> None:
        """Reset all tracking."""
        self.l1 = LayerCost()
        self.l2 = LayerCost()
        self.l3 = LayerCost()
        self.l4 = LayerCost()
        self.l5 = LayerCost()
        self.judge = LayerCost()
        self.start_time = datetime.now()
        self.items_evaluated = 0

    def __repr__(self) -> str:
        return (
            f"CostTracker(items={self.items_evaluated}, "
            f"tokens={self.get_total_tokens()}, "
            f"cost=${self.get_total_cost():.4f})"
        )


# =============================================================================
# Utility Functions
# =============================================================================


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses simple heuristic: ~4 characters per token for English.
    For more accurate counts, use tiktoken.
    """
    return len(text) // 4


def estimate_embedding_cost(texts: list[str], model: str = "text-embedding-3-small") -> float:
    """
    Estimate embedding cost for a list of texts.

    Args:
        texts: List of texts to embed
        model: Embedding model name

    Returns:
        Estimated cost in USD
    """
    total_tokens = sum(estimate_tokens(t) for t in texts)
    price_per_m, _source = resolve_embedding_pricing(model)
    return (total_tokens / 1_000_000) * price_per_m


def estimate_llm_cost(
    prompt: str, expected_output_tokens: int = 500, model: str = "gpt-4.1-mini"
) -> float:
    """
    Estimate LLM call cost.

    Args:
        prompt: Input prompt
        expected_output_tokens: Expected output token count
        model: LLM model name

    Returns:
        Estimated cost in USD
    """
    prompt_tokens = estimate_tokens(prompt)
    input_per_m, output_per_m, _source = resolve_llm_pricing(model)
    input_cost = (prompt_tokens / 1_000_000) * input_per_m
    output_cost = (expected_output_tokens / 1_000_000) * output_per_m
    return input_cost + output_cost
