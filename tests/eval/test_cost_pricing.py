"""Tests for F4: LLM pricing correction (litellm-first lookup, fallback table).

Regression target: gpt-4.1-mini published pricing is $0.40/$1.60 per 1M tokens
(input/output). The previous internal fallback table used $0.15/$0.60, which
under-counted reported evaluation cost by a factor of ~2.667.
"""

import pytest

from eval.cost_tracker import (
    LLM_PRICING,
    CostTracker,
    resolve_embedding_pricing,
    resolve_llm_pricing,
)


class TestResolveLlmPricing:
    """Tests for litellm-first pricing resolution with fallback."""

    def test_gpt_4_1_mini_uses_litellm_and_matches_published_price(self):
        """litellm should resolve gpt-4.1-mini to the $0.40/$1.60 published price."""
        input_per_1m, output_per_1m, source = resolve_llm_pricing("gpt-4.1-mini")

        assert input_per_1m == pytest.approx(0.40, rel=0.01)
        assert output_per_1m == pytest.approx(1.60, rel=0.01)
        assert source == "litellm"

    def test_unknown_model_falls_back_to_internal_table(self):
        """A model litellm doesn't know about should fall back to LLM_PRICING."""
        input_per_1m, output_per_1m, source = resolve_llm_pricing("totally-fake-model-xyz")

        assert source == "fallback_table"
        default = LLM_PRICING["default"]
        assert input_per_1m == pytest.approx(default["input"])
        assert output_per_1m == pytest.approx(default["output"])

    def test_fallback_table_gpt_4_1_mini_is_corrected(self):
        """Even the internal fallback entry for gpt-4.1-mini must carry the true price,
        in case litellm is ever unavailable."""
        pricing = LLM_PRICING["gpt-4.1-mini"]

        assert pricing["input"] == pytest.approx(0.40)
        assert pricing["output"] == pytest.approx(1.60)


class TestResolveEmbeddingPricing:
    """Tests for embedding pricing resolution."""

    def test_text_embedding_3_small_uses_litellm(self):
        price_per_1m, source = resolve_embedding_pricing("text-embedding-3-small")

        assert price_per_1m == pytest.approx(0.02, rel=0.01)
        assert source == "litellm"

    def test_unknown_embedding_model_falls_back(self):
        price_per_1m, source = resolve_embedding_pricing("totally-fake-embedding-xyz")

        assert source == "fallback_table"


class TestCostTrackerCorrectedPricing:
    """Tests that CostTracker now computes cost using the corrected price."""

    def test_l5_cost_matches_published_price(self):
        """1M prompt + 1M completion tokens on gpt-4.1-mini should cost $0.40 + $1.60 = $2.00."""
        tracker = CostTracker(llm_model="gpt-4.1-mini", embedding_model="text-embedding-3-small")
        tracker.track_l5_tokens(prompt_tokens=1_000_000, completion_tokens=1_000_000)

        cost = tracker.get_l5_cost()

        assert cost == pytest.approx(2.00, rel=0.01)

    def test_judge_cost_matches_published_price(self):
        tracker = CostTracker(judge_model="gpt-4.1-mini")
        tracker.track_judge_tokens(prompt_tokens=1_000_000, completion_tokens=1_000_000)

        cost = tracker.get_judge_cost()

        assert cost == pytest.approx(2.00, rel=0.01)

    def test_get_total_cost_no_longer_undercounts(self):
        """Sanity check against the old bug: old pricing (0.15/0.60) would have
        given 0.75 for this input; the corrected pricing must give 2.00."""
        tracker = CostTracker(llm_model="gpt-4.1-mini")
        tracker.track_l5_tokens(prompt_tokens=1_000_000, completion_tokens=1_000_000)

        cost = tracker.get_l5_cost()

        old_buggy_cost = 0.15 + 0.60
        assert cost != pytest.approx(old_buggy_cost, rel=0.01)
        assert cost == pytest.approx(2.00, rel=0.01)


class TestCostTracePricingMetadata:
    """Tests that CostTrace records which unit prices/source were used."""

    def test_to_cost_trace_includes_pricing_metadata(self):
        tracker = CostTracker(
            llm_model="gpt-4.1-mini",
            embedding_model="text-embedding-3-small",
            judge_model="gpt-4.1-mini",
        )
        tracker.track_l5_tokens(prompt_tokens=1000, completion_tokens=500)
        tracker.track_l2_tokens(embedding_tokens=2000)

        trace = tracker.to_cost_trace()

        assert "gpt-4.1-mini" in trace.pricing
        assert trace.pricing["gpt-4.1-mini"]["source"] == "litellm"
        assert trace.pricing["gpt-4.1-mini"]["input_per_1m_usd"] == pytest.approx(0.40, rel=0.01)
        assert trace.pricing["gpt-4.1-mini"]["output_per_1m_usd"] == pytest.approx(1.60, rel=0.01)
        assert "text-embedding-3-small" in trace.pricing

    def test_to_cost_trace_includes_granular_token_breakdown(self):
        tracker = CostTracker()
        tracker.track_l5_tokens(prompt_tokens=1000, completion_tokens=500)
        tracker.track_l2_tokens(embedding_tokens=2000)
        tracker.track_judge_tokens(prompt_tokens=300, completion_tokens=100)

        trace = tracker.to_cost_trace()

        assert trace.l5_prompt_tokens == 1000
        assert trace.l5_completion_tokens == 500
        assert trace.l2_embedding_tokens == 2000
        assert trace.judge_prompt_tokens == 300
        assert trace.judge_completion_tokens == 100

    def test_cost_trace_backward_compatible_with_old_json(self):
        """Old report.json/baseline files won't have the new fields; they must
        still load with safe defaults."""
        from eval.schemas import CostTrace

        old_style_payload = {
            "l1_tokens": 0,
            "l2_tokens": 100,
            "l3_tokens": 0,
            "l4_tokens": 0,
            "l5_tokens": 1500,
            "judge_tokens": 200,
            "l1_cost_usd": 0.0,
            "l2_cost_usd": 0.002,
            "l3_cost_usd": 0.0,
            "l4_cost_usd": 0.0,
            "l5_cost_usd": 0.001,
            "judge_cost_usd": 0.0001,
        }

        trace = CostTrace(**old_style_payload)

        assert trace.l5_prompt_tokens == 0
        assert trace.l5_completion_tokens == 0
        assert trace.pricing == {}
        assert trace.l5_tokens == 1500  # old field still present and correct
