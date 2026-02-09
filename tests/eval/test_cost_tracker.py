"""Tests for cost tracking module."""

import pytest

from eval.cost_tracker import (
    CostTracker,
    LayerCost,
    estimate_embedding_cost,
    estimate_llm_cost,
    estimate_tokens,
)


class TestLayerCost:
    """Tests for LayerCost dataclass."""

    def test_initial_values(self):
        """Test initial values are zero."""
        layer = LayerCost()
        assert layer.prompt_tokens == 0
        assert layer.completion_tokens == 0
        assert layer.embedding_tokens == 0
        assert layer.calls == 0
        assert layer.total_tokens == 0

    def test_add_llm_call(self):
        """Test adding an LLM call."""
        layer = LayerCost()
        layer.add_llm_call(100, 50)

        assert layer.prompt_tokens == 100
        assert layer.completion_tokens == 50
        assert layer.total_tokens == 150
        assert layer.calls == 1

    def test_add_embedding_call(self):
        """Test adding an embedding call."""
        layer = LayerCost()
        layer.add_embedding_call(200)

        assert layer.embedding_tokens == 200
        assert layer.total_tokens == 200
        assert layer.calls == 1

    def test_multiple_calls(self):
        """Test accumulating multiple calls."""
        layer = LayerCost()
        layer.add_llm_call(100, 50)
        layer.add_llm_call(200, 100)

        assert layer.prompt_tokens == 300
        assert layer.completion_tokens == 150
        assert layer.calls == 2


class TestCostTracker:
    """Tests for CostTracker."""

    @pytest.fixture
    def tracker(self):
        """Create cost tracker instance."""
        return CostTracker(
            llm_model="gpt-4.1-mini",
            embedding_model="text-embedding-3-small",
        )

    def test_track_l5_tokens(self, tracker):
        """Test tracking L5 tokens."""
        tracker.track_l5_tokens(1000, 500)

        assert tracker.l5.prompt_tokens == 1000
        assert tracker.l5.completion_tokens == 500
        assert tracker.l5.calls == 1

    def test_track_l2_embedding(self, tracker):
        """Test tracking L2 embedding tokens."""
        tracker.track_l2_tokens(2000)

        assert tracker.l2.embedding_tokens == 2000
        assert tracker.l2.calls == 1

    def test_get_total_tokens(self, tracker):
        """Test getting total tokens across layers."""
        tracker.track_l1_tokens(100, 50)
        tracker.track_l2_tokens(200)
        tracker.track_l5_tokens(1000, 500)

        total = tracker.get_total_tokens()
        assert total == 100 + 50 + 200 + 1000 + 500

    def test_get_l5_cost(self, tracker):
        """Test L5 cost calculation."""
        # gpt-4.1-mini: $0.15/1M input, $0.60/1M output
        tracker.track_l5_tokens(1_000_000, 1_000_000)

        cost = tracker.get_l5_cost()
        assert cost == pytest.approx(0.15 + 0.60, rel=0.01)

    def test_get_l2_cost(self, tracker):
        """Test L2 embedding cost calculation."""
        # text-embedding-3-small: $0.02/1M tokens
        tracker.track_l2_tokens(1_000_000)

        cost = tracker.get_l2_cost()
        assert cost == pytest.approx(0.02, rel=0.01)

    def test_get_total_cost(self, tracker):
        """Test total cost calculation."""
        tracker.track_l5_tokens(100_000, 50_000)
        tracker.track_l2_tokens(100_000)

        total = tracker.get_total_cost()
        assert total > 0

    def test_track_item_completed(self, tracker):
        """Test item completion tracking."""
        tracker.track_item_completed()
        tracker.track_item_completed()

        assert tracker.items_evaluated == 2

    def test_get_summary(self, tracker):
        """Test getting cost summary."""
        tracker.track_l5_tokens(1000, 500)
        tracker.track_l2_tokens(2000)
        tracker.track_item_completed()

        summary = tracker.get_summary()

        assert "tokens" in summary
        assert "cost_usd" in summary
        assert "averages" in summary
        assert summary["items_evaluated"] == 1
        assert summary["tokens"]["l5"] == 1500
        assert summary["tokens"]["l2"] == 2000

    def test_reset(self, tracker):
        """Test resetting tracker."""
        tracker.track_l5_tokens(1000, 500)
        tracker.track_item_completed()

        tracker.reset()

        assert tracker.get_total_tokens() == 0
        assert tracker.items_evaluated == 0

    def test_get_cost_breakdown_by_layer(self, tracker):
        """Test cost breakdown by layer."""
        tracker.track_l1_tokens(100, 50)
        tracker.track_l5_tokens(1000, 500)

        breakdown = tracker.get_cost_breakdown_by_layer()

        assert "l1" in breakdown
        assert "l5" in breakdown
        assert breakdown["l1"] > 0
        assert breakdown["l5"] > 0

    def test_to_cost_trace(self, tracker):
        """Test converting to CostTrace schema."""
        tracker.track_l1_tokens(100, 50)
        tracker.track_l5_tokens(1000, 500)

        trace = tracker.to_cost_trace()

        assert trace.l1_tokens == 150
        assert trace.l5_tokens == 1500
        assert trace.l1_cost_usd > 0
        assert trace.l5_cost_usd > 0


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "a" * 400  # 400 characters
        tokens = estimate_tokens(text)
        assert tokens == 100  # ~4 chars per token

    def test_estimate_embedding_cost(self):
        """Test embedding cost estimation."""
        texts = ["hello world"] * 100  # Simple texts

        cost = estimate_embedding_cost(texts, model="text-embedding-3-small")
        assert cost > 0
        assert cost < 0.01  # Should be very small

    def test_estimate_llm_cost(self):
        """Test LLM cost estimation."""
        prompt = "a" * 4000  # ~1000 tokens

        cost = estimate_llm_cost(prompt, expected_output_tokens=500, model="gpt-4.1-mini")
        assert cost > 0
