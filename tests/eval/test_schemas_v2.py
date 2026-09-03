"""Tests for CostBreakdown and RegressionComparison schemas."""

import pytest

from eval.schemas import CostBreakdown, LayerCost, RegressionComparison, RegressionItem


class TestLayerCost:
    def test_defaults(self):
        lc = LayerCost()
        assert lc.tokens == 0
        assert lc.cost_usd == 0.0


class TestCostBreakdown:
    def test_creation(self):
        cb = CostBreakdown(run_id="run-001", total_tokens=1000, total_cost_usd=0.05)
        assert cb.run_id == "run-001"
        assert cb.total_tokens == 1000

    def test_cost_delta_pct(self):
        cb = CostBreakdown(run_id="run-002", total_cost_usd=0.06, prev_run_cost_usd=0.05)
        assert cb.cost_delta_pct == pytest.approx(20.0)

    def test_cost_delta_pct_no_prev(self):
        cb = CostBreakdown(run_id="run-003", total_cost_usd=0.05)
        assert cb.cost_delta_pct is None

    def test_cost_delta_pct_zero_prev(self):
        cb = CostBreakdown(run_id="run-004", total_cost_usd=0.05, prev_run_cost_usd=0.0)
        assert cb.cost_delta_pct is None

    def test_summary(self):
        cb = CostBreakdown(
            run_id="run-005",
            total_tokens=2000,
            total_cost_usd=0.10,
            by_layer={
                "l1": LayerCost(tokens=500, cost_usd=0.02),
                "l5": LayerCost(tokens=1500, cost_usd=0.08),
            },
        )
        s = cb.summary()
        assert s["run_id"] == "run-005"
        assert s["total_tokens"] == 2000
        assert "l1" in s["layers"]

    def test_empty(self):
        cb = CostBreakdown(run_id="empty")
        assert cb.total_tokens == 0
        assert cb.total_cost_usd == 0.0
        assert cb.summary()["total_tokens"] == 0


class TestRegressionItem:
    def test_creation(self):
        ri = RegressionItem(
            metric="entity_f1",
            baseline_value=0.85,
            current_value=0.75,
            delta=-0.10,
            severity="high",
        )
        assert ri.metric == "entity_f1"
        assert ri.severity == "high"


class TestRegressionComparison:
    def test_creation(self):
        rc = RegressionComparison(baseline_run_id="run-001", current_run_id="run-002")
        assert rc.baseline_run_id == "run-001"
        assert not rc.has_regressions()

    def test_has_regressions_above_threshold(self):
        rc = RegressionComparison(
            baseline_run_id="run-001",
            current_run_id="run-002",
            regressions=[
                RegressionItem(
                    metric="f1",
                    baseline_value=0.9,
                    current_value=0.8,
                    delta=-0.1,
                    severity="high",
                )
            ],
        )
        assert rc.has_regressions(threshold=0.05)
        assert not rc.has_regressions(threshold=0.15)

    def test_summary(self):
        rc = RegressionComparison(
            baseline_run_id="run-001",
            current_run_id="run-002",
            regressions=[
                RegressionItem(metric="f1", baseline_value=0.9, current_value=0.8, delta=-0.1)
            ],
            improvements=[
                RegressionItem(metric="recall", baseline_value=0.7, current_value=0.85, delta=0.15)
            ],
        )
        s = rc.summary()
        assert s["total_regressions"] == 1
        assert s["total_improvements"] == 1

    def test_empty_summary(self):
        rc = RegressionComparison(baseline_run_id="a", current_run_id="b")
        s = rc.summary()
        assert s["total_regressions"] == 0
        assert s["worst_regression"] == 0.0
