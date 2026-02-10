"""Tests for regression testing framework."""

from datetime import datetime

import pytest

from eval.regression import (
    ComparisonReport,
    RegressionResult,
    RegressionTester,
    RegressionThresholds,
    check_regression,
)
from eval.schemas import AggregateMetrics, EvalConfig, EvalReport, ItemResult


class TestRegressionThresholds:
    """Tests for RegressionThresholds."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = RegressionThresholds()
        assert thresholds.pass_rate == 0.03
        assert thresholds.avg_overall_score == 0.05

    def test_strict_thresholds(self):
        """Test strict thresholds."""
        thresholds = RegressionThresholds.strict()
        assert thresholds.pass_rate == 0.01
        assert thresholds.avg_overall_score == 0.02

    def test_relaxed_thresholds(self):
        """Test relaxed thresholds."""
        thresholds = RegressionThresholds.relaxed()
        assert thresholds.pass_rate == 0.10
        assert thresholds.avg_overall_score == 0.10


class TestRegressionResult:
    """Tests for RegressionResult."""

    def test_no_regression(self):
        """Test when no regression occurs."""
        result = RegressionResult(
            metric_name="pass_rate",
            baseline_value=0.80,
            current_value=0.82,  # Improved
            threshold=0.03,
            is_regression=False,
            delta=0.02,
            delta_pct=2.5,
        )

        assert not result.is_regression
        assert result.severity == "ok"

    def test_minor_regression(self):
        """Test minor regression detection."""
        result = RegressionResult(
            metric_name="pass_rate",
            baseline_value=0.80,
            current_value=0.77,  # Dropped
            threshold=0.03,
            is_regression=True,
            delta=-0.03,
            delta_pct=-3.75,
        )

        assert result.is_regression
        assert result.severity == "minor"

    def test_critical_regression(self):
        """Test critical regression detection."""
        result = RegressionResult(
            metric_name="pass_rate",
            baseline_value=0.80,
            current_value=0.60,  # Major drop
            threshold=0.03,
            is_regression=True,
            delta=-0.20,
            delta_pct=-25.0,
        )

        assert result.is_regression
        assert result.severity == "critical"


class TestComparisonReport:
    """Tests for ComparisonReport."""

    @pytest.fixture
    def report_with_regressions(self):
        """Create a comparison report with regressions."""
        return ComparisonReport(
            baseline_name="v1.0",
            baseline_timestamp=datetime.now(),
            current_timestamp=datetime.now(),
            thresholds=RegressionThresholds(),
            metric_results=[
                RegressionResult(
                    metric_name="pass_rate",
                    baseline_value=0.80,
                    current_value=0.70,
                    threshold=0.03,
                    is_regression=True,
                    delta=-0.10,
                    delta_pct=-12.5,
                ),
                RegressionResult(
                    metric_name="avg_overall_score",
                    baseline_value=0.75,
                    current_value=0.78,
                    threshold=0.05,
                    is_regression=False,
                    delta=0.03,
                    delta_pct=4.0,
                ),
            ],
            new_failures=["q001", "q002"],
        )

    def test_has_regressions(self, report_with_regressions):
        """Test regression detection."""
        assert report_with_regressions.has_regressions

    def test_regression_count(self, report_with_regressions):
        """Test regression count."""
        assert report_with_regressions.regression_count == 1

    def test_no_regressions(self):
        """Test report with no regressions."""
        report = ComparisonReport(
            baseline_name="v1.0",
            baseline_timestamp=datetime.now(),
            current_timestamp=datetime.now(),
            thresholds=RegressionThresholds(),
            metric_results=[
                RegressionResult(
                    metric_name="pass_rate",
                    baseline_value=0.80,
                    current_value=0.85,
                    threshold=0.03,
                    is_regression=False,
                    delta=0.05,
                    delta_pct=6.25,
                ),
            ],
        )
        assert not report.has_regressions

    def test_summary_dict(self, report_with_regressions):
        """Test summary dictionary generation."""
        summary = report_with_regressions.summary_dict()

        assert summary["baseline"] == "v1.0"
        assert summary["has_regressions"] is True
        assert summary["regression_count"] == 1
        assert summary["new_failures_count"] == 2

    def test_markdown_report(self, report_with_regressions):
        """Test markdown report generation."""
        md = report_with_regressions.markdown_report()

        assert "# Regression Report" in md
        assert "pass_rate" in md
        assert "v1.0" in md


class TestRegressionTester:
    """Tests for RegressionTester."""

    @pytest.fixture
    def tester(self, tmp_path):
        """Create regression tester with temp directory."""
        return RegressionTester(baseline_dir=tmp_path)

    @pytest.fixture
    def sample_report(self):
        """Create a sample evaluation report."""
        return EvalReport(
            timestamp=datetime.now(),
            config=EvalConfig(),
            aggregates=AggregateMetrics(
                total=10,
                passed=8,
                failed=2,
                pass_rate=0.80,
                avg_overall_score=0.75,
                avg_latency_ms=100.0,
                by_layer={
                    "l1_entity_link_f1": 0.85,
                    "l2_context_recall": 0.80,
                    "l5_answer_f1": 0.70,
                },
            ),
            items=[
                ItemResult(item_id="q001", passed=True, overall_score=0.9),
                ItemResult(item_id="q002", passed=True, overall_score=0.8),
                ItemResult(item_id="q003", passed=False, overall_score=0.4),
            ],
        )

    def test_save_baseline(self, tester, sample_report, tmp_path):
        """Test saving a baseline."""
        path = tester.save_baseline("v1.0", sample_report)

        assert path.exists()
        assert (path / "report.json").exists()
        assert (path / "metadata.json").exists()
        assert (path / "items").exists()

    def test_load_baseline(self, tester, sample_report):
        """Test loading a baseline."""
        tester.save_baseline("v1.0", sample_report)
        loaded = tester.load_baseline("v1.0")

        assert loaded.aggregates.total == sample_report.aggregates.total
        assert loaded.aggregates.pass_rate == sample_report.aggregates.pass_rate

    def test_load_nonexistent_baseline(self, tester):
        """Test loading nonexistent baseline raises error."""
        with pytest.raises(FileNotFoundError):
            tester.load_baseline("nonexistent")

    def test_list_baselines(self, tester, sample_report):
        """Test listing baselines."""
        tester.save_baseline("v1.0", sample_report)
        tester.save_baseline("v2.0", sample_report)

        baselines = tester.list_baselines()

        assert len(baselines) == 2
        names = [b["name"] for b in baselines]
        assert "v1.0" in names
        assert "v2.0" in names

    def test_delete_baseline(self, tester, sample_report, tmp_path):
        """Test deleting a baseline."""
        tester.save_baseline("v1.0", sample_report)
        assert (tmp_path / "v1.0").exists()

        result = tester.delete_baseline("v1.0")

        assert result is True
        assert not (tmp_path / "v1.0").exists()

    def test_compare_no_regression(self, tester, sample_report):
        """Test comparison with no regression."""
        tester.save_baseline("v1.0", sample_report)

        # Create improved report
        improved = EvalReport(
            aggregates=AggregateMetrics(
                total=10,
                passed=9,
                failed=1,
                pass_rate=0.90,  # Improved
                avg_overall_score=0.80,  # Improved
                by_layer={
                    "l1_entity_link_f1": 0.90,
                    "l2_context_recall": 0.85,
                    "l5_answer_f1": 0.75,
                },
            ),
            items=[
                ItemResult(item_id="q001", passed=True),
                ItemResult(item_id="q002", passed=True),
                ItemResult(item_id="q003", passed=True),  # Fixed
            ],
        )

        comparison = tester.compare("v1.0", improved)

        assert not comparison.has_regressions
        assert "q003" in comparison.fixed_items

    def test_compare_with_regression(self, tester, sample_report):
        """Test comparison with regression."""
        tester.save_baseline("v1.0", sample_report)

        # Create worse report
        worse = EvalReport(
            aggregates=AggregateMetrics(
                total=10,
                passed=6,
                failed=4,
                pass_rate=0.60,  # Regressed
                avg_overall_score=0.60,  # Regressed
                by_layer={
                    "l1_entity_link_f1": 0.70,
                    "l2_context_recall": 0.65,
                    "l5_answer_f1": 0.50,
                },
            ),
            items=[
                ItemResult(item_id="q001", passed=False),  # New failure
                ItemResult(item_id="q002", passed=True),
                ItemResult(item_id="q003", passed=False),
            ],
        )

        comparison = tester.compare("v1.0", worse)

        assert comparison.has_regressions
        assert comparison.regression_count > 0
        assert "q001" in comparison.new_failures


class TestCheckRegressionFunction:
    """Tests for check_regression convenience function."""

    def test_check_regression_no_baseline(self, tmp_path):
        """Test check_regression when no baseline exists."""
        report = EvalReport(
            aggregates=AggregateMetrics(total=10, passed=8, pass_rate=0.80),
        )

        success, comparison = check_regression(
            report,
            baseline_dir=tmp_path,
        )

        assert success is True
        assert comparison is None

    def test_check_regression_pass(self, tmp_path):
        """Test check_regression when no regressions."""
        # Save baseline
        tester = RegressionTester(baseline_dir=tmp_path)
        baseline = EvalReport(
            aggregates=AggregateMetrics(
                total=10,
                passed=8,
                pass_rate=0.80,
                avg_overall_score=0.75,
                by_layer={},
            ),
            items=[],
        )
        tester.save_baseline("v1.0", baseline)

        # Check improved report
        current = EvalReport(
            aggregates=AggregateMetrics(
                total=10,
                passed=9,
                pass_rate=0.90,
                avg_overall_score=0.85,
                by_layer={},
            ),
            items=[],
        )

        success, comparison = check_regression(
            current,
            baseline_dir=tmp_path,
            baseline_name="v1.0",
        )

        assert success is True
        assert comparison is not None
        assert not comparison.has_regressions

    def test_check_regression_fail(self, tmp_path):
        """Test check_regression when regression detected."""
        # Save baseline
        tester = RegressionTester(baseline_dir=tmp_path)
        baseline = EvalReport(
            aggregates=AggregateMetrics(
                total=10,
                passed=8,
                pass_rate=0.80,
                avg_overall_score=0.75,
                by_layer={},
            ),
            items=[],
        )
        tester.save_baseline("v1.0", baseline)

        # Check regressed report
        current = EvalReport(
            aggregates=AggregateMetrics(
                total=10,
                passed=5,
                pass_rate=0.50,  # Major regression
                avg_overall_score=0.55,
                by_layer={},
            ),
            items=[],
        )

        success, comparison = check_regression(
            current,
            baseline_dir=tmp_path,
            baseline_name="v1.0",
            fail_on_regression=True,
        )

        assert success is False
        assert comparison is not None
        assert comparison.has_regressions
