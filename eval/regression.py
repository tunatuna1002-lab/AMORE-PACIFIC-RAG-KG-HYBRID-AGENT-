"""
Regression Testing Framework
============================
Compare evaluation results against baselines to detect regressions.

Features:
- Store baseline results (golden snapshots)
- Compare new runs against baselines
- Detect metric regressions with configurable thresholds
- Generate diff reports

Usage:
    # Save baseline
    regression = RegressionTester(baseline_dir="eval/baselines")
    regression.save_baseline("v1.0", results)

    # Compare against baseline
    comparison = regression.compare("v1.0", new_results)
    if comparison.has_regressions:
        print(comparison.regression_report())

Directory Structure:
    eval/baselines/
    ├── v1.0/
    │   ├── report.json
    │   ├── summary.md
    │   └── items/
    │       ├── q001.json
    │       └── q002.json
    └── v2.0/
        └── ...
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.schemas import AggregateMetrics, EvalReport

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RegressionThresholds:
    """
    Thresholds for regression detection.

    A regression is flagged if metric decreases more than the threshold.
    Example: If pass_rate drops from 0.85 to 0.80 and threshold is 0.03,
    this is a regression (0.05 > 0.03).
    """

    # Aggregate metrics (absolute decrease tolerance)
    pass_rate: float = 0.03  # 3% drop in pass rate
    avg_overall_score: float = 0.05  # 5% drop in overall score
    avg_latency_ms: float = 500  # 500ms increase in latency

    # Per-layer metrics (absolute decrease tolerance)
    l1_entity_link_f1: float = 0.05
    l1_concept_map_f1: float = 0.05
    l2_context_recall: float = 0.05
    l2_mrr: float = 0.05
    l3_hits_at_k: float = 0.05
    l3_kg_edge_f1: float = 0.05
    l4_constraint_violation: float = 0.02  # Increase tolerance (lower is better)
    l5_answer_f1: float = 0.05
    l5_groundedness: float = 0.05

    # Cost thresholds (relative increase tolerance)
    cost_increase_pct: float = 0.20  # 20% cost increase

    @classmethod
    def strict(cls) -> "RegressionThresholds":
        """Strict thresholds for critical systems."""
        return cls(
            pass_rate=0.01,
            avg_overall_score=0.02,
            l1_entity_link_f1=0.02,
            l2_context_recall=0.02,
            l5_answer_f1=0.02,
            l5_groundedness=0.02,
        )

    @classmethod
    def relaxed(cls) -> "RegressionThresholds":
        """Relaxed thresholds for early development."""
        return cls(
            pass_rate=0.10,
            avg_overall_score=0.10,
            l1_entity_link_f1=0.10,
            l2_context_recall=0.10,
            l5_answer_f1=0.10,
            l5_groundedness=0.10,
        )


@dataclass
class RegressionResult:
    """Result of a single metric comparison."""

    metric_name: str
    baseline_value: float
    current_value: float
    threshold: float
    is_regression: bool
    delta: float  # Positive = improvement, Negative = regression
    delta_pct: float  # Percentage change

    @property
    def severity(self) -> str:
        """Severity level based on how much threshold is exceeded."""
        if not self.is_regression:
            return "ok"
        exceeded_by = abs(self.delta) - self.threshold
        if exceeded_by > self.threshold * 2:
            return "critical"
        elif exceeded_by > self.threshold:
            return "warning"
        else:
            return "minor"


@dataclass
class ComparisonReport:
    """Complete comparison report between baseline and current results."""

    baseline_name: str
    baseline_timestamp: datetime
    current_timestamp: datetime
    thresholds: RegressionThresholds

    # Per-metric results
    metric_results: list[RegressionResult] = field(default_factory=list)

    # Per-item regressions
    item_regressions: list[dict[str, Any]] = field(default_factory=list)

    # New failures (items that passed before but fail now)
    new_failures: list[str] = field(default_factory=list)

    # Fixed items (items that failed before but pass now)
    fixed_items: list[str] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        """Check if any regressions detected."""
        return any(r.is_regression for r in self.metric_results) or bool(self.new_failures)

    @property
    def regression_count(self) -> int:
        """Count of regressed metrics."""
        return sum(1 for r in self.metric_results if r.is_regression)

    @property
    def critical_regressions(self) -> list[RegressionResult]:
        """Get critical severity regressions."""
        return [r for r in self.metric_results if r.severity == "critical"]

    def summary_dict(self) -> dict[str, Any]:
        """Get summary as dictionary."""
        return {
            "baseline": self.baseline_name,
            "has_regressions": self.has_regressions,
            "regression_count": self.regression_count,
            "new_failures_count": len(self.new_failures),
            "fixed_items_count": len(self.fixed_items),
            "critical_count": len(self.critical_regressions),
            "regressions": [
                {
                    "metric": r.metric_name,
                    "baseline": r.baseline_value,
                    "current": r.current_value,
                    "delta": r.delta,
                    "severity": r.severity,
                }
                for r in self.metric_results
                if r.is_regression
            ],
        }

    def markdown_report(self) -> str:
        """Generate markdown regression report."""
        lines = [
            "# Regression Report",
            "",
            f"**Baseline:** {self.baseline_name}",
            f"**Compared at:** {self.current_timestamp.isoformat()}",
            "",
        ]

        if not self.has_regressions:
            lines.append("✅ **No regressions detected!**")
            if self.fixed_items:
                lines.append(f"\n🎉 **Fixed:** {len(self.fixed_items)} items now pass")
            return "\n".join(lines)

        # Critical regressions
        critical = self.critical_regressions
        if critical:
            lines.append("## ❌ Critical Regressions")
            lines.append("")
            lines.append("| Metric | Baseline | Current | Delta |")
            lines.append("|--------|----------|---------|-------|")
            for r in critical:
                lines.append(
                    f"| {r.metric_name} | {r.baseline_value:.4f} | {r.current_value:.4f} | {r.delta:+.4f} |"
                )
            lines.append("")

        # All regressions
        regressions = [r for r in self.metric_results if r.is_regression]
        if regressions:
            lines.append("## ⚠️ All Regressions")
            lines.append("")
            lines.append("| Metric | Baseline | Current | Delta | Severity |")
            lines.append("|--------|----------|---------|-------|----------|")
            for r in regressions:
                emoji = {"critical": "🔴", "warning": "🟡", "minor": "🟢"}.get(r.severity, "")
                lines.append(
                    f"| {r.metric_name} | {r.baseline_value:.4f} | "
                    f"{r.current_value:.4f} | {r.delta:+.4f} | {emoji} {r.severity} |"
                )
            lines.append("")

        # New failures
        if self.new_failures:
            lines.append("## 🆕 New Failures")
            lines.append("")
            for item_id in self.new_failures[:10]:  # Limit to 10
                lines.append(f"- {item_id}")
            if len(self.new_failures) > 10:
                lines.append(f"- ... and {len(self.new_failures) - 10} more")
            lines.append("")

        # Fixed items
        if self.fixed_items:
            lines.append("## 🎉 Fixed Items")
            lines.append("")
            for item_id in self.fixed_items[:10]:
                lines.append(f"- {item_id}")
            if len(self.fixed_items) > 10:
                lines.append(f"- ... and {len(self.fixed_items) - 10} more")

        return "\n".join(lines)


# =============================================================================
# Regression Tester
# =============================================================================


class RegressionTester:
    """
    Compare evaluation results against stored baselines.

    Stores baselines in a structured directory format and provides
    comparison utilities for CI/CD integration.
    """

    def __init__(
        self,
        baseline_dir: str | Path = "eval/baselines",
        thresholds: RegressionThresholds | None = None,
    ):
        """
        Initialize regression tester.

        Args:
            baseline_dir: Directory to store baselines
            thresholds: Regression detection thresholds
        """
        self.baseline_dir = Path(baseline_dir)
        self.thresholds = thresholds or RegressionThresholds()

    # =========================================================================
    # Baseline Management
    # =========================================================================

    def save_baseline(
        self,
        name: str,
        report: EvalReport,
        save_items: bool = True,
    ) -> Path:
        """
        Save evaluation report as a baseline.

        Args:
            name: Baseline name (e.g., "v1.0", "main-2026-02-03")
            report: Evaluation report to save
            save_items: Whether to save individual item results

        Returns:
            Path to baseline directory
        """
        baseline_path = self.baseline_dir / name
        baseline_path.mkdir(parents=True, exist_ok=True)

        # Save main report
        report_path = baseline_path / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        # Save individual items (for per-item comparison)
        if save_items and report.items:
            items_dir = baseline_path / "items"
            items_dir.mkdir(exist_ok=True)
            for item in report.items:
                item_path = items_dir / f"{item.item_id}.json"
                with open(item_path, "w", encoding="utf-8") as f:
                    f.write(item.model_dump_json(indent=2))

        # Save metadata
        meta = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "total_items": len(report.items),
            "pass_rate": report.aggregates.pass_rate,
        }
        meta_path = baseline_path / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved baseline '{name}' to {baseline_path}")
        return baseline_path

    def load_baseline(self, name: str) -> EvalReport:
        """
        Load a baseline report.

        Args:
            name: Baseline name

        Returns:
            EvalReport from baseline

        Raises:
            FileNotFoundError: If baseline doesn't exist
        """
        baseline_path = self.baseline_dir / name / "report.json"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline '{name}' not found at {baseline_path}")

        with open(baseline_path, encoding="utf-8") as f:
            data = json.load(f)

        return EvalReport.model_validate(data)

    def list_baselines(self) -> list[dict[str, Any]]:
        """
        List all available baselines.

        Returns:
            List of baseline metadata dictionaries
        """
        baselines = []

        if not self.baseline_dir.exists():
            return baselines

        for path in self.baseline_dir.iterdir():
            if path.is_dir():
                meta_path = path / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                        baselines.append(meta)

        return sorted(baselines, key=lambda x: x.get("timestamp", ""), reverse=True)

    def delete_baseline(self, name: str) -> bool:
        """
        Delete a baseline.

        Args:
            name: Baseline name

        Returns:
            True if deleted, False if not found
        """
        baseline_path = self.baseline_dir / name
        if not baseline_path.exists():
            return False

        import shutil

        shutil.rmtree(baseline_path)
        logger.info(f"Deleted baseline '{name}'")
        return True

    # =========================================================================
    # Comparison
    # =========================================================================

    def compare(
        self,
        baseline_name: str,
        current: EvalReport,
    ) -> ComparisonReport:
        """
        Compare current results against a baseline.

        Args:
            baseline_name: Name of baseline to compare against
            current: Current evaluation report

        Returns:
            ComparisonReport with regression analysis
        """
        baseline = self.load_baseline(baseline_name)

        # Load baseline metadata for timestamp
        meta_path = self.baseline_dir / baseline_name / "metadata.json"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        comparison = ComparisonReport(
            baseline_name=baseline_name,
            baseline_timestamp=datetime.fromisoformat(meta["timestamp"]),
            current_timestamp=current.timestamp,
            thresholds=self.thresholds,
        )

        # Compare aggregate metrics
        comparison.metric_results = self._compare_aggregates(
            baseline.aggregates, current.aggregates
        )

        # Compare per-item results
        baseline_items = {item.item_id: item for item in baseline.items}
        current_items = {item.item_id: item for item in current.items}

        # Find new failures and fixed items
        for item_id, current_item in current_items.items():
            baseline_item = baseline_items.get(item_id)
            if baseline_item:
                if baseline_item.passed and not current_item.passed:
                    comparison.new_failures.append(item_id)
                elif not baseline_item.passed and current_item.passed:
                    comparison.fixed_items.append(item_id)

        return comparison

    def _compare_aggregates(
        self,
        baseline: AggregateMetrics,
        current: AggregateMetrics,
    ) -> list[RegressionResult]:
        """Compare aggregate metrics."""
        results = []

        # Define metrics to compare: (name, baseline_val, current_val, threshold, lower_is_better)
        comparisons = [
            ("pass_rate", baseline.pass_rate, current.pass_rate, self.thresholds.pass_rate, False),
            (
                "avg_overall_score",
                baseline.avg_overall_score,
                current.avg_overall_score,
                self.thresholds.avg_overall_score,
                False,
            ),
            (
                "avg_latency_ms",
                baseline.avg_latency_ms,
                current.avg_latency_ms,
                self.thresholds.avg_latency_ms,
                True,
            ),
        ]

        # Add per-layer metrics from by_layer dict
        layer_metrics = [
            ("l1_entity_link_f1", self.thresholds.l1_entity_link_f1),
            ("l1_concept_map_f1", self.thresholds.l1_concept_map_f1),
            ("l2_context_recall", self.thresholds.l2_context_recall),
            ("l2_mrr", self.thresholds.l2_mrr),
            ("l3_hits_at_k", self.thresholds.l3_hits_at_k),
            ("l3_kg_edge_f1", self.thresholds.l3_kg_edge_f1),
            ("l5_answer_f1", self.thresholds.l5_answer_f1),
            ("l5_groundedness", self.thresholds.l5_groundedness),
        ]

        for metric_name, threshold in layer_metrics:
            baseline_val = baseline.by_layer.get(metric_name, 0.0)
            current_val = current.by_layer.get(metric_name, 0.0)
            comparisons.append((metric_name, baseline_val, current_val, threshold, False))

        # Constraint violation (lower is better)
        if "l4_constraint_violation" in baseline.by_layer:
            comparisons.append(
                (
                    "l4_constraint_violation",
                    baseline.by_layer.get("l4_constraint_violation", 0.0),
                    current.by_layer.get("l4_constraint_violation", 0.0),
                    self.thresholds.l4_constraint_violation,
                    True,  # Lower is better
                )
            )

        for name, baseline_val, current_val, threshold, lower_is_better in comparisons:
            delta = current_val - baseline_val
            if lower_is_better:
                delta = -delta  # Flip for "lower is better" metrics

            delta_pct = (delta / baseline_val * 100) if baseline_val != 0 else 0

            # Regression if delta is negative beyond threshold
            is_regression = delta < -threshold

            results.append(
                RegressionResult(
                    metric_name=name,
                    baseline_value=baseline_val,
                    current_value=current_val,
                    threshold=threshold,
                    is_regression=is_regression,
                    delta=delta,
                    delta_pct=delta_pct,
                )
            )

        return results

    def compare_with_latest(self, current: EvalReport) -> ComparisonReport | None:
        """
        Compare against the most recent baseline.

        Args:
            current: Current evaluation report

        Returns:
            ComparisonReport or None if no baselines exist
        """
        baselines = self.list_baselines()
        if not baselines:
            logger.warning("No baselines found for comparison")
            return None

        latest = baselines[0]["name"]
        return self.compare(latest, current)


# =============================================================================
# CI/CD Integration
# =============================================================================


def check_regression(
    current: EvalReport,
    baseline_dir: str | Path = "eval/baselines",
    baseline_name: str | None = None,
    thresholds: RegressionThresholds | None = None,
    fail_on_regression: bool = True,
) -> tuple[bool, ComparisonReport | None]:
    """
    Check for regressions (convenience function for CI/CD).

    Args:
        current: Current evaluation report
        baseline_dir: Baseline directory
        baseline_name: Specific baseline to compare (None = latest)
        thresholds: Regression thresholds
        fail_on_regression: Whether to return failure status on regression

    Returns:
        Tuple of (success: bool, comparison: ComparisonReport | None)
    """
    tester = RegressionTester(baseline_dir, thresholds)

    if baseline_name:
        comparison = tester.compare(baseline_name, current)
    else:
        comparison = tester.compare_with_latest(current)

    if comparison is None:
        logger.info("No baseline found, skipping regression check")
        return True, None

    if fail_on_regression and comparison.has_regressions:
        logger.error(f"Regression detected: {comparison.regression_count} metrics regressed")
        return False, comparison

    return True, comparison
