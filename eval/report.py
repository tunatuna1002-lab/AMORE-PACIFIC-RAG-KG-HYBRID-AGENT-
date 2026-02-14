"""
Report Generator
================
Generates JSON and Markdown reports from evaluation results.

Output formats:
- report.json: Full structured results
- summary.md: Human-readable summary
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.metrics.aggregator import FAIL_REASONS
from eval.schemas import AggregateMetrics, EvalConfig, EvalReport, ItemResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates evaluation reports in JSON and Markdown formats.
    """

    def __init__(self, config: EvalConfig | None = None):
        """
        Initialize report generator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvalConfig()

    def generate_report(
        self,
        results: list[ItemResult],
        out_dir: Path | str,
    ) -> EvalReport:
        """
        Generate full evaluation report.

        Args:
            results: List of evaluation results
            out_dir: Output directory

        Returns:
            EvalReport object
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Compute aggregates
        aggregates = self._compute_aggregates(results)

        # Create report
        report = EvalReport(
            timestamp=datetime.now(),
            config=self.config,
            aggregates=aggregates,
            items=results,
        )

        # Write JSON report
        self._write_json_report(report, out_dir / "report.json")

        # Write Markdown summary
        self._write_markdown_summary(report, out_dir / "summary.md")

        # Write individual traces (optional)
        if self.config.save_traces:
            self._write_traces(results, out_dir / "traces")

        logger.info(f"Report generated at {out_dir}")
        return report

    def _compute_aggregates(self, results: list[ItemResult]) -> AggregateMetrics:
        """
        Compute aggregate metrics from results.

        Args:
            results: List of evaluation results

        Returns:
            AggregateMetrics
        """
        total = len(results)
        if total == 0:
            return AggregateMetrics(
                total=0,
                passed=0,
                failed=0,
                pass_rate=0.0,
                avg_overall_score=0.0,
                avg_latency_ms=0.0,
                by_layer={},
                by_domain={},
                by_difficulty={},
                top_fail_reasons={},
            )

        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # Average overall score
        avg_score = sum(r.overall_score for r in results) / total

        # Average latency
        latencies = [r.trace.latency_ms for r in results if r.trace.latency_ms]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # By-layer averages
        by_layer = self._compute_layer_averages(results)

        # By-domain breakdown
        by_domain = self._compute_domain_breakdown(results)

        # By-difficulty breakdown
        by_difficulty = self._compute_difficulty_breakdown(results)

        # Top fail reasons
        top_fail_reasons = self._compute_fail_reason_counts(results)

        return AggregateMetrics(
            total=total,
            passed=passed,
            failed=failed,
            pass_rate=passed / total,
            avg_overall_score=avg_score,
            avg_latency_ms=avg_latency,
            by_layer=by_layer,
            by_domain=by_domain,
            by_difficulty=by_difficulty,
            top_fail_reasons=top_fail_reasons,
        )

    def _compute_layer_averages(self, results: list[ItemResult]) -> dict[str, float]:
        """Compute average metrics for each layer."""
        if not results:
            return {}

        n = len(results)
        return {
            # L1
            "l1_entity_link_f1": sum(r.l1.entity_link_f1 for r in results) / n,
            "l1_concept_map_f1": sum(r.l1.concept_map_f1 for r in results) / n,
            "l1_constraint_extraction_f1": sum(r.l1.constraint_extraction_f1 for r in results) / n,
            # L2
            "l2_context_recall": sum(r.l2.context_recall_at_k for r in results) / n,
            "l2_context_precision": sum(r.l2.context_precision_at_k for r in results) / n,
            "l2_mrr": sum(r.l2.mrr for r in results) / n,
            # L3
            "l3_hits_at_k": sum(r.l3.hits_at_k for r in results) / n,
            "l3_kg_edge_f1": sum(r.l3.kg_edge_f1 for r in results) / n,
            # L4
            "l4_constraint_violation_rate": sum(r.l4.constraint_violation_rate for r in results)
            / n,
            "l4_type_consistency_rate": sum(r.l4.type_consistency_rate for r in results) / n,
            # L5
            "l5_exact_match": sum(r.l5.answer_exact_match for r in results) / n,
            "l5_answer_f1": sum(r.l5.answer_f1 for r in results) / n,
            "l5_groundedness": sum(
                r.l5.groundedness_score for r in results if r.l5.groundedness_score is not None
            )
            / max(1, sum(1 for r in results if r.l5.groundedness_score is not None)),
            "l5_relevance": sum(
                r.l5.answer_relevance_score
                for r in results
                if r.l5.answer_relevance_score is not None
            )
            / max(1, sum(1 for r in results if r.l5.answer_relevance_score is not None)),
        }

    def _compute_domain_breakdown(self, results: list[ItemResult]) -> dict[str, dict[str, float]]:
        """Compute metrics breakdown by domain."""
        by_domain: dict[str, list[ItemResult]] = defaultdict(list)

        for r in results:
            # Use domain from metadata
            domain = r.metadata.domain
            by_domain[domain].append(r)

        breakdown = {}
        for domain, domain_results in by_domain.items():
            n = len(domain_results)
            breakdown[domain] = {
                "count": n,
                "pass_rate": sum(1 for r in domain_results if r.passed) / n,
                "avg_score": sum(r.overall_score for r in domain_results) / n,
            }

        return breakdown

    def _compute_difficulty_breakdown(
        self, results: list[ItemResult]
    ) -> dict[str, dict[str, float]]:
        """Compute metrics breakdown by difficulty."""
        by_difficulty: dict[str, list[ItemResult]] = defaultdict(list)

        for r in results:
            # Use difficulty from metadata
            difficulty = r.metadata.difficulty
            by_difficulty[difficulty].append(r)

        breakdown = {}
        for difficulty, difficulty_results in by_difficulty.items():
            n = len(difficulty_results)
            breakdown[difficulty] = {
                "count": n,
                "pass_rate": sum(1 for r in difficulty_results if r.passed) / n,
                "avg_score": sum(r.overall_score for r in difficulty_results) / n,
            }

        return breakdown

    def _compute_fail_reason_counts(self, results: list[ItemResult]) -> dict[str, int]:
        """Count occurrences of each fail reason."""
        counts: dict[str, int] = defaultdict(int)

        for r in results:
            for reason in r.fail_reason_tags:
                counts[reason] += 1

        # Sort by count descending
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def _write_json_report(self, report: EvalReport, path: Path) -> None:
        """Write JSON report to file."""

        def serialize(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Path):
                return str(obj)
            return obj

        report_dict = report.model_dump()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, default=serialize, ensure_ascii=False)

        logger.info(f"JSON report written to {path}")

    def _write_markdown_summary(self, report: EvalReport, path: Path) -> None:
        """Write Markdown summary to file."""
        lines = []

        # Header
        lines.append("# Evaluation Summary")
        lines.append("")
        lines.append(f"**Generated**: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total Items**: {report.aggregates.total}")
        lines.append(f"**Pass Rate**: {report.aggregates.pass_rate:.1%}")
        lines.append(f"**Avg Score**: {report.aggregates.avg_overall_score:.3f}")
        lines.append("")

        # Configuration
        lines.append("## Configuration")
        lines.append("")
        lines.append(f"- Top-K: {report.config.top_k}")
        lines.append(f"- Judge: {'enabled' if report.config.use_judge else 'stub'}")
        lines.append("")

        # Overall Results
        lines.append("## Overall Results")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Passed | {report.aggregates.passed} |")
        lines.append(f"| Failed | {report.aggregates.failed} |")
        lines.append(f"| Pass Rate | {report.aggregates.pass_rate:.1%} |")
        lines.append(f"| Avg Score | {report.aggregates.avg_overall_score:.3f} |")
        lines.append(f"| Avg Latency | {report.aggregates.avg_latency_ms:.0f}ms |")
        lines.append("")

        # Layer Metrics
        lines.append("## Layer Metrics")
        lines.append("")
        lines.append("| Layer | Metric | Score |")
        lines.append("|-------|--------|-------|")

        by_layer = report.aggregates.by_layer
        lines.append(f"| L1 | Entity Link F1 | {by_layer.get('l1_entity_link_f1', 0):.3f} |")
        lines.append(f"| L1 | Concept Map F1 | {by_layer.get('l1_concept_map_f1', 0):.3f} |")
        lines.append(f"| L2 | Context Recall | {by_layer.get('l2_context_recall', 0):.3f} |")
        lines.append(f"| L2 | MRR | {by_layer.get('l2_mrr', 0):.3f} |")
        lines.append(f"| L3 | Hits@k | {by_layer.get('l3_hits_at_k', 0):.3f} |")
        lines.append(f"| L3 | KG Edge F1 | {by_layer.get('l3_kg_edge_f1', 0):.3f} |")
        lines.append(
            f"| L4 | Violation Rate | {by_layer.get('l4_constraint_violation_rate', 0):.3f} |"
        )
        lines.append(
            f"| L4 | Type Consistency | {by_layer.get('l4_type_consistency_rate', 0):.3f} |"
        )
        lines.append(f"| L5 | Answer F1 | {by_layer.get('l5_answer_f1', 0):.3f} |")
        lines.append(f"| L5 | Groundedness | {by_layer.get('l5_groundedness', 0):.3f} |")
        lines.append("")

        # Top Failures
        if report.aggregates.top_fail_reasons:
            lines.append("## Top Failure Reasons")
            lines.append("")

            for tag, count in list(report.aggregates.top_fail_reasons.items())[:10]:
                description = FAIL_REASONS.get(tag, tag)
                lines.append(f"### {tag} ({count} items)")
                lines.append(f"*{description}*")
                lines.append("")

                # List affected items
                affected = [r for r in report.items if tag in r.fail_reason_tags][:5]
                for item in affected:
                    lines.append(f"- `{item.item_id}`: score={item.overall_score:.3f}")
                lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        self._add_recommendations(lines, report)

        # Write file
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown summary written to {path}")

    def _add_recommendations(self, lines: list[str], report: EvalReport) -> None:
        """Add recommendations based on metrics."""
        recs = []

        by_layer = report.aggregates.by_layer

        # L1 recommendations
        if by_layer.get("l1_entity_link_f1", 1.0) < 0.7:
            recs.append(
                "- **Improve entity extraction**: Consider expanding brand/product aliases "
                "or using fuzzy matching for entity linking"
            )

        # L2 recommendations
        if by_layer.get("l2_context_recall", 1.0) < 0.8:
            recs.append(
                "- **Improve document retrieval**: Consider expanding query terms, "
                "using hybrid search, or re-indexing with better chunking"
            )

        # L3 recommendations
        if by_layer.get("l3_hits_at_k", 1.0) < 0.8:
            recs.append(
                "- **Improve KG coverage**: Consider adding more entities to the "
                "knowledge graph or improving entity linking to KG nodes"
            )

        # L4 recommendations
        if by_layer.get("l4_constraint_violation_rate", 0.0) > 0.05:
            recs.append(
                "- **Fix ontology violations**: Review inference rules and ensure "
                "type constraints are properly enforced"
            )

        # L5 recommendations
        if by_layer.get("l5_answer_f1", 1.0) < 0.7:
            recs.append(
                "- **Improve answer generation**: Consider providing better context, "
                "adjusting prompts, or using more capable models"
            )

        if by_layer.get("l5_groundedness", 1.0) < 0.7:
            recs.append(
                "- **Improve groundedness**: Ensure answers cite retrieved context "
                "and avoid hallucination"
            )

        if not recs:
            recs.append("- No critical issues detected. Continue monitoring metrics.")

        lines.extend(recs)

    def _write_traces(self, results: list[ItemResult], traces_dir: Path) -> None:
        """Write individual traces to files."""
        traces_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            trace_path = traces_dir / f"{result.item_id}.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(
                    result.trace.model_dump(),
                    f,
                    indent=2,
                    default=str,
                    ensure_ascii=False,
                )

        logger.info(f"Traces written to {traces_dir}")


def generate_json_report(results: list[ItemResult], out_dir: Path) -> None:
    """Convenience function to generate JSON report."""
    generator = ReportGenerator()
    generator.generate_report(results, out_dir)


def generate_markdown_summary(results: list[ItemResult], out_dir: Path) -> None:
    """Convenience function to generate Markdown summary."""
    generator = ReportGenerator()
    generator.generate_report(results, out_dir)
