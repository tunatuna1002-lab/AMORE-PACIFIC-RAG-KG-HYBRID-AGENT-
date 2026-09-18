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
from eval.metrics.l3_kg import aggregate_l3_extended
from eval.metrics.l4_ontology import aggregate_l4_extended
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
        baseline_path: str | Path | None = None,
    ) -> EvalReport:
        """
        Generate full evaluation report.

        Args:
            results: List of evaluation results
            out_dir: Output directory
            baseline_path: Optional path to baseline report.json for regression analysis

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
        self._write_markdown_summary(report, out_dir / "summary.md", baseline_path=baseline_path)

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
        # 인프라 실패(타임아웃·API 오류)는 채점에서 분리한다. 답변을 얻지 못한
        # 문항을 0점으로 평균에 넣으면 지표가 모델 품질이 아니라 실행 환경을
        # 측정하게 된다. 비용은 실제로 쓴 만큼이므로 실패 문항도 합산한다.
        errored_results = [r for r in results if r.trace is not None and r.trace.error]
        error_item_ids = [r.item_id for r in errored_results]
        scored = [r for r in results if not (r.trace is not None and r.trace.error)]

        total = len(scored)
        if total == 0:
            return AggregateMetrics(
                total=0,
                passed=0,
                failed=0,
                errored=len(errored_results),
                error_item_ids=error_item_ids,
                pass_rate=0.0,
                avg_overall_score=0.0,
                avg_latency_ms=0.0,
                by_layer={},
                by_domain={},
                by_difficulty={},
                top_fail_reasons={},
            )

        results = scored
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # 선택 기능(비핵심) 실패는 채점을 막지 않는다 — errored와 달리 채점된
        # 문항에 포함되므로 몇 개가 저하된 채로 채점됐는지만 집계한다 (F3)
        degraded_items = sum(1 for r in results if r.trace is not None and r.trace.degraded)

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

        # Route / confidence distribution (route_trace, v4 전용 — 커밋 31040bf)
        route_counts, confidence_level_counts = self._compute_route_distribution(results)
        react_items = route_counts.get("react", 0)

        # 규칙 추론 발동 분포 (l4_ontology.inferences, v1/v4 공통)
        rule_fired_items, rule_inference_total = self._compute_rule_distribution(results)

        # [2026-09 사후] O0-A: 골드 엣지 있는 문항만의 L3·술어별 recall, 새 L4 지표
        ontology_metrics = {
            "l3": aggregate_l3_extended([r.l3 for r in results]),
            "l4": aggregate_l4_extended([r.l4 for r in results]),
        }
        for key, value in (
            ("l3_kg_edge_recall_gold_only", ontology_metrics["l3"]["kg_edge_recall_gold_only"]),
            (
                # [2026-09 사후] O0-추가: 술어 별칭(ownedBy↔ownedByGroup)·브랜드/그룹 표기를
                # 온톨로지 로더로 정식화한 뒤의 같은 지표. eval/metrics/l3_kg.py 참고.
                "l3_kg_edge_recall_gold_only_canonical",
                ontology_metrics["l3"]["kg_edge_recall_gold_only_canonical"],
            ),
            (
                "l4_rule_constraint_violation_rate",
                ontology_metrics["l4"]["rule_constraint_violation_rate"],
            ),
            ("l4_typed_consistency_rate", ontology_metrics["l4"]["typed_consistency_rate"]),
        ):
            if value is not None:
                by_layer[key] = value

        # 규칙 정답 일치 관측 (트랙 3-C) — rule_gold가 있는 문항만 대상
        rule_agreement_rate, rule_agreement_items = self._compute_rule_agreement_rate(results)
        non_fire_reason_top = self._compute_non_fire_reasons(results)

        # 답변 수치 검증 관측 (트랙 2-D) — 채점된 문항만
        (
            nv_items,
            nv_counts,
            nv_skipped,
            nv_items_unverified,
        ) = self._compute_numeric_verification(results)

        # Cost aggregation
        total_tokens = 0
        total_cost_usd = 0.0
        cost_by_layer: dict[str, float] = defaultdict(float)

        # 비용만은 실패 문항 포함 — 토큰은 실제로 소비됐다
        for r in results + errored_results:
            if r.trace and r.trace.cost:
                c = r.trace.cost
                total_tokens += c.total_tokens
                total_cost_usd += c.total_cost_usd
                cost_by_layer["l1"] += c.l1_cost_usd
                cost_by_layer["l2"] += c.l2_cost_usd
                cost_by_layer["l3"] += c.l3_cost_usd
                cost_by_layer["l4"] += c.l4_cost_usd
                cost_by_layer["l5"] += c.l5_cost_usd
                cost_by_layer["judge"] += c.judge_cost_usd

        cost_items = total + len(errored_results)
        return AggregateMetrics(
            total=total,
            passed=passed,
            failed=failed,
            errored=len(errored_results),
            error_item_ids=error_item_ids,
            degraded_items=degraded_items,
            pass_rate=passed / total,
            avg_overall_score=avg_score,
            avg_latency_ms=avg_latency,
            by_layer=by_layer,
            by_domain=by_domain,
            by_difficulty=by_difficulty,
            top_fail_reasons=top_fail_reasons,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            avg_tokens_per_item=total_tokens / cost_items if cost_items else 0.0,
            avg_cost_per_item_usd=total_cost_usd / cost_items if cost_items else 0.0,
            cost_by_layer=dict(cost_by_layer),
            route_counts=route_counts,
            confidence_level_counts=confidence_level_counts,
            react_items=react_items,
            rule_fired_items=rule_fired_items,
            rule_inference_total=rule_inference_total,
            rule_agreement_rate=rule_agreement_rate,
            rule_agreement_items=rule_agreement_items,
            non_fire_reason_top=non_fire_reason_top,
            ontology_metrics=ontology_metrics,
            numeric_verification_items=nv_items,
            numeric_verification_counts=nv_counts,
            numeric_verification_skipped=nv_skipped,
            numeric_verification_items_with_unverified=nv_items_unverified,
        )

    NUMERIC_VERIFICATION_KEYS = (
        "checked",
        "verified",
        "mismatch",
        "no_citation",
        "unknown_card",
        "replaced",
        "found_in_other_cards",
    )

    @classmethod
    def _compute_numeric_verification(
        cls, results: list[ItemResult]
    ) -> tuple[int, dict[str, int], dict[str, int], int]:
        """trace.numeric_verification(트랙 2-D)을 합산한다.

        반환: (검증기가 실행된 문항 수, skipped 아닌 문항의 개수 합계, skipped 사유별 문항 수,
        mismatch·unknown_card가 있는 문항 수). 결과가 없는 문항(v1, 플래그 off)은 건너뛴다.
        """
        items = 0
        counts: dict[str, int] = dict.fromkeys(cls.NUMERIC_VERIFICATION_KEYS, 0)
        skipped: dict[str, int] = defaultdict(int)
        with_unverified = 0
        for r in results:
            nv = r.trace.numeric_verification if r.trace is not None else None
            if not isinstance(nv, dict):
                continue
            items += 1
            if nv.get("skipped"):
                skipped[str(nv["skipped"])] += 1
                continue
            for key in cls.NUMERIC_VERIFICATION_KEYS:
                counts[key] += int(nv.get(key) or 0)
            if int(nv.get("mismatch") or 0) or int(nv.get("unknown_card") or 0):
                with_unverified += 1
        return items, counts, dict(skipped), with_unverified

    @staticmethod
    def _compute_route_distribution(
        results: list[ItemResult],
    ) -> tuple[dict[str, int], dict[str, int]]:
        """trace.route_trace에서 route·confidence_level 분포를 센다.

        route_trace가 없는 문항(v1 경로, 구형 트레이스)은 조용히 건너뛴다 —
        인프라 실패 제외는 호출부(scored된 results만 넘어옴)에서 이미 처리됐다.
        """
        route_counts: dict[str, int] = defaultdict(int)
        confidence_level_counts: dict[str, int] = defaultdict(int)

        for r in results:
            trace = r.trace.route_trace if r.trace is not None else None
            if not isinstance(trace, dict):
                continue
            route = trace.get("route")
            if route:
                route_counts[route] += 1
            level = trace.get("confidence_level")
            if level:
                confidence_level_counts[level] += 1

        return dict(route_counts), dict(confidence_level_counts)

    @staticmethod
    def _compute_rule_distribution(results: list[ItemResult]) -> tuple[int, int]:
        """l4_ontology.inferences가 비어있지 않은 문항 수와 총 inferences 개수."""
        fired_items = 0
        total_inferences = 0

        for r in results:
            if r.trace is None:
                continue
            inferences = r.trace.l4_ontology.inferences
            if inferences:
                fired_items += 1
                total_inferences += len(inferences)

        return fired_items, total_inferences

    @staticmethod
    def _compute_rule_agreement_rate(results: list[ItemResult]) -> tuple[float | None, int]:
        """ItemResult.rule_agreement(트랙 3-C)가 판정된 문항 중 일치 비율과 판정 문항 수.

        rule_agreement가 None인 문항(rule_gold 없음, 구형 리포트)은 제외한다.
        판정 가능한 문항이 하나도 없으면 (None, 0).
        """
        judged = [r.rule_agreement for r in results if r.rule_agreement is not None]
        if not judged:
            return None, 0
        return sum(1 for v in judged if v) / len(judged), len(judged)

    @staticmethod
    def _compute_non_fire_reasons(
        results: list[ItemResult], top_n: int = 15
    ) -> list[tuple[str, int]]:
        """채점된 문항의 trace.rule_evaluation.non_fire_top을 라벨별로 합산해 상위 top_n개.

        rule_evaluation이 없는 문항(3-B 미병합, v1 구형)은 조용히 건너뛴다.
        """
        counts: dict[str, int] = defaultdict(int)
        for r in results:
            rule_evaluation = r.trace.rule_evaluation if r.trace is not None else None
            if not isinstance(rule_evaluation, dict):
                continue
            for entry in rule_evaluation.get("non_fire_top") or []:
                if isinstance(entry, list | tuple) and len(entry) == 2:
                    label, count = entry
                    counts[str(label)] += int(count)
        return sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]

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
            "l2_context_recall_doc": sum(r.l2.context_recall_at_k_doc for r in results) / n,
            "l2_context_recall_concept": sum(r.l2.context_recall_at_k_concept for r in results) / n,
            # L3
            "l3_hits_at_k": sum(r.l3.hits_at_k for r in results) / n,
            "l3_kg_edge_f1": sum(r.l3.kg_edge_f1 for r in results) / n,
            "l3_kg_edge_recall": sum(r.l3.kg_edge_recall for r in results) / n,
            "l3_kg_edge_precision": sum(r.l3.kg_edge_precision for r in results) / n,
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
            # 수치 정확도는 expected_values가 있는 문항에서만 계산되므로
            # 그 문항들만의 평균으로 보고한다 (없으면 0.0)
            "l5_numeric_accuracy": (
                sum(r.l5.numeric_accuracy for r in results if r.l5.numeric_accuracy is not None)
                / max(1, sum(1 for r in results if r.l5.numeric_accuracy is not None))
            ),
            "l5_numeric_accuracy_items": float(
                sum(1 for r in results if r.l5.numeric_accuracy is not None)
            ),
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

    def _write_markdown_summary(
        self,
        report: EvalReport,
        path: Path,
        baseline_path: str | Path | None = None,
    ) -> None:
        """Write Markdown summary to file."""
        lines = []

        # Header
        lines.append("# Evaluation Summary")
        lines.append("")
        lines.append(f"**Generated**: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Scored Items**: {report.aggregates.total}")
        if report.aggregates.errored:
            lines.append(
                f"**Excluded (infra failure)**: {report.aggregates.errored} "
                f"— {', '.join(report.aggregates.error_item_ids[:10])}"
            )
        if report.aggregates.degraded_items:
            lines.append(
                f"**Degraded (scored, optional feature failed)**: {report.aggregates.degraded_items}"
            )
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
        if report.aggregates.errored:
            lines.append(f"| Excluded (infra) | {report.aggregates.errored} |")
        if report.aggregates.degraded_items:
            lines.append(
                f"| Degraded (optional feature failed) | {report.aggregates.degraded_items} |"
            )
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
        lines.append(
            f"| L2 | Context Recall (concept) | "
            f"{by_layer.get('l2_context_recall_concept', 0):.3f} |"
        )
        lines.append(
            f"| L2 | Context Recall (doc) | {by_layer.get('l2_context_recall_doc', 0):.3f} |"
        )
        lines.append(f"| L2 | MRR | {by_layer.get('l2_mrr', 0):.3f} |")
        lines.append(f"| L3 | Hits@k | {by_layer.get('l3_hits_at_k', 0):.3f} |")
        lines.append(f"| L3 | KG Edge F1 | {by_layer.get('l3_kg_edge_f1', 0):.3f} |")
        lines.append(f"| L3 | KG Edge Recall | {by_layer.get('l3_kg_edge_recall', 0):.3f} |")
        lines.append(f"| L3 | KG Edge Precision | {by_layer.get('l3_kg_edge_precision', 0):.3f} |")
        lines.append(
            f"| L4 | Violation Rate | {by_layer.get('l4_constraint_violation_rate', 0):.3f} |"
        )
        lines.append(
            f"| L4 | Type Consistency | {by_layer.get('l4_type_consistency_rate', 0):.3f} |"
        )
        for key, label in (
            ("l3_kg_edge_recall_gold_only", "| L3 | KG Edge Recall (gold-edge items) |"),
            (
                "l3_kg_edge_recall_gold_only_canonical",
                "| L3 | KG Edge Recall (gold-edge items, canonical) |",
            ),
            ("l4_rule_constraint_violation_rate", "| L4 | Rule Constraint Violation |"),
            ("l4_typed_consistency_rate", "| L4 | Typed Consistency |"),
        ):
            if key in by_layer:
                lines.append(f"{label} {by_layer[key]:.3f} |")
        lines.append(f"| L5 | Answer F1 | {by_layer.get('l5_answer_f1', 0):.3f} |")
        lines.append(f"| L5 | Groundedness | {by_layer.get('l5_groundedness', 0):.3f} |")
        if by_layer.get("l5_numeric_accuracy_items", 0):
            lines.append(
                f"| L5 | Numeric Accuracy | {by_layer.get('l5_numeric_accuracy', 0):.3f} "
                f"({int(by_layer['l5_numeric_accuracy_items'])} items) |"
            )
        lines.append("")

        # Route / Confidence (route_trace 기반, v4 전용 — 데이터 없으면 생략)
        if report.aggregates.route_counts:
            lines.append("## Route / Confidence")
            lines.append("")
            lines.append("| Route | Count |")
            lines.append("|-------|-------|")
            for route, count in sorted(report.aggregates.route_counts.items(), key=lambda x: -x[1]):
                lines.append(f"| {route} | {count} |")
            lines.append("")
            if report.aggregates.confidence_level_counts:
                lines.append("| Confidence Level | Count |")
                lines.append("|------------------|-------|")
                for level, count in sorted(
                    report.aggregates.confidence_level_counts.items(), key=lambda x: -x[1]
                ):
                    lines.append(f"| {level} | {count} |")
                lines.append("")

        # Numeric Verification (트랙 2-D, 검증기가 실행된 문항이 없으면 생략)
        if report.aggregates.numeric_verification_items:
            agg_nv = report.aggregates
            lines.append("## Numeric Verification")
            lines.append("")
            lines.append(
                f"Items verified: {agg_nv.numeric_verification_items} "
                f"(with mismatch/unknown_card: {agg_nv.numeric_verification_items_with_unverified}"
                f", skipped: {dict(agg_nv.numeric_verification_skipped) or 0})"
            )
            lines.append("")
            lines.append("| Class | Count |")
            lines.append("|-------|-------|")
            for key, count in agg_nv.numeric_verification_counts.items():
                lines.append(f"| {key} | {count} |")
            lines.append("")

        # Rules (규칙 엔진 추론 관측 — 발화 여부는 v1/v4 공통, 정답 일치는 rule_gold가
        # 있는 문항만. 데이터가 전혀 없으면(비-rule 데이터셋) 섹션을 생략한다)
        agg = report.aggregates
        if agg.rule_fired_items or agg.rule_agreement_items or agg.non_fire_reason_top:
            lines.append("## Rules")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Rule-fired items | {agg.rule_fired_items} |")
            lines.append(f"| Rule inferences (total) | {agg.rule_inference_total} |")
            if agg.rule_agreement_rate is not None:
                lines.append(
                    f"| Rule agreement rate | {agg.rule_agreement_rate:.1%} "
                    f"({agg.rule_agreement_items} judged) |"
                )
            else:
                lines.append("| Rule agreement rate | — (no judged items) |")
            lines.append("")

            if agg.non_fire_reason_top:
                lines.append("**Top non-fire reasons** (top 10)")
                lines.append("")
                lines.append("| Reason | Count |")
                lines.append("|--------|-------|")
                for label, count in agg.non_fire_reason_top[:10]:
                    lines.append(f"| {label} | {count} |")
                lines.append("")

        # Cost Summary (only if cost data exists)
        if report.aggregates.total_tokens > 0:
            lines.append("## Cost Summary")
            lines.append("")
            lines.append("| Layer | Tokens | Cost (USD) |")
            lines.append("|-------|--------|-----------|")

            layer_names = ["l1", "l2", "l3", "l4", "l5", "judge"]
            layer_token_fields = {
                "l1": "l1_tokens",
                "l2": "l2_tokens",
                "l3": "l3_tokens",
                "l4": "l4_tokens",
                "l5": "l5_tokens",
                "judge": "judge_tokens",
            }
            for layer in layer_names:
                cost = report.aggregates.cost_by_layer.get(layer, 0.0)
                # Sum tokens per layer from individual items
                token_field = layer_token_fields[layer]
                tokens = sum(
                    getattr(r.trace.cost, token_field, 0)
                    for r in report.items
                    if r.trace and r.trace.cost
                )
                if tokens > 0 or cost > 0:
                    lines.append(f"| {layer.upper()} | {tokens} | ${cost:.4f} |")

            lines.append(
                f"| **Total** | **{report.aggregates.total_tokens}** "
                f"| **${report.aggregates.total_cost_usd:.4f}** |"
            )
            lines.append("")
            lines.append(
                f"Average: {report.aggregates.avg_tokens_per_item:.0f} tokens "
                f"/ ${report.aggregates.avg_cost_per_item_usd:.4f} per item"
            )
            lines.append("")

            # Pricing used (same for the whole run; pulled from the first item
            # that carries pricing metadata). Source is "litellm" or
            # "fallback_table" — see eval/cost_tracker.py resolve_llm_pricing.
            pricing = next(
                (
                    r.trace.cost.pricing
                    for r in report.items
                    if r.trace and r.trace.cost and r.trace.cost.pricing
                ),
                {},
            )
            if pricing:
                lines.append("Pricing used (USD per 1M tokens):")
                lines.append("")
                for model, rates in pricing.items():
                    lines.append(
                        f"- `{model}`: input ${rates.get('input_per_1m_usd', 0):.4f}, "
                        f"output ${rates.get('output_per_1m_usd', 0):.4f} "
                        f"(source: {rates.get('source', 'unknown')})"
                    )
                lines.append("")

        # Regression Analysis (only if baseline provided)
        if baseline_path:
            self._write_regression_section(lines, report, baseline_path)

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

    def _write_regression_section(
        self,
        lines: list[str],
        report: EvalReport,
        baseline_path: str | Path,
    ) -> None:
        """Add regression analysis section comparing against a baseline."""
        from eval.regression import RegressionTester

        baseline_path = Path(baseline_path)
        if not baseline_path.exists():
            lines.append("## Regression Analysis")
            lines.append("")
            lines.append(f"*Baseline not found: {baseline_path}*")
            lines.append("")
            return

        try:
            # Determine baseline_dir and baseline_name from path
            # baseline_path can be a report.json or a baseline directory
            if baseline_path.is_file():
                baseline_dir = baseline_path.parent.parent
                baseline_name = baseline_path.parent.name
            else:
                baseline_dir = baseline_path.parent
                baseline_name = baseline_path.name

            tester = RegressionTester(baseline_dir=baseline_dir)
            comparison = tester.compare(baseline_name, report)

            lines.append("## Regression Analysis")
            lines.append("")
            lines.append(f"**Baseline**: {comparison.baseline_name}")
            lines.append("")

            if not comparison.has_regressions:
                lines.append("No regressions detected.")
            else:
                lines.append(f"**{comparison.regression_count} regression(s) detected**")
                lines.append("")
                lines.append("| Metric | Baseline | Current | Delta | Severity |")
                lines.append("|--------|----------|---------|-------|----------|")
                for r in comparison.metric_results:
                    if r.is_regression:
                        lines.append(
                            f"| {r.metric_name} | {r.baseline_value:.4f} "
                            f"| {r.current_value:.4f} | {r.delta:+.4f} "
                            f"| {r.severity} |"
                        )
                lines.append("")

            if comparison.new_failures:
                lines.append(f"**New failures**: {len(comparison.new_failures)}")
                for item_id in comparison.new_failures[:5]:
                    lines.append(f"- {item_id}")
                lines.append("")

            if comparison.fixed_items:
                lines.append(f"**Fixed items**: {len(comparison.fixed_items)}")
                for item_id in comparison.fixed_items[:5]:
                    lines.append(f"- {item_id}")
                lines.append("")

        except Exception as e:
            lines.append("## Regression Analysis")
            lines.append("")
            lines.append(f"*Error loading baseline: {e}*")
            lines.append("")

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
        if by_layer.get("l2_context_recall_concept", 1.0) < 0.8:
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
