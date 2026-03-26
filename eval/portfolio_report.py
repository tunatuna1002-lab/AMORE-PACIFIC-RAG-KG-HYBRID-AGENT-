"""
Portfolio Report Generator
==========================
Generates a portfolio-grade evaluation report with quantitative metrics
and publication-quality visualizations.

Usage:
    # From existing report.json (offline, no API calls)
    python -m eval portfolio --report eval_output/report.json \\
        --dataset eval/data/golden/laneige_golden_v1.jsonl --out portfolio_output

    # Programmatic
    from eval.portfolio_report import PortfolioReportGenerator
    gen = PortfolioReportGenerator()
    gen.generate(report, dataset_path, out_dir)

Output:
    portfolio_output/
    ├── portfolio_report.md       # Main Markdown report
    ├── metrics_summary.json      # Structured metrics JSON
    └── charts/                   # PNG visualizations (6 charts)
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.loader import load_dataset
from eval.portfolio_charts import PortfolioChartGenerator
from eval.schemas import EvalConfig, EvalItem, EvalReport, ItemResult

logger = logging.getLogger(__name__)


class PortfolioReportGenerator:
    """포트폴리오/이력서용 RAG 평가 보고서 생성기"""

    def __init__(self, config: EvalConfig | None = None, dpi: int = 200):
        self.config = config or EvalConfig()
        self.dpi = dpi

    def generate(
        self,
        report: EvalReport,
        dataset_path: str | Path | None = None,
        out_dir: str | Path = "./portfolio_output",
        title: str = "RAG-KG Hybrid Agent Evaluation Report",
    ) -> Path:
        """
        포트폴리오 보고서 생성 메인 엔트리.

        Args:
            report: 기존 평가 결과 (EvalReport)
            dataset_path: 골든셋 경로 (Confusion Matrix 계산용)
            out_dir: 출력 디렉토리
            title: 보고서 제목

        Returns:
            생성된 보고서 디렉토리 경로
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. 골든셋 로드 (Confusion Matrix용)
        gold_map: dict[str, EvalItem] = {}
        if dataset_path:
            items = load_dataset(dataset_path)
            gold_map = {item.id: item for item in items}
            logger.info(f"Loaded {len(gold_map)} golden items from {dataset_path}")

        # 2. 포트폴리오 메트릭 계산
        metrics = self._compute_portfolio_metrics(report, gold_map)

        # 3. 차트 생성
        chart_gen = PortfolioChartGenerator(output_dir=out_dir, dpi=self.dpi)
        chart_data = self._prepare_chart_data(report, metrics)
        charts = chart_gen.generate_all(chart_data)

        # 4. Markdown 보고서 생성
        self._generate_markdown(report, metrics, charts, out_dir / "portfolio_report.md", title)

        # 5. 메트릭 요약 JSON 저장
        self._save_metrics_json(metrics, out_dir / "metrics_summary.json")

        logger.info(f"Portfolio report generated at {out_dir}")
        return out_dir

    # =========================================================================
    # Metrics Computation
    # =========================================================================

    def _compute_portfolio_metrics(
        self, report: EvalReport, gold_map: dict[str, EvalItem]
    ) -> dict[str, Any]:
        """모든 포트폴리오 지표 계산."""
        items = report.items

        # Confusion Matrix (L2 Retrieval)
        cm = self._compute_confusion_matrix(items, gold_map, k=self.config.top_k)

        # NDCG@K
        ndcg = self._compute_ndcg(items, gold_map, k=self.config.top_k)

        # Layer scores (L1-L5 평균)
        by_layer = report.aggregates.by_layer
        layer_scores = {
            "L1 Query\nUnderstanding": _avg([
                by_layer.get("l1_entity_link_f1", 0),
                by_layer.get("l1_concept_map_f1", 0),
            ]),
            "L2 Document\nRetrieval": _avg([
                by_layer.get("l2_context_recall", 0),
                by_layer.get("l2_context_precision", 0),
                by_layer.get("l2_mrr", 0),
            ]),
            "L3 Knowledge\nGraph": _avg([
                by_layer.get("l3_hits_at_k", 0),
                by_layer.get("l3_kg_edge_f1", 0),
            ]),
            "L4 Ontology\nCompliance": _avg([
                1.0 - by_layer.get("l4_constraint_violation_rate", 0),
                by_layer.get("l4_type_consistency_rate", 0),
            ]),
            "L5 Answer\nQuality": _avg([
                by_layer.get("l5_answer_f1", 0),
                by_layer.get("l5_groundedness", 0),
                by_layer.get("l5_relevance", 0),
            ]),
        }

        # Precision/Recall/F1 per layer
        pr_metrics = self._compute_pr_per_layer(report, cm)

        # Overall scores list
        scores = [r.overall_score for r in items]

        return {
            "confusion_matrix": cm,
            "ndcg": ndcg,
            "layer_scores": layer_scores,
            "pr_metrics": pr_metrics,
            "scores": scores,
            "by_layer": by_layer,
            "by_domain": report.aggregates.by_domain,
            "by_difficulty": report.aggregates.by_difficulty,
            "summary": {
                "total": report.aggregates.total,
                "passed": report.aggregates.passed,
                "failed": report.aggregates.failed,
                "pass_rate": report.aggregates.pass_rate,
                "avg_score": report.aggregates.avg_overall_score,
                "avg_latency_ms": report.aggregates.avg_latency_ms,
            },
        }

    def _compute_confusion_matrix(
        self,
        items: list[ItemResult],
        gold_map: dict[str, EvalItem],
        k: int = 8,
    ) -> dict[str, Any]:
        """L2 검색 Confusion Matrix 계산."""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        items_evaluated = 0

        for item in items:
            if item.item_id not in gold_map:
                continue
            gold = gold_map[item.item_id]
            gold_chunks = set(gold.gold.doc_chunk_ids)
            if not gold_chunks:
                continue  # 골든 청크 없으면 제외

            retrieved: set[str] = set()
            if item.trace and item.trace.l2_doc_retrieval:
                retrieved = set(item.trace.l2_doc_retrieval.chunk_ids[:k])

            tp = len(gold_chunks & retrieved)
            fp = len(retrieved - gold_chunks)
            fn = len(gold_chunks - retrieved)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            items_evaluated += 1

        # TN은 RAG에서 직접 측정 불가 — 전체 청크 수 기반 추정
        # 여기서는 실질적인 TP/FP/FN만 보고
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": 0,  # RAG 특성상 TN은 미정의 (전체 코퍼스 규모)
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "items_evaluated": items_evaluated,
        }

    def _compute_ndcg(
        self,
        items: list[ItemResult],
        gold_map: dict[str, EvalItem],
        k: int = 8,
    ) -> float:
        """NDCG@K 계산."""
        ndcg_scores = []

        for item in items:
            if item.item_id not in gold_map:
                continue
            gold = gold_map[item.item_id]
            gold_chunks = set(gold.gold.doc_chunk_ids)
            if not gold_chunks:
                continue

            retrieved: list[str] = []
            if item.trace and item.trace.l2_doc_retrieval:
                retrieved = item.trace.l2_doc_retrieval.chunk_ids[:k]

            # Relevance vector
            relevance = [1.0 if c in gold_chunks else 0.0 for c in retrieved]

            # DCG
            dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))

            # IDCG (perfect ordering)
            ideal_len = min(len(gold_chunks), k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return _avg(ndcg_scores) if ndcg_scores else 0.0

    def _compute_pr_per_layer(
        self, report: EvalReport, cm: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """각 평가 레이어별 Precision/Recall/F1."""
        by_layer = report.aggregates.by_layer

        # L1: Entity Linking
        l1_f1 = by_layer.get("l1_entity_link_f1", 0)
        # F1에서 P, R 추정 (개별 값이 없으므로 대칭 가정)
        l1_p = l1_f1  # 근사치
        l1_r = l1_f1

        # L2: Retrieval (Confusion Matrix에서 직접 계산)
        l2_p = cm["precision"]
        l2_r = cm["recall"]
        l2_f1 = cm["f1"]

        # L3: KG
        l3_f1 = by_layer.get("l3_kg_edge_f1", 0)
        l3_hits = by_layer.get("l3_hits_at_k", 0)

        # L5: Answer Quality
        l5_f1 = by_layer.get("l5_answer_f1", 0)

        return {
            "L1 Entity Linking": {"precision": l1_p, "recall": l1_r, "f1": l1_f1},
            "L2 Doc Retrieval": {"precision": l2_p, "recall": l2_r, "f1": l2_f1},
            "L3 KG Edges": {"precision": l3_f1, "recall": l3_hits, "f1": l3_f1},
            "L5 Answer": {"precision": l5_f1, "recall": l5_f1, "f1": l5_f1},
        }

    # =========================================================================
    # Chart Data Preparation
    # =========================================================================

    def _prepare_chart_data(
        self, report: EvalReport, metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """차트 생성기에 전달할 데이터 준비."""
        return {
            "layer_scores": metrics["layer_scores"],
            "confusion_matrix": metrics["confusion_matrix"],
            "pr_metrics": metrics["pr_metrics"],
            "scores": metrics["scores"],
            "by_domain": metrics["by_domain"],
            "by_difficulty": metrics["by_difficulty"],
        }

    # =========================================================================
    # Markdown Report Generation
    # =========================================================================

    def _generate_markdown(
        self,
        report: EvalReport,
        metrics: dict[str, Any],
        charts: dict[str, Path],
        out_path: Path,
        title: str,
    ) -> None:
        """포트폴리오급 Markdown 보고서 생성."""
        lines: list[str] = []
        summary = metrics["summary"]
        by_layer = metrics["by_layer"]
        cm = metrics["confusion_matrix"]

        # ===== Header =====
        lines.append(f"# {title}")
        lines.append("> AMOREPACIFIC LANEIGE Brand Intelligence System")
        lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # ===== Executive Summary =====
        lines.append("## 1. Executive Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Test Cases | {summary['total']} |")
        lines.append(f"| Pass Rate | {summary['pass_rate']:.1%} |")
        lines.append(f"| Average Score | {summary['avg_score']:.3f} |")
        lines.append(f"| Avg Latency | {summary['avg_latency_ms']:.0f}ms |")
        lines.append(f"| Retrieval Precision | {cm['precision']:.3f} |")
        lines.append(f"| Retrieval Recall | {cm['recall']:.3f} |")
        lines.append(f"| Retrieval F1 | {cm['f1']:.3f} |")
        lines.append(f"| NDCG@{self.config.top_k} | {metrics['ndcg']:.3f} |")
        lines.append("")

        # ===== Methodology =====
        lines.append("## 2. Evaluation Methodology")
        lines.append("")
        lines.append("### 2.1 5-Layer Evaluation Architecture")
        lines.append("")
        lines.append("| Layer | Evaluation Target | Key Metrics | Weight |")
        lines.append("|-------|-------------------|-------------|--------|")
        lines.append(
            "| **L1** | Query Understanding | Entity Link F1, Concept Map F1 | 10% |"
        )
        lines.append(
            "| **L2** | Document Retrieval | Context Recall@K, Precision@K, MRR | 35% |"
        )
        lines.append(
            "| **L3** | Knowledge Graph | Hits@K, KG Edge F1 | (combined with L2) |"
        )
        lines.append(
            "| **L4** | Ontology Compliance | Constraint Violation Rate, Type Consistency | 10% |"
        )
        lines.append(
            "| **L5** | Answer Quality | Token F1, Groundedness, Relevance | 45% |"
        )
        lines.append("")
        lines.append("### 2.2 System Architecture")
        lines.append("")
        lines.append("```")
        lines.append("Query → Embedding (OpenAI text-embedding-3-small)")
        lines.append("     → ChromaDB Cosine Similarity Search")
        lines.append("     → RRF Fusion (Dense + BM25 Sparse)")
        lines.append("     → Knowledge Graph + Ontology Reasoning")
        lines.append("     → Confidence Fusion (Vector 40% + Ontology 35% + Entity 25%)")
        lines.append("     → LLM Answer Generation (GPT-4.1-mini)")
        lines.append("```")
        lines.append("")

        # ===== Multi-Layer Performance =====
        lines.append("## 3. Multi-Layer Performance")
        lines.append("")
        if "radar" in charts:
            lines.append(f"![L1-L5 Radar Chart](charts/{charts['radar'].name})")
            lines.append("")

        lines.append("| Layer | Metric | Score | Threshold | Status |")
        lines.append("|-------|--------|-------|-----------|--------|")
        self._add_metric_row(
            lines, "L1", "Entity Link F1",
            by_layer.get("l1_entity_link_f1", 0), 0.50,
        )
        self._add_metric_row(
            lines, "L1", "Concept Map F1",
            by_layer.get("l1_concept_map_f1", 0), 0.50,
        )
        self._add_metric_row(
            lines, "L2", "Context Recall@K",
            by_layer.get("l2_context_recall", 0), 0.80,
        )
        self._add_metric_row(
            lines, "L2", "Context Precision@K",
            by_layer.get("l2_context_precision", 0), None,
        )
        self._add_metric_row(
            lines, "L2", "MRR",
            by_layer.get("l2_mrr", 0), None,
        )
        self._add_metric_row(
            lines, "L3", "Hits@K",
            by_layer.get("l3_hits_at_k", 0), 0.80,
        )
        self._add_metric_row(
            lines, "L3", "KG Edge F1",
            by_layer.get("l3_kg_edge_f1", 0), 0.50,
        )
        self._add_metric_row(
            lines, "L4", "Violation Rate",
            by_layer.get("l4_constraint_violation_rate", 0), 0.05, lower_is_better=True,
        )
        self._add_metric_row(
            lines, "L4", "Type Consistency",
            by_layer.get("l4_type_consistency_rate", 0), 0.90,
        )
        self._add_metric_row(
            lines, "L5", "Answer F1",
            by_layer.get("l5_answer_f1", 0), 0.50,
        )
        self._add_metric_row(
            lines, "L5", "Groundedness",
            by_layer.get("l5_groundedness", 0), 0.70,
        )
        self._add_metric_row(
            lines, "L5", "Relevance",
            by_layer.get("l5_relevance", 0), 0.70,
        )
        lines.append("")

        # ===== Retrieval Quality =====
        lines.append("## 4. Retrieval Quality Analysis (L2)")
        lines.append("")

        if "confusion_matrix" in charts:
            lines.append(
                f"![Confusion Matrix](charts/{charts['confusion_matrix'].name})"
            )
            lines.append("")

        lines.append("### 4.1 Confusion Matrix")
        lines.append("")
        lines.append(f"- **True Positives (TP)**: {cm['tp']} — "
                      "relevant documents correctly retrieved")
        lines.append(f"- **False Positives (FP)**: {cm['fp']} — "
                      "irrelevant documents incorrectly retrieved")
        lines.append(f"- **False Negatives (FN)**: {cm['fn']} — "
                      "relevant documents missed")
        lines.append(f"- **Items Evaluated**: {cm['items_evaluated']} "
                      "(queries with gold doc_chunk_ids)")
        lines.append("")

        lines.append("### 4.2 Information Retrieval Metrics")
        lines.append("")
        lines.append("| Metric | Value | Description |")
        lines.append("|--------|-------|-------------|")
        lines.append(
            f"| **Precision** | {cm['precision']:.3f} | "
            "Fraction of retrieved docs that are relevant |"
        )
        lines.append(
            f"| **Recall** | {cm['recall']:.3f} | "
            "Fraction of relevant docs that were retrieved |"
        )
        lines.append(
            f"| **F1 Score** | {cm['f1']:.3f} | "
            "Harmonic mean of Precision and Recall |"
        )
        lines.append(
            f"| **NDCG@{self.config.top_k}** | {metrics['ndcg']:.3f} | "
            "Normalized Discounted Cumulative Gain |"
        )
        lines.append(
            f"| **MRR** | {by_layer.get('l2_mrr', 0):.3f} | "
            "Mean Reciprocal Rank of first relevant doc |"
        )
        lines.append("")

        if "precision_recall" in charts:
            lines.append(
                f"![Precision/Recall/F1](charts/{charts['precision_recall'].name})"
            )
            lines.append("")

        # ===== Answer Quality =====
        lines.append("## 5. Answer Quality Analysis (L5)")
        lines.append("")
        lines.append("| Metric | Score | Description |")
        lines.append("|--------|-------|-------------|")
        lines.append(
            f"| Exact Match | {by_layer.get('l5_exact_match', 0):.3f} | "
            "Exact answer match rate |"
        )
        lines.append(
            f"| Token F1 | {by_layer.get('l5_answer_f1', 0):.3f} | "
            "Token-level overlap with gold answer |"
        )
        lines.append(
            f"| Groundedness | {by_layer.get('l5_groundedness', 0):.3f} | "
            "Answer grounded in retrieved context |"
        )
        lines.append(
            f"| Relevance | {by_layer.get('l5_relevance', 0):.3f} | "
            "Answer relevance to the question |"
        )
        lines.append("")

        if "score_distribution" in charts:
            lines.append(
                f"![Score Distribution](charts/{charts['score_distribution'].name})"
            )
            lines.append("")

        # ===== Domain Analysis =====
        lines.append("## 6. Domain Analysis")
        lines.append("")

        if metrics["by_domain"]:
            lines.append("| Domain | Count | Pass Rate | Avg Score |")
            lines.append("|--------|-------|-----------|-----------|")
            for domain, data in sorted(metrics["by_domain"].items()):
                count = int(data.get("count", 0))
                pr = data.get("pass_rate", 0)
                avg = data.get("avg_score", 0)
                lines.append(f"| {domain} | {count} | {pr:.1%} | {avg:.3f} |")
            lines.append("")

        if "domain_breakdown" in charts:
            lines.append(
                f"![Domain Breakdown](charts/{charts['domain_breakdown'].name})"
            )
            lines.append("")

        # ===== Difficulty Analysis =====
        lines.append("## 7. Difficulty Analysis")
        lines.append("")

        if metrics["by_difficulty"]:
            lines.append("| Difficulty | Count | Pass Rate | Avg Score |")
            lines.append("|------------|-------|-----------|-----------|")
            for diff in ["easy", "medium", "hard"]:
                if diff in metrics["by_difficulty"]:
                    data = metrics["by_difficulty"][diff]
                    count = int(data.get("count", 0))
                    pr = data.get("pass_rate", 0)
                    avg = data.get("avg_score", 0)
                    lines.append(f"| {diff.capitalize()} | {count} | {pr:.1%} | {avg:.3f} |")
            lines.append("")

        if "difficulty_comparison" in charts:
            lines.append(
                f"![Difficulty Comparison](charts/{charts['difficulty_comparison'].name})"
            )
            lines.append("")

        # ===== Failure Analysis =====
        if report.aggregates.top_fail_reasons:
            lines.append("## 8. Failure Analysis")
            lines.append("")
            lines.append("| Failure Reason | Count | Percentage |")
            lines.append("|----------------|-------|------------|")
            total = max(summary["total"], 1)
            for reason, count in list(report.aggregates.top_fail_reasons.items())[:10]:
                pct = count / total
                lines.append(f"| {reason} | {count} | {pct:.1%} |")
            lines.append("")

        # ===== Conclusion =====
        section_num = 9 if report.aggregates.top_fail_reasons else 8
        lines.append(f"## {section_num}. Conclusion")
        lines.append("")
        lines.append(
            f"This RAG-KG Hybrid Agent was evaluated across **{summary['total']} queries** "
            f"spanning 5 domains and 3 difficulty levels."
        )
        lines.append("")

        # 핵심 성과 요약
        lines.append("### Key Results")
        lines.append("")
        lines.append(f"- **Overall Pass Rate**: {summary['pass_rate']:.1%}")
        lines.append(f"- **Average Score**: {summary['avg_score']:.3f}")
        lines.append(
            f"- **Retrieval F1**: {cm['f1']:.3f} "
            f"(Precision {cm['precision']:.3f}, Recall {cm['recall']:.3f})"
        )
        lines.append(f"- **NDCG@{self.config.top_k}**: {metrics['ndcg']:.3f}")
        lines.append(f"- **Answer Token F1**: {by_layer.get('l5_answer_f1', 0):.3f}")
        lines.append("")

        lines.append("### Architecture Highlights")
        lines.append("")
        lines.append("- **Hybrid Retrieval**: Dense (embedding) + Sparse (BM25) with RRF fusion")
        lines.append("- **Multi-source Confidence Fusion**: "
                      "Vector (40%) + Ontology (35%) + Entity Linking (25%)")
        lines.append("- **5-Layer Evaluation**: "
                      "Query understanding → Retrieval → KG → Ontology → Answer quality")
        lines.append("- **Knowledge Graph Integration**: "
                      "OWL ontology with rule-based reasoning engine")
        lines.append("")

        # Write file
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Portfolio report written to {out_path}")

    def _add_metric_row(
        self,
        lines: list[str],
        layer: str,
        metric: str,
        value: float,
        threshold: float | None,
        lower_is_better: bool = False,
    ) -> None:
        """메트릭 테이블 행 추가 (Pass/Fail 상태 포함)."""
        if threshold is not None:
            if lower_is_better:
                passed = value <= threshold
                threshold_str = f"<= {threshold:.2f}"
            else:
                passed = value >= threshold
                threshold_str = f">= {threshold:.2f}"
            status = "PASS" if passed else "FAIL"
        else:
            threshold_str = "-"
            status = "-"

        lines.append(f"| {layer} | {metric} | {value:.3f} | {threshold_str} | {status} |")

    # =========================================================================
    # JSON Output
    # =========================================================================

    def _save_metrics_json(self, metrics: dict[str, Any], path: Path) -> None:
        """메트릭 요약을 JSON으로 저장."""
        # Path 객체를 문자열로 변환
        serializable = _make_serializable(metrics)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Metrics summary written to {path}")


# =============================================================================
# Utility Functions
# =============================================================================


def _avg(values: list[float]) -> float:
    """안전한 평균 계산."""
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else 0.0


def _make_serializable(obj: Any) -> Any:
    """JSON 직렬화 가능하도록 변환."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def load_report_from_json(path: str | Path) -> EvalReport:
    """
    기존 report.json 파일에서 EvalReport 로드.

    Args:
        path: report.json 경로

    Returns:
        EvalReport 객체
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return EvalReport.model_validate(data)
