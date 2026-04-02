"""
Portfolio Chart Generator
=========================
Generates publication-quality evaluation charts for portfolio/resume use.

Charts:
1. L1-L5 Radar Chart (multi-layer performance)
2. Confusion Matrix Heatmap (retrieval TP/FP/FN/TN)
3. Precision-Recall-F1 Grouped Bar Chart
4. Score Distribution Histogram
5. Domain Performance Breakdown
6. Difficulty Comparison Bar Chart

Design System: AMOREPACIFIC (Pacific Blue #001C58, Amore Blue #1F5795)
"""

import matplotlib

matplotlib.use("Agg")

import logging
import math
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class PortfolioChartGenerator:
    """포트폴리오용 평가 차트 생성기 (AMOREPACIFIC 디자인 시스템)"""

    # AMOREPACIFIC 컬러 팔레트
    PACIFIC_BLUE = "#001C58"
    AMORE_BLUE = "#1F5795"
    GRAY = "#7D7D7D"
    WHITE = "#FFFFFF"
    LIGHT_BLUE = "#4A90D9"
    ACCENT_GREEN = "#43A047"
    ACCENT_RED = "#E53935"
    ACCENT_ORANGE = "#FF9800"

    # 차트 색상 세트
    LAYER_COLORS = ["#001C58", "#1F5795", "#4A90D9", "#7DBCEA", "#A8D5F2"]
    BAR_COLORS = ["#1F5795", "#43A047", "#FF9800"]  # Precision, Recall, F1

    def __init__(self, output_dir: str | Path, dpi: int = 200):
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self._setup_korean_font()
        self._setup_global_style()

    def _setup_korean_font(self):
        """한글 폰트 설정 (macOS, Linux/Docker 모두 지원)"""
        korean_fonts = [
            "Noto Sans CJK JP",
            "Noto Sans CJK KR",
            "Noto Sans CJK SC",
            "AppleGothic",
            "Malgun Gothic",
            "NanumGothic",
            "DejaVu Sans",
        ]

        font_found = False
        for font_name in korean_fonts:
            try:
                font_path = fm.findfont(
                    fm.FontProperties(family=font_name), fallback_to_default=False
                )
                if font_path and "DejaVuSans" not in font_path:
                    plt.rcParams["font.family"] = font_name
                    font_found = True
                    break
            except Exception:
                continue

        if not font_found:
            try:
                for font in fm.fontManager.ttflist:
                    if "Noto" in font.name and "CJK" in font.name:
                        plt.rcParams["font.family"] = font.name
                        font_found = True
                        break
            except Exception:
                pass

        if not font_found:
            logger.warning("No Korean font found. Charts may show broken characters.")

        plt.rcParams["axes.unicode_minus"] = False

    def _setup_global_style(self):
        """AMOREPACIFIC 디자인 시스템 기반 전역 스타일"""
        plt.rcParams["figure.facecolor"] = self.WHITE
        plt.rcParams["axes.facecolor"] = self.WHITE
        plt.rcParams["axes.edgecolor"] = "#E0E0E0"
        plt.rcParams["axes.linewidth"] = 0.8
        plt.rcParams["grid.color"] = "#E0E0E0"
        plt.rcParams["grid.linewidth"] = 0.5

    def generate_radar_chart(self, layer_scores: dict[str, float]) -> Path:
        """
        L1-L5 레이더 차트 생성.

        Args:
            layer_scores: {"L1 Query Understanding": 0.85, "L2 Retrieval": 0.78, ...}

        Returns:
            생성된 PNG 파일 경로
        """
        labels = list(layer_scores.keys())
        values = list(layer_scores.values())

        n = len(labels)
        angles = [i / n * 2 * math.pi for i in range(n)]
        values_closed = values + [values[0]]
        angles_closed = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

        # 배경 원 그리기
        for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
            circle_angles = np.linspace(0, 2 * math.pi, 100)
            ax.plot(circle_angles, [level] * 100, color="#E0E0E0", linewidth=0.5)

        # 데이터 영역
        ax.fill(angles_closed, values_closed, color=self.AMORE_BLUE, alpha=0.15)
        ax.plot(angles_closed, values_closed, color=self.PACIFIC_BLUE, linewidth=2.5)

        # 데이터 포인트
        ax.scatter(angles, values, color=self.PACIFIC_BLUE, s=80, zorder=5)
        for angle, value in zip(angles, values, strict=True):
            ax.annotate(
                f"{value:.2f}",
                xy=(angle, value),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=11,
                fontweight="bold",
                color=self.PACIFIC_BLUE,
            )

        # 축 설정
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=10, color=self.GRAY)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color=self.GRAY)

        ax.set_title(
            "Multi-Layer Evaluation (L1-L5)",
            fontsize=14,
            fontweight="bold",
            color=self.PACIFIC_BLUE,
            pad=20,
        )

        path = self.charts_dir / "radar_l1_l5.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=self.WHITE)
        plt.close(fig)
        return path

    def generate_confusion_matrix_heatmap(
        self, tp: int, fp: int, fn: int, tn: int
    ) -> Path:
        """
        검색 Confusion Matrix 히트맵 생성.

        Args:
            tp, fp, fn, tn: Confusion Matrix 값

        Returns:
            생성된 PNG 파일 경로
        """
        matrix = np.array([[tp, fp], [fn, tn]])
        labels_pred = ["Retrieved", "Not Retrieved"]
        labels_actual = ["Relevant", "Not Relevant"]

        fig, ax = plt.subplots(figsize=(7, 6))

        # 히트맵
        cmap = plt.cm.Blues
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0)

        # 셀 텍스트
        cell_labels = [["TP", "FP"], ["FN", "TN"]]
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                text_color = self.WHITE if val > matrix.max() * 0.5 else self.PACIFIC_BLUE
                ax.text(
                    j, i, f"{cell_labels[i][j]}\n{val}",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color=text_color,
                )

        # 축 설정
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels_pred, fontsize=11, color=self.GRAY)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels_actual, fontsize=11, color=self.GRAY)
        ax.set_xlabel("Predicted", fontsize=12, color=self.PACIFIC_BLUE, labelpad=10)
        ax.set_ylabel("Actual", fontsize=12, color=self.PACIFIC_BLUE, labelpad=10)

        ax.set_title(
            "Retrieval Confusion Matrix",
            fontsize=14, fontweight="bold",
            color=self.PACIFIC_BLUE, pad=15,
        )

        fig.colorbar(im, ax=ax, shrink=0.8, label="Count")

        path = self.charts_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=self.WHITE)
        plt.close(fig)
        return path

    def generate_precision_recall_bar(self, metrics: dict[str, dict[str, float]]) -> Path:
        """
        Precision/Recall/F1 그룹 바 차트.

        Args:
            metrics: {
                "L2 Retrieval": {"precision": 0.8, "recall": 0.75, "f1": 0.77},
                "L3 KG": {"precision": 0.9, "recall": 0.85, "f1": 0.87},
            }

        Returns:
            생성된 PNG 파일 경로
        """
        categories = list(metrics.keys())
        precision_vals = [m["precision"] for m in metrics.values()]
        recall_vals = [m["recall"] for m in metrics.values()]
        f1_vals = [m["f1"] for m in metrics.values()]

        x = np.arange(len(categories))
        width = 0.22

        fig, ax = plt.subplots(figsize=(10, 6))

        bars_p = ax.bar(x - width, precision_vals, width, label="Precision",
                        color=self.AMORE_BLUE, edgecolor="white", linewidth=0.5)
        bars_r = ax.bar(x, recall_vals, width, label="Recall",
                        color=self.ACCENT_GREEN, edgecolor="white", linewidth=0.5)
        bars_f = ax.bar(x + width, f1_vals, width, label="F1 Score",
                        color=self.ACCENT_ORANGE, edgecolor="white", linewidth=0.5)

        # 값 표시
        for bars in [bars_p, bars_r, bars_f]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f"{height:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=self.PACIFIC_BLUE,
                )

        ax.set_ylabel("Score", fontsize=12, color=self.PACIFIC_BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11, color=self.GRAY)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        ax.set_title(
            "Precision / Recall / F1 by Evaluation Layer",
            fontsize=14, fontweight="bold",
            color=self.PACIFIC_BLUE, pad=15,
        )

        path = self.charts_dir / "precision_recall_f1.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=self.WHITE)
        plt.close(fig)
        return path

    def generate_score_distribution(self, scores: list[float]) -> Path:
        """
        전체 점수 분포 히스토그램.

        Args:
            scores: 각 쿼리의 overall_score 리스트

        Returns:
            생성된 PNG 파일 경로
        """
        fig, ax = plt.subplots(figsize=(9, 6))

        if not scores:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
            path = self.charts_dir / "score_distribution.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=self.WHITE)
            plt.close(fig)
            return path

        scores_arr = np.array(scores)
        bins = np.arange(0, 1.05, 0.1)

        ax.hist(
            scores_arr, bins=bins,
            color=self.AMORE_BLUE, edgecolor=self.WHITE,
            alpha=0.85, linewidth=0.8,
        )

        # 평균/중앙값 라인
        mean_val = float(np.mean(scores_arr))
        median_val = float(np.median(scores_arr))
        ax.axvline(mean_val, color=self.ACCENT_RED, linestyle="--", linewidth=2,
                    label=f"Mean: {mean_val:.3f}")
        ax.axvline(median_val, color=self.ACCENT_GREEN, linestyle="-.", linewidth=2,
                    label=f"Median: {median_val:.3f}")

        ax.set_xlabel("Overall Score", fontsize=12, color=self.PACIFIC_BLUE)
        ax.set_ylabel("Count", fontsize=12, color=self.PACIFIC_BLUE)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        ax.set_title(
            "Score Distribution Across Queries",
            fontsize=14, fontweight="bold",
            color=self.PACIFIC_BLUE, pad=15,
        )

        path = self.charts_dir / "score_distribution.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=self.WHITE)
        plt.close(fig)
        return path

    def generate_domain_breakdown(self, by_domain: dict[str, dict[str, Any]]) -> Path:
        """
        도메인별 성능 수평 바 차트.

        Args:
            by_domain: {"market": {"count": 10, "pass_rate": 0.8, "avg_score": 0.75}, ...}

        Returns:
            생성된 PNG 파일 경로
        """
        domains = list(by_domain.keys())
        avg_scores = [by_domain[d].get("avg_score", 0) for d in domains]
        pass_rates = [by_domain[d].get("pass_rate", 0) for d in domains]
        counts = [int(by_domain[d].get("count", 0)) for d in domains]

        y = np.arange(len(domains))
        height = 0.35

        fig, ax = plt.subplots(figsize=(10, max(5, len(domains) * 1.2)))

        ax.barh(y - height / 2, avg_scores, height,
                label="Avg Score", color=self.AMORE_BLUE,
                edgecolor="white", linewidth=0.5)
        ax.barh(y + height / 2, pass_rates, height,
                label="Pass Rate", color=self.LIGHT_BLUE,
                edgecolor="white", linewidth=0.5)

        # 값 + 건수 표시
        for i, (score, rate, _count) in enumerate(
            zip(avg_scores, pass_rates, counts, strict=True)
        ):
            ax.text(score + 0.02, i - height / 2, f"{score:.2f}", va="center",
                    fontsize=9, fontweight="bold", color=self.PACIFIC_BLUE)
            ax.text(rate + 0.02, i + height / 2, f"{rate:.0%}", va="center",
                    fontsize=9, fontweight="bold", color=self.PACIFIC_BLUE)

        # 도메인 레이블에 건수 포함
        domain_labels = [f"{d} (n={c})" for d, c in zip(domains, counts, strict=True)]
        ax.set_yticks(y)
        ax.set_yticklabels(domain_labels, fontsize=11, color=self.GRAY)
        ax.set_xlim(0, 1.2)
        ax.set_xlabel("Score", fontsize=12, color=self.PACIFIC_BLUE)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(axis="x", alpha=0.3)

        ax.set_title(
            "Performance by Domain",
            fontsize=14, fontweight="bold",
            color=self.PACIFIC_BLUE, pad=15,
        )

        path = self.charts_dir / "domain_breakdown.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=self.WHITE)
        plt.close(fig)
        return path

    def generate_difficulty_comparison(
        self, by_difficulty: dict[str, dict[str, Any]]
    ) -> Path:
        """
        난이도별 Pass Rate 및 Avg Score 비교 바 차트.

        Args:
            by_difficulty: {"easy": {"count": 15, "pass_rate": 0.93, "avg_score": 0.88}, ...}

        Returns:
            생성된 PNG 파일 경로
        """
        order = ["easy", "medium", "hard"]
        difficulties = [d for d in order if d in by_difficulty]
        if not difficulties:
            difficulties = list(by_difficulty.keys())

        avg_scores = [by_difficulty[d].get("avg_score", 0) for d in difficulties]
        pass_rates = [by_difficulty[d].get("pass_rate", 0) for d in difficulties]
        counts = [int(by_difficulty[d].get("count", 0)) for d in difficulties]

        x = np.arange(len(difficulties))
        width = 0.3

        fig, ax = plt.subplots(figsize=(8, 6))

        colors_score = [self.ACCENT_GREEN, self.ACCENT_ORANGE, self.ACCENT_RED]

        bars1 = ax.bar(x - width / 2, avg_scores, width, label="Avg Score",
                        color=[colors_score[i] for i in range(len(difficulties))],
                        edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, pass_rates, width, label="Pass Rate",
                        color=[self.AMORE_BLUE] * len(difficulties),
                        edgecolor="white", linewidth=0.5, alpha=0.7)

        for bars in [bars1, bars2]:
            for bar in bars:
                height_val = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height_val + 0.02,
                    f"{height_val:.2f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=self.PACIFIC_BLUE,
                )

        diff_labels = [
            f"{d.capitalize()} (n={c})" for d, c in zip(difficulties, counts, strict=True)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(diff_labels, fontsize=11, color=self.GRAY)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score", fontsize=12, color=self.PACIFIC_BLUE)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        ax.set_title(
            "Performance by Difficulty Level",
            fontsize=14, fontweight="bold",
            color=self.PACIFIC_BLUE, pad=15,
        )

        path = self.charts_dir / "difficulty_comparison.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=self.WHITE)
        plt.close(fig)
        return path

    # =========================================================================
    # Ablation Study Comparison Charts
    # =========================================================================

    def generate_ablation_bar(self, mode_metrics: dict[str, dict[str, float]]) -> Path:
        """
        모드별 핵심 지표 비교 그룹 바 차트.

        Args:
            mode_metrics: {
                "RAG Only": {"recall": 0.62, "f1": 0.58, "answer_f1": 0.61, "ndcg": 0.65},
                "RAG + KG": {"recall": 0.78, ...},
                "Full Hybrid": {"recall": 0.85, ...},
            }
        """
        modes = list(mode_metrics.keys())
        metric_names = list(next(iter(mode_metrics.values())).keys())

        x = np.arange(len(metric_names))
        n_modes = len(modes)
        width = 0.7 / n_modes

        mode_colors = [self.GRAY, self.AMORE_BLUE, self.PACIFIC_BLUE]

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, mode in enumerate(modes):
            values = [mode_metrics[mode].get(m, 0) for m in metric_names]
            offset = (i - n_modes / 2 + 0.5) * width
            color = mode_colors[i] if i < len(mode_colors) else self.LAYER_COLORS[i]
            bars = ax.bar(
                x + offset, values, width,
                label=mode, color=color,
                edgecolor="white", linewidth=0.5,
            )
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=self.PACIFIC_BLUE,
                )

        ax.set_ylabel("Score", fontsize=12, color=self.PACIFIC_BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=10, color=self.GRAY)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        ax.set_title(
            "Ablation Study: Component Contribution",
            fontsize=14, fontweight="bold",
            color=self.PACIFIC_BLUE, pad=15,
        )

        path = self.charts_dir / "ablation_bar.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=self.WHITE)
        plt.close(fig)
        return path

    def generate_ablation_radar(
        self, mode_layer_scores: dict[str, dict[str, float]]
    ) -> Path:
        """
        모드별 L1-L5 레이더 차트 오버레이.

        Args:
            mode_layer_scores: {
                "RAG Only": {"L1": 0.7, "L2": 0.5, "L3": 0.3, "L4": 0.6, "L5": 0.5},
                "RAG + KG": {"L1": 0.75, ...},
                "Full Hybrid": {"L1": 0.8, ...},
            }
        """
        modes = list(mode_layer_scores.keys())
        labels = list(next(iter(mode_layer_scores.values())).keys())
        n = len(labels)

        angles = [i / n * 2 * math.pi for i in range(n)]
        angles_closed = angles + [angles[0]]

        mode_colors = [self.GRAY, self.AMORE_BLUE, self.PACIFIC_BLUE]
        mode_styles = ["--", "-.", "-"]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

        # 배경 원
        for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
            circle = np.linspace(0, 2 * math.pi, 100)
            ax.plot(circle, [level] * 100, color="#E0E0E0", linewidth=0.5)

        for i, mode in enumerate(modes):
            values = [mode_layer_scores[mode].get(lbl, 0) for lbl in labels]
            values_closed = values + [values[0]]
            color = mode_colors[i] if i < len(mode_colors) else self.LAYER_COLORS[i]
            style = mode_styles[i] if i < len(mode_styles) else "-"
            lw = 2.0 if i < len(modes) - 1 else 2.5

            ax.fill(angles_closed, values_closed, color=color, alpha=0.08)
            ax.plot(
                angles_closed, values_closed,
                color=color, linewidth=lw, linestyle=style, label=mode,
            )
            ax.scatter(angles, values, color=color, s=40, zorder=5)

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=10, color=self.GRAY)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color=self.GRAY)
        ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.3, 1.1))

        ax.set_title(
            "Ablation: Layer Performance Comparison",
            fontsize=14, fontweight="bold",
            color=self.PACIFIC_BLUE, pad=20,
        )

        path = self.charts_dir / "ablation_radar.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=self.WHITE)
        plt.close(fig)
        return path

    def generate_all(self, report_data: dict[str, Any]) -> dict[str, Path]:
        """
        모든 차트를 한번에 생성.

        Args:
            report_data: 포트폴리오 보고서 데이터 (PortfolioReportGenerator에서 전달)
                - layer_scores: dict[str, float]
                - confusion_matrix: {"tp": int, "fp": int, "fn": int, "tn": int}
                - pr_metrics: dict[str, dict[str, float]]
                - scores: list[float]
                - by_domain: dict
                - by_difficulty: dict

        Returns:
            {chart_name: file_path} 딕셔너리
        """
        charts = {}

        if "layer_scores" in report_data:
            charts["radar"] = self.generate_radar_chart(report_data["layer_scores"])

        if "confusion_matrix" in report_data:
            cm = report_data["confusion_matrix"]
            charts["confusion_matrix"] = self.generate_confusion_matrix_heatmap(
                tp=cm["tp"], fp=cm["fp"], fn=cm["fn"], tn=cm.get("tn", 0),
            )

        if "pr_metrics" in report_data:
            charts["precision_recall"] = self.generate_precision_recall_bar(
                report_data["pr_metrics"]
            )

        if "scores" in report_data:
            charts["score_distribution"] = self.generate_score_distribution(
                report_data["scores"]
            )

        if "by_domain" in report_data:
            charts["domain_breakdown"] = self.generate_domain_breakdown(
                report_data["by_domain"]
            )

        if "by_difficulty" in report_data:
            charts["difficulty_comparison"] = self.generate_difficulty_comparison(
                report_data["by_difficulty"]
            )

        logger.info(f"Generated {len(charts)} portfolio charts in {self.charts_dir}")
        return charts
