"""
차트 생성기
==========
기간별 분석 데이터를 시각화하여 PNG 이미지로 생성

## 필요 차트
1. SoS 일별 추이 (Line Chart)
2. 카테고리별 SoS 비교 (Bar Chart)
3. 브랜드별 점유율 변화 (Horizontal Bar)
4. 제품별 순위 변동 (Table or Bump Chart)

## 디자인 시스템 (AMOREPACIFIC)
- Pacific Blue: #001C58 (메인)
- Amore Blue: #1F5795 (보조)
- Gray: #7D7D7D (텍스트)
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tempfile
import io


class ChartGenerator:
    """차트 생성기"""

    # AMOREPACIFIC 컬러 팔레트
    PACIFIC_BLUE = "#001C58"
    AMORE_BLUE = "#1F5795"
    GRAY = "#7D7D7D"
    WHITE = "#FFFFFF"

    # 차트 색상 팔레트 (브랜드별)
    BRAND_COLORS = [
        "#001C58",  # LANEIGE (Pacific Blue)
        "#1F5795",  # Amore Blue
        "#4A90D9",
        "#7DBCEA",
        "#A8D5F2",
        "#D4E8F7",
        "#FF6B6B",  # 경쟁사 (빨강 계열)
        "#FFB347",
        "#77DD77",
        "#AEC6CF"
    ]

    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: 차트 이미지 저장 디렉토리 (None이면 임시 폴더)
        """
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 한글 폰트 설정 (macOS)
        self._setup_korean_font()

    def _setup_korean_font(self):
        """한글 폰트 설정"""
        # macOS 기본 한글 폰트
        korean_fonts = [
            "AppleGothic",
            "Malgun Gothic",
            "NanumGothic",
            "Arial Unicode MS"
        ]

        for font_name in korean_fonts:
            try:
                fm.findfont(fm.FontProperties(family=font_name))
                plt.rcParams["font.family"] = font_name
                break
            except:
                continue

        plt.rcParams["axes.unicode_minus"] = False

    def generate_sos_trend_chart(
        self,
        daily_trends: List[Dict[str, Any]],
        title: str = "LANEIGE SoS 추이",
        filename: str = "sos_trend.png"
    ) -> Path:
        """
        SoS 일별 추이 차트 생성

        Args:
            daily_trends: [{"date": "2026-01-14", "laneige_sos": 2.5, ...}, ...]
            title: 차트 제목
            filename: 저장 파일명

        Returns:
            생성된 이미지 파일 경로
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        dates = [d["date"][-5:] for d in daily_trends]  # MM-DD 형식
        sos_values = [d["laneige_sos"] for d in daily_trends]

        # 라인 차트
        ax.plot(dates, sos_values, marker="o", color=self.PACIFIC_BLUE,
                linewidth=2, markersize=6, label="LANEIGE SoS")

        # 평균선
        avg_sos = sum(sos_values) / len(sos_values) if sos_values else 0
        ax.axhline(y=avg_sos, color=self.AMORE_BLUE, linestyle="--",
                   linewidth=1, label=f"평균: {avg_sos:.2f}%")

        # 스타일링
        ax.set_title(title, fontsize=14, fontweight="bold", color=self.PACIFIC_BLUE)
        ax.set_xlabel("날짜", fontsize=10, color=self.GRAY)
        ax.set_ylabel("SoS (%)", fontsize=10, color=self.GRAY)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Y축 범위 설정 (최소 0)
        ax.set_ylim(bottom=0)

        plt.tight_layout()

        # 저장
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=self.WHITE, edgecolor="none")
        plt.close(fig)

        return output_path

    def generate_category_sos_chart(
        self,
        category_analysis: Dict[str, Dict[str, Any]],
        title: str = "카테고리별 LANEIGE SoS",
        filename: str = "category_sos.png"
    ) -> Path:
        """
        카테고리별 SoS 비교 차트

        Args:
            category_analysis: {"lip_care": {"avg_sos": 5.2, ...}, ...}
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        categories = list(category_analysis.keys())
        sos_values = [v.get("avg_sos", 0) for v in category_analysis.values()]

        # 카테고리명 정리
        display_names = [self._format_category_name(c) for c in categories]

        # 수평 바 차트
        bars = ax.barh(display_names, sos_values, color=self.AMORE_BLUE, height=0.6)

        # 값 표시
        for bar, val in zip(bars, sos_values):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=9, color=self.GRAY)

        ax.set_title(title, fontsize=14, fontweight="bold", color=self.PACIFIC_BLUE)
        ax.set_xlabel("SoS (%)", fontsize=10, color=self.GRAY)
        ax.set_xlim(right=max(sos_values) * 1.2 if sos_values else 10)
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=self.WHITE, edgecolor="none")
        plt.close(fig)

        return output_path

    def generate_brand_comparison_chart(
        self,
        brand_performance: List[Dict[str, Any]],
        top_n: int = 10,
        title: str = "브랜드별 점유율 (Top 10)",
        filename: str = "brand_comparison.png"
    ) -> Path:
        """
        브랜드별 점유율 비교 차트

        Args:
            brand_performance: [{"brand": "LANEIGE", "avg_sos": 2.5, "sos_change": +0.3}, ...]
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Top N 브랜드만
        top_brands = brand_performance[:top_n]

        brands = [b["brand"] for b in top_brands]
        sos_values = [b["avg_sos"] for b in top_brands]
        sos_changes = [b.get("sos_change", 0) for b in top_brands]

        # 색상 (LANEIGE는 Pacific Blue, 나머지는 그라데이션)
        colors = []
        for brand in brands:
            if "LANEIGE" in brand.upper():
                colors.append(self.PACIFIC_BLUE)
            else:
                colors.append(self.AMORE_BLUE)

        # 바 차트
        bars = ax.barh(brands[::-1], sos_values[::-1], color=colors[::-1], height=0.6)

        # 변화율 표시
        for i, (bar, val, change) in enumerate(zip(bars, sos_values[::-1], sos_changes[::-1])):
            change_str = f"+{change:.1f}" if change > 0 else f"{change:.1f}"
            change_color = "#27AE60" if change > 0 else "#E74C3C" if change < 0 else self.GRAY

            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}% ({change_str})", va="center", fontsize=9, color=change_color)

        ax.set_title(title, fontsize=14, fontweight="bold", color=self.PACIFIC_BLUE)
        ax.set_xlabel("SoS (%)", fontsize=10, color=self.GRAY)
        ax.set_xlim(right=max(sos_values) * 1.3 if sos_values else 10)
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=self.WHITE, edgecolor="none")
        plt.close(fig)

        return output_path

    def generate_hhi_trend_chart(
        self,
        daily_trends: List[Dict[str, Any]],
        title: str = "시장 집중도(HHI) 추이",
        filename: str = "hhi_trend.png"
    ) -> Path:
        """HHI 추이 차트"""
        fig, ax = plt.subplots(figsize=(10, 4))

        dates = [d["date"][-5:] for d in daily_trends]
        hhi_values = [d.get("hhi", 0) for d in daily_trends]

        ax.plot(dates, hhi_values, marker="s", color=self.AMORE_BLUE,
                linewidth=2, markersize=5)

        # 집중도 기준선
        ax.axhline(y=1500, color="#27AE60", linestyle="--", linewidth=1,
                   label="경쟁적 시장 (<1500)")
        ax.axhline(y=2500, color="#E74C3C", linestyle="--", linewidth=1,
                   label="고집중 시장 (>2500)")

        ax.set_title(title, fontsize=14, fontweight="bold", color=self.PACIFIC_BLUE)
        ax.set_xlabel("날짜", fontsize=10, color=self.GRAY)
        ax.set_ylabel("HHI", fontsize=10, color=self.GRAY)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=self.WHITE, edgecolor="none")
        plt.close(fig)

        return output_path

    def generate_product_rank_table(
        self,
        top_products: List[Dict[str, Any]],
        title: str = "LANEIGE 제품별 순위 변동",
        filename: str = "product_ranks.png"
    ) -> Path:
        """제품별 순위 변동 테이블 (이미지)"""
        fig, ax = plt.subplots(figsize=(12, len(top_products) * 0.5 + 1))
        ax.axis("off")

        # 테이블 데이터
        headers = ["제품명", "시작 순위", "종료 순위", "변동"]
        rows = []
        cell_colors = []

        for p in top_products[:10]:
            title_short = p.get("title", "")[:40] + "..." if len(p.get("title", "")) > 40 else p.get("title", "")
            change = p.get("change", 0)
            change_str = f"+{change}" if change > 0 else str(change)

            rows.append([
                title_short,
                str(p.get("start_rank", "-")),
                str(p.get("end_rank", "-")),
                change_str
            ])

            # 변동에 따른 색상
            if change > 5:
                cell_colors.append([self.WHITE, self.WHITE, self.WHITE, "#D4EDDA"])
            elif change < -5:
                cell_colors.append([self.WHITE, self.WHITE, self.WHITE, "#F8D7DA"])
            else:
                cell_colors.append([self.WHITE] * 4)

        if rows:
            table = ax.table(
                cellText=rows,
                colLabels=headers,
                cellLoc="center",
                loc="center",
                colColours=[self.PACIFIC_BLUE] * 4,
                cellColours=cell_colors
            )

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # 헤더 텍스트 색상
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(color=self.WHITE, fontweight="bold")

        ax.set_title(title, fontsize=14, fontweight="bold", color=self.PACIFIC_BLUE, pad=20)

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=self.WHITE, edgecolor="none")
        plt.close(fig)

        return output_path

    def generate_all_charts(self, analysis) -> Dict[str, Path]:
        """
        모든 차트 일괄 생성

        Args:
            analysis: PeriodAnalysis 객체

        Returns:
            {"sos_trend": Path, "category_sos": Path, ...}
        """
        charts = {}

        # 1. SoS 추이
        if analysis.daily_trends:
            charts["sos_trend"] = self.generate_sos_trend_chart(
                analysis.daily_trends,
                title=f"LANEIGE SoS 추이 ({analysis.start_date} ~ {analysis.end_date})"
            )

        # 2. 카테고리별 SoS
        if analysis.category_analysis:
            charts["category_sos"] = self.generate_category_sos_chart(
                analysis.category_analysis
            )

        # 3. 브랜드 비교
        if analysis.brand_performance:
            charts["brand_comparison"] = self.generate_brand_comparison_chart(
                analysis.brand_performance
            )

        # 4. HHI 추이
        if analysis.daily_trends:
            charts["hhi_trend"] = self.generate_hhi_trend_chart(
                analysis.daily_trends
            )

        # 5. 제품 순위 테이블
        if analysis.laneige_metrics.get("top_products"):
            charts["product_ranks"] = self.generate_product_rank_table(
                analysis.laneige_metrics["top_products"]
            )

        return charts

    def _format_category_name(self, category: str) -> str:
        """카테고리명 포맷팅"""
        name_map = {
            "beauty_personal_care": "Beauty & Personal Care",
            "skin_care": "Skin Care",
            "lip_care": "Lip Care",
            "lip_makeup": "Lip Makeup",
            "face_powder": "Face Powder"
        }
        return name_map.get(category, category.replace("_", " ").title())

    def to_bytes(self, chart_path: Path) -> bytes:
        """차트 파일을 바이트로 읽기 (DOCX 삽입용)"""
        with open(chart_path, "rb") as f:
            return f.read()
