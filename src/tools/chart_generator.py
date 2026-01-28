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

# Headless 서버 환경을 위해 Agg 백엔드 설정 (pyplot import 전에 필수)
import matplotlib

matplotlib.use("Agg")

import tempfile
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


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
        "#AEC6CF",
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
        """한글 폰트 설정 (macOS, Linux/Docker 환경 모두 지원)"""
        import logging

        logger = logging.getLogger(__name__)

        # 폰트 우선순위: Docker (Noto Sans CJK) > macOS > Windows > fallback
        korean_fonts = [
            "Noto Sans CJK JP",  # Docker/Linux (fonts-noto-cjk)
            "Noto Sans CJK KR",  # Docker/Linux 대체
            "Noto Sans CJK SC",  # Docker/Linux 대체
            "AppleGothic",  # macOS
            "Malgun Gothic",  # Windows
            "NanumGothic",  # 설치된 경우
            "DejaVu Sans",  # fallback (한글 제한)
        ]

        font_found = False
        for font_name in korean_fonts:
            try:
                # 폰트 존재 여부 확인
                font_path = fm.findfont(
                    fm.FontProperties(family=font_name), fallback_to_default=False
                )
                if font_path and "DejaVuSans" not in font_path:  # fallback 아닌 경우만
                    plt.rcParams["font.family"] = font_name
                    logger.info(f"Korean font set: {font_name}")
                    font_found = True
                    break
            except Exception:
                continue

        if not font_found:
            # 시스템 폰트 캐시에서 한글 지원 폰트 직접 탐색
            try:
                for font in fm.fontManager.ttflist:
                    if "Noto" in font.name and "CJK" in font.name:
                        plt.rcParams["font.family"] = font.name
                        logger.info(f"Korean font found via fontManager: {font.name}")
                        font_found = True
                        break
            except Exception as e:
                logger.warning(f"Font search failed: {e}")

        if not font_found:
            logger.warning("No Korean font found. Charts may show broken characters.")

        plt.rcParams["axes.unicode_minus"] = False

    def generate_sos_trend_chart(
        self,
        daily_trends: list[dict[str, Any]],
        title: str = "LANEIGE SoS 추이",
        filename: str = "sos_trend.png",
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

        # 날짜 정렬 (혹시 정렬되지 않은 경우 대비)
        sorted_trends = sorted(daily_trends, key=lambda d: d["date"])

        # 날짜 포맷팅: MM-DD 형식 (최소 7일 이상 표시 보장)
        dates = [d["date"][-5:] for d in sorted_trends]  # MM-DD 형식
        full_dates = [d["date"] for d in sorted_trends]  # YYYY-MM-DD 형식 (기간 계산용)
        sos_values = [d.get("laneige_sos", 0) for d in sorted_trends]

        # 제목에서 기간 추출하여 기대 일수 계산
        expected_days = None
        data_warning = None
        import re
        from datetime import datetime

        date_match = re.search(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})", title)
        if date_match and full_dates:
            try:
                start = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                end = datetime.strptime(date_match.group(2), "%Y-%m-%d")
                expected_days = (end - start).days + 1
                actual_days = len(dates)

                if actual_days < expected_days:
                    missing = expected_days - actual_days
                    data_warning = f"※ 데이터 {actual_days}일분만 존재 (기간 {expected_days}일 중 {missing}일 누락)"
            except ValueError:
                pass

        # 데이터가 없거나 적을 때 처리
        if not dates or not sos_values:
            # 빈 차트 생성
            ax.text(
                0.5,
                0.5,
                "데이터 없음",
                ha="center",
                va="center",
                fontsize=14,
                color=self.GRAY,
                transform=ax.transAxes,
            )
            ax.set_title(title, fontsize=14, fontweight="bold", color=self.PACIFIC_BLUE)
            output_path = self.output_dir / filename
            fig.savefig(
                output_path, dpi=150, bbox_inches="tight", facecolor=self.WHITE, edgecolor="none"
            )
            plt.close(fig)
            return output_path

        # 라인 차트
        ax.plot(
            dates,
            sos_values,
            marker="o",
            color=self.PACIFIC_BLUE,
            linewidth=2,
            markersize=6,
            label="LANEIGE SoS",
        )

        # 평균선
        valid_values = [v for v in sos_values if v is not None and v > 0]
        avg_sos = sum(valid_values) / len(valid_values) if valid_values else 0
        ax.axhline(
            y=avg_sos,
            color=self.AMORE_BLUE,
            linestyle="--",
            linewidth=1,
            label=f"평균: {avg_sos:.2f}%",
        )

        # 스타일링
        ax.set_title(title, fontsize=14, fontweight="bold", color=self.PACIFIC_BLUE)
        ax.set_xlabel("날짜", fontsize=10, color=self.GRAY)
        ax.set_ylabel("SoS (%)", fontsize=10, color=self.GRAY)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Y축 범위 설정 (적절한 스케일)
        max_sos = max(valid_values) if valid_values else 1
        min_sos = min(valid_values) if valid_values else 0

        # 여유 공간 계산 (데이터 범위의 20%)
        margin = max(0.2, (max_sos - min_sos) * 0.2) if max_sos > min_sos else 0.5
        ax.set_ylim(bottom=max(0, min_sos - margin), top=max_sos + margin)

        # X축 틱 설정 (날짜가 많을 경우 간격 조정)
        num_dates = len(dates)
        if num_dates > 14:
            # 14일 이상: 3일 간격
            tick_step = 3
        elif num_dates > 7:
            # 7-14일: 2일 간격
            tick_step = 2
        else:
            # 7일 이하: 모든 날짜 표시
            tick_step = 1

        tick_indices = list(range(0, num_dates, tick_step))
        if (num_dates - 1) not in tick_indices:
            tick_indices.append(num_dates - 1)  # 마지막 날짜 포함

        ax.set_xticks([dates[i] for i in tick_indices])
        ax.set_xticklabels([dates[i] for i in tick_indices])

        # 데이터 누락 경고 표시
        if data_warning:
            fig.text(
                0.5,
                0.02,
                data_warning,
                ha="center",
                fontsize=9,
                color="#E74C3C",
                style="italic",
                weight="bold",
            )
            plt.tight_layout(rect=[0, 0.05, 1, 1])
        else:
            plt.tight_layout()

        # 저장
        output_path = self.output_dir / filename
        fig.savefig(
            output_path, dpi=150, bbox_inches="tight", facecolor=self.WHITE, edgecolor="none"
        )
        plt.close(fig)

        return output_path

    def generate_category_sos_chart(
        self,
        category_analysis: dict[str, dict[str, Any]],
        title: str = "카테고리별 LANEIGE SoS",
        filename: str = "category_sos.png",
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
        for bar, val in zip(bars, sos_values, strict=False):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=9,
                color=self.GRAY,
            )

        ax.set_title(title, fontsize=14, fontweight="bold", color=self.PACIFIC_BLUE)
        ax.set_xlabel("SoS (%)", fontsize=10, color=self.GRAY)
        ax.set_xlim(right=max(sos_values) * 1.2 if sos_values else 10)
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(
            output_path, dpi=150, bbox_inches="tight", facecolor=self.WHITE, edgecolor="none"
        )
        plt.close(fig)

        return output_path

    def generate_brand_comparison_chart(
        self,
        brand_performance: list[dict[str, Any]],
        top_n: int = 10,
        title: str = "브랜드별 점유율 (Top 10)",
        filename: str = "brand_comparison.png",
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
        for _i, (bar, val, change) in enumerate(
            zip(bars, sos_values[::-1], sos_changes[::-1], strict=False)
        ):
            change_str = f"+{change:.1f}" if change > 0 else f"{change:.1f}"
            change_color = "#27AE60" if change > 0 else "#E74C3C" if change < 0 else self.GRAY

            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}% ({change_str})",
                va="center",
                fontsize=9,
                color=change_color,
            )

        ax.set_title(title, fontsize=14, fontweight="bold", color=self.PACIFIC_BLUE)
        ax.set_xlabel("SoS (%)", fontsize=10, color=self.GRAY)
        ax.set_xlim(right=max(sos_values) * 1.3 if sos_values else 10)
        ax.grid(True, axis="x", alpha=0.3)

        # "Others" 또는 "소규모 브랜드" 범주가 있으면 각주 추가
        has_others = any("others" in b.lower() or "소규모" in b or "기타" in b for b in brands)
        if has_others:
            footnote = "※ Others/소규모 브랜드: 개별 점유율 1% 미만 브랜드 합산"
            fig.text(0.5, 0.01, footnote, ha="center", fontsize=8, color=self.GRAY, style="italic")

        plt.tight_layout(rect=[0, 0.03, 1, 1] if has_others else [0, 0, 1, 1])

        output_path = self.output_dir / filename
        fig.savefig(
            output_path, dpi=150, bbox_inches="tight", facecolor=self.WHITE, edgecolor="none"
        )
        plt.close(fig)

        return output_path

    def generate_hhi_trend_chart(
        self,
        daily_trends: list[dict[str, Any]],
        title: str = "시장 집중도(HHI) 추이",
        filename: str = "hhi_trend.png",
    ) -> Path:
        """HHI 추이 차트"""
        import re
        from datetime import datetime

        fig, ax = plt.subplots(figsize=(10, 4))

        dates = [d["date"][-5:] for d in daily_trends]
        full_dates = [d["date"] for d in daily_trends]
        hhi_values = [d.get("hhi", 0) for d in daily_trends]

        # 데이터 누락 경고 계산
        data_warning = None
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})", title)
        if date_match and full_dates:
            try:
                start = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                end = datetime.strptime(date_match.group(2), "%Y-%m-%d")
                expected_days = (end - start).days + 1
                actual_days = len(dates)
                if actual_days < expected_days:
                    missing = expected_days - actual_days
                    data_warning = f"※ 데이터 {actual_days}일분만 존재 ({missing}일 누락)"
            except ValueError:
                pass

        ax.plot(dates, hhi_values, marker="s", color=self.AMORE_BLUE, linewidth=2, markersize=5)

        # 집중도 기준선
        ax.axhline(
            y=1500, color="#27AE60", linestyle="--", linewidth=1, label="경쟁적 시장 (<1500)"
        )
        ax.axhline(
            y=2500, color="#E74C3C", linestyle="--", linewidth=1, label="고집중 시장 (>2500)"
        )

        ax.set_title(title, fontsize=14, fontweight="bold", color=self.PACIFIC_BLUE)
        ax.set_xlabel("날짜", fontsize=10, color=self.GRAY)
        ax.set_ylabel("HHI", fontsize=10, color=self.GRAY)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 데이터 누락 경고 표시
        if data_warning:
            fig.text(
                0.5,
                0.02,
                data_warning,
                ha="center",
                fontsize=9,
                color="#E74C3C",
                style="italic",
                weight="bold",
            )
            plt.tight_layout(rect=[0, 0.05, 1, 1])
        else:
            plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(
            output_path, dpi=150, bbox_inches="tight", facecolor=self.WHITE, edgecolor="none"
        )
        plt.close(fig)

        return output_path

    def generate_product_rank_table(
        self,
        top_products: list[dict[str, Any]],
        title: str = "LANEIGE 제품별 순위 변동",
        filename: str = "product_ranks.png",
    ) -> Path:
        """
        제품별 순위 변동 테이블 (이미지)

        AMOREPACIFIC 디자인 가이드 적용:
        - Pacific Blue 헤더
        - 큰 폰트 (12pt)
        - 충분한 행 높이
        - 제품명 가독성 확보
        """
        # 테이블 행 수에 따라 높이 조정 (최소 높이 확보)
        num_rows = min(len(top_products), 10)
        row_height = 1.0  # 각 행의 높이 (기존 0.8 → 1.0으로 더 증가)
        fig_height = max(5, num_rows * row_height + 3)  # 최소 높이 5

        fig, ax = plt.subplots(figsize=(16, fig_height))  # 너비 14 → 16으로 더 증가
        ax.axis("off")

        # 테이블 데이터
        headers = ["제품명", "시작 순위", "종료 순위", "변동"]
        rows = []
        cell_colors = []

        for p in top_products[:10]:
            # 제품명 길이 제한 (55자로 증가하여 가독성 개선)
            raw_title = p.get("title", "")
            title_short = raw_title[:55] + "..." if len(raw_title) > 55 else raw_title
            change = p.get("change", 0)
            change_str = f"+{change}" if change > 0 else str(change)

            rows.append(
                [
                    title_short,
                    str(p.get("start_rank", "-")),
                    str(p.get("end_rank", "-")),
                    change_str,
                ]
            )

            # 변동에 따른 색상 (AMOREPACIFIC 색상 적용)
            # 상승: 연한 파랑 (#E8F4FD), 하락: 연한 분홍 (#FCE8EC)
            if change > 5:
                cell_colors.append([self.WHITE, self.WHITE, self.WHITE, "#E8F4FD"])
            elif change < -5:
                cell_colors.append([self.WHITE, self.WHITE, self.WHITE, "#FCE8EC"])
            else:
                cell_colors.append([self.WHITE] * 4)

        if rows:
            table = ax.table(
                cellText=rows,
                colLabels=headers,
                cellLoc="center",
                loc="center",
                colColours=[self.PACIFIC_BLUE] * 4,
                cellColours=cell_colors,
                colWidths=[0.52, 0.16, 0.16, 0.16],  # 컬럼 너비 비율 조정
            )

            table.auto_set_font_size(False)
            table.set_fontsize(12)  # 폰트 크기 10 → 12로 더 증가 (가독성 대폭 개선)
            table.scale(1.1, 2.2)  # 가로 1.1, 세로 2.2로 스케일 확대

            # 셀별 스타일 조정 (Arita 디자인 철학 반영)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    # 헤더 스타일 (Pacific Blue 배경, 흰색 텍스트)
                    cell.set_text_props(color=self.WHITE, fontweight="bold", fontsize=13)
                    cell.set_height(0.14)  # 헤더 높이 증가
                else:
                    # 데이터 셀 (Pacific Blue 텍스트)
                    cell.set_text_props(color=self.PACIFIC_BLUE, fontsize=12)
                    cell.set_height(0.12)  # 데이터 행 높이 증가

                    # 변동 열은 변화에 따라 색상 적용
                    if col == 3:
                        change_val = rows[row - 1][3]
                        if change_val.startswith("+"):
                            cell.set_text_props(
                                color=self.AMORE_BLUE, fontweight="bold", fontsize=12
                            )
                        elif change_val.startswith("-"):
                            cell.set_text_props(
                                color="#8B2635", fontweight="bold", fontsize=12
                            )  # 위험 색상

                # 첫 번째 컬럼(제품명)은 왼쪽 정렬
                if col == 0:
                    cell._loc = "left"
                    cell.PAD = 0.03  # 패딩

                # 테두리 스타일 (연한 회색)
                cell.set_edgecolor("#E5E7EB")
                cell.set_linewidth(0.5)

        ax.set_title(title, fontsize=16, fontweight="bold", color=self.PACIFIC_BLUE, pad=35)

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",  # dpi 150 → 180으로 증가
            facecolor=self.WHITE,
            edgecolor="none",
        )
        plt.close(fig)

        return output_path

    def generate_all_charts(self, analysis) -> dict[str, Path]:
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
                title=f"LANEIGE SoS 추이 ({analysis.start_date} ~ {analysis.end_date})",
            )

        # 2. 카테고리별 SoS
        if analysis.category_analysis:
            charts["category_sos"] = self.generate_category_sos_chart(analysis.category_analysis)

        # 3. 브랜드 비교
        if analysis.brand_performance:
            charts["brand_comparison"] = self.generate_brand_comparison_chart(
                analysis.brand_performance
            )

        # 4. HHI 추이
        if analysis.daily_trends:
            charts["hhi_trend"] = self.generate_hhi_trend_chart(analysis.daily_trends)

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
            "face_powder": "Face Powder",
        }
        return name_map.get(category, category.replace("_", " ").title())

    def to_bytes(self, chart_path: Path) -> bytes:
        """차트 파일을 바이트로 읽기 (DOCX 삽입용)"""
        with open(chart_path, "rb") as f:
            return f.read()
