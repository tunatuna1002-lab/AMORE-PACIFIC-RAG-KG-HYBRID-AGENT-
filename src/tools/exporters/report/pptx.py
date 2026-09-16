"""
PPTX report generator
=====================
발표용 슬라이드 생성. ``python-pptx`` 가 없으면 생성 시점에 명시적으로 실패한다.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .design import DesignSystem

logger = logging.getLogger(__name__)


# =============================================================================
# PPTX Report Generator
# =============================================================================


class PptxReportGenerator:
    """
    PPTX 리포트 생성기

    AMOREPACIFIC IR 스타일의 프레젠테이션 생성
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # python-pptx 가용성 확인
        try:
            import importlib.util

            self._pptx_available = importlib.util.find_spec("pptx") is not None
        except Exception:
            self._pptx_available = False

        if not self._pptx_available:
            logger.warning("python-pptx not installed. PPTX generation disabled.")

        # Project root 찾기
        self.project_root = self._find_project_root()

    def _find_project_root(self) -> Path:
        """프로젝트 루트 디렉토리 찾기"""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "CLAUDE.md").exists() or (
                parent / "src" / "api" / "dashboard_api.py"
            ).exists():
                return parent
        return Path.cwd()

    def _get_logo_path(self, logo_type: str = "reverse") -> Path | None:
        """로고 파일 경로 반환 (PPTX는 어두운 배경이므로 reverse 권장)"""
        logo_map = {
            "color": DesignSystem.LOGO_COLOR_EN,
            "basic": DesignSystem.LOGO_BASIC_EN,
            "reverse": DesignSystem.LOGO_REVERSE_EN,
        }
        logo_rel = logo_map.get(logo_type, DesignSystem.LOGO_REVERSE_EN)
        logo_path = self.project_root / logo_rel

        if logo_path.exists():
            return logo_path
        return None

    def generate_presentation(
        self,
        report_data: dict[str, Any],
        chart_paths: dict[str, Path] = None,
        output_filename: str = None,
    ) -> Path | None:
        """
        프레젠테이션 생성

        Args:
            report_data: 리포트 데이터 (DocxReportGenerator와 동일 형식)
            chart_paths: 차트 이미지 경로
            output_filename: 출력 파일명

        Returns:
            생성된 파일 경로 (python-pptx 미설치 시 None)
        """
        if not self._pptx_available:
            logger.error("python-pptx not installed")
            return None

        from pptx import Presentation
        from pptx.dml.color import RGBColor as PptxRGBColor
        from pptx.util import Inches as PptxInches
        from pptx.util import Pt as PptxPt

        prs = Presentation()
        prs.slide_width = PptxInches(13.333)  # 16:9 비율
        prs.slide_height = PptxInches(7.5)

        # 색상 상수
        PACIFIC_BLUE = PptxRGBColor(0, 28, 88)
        WHITE = PptxRGBColor(255, 255, 255)
        GRAY = PptxRGBColor(125, 125, 125)
        _ = PACIFIC_BLUE, WHITE, GRAY  # 사용 표시

        # 메타 정보
        title = report_data.get("title", "AMORE Insight Report")
        start_date = report_data.get("start_date", "")
        end_date = report_data.get("end_date", "")

        # ========== 슬라이드 1: 표지 ==========
        slide_layout = prs.slide_layouts[6]  # Blank layout
        slide = prs.slides.add_slide(slide_layout)

        # 배경 (Pacific Blue)
        background = slide.shapes.add_shape(
            1,  # Rectangle
            PptxInches(0),
            PptxInches(0),
            prs.slide_width,
            prs.slide_height,
        )
        background.fill.solid()
        background.fill.fore_color.rgb = PACIFIC_BLUE
        background.line.fill.background()

        # 로고
        logo_path = self._get_logo_path("reverse")
        if logo_path:
            slide.shapes.add_picture(
                str(logo_path), PptxInches(0.5), PptxInches(0.5), width=PptxInches(2.5)
            )

        # 제목
        title_box = slide.shapes.add_textbox(
            PptxInches(0.5), PptxInches(3), PptxInches(12), PptxInches(1.5)
        )
        tf = title_box.text_frame
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = PptxPt(44)
        p.font.bold = True
        p.font.color.rgb = WHITE

        # 날짜
        date_box = slide.shapes.add_textbox(
            PptxInches(0.5), PptxInches(5), PptxInches(6), PptxInches(0.5)
        )
        tf = date_box.text_frame
        p = tf.paragraphs[0]
        p.text = f"{start_date} ~ {end_date}" if start_date else datetime.now().strftime("%Y-%m-%d")
        p.font.size = PptxPt(18)
        p.font.color.rgb = WHITE

        # ========== 슬라이드 2: 목차 ==========
        slide = prs.slides.add_slide(slide_layout)

        # AMORE PACIFIC 로고 이미지
        logo_path = self._get_logo_path("color")
        if logo_path:
            slide.shapes.add_picture(
                str(logo_path), PptxInches(0.5), PptxInches(0.3), width=PptxInches(3.5)
            )

        # 목차 항목 (IR 스타일 - 왼쪽 파란 바)
        sections = report_data.get("sections", [])
        y_pos = 1.5  # 시작 Y 위치 (로고 아래)

        for _i, section in enumerate(sections):
            # 왼쪽 파란 바 (Rectangle)
            bar = slide.shapes.add_shape(
                1,  # Rectangle
                PptxInches(0.8),
                PptxInches(y_pos),
                PptxInches(0.08),
                PptxInches(0.4),
            )
            bar.fill.solid()
            bar.fill.fore_color.rgb = PACIFIC_BLUE
            bar.line.fill.background()

            # 섹션명 텍스트
            section_box = slide.shapes.add_textbox(
                PptxInches(1.0), PptxInches(y_pos), PptxInches(10), PptxInches(0.5)
            )
            tf = section_box.text_frame
            p = tf.paragraphs[0]
            p.text = section.get("title", "")
            p.font.size = PptxPt(18)
            p.font.bold = True
            p.font.color.rgb = PACIFIC_BLUE

            y_pos += 0.6  # 다음 항목 위치

        # DISCLAIMER (하단)
        disclaimer_box = slide.shapes.add_textbox(
            PptxInches(0.5), PptxInches(6.5), PptxInches(12), PptxInches(0.8)
        )
        tf = disclaimer_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = "DISCLAIMER"
        p.font.size = PptxPt(10)
        p.font.bold = True
        p.font.color.rgb = GRAY

        p2 = tf.add_paragraph()
        p2.text = (
            "본 자료는 Amazon.com 웹사이트에서 공개적으로 수집된 베스트셀러 순위 데이터를 기반으로 작성되었습니다. "
            "모든 분석 인사이트는 AI(인공지능)에 의해 자동 생성된 것으로, 실제 시장 상황과 다를 수 있습니다. "
            "본 자료의 정보를 활용한 의사결정에 대해 당사는 어떠한 책임도 지지 않습니다."
        )
        p2.font.size = PptxPt(8)
        p2.font.color.rgb = GRAY

        # ========== 본문 슬라이드들 ==========
        for section in sections:
            slide = prs.slides.add_slide(slide_layout)

            # 섹션 제목
            title_box = slide.shapes.add_textbox(
                PptxInches(0.5), PptxInches(0.3), PptxInches(12), PptxInches(0.8)
            )
            tf = title_box.text_frame
            p = tf.paragraphs[0]
            p.text = f"{section.get('id', '')}. {section.get('title', '')}"
            p.font.size = PptxPt(28)
            p.font.bold = True
            p.font.color.rgb = PACIFIC_BLUE

            # 내용 또는 차트
            chart_key = section.get("chart_key")
            if chart_key and chart_paths and chart_key in chart_paths:
                chart_path = chart_paths[chart_key]
                if Path(chart_path).exists():
                    slide.shapes.add_picture(
                        str(chart_path), PptxInches(1), PptxInches(1.5), width=PptxInches(11)
                    )
            else:
                # 텍스트 내용
                content = section.get("content", "")
                if content:
                    content_box = slide.shapes.add_textbox(
                        PptxInches(0.5), PptxInches(1.5), PptxInches(12), PptxInches(5.5)
                    )
                    tf = content_box.text_frame
                    tf.word_wrap = True

                    lines = content.split("\n")[:15]  # 최대 15줄
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
                        p.text = line
                        p.font.size = PptxPt(16)
                        p.font.color.rgb = PACIFIC_BLUE if line.startswith("■") else GRAY
                        p.space_before = PptxPt(8)

        # 저장
        if not output_filename:
            output_filename = f"AMORE_Presentation_{start_date}_{end_date}.pptx"

        output_path = self.output_dir / output_filename
        prs.save(output_path)

        logger.info(f"PPTX presentation generated: {output_path}")
        return output_path
