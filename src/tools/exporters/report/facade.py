"""
Unified report generator
========================
DOCX / PPTX / PDF 세 생성기를 조합하는 단일 진입점.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from .docx import DocxReportGenerator
from .pdf import PdfReportGenerator
from .pptx import PptxReportGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# Unified Report Generator
# =============================================================================


class ReportGenerator:
    """
    통합 리포트 생성기

    DOCX, PPTX, PDF 포맷을 모두 지원하는 통합 인터페이스
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.docx_generator = DocxReportGenerator(output_dir)
        self.pptx_generator = PptxReportGenerator(output_dir)
        self.pdf_generator = PdfReportGenerator(output_dir)

    def generate(
        self,
        report_data: dict[str, Any],
        chart_paths: dict[str, Path] = None,
        formats: list[str] = None,
        base_filename: str = None,
    ) -> dict[str, Path]:
        """
        지정된 포맷으로 리포트 생성

        Args:
            report_data: 리포트 데이터
            chart_paths: 차트 이미지 경로
            formats: 생성할 포맷 목록 ["docx", "pptx", "pdf"]
            base_filename: 기본 파일명 (확장자 제외)

        Returns:
            {"docx": Path, "pptx": Path, "pdf": Path}
        """
        if formats is None:
            formats = ["docx"]

        start_date = report_data.get("start_date", "")
        end_date = report_data.get("end_date", "")

        if not base_filename:
            base_filename = f"AMORE_Report_{start_date}_{end_date}"

        results = {}

        # DOCX 생성
        if "docx" in formats:
            docx_path = self.docx_generator.generate_analyst_report(
                report_data,
                chart_paths,
                output_filename=f"{base_filename}.docx",
            )
            results["docx"] = docx_path

        # PPTX 생성
        if "pptx" in formats:
            pptx_path = self.pptx_generator.generate_presentation(
                report_data,
                chart_paths,
                output_filename=f"{base_filename}.pptx",
            )
            if pptx_path:
                results["pptx"] = pptx_path

        # PDF 생성 (DOCX 변환)
        if "pdf" in formats:
            if "docx" in results:
                pdf_path = self.pdf_generator.convert_docx_to_pdf(results["docx"])
                if pdf_path:
                    results["pdf"] = pdf_path
            else:
                # DOCX 없이 PDF만 요청된 경우, 먼저 DOCX 생성 후 변환
                temp_docx = self.docx_generator.generate_analyst_report(
                    report_data,
                    chart_paths,
                    output_filename=f"{base_filename}_temp.docx",
                )
                pdf_path = self.pdf_generator.convert_docx_to_pdf(temp_docx)
                if pdf_path:
                    results["pdf"] = pdf_path
                # 임시 DOCX 삭제
                temp_docx.unlink(missing_ok=True)

        return results
