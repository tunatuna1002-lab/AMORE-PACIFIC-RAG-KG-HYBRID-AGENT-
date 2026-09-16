"""
PDF report generator
====================
WeasyPrint 직접 생성 또는 DOCX → PDF 변환(LibreOffice).
"""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# PDF Report Generator
# =============================================================================


class PdfReportGenerator:
    """
    PDF 리포트 생성기

    DOCX를 PDF로 변환하거나 WeasyPrint로 직접 생성
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # WeasyPrint 가용성 확인
        try:
            import importlib.util

            self._weasyprint_available = importlib.util.find_spec("weasyprint") is not None
        except Exception:
            self._weasyprint_available = False

        if not self._weasyprint_available:
            logger.warning("WeasyPrint not installed. PDF generation limited.")

    def generate_from_html(
        self,
        html_content: str,
        output_filename: str = None,
    ) -> Path | None:
        """
        HTML에서 PDF 생성

        Args:
            html_content: HTML 문자열
            output_filename: 출력 파일명

        Returns:
            생성된 파일 경로
        """
        if not self._weasyprint_available:
            logger.error("WeasyPrint not installed")
            return None

        from weasyprint import CSS, HTML

        if not output_filename:
            output_filename = f"AMORE_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        output_path = self.output_dir / output_filename

        # CSS 스타일
        css = CSS(
            string="""
            @page {
                size: A4;
                margin: 2cm;
            }
            body {
                font-family: Arial, 'Noto Sans KR', sans-serif;
                font-size: 11pt;
                line-height: 1.5;
                color: #333333;
            }
            h1 {
                color: #001C58;
                font-size: 24pt;
                border-bottom: 2px solid #001C58;
                padding-bottom: 8px;
            }
            h2 {
                color: #1F5795;
                font-size: 16pt;
            }
            .highlight {
                color: #001C58;
                font-weight: bold;
            }
            .kpi-card {
                background: #F5F5F5;
                padding: 16px;
                border-radius: 8px;
                text-align: center;
            }
            .kpi-value {
                font-size: 28pt;
                font-weight: bold;
                color: #001C58;
            }
            .footer {
                text-align: center;
                font-size: 9pt;
                color: #7D7D7D;
                margin-top: 32px;
            }
        """
        )

        HTML(string=html_content).write_pdf(output_path, stylesheets=[css])

        logger.info(f"PDF generated: {output_path}")
        return output_path

    def convert_docx_to_pdf(self, docx_path: Path) -> Path | None:
        """
        DOCX를 PDF로 변환 (LibreOffice 또는 docx2pdf 사용)

        Args:
            docx_path: DOCX 파일 경로

        Returns:
            생성된 PDF 경로 (변환 실패 시 None)
        """
        import subprocess

        output_path = docx_path.with_suffix(".pdf")

        # 1. docx2pdf 시도 (Windows/macOS)
        try:
            from docx2pdf import convert

            convert(str(docx_path), str(output_path))
            if output_path.exists():
                logger.info(f"PDF converted via docx2pdf: {output_path}")
                return output_path
        except ImportError:
            logger.warning("Suppressed ImportError", exc_info=True)
        except Exception as e:
            logger.warning(f"docx2pdf failed: {e}")

        # 2. LibreOffice 시도 (Linux/Docker)
        try:
            subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(self.output_dir),
                    str(docx_path),
                ],
                check=True,
                capture_output=True,
            )
            if output_path.exists():
                logger.info(f"PDF converted via LibreOffice: {output_path}")
                return output_path
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"LibreOffice conversion failed: {e}")

        logger.error("PDF conversion failed: no converter available")
        return None
