"""
리포트 생성기 (Report Generator) — 호환 파사드
==============================================
실제 구현은 ``src.tools.exporters.report`` 패키지로 분할됐다 (Phase 4).

- ``report.design`` : DesignSystem (Pacific Blue #001C58 / Amore Blue #1F5795 ...)
- ``report.docx``   : DocxReportGenerator (표지·목차·본문·KPI 카드·차트)
- ``report.pptx``   : PptxReportGenerator
- ``report.pdf``    : PdfReportGenerator (WeasyPrint / LibreOffice 변환)
- ``report.facade`` : ReportGenerator (세 포맷 통합)

새 코드는 ``from src.tools.exporters.report import ...`` 를 쓴다.
"""

from src.tools.exporters.report import (
    DesignSystem,
    DocxReportGenerator,
    PdfReportGenerator,
    PptxReportGenerator,
    ReportGenerator,
)

__all__ = [
    "DesignSystem",
    "DocxReportGenerator",
    "PdfReportGenerator",
    "PptxReportGenerator",
    "ReportGenerator",
]
