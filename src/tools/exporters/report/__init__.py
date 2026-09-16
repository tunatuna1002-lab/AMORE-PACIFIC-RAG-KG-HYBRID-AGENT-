"""
Report generation (AMOREPACIFIC IR style)
=========================================
``report_generator.py`` (1,257줄)를 책임 단위로 분할한 패키지 (Phase 4).

- ``design``  : DesignSystem (색상·폰트·로고 경로 상수)
- ``assets``  : 프로젝트 루트·로고 파일 탐색 (docx/pptx 공용)
- ``docx``    : DocxReportGenerator
- ``pptx``    : PptxReportGenerator
- ``pdf``     : PdfReportGenerator
- ``facade``  : ReportGenerator (세 포맷 통합 진입점)

``src.tools.exporters.report_generator`` 는 이 패키지를 재수출하는 호환 파사드다.
"""

from .design import DesignSystem
from .docx import DocxReportGenerator
from .facade import ReportGenerator
from .pdf import PdfReportGenerator
from .pptx import PptxReportGenerator

__all__ = [
    "DesignSystem",
    "DocxReportGenerator",
    "PdfReportGenerator",
    "PptxReportGenerator",
    "ReportGenerator",
]
