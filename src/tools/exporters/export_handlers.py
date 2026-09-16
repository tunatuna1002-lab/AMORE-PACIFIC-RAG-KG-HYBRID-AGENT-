"""
Export Renderers & Job Handlers

DOCX 렌더링(동기 API 경로와 비동기 작업 큐가 **함께** 쓰는 단일 구현)과
job_queue 핸들러들. 페이지 새로고침에도 다운로드가 지속되도록 job_queue.py와
함께 사용한다.

Usage:
    from src.tools.utilities.job_queue import get_job_queue
    from src.tools.exporters.export_handlers import register_all_handlers

    queue = get_job_queue()
    await queue.initialize()
    register_all_handlers(queue)
    await queue.start_worker()

데이터 수집/집계는 ``src/application/services/export_service.py``가, 문서 조립은
이 모듈이 담당한다. 이 모듈은 ``src.api``를 import 하지 않는다 (계층 역전 제거).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

from src.application.services.export_service import (
    ANALYST_REPORT_TOC,
    AnalystReportContext,
    build_analyst_report_context,
    filter_reference_lines,
)
from src.tools.storage.sqlite_storage import get_sqlite_storage
from src.tools.utilities.job_queue import JobQueue, JobType

logger = logging.getLogger(__name__)

# AMOREPACIFIC colors
PACIFIC_BLUE = RGBColor(0, 28, 88)  # #001C58
AMORE_BLUE = RGBColor(31, 87, 149)  # #1F5795
GRAY = RGBColor(125, 125, 125)  # #7D7D7D

STRATEGY_FALLBACK = """
1. Top 10 유지 전략: 현재 상위권 제품의 리뷰 관리 및 재고 확보를 통한 포지션 유지

2. 경쟁사 모니터링: e.l.f., Maybelline 등 주요 경쟁사의 가격 및 프로모션 동향 파악

3. 신규 진입 기회: Lip Care 카테고리 외 Face Powder, Toner 등 확장 가능성 검토
"""

NO_SIGNALS_HINT = (
    "외부 신호를 수집하려면:\n"
    "1. RSS 피드 자동 수집: /api/signals/fetch/rss\n"
    "2. Reddit 트렌드 수집: /api/signals/fetch/reddit\n"
    "3. 수동 입력: /api/signals/manual"
)


def register_all_handlers(queue: JobQueue) -> None:
    """모든 export 핸들러 등록"""
    queue.register_handler(JobType.EXPORT_ANALYST_REPORT.value, handle_export_analyst_report)
    queue.register_handler(JobType.EXPORT_EXCEL.value, handle_export_excel)
    logger.info("Registered all export handlers")


# ===========================================================================
# 1. 인사이트 리포트 (대시보드 스냅샷 기반)
# ===========================================================================


@dataclass
class InsightReportView:
    """
    인사이트 리포트가 필요로 하는 값들 (대시보드 JSON을 호출자가 평탄화해서 넘긴다).

    ``src/api/dashboard_shape.py``의 어댑터가 exporter/legacy 두 형태를 흡수하므로
    렌더러는 한 가지 형태만 알면 된다.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    brands: list[dict[str, Any]] = field(default_factory=list)
    categories: list[dict[str, Any]] = field(default_factory=list)
    products: list[dict[str, Any]] = field(default_factory=list)
    strategic_insights: list[dict[str, Any]] = field(default_factory=list)
    include_strategy: bool = True
    include_external_signals: bool = False
    signals_section: str = ""
    signals_days: int = 7

    @property
    def filename(self) -> str:
        return f"AMORE_Insight_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.docx"


def render_insight_report(view: InsightReportView) -> Document:
    """인사이트 리포트 DOCX (동기 API 경로와 비동기 작업이 공유하는 단일 구현)."""
    doc = Document()

    # 스타일 설정
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Arial"
    font.size = Pt(11)

    # ===== 표지 =====
    title = doc.add_heading("AMORE INSIGHT Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph("LANEIGE Amazon US 분석 리포트")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 날짜
    metadata = view.metadata
    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_para.add_run(
        f"분석 기준일: {metadata.get('data_date', datetime.now().strftime('%Y-%m-%d'))}"
    )
    date_para.add_run(f"\n생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    doc.add_page_break()

    # ===== 목차 =====
    doc.add_heading("목차", 1)
    for item in [
        "1. 요약 통계",
        "2. 브랜드별 성과",
        "3. 카테고리별 분석",
        "4. 주요 제품",
        "5. AI 인사이트 및 전략 제언",
        "6. 외부 트렌드 신호",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_page_break()

    # ===== 1. 요약 통계 =====
    doc.add_heading("1. 요약 통계", 1)
    summary = view.summary

    table = doc.add_table(rows=5, cols=2)
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    stats = [
        ("총 제품 수", summary.get("total_products", 0)),
        ("크롤링 카테고리", summary.get("categories_count", 0)),
        ("LANEIGE 제품 수", summary.get("laneige_products", 0)),
        ("평균 가격", f"${summary.get('avg_price', 0):.2f}"),
        ("데이터 날짜", metadata.get("data_date", "N/A")),
    ]

    for i, (label, value) in enumerate(stats):
        table.rows[i].cells[0].text = label
        table.rows[i].cells[1].text = str(value)

    doc.add_paragraph()

    # ===== 2. 브랜드별 성과 =====
    doc.add_heading("2. 브랜드별 성과", 1)

    if view.brands:
        # 상위 10개 브랜드
        top_brands = sorted(view.brands, key=lambda x: x.get("product_count", 0), reverse=True)[:10]

        brand_table = doc.add_table(rows=len(top_brands) + 1, cols=4)
        brand_table.style = "Light Grid Accent 1"

        headers = brand_table.rows[0].cells
        headers[0].text = "브랜드"
        headers[1].text = "제품 수"
        headers[2].text = "평균 순위"
        headers[3].text = "SoS (%)"

        for i, brand in enumerate(top_brands, start=1):
            cells = brand_table.rows[i].cells
            cells[0].text = brand.get("brand", "Unknown")
            cells[1].text = str(brand.get("product_count", 0))
            cells[2].text = f"{brand.get('avg_rank', 0):.1f}"
            cells[3].text = f"{brand.get('sos', 0):.2f}%"
    else:
        doc.add_paragraph("브랜드 데이터가 없습니다.")

    doc.add_page_break()

    # ===== 3. 카테고리별 분석 =====
    doc.add_heading("3. 카테고리별 분석", 1)

    if view.categories:
        for category in view.categories:
            doc.add_heading(category.get("category", "Unknown"), 2)

            for label, value in [
                ("총 제품 수", category.get("total_products", 0)),
                ("LANEIGE 제품 수", category.get("laneige_products", 0)),
                ("평균 가격", f"${category.get('avg_price', 0):.2f}"),
                ("HHI", f"{category.get('hhi', 0):.2f}"),
                ("CPI", f"{category.get('cpi', 0):.2f}"),
            ]:
                doc.add_paragraph(f"{label}: {value}")

            doc.add_paragraph()
    else:
        doc.add_paragraph("카테고리 데이터가 없습니다.")

    doc.add_page_break()

    # ===== 4. 주요 제품 =====
    doc.add_heading("4. 주요 제품 (LANEIGE Top 10)", 1)

    laneige_products = sorted(view.products, key=lambda x: x.get("rank", 999))[:10]

    if laneige_products:
        product_table = doc.add_table(rows=len(laneige_products) + 1, cols=5)
        product_table.style = "Light Grid Accent 1"

        headers = product_table.rows[0].cells
        headers[0].text = "순위"
        headers[1].text = "제품명"
        headers[2].text = "카테고리"
        headers[3].text = "가격"
        headers[4].text = "평점"

        for i, product in enumerate(laneige_products, start=1):
            cells = product_table.rows[i].cells
            cells[0].text = str(product.get("rank", "N/A"))
            cells[1].text = product.get("title", "Unknown")[:50]
            cells[2].text = product.get("category", "Unknown")
            cells[3].text = f"${product.get('price', 0):.2f}" if product.get("price") else "N/A"
            cells[4].text = str(product.get("rating", "N/A"))
    else:
        doc.add_paragraph("LANEIGE 제품이 없습니다.")

    doc.add_page_break()

    # ===== 5. AI 인사이트 및 전략 제언 =====
    if view.include_strategy:
        doc.add_heading("5. AI 인사이트 및 전략 제언", 1)

        if view.strategic_insights:
            for insight in view.strategic_insights:
                doc.add_heading(insight.get("title", "Insight"), 2)
                doc.add_paragraph(insight.get("content", ""))
                doc.add_paragraph()
        else:
            # 폴백 전략
            doc.add_paragraph(STRATEGY_FALLBACK)

    # ===== 6. 외부 트렌드 신호 =====
    if view.include_external_signals:
        doc.add_heading("6. 외부 트렌드 신호", 1)

        if view.signals_section:
            doc.add_paragraph(f"분석 기간: 최근 {view.signals_days}일")
            doc.add_paragraph()

            # 신호 섹션별로 파싱하여 추가
            for line in view.signals_section.split("\n"):
                if line.startswith("■"):
                    doc.add_heading(line.replace("■ ", ""), 2)
                elif line.startswith("•"):
                    doc.add_paragraph(line, style="List Bullet")
                elif line.strip():
                    doc.add_paragraph(line)
        else:
            doc.add_paragraph("수집된 외부 트렌드 신호가 없습니다.")
            doc.add_paragraph()
            doc.add_paragraph(NO_SIGNALS_HINT)

    # ===== 푸터 =====
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run(f"© {datetime.now().year} AMORE Pacific - Confidential").italic = True

    return doc


# ===========================================================================
# 2. 애널리스트 리포트 (8 섹션, IR 표지)
# ===========================================================================


def render_analyst_report(ctx: AnalystReportContext, include_charts: bool = True) -> Document:
    """
    애널리스트 리포트 DOCX (8 섹션).

    동기 ``POST /api/export/analyst-report``와 비동기 작업이 **같은** 문서를
    생성한다 (F6: 두 경로가 서로 다른 문서를 만들던 중복 제거).
    """
    from src.tools.exporters.report_generator import DocxReportGenerator

    design_gen = DocxReportGenerator()
    doc = Document()
    report = ctx.report
    chart_paths = ctx.chart_paths

    # 표지/목차에 Arita 폰트 및 페이지 여백 설정
    design_gen._setup_document_styles(doc)
    design_gen._setup_page_margins(doc)

    # ===== 표지 페이지 - Pacific Blue 헤더바 + AMOREPACIFIC 로고 =====
    design_gen._add_cover_page(
        doc,
        title="LANEIGE Amazon US 경쟁력 분석 보고서",
        subtitle="Weekly Insight Report",
        date_range=f"{ctx.start_date} ~ {ctx.end_date}",
        generation_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    # ===== 목차 페이지 =====
    design_gen._add_toc_page(doc, list(ANALYST_REPORT_TOC))

    def add_section_content(section, section_charts: list[str] | None = None):
        """DocxReportGenerator 스타일로 섹션 콘텐츠 추가"""
        design_gen._add_section_heading(doc, section.section_id, section.section_title)

        # 본문 내용 (참고자료 라인은 8장으로 모으므로 본문에서 제거)
        if section.content:
            is_first_heading = True
            for raw_line in filter_reference_lines(section.content).split("\n"):
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("■"):
                    # 목차(■) - IR 스타일 적용
                    design_gen._add_content_paragraph(
                        doc, line, is_highlight=True, add_space_before=not is_first_heading
                    )
                    is_first_heading = False
                elif line.startswith("•") or line.startswith("-"):
                    # 불릿 포인트
                    design_gen._add_content_paragraph(doc, line.lstrip("•- "), is_bullet=True)
                else:
                    design_gen._add_content_paragraph(doc, line)

        # 차트 추가
        if section_charts and include_charts:
            for chart_key in section_charts:
                if chart_key in chart_paths:
                    doc.add_paragraph()
                    design_gen._add_chart_image(doc, chart_paths[chart_key])

        doc.add_page_break()

    if report.executive_summary:
        add_section_content(report.executive_summary, ["sos_trend"])
    if report.laneige_analysis:
        add_section_content(report.laneige_analysis, ["sos_trend", "product_ranks"])
    if report.competitive_analysis:
        add_section_content(report.competitive_analysis, ["brand_comparison"])
    if report.market_trends:
        add_section_content(report.market_trends, ["hhi_trend"])
    if report.external_signals:
        add_section_content(report.external_signals)
    if report.risks_opportunities:
        add_section_content(report.risks_opportunities)
    if report.strategic_recommendations:
        add_section_content(report.strategic_recommendations)

    # ===== Section 8: 참고자료 =====
    design_gen._add_section_heading(doc, 8, "참고자료 (References)")

    # 8.1 외부 자료
    doc.add_heading("8.1 외부 자료", 2)
    for run in doc.paragraphs[-1].runs:
        run.font.color.rgb = AMORE_BLUE
    external_refs = ctx.tracker.get_formatted_references(source_type="external")
    if external_refs:
        for ref in external_refs.split("\n"):
            if ref.strip():
                design_gen._add_content_paragraph(doc, ref, is_bullet=True)
    else:
        design_gen._add_content_paragraph(doc, "외부 자료 없음")

    # 8.2 데이터 소스
    doc.add_heading("8.2 데이터 소스", 2)
    for run in doc.paragraphs[-1].runs:
        run.font.color.rgb = AMORE_BLUE
    data_refs = ctx.tracker.get_formatted_references(source_type="data")
    if data_refs:
        for ref in data_refs.split("\n"):
            if ref.strip():
                design_gen._add_content_paragraph(doc, ref, is_bullet=True)

    return doc


# ===========================================================================
# 3. Job handlers
# ===========================================================================


async def handle_export_analyst_report(job_id: str, params: dict, queue: JobQueue) -> str:
    """
    애널리스트 리포트 (8 섹션) 생성 핸들러

    Args:
        job_id: 작업 ID
        params: {start_date, end_date, include_charts, include_external_signals}
        queue: JobQueue 인스턴스

    Returns:
        생성된 파일 경로
    """
    start_date = params.get("start_date")
    end_date = params.get("end_date")
    include_charts = params.get("include_charts", True)
    include_external_signals = params.get("include_external_signals", True)

    if not start_date or not end_date:
        raise ValueError("start_date and end_date are required")

    async def _progress(pct: int, message: str) -> None:
        await queue.update_progress(job_id, pct, message)

    ctx = await build_analyst_report_context(
        start_date=start_date,
        end_date=end_date,
        include_charts=include_charts,
        include_external_signals=include_external_signals,
        progress=_progress,
    )

    await queue.update_progress(job_id, 70, "DOCX 문서 생성 중...")
    doc = render_analyst_report(ctx, include_charts=include_charts)

    await queue.update_progress(job_id, 95, "파일 저장 중...")
    output_path = os.path.join(queue.output_dir, ctx.filename)
    doc.save(output_path)
    logger.info(f"Analyst report saved: {output_path}")

    ctx.cleanup_charts()

    return output_path


async def handle_export_excel(job_id: str, params: dict, queue: JobQueue) -> str:
    """
    Excel 데이터 내보내기 핸들러

    Args:
        job_id: 작업 ID
        params: {start_date, end_date, include_metrics}
        queue: JobQueue 인스턴스

    Returns:
        생성된 파일 경로
    """
    start_date = params.get("start_date")
    end_date = params.get("end_date")
    include_metrics = params.get("include_metrics", True)

    await queue.update_progress(job_id, 20, "SQLite 데이터 로드 중...")

    storage = get_sqlite_storage()
    await storage.initialize()

    await queue.update_progress(job_id, 50, "Excel 파일 생성 중...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"AMORE_Data_{timestamp}.xlsx"
    output_path = os.path.join(queue.output_dir, filename)

    storage.export_to_excel(
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
        include_metrics=include_metrics,
    )

    await queue.update_progress(job_id, 100, "완료")

    logger.info(f"Excel saved: {output_path}")
    return output_path
