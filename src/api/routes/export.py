"""
Export Routes - Document and data export endpoints
"""
import logging
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
import tempfile
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from src.api.dependencies import load_dashboard_data
from src.tools.sqlite_storage import get_sqlite_storage
from src.tools.external_signal_collector import ExternalSignalCollector, SignalTier
from src.tools.period_analyzer import PeriodAnalyzer
from src.tools.chart_generator import ChartGenerator
from src.tools.reference_tracker import ReferenceTracker, ReferenceType
from src.agents.period_insight_agent import PeriodInsightAgent

router = APIRouter(prefix="/api/export", tags=["export"])


class ExportRequest(BaseModel):
    """내보내기 요청"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    include_strategy: bool = True
    include_external_signals: bool = True  # External Signal 포함 여부


class AnalystReportRequest(BaseModel):
    """애널리스트 리포트 요청"""
    start_date: str  # Required: YYYY-MM-DD
    end_date: str    # Required: YYYY-MM-DD
    include_charts: bool = True
    include_external_signals: bool = True


async def _get_external_signals(
    days: int = 7,
    brands: list = None,
    include_tavily: bool = True
) -> dict:
    """
    External Signal 수집 및 보고서 섹션 생성

    Args:
        days: 검색 기간 (일)
        brands: 검색할 브랜드 리스트
        include_tavily: Tavily 뉴스 검색 포함 여부

    Returns:
        {
            "signals": List[ExternalSignal],
            "report_section": str
        }
    """
    logger = logging.getLogger(__name__)
    all_signals = []

    try:
        collector = ExternalSignalCollector()
        await collector.initialize()

        # 브랜드 기본값
        if not brands:
            brands = ["LANEIGE", "COSRX", "K-Beauty"]

        # 1. Tavily 뉴스 검색 (최적화)
        if include_tavily:
            try:
                # 검색 기간: 최소 14일, 최대 30일 (분석 기간에 맞춤)
                search_days = min(max(days, 14), 30)
                tavily_signals = await collector.fetch_tavily_news(
                    brands=brands[:3],
                    topics=["K-Beauty skincare", "Amazon beauty bestseller"],
                    days=search_days,
                    max_results=15  # 더 많은 결과 수집
                )
                all_signals.extend(tavily_signals)
                logger.info(f"Collected {len(tavily_signals)} Tavily news signals (search_days={search_days})")
            except Exception as e:
                logger.warning(f"Tavily news fetch failed: {e}")

        # 2. RSS 피드 수집 (기존)
        try:
            rss_signals = await collector.fetch_all_rss_feeds(
                keywords=brands + ["skincare", "lip care"]
            )
            all_signals.extend(rss_signals)
            logger.info(f"Collected {len(rss_signals)} RSS signals")
        except Exception as e:
            logger.warning(f"RSS fetch failed: {e}")

        # 3. Reddit 트렌드 수집 (기존)
        try:
            reddit_signals = await collector.fetch_reddit_trends()
            all_signals.extend(reddit_signals)
            logger.info(f"Collected {len(reddit_signals)} Reddit signals")
        except Exception as e:
            logger.warning(f"Reddit fetch failed: {e}")

        # 기존 수집된 신호도 추가
        if collector.signals:
            all_signals.extend(collector.signals)

        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_signals = []
        for signal in all_signals:
            url = getattr(signal, 'url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_signals.append(signal)
            elif not url:
                unique_signals.append(signal)

        # 보고서 섹션 생성
        if unique_signals:
            # 컬렉터에 신호 추가 후 보고서 생성
            for signal in unique_signals:
                if signal not in collector.signals:
                    collector.signals.append(signal)
            report_section = collector.generate_report_section(days=days)
        else:
            report_section = ""

        return {
            "signals": unique_signals,
            "report_section": report_section
        }

    except Exception as e:
        logger.warning(f"External signal collection failed: {e}")
        return {"signals": [], "report_section": ""}
    finally:
        try:
            await collector.close()
        except Exception:
            pass


@router.post("/docx")
async def export_docx(request: ExportRequest):
    """
    인사이트 리포트 DOCX 생성 및 다운로드
    """
    data = load_dashboard_data()
    if not data:
        raise HTTPException(status_code=404, detail="Dashboard data not found")

    # DOCX 문서 생성
    doc = Document()

    # 스타일 설정
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(11)

    # ===== 표지 =====
    title = doc.add_heading('AMORE INSIGHT Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph('LANEIGE Amazon US 분석 리포트')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 날짜
    metadata = data.get("metadata", {})
    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_para.add_run(f"분석 기준일: {metadata.get('data_date', datetime.now().strftime('%Y-%m-%d'))}")
    date_para.add_run(f"\n생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    doc.add_page_break()

    # ===== 목차 =====
    doc.add_heading('목차', 1)
    toc_items = [
        "1. 요약 통계",
        "2. 브랜드별 성과",
        "3. 카테고리별 분석",
        "4. 주요 제품",
        "5. AI 인사이트 및 전략 제언",
        "6. 외부 트렌드 신호"
    ]
    for item in toc_items:
        doc.add_paragraph(item, style='List Bullet')

    doc.add_page_break()

    # ===== 1. 요약 통계 =====
    doc.add_heading('1. 요약 통계', 1)
    summary = data.get("summary", {})

    table = doc.add_table(rows=5, cols=2)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    stats = [
        ("총 제품 수", summary.get("total_products", 0)),
        ("크롤링 카테고리", summary.get("categories_count", 0)),
        ("LANEIGE 제품 수", summary.get("laneige_products", 0)),
        ("평균 가격", f"${summary.get('avg_price', 0):.2f}"),
        ("데이터 날짜", metadata.get("data_date", "N/A"))
    ]

    for i, (label, value) in enumerate(stats):
        table.rows[i].cells[0].text = label
        table.rows[i].cells[1].text = str(value)

    doc.add_paragraph()

    # ===== 2. 브랜드별 성과 =====
    doc.add_heading('2. 브랜드별 성과', 1)

    brand_data = data.get("brand_metrics", [])
    if brand_data:
        # 상위 10개 브랜드
        top_brands = sorted(brand_data, key=lambda x: x.get("product_count", 0), reverse=True)[:10]

        brand_table = doc.add_table(rows=len(top_brands) + 1, cols=4)
        brand_table.style = 'Light Grid Accent 1'

        # 헤더
        headers = brand_table.rows[0].cells
        headers[0].text = "브랜드"
        headers[1].text = "제품 수"
        headers[2].text = "평균 순위"
        headers[3].text = "SoS (%)"

        # 데이터
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
    doc.add_heading('3. 카테고리별 분석', 1)

    category_data = data.get("category_metrics", [])
    if category_data:
        for category in category_data:
            cat_name = category.get("category", "Unknown")
            doc.add_heading(cat_name, 2)

            cat_stats = [
                ("총 제품 수", category.get("total_products", 0)),
                ("LANEIGE 제품 수", category.get("laneige_products", 0)),
                ("평균 가격", f"${category.get('avg_price', 0):.2f}"),
                ("HHI", f"{category.get('hhi', 0):.2f}"),
                ("CPI", f"{category.get('cpi', 0):.2f}")
            ]

            for label, value in cat_stats:
                doc.add_paragraph(f"{label}: {value}")

            doc.add_paragraph()
    else:
        doc.add_paragraph("카테고리 데이터가 없습니다.")

    doc.add_page_break()

    # ===== 4. 주요 제품 =====
    doc.add_heading('4. 주요 제품 (LANEIGE Top 10)', 1)

    products = data.get("products", [])
    laneige_products = [p for p in products if p.get("brand", "").upper() == "LANEIGE"]
    laneige_products = sorted(laneige_products, key=lambda x: x.get("rank", 999))[:10]

    if laneige_products:
        product_table = doc.add_table(rows=len(laneige_products) + 1, cols=5)
        product_table.style = 'Light Grid Accent 1'

        # 헤더
        headers = product_table.rows[0].cells
        headers[0].text = "순위"
        headers[1].text = "제품명"
        headers[2].text = "카테고리"
        headers[3].text = "가격"
        headers[4].text = "평점"

        # 데이터
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
    if request.include_strategy:
        doc.add_heading('5. AI 인사이트 및 전략 제언', 1)

        insights = data.get("ai_insights", {})
        strategic_insights = insights.get("strategic_insights", [])

        if strategic_insights:
            for insight in strategic_insights:
                doc.add_heading(insight.get("title", "Insight"), 2)
                doc.add_paragraph(insight.get("content", ""))
                doc.add_paragraph()
        else:
            # 폴백 전략
            doc.add_paragraph("""
1. Top 10 유지 전략: 현재 상위권 제품의 리뷰 관리 및 재고 확보를 통한 포지션 유지

2. 경쟁사 모니터링: e.l.f., Maybelline 등 주요 경쟁사의 가격 및 프로모션 동향 파악

3. 신규 진입 기회: Lip Care 카테고리 외 Face Powder, Toner 등 확장 가능성 검토
""")

    # ===== 6. 외부 트렌드 신호 =====
    if request.include_external_signals:
        doc.add_heading('6. 외부 트렌드 신호', 1)

        # 분석 기간 계산
        if request.start_date and request.end_date:
            try:
                start = datetime.strptime(request.start_date, "%Y-%m-%d")
                end = datetime.strptime(request.end_date, "%Y-%m-%d")
                days = (end - start).days + 1
            except ValueError:
                days = 7
        else:
            days = 7

        signals_result = await _get_external_signals(days=days)
        external_signals_text = signals_result.get("report_section", "")

        if external_signals_text:
            doc.add_paragraph(f"분석 기간: 최근 {days}일")
            doc.add_paragraph()

            # 신호 섹션별로 파싱하여 추가
            for line in external_signals_text.split("\n"):
                if line.startswith("■"):
                    doc.add_heading(line.replace("■ ", ""), 2)
                elif line.startswith("•"):
                    doc.add_paragraph(line, style='List Bullet')
                elif line.strip():
                    doc.add_paragraph(line)
        else:
            doc.add_paragraph("수집된 외부 트렌드 신호가 없습니다.")
            doc.add_paragraph()
            doc.add_paragraph(
                "외부 신호를 수집하려면:\n"
                "1. RSS 피드 자동 수집: /api/signals/fetch/rss\n"
                "2. Reddit 트렌드 수집: /api/signals/fetch/reddit\n"
                "3. 수동 입력: /api/signals/manual"
            )

    # ===== 푸터 =====
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("© 2025 AMORE Pacific - Confidential").italic = True

    # BytesIO로 저장
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    # 파일명 생성
    filename = f"AMORE_Insight_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.docx"

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.post("/analyst-report")
async def export_analyst_report(request: AnalystReportRequest):
    """
    기간별 애널리스트 리포트 DOCX 생성 (8 Sections)

    Args:
        request: AnalystReportRequest with start_date, end_date, options

    Returns:
        StreamingResponse with DOCX file
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating analyst report: {request.start_date} ~ {request.end_date}")

    # AMOREPACIFIC colors
    PACIFIC_BLUE = RGBColor(0, 28, 88)  # #001C58
    AMORE_BLUE = RGBColor(31, 87, 149)  # #1F5795
    GRAY = RGBColor(125, 125, 125)      # #7D7D7D

    try:
        # 1. Period Analysis
        analyzer = PeriodAnalyzer()
        analysis = await analyzer.analyze(request.start_date, request.end_date)

        if analysis.total_days == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for period {request.start_date} ~ {request.end_date}"
            )

        # 2. External Signals (optional) - Tavily 뉴스 포함
        external_signals = None
        external_signals_list = []
        if request.include_external_signals:
            try:
                # 새로운 통합 함수 사용 (Tavily + RSS + Reddit)
                signals_result = await _get_external_signals(
                    days=analysis.total_days,
                    brands=["LANEIGE", "COSRX", "K-Beauty"],
                    include_tavily=True
                )
                if signals_result.get("signals"):
                    external_signals = signals_result
                    external_signals_list = signals_result.get("signals", [])
                    logger.info(f"Collected {len(external_signals_list)} external signals for report")
            except Exception as e:
                logger.warning(f"External signal collection failed: {e}")

        # 3. Generate Insights
        insight_agent = PeriodInsightAgent()
        report = await insight_agent.generate_report(analysis, external_signals=external_signals)

        # 4. Generate Charts
        charts = {}
        chart_paths = {}
        if request.include_charts:
            temp_dir = tempfile.mkdtemp()
            chart_gen = ChartGenerator(output_dir=temp_dir)
            chart_paths = chart_gen.generate_all_charts(analysis)
            logger.info(f"Generated {len(chart_paths)} charts in {temp_dir}")

        # 5. Reference Tracker
        tracker = ReferenceTracker()
        tracker.auto_add_amazon_sources(
            start_date=request.start_date,
            end_date=request.end_date
        )

        # 외부 신호를 참고자료에 자동 추가 (Tavily 뉴스, RSS, Reddit 등)
        if external_signals_list:
            added_refs = tracker.add_external_signals(external_signals_list)
            logger.info(f"Added {added_refs} external signal references")

        # 6. Create DOCX Document
        doc = Document()

        # Style setup
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Arial'
        font.size = Pt(11)

        # ===== 표지 (Cover Page) =====
        title = doc.add_heading('LANEIGE Amazon US 경쟁력 분석 보고서', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = title.runs[0]
        run.font.color.rgb = PACIFIC_BLUE

        subtitle = doc.add_paragraph(f'분석 기간: {request.start_date} ~ {request.end_date}')
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

        date_para = doc.add_paragraph()
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_para.add_run(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        doc.add_page_break()

        # ===== 목차 (Table of Contents) =====
        toc_heading = doc.add_heading('목차', 1)
        toc_heading.runs[0].font.color.rgb = PACIFIC_BLUE

        toc_items = [
            "1. Executive Summary",
            "2. LANEIGE 심층 분석",
            "3. 경쟁 환경 분석",
            "4. 시장 동향",
            "5. 외부 신호 분석",
            "6. 리스크 및 기회 요인",
            "7. 전략 제언",
            "8. 참고자료 (References)"
        ]
        for item in toc_items:
            doc.add_paragraph(item, style='List Bullet')

        doc.add_page_break()

        # ===== Section 1: Executive Summary =====
        if report.executive_summary:
            section = report.executive_summary
            heading = doc.add_heading(f'{section.section_id}. {section.section_title}', 1)
            heading.runs[0].font.color.rgb = PACIFIC_BLUE

            # Content
            for line in section.content.split('\n'):
                if line.strip():
                    if line.startswith('■'):
                        para = doc.add_paragraph(line)
                        para.runs[0].font.bold = True
                    else:
                        doc.add_paragraph(line)

            # Insert SoS chart if available
            if request.include_charts and 'sos_trend' in chart_paths:
                doc.add_paragraph()
                doc.add_picture(str(chart_paths['sos_trend']), width=Inches(6))

            doc.add_page_break()

        # ===== Section 2: LANEIGE 심층 분석 =====
        if report.laneige_analysis:
            section = report.laneige_analysis
            heading = doc.add_heading(f'{section.section_id}. {section.section_title}', 1)
            heading.runs[0].font.color.rgb = PACIFIC_BLUE

            # Content
            for line in section.content.split('\n'):
                if line.strip():
                    if line.startswith('■'):
                        para = doc.add_paragraph(line)
                        para.runs[0].font.bold = True
                    elif line.startswith('2.'):
                        doc.add_heading(line, 2)
                    else:
                        doc.add_paragraph(line)

            # Insert charts
            if request.include_charts:
                doc.add_paragraph()
                if 'sos_trend' in chart_paths:
                    doc.add_heading('일별 SoS 추이', 3)
                    doc.add_picture(str(chart_paths['sos_trend']), width=Inches(6))
                    doc.add_paragraph()

                if 'product_ranks' in chart_paths:
                    doc.add_heading('제품별 순위 변동', 3)
                    doc.add_picture(str(chart_paths['product_ranks']), width=Inches(6))

            doc.add_page_break()

        # ===== Section 3: 경쟁 환경 분석 =====
        if report.competitive_analysis:
            section = report.competitive_analysis
            heading = doc.add_heading(f'{section.section_id}. {section.section_title}', 1)
            heading.runs[0].font.color.rgb = PACIFIC_BLUE

            for line in section.content.split('\n'):
                if line.strip():
                    if line.startswith('■'):
                        para = doc.add_paragraph(line)
                        para.runs[0].font.bold = True
                    elif line.startswith('3.'):
                        doc.add_heading(line, 2)
                    else:
                        doc.add_paragraph(line)

            # Insert brand comparison chart
            if request.include_charts and 'brand_comparison' in chart_paths:
                doc.add_paragraph()
                doc.add_heading('브랜드별 점유율', 3)
                doc.add_picture(str(chart_paths['brand_comparison']), width=Inches(6))

            doc.add_page_break()

        # ===== Section 4: 시장 동향 =====
        if report.market_trends:
            section = report.market_trends
            heading = doc.add_heading(f'{section.section_id}. {section.section_title}', 1)
            heading.runs[0].font.color.rgb = PACIFIC_BLUE

            for line in section.content.split('\n'):
                if line.strip():
                    if line.startswith('■'):
                        para = doc.add_paragraph(line)
                        para.runs[0].font.bold = True
                    elif line.startswith('4.'):
                        doc.add_heading(line, 2)
                    else:
                        doc.add_paragraph(line)

            # Insert HHI chart
            if request.include_charts and 'hhi_trend' in chart_paths:
                doc.add_paragraph()
                doc.add_heading('HHI 추이', 3)
                doc.add_picture(str(chart_paths['hhi_trend']), width=Inches(6))

            doc.add_page_break()

        # ===== Section 5: 외부 신호 분석 =====
        if report.external_signals:
            section = report.external_signals
            heading = doc.add_heading(f'{section.section_id}. {section.section_title}', 1)
            heading.runs[0].font.color.rgb = PACIFIC_BLUE

            for line in section.content.split('\n'):
                if line.strip():
                    if line.startswith('■'):
                        para = doc.add_paragraph(line)
                        para.runs[0].font.bold = True
                    elif line.startswith('5.'):
                        doc.add_heading(line, 2)
                    else:
                        doc.add_paragraph(line)

            doc.add_page_break()

        # ===== Section 6: 리스크 및 기회 요인 =====
        if report.risks_opportunities:
            section = report.risks_opportunities
            heading = doc.add_heading(f'{section.section_id}. {section.section_title}', 1)
            heading.runs[0].font.color.rgb = PACIFIC_BLUE

            for line in section.content.split('\n'):
                if line.strip():
                    if line.startswith('■'):
                        para = doc.add_paragraph(line)
                        para.runs[0].font.bold = True
                    elif line.startswith('6.'):
                        doc.add_heading(line, 2)
                    else:
                        doc.add_paragraph(line)

            doc.add_page_break()

        # ===== Section 7: 전략 제언 =====
        if report.strategic_recommendations:
            section = report.strategic_recommendations
            heading = doc.add_heading(f'{section.section_id}. {section.section_title}', 1)
            heading.runs[0].font.color.rgb = PACIFIC_BLUE

            for line in section.content.split('\n'):
                if line.strip():
                    if line.startswith('■'):
                        para = doc.add_paragraph(line)
                        para.runs[0].font.bold = True
                    elif line.startswith('7.'):
                        doc.add_heading(line, 2)
                    else:
                        doc.add_paragraph(line)

            doc.add_page_break()

        # ===== Section 8: 참고자료 (References) =====
        ref_heading = doc.add_heading('8. 참고자료 (References)', 1)
        ref_heading.runs[0].font.color.rgb = PACIFIC_BLUE

        # Format reference section
        ref_section = tracker.format_section()
        for line in ref_section.split('\n'):
            if line.strip():
                if line.startswith('8.'):
                    doc.add_heading(line, 2)
                else:
                    doc.add_paragraph(line)

        # ===== Footer =====
        doc.add_paragraph()
        footer = doc.add_paragraph()
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer.add_run("© 2025 AMORE Pacific - Confidential")
        run.italic = True
        run.font.color.rgb = GRAY

        # Save to BytesIO
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        # Generate filename
        filename = f"AMORE_Analyst_Report_{request.start_date}_{request.end_date}.docx"

        # Cleanup temp charts
        if request.include_charts and chart_paths:
            try:
                for chart_path in chart_paths.values():
                    if chart_path.exists():
                        chart_path.unlink()
                if temp_dir:
                    Path(temp_dir).rmdir()
            except Exception as e:
                logger.warning(f"Chart cleanup failed: {e}")

        logger.info(f"Analyst report generated successfully: {filename}")

        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analyst report generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.post("/excel")
async def export_excel(request: Request):
    """
    엑셀 데이터 내보내기 (SQLite → Excel)
    """
    try:
        # Parse request body
        body = await request.json()
        start_date = body.get("start_date")
        end_date = body.get("end_date")
        include_metrics = body.get("include_metrics", True)

        # SQLite storage 사용
        storage = get_sqlite_storage()
        await storage.initialize()

        # 출력 경로
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./data/exports/AMORE_Data_{timestamp}.xlsx"

        # 엑셀 생성
        result = storage.export_to_excel(
            output_path=output_path,
            start_date=start_date,
            end_date=end_date,
            include_metrics=include_metrics
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Export failed"))

        file_path = Path(result["file_path"])
        if not file_path.exists():
            raise HTTPException(status_code=500, detail="Generated file not found")

        return FileResponse(
            path=str(file_path),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={file_path.name}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Excel export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
