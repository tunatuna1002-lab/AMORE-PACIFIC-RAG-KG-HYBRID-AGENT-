"""
Export Routes - Document and data export endpoints
"""
import logging
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
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

router = APIRouter(prefix="/api/export", tags=["export"])


class ExportRequest(BaseModel):
    """내보내기 요청"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    include_strategy: bool = True
    include_external_signals: bool = True  # External Signal 포함 여부


async def _get_external_signals(days: int = 7) -> str:
    """External Signal 수집 및 보고서 섹션 생성"""
    try:
        collector = ExternalSignalCollector()
        await collector.initialize()

        # 기존 신호가 있으면 반환
        if collector.signals:
            return collector.generate_report_section(days=days)

        # 신호가 없으면 빈 문자열
        return ""
    except Exception as e:
        logging.warning(f"External signal collection failed: {e}")
        return ""
    finally:
        await collector.close()


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

        external_signals = await _get_external_signals(days=days)

        if external_signals:
            doc.add_paragraph(f"분석 기간: 최근 {days}일")
            doc.add_paragraph()

            # 신호 섹션별로 파싱하여 추가
            for line in external_signals.split("\n"):
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
