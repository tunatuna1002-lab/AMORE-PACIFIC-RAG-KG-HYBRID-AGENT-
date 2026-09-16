"""
Export Routes - Document and data export endpoints

라우트는 검증 + 서비스 호출 + 응답 변환만 담당한다 (F6):
- 수집/집계: ``src/application/services/export_service.py``
- 문서 렌더링: ``src/tools/exporters/export_handlers.py`` (동기/비동기 공용)
"""

import logging
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from src.api.dashboard_shape import (
    ai_insights_from,
    brand_metrics_from,
    category_metrics_from,
    products_as_list,
    summary_from,
)
from src.api.dependencies import (
    get_data_service,
    limiter,
    load_dashboard_data,
    verify_api_key,
)
from src.api.models import AnalystReportRequest, AsyncExportRequest, ExportRequest
from src.application.services.date_range import days_in_range
from src.application.services.export_service import (
    NoDataForPeriod,
    build_analyst_report_context,
)
from src.application.services.external_signals_service import (
    get_external_signals,
    signal_source_status,
)
from src.domain.brand import is_target_brand
from src.tools.exporters.export_handlers import (
    InsightReportView,
    render_analyst_report,
    render_insight_report,
)
from src.tools.storage.sqlite_storage import get_sqlite_storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/export", tags=["export"])

DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _streamed(doc, filename: str) -> StreamingResponse:
    """python-docx Document -> 첨부파일 스트리밍 응답"""
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type=DOCX_MEDIA_TYPE,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/docx", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def export_docx(request: Request, payload: ExportRequest):
    """
    인사이트 리포트 DOCX 생성 및 다운로드
    """
    data = load_dashboard_data()
    if not data:
        raise HTTPException(status_code=404, detail="Dashboard data not found")

    # 외부 신호 (선택): 분석 기간만큼 수집해 보고서 섹션 문자열로
    signals_section = ""
    days = 7
    if payload.include_external_signals:
        days = days_in_range(payload.start_date, payload.end_date, fallback=7)
        signals_result = await get_external_signals(
            days=days, start_date=payload.start_date, end_date=payload.end_date
        )
        signals_section = signals_result.get("report_section", "")

    view = InsightReportView(
        metadata=data.get("metadata", {}),
        summary=summary_from(data),
        brands=brand_metrics_from(data),
        categories=category_metrics_from(data),
        products=[p for p in products_as_list(data) if is_target_brand(p.get("brand"))],
        strategic_insights=ai_insights_from(data).get("strategic_insights", []),
        include_strategy=payload.include_strategy,
        include_external_signals=payload.include_external_signals,
        signals_section=signals_section,
        signals_days=days,
    )

    return _streamed(render_insight_report(view), view.filename)


@router.post("/analyst-report", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def export_analyst_report(request: Request, payload: AnalystReportRequest):
    """
    기간별 애널리스트 리포트 DOCX 생성 (8 Sections)

    Args:
        payload: AnalystReportRequest with start_date, end_date, options

    Returns:
        StreamingResponse with DOCX file
    """
    logger.info(f"Generating analyst report: {payload.start_date} ~ {payload.end_date}")

    try:
        ctx = await build_analyst_report_context(
            start_date=payload.start_date,
            end_date=payload.end_date,
            include_charts=payload.include_charts,
            include_external_signals=payload.include_external_signals,
        )
    except NoDataForPeriod as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Analyst report generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}") from e

    try:
        doc = render_analyst_report(ctx, include_charts=payload.include_charts)
        response = _streamed(doc, ctx.filename)
    except Exception as e:
        logger.error(f"Analyst report generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}") from e
    finally:
        ctx.cleanup_charts()

    logger.info(f"Analyst report generated successfully: {ctx.filename}")
    return response


@router.post("/excel", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
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
        output_path = str(get_data_service().path_for("exports", f"AMORE_Data_{timestamp}.xlsx"))

        # 엑셀 생성
        result = storage.export_to_excel(
            output_path=output_path,
            start_date=start_date,
            end_date=end_date,
            include_metrics=include_metrics,
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Export failed"))

        file_path = Path(result["file_path"])
        if not file_path.exists():
            raise HTTPException(status_code=500, detail="Generated file not found")

        return FileResponse(
            path=str(file_path),
            media_type=XLSX_MEDIA_TYPE,
            headers={"Content-Disposition": f"attachment; filename={file_path.name}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Excel export error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/signals/status")
@limiter.limit("5/minute")
async def get_signal_status(request: Request):
    """
    외부 신호 API 상태 확인

    Returns:
        각 외부 데이터 소스의 설정/사용 가능 상태
    """
    return signal_source_status()


# ============================================================
# 비동기 작업 API (페이지 새로고침에도 다운로드 지속)
# ============================================================


@router.post("/async/start", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def start_async_export(request: Request, payload: AsyncExportRequest):
    """
    비동기 내보내기 작업 시작

    페이지 새로고침에도 다운로드가 지속됩니다.
    작업 ID를 반환하며, /async/status/{job_id}로 진행 상태 확인 가능.

    Args:
        payload: AsyncExportRequest

    Returns:
        {
            "job_id": "abc12345",
            "status": "pending",
            "message": "작업이 큐에 추가되었습니다."
        }
    """
    from src.tools.utilities.job_queue import get_job_queue

    queue = get_job_queue()
    await queue.initialize()

    # 파라미터 준비
    params = {
        "start_date": payload.start_date,
        "end_date": payload.end_date,
        "include_charts": payload.include_charts,
        "include_external_signals": payload.include_external_signals,
        "include_metrics": payload.include_metrics,
    }

    # 작업 생성
    job_id = await queue.create_job(payload.job_type, params)

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "작업이 큐에 추가되었습니다. 진행 상태는 /api/export/async/status/{job_id}에서 확인하세요.",
    }


@router.get("/async/status/{job_id}")
@limiter.limit("5/minute")
async def get_async_export_status(request: Request, job_id: str):
    """
    비동기 내보내기 작업 상태 조회

    Args:
        job_id: 작업 ID

    Returns:
        {
            "id": "abc12345",
            "status": "running" | "completed" | "failed" | "pending",
            "progress": 50,
            "progress_message": "차트 생성 중...",
            "download_url": "/api/export/download/abc12345" (완료 시)
        }
    """
    from src.tools.utilities.job_queue import get_job_queue

    queue = get_job_queue()
    status = await queue.get_job_status(job_id)

    if not status:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return status


@router.get("/download/{job_id}")
@limiter.limit("5/minute")
async def download_export_file(request: Request, job_id: str):
    """
    완료된 내보내기 파일 다운로드

    Args:
        job_id: 작업 ID

    Returns:
        FileResponse with the generated file
    """
    from src.tools.utilities.job_queue import JobStatus, get_job_queue

    queue = get_job_queue()
    status = await queue.get_job_status(job_id)

    if not status:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if status["status"] != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400, detail=f"Job not completed yet. Current status: {status['status']}"
        )

    file_path = status.get("result_file")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found or expired")

    filename = os.path.basename(file_path)

    # MIME type 결정
    if filename.endswith(".docx"):
        media_type = DOCX_MEDIA_TYPE
    elif filename.endswith(".xlsx"):
        media_type = XLSX_MEDIA_TYPE
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/async/jobs")
@limiter.limit("5/minute")
async def list_export_jobs(request: Request, status: str | None = None, limit: int = 20):
    """
    내보내기 작업 목록 조회

    Args:
        status: 필터링할 상태 (pending, running, completed, failed)
        limit: 최대 개수

    Returns:
        작업 목록
    """
    from src.tools.utilities.job_queue import get_job_queue

    queue = get_job_queue()
    jobs = await queue.get_all_jobs(status=status, limit=limit)

    return {"jobs": jobs, "total": len(jobs)}
