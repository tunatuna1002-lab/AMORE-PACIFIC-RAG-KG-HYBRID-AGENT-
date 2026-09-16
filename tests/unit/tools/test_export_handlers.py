"""
ExportHandlers 단위 테스트

테스트 대상: src/tools/exporters/export_handlers.py
Coverage target: 60%+
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src.api.dependencies can be imported (needs API_KEY env)
with patch.dict(
    "os.environ",
    {"API_KEY": "test-key-for-export"},  # pragma: allowlist secret
    clear=False,
):
    import src.api.dependencies  # noqa: F401


class TestRegisterAllHandlers:
    """register_all_handlers 함수 테스트"""

    def test_register_all_handlers(self):
        """모든 핸들러가 등록되어야 함"""
        from src.tools.exporters.export_handlers import register_all_handlers

        mock_queue = MagicMock()
        register_all_handlers(mock_queue)

        # CHANGED (F6): 핸들러 3개 -> 2개. "simple docx" 비동기 핸들러
        # (handle_export_docx)는 brand.kpis를 {"value","change"} dict로 가정해
        # exporter가 쓰는 스칼라 값에서는 항상 AttributeError로 실패했고,
        # 대시보드는 export_docx 작업을 만들지 않는다. 인사이트 DOCX는 이제
        # render_insight_report 한 곳에서만 생성된다(동기 /api/export/docx).
        assert mock_queue.register_handler.call_count == 2
        registered_types = [call.args[0] for call in mock_queue.register_handler.call_args_list]
        assert "export_docx" not in registered_types
        assert "export_analyst_report" in registered_types
        assert "export_excel" in registered_types


class TestRenderInsightReport:
    """
    render_insight_report - 인사이트 리포트 DOCX의 단일 렌더러.

    CHANGED (F6): 이 클래스는 삭제된 handle_export_docx("simple docx") 테스트를
    대체한다. 그 핸들러는 kpis를 dict-of-dict로 가정해 실제 대시보드 데이터에서는
    항상 실패했다. 문서 생성은 이제 동기 라우트와 공유하는 이 렌더러가 담당한다.
    """

    def test_renders_a_document_from_a_populated_view(self, tmp_path):
        from src.tools.exporters.export_handlers import (
            InsightReportView,
            render_insight_report,
        )

        view = InsightReportView(
            metadata={"data_date": "2026-09-01"},
            summary={
                "total_products": 2,
                "categories_count": 1,
                "laneige_products": 1,
                "avg_price": 16.5,
            },
            brands=[{"brand": "LANEIGE", "product_count": 1, "avg_rank": 1.0, "sos": 50.0}],
            categories=[
                {
                    "category": "Lip Care",
                    "total_products": 2,
                    "laneige_products": 1,
                    "avg_price": 16.5,
                    "hhi": 5000.0,
                    "cpi": 150.0,
                }
            ],
            products=[
                {
                    "rank": 1,
                    "title": "LANEIGE Lip Sleeping Mask",
                    "category": "lip_care",
                    "price": 24.0,
                    "rating": 4.6,
                }
            ],
            strategic_insights=[{"title": "Insight", "content": "본문"}],
        )

        doc = render_insight_report(view)
        text = "\n".join(p.text for p in doc.paragraphs)
        for table in doc.tables:
            for row in table.rows:
                text += "\n" + "\n".join(cell.text for cell in row.cells)

        assert "AMORE INSIGHT Report" in text
        assert "LANEIGE Lip Sleeping Mask" in text
        assert "Lip Care" in text
        assert "브랜드 데이터가 없습니다." not in text
        assert view.filename.startswith("AMORE_Insight_Report_")
        assert view.filename.endswith(".docx")

        output = tmp_path / view.filename
        doc.save(str(output))
        assert output.exists()

    def test_renders_placeholders_for_an_empty_view(self):
        from src.tools.exporters.export_handlers import (
            InsightReportView,
            render_insight_report,
        )

        doc = render_insight_report(InsightReportView())
        text = "\n".join(p.text for p in doc.paragraphs)

        assert "브랜드 데이터가 없습니다." in text
        assert "카테고리 데이터가 없습니다." in text
        assert "LANEIGE 제품이 없습니다." in text

    def test_external_signal_section_is_rendered_when_requested(self):
        from src.tools.exporters.export_handlers import (
            InsightReportView,
            render_insight_report,
        )

        view = InsightReportView(
            include_external_signals=True,
            signals_section="■ 뉴스\n• LANEIGE 신제품 출시",
            signals_days=14,
        )
        doc = render_insight_report(view)
        text = "\n".join(p.text for p in doc.paragraphs)

        assert "분석 기간: 최근 14일" in text
        assert "• LANEIGE 신제품 출시" in text

        empty = render_insight_report(InsightReportView(include_external_signals=True))
        empty_text = "\n".join(p.text for p in empty.paragraphs)
        assert "수집된 외부 트렌드 신호가 없습니다." in empty_text


class TestHandleExportAnalystReport:
    """handle_export_analyst_report 핸들러 테스트"""

    @pytest.mark.asyncio
    async def test_missing_dates_raises_error(self):
        """start_date/end_date 없으면 에러"""
        from src.tools.exporters.export_handlers import handle_export_analyst_report

        mock_queue = MagicMock()
        mock_queue.update_progress = AsyncMock()

        with pytest.raises(ValueError, match="start_date and end_date are required"):
            await handle_export_analyst_report("job-010", {}, mock_queue)

    @pytest.mark.asyncio
    async def test_missing_start_date_raises_error(self):
        """start_date만 없으면 에러"""
        from src.tools.exporters.export_handlers import handle_export_analyst_report

        mock_queue = MagicMock()
        mock_queue.update_progress = AsyncMock()

        with pytest.raises(ValueError, match="start_date and end_date are required"):
            await handle_export_analyst_report("job-011", {"end_date": "2026-01-15"}, mock_queue)

    @pytest.mark.asyncio
    async def test_zero_days_analysis_raises_error(self, tmp_path):
        """분석 기간이 0일이면 에러"""
        from src.tools.exporters.export_handlers import handle_export_analyst_report

        mock_queue = MagicMock()
        mock_queue.update_progress = AsyncMock()
        mock_queue.output_dir = str(tmp_path)

        mock_analysis = MagicMock()
        mock_analysis.total_days = 0

        # CHANGED (F6): PeriodAnalyzer는 export_handlers가 아니라
        # application/services/export_service.py가 (지연) import 한다.
        with patch("src.tools.calculators.period_analyzer.PeriodAnalyzer") as mock_analyzer_cls:
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze = AsyncMock(return_value=mock_analysis)
            mock_analyzer_cls.return_value = mock_analyzer

            with pytest.raises(ValueError, match="No data found"):
                await handle_export_analyst_report(
                    "job-012",
                    {"start_date": "2026-01-01", "end_date": "2026-01-15"},
                    mock_queue,
                )

    @pytest.mark.asyncio
    async def test_analyst_report_params_parsing(self):
        """파라미터 파싱 테스트"""
        from src.tools.exporters.export_handlers import handle_export_analyst_report

        mock_queue = MagicMock()
        mock_queue.update_progress = AsyncMock()

        params = {
            "start_date": "2026-01-01",
            "end_date": "2026-01-15",
            "include_charts": False,
            "include_external_signals": False,
        }

        mock_analysis = MagicMock()
        mock_analysis.total_days = 0

        # CHANGED (F6): PeriodAnalyzer는 export_handlers가 아니라
        # application/services/export_service.py가 (지연) import 한다.
        with patch("src.tools.calculators.period_analyzer.PeriodAnalyzer") as mock_analyzer_cls:
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze = AsyncMock(return_value=mock_analysis)
            mock_analyzer_cls.return_value = mock_analyzer

            with pytest.raises(ValueError, match="No data found"):
                await handle_export_analyst_report("job-013", params, mock_queue)


class TestHandleExportExcel:
    """handle_export_excel 핸들러 테스트"""

    @pytest.mark.asyncio
    async def test_export_excel_basic(self, tmp_path):
        """기본 Excel 내보내기"""
        from src.tools.exporters.export_handlers import handle_export_excel

        mock_queue = MagicMock()
        mock_queue.update_progress = AsyncMock()
        mock_queue.output_dir = str(tmp_path)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.export_to_excel = MagicMock()

        with patch(
            "src.tools.exporters.export_handlers.get_sqlite_storage",
            return_value=mock_storage,
        ):
            result = await handle_export_excel("job-020", {}, mock_queue)

        assert result.endswith(".xlsx")
        assert "AMORE_Data_" in result
        mock_storage.initialize.assert_called_once()
        mock_storage.export_to_excel.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_excel_with_params(self, tmp_path):
        """날짜 범위 및 메트릭 포함 파라미터"""
        from src.tools.exporters.export_handlers import handle_export_excel

        mock_queue = MagicMock()
        mock_queue.update_progress = AsyncMock()
        mock_queue.output_dir = str(tmp_path)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.export_to_excel = MagicMock()

        params = {
            "start_date": "2026-01-01",
            "end_date": "2026-01-15",
            "include_metrics": True,
        }

        with patch(
            "src.tools.exporters.export_handlers.get_sqlite_storage",
            return_value=mock_storage,
        ):
            result = await handle_export_excel("job-021", params, mock_queue)

        call_kwargs = mock_storage.export_to_excel.call_args.kwargs
        assert call_kwargs["start_date"] == "2026-01-01"
        assert call_kwargs["end_date"] == "2026-01-15"
        assert call_kwargs["include_metrics"] is True

    @pytest.mark.asyncio
    async def test_export_excel_default_include_metrics(self, tmp_path):
        """include_metrics 기본값은 True"""
        from src.tools.exporters.export_handlers import handle_export_excel

        mock_queue = MagicMock()
        mock_queue.update_progress = AsyncMock()
        mock_queue.output_dir = str(tmp_path)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.export_to_excel = MagicMock()

        with patch(
            "src.tools.exporters.export_handlers.get_sqlite_storage",
            return_value=mock_storage,
        ):
            await handle_export_excel("job-022", {}, mock_queue)

        call_kwargs = mock_storage.export_to_excel.call_args.kwargs
        assert call_kwargs["include_metrics"] is True

    @pytest.mark.asyncio
    async def test_export_excel_progress_updates(self, tmp_path):
        """진행률 업데이트 호출 확인"""
        from src.tools.exporters.export_handlers import handle_export_excel

        mock_queue = MagicMock()
        mock_queue.update_progress = AsyncMock()
        mock_queue.output_dir = str(tmp_path)

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.export_to_excel = MagicMock()

        with patch(
            "src.tools.exporters.export_handlers.get_sqlite_storage",
            return_value=mock_storage,
        ):
            await handle_export_excel("job-023", {}, mock_queue)

        # 적어도 3번의 progress 업데이트가 있어야 함
        assert mock_queue.update_progress.call_count >= 3


class TestModuleConstants:
    """모듈 수준 상수 테스트"""

    def test_pacific_blue_color(self):
        """PACIFIC_BLUE 색상 상수"""
        from src.tools.exporters.export_handlers import PACIFIC_BLUE

        assert PACIFIC_BLUE is not None

    def test_amore_blue_color(self):
        """AMORE_BLUE 색상 상수"""
        from src.tools.exporters.export_handlers import AMORE_BLUE

        assert AMORE_BLUE is not None

    def test_gray_color(self):
        """GRAY 색상 상수"""
        from src.tools.exporters.export_handlers import GRAY

        assert GRAY is not None


class TestRenderAnalystReport:
    """
    render_analyst_report - 애널리스트 리포트의 단일 렌더러.

    CHANGED (F6): 동기 ``POST /api/export/analyst-report``와 비동기
    ``export_analyst_report`` 작업이 서로 다른 문서를 만들던 중복을 없앴다.
    남은 경로는 비동기 쪽 레이아웃(DocxReportGenerator IR 스타일)이며,
    동기 경로에만 있던 본문 참고자료 라인 필터링을 흡수했다.
    """

    @staticmethod
    def _context():
        from types import SimpleNamespace

        from src.application.services.export_service import AnalystReportContext

        def section(section_id, title, content):
            return SimpleNamespace(section_id=section_id, section_title=title, content=content)

        report = SimpleNamespace(
            executive_summary=section(
                1,
                "Executive Summary",
                "■ 요약\n• LANEIGE는 Lip Care 1위\n참고자료:\n[1] https://example.com/news",
            ),
            laneige_analysis=section(2, "LANEIGE 심층 분석", "■ 순위\n• 1위 유지"),
            competitive_analysis=None,
            market_trends=None,
            external_signals=None,
            risks_opportunities=None,
            strategic_recommendations=section(7, "전략 제언", "• 리뷰 관리"),
        )

        class _Tracker:
            def get_formatted_references(self, source_type: str) -> str:
                return "[1] Amazon Best Sellers" if source_type == "data" else ""

        return AnalystReportContext(
            start_date="2026-09-01",
            end_date="2026-09-07",
            analysis=SimpleNamespace(total_days=7),
            report=report,
            tracker=_Tracker(),
        )

    def test_renders_the_eight_section_document(self):
        from src.tools.exporters.export_handlers import render_analyst_report

        ctx = self._context()
        doc = render_analyst_report(ctx, include_charts=False)
        text = "\n".join(p.text for p in doc.paragraphs)

        # 표지 제목은 두 줄로 나뉘어 렌더링된다
        assert "LANEIGE Amazon US" in text
        assert "경쟁력 분석 보고서" in text
        assert "1. Executive Summary" in text  # 목차 + 섹션 제목
        assert "LANEIGE는 Lip Care 1위" in text
        assert "8.1 외부 자료" in text
        assert "외부 자료 없음" in text
        assert "8.2 데이터 소스" in text
        assert "[1] Amazon Best Sellers" in text
        assert ctx.filename == "AMORE_Analyst_Report_2026-09-01_2026-09-07.docx"

    def test_body_reference_lines_are_filtered_out(self):
        from src.tools.exporters.export_handlers import render_analyst_report

        doc = render_analyst_report(self._context(), include_charts=False)
        text = "\n".join(p.text for p in doc.paragraphs)

        # 참고자료는 8장에만 남는다 (본문의 "참고자료:" 블록은 제거)
        assert "https://example.com/news" not in text
