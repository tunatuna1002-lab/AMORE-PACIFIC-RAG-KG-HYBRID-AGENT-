"""
ExportHandlers 단위 테스트

테스트 대상: src/tools/exporters/export_handlers.py
Coverage target: 60%+
"""

import os
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

        assert mock_queue.register_handler.call_count == 3
        registered_types = [call.args[0] for call in mock_queue.register_handler.call_args_list]
        assert "export_docx" in registered_types
        assert "export_analyst_report" in registered_types
        assert "export_excel" in registered_types


class TestHandleExportDocx:
    """handle_export_docx 핸들러 테스트"""

    @pytest.mark.asyncio
    async def test_export_docx_basic(self, tmp_path):
        """기본 DOCX 리포트 생성"""
        from src.tools.exporters.export_handlers import handle_export_docx

        mock_queue = MagicMock()
        mock_queue.update_progress = AsyncMock()
        mock_queue.output_dir = str(tmp_path)

        mock_data = {
            "home": {"insight_message": "Test insight message"},
            "brand": {
                "kpis": {
                    "SoS": {"value": "12.5%", "change": "+1.2%"},
                    "HHI": {"value": "0.15", "change": "-0.02"},
                }
            },
        }

        with patch(
            "src.api.dependencies.load_dashboard_data",
            return_value=mock_data,
        ):
            result = await handle_export_docx("job-001", {}, mock_queue)

        assert result.endswith(".docx")
        assert os.path.exists(result)
        assert "AMORE_Insight_Report_" in result
        # progress가 여러번 업데이트되어야 함
        assert mock_queue.update_progress.call_count >= 4

    @pytest.mark.asyncio
    async def test_export_docx_empty_data(self, tmp_path):
        """데이터가 없는 경우 기본 문구 사용"""
        from src.tools.exporters.export_handlers import handle_export_docx

        mock_queue = MagicMock()
        mock_queue.update_progress = AsyncMock()
        mock_queue.output_dir = str(tmp_path)

        with patch(
            "src.api.dependencies.load_dashboard_data",
            return_value={},
        ):
            result = await handle_export_docx("job-002", {}, mock_queue)

        assert result.endswith(".docx")
        assert os.path.exists(result)

    @pytest.mark.asyncio
    async def test_export_docx_no_insight_message(self, tmp_path):
        """insight_message가 없는 경우"""
        from src.tools.exporters.export_handlers import handle_export_docx

        mock_queue = MagicMock()
        mock_queue.update_progress = AsyncMock()
        mock_queue.output_dir = str(tmp_path)

        mock_data = {"home": {}, "brand": {"kpis": {}}}

        with patch(
            "src.api.dependencies.load_dashboard_data",
            return_value=mock_data,
        ):
            result = await handle_export_docx("job-003", {}, mock_queue)

        assert os.path.exists(result)

    @pytest.mark.asyncio
    async def test_export_docx_with_kpis(self, tmp_path):
        """KPI 데이터가 포함된 경우 테이블 생성"""
        from src.tools.exporters.export_handlers import handle_export_docx

        mock_queue = MagicMock()
        mock_queue.update_progress = AsyncMock()
        mock_queue.output_dir = str(tmp_path)

        mock_data = {
            "home": {"insight_message": "Some insight"},
            "brand": {
                "kpis": {
                    "SoS": {"value": "12.5%", "change": "+1.2%"},
                }
            },
        }

        with patch(
            "src.api.dependencies.load_dashboard_data",
            return_value=mock_data,
        ):
            result = await handle_export_docx("job-004", {}, mock_queue)

        assert os.path.exists(result)


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

        with patch("src.tools.exporters.export_handlers.PeriodAnalyzer") as mock_analyzer_cls:
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

        with patch("src.tools.exporters.export_handlers.PeriodAnalyzer") as mock_analyzer_cls:
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
