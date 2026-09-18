"""
Phase 3 배선 테스트 (§3.1 crawl_failed / §3.2 신선도 / §3.3 무결성 / §3.4 발화 위치)

이 파일이 검증하는 것은 "구현이 존재하는가"가 아니라 "실제로 호출되는가"다.
Phase 3 이전에는 모두 구현만 있고 호출처가 0건이었다.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.alert_manager import AlertManager
from src.core.brain import UnifiedBrain

# =============================================================================
# §3.1 crawl_failed → 알림 발송
# =============================================================================


class TestCrawlFailedAlert:
    @pytest.mark.asyncio
    async def test_check_conditions_produces_critical_alert(self):
        manager = AlertManager()
        alerts = await manager.check_conditions("crawl_failed", {"error": "Connection timeout"})

        assert len(alerts) == 1
        assert alerts[0]["type"] == "crawl_failed"
        assert alerts[0]["severity"] == "critical"
        assert "Connection timeout" in alerts[0]["message"]

    @pytest.mark.asyncio
    async def test_includes_category_when_given(self):
        manager = AlertManager()
        alerts = await manager.check_conditions(
            "crawl_failed", {"error": "blocked", "category": "lip_care"}
        )
        assert "lip_care" in alerts[0]["message"]

    @pytest.mark.asyncio
    async def test_unknown_error_fallback(self):
        manager = AlertManager()
        alerts = await manager.check_conditions("crawl_failed", {})
        assert "Unknown error" in alerts[0]["message"]

    @pytest.mark.asyncio
    async def test_emit_event_routes_to_alert_manager(self):
        """brain.emit_event('crawl_failed')가 알림 처리까지 도달한다"""
        brain = UnifiedBrain()
        manager = MagicMock()
        manager.check_conditions = AsyncMock(
            return_value=[{"type": "crawl_failed", "severity": "critical", "message": "x"}]
        )
        manager.process_alert = AsyncMock(return_value=True)
        brain._alert_manager = manager  # alert_manager는 lazy property

        await brain.emit_event("crawl_failed", {"error": "boom"})

        manager.check_conditions.assert_awaited_once()
        assert manager.check_conditions.await_args.args[0] == "crawl_failed"
        manager.process_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_alert_maps_critical_priority(self):
        from src.agents.alert_agent import AlertPriority

        manager = AlertManager()
        agent = MagicMock()
        agent.create_alert = MagicMock()
        agent.send_pending_alerts = AsyncMock(return_value={"sent": 1})
        manager._alert_agent = agent
        manager._initialized = True

        sent = await manager.process_alert(
            {"type": "crawl_failed", "message": "크롤링 실패: timeout"}
        )

        assert sent is True
        assert agent.create_alert.call_args.kwargs["priority"] == AlertPriority.CRITICAL

    def test_alert_agent_handlers_removed(self):
        """D2: 미배선 핸들러 3종은 삭제됐다"""
        from src.agents.alert_agent import AlertAgent

        for name in ("on_crawl_complete", "on_crawl_failed", "on_error"):
            assert not hasattr(AlertAgent, name), f"{name}이 되살아났습니다"

    def test_alert_protocol_handlers_removed(self):
        from src.domain.interfaces.alert import AlertAgentProtocol

        for name in ("on_crawl_complete", "on_crawl_failed", "on_error"):
            assert not hasattr(AlertAgentProtocol, name)


# =============================================================================
# §3.2 / §3.4 워크플로우 완료 지점 배선
# =============================================================================


class TestWorkflowNotifications:
    def _workflow(self):
        from src.application.workflows.batch_workflow import BatchWorkflow

        wf = BatchWorkflow.__new__(BatchWorkflow)
        wf.logger = MagicMock()
        return wf

    @pytest.mark.asyncio
    async def test_complete_marks_crawled_and_emits(self):
        wf = self._workflow()
        brain = MagicMock()
        brain.state = MagicMock()
        brain.emit_event = AsyncMock()

        with patch("src.core.brain.get_brain", new=AsyncMock(return_value=brain)):
            await wf._notify_workflow_complete(
                {
                    "status": "completed",
                    "summary": {
                        "products_crawled": 492,
                        "laneige_tracked": 5,
                        "categories": ["lip_care"],
                    },
                }
            )

        # §3.2: data_freshness가 "fresh"로 기록된다 (상시 unknown이던 문제)
        brain.state.mark_crawled.assert_called_once_with(products_count=492)

        # §3.4/D4: crawl_complete가 워크플로우 완료 지점에서 발화된다
        brain.emit_event.assert_awaited_once()
        event_name, payload = brain.emit_event.await_args.args
        assert event_name == "crawl_complete"
        assert payload["result"]["success"] is True
        assert payload["total_products"] == 492

    @pytest.mark.asyncio
    async def test_failure_marks_stale_and_emits_crawl_failed(self):
        wf = self._workflow()
        brain = MagicMock()
        brain.state = MagicMock()
        brain.emit_event = AsyncMock()

        with patch("src.core.brain.get_brain", new=AsyncMock(return_value=brain)):
            await wf._notify_workflow_failed("스크래퍼 예외")

        brain.state.mark_data_stale.assert_called_once()
        event_name, payload = brain.emit_event.await_args.args
        assert event_name == "crawl_failed"
        assert payload["error"] == "스크래퍼 예외"

    @pytest.mark.asyncio
    async def test_notification_failure_does_not_raise(self):
        """알림 실패가 워크플로우 결과를 깨뜨리지 않는다"""
        wf = self._workflow()
        with patch("src.core.brain.get_brain", new=AsyncMock(side_effect=RuntimeError("no brain"))):
            await wf._notify_workflow_complete({"status": "completed", "summary": {}})
            await wf._notify_workflow_failed("err")


class TestStateOwnership:
    """§3.2: 크롤 상태 정본은 OrchestratorState 하나뿐"""

    def test_state_manager_no_longer_duplicates_crawl_state(self):
        from src.core.state_manager import StateManager

        for name in ("mark_crawled", "mark_data_stale", "is_crawl_needed", "get_data_age_hours"):
            assert not hasattr(StateManager, name), f"StateManager.{name} 중복이 남아있습니다"

    def test_orchestrator_state_owns_crawl_state(self, tmp_path):
        from src.core.state import OrchestratorState

        state = OrchestratorState(_persist_path=tmp_path / "s.json")
        assert state.data_freshness == "unknown"
        state.mark_crawled(products_count=100)
        assert state.data_freshness == "fresh"
        state.mark_data_stale()
        assert state.data_freshness == "stale"


# =============================================================================
# §3.3 무결성 검사 배선
# =============================================================================


class TestIntegrityCheckWiring:
    def test_scheduler_registers_daily_task(self):
        from src.core.scheduler import AutonomousScheduler

        scheduler = AutonomousScheduler()
        actions = {s["action"] for s in scheduler.schedules}
        assert "check_integrity" in actions

    @pytest.mark.asyncio
    async def test_ok_severity_does_not_alert(self):
        brain = UnifiedBrain()
        brain._process_alert = AsyncMock()

        with patch(
            "src.tools.utilities.data_integrity_checker.check_data_integrity",
            new=AsyncMock(return_value={"severity": "OK"}),
        ):
            result = await brain._execute_scheduled_task(
                {"name": "정합성", "action": "check_integrity"}
            )

        assert result["severity"] == "OK"
        brain._process_alert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_critical_severity_raises_alert(self):
        brain = UnifiedBrain()
        brain._process_alert = AsyncMock()

        with patch(
            "src.tools.utilities.data_integrity_checker.check_data_integrity",
            new=AsyncMock(
                return_value={
                    "severity": "CRITICAL",
                    "missing_dates": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31"],
                    "sync_status": {"gap": 900},
                    "recommendations": ["sync_sheets_to_sqlite.py 실행"],
                }
            ),
        ):
            result = await brain._execute_scheduled_task(
                {"name": "정합성", "action": "check_integrity"}
            )

        assert result["severity"] == "CRITICAL"
        brain._process_alert.assert_awaited_once()
        alert = brain._process_alert.await_args.args[0]
        assert alert["type"] == "data_integrity"
        assert alert["severity"] == "critical"
        assert "900" in alert["message"]

    def test_route_is_registered(self):
        from src.api.app_factory import create_app
        from tests.unit.api.route_utils import collect_app_paths

        paths = collect_app_paths(create_app())
        assert "/api/health/integrity" in paths
