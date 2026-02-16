"""
AlertManager 단위 테스트
============================
src/core/alert_manager.py 커버리지 11.64% → 50%+ 목표

테스트 대상:
- AlertManager 초기화 (with/without dependencies)
- initialize() 성공 및 실패
- check_conditions() - event routing (rank_changed, crawl_complete, metrics_calculated)
- _check_rank_change() - 순위 변동 임계값 체크
- _check_crawl_complete() - 크롤링 성공/실패 감지
- _check_metrics_calculated() - placeholder 반환
- check_metrics_alerts() - rank_delta/sos_delta 파싱 및 알림 생성
- process_alert() - 알림 생성, 발송, 히스토리 관리, lazy init
- get_recent_alerts() - 히스토리 조회
- get_stats() - 통계 반환
- History pruning (>100 items)
- Exception handling
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.core.alert_manager import AlertManager

# =========================================================================
# Fixtures
# =========================================================================


@pytest_asyncio.fixture
async def mock_state_manager():
    """Mock StateManager"""
    mock = MagicMock()
    return mock


@pytest_asyncio.fixture
async def mock_alert_agent():
    """Mock AlertAgent with async methods"""
    mock = MagicMock()
    mock.create_alert = MagicMock(return_value=MagicMock())
    mock.send_pending_alerts = AsyncMock(return_value={"sent": 1, "failed": 0})
    return mock


@pytest_asyncio.fixture
async def manager(mock_state_manager, mock_alert_agent):
    """AlertManager with mocked dependencies"""
    mgr = AlertManager(
        state_manager=mock_state_manager,
        alert_agent=mock_alert_agent,
    )
    mgr._initialized = True  # Skip lazy init
    return mgr


@pytest_asyncio.fixture
async def manager_no_deps():
    """AlertManager without dependencies (for lazy init testing)"""
    return AlertManager()


# =========================================================================
# 초기화
# =========================================================================


class TestAlertManagerInit:
    def test_defaults(self):
        mgr = AlertManager()
        assert mgr._state_manager is None
        assert mgr._alert_agent is None
        assert mgr._initialized is False
        assert mgr._alert_history == []
        assert mgr._stats == {"alerts_generated": 0, "alerts_sent": 0, "send_failures": 0}

    def test_with_dependencies(self, mock_state_manager, mock_alert_agent):
        mgr = AlertManager(
            state_manager=mock_state_manager,
            alert_agent=mock_alert_agent,
        )
        assert mgr._state_manager is mock_state_manager
        assert mgr._alert_agent is mock_alert_agent
        assert mgr._initialized is False

    def test_constants(self):
        assert AlertManager.RANK_CHANGE_THRESHOLD == 10
        assert AlertManager.SOS_CHANGE_THRESHOLD == 2.0


# =========================================================================
# initialize()
# =========================================================================


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_success_with_provided_deps(
        self, mock_state_manager, mock_alert_agent
    ):
        mgr = AlertManager(state_manager=mock_state_manager, alert_agent=mock_alert_agent)
        await mgr.initialize()
        assert mgr._initialized is True
        assert mgr._state_manager is mock_state_manager
        assert mgr._alert_agent is mock_alert_agent

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, manager):
        assert manager._initialized is True
        await manager.initialize()
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_lazy_import_state_manager(self):
        mgr = AlertManager(alert_agent=MagicMock())
        # Patch where the class is imported (in the initialize method)
        with patch("src.core.state_manager.StateManager") as MockStateManager:
            mock_sm = MagicMock()
            MockStateManager.return_value = mock_sm
            # Also patch AlertAgent to prevent secondary import
            with patch("src.agents.alert_agent.AlertAgent"):
                await mgr.initialize()
                assert mgr._state_manager is mock_sm
                assert mgr._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_lazy_import_alert_agent(self, mock_state_manager):
        mgr = AlertManager(state_manager=mock_state_manager)
        with patch("src.agents.alert_agent.AlertAgent") as MockAlertAgent:
            mock_agent = MagicMock()
            MockAlertAgent.return_value = mock_agent
            await mgr.initialize()
            MockAlertAgent.assert_called_once_with(mock_state_manager)
            assert mgr._alert_agent is mock_agent
            assert mgr._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_exception_handling(self):
        mgr = AlertManager()
        # Trigger exception during StateManager import
        with patch("src.core.state_manager.StateManager", side_effect=Exception("Import failed")):
            await mgr.initialize()
            assert mgr._initialized is False


# =========================================================================
# check_conditions() - Event Routing
# =========================================================================


class TestCheckConditions:
    @pytest.mark.asyncio
    async def test_check_conditions_rank_changed(self, manager):
        data = {
            "product": {
                "name": "Lip Mask",
                "asin": "B001",
                "rank": 5,
            },
            "change": 15,
        }
        alerts = await manager.check_conditions("rank_changed", data)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "rank_drop"
        assert alerts[0]["change"] == 15

    @pytest.mark.asyncio
    async def test_check_conditions_crawl_complete(self, manager):
        data = {"result": {"success": False, "error": "Timeout"}}
        alerts = await manager.check_conditions("crawl_complete", data)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "crawl_failed"

    @pytest.mark.asyncio
    async def test_check_conditions_metrics_calculated(self, manager):
        data = {}
        alerts = await manager.check_conditions("metrics_calculated", data)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_check_conditions_unknown_event(self, manager):
        alerts = await manager.check_conditions("unknown_event", {})
        assert alerts == []


# =========================================================================
# _check_rank_change()
# =========================================================================


class TestCheckRankChange:
    def test_rank_drop_above_threshold(self, manager):
        data = {
            "product": {"name": "Lip Mask", "asin": "B001", "rank": 15},
            "change": 12,
        }
        alerts = manager._check_rank_change(data)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "rank_drop"
        assert alerts[0]["severity"] == "warning"
        assert alerts[0]["change"] == 12
        assert "급락" in alerts[0]["message"]
        assert alerts[0]["product"] == "Lip Mask"
        assert alerts[0]["asin"] == "B001"
        assert alerts[0]["current_rank"] == 15
        assert "timestamp" in alerts[0]

    def test_rank_surge_above_threshold(self, manager):
        data = {
            "product": {"name": "Lip Mask", "asin": "B001", "rank": 3},
            "change": -15,
        }
        alerts = manager._check_rank_change(data)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "rank_surge"
        assert alerts[0]["severity"] == "warning"
        assert alerts[0]["change"] == -15
        assert "급등" in alerts[0]["message"]

    def test_rank_drop_critical(self, manager):
        data = {
            "product": {"name": "Lip Mask", "asin": "B001", "rank": 30},
            "change": 25,
        }
        alerts = manager._check_rank_change(data)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"

    def test_rank_surge_critical(self, manager):
        data = {
            "product": {"name": "Lip Mask", "asin": "B001", "rank": 2},
            "change": -20,
        }
        alerts = manager._check_rank_change(data)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"
        assert alerts[0]["type"] == "rank_surge"

    def test_rank_change_below_threshold(self, manager):
        data = {
            "product": {"name": "Lip Mask", "asin": "B001", "rank": 10},
            "change": 5,
        }
        alerts = manager._check_rank_change(data)
        assert len(alerts) == 0

    def test_rank_change_exactly_threshold(self, manager):
        data = {
            "product": {"name": "Lip Mask", "asin": "B001", "rank": 10},
            "change": 10,
        }
        alerts = manager._check_rank_change(data)
        assert len(alerts) == 1

    def test_rank_change_zero(self, manager):
        data = {
            "product": {"name": "Lip Mask", "asin": "B001", "rank": 10},
            "change": 0,
        }
        alerts = manager._check_rank_change(data)
        assert len(alerts) == 0

    def test_rank_change_missing_product_fields(self, manager):
        data = {"product": {}, "change": 15}
        alerts = manager._check_rank_change(data)
        assert len(alerts) == 1
        assert alerts[0]["product"] is None
        assert alerts[0]["asin"] is None


# =========================================================================
# _check_crawl_complete()
# =========================================================================


class TestCheckCrawlComplete:
    def test_crawl_success(self, manager):
        data = {"result": {"success": True}}
        alerts = manager._check_crawl_complete(data)
        assert len(alerts) == 0

    def test_crawl_failure(self, manager):
        data = {"result": {"success": False, "error": "Network timeout"}}
        alerts = manager._check_crawl_complete(data)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "crawl_failed"
        assert alerts[0]["severity"] == "critical"
        assert "Network timeout" in alerts[0]["message"]
        assert "timestamp" in alerts[0]

    def test_crawl_failure_no_error_message(self, manager):
        data = {"result": {"success": False}}
        alerts = manager._check_crawl_complete(data)
        assert len(alerts) == 1
        assert "Unknown error" in alerts[0]["message"]

    def test_crawl_missing_result(self, manager):
        data = {}
        alerts = manager._check_crawl_complete(data)
        assert len(alerts) == 0

    def test_crawl_missing_success_field(self, manager):
        data = {"result": {}}
        alerts = manager._check_crawl_complete(data)
        assert len(alerts) == 0


# =========================================================================
# _check_metrics_calculated()
# =========================================================================


class TestCheckMetricsCalculated:
    def test_returns_empty_list(self, manager):
        alerts = manager._check_metrics_calculated({})
        assert alerts == []


# =========================================================================
# check_metrics_alerts()
# =========================================================================


class TestCheckMetricsAlerts:
    @pytest.mark.asyncio
    async def test_rank_delta_positive_above_threshold(self, manager):
        data = {
            "products": {
                "B001": {
                    "name": "Lip Mask",
                    "rank": 20,
                    "rank_delta": "+15",
                }
            }
        }
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "rank_change"
        assert alerts[0]["severity"] == "warning"
        assert alerts[0]["change"] == 15
        assert "급락" in alerts[0]["message"]

    @pytest.mark.asyncio
    async def test_rank_delta_negative_above_threshold(self, manager):
        data = {
            "products": {
                "B002": {
                    "name": "Lip Gloss",
                    "rank": 5,
                    "rank_delta": "-12",
                }
            }
        }
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "rank_change"
        assert alerts[0]["change"] == -12
        assert "급등" in alerts[0]["message"]

    @pytest.mark.asyncio
    async def test_rank_delta_critical(self, manager):
        data = {
            "products": {
                "B003": {
                    "name": "Product",
                    "rank": 50,
                    "rank_delta": "+25",
                }
            }
        }
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_rank_delta_below_threshold(self, manager):
        data = {
            "products": {
                "B004": {
                    "name": "Product",
                    "rank": 10,
                    "rank_delta": "+5",
                }
            }
        }
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_rank_delta_invalid_format(self, manager):
        data = {
            "products": {
                "B005": {
                    "name": "Product",
                    "rank": 10,
                    "rank_delta": "invalid",
                }
            }
        }
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_rank_delta_empty_string(self, manager):
        data = {
            "products": {
                "B006": {
                    "name": "Product",
                    "rank": 10,
                    "rank_delta": "",
                }
            }
        }
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_sos_drop_above_threshold(self, manager):
        data = {
            "products": {},
            "brand": {
                "kpis": {
                    "sos": "15.2%",
                    "sos_delta": "-3.5%p",
                }
            },
        }
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "sos_drop"
        assert alerts[0]["severity"] == "warning"
        assert alerts[0]["change"] == -3.5
        assert "LANEIGE SoS" in alerts[0]["message"]

    @pytest.mark.asyncio
    async def test_sos_drop_exactly_threshold(self, manager):
        data = {
            "products": {},
            "brand": {
                "kpis": {
                    "sos": "15.0%",
                    "sos_delta": "-2.0%p",
                }
            },
        }
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 1

    @pytest.mark.asyncio
    async def test_sos_drop_below_threshold(self, manager):
        data = {
            "products": {},
            "brand": {
                "kpis": {
                    "sos": "15.0%",
                    "sos_delta": "-1.5%p",
                }
            },
        }
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_sos_positive_change(self, manager):
        data = {
            "products": {},
            "brand": {
                "kpis": {
                    "sos": "18.0%",
                    "sos_delta": "+2.5%p",
                }
            },
        }
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_sos_invalid_format(self, manager):
        data = {
            "products": {},
            "brand": {
                "kpis": {
                    "sos": "15.0%",
                    "sos_delta": "invalid",
                }
            },
        }
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_multiple_alerts(self, manager):
        data = {
            "products": {
                "B001": {
                    "name": "Product A",
                    "rank": 20,
                    "rank_delta": "+15",
                },
                "B002": {
                    "name": "Product B",
                    "rank": 5,
                    "rank_delta": "-12",
                },
            },
            "brand": {
                "kpis": {
                    "sos": "15.0%",
                    "sos_delta": "-3.0%p",
                }
            },
        }
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 3

    @pytest.mark.asyncio
    async def test_empty_data(self, manager):
        alerts = await manager.check_metrics_alerts({})
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_missing_kpis(self, manager):
        data = {"products": {}, "brand": {}}
        alerts = await manager.check_metrics_alerts(data)
        assert len(alerts) == 0


# =========================================================================
# process_alert()
# =========================================================================


class TestProcessAlert:
    @pytest.mark.asyncio
    async def test_process_alert_success(self, manager, mock_alert_agent):
        alert = {
            "type": "rank_drop",
            "severity": "warning",
            "message": "Test alert",
            "timestamp": datetime.now().isoformat(),
        }
        result = await manager.process_alert(alert)
        assert result is True
        assert manager._stats["alerts_generated"] == 1
        assert manager._stats["alerts_sent"] == 1
        assert len(manager._alert_history) == 1
        mock_alert_agent.create_alert.assert_called_once()
        mock_alert_agent.send_pending_alerts.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_alert_no_alert_agent(self, mock_state_manager):
        mgr = AlertManager(state_manager=mock_state_manager, alert_agent=None)
        mgr._initialized = True
        alert = {"type": "rank_drop", "message": "Test"}
        result = await mgr.process_alert(alert)
        assert result is False
        assert mgr._stats["alerts_generated"] == 1
        assert mgr._stats["alerts_sent"] == 0

    @pytest.mark.asyncio
    async def test_process_alert_lazy_init(self, manager_no_deps):
        alert = {"type": "rank_drop", "message": "Test"}
        with patch.object(manager_no_deps, "initialize", new_callable=AsyncMock) as mock_init:
            manager_no_deps._alert_agent = None
            result = await manager_no_deps.process_alert(alert)
            mock_init.assert_called_once()
            assert result is False

    @pytest.mark.asyncio
    async def test_process_alert_send_failed(self, manager, mock_alert_agent):
        mock_alert_agent.send_pending_alerts.return_value = {"sent": 0, "failed": 1}
        alert = {"type": "rank_drop", "message": "Test"}
        result = await manager.process_alert(alert)
        assert result is False
        assert manager._stats["alerts_generated"] == 1
        assert manager._stats["alerts_sent"] == 0

    @pytest.mark.asyncio
    async def test_process_alert_exception_handling(self, manager, mock_alert_agent):
        mock_alert_agent.send_pending_alerts.side_effect = Exception("Send failed")
        alert = {"type": "rank_drop", "message": "Test"}
        result = await manager.process_alert(alert)
        assert result is False
        assert manager._stats["send_failures"] == 1

    @pytest.mark.asyncio
    async def test_process_alert_priority_mapping(self, manager, mock_alert_agent):
        alerts_with_priorities = [
            ("rank_drop", "HIGH"),
            ("rank_surge", "NORMAL"),
            ("sos_drop", "HIGH"),
            ("sos_surge", "NORMAL"),
            ("error", "CRITICAL"),
            ("crawl_complete", "LOW"),
            ("crawl_failed", "CRITICAL"),
            ("unknown_type", "NORMAL"),
        ]
        for alert_type, expected_priority in alerts_with_priorities:
            mock_alert_agent.reset_mock()
            alert = {"type": alert_type, "message": "Test"}
            await manager.process_alert(alert)
            call_kwargs = mock_alert_agent.create_alert.call_args[1]
            assert call_kwargs["priority"].name == expected_priority

    @pytest.mark.asyncio
    async def test_process_alert_history_pruning(self, manager):
        # Fill history to >100 items
        # Pruning happens when history > 100, then it's cut to 50 most recent
        for i in range(101):
            alert = {"type": "rank_drop", "message": f"Alert {i}"}
            await manager.process_alert(alert)

        # After 101st alert, history should be pruned to 50 most recent (items 51-100)
        assert len(manager._alert_history) == 50
        assert manager._alert_history[0]["message"] == "Alert 51"
        assert manager._alert_history[-1]["message"] == "Alert 100"

    @pytest.mark.asyncio
    async def test_process_alert_with_details_field(self, manager, mock_alert_agent):
        alert = {
            "type": "rank_drop",
            "message": "Short message",
            "details": "Long detailed message",
        }
        await manager.process_alert(alert)
        call_kwargs = mock_alert_agent.create_alert.call_args[1]
        assert call_kwargs["message"] == "Long detailed message"

    @pytest.mark.asyncio
    async def test_process_alert_without_details_field(self, manager, mock_alert_agent):
        alert = {"type": "rank_drop", "message": "Short message"}
        await manager.process_alert(alert)
        call_kwargs = mock_alert_agent.create_alert.call_args[1]
        assert call_kwargs["message"] == "Short message"


# =========================================================================
# get_recent_alerts()
# =========================================================================


class TestGetRecentAlerts:
    @pytest.mark.asyncio
    async def test_get_recent_alerts_default_limit(self, manager):
        for i in range(15):
            await manager.process_alert({"type": "test", "message": f"Alert {i}"})

        recent = manager.get_recent_alerts()
        assert len(recent) == 10
        assert recent[0]["message"] == "Alert 5"
        assert recent[-1]["message"] == "Alert 14"

    @pytest.mark.asyncio
    async def test_get_recent_alerts_custom_limit(self, manager):
        for i in range(20):
            await manager.process_alert({"type": "test", "message": f"Alert {i}"})

        recent = manager.get_recent_alerts(limit=5)
        assert len(recent) == 5
        assert recent[0]["message"] == "Alert 15"
        assert recent[-1]["message"] == "Alert 19"

    def test_get_recent_alerts_empty_history(self, manager):
        recent = manager.get_recent_alerts()
        assert len(recent) == 0

    @pytest.mark.asyncio
    async def test_get_recent_alerts_fewer_than_limit(self, manager):
        for i in range(3):
            await manager.process_alert({"type": "test", "message": f"Alert {i}"})

        recent = manager.get_recent_alerts(limit=10)
        assert len(recent) == 3


# =========================================================================
# get_stats()
# =========================================================================


class TestGetStats:
    def test_get_stats_initial(self, manager):
        stats = manager.get_stats()
        assert stats["alerts_generated"] == 0
        assert stats["alerts_sent"] == 0
        assert stats["send_failures"] == 0
        assert stats["history_size"] == 0
        assert stats["initialized"] is True

    @pytest.mark.asyncio
    async def test_get_stats_after_processing(self, manager):
        await manager.process_alert({"type": "test", "message": "Alert 1"})
        await manager.process_alert({"type": "test", "message": "Alert 2"})

        stats = manager.get_stats()
        assert stats["alerts_generated"] == 2
        assert stats["alerts_sent"] == 2
        assert stats["history_size"] == 2

    @pytest.mark.asyncio
    async def test_get_stats_with_failures(self, manager, mock_alert_agent):
        mock_alert_agent.send_pending_alerts.side_effect = Exception("Fail")
        await manager.process_alert({"type": "test", "message": "Alert 1"})
        await manager.process_alert({"type": "test", "message": "Alert 2"})

        stats = manager.get_stats()
        assert stats["alerts_generated"] == 2
        assert stats["send_failures"] == 2
        assert stats["alerts_sent"] == 0

    def test_get_stats_not_initialized(self, manager_no_deps):
        stats = manager_no_deps.get_stats()
        assert stats["initialized"] is False
