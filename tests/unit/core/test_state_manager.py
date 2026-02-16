"""
StateManager 단위 테스트
============================
src/core/state_manager.py 커버리지 22% → 60%+ 목표

테스트 대상:
- Enum (AlertType, DataFreshness)
- Dataclass (EmailSubscription, AgentStatus)
- StateManager 초기화 및 영속화
- 크롤링 상태 (is_crawl_needed, mark_crawled, mark_data_stale, get_data_age_hours)
- KG 상태 (mark_kg_initialized, update_kg_stats)
- 에이전트 상태 (start_agent, complete_agent, is_agent_running, etc.)
- 이메일 구독 (register_email, update, revoke, get_recipients)
- 세션 (set_session, clear_session)
- 메트릭 (mark_metrics_calculated, is_metrics_fresh)
- 직렬화 (to_dict, to_context_summary)
- 영속화 (_save_state, _load_state, _save_subscriptions, _load_subscriptions)
- 리셋 (reset)
- 싱글톤 (get_state_manager, reset_state_manager)
"""

import json
from datetime import datetime, timedelta

import pytest

from src.core.state_manager import (
    AgentStatus,
    AlertType,
    DataFreshness,
    EmailSubscription,
    StateManager,
    get_state_manager,
    reset_state_manager,
)

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def state_mgr(tmp_path):
    """tmp_path에 영속화되는 StateManager"""
    return StateManager(persist_dir=tmp_path)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """각 테스트 후 싱글톤 초기화"""
    yield
    reset_state_manager()


# =========================================================================
# Enum
# =========================================================================


class TestAlertType:
    def test_values(self):
        assert AlertType.RANK_CHANGE.value == "rank_change"
        assert AlertType.IMPORTANT_INSIGHT.value == "important_insight"
        assert AlertType.CRAWL_COMPLETE.value == "crawl_complete"
        assert AlertType.ERROR.value == "error"
        assert AlertType.DAILY_SUMMARY.value == "daily_summary"


class TestDataFreshness:
    def test_values(self):
        assert DataFreshness.FRESH.value == "fresh"
        assert DataFreshness.STALE.value == "stale"
        assert DataFreshness.UNKNOWN.value == "unknown"


# =========================================================================
# EmailSubscription
# =========================================================================


class TestEmailSubscription:
    def test_to_dict_minimal(self):
        sub = EmailSubscription(email="test@example.com", consent=True)
        d = sub.to_dict()
        assert d["email"] == "test@example.com"
        assert d["consent"] is True
        assert d["consent_date"] is None
        assert d["alert_types"] == []
        assert d["active"] is True
        assert d["verified"] is False
        assert d["verified_at"] is None

    def test_to_dict_full(self):
        now = datetime.now()
        sub = EmailSubscription(
            email="user@example.com",
            consent=True,
            consent_date=now,
            alert_types=["rank_change", "error"],
            active=True,
            verified=True,
            verified_at=now,
        )
        d = sub.to_dict()
        assert d["consent_date"] == now.isoformat()
        assert d["verified_at"] == now.isoformat()
        assert len(d["alert_types"]) == 2

    def test_from_dict_minimal(self):
        data = {"email": "test@example.com", "consent": True}
        sub = EmailSubscription.from_dict(data)
        assert sub.email == "test@example.com"
        assert sub.consent is True
        assert sub.consent_date is None
        assert sub.active is True
        assert sub.verified is False

    def test_from_dict_full(self):
        data = {
            "email": "user@example.com",
            "consent": True,
            "consent_date": "2025-01-15T10:00:00",
            "alert_types": ["rank_change"],
            "active": False,
            "verified": True,
            "verified_at": "2025-01-16T12:00:00",
        }
        sub = EmailSubscription.from_dict(data)
        assert sub.consent_date == datetime(2025, 1, 15, 10, 0)
        assert sub.verified_at == datetime(2025, 1, 16, 12, 0)
        assert sub.active is False

    def test_roundtrip(self):
        now = datetime(2025, 6, 1, 12, 0)
        original = EmailSubscription(
            email="rt@test.com",
            consent=True,
            consent_date=now,
            alert_types=["error"],
            active=True,
            verified=True,
            verified_at=now,
        )
        restored = EmailSubscription.from_dict(original.to_dict())
        assert restored.email == original.email
        assert restored.consent_date == original.consent_date
        assert restored.verified_at == original.verified_at


# =========================================================================
# AgentStatus
# =========================================================================


class TestAgentStatus:
    def test_defaults(self):
        status = AgentStatus(name="crawler", status="idle")
        assert status.last_run is None
        assert status.last_error is None
        assert status.run_count == 0
        assert status.success_count == 0
        assert status.fail_count == 0


# =========================================================================
# StateManager 초기화
# =========================================================================


class TestStateManagerInit:
    def test_defaults(self, state_mgr):
        assert state_mgr.last_crawl_time is None
        assert state_mgr.last_crawl_success is False
        assert state_mgr.last_crawl_count == 0
        assert state_mgr.data_freshness == DataFreshness.UNKNOWN
        assert state_mgr.kg_initialized is False
        assert state_mgr.kg_triple_count == 0
        assert state_mgr.current_session_id is None
        assert state_mgr.last_metrics_time is None

    def test_persist_dir_created(self, tmp_path):
        nested = tmp_path / "nested" / "dir"
        mgr = StateManager(persist_dir=nested)
        assert nested.exists()
        assert mgr.persist_dir == nested


# =========================================================================
# 크롤링 상태
# =========================================================================


class TestCrawlState:
    def test_is_crawl_needed_no_history(self, state_mgr):
        assert state_mgr.is_crawl_needed() is True

    def test_is_crawl_needed_today(self, state_mgr):
        state_mgr.last_crawl_time = datetime.now()
        assert state_mgr.is_crawl_needed() is False

    def test_is_crawl_needed_yesterday(self, state_mgr):
        state_mgr.last_crawl_time = datetime.now() - timedelta(days=1)
        assert state_mgr.is_crawl_needed() is True

    def test_mark_crawled_success(self, state_mgr):
        state_mgr.mark_crawled(success=True, products_count=50)
        assert state_mgr.last_crawl_time is not None
        assert state_mgr.last_crawl_success is True
        assert state_mgr.last_crawl_count == 50
        assert state_mgr.data_freshness == DataFreshness.FRESH

    def test_mark_crawled_failure(self, state_mgr):
        state_mgr.mark_crawled(success=False, products_count=0)
        assert state_mgr.last_crawl_success is False
        assert state_mgr.data_freshness == DataFreshness.UNKNOWN

    def test_mark_data_stale(self, state_mgr):
        state_mgr.mark_crawled(success=True)
        state_mgr.mark_data_stale()
        assert state_mgr.data_freshness == DataFreshness.STALE

    def test_get_data_age_hours_none(self, state_mgr):
        assert state_mgr.get_data_age_hours() is None

    def test_get_data_age_hours_recent(self, state_mgr):
        state_mgr.last_crawl_time = datetime.now() - timedelta(hours=3)
        age = state_mgr.get_data_age_hours()
        assert age is not None
        assert 2.9 < age < 3.1


# =========================================================================
# KG 상태
# =========================================================================


class TestKGState:
    def test_mark_kg_initialized(self, state_mgr):
        state_mgr.mark_kg_initialized(triple_count=1000)
        assert state_mgr.kg_initialized is True
        assert state_mgr.kg_triple_count == 1000
        assert state_mgr.kg_last_update is not None

    def test_update_kg_stats(self, state_mgr):
        state_mgr.mark_kg_initialized(500)
        state_mgr.update_kg_stats(triple_count=750)
        assert state_mgr.kg_triple_count == 750


# =========================================================================
# 에이전트 상태
# =========================================================================


class TestAgentState:
    def test_start_agent(self, state_mgr):
        state_mgr.start_agent("crawler")
        assert state_mgr.is_agent_running("crawler") is True
        assert state_mgr.has_active_agents() is True

    def test_complete_agent_success(self, state_mgr):
        state_mgr.start_agent("crawler")
        state_mgr.complete_agent("crawler", success=True)
        assert state_mgr.is_agent_running("crawler") is False
        status = state_mgr.get_agent_status("crawler")
        assert status.status == "completed"
        assert status.success_count == 1

    def test_complete_agent_failure(self, state_mgr):
        state_mgr.start_agent("crawler")
        state_mgr.complete_agent("crawler", success=False, error="Timeout")
        status = state_mgr.get_agent_status("crawler")
        assert status.status == "failed"
        assert status.fail_count == 1
        assert status.last_error == "Timeout"

    def test_complete_agent_not_started(self, state_mgr):
        state_mgr.complete_agent("unknown", success=True)
        status = state_mgr.get_agent_status("unknown")
        assert status is not None
        assert status.status == "completed"

    def test_multiple_agents(self, state_mgr):
        state_mgr.start_agent("crawler")
        state_mgr.start_agent("metrics")
        assert state_mgr.has_active_agents() is True
        assert state_mgr.is_agent_running("crawler") is True
        assert state_mgr.is_agent_running("metrics") is True

        state_mgr.complete_agent("crawler")
        assert state_mgr.is_agent_running("crawler") is False
        assert state_mgr.has_active_agents() is True

    def test_get_agent_status_missing(self, state_mgr):
        assert state_mgr.get_agent_status("nonexistent") is None

    def test_get_all_agent_statuses(self, state_mgr):
        state_mgr.start_agent("a")
        state_mgr.start_agent("b")
        statuses = state_mgr.get_all_agent_statuses()
        assert "a" in statuses
        assert "b" in statuses
        # Copy check
        statuses["c"] = AgentStatus(name="c", status="idle")
        assert "c" not in state_mgr._agent_status

    def test_run_count_increments(self, state_mgr):
        state_mgr.start_agent("crawler")
        state_mgr.complete_agent("crawler")
        state_mgr.start_agent("crawler")
        state_mgr.complete_agent("crawler")
        status = state_mgr.get_agent_status("crawler")
        assert status.run_count == 2
        assert status.success_count == 2


# =========================================================================
# 이메일 구독
# =========================================================================


class TestEmailSubscriptionManagement:
    def test_register_email_success(self, state_mgr):
        result = state_mgr.register_email("user@example.com", consent=True)
        assert result is True
        sub = state_mgr.get_subscription("user@example.com")
        assert sub is not None
        assert sub.consent is True
        assert sub.consent_date is not None

    def test_register_email_no_consent(self, state_mgr):
        result = state_mgr.register_email("user@example.com", consent=False)
        assert result is False
        assert state_mgr.get_subscription("user@example.com") is None

    def test_register_email_invalid(self, state_mgr):
        result = state_mgr.register_email("not-an-email", consent=True)
        assert result is False

    def test_register_email_custom_alert_types(self, state_mgr):
        result = state_mgr.register_email(
            "user@example.com",
            consent=True,
            alert_types=["rank_change"],
        )
        assert result is True
        sub = state_mgr.get_subscription("user@example.com")
        assert sub.alert_types == ["rank_change"]

    def test_register_email_default_alert_types(self, state_mgr):
        state_mgr.register_email("user@example.com", consent=True)
        sub = state_mgr.get_subscription("user@example.com")
        assert AlertType.RANK_CHANGE.value in sub.alert_types
        assert AlertType.IMPORTANT_INSIGHT.value in sub.alert_types
        assert AlertType.ERROR.value in sub.alert_types

    def test_update_subscription(self, state_mgr):
        state_mgr.register_email("user@example.com", consent=True)
        result = state_mgr.update_email_subscription(
            "user@example.com",
            alert_types=["daily_summary"],
            active=False,
        )
        assert result is True
        sub = state_mgr.get_subscription("user@example.com")
        assert sub.alert_types == ["daily_summary"]
        assert sub.active is False

    def test_update_subscription_nonexistent(self, state_mgr):
        result = state_mgr.update_email_subscription("no@example.com", active=False)
        assert result is False

    def test_revoke_consent(self, state_mgr):
        state_mgr.register_email("user@example.com", consent=True)
        result = state_mgr.revoke_email_consent("user@example.com")
        assert result is True
        sub = state_mgr.get_subscription("user@example.com")
        assert sub.consent is False
        assert sub.active is False

    def test_revoke_consent_nonexistent(self, state_mgr):
        result = state_mgr.revoke_email_consent("no@example.com")
        assert result is False

    def test_get_alert_recipients(self, state_mgr):
        state_mgr.register_email("a@test.com", consent=True, alert_types=["rank_change"])
        state_mgr.register_email("b@test.com", consent=True, alert_types=["rank_change", "error"])
        state_mgr.register_email("c@test.com", consent=True, alert_types=["error"])

        recipients = state_mgr.get_alert_recipients("rank_change")
        assert "a@test.com" in recipients
        assert "b@test.com" in recipients
        assert "c@test.com" not in recipients

    def test_get_alert_recipients_revoked(self, state_mgr):
        state_mgr.register_email("user@test.com", consent=True, alert_types=["rank_change"])
        state_mgr.revoke_email_consent("user@test.com")
        recipients = state_mgr.get_alert_recipients("rank_change")
        assert len(recipients) == 0

    def test_get_all_subscriptions(self, state_mgr):
        state_mgr.register_email("a@test.com", consent=True)
        state_mgr.register_email("b@test.com", consent=True)
        subs = state_mgr.get_all_subscriptions()
        assert len(subs) == 2
        # Copy check
        subs["c@test.com"] = EmailSubscription(email="c", consent=False)
        assert "c@test.com" not in state_mgr._email_subscriptions

    def test_validate_email_valid(self, state_mgr):
        assert state_mgr._validate_email("user@example.com") is True
        assert state_mgr._validate_email("user.name+tag@domain.co.kr") is True

    def test_validate_email_invalid(self, state_mgr):
        assert state_mgr._validate_email("not-email") is False
        assert state_mgr._validate_email("@domain.com") is False
        assert state_mgr._validate_email("user@") is False
        assert state_mgr._validate_email("") is False


# =========================================================================
# 세션
# =========================================================================


class TestSession:
    def test_set_session(self, state_mgr):
        state_mgr.set_session("sess-123")
        assert state_mgr.current_session_id == "sess-123"

    def test_clear_session(self, state_mgr):
        state_mgr.set_session("sess-123")
        state_mgr.clear_session()
        assert state_mgr.current_session_id is None


# =========================================================================
# 메트릭 상태
# =========================================================================


class TestMetrics:
    def test_mark_metrics_calculated(self, state_mgr):
        state_mgr.mark_metrics_calculated()
        assert state_mgr.last_metrics_time is not None

    def test_is_metrics_fresh_no_calc(self, state_mgr):
        assert state_mgr.is_metrics_fresh() is False

    def test_is_metrics_fresh_recent(self, state_mgr):
        state_mgr.last_metrics_time = datetime.now() - timedelta(hours=1)
        assert state_mgr.is_metrics_fresh(max_age_hours=24) is True

    def test_is_metrics_fresh_stale(self, state_mgr):
        state_mgr.last_metrics_time = datetime.now() - timedelta(hours=25)
        assert state_mgr.is_metrics_fresh(max_age_hours=24) is False


# =========================================================================
# 직렬화
# =========================================================================


class TestSerialization:
    def test_to_dict_defaults(self, state_mgr):
        d = state_mgr.to_dict()
        assert d["last_crawl_time"] is None
        assert d["last_crawl_success"] is False
        assert d["last_crawl_count"] == 0
        assert d["data_freshness"] == "unknown"
        assert d["kg_initialized"] is False
        assert d["kg_triple_count"] == 0
        assert d["kg_last_update"] is None
        assert d["last_metrics_time"] is None

    def test_to_dict_with_data(self, state_mgr):
        state_mgr.mark_crawled(success=True, products_count=50)
        state_mgr.mark_kg_initialized(100)
        state_mgr.mark_metrics_calculated()
        d = state_mgr.to_dict()
        assert d["last_crawl_time"] is not None
        assert d["last_crawl_success"] is True
        assert d["data_freshness"] == "fresh"
        assert d["kg_initialized"] is True
        assert d["kg_last_update"] is not None

    def test_to_context_summary_no_data(self, state_mgr):
        summary = state_mgr.to_context_summary()
        assert "크롤링: 없음" in summary
        assert "KG: 미초기화" in summary

    def test_to_context_summary_with_data(self, state_mgr):
        state_mgr.mark_crawled(success=True, products_count=50)
        state_mgr.mark_kg_initialized(200)
        summary = state_mgr.to_context_summary()
        assert "크롤링:" in summary
        assert "시간 전" in summary
        assert "KG: 200 트리플" in summary

    def test_to_context_summary_active_tools(self, state_mgr):
        state_mgr.start_agent("crawler")
        summary = state_mgr.to_context_summary()
        assert "실행 중: crawler" in summary


# =========================================================================
# 영속화
# =========================================================================


class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        mgr1 = StateManager(persist_dir=tmp_path)
        mgr1.mark_crawled(success=True, products_count=100)
        mgr1.mark_kg_initialized(500)
        mgr1.mark_metrics_calculated()

        mgr2 = StateManager(persist_dir=tmp_path)
        assert mgr2.last_crawl_time is not None
        assert mgr2.last_crawl_success is True
        assert mgr2.last_crawl_count == 100
        assert mgr2.data_freshness == DataFreshness.FRESH
        assert mgr2.kg_initialized is True
        assert mgr2.kg_triple_count == 500
        assert mgr2.last_metrics_time is not None

    def test_save_subscriptions_and_reload(self, tmp_path):
        mgr1 = StateManager(persist_dir=tmp_path)
        mgr1.register_email("user@test.com", consent=True, alert_types=["rank_change"])

        mgr2 = StateManager(persist_dir=tmp_path)
        sub = mgr2.get_subscription("user@test.com")
        assert sub is not None
        assert sub.consent is True
        assert sub.alert_types == ["rank_change"]

    def test_load_state_no_file(self, tmp_path):
        mgr = StateManager(persist_dir=tmp_path)
        assert mgr.last_crawl_time is None

    def test_load_state_corrupt_file(self, tmp_path):
        state_path = tmp_path / "system_state.json"
        state_path.write_text("not valid json!", encoding="utf-8")
        mgr = StateManager(persist_dir=tmp_path)
        assert mgr.data_freshness == DataFreshness.UNKNOWN

    def test_load_subscriptions_corrupt(self, tmp_path):
        sub_path = tmp_path / "email_subscriptions.json"
        sub_path.write_text("{invalid}", encoding="utf-8")
        mgr = StateManager(persist_dir=tmp_path)
        assert len(mgr.get_all_subscriptions()) == 0

    def test_save_state_creates_file(self, state_mgr):
        state_mgr.mark_crawled()
        state_path = state_mgr.persist_dir / "system_state.json"
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["data_freshness"] == "fresh"
        assert "saved_at" in data


# =========================================================================
# 리셋
# =========================================================================


class TestReset:
    def test_reset_clears_all(self, state_mgr):
        state_mgr.mark_crawled(success=True, products_count=100)
        state_mgr.mark_kg_initialized(500)
        state_mgr.set_session("sess-1")
        state_mgr.start_agent("crawler")
        state_mgr.mark_metrics_calculated()

        state_mgr.reset()

        assert state_mgr.last_crawl_time is None
        assert state_mgr.last_crawl_success is False
        assert state_mgr.last_crawl_count == 0
        assert state_mgr.data_freshness == DataFreshness.UNKNOWN
        assert state_mgr.kg_initialized is False
        assert state_mgr.kg_triple_count == 0
        assert state_mgr.kg_last_update is None
        assert state_mgr.current_session_id is None
        assert state_mgr.last_metrics_time is None
        assert not state_mgr.has_active_agents()

    def test_reset_deletes_file(self, state_mgr):
        state_mgr.mark_crawled()
        state_path = state_mgr.persist_dir / "system_state.json"
        assert state_path.exists()

        state_mgr.reset()
        assert not state_path.exists()

    def test_reset_no_file_no_error(self, state_mgr):
        state_mgr.reset()  # No exception


# =========================================================================
# 싱글톤
# =========================================================================


class TestSingleton:
    def test_get_state_manager_returns_same_instance(self):
        mgr1 = get_state_manager()
        mgr2 = get_state_manager()
        assert mgr1 is mgr2

    def test_reset_state_manager(self):
        mgr1 = get_state_manager()
        reset_state_manager()
        mgr2 = get_state_manager()
        assert mgr1 is not mgr2
