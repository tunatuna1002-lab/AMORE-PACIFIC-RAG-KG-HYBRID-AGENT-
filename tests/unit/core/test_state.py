"""
OrchestratorState 단위 테스트
==============================
src/core/state.py 커버리지 48% → 90%+ 목표

테스트 대상:
- 초기화 및 영속화 (_load_state, _save_state)
- 크롤링 상태 (is_crawl_needed, mark_crawled, mark_data_stale, get_data_age_hours)
- 지표 상태 (mark_metrics_calculated, is_metrics_fresh)
- KG 상태 (mark_kg_initialized, update_kg_stats)
- 도구 실행 (start_tool, end_tool, is_tool_running, has_active_tools)
- 세션 관리 (set_session, clear_session)
- 직렬화 (to_dict, to_context_summary)
- 리셋 (reset)
"""

import json
from datetime import datetime, timedelta

import pytest

from src.core.state import OrchestratorState


@pytest.fixture
def state(tmp_path):
    """tmp_path에 영속화되는 OrchestratorState"""
    persist = tmp_path / "state.json"
    return OrchestratorState(_persist_path=persist)


@pytest.fixture
def state_with_crawl(tmp_path):
    """크롤링 완료 상태"""
    persist = tmp_path / "state.json"
    s = OrchestratorState(_persist_path=persist)
    s.mark_crawled(products_count=100)
    return s


# =========================================================================
# 초기화
# =========================================================================


class TestInitialization:
    """초기화 테스트"""

    def test_defaults(self, state):
        """기본값 확인"""
        assert state.last_crawl_time is None
        assert state.data_freshness == "unknown"
        assert state.kg_initialized is False
        assert state.kg_triple_count == 0
        assert state.current_session_id is None
        assert state.active_tools == []
        assert state.last_metrics_time is None

    def test_default_persist_path(self):
        """기본 영속화 경로"""
        s = OrchestratorState()
        assert str(s._persist_path).endswith("orchestrator_state.json")

    def test_load_state_on_init(self, tmp_path):
        """초기화 시 영속화된 상태 로드"""
        persist = tmp_path / "state.json"
        data = {
            "last_crawl_time": "2025-01-15T10:00:00",
            "last_metrics_time": "2025-01-15T12:00:00",
            "data_freshness": "fresh",
            "kg_initialized": True,
            "kg_triple_count": 500,
        }
        persist.write_text(json.dumps(data), encoding="utf-8")

        s = OrchestratorState(_persist_path=persist)
        assert s.last_crawl_time == datetime(2025, 1, 15, 10, 0)
        assert s.last_metrics_time == datetime(2025, 1, 15, 12, 0)
        assert s.data_freshness == "fresh"
        assert s.kg_initialized is True
        assert s.kg_triple_count == 500

    def test_load_state_no_file(self, tmp_path):
        """영속화 파일이 없을 때 기본값 유지"""
        persist = tmp_path / "nonexistent.json"
        s = OrchestratorState(_persist_path=persist)
        assert s.last_crawl_time is None

    def test_load_state_corrupt_file(self, tmp_path):
        """파일 손상 시 기본값 유지"""
        persist = tmp_path / "state.json"
        persist.write_text("not valid json!", encoding="utf-8")
        s = OrchestratorState(_persist_path=persist)
        assert s.data_freshness == "unknown"

    def test_load_state_partial_data(self, tmp_path):
        """일부 필드만 있는 상태 파일"""
        persist = tmp_path / "state.json"
        data = {"data_freshness": "stale"}
        persist.write_text(json.dumps(data), encoding="utf-8")

        s = OrchestratorState(_persist_path=persist)
        assert s.data_freshness == "stale"
        assert s.last_crawl_time is None
        assert s.kg_initialized is False


# =========================================================================
# 크롤링 상태
# =========================================================================


class TestCrawlState:
    """크롤링 상태 관리 테스트"""

    def test_is_crawl_needed_no_history(self, state):
        """크롤링 기록 없으면 필요"""
        assert state.is_crawl_needed() is True

    def test_is_crawl_needed_today_crawled(self, state):
        """오늘 크롤링 완료면 불필요"""
        state.last_crawl_time = datetime.now()
        assert state.is_crawl_needed() is False

    def test_is_crawl_needed_yesterday(self, state):
        """어제 크롤링이면 필요"""
        state.last_crawl_time = datetime.now() - timedelta(days=1)
        assert state.is_crawl_needed() is True

    def test_mark_crawled(self, state):
        """크롤링 완료 표시"""
        state.mark_crawled(products_count=50)
        assert state.last_crawl_time is not None
        assert state.data_freshness == "fresh"

    def test_mark_crawled_persists(self, state):
        """크롤링 완료가 파일에 저장됨"""
        state.mark_crawled(products_count=10)
        assert state._persist_path.exists()
        data = json.loads(state._persist_path.read_text())
        assert data["data_freshness"] == "fresh"
        assert data["last_crawl_time"] is not None

    def test_mark_data_stale(self, state):
        """데이터 stale 표시"""
        state.mark_crawled()
        state.mark_data_stale()
        assert state.data_freshness == "stale"

    def test_get_data_age_hours_no_crawl(self, state):
        """크롤링 기록 없으면 None"""
        assert state.get_data_age_hours() is None

    def test_get_data_age_hours_recent(self, state):
        """최근 크롤링 경과 시간"""
        state.last_crawl_time = datetime.now() - timedelta(hours=2)
        age = state.get_data_age_hours()
        assert age is not None
        assert 1.9 < age < 2.1


# =========================================================================
# 지표 상태
# =========================================================================


class TestMetricsState:
    """지표 계산 상태 테스트"""

    def test_mark_metrics_calculated(self, state):
        """지표 계산 완료 표시"""
        state.mark_metrics_calculated()
        assert state.last_metrics_time is not None

    def test_is_metrics_fresh_no_calc(self, state):
        """지표 계산 기록 없으면 False"""
        assert state.is_metrics_fresh() is False

    def test_is_metrics_fresh_recent(self, state):
        """최근 계산이면 True"""
        state.last_metrics_time = datetime.now() - timedelta(hours=1)
        assert state.is_metrics_fresh(max_age_hours=24) is True

    def test_is_metrics_fresh_stale(self, state):
        """오래된 계산이면 False"""
        state.last_metrics_time = datetime.now() - timedelta(hours=25)
        assert state.is_metrics_fresh(max_age_hours=24) is False

    def test_is_metrics_fresh_custom_age(self, state):
        """커스텀 max_age_hours"""
        state.last_metrics_time = datetime.now() - timedelta(hours=5)
        assert state.is_metrics_fresh(max_age_hours=6) is True
        assert state.is_metrics_fresh(max_age_hours=4) is False


# =========================================================================
# KG 상태
# =========================================================================


class TestKGState:
    """KG 상태 관리 테스트"""

    def test_mark_kg_initialized(self, state):
        """KG 초기화 완료 표시"""
        state.mark_kg_initialized(triple_count=1000)
        assert state.kg_initialized is True
        assert state.kg_triple_count == 1000

    def test_update_kg_stats(self, state):
        """KG 통계 업데이트"""
        state.mark_kg_initialized(triple_count=500)
        state.update_kg_stats(triple_count=750)
        assert state.kg_triple_count == 750


# =========================================================================
# 도구 실행 상태
# =========================================================================


class TestToolState:
    """도구 실행 상태 테스트"""

    def test_start_tool(self, state):
        """도구 시작"""
        state.start_tool("crawler")
        assert state.is_tool_running("crawler") is True
        assert state.has_active_tools() is True

    def test_end_tool(self, state):
        """도구 종료"""
        state.start_tool("crawler")
        state.end_tool("crawler")
        assert state.is_tool_running("crawler") is False
        assert state.has_active_tools() is False

    def test_start_tool_no_duplicate(self, state):
        """같은 도구 중복 시작 방지"""
        state.start_tool("crawler")
        state.start_tool("crawler")
        assert state.active_tools.count("crawler") == 1

    def test_end_tool_not_running(self, state):
        """실행 중이 아닌 도구 종료 — 에러 없음"""
        state.end_tool("nonexistent")
        assert state.has_active_tools() is False

    def test_multiple_tools(self, state):
        """여러 도구 동시 실행"""
        state.start_tool("crawler")
        state.start_tool("metrics")
        assert state.has_active_tools() is True
        assert state.is_tool_running("crawler") is True
        assert state.is_tool_running("metrics") is True

        state.end_tool("crawler")
        assert state.is_tool_running("crawler") is False
        assert state.has_active_tools() is True


# =========================================================================
# 세션 관리
# =========================================================================


class TestSessionManagement:
    """세션 관리 테스트"""

    def test_set_session(self, state):
        """세션 설정"""
        state.set_session("session-123")
        assert state.current_session_id == "session-123"

    def test_clear_session(self, state):
        """세션 초기화"""
        state.set_session("session-123")
        state.clear_session()
        assert state.current_session_id is None


# =========================================================================
# 직렬화
# =========================================================================


class TestSerialization:
    """직렬화 테스트"""

    def test_to_dict_defaults(self, state):
        """기본 상태 직렬화"""
        d = state.to_dict()
        assert d["last_crawl_time"] is None
        assert d["data_freshness"] == "unknown"
        assert d["kg_initialized"] is False
        assert d["kg_triple_count"] == 0
        assert d["current_session_id"] is None
        assert d["active_tools"] == []
        assert d["last_metrics_time"] is None

    def test_to_dict_with_data(self, state):
        """데이터가 있는 상태 직렬화"""
        state.mark_crawled(50)
        state.mark_kg_initialized(100)
        state.set_session("sess-1")
        state.start_tool("crawler")
        state.mark_metrics_calculated()

        d = state.to_dict()
        assert d["last_crawl_time"] is not None
        assert d["data_freshness"] == "fresh"
        assert d["kg_initialized"] is True
        assert d["kg_triple_count"] == 100
        assert d["current_session_id"] == "sess-1"
        assert "crawler" in d["active_tools"]
        assert d["last_metrics_time"] is not None

    def test_to_context_summary_no_crawl(self, state):
        """크롤링 기록 없는 컨텍스트 요약"""
        summary = state.to_context_summary()
        assert "마지막 크롤링: 없음" in summary
        assert "KG: 미초기화" in summary

    def test_to_context_summary_with_crawl(self, state):
        """크롤링 기록 있는 컨텍스트 요약"""
        state.mark_crawled(50)
        state.mark_kg_initialized(200)
        summary = state.to_context_summary()
        assert "마지막 크롤링:" in summary
        assert "시간 전" in summary
        assert "KG: 초기화됨" in summary
        assert "200 트리플" in summary

    def test_to_context_summary_kg_not_initialized(self, state):
        """KG 미초기화 상태 요약"""
        summary = state.to_context_summary()
        assert "KG: 미초기화" in summary


# =========================================================================
# 영속화
# =========================================================================


class TestPersistence:
    """영속화 테스트"""

    def test_save_and_reload(self, tmp_path):
        """저장 후 새 인스턴스로 로드"""
        persist = tmp_path / "state.json"
        s1 = OrchestratorState(_persist_path=persist)
        s1.mark_crawled(100)
        s1.mark_kg_initialized(500)
        s1.mark_metrics_calculated()

        # 새 인스턴스 생성 — 파일에서 로드
        s2 = OrchestratorState(_persist_path=persist)
        assert s2.last_crawl_time is not None
        assert s2.data_freshness == "fresh"
        assert s2.kg_initialized is True
        assert s2.kg_triple_count == 500
        assert s2.last_metrics_time is not None

    def test_save_creates_directory(self, tmp_path):
        """중간 디렉토리 자동 생성"""
        persist = tmp_path / "nested" / "dir" / "state.json"
        s = OrchestratorState(_persist_path=persist)
        s.mark_crawled()
        assert persist.exists()

    def test_save_failure_no_exception(self, state):
        """저장 실패 시 예외 없이 경고만"""
        state._persist_path = None  # None이면 AttributeError
        # _save_state에서 예외가 잡혀야 함
        state._persist_path = type(
            "FakePath",
            (),
            {
                "parent": type(
                    "FakeParent",
                    (),
                    {"mkdir": lambda self, **kw: (_ for _ in ()).throw(PermissionError("denied"))},
                )()
            },
        )()
        # Should not raise
        state._save_state()


# =========================================================================
# 리셋
# =========================================================================


class TestReset:
    """상태 리셋 테스트"""

    def test_reset_clears_all(self, state):
        """전체 초기화"""
        state.mark_crawled(100)
        state.mark_kg_initialized(500)
        state.set_session("sess-1")
        state.start_tool("crawler")
        state.mark_metrics_calculated()

        state.reset()

        assert state.last_crawl_time is None
        assert state.data_freshness == "unknown"
        assert state.kg_initialized is False
        assert state.kg_triple_count == 0
        assert state.current_session_id is None
        assert state.active_tools == []
        assert state.last_metrics_time is None

    def test_reset_deletes_file(self, tmp_path):
        """리셋 시 영속화 파일 삭제"""
        persist = tmp_path / "state.json"
        s = OrchestratorState(_persist_path=persist)
        s.mark_crawled()
        assert persist.exists()

        s.reset()
        assert not persist.exists()

    def test_reset_no_file_no_error(self, state):
        """파일이 없어도 리셋 가능"""
        state.reset()  # No exception
        assert state.data_freshness == "unknown"
