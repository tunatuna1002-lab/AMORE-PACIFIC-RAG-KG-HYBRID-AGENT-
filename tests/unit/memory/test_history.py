"""Tests for src.memory.history module."""

from datetime import datetime, timedelta

import pytest

from src.memory.history import HistoryManager


class TestHistoryManager:
    @pytest.fixture
    def history(self, tmp_path):
        return HistoryManager(history_dir=str(tmp_path / "history"))

    def test_add_execution(self, history):
        history.add_execution({"session_id": "abc123", "status": "completed", "agents": {}})
        recent = history.get_recent_executions(days=1)
        assert len(recent) == 1
        assert recent[0]["session_id"] == "abc123"
        assert "timestamp" in recent[0]

    def test_persistence(self, tmp_path):
        history_dir = str(tmp_path / "history")
        h1 = HistoryManager(history_dir=history_dir)
        h1.add_execution({"session_id": "s1", "status": "completed"})

        h2 = HistoryManager(history_dir=history_dir)
        recent = h2.get_recent_executions(days=1)
        assert len(recent) == 1

    def test_max_history_limit(self, history):
        for i in range(1010):
            history.add_execution({"session_id": f"s{i}", "status": "completed"})
        assert len(history._history) == 1000

    def test_get_recent_executions_date_filter(self, history):
        # Add an old record by manipulating history directly
        old_record = {
            "timestamp": (datetime.now() - timedelta(days=30)).isoformat(),
            "session_id": "old",
            "status": "completed",
        }
        history._history.append(old_record)

        history.add_execution({"session_id": "new", "status": "completed"})

        recent = history.get_recent_executions(days=7)
        assert len(recent) == 1
        assert recent[0]["session_id"] == "new"

    def test_get_recent_executions_limit(self, history):
        for i in range(10):
            history.add_execution({"session_id": f"s{i}", "status": "completed"})
        recent = history.get_recent_executions(days=7, limit=3)
        assert len(recent) == 3

    def test_get_success_rate_empty(self, history):
        rates = history.get_success_rate()
        assert rates == {"overall": 0.0}

    def test_get_success_rate(self, history):
        for i in range(8):
            history.add_execution({"session_id": f"s{i}", "status": "completed"})
        for i in range(2):
            history.add_execution({"session_id": f"f{i}", "status": "failed"})

        rates = history.get_success_rate()
        assert rates["overall"] == 0.8

    def test_get_success_rate_per_agent(self, history):
        history.add_execution(
            {
                "session_id": "s1",
                "status": "completed",
                "agents": {
                    "crawler_agent": {"status": "completed"},
                    "insight_agent": {"status": "failed"},
                },
            }
        )
        history.add_execution(
            {
                "session_id": "s2",
                "status": "completed",
                "agents": {
                    "crawler_agent": {"status": "completed"},
                    "insight_agent": {"status": "completed"},
                },
            }
        )
        rates = history.get_success_rate()
        assert rates["crawler_agent"] == 1.0
        assert rates["insight_agent"] == 0.5

    def test_get_average_duration_empty(self, history):
        assert history.get_average_duration() == {}

    def test_get_average_duration(self, history):
        history.add_execution(
            {
                "session_id": "s1",
                "status": "completed",
                "total_duration_seconds": 10.0,
                "agents": {"crawler_agent": {"duration_seconds": 5.0}},
            }
        )
        history.add_execution(
            {
                "session_id": "s2",
                "status": "completed",
                "total_duration_seconds": 20.0,
                "agents": {"crawler_agent": {"duration_seconds": 15.0}},
            }
        )
        avg = history.get_average_duration()
        assert avg["overall"] == 15.0
        assert avg["crawler_agent"] == 10.0

    def test_get_error_summary(self, history):
        history.add_execution(
            {"session_id": "s1", "status": "failed", "error": "DB connection timeout", "agents": {}}
        )
        history.add_execution(
            {
                "session_id": "s2",
                "status": "completed",
                "agents": {"crawler_agent": {"status": "failed", "error": "WAF blocked"}},
            }
        )
        errors = history.get_error_summary()
        assert len(errors) == 2
        assert errors[0]["error"] == "DB connection timeout"
        assert errors[1]["agent"] == "crawler_agent"

    def test_get_daily_stats(self, history):
        history.add_execution({"session_id": "s1", "status": "completed"})
        history.add_execution({"session_id": "s2", "status": "failed"})
        history.add_execution({"session_id": "s3", "status": "completed"})

        stats = history.get_daily_stats()
        assert len(stats) == 1
        today_stats = stats[0]
        assert today_stats["total"] == 3
        assert today_stats["success"] == 2
        assert today_stats["failed"] == 1
        assert today_stats["success_rate"] == pytest.approx(0.667, abs=0.001)

    def test_load_corrupted_history(self, tmp_path):
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        (history_dir / "execution_history.json").write_text("invalid{json")

        h = HistoryManager(history_dir=str(history_dir))
        assert h._history == []
