"""Tests for src.memory.session module."""

import pytest

from src.memory.session import (
    AgentState,
    AgentStatus,
    SessionManager,
    SessionStatus,
)


class TestSessionStatus:
    def test_values(self):
        assert SessionStatus.CREATED == "created"
        assert SessionStatus.RUNNING == "running"
        assert SessionStatus.COMPLETED == "completed"
        assert SessionStatus.FAILED == "failed"
        assert SessionStatus.CANCELLED == "cancelled"


class TestAgentStatus:
    def test_values(self):
        assert AgentStatus.PENDING == "pending"
        assert AgentStatus.RUNNING == "running"
        assert AgentStatus.COMPLETED == "completed"
        assert AgentStatus.FAILED == "failed"
        assert AgentStatus.SKIPPED == "skipped"


class TestAgentState:
    def test_defaults(self):
        state = AgentState(agent_name="crawler_agent")
        assert state.agent_name == "crawler_agent"
        assert state.status == AgentStatus.PENDING
        assert state.started_at is None
        assert state.completed_at is None
        assert state.input_data is None
        assert state.output_data is None
        assert state.error_message is None


class TestSessionManager:
    @pytest.fixture
    def manager(self):
        return SessionManager()

    def test_create_session(self, manager):
        session_id = manager.create_session()
        assert len(session_id) == 36
        session = manager.get_current_session()
        assert session is not None
        assert session.session_id == session_id
        assert session.status == SessionStatus.CREATED

    def test_create_session_initializes_agents(self, manager):
        manager.create_session()
        session = manager.get_current_session()
        assert set(session.agents.keys()) == {
            "crawler_agent",
            "storage_agent",
            "metrics_agent",
            "insight_agent",
        }
        for agent in session.agents.values():
            assert agent.status == AgentStatus.PENDING

    def test_start_session(self, manager):
        manager.create_session()
        manager.start_session()
        session = manager.get_current_session()
        assert session.status == SessionStatus.RUNNING
        assert session.started_at is not None

    def test_complete_session_success(self, manager):
        manager.create_session()
        manager.start_session()
        manager.complete_session(success=True)
        session = manager.get_current_session()
        assert session.status == SessionStatus.COMPLETED
        assert session.completed_at is not None

    def test_complete_session_failure(self, manager):
        manager.create_session()
        manager.start_session()
        manager.complete_session(success=False, error="Something went wrong")
        session = manager.get_current_session()
        assert session.status == SessionStatus.FAILED
        assert session.error_message == "Something went wrong"

    def test_end_session(self, manager):
        sid = manager.create_session()
        manager.start_session()
        summary = manager.end_session(sid)
        assert summary["session_id"] == sid
        assert summary["status"] == "completed"

    def test_start_agent(self, manager):
        sid = manager.create_session()
        manager.start_session()
        manager.start_agent(sid, "crawler_agent", input_data={"url": "test"})

        session = manager.get_current_session()
        agent = session.agents["crawler_agent"]
        assert agent.status == AgentStatus.RUNNING
        assert agent.started_at is not None
        assert agent.input_data == {"url": "test"}

    def test_complete_agent_success(self, manager):
        sid = manager.create_session()
        manager.start_session()
        manager.start_agent(sid, "crawler_agent")
        manager.complete_agent(sid, "crawler_agent", output_data={"count": 100})

        session = manager.get_current_session()
        agent = session.agents["crawler_agent"]
        assert agent.status == AgentStatus.COMPLETED
        assert agent.output_data == {"count": 100}

    def test_fail_agent(self, manager):
        sid = manager.create_session()
        manager.start_session()
        manager.start_agent(sid, "crawler_agent")
        manager.fail_agent(sid, "crawler_agent", "Timeout")

        session = manager.get_current_session()
        agent = session.agents["crawler_agent"]
        assert agent.status == AgentStatus.FAILED
        assert agent.error_message == "Timeout"

    def test_set_and_get_context(self, manager):
        manager.create_session()
        manager.set_context("key1", "value1")
        assert manager.get_context("key1") == "value1"
        assert manager.get_context("nonexistent", "default") == "default"

    def test_get_context_no_session(self, manager):
        assert manager.get_context("key") is None

    def test_get_agent_output(self, manager):
        sid = manager.create_session()
        manager.start_agent(sid, "crawler_agent")
        manager.complete_agent(sid, "crawler_agent", output_data={"data": [1, 2, 3]})

        assert manager.get_agent_output("crawler_agent") == {"data": [1, 2, 3]}
        assert manager.get_agent_output("nonexistent") is None

    def test_get_agent_output_no_session(self, manager):
        assert manager.get_agent_output("crawler_agent") is None

    def test_get_next_agent(self, manager):
        manager.create_session()
        assert manager.get_next_agent() == "crawler_agent"

        sid = manager.get_current_session().session_id
        manager.start_agent(sid, "crawler_agent")
        manager.complete_agent(sid, "crawler_agent")
        assert manager.get_next_agent() == "storage_agent"

    def test_get_next_agent_all_done(self, manager):
        sid = manager.create_session()
        for agent_name in SessionManager.AGENT_EXECUTION_ORDER:
            manager.start_agent(sid, agent_name)
            manager.complete_agent(sid, agent_name)
        assert manager.get_next_agent() is None

    def test_get_next_agent_no_session(self, manager):
        assert manager.get_next_agent() is None

    def test_get_session_summary(self, manager):
        sid = manager.create_session()
        manager.start_session()
        manager.start_agent(sid, "crawler_agent")
        manager.complete_agent(sid, "crawler_agent", output_data={"count": 50})
        manager.complete_session()

        summary = manager.get_session_summary()
        assert summary["session_id"] == sid
        assert summary["status"] == "completed"
        assert summary["total_duration_seconds"] is not None
        assert summary["agents"]["crawler_agent"]["status"] == "completed"
        assert summary["agents"]["crawler_agent"]["has_output"] is True

    def test_get_session_summary_no_session(self, manager):
        assert manager.get_session_summary() == {}

    def test_get_session_by_id(self, manager):
        sid = manager.create_session()
        session = manager.get_session(sid)
        assert session is not None
        assert session.session_id == sid

    def test_get_nonexistent_session(self, manager):
        assert manager.get_session("nonexistent") is None

    def test_multiple_sessions(self, manager):
        sid1 = manager.create_session()
        sid2 = manager.create_session()
        assert sid1 != sid2
        assert manager.get_current_session().session_id == sid2
        assert manager.get_session(sid1) is not None

    def test_start_agent_no_session(self, manager):
        # Should not raise
        manager.start_agent("fake", "crawler_agent")

    def test_complete_agent_no_session(self, manager):
        # Should not raise
        manager.complete_agent("fake", "crawler_agent")

    def test_start_agent_unknown_name(self, manager):
        sid = manager.create_session()
        # Should not raise for unknown agent name
        manager.start_agent(sid, "unknown_agent")

    def test_complete_session_no_current(self, manager):
        # Should not raise
        manager.complete_session()
