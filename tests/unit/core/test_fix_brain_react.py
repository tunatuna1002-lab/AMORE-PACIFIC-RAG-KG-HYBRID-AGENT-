"""
D3: UnifiedBrain ReAct agent wiring
====================================
brain.py imported ``get_react_agent`` from a non-existent module
(``src.agents.react_agent``) and swallowed the ImportError at DEBUG level,
so the ReAct agent was never available at runtime.

Fix contract:
- ``UnifiedBrain._init_react_agent()`` reads ``ENABLE_REACT_AGENT`` (default "false").
- When enabled, it imports ``get_react_agent`` from ``src.core.react_agent`` and
  wires the brain's tool executor into it.
- When disabled, no import is attempted and nothing is logged at WARNING.
- If the import/construction fails while enabled, a WARNING is logged.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.brain import UnifiedBrain, reset_brain
from src.core.models import Context, Response, ToolResult


class FakeReActAgent:
    def __init__(self) -> None:
        self.tool_executor = None

    def set_tool_executor(self, executor) -> None:
        self.tool_executor = executor


@pytest.fixture(autouse=True)
def _reset_brain():
    yield
    reset_brain()


@pytest.fixture
def brain() -> UnifiedBrain:
    gatherer = MagicMock()
    gatherer.initialize = AsyncMock()
    gatherer.gather = AsyncMock(
        return_value=Context(query="q", entities={}, rag_docs=[], kg_facts=[], summary="")
    )
    tool_executor = MagicMock()
    tool_executor.execute = AsyncMock(return_value=ToolResult(tool_name="t", success=True, data={}))
    pipeline = MagicMock()
    pipeline.generate = AsyncMock(return_value=Response(text="ok", confidence_score=0.5))
    return UnifiedBrain(
        context_gatherer=gatherer, tool_executor=tool_executor, response_pipeline=pipeline
    )


def test_react_agent_enabled_by_env_flag(monkeypatch: pytest.MonkeyPatch, brain: UnifiedBrain):
    """ENABLE_REACT_AGENT=true -> react agent is constructed from src.core.react_agent."""
    fake = FakeReActAgent()
    monkeypatch.setenv("ENABLE_REACT_AGENT", "true")
    monkeypatch.setattr("src.core.react_agent.get_react_agent", lambda: fake)

    brain._init_react_agent()

    assert brain._react_agent is fake
    assert fake.tool_executor is brain._tool_executor


def test_react_agent_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch, brain: UnifiedBrain, caplog: pytest.LogCaptureFixture
):
    """Without the flag the agent stays None and nothing is logged at WARNING+."""
    monkeypatch.delenv("ENABLE_REACT_AGENT", raising=False)

    def _boom() -> None:  # must never be called when disabled
        raise AssertionError("get_react_agent must not be called when disabled")

    monkeypatch.setattr("src.core.react_agent.get_react_agent", _boom)

    with caplog.at_level(logging.WARNING, logger="src.core.brain"):
        brain._init_react_agent()

    assert brain._react_agent is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_react_agent_explicit_false(monkeypatch: pytest.MonkeyPatch, brain: UnifiedBrain):
    monkeypatch.setenv("ENABLE_REACT_AGENT", "false")
    monkeypatch.setattr("src.core.react_agent.get_react_agent", lambda: FakeReActAgent())

    brain._init_react_agent()

    assert brain._react_agent is None


def test_react_agent_failure_logs_warning(
    monkeypatch: pytest.MonkeyPatch, brain: UnifiedBrain, caplog: pytest.LogCaptureFixture
):
    """A failure while enabled must surface at WARNING, not be hidden at DEBUG."""
    monkeypatch.setenv("ENABLE_REACT_AGENT", "true")

    def _boom() -> None:
        raise ImportError("simulated import failure")

    monkeypatch.setattr("src.core.react_agent.get_react_agent", _boom)

    with caplog.at_level(logging.DEBUG, logger="src.core.brain"):
        brain._init_react_agent()

    assert brain._react_agent is None
    warnings = [
        r for r in caplog.records if r.levelno >= logging.WARNING and "ReAct" in r.getMessage()
    ]
    assert warnings, "expected a WARNING about ReAct agent initialization failure"


@pytest.mark.asyncio
async def test_initialize_wires_react_agent_into_query_graph(
    monkeypatch: pytest.MonkeyPatch, brain: UnifiedBrain, tmp_path
):
    """Full initialize(): with the flag on, the QueryGraph receives the react agent."""
    fake = FakeReActAgent()
    monkeypatch.setenv("ENABLE_REACT_AGENT", "true")
    monkeypatch.setenv("DASHBOARD_DATA_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr("src.core.react_agent.get_react_agent", lambda: fake)
    monkeypatch.setattr("src.agents.alert_agent.AlertAgent", lambda *a, **k: MagicMock())
    brain._alert_manager = MagicMock(initialize=AsyncMock())

    await brain.initialize()

    assert brain._react_agent is fake
    assert brain._query_graph is not None
    assert brain._query_graph._react_agent is fake
