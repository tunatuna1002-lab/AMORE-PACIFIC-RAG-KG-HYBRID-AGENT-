"""
F7: the v4 chat path carries the session's previous turns
=========================================================
Before this, ``/api/v4/chat`` and ``/api/v4/chat/stream`` wrote nothing to the session
memory and read nothing from it, so every question was answered as if it were the first
one (defect D15: the memory had writers on the v1 route only and no readers at all).

Now both endpoints hand ``get_recent_turns(session_id)`` to the brain and record the
turn afterwards, and ``ResponsePipeline`` turns that list into real chat messages placed
right before the current question.

Only public entry points are used: the FastAPI app through ``TestClient``, the module
functions re-exported by ``src.api.dependencies``, and ``ResponsePipeline._build_messages``
via its public ``generate``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.core.models import Context, Response
from src.core.response_pipeline import ResponsePipeline

API_KEY = "test-api-key-12345"  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# ResponsePipeline: history becomes LLM messages
# ---------------------------------------------------------------------------


def _ctx(query: str) -> Context:
    return Context(query=query, entities={}, rag_docs=[], kg_facts=[])


def test_history_is_rendered_as_chat_messages_before_the_question() -> None:
    pipeline = ResponsePipeline()
    messages = pipeline._build_messages(
        "그럼 2위는?",
        _ctx("그럼 2위는?"),
        conversation_history=[
            {"role": "user", "content": "LANEIGE 순위 알려줘"},
            {"role": "assistant", "content": "1위입니다"},
        ],
    )

    roles_and_text = [(m["role"], m["content"]) for m in messages]
    assert roles_and_text[-3:] == [
        ("user", "LANEIGE 순위 알려줘"),
        ("assistant", "1위입니다"),
        ("user", "그럼 2위는?"),
    ]


def test_no_history_leaves_the_message_list_unchanged() -> None:
    pipeline = ResponsePipeline()
    without = pipeline._build_messages("질문", _ctx("질문"))
    with_empty = pipeline._build_messages("질문", _ctx("질문"), conversation_history=[])
    assert without == with_empty
    assert without[-1] == {"role": "user", "content": "질문"}


def test_history_is_capped_and_blank_turns_are_dropped() -> None:
    pipeline = ResponsePipeline()
    history: list[dict[str, str]] = [{"role": "user", "content": "   "}]
    history += [{"role": "user", "content": f"m{i}"} for i in range(10)]

    messages = pipeline._build_messages(
        "이번 질문", _ctx("이번 질문"), conversation_history=history
    )
    carried = [m["content"] for m in messages if m["role"] in ("user", "assistant")][:-1]

    assert len(carried) == ResponsePipeline.MAX_HISTORY_TURNS
    assert carried == [f"m{i}" for i in range(4, 10)]  # 최근 6턴, 공백 턴 제외


def test_unknown_roles_are_folded_into_assistant() -> None:
    pipeline = ResponsePipeline()
    messages = pipeline._build_messages(
        "질문",
        _ctx("질문"),
        conversation_history=[{"role": "tool", "content": "도구 출력"}],
    )
    assert messages[-2] == {"role": "assistant", "content": "도구 출력"}


# ---------------------------------------------------------------------------
# API routes: history in, turn recorded out
# ---------------------------------------------------------------------------


class _RecordingBrain:
    """Minimal stand-in for UnifiedBrain (only what the chat routes call)."""

    class _Mode:
        value = "responding"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.mode = self._Mode()

    async def process_query(self, **kwargs: Any) -> Response:
        self.calls.append(kwargs)
        return Response(text="ANSWER", confidence_score=0.9)

    async def process_query_stream(self, **kwargs: Any):
        self.calls.append(kwargs)
        yield {"type": "text", "content": "AN"}
        yield {"type": "text", "content": "SWER"}
        yield {"type": "done", "content": {}}


@pytest.fixture
def brain(monkeypatch: pytest.MonkeyPatch) -> _RecordingBrain:
    """Inject the fake at the one boundary the routes resolve the brain through."""
    import src.api.routes.chat as chat_route

    recording = _RecordingBrain()

    async def _get_initialized_brain():
        return recording

    monkeypatch.setattr(chat_route, "get_initialized_brain", _get_initialized_brain)
    return recording


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("API_KEY", API_KEY)
    deps.conversation_memory.clear()
    from src.api.dashboard_api import app

    # base_url must be http://localhost - TrustedHostMiddleware rejects "testserver".
    return TestClient(app, base_url="http://localhost")


def _post(client: TestClient, path: str, message: str, session_id: str):
    return client.post(
        path,
        json={"message": message, "session_id": session_id},
        headers={"X-API-Key": API_KEY},
    )


def test_v4_chat_records_the_turn_and_replays_it_on_the_next_question(
    client: TestClient, brain: _RecordingBrain
) -> None:
    assert _post(client, "/api/v4/chat", "LANEIGE 순위 알려줘", "s1").status_code == 200

    # 첫 질문에는 이력이 없다
    assert brain.calls[0]["conversation_history"] == []

    assert _post(client, "/api/v4/chat", "그럼 2위는?", "s1").status_code == 200
    assert brain.calls[1]["conversation_history"] == [
        {"role": "user", "content": "LANEIGE 순위 알려줘"},
        {"role": "assistant", "content": "ANSWER"},
    ]


def test_v4_chat_sessions_do_not_leak_into_each_other(
    client: TestClient, brain: _RecordingBrain
) -> None:
    _post(client, "/api/v4/chat", "세션 A 질문", "a")
    _post(client, "/api/v4/chat", "세션 B 질문", "b")
    assert brain.calls[1]["conversation_history"] == []


def test_v4_stream_passes_history_and_records_the_streamed_answer(
    client: TestClient, brain: _RecordingBrain
) -> None:
    first = _post(client, "/api/v4/chat/stream", "LANEIGE 순위 알려줘", "s2")
    assert first.status_code == 200
    assert "AN" in first.text and "SWER" in first.text
    assert brain.calls[0]["conversation_history"] == []

    second = _post(client, "/api/v4/chat/stream", "그럼 2위는?", "s2")
    assert second.status_code == 200
    # 스트림으로 흘러간 토큰이 합쳐져 한 턴으로 기록된다
    assert brain.calls[1]["conversation_history"] == [
        {"role": "user", "content": "LANEIGE 순위 알려줘"},
        {"role": "assistant", "content": "ANSWER"},
    ]


def test_clearing_the_session_drops_the_history(client: TestClient, brain: _RecordingBrain) -> None:
    _post(client, "/api/v4/chat", "첫 질문", "s3")
    assert client.delete("/api/chat/memory/s3", headers={"X-API-Key": API_KEY}).status_code == 200

    _post(client, "/api/v4/chat", "두 번째 질문", "s3")
    assert brain.calls[-1]["conversation_history"] == []


def test_stream_still_emits_sse_frames(client: TestClient, brain: _RecordingBrain) -> None:
    body = _post(client, "/api/v4/chat/stream", "질문", "s4").text
    frames = [
        json.loads(line[len("data: ") :]) for line in body.splitlines() if line.startswith("data: ")
    ]
    assert [f["type"] for f in frames] == ["text", "text", "done"]
