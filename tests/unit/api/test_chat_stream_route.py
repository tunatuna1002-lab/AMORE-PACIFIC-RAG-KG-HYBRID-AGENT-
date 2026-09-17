"""`/api/v4/chat/stream`의 SSE 프레이밍 고정 테스트 (트랙 5-A).

대시보드는 응답 본문을 `\\n`으로 쪼개 `data: ` 접두사가 붙은 줄만 JSON으로 파싱한다
(`dashboard/amore_unified_dashboard_v4.html`). 그 계약을 라우트 수준에서 못박는다.
Brain 자체는 `tests/unit/core/test_stream_graph_parity.py`가 검증하므로 여기서는
Brain만 가짜로 두고 라우트의 직렬화·헤더만 본다.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.api.models import BrainChatRequest
from src.api.routes import chat as chat_routes

# UnifiedBrain.process_query_stream이 실제로 내보내는 이벤트 모양 그대로
EVENTS: list[dict[str, Any]] = [
    {"type": "status", "content": "컨텍스트 수집 중..."},
    {"type": "status", "content": "분석 중..."},
    {"type": "tool_call", "content": {"name": "get_metrics", "status": "calling"}},
    {"type": "status", "content": "응답 생성 중..."},
    {"type": "text", "content": "LANEIGE의 Lip Care SoS는 2%입니다."},
    {
        "type": "done",
        "content": {
            "confidence": 0.9,
            "sources": ["sqlite:brand_metrics (2026-08-31)"],
            "tools_used": ["get_metrics"],
            "suggestions": ["경쟁사 비교해줘"],
            "processing_time_ms": 12.3,
            "mode": "direct",
            "confidence_level": "high",
            "metadata": {"route_trace": {"route": "decide", "confidence_level": "high"}},
        },
    },
]


class _FakeBrain:
    """process_query_stream만 흉내 내는 Brain 대역."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.calls: list[dict[str, Any]] = []

    async def process_query_stream(self, **kwargs: Any):
        self.calls.append(kwargs)
        for event in self.events:
            yield event


async def _collect(body: BrainChatRequest, brain: Any) -> tuple[str, Any]:
    """라우트를 돌려 (SSE 본문 전체, StreamingResponse)를 돌려준다."""
    with (
        patch.object(chat_routes, "get_initialized_brain", AsyncMock(return_value=brain)),
        patch.object(chat_routes, "load_dashboard_data", lambda: {"brand": {}}),
    ):
        # 데코레이터(rate limit)는 Request 객체를 요구하므로 원 함수를 직접 부른다
        response = await chat_routes.chat_v4_stream.__wrapped__(request=None, body=body)
        chunks = [chunk async for chunk in response.body_iterator]
    return "".join(chunks), response


def _parse(payload: str) -> list[dict[str, Any]]:
    """대시보드와 같은 방식으로 파싱한다."""
    return [
        json.loads(line[len("data: ") :])
        for line in payload.split("\n")
        if line.startswith("data: ")
    ]


@pytest.mark.asyncio
async def test_stream_route_frames_every_event_as_sse_data_line():
    brain = _FakeBrain(EVENTS)
    payload, response = await _collect(BrainChatRequest(message="LANEIGE 점유율"), brain)

    assert response.media_type == "text/event-stream"
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers["X-Accel-Buffering"] == "no"

    # 이벤트마다 정확히 "data: {json}\n\n" 한 덩어리
    assert payload.count("data: ") == len(EVENTS)
    assert payload.endswith("\n\n")
    assert _parse(payload) == EVENTS


@pytest.mark.asyncio
async def test_stream_route_passes_session_and_metrics_to_brain():
    brain = _FakeBrain(EVENTS)
    await _collect(BrainChatRequest(message="LANEIGE 점유율", session_id="s1"), brain)

    assert brain.calls == [
        {"query": "LANEIGE 점유율", "session_id": "s1", "current_metrics": {"brand": {}}}
    ]


@pytest.mark.asyncio
async def test_stream_route_keeps_korean_text_unescaped():
    """ensure_ascii=False — 대시보드가 그대로 렌더한다."""
    brain = _FakeBrain([{"type": "text", "content": "라네즈 점유율"}])
    payload, _ = await _collect(BrainChatRequest(message="점유율"), brain)

    assert "라네즈 점유율" in payload


@pytest.mark.asyncio
async def test_stream_route_emits_error_event_when_brain_raises():
    class _BoomBrain:
        async def process_query_stream(self, **_: Any):
            yield {"type": "status", "content": "컨텍스트 수집 중..."}
            raise RuntimeError("brain down")

    payload, _ = await _collect(BrainChatRequest(message="점유율"), _BoomBrain())

    events = _parse(payload)
    assert events[-1] == {"type": "error", "content": "brain down"}


@pytest.mark.asyncio
async def test_stream_route_rejects_empty_message():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await chat_routes.chat_v4_stream.__wrapped__(
            request=None, body=BrainChatRequest(message="   ")
        )
    assert exc.value.status_code == 400
