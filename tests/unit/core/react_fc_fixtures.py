"""ReAct function calling 테스트용 가짜 LLM 응답 빌더 (트랙 5-C)

ReAct 루프가 JSON 문자열 파싱 대신 네이티브 function calling을 쓰므로, 가짜 응답도
``message.tool_calls``를 갖춘 모양이어야 한다. 여러 테스트 모듈이 같은 모양을 쓰므로
여기에 모아 둔다. **네트워크 호출은 일어나지 않는다** — acompletion만 대체한다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


def _usage(total: int | None) -> SimpleNamespace | None:
    if total is None:
        return None
    return SimpleNamespace(
        prompt_tokens=total // 2, completion_tokens=total - total // 2, total_tokens=total
    )


def text_reply(content: str, *, total_tokens: int | None = None) -> SimpleNamespace:
    """도구 호출 없이 본문만 돌려주는 응답 (최종 답변 · reflection · 강제 마무리)."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
        usage=_usage(total_tokens),
    )


def json_reply(payload: dict[str, Any], *, total_tokens: int | None = None) -> SimpleNamespace:
    """본문이 JSON인 응답 (self-reflection)."""
    return text_reply(json.dumps(payload, ensure_ascii=False), total_tokens=total_tokens)


def tool_call_reply(
    thought: str,
    action: str,
    action_input: dict[str, Any] | None = None,
    *,
    call_id: str = "call_1",
    total_tokens: int | None = None,
) -> SimpleNamespace:
    """``thought``를 본문으로, ``action``을 tool_call로 내보내는 응답."""
    call = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=action, arguments=json.dumps(action_input or {}, ensure_ascii=False)
        ),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=thought, tool_calls=[call]))],
        usage=_usage(total_tokens),
    )


def reply(payload: dict[str, Any], *, total_tokens: int | None = None) -> SimpleNamespace:
    """``{"thought", "action", "action_input"}``이면 tool_call, 아니면 본문 JSON 응답."""
    if "action" in payload:
        return tool_call_reply(
            payload.get("thought", ""),
            payload["action"],
            payload.get("action_input"),
            total_tokens=total_tokens,
        )
    return json_reply(payload, total_tokens=total_tokens)
