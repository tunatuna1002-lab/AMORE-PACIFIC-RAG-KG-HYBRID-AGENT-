"""
ReAct 도구 실행기 어댑터
========================
ReAct 루프는 ``execute(action, action_input) -> ToolResult``와 ``get_available_tools()``를
쓰고, 관찰을 ``str(result.data)``로 만든다. 이 모듈은 그 인터페이스를 단일 도구 레지스트리
(``src/core/tool_registry.py``)에 연결한다 (트랙 4-A).

왜 어댑터인가
-------------
예전에는 ReAct 전용 도구 3종(query_data·query_knowledge_graph·calculate_metrics)을 별도
``ToolExecutor``에 등록해, DecisionMaker가 쓰는 도구 목록과 갈라져 있었다. 같은 질문에
대해 경로에 따라 다른 도구·다른 근거가 쓰였고 어느 쪽이 답변의 근거였는지 추적되지 않았다.
이제 두 경로가 같은 레지스트리 객체를 본다.

관찰 형식
---------
관찰은 증거 카드를 프롬프트와 같은 형식(``[id] 내용 (as_of, source)``)으로 렌더링한 문자열이다.
ReAct 루프가 ``str(result.data)``를 쓰므로, ``data``는 dict이면서 문자열로 바꾸면 그 렌더링이
되도록 ``_Observation``(dict 하위 클래스)로 싣는다. 카드 id가 관찰에 남아야 최종 답변이
같은 id로 인용할 수 있다. (ReAct를 function calling으로 바꾸는 일은 트랙 5-C다.)
"""

from __future__ import annotations

import logging
from typing import Any

from .models import ToolResult
from .tool_registry import EVIDENCE_KEY, ToolRegistry, render_tool_observation

logger = logging.getLogger(__name__)


class _Observation(dict):
    """``str()``이 카드 렌더링이 되는 도구 결과 dict."""

    def __init__(self, data: dict[str, Any], rendered: str) -> None:
        super().__init__(data)
        self._rendered = rendered

    def __str__(self) -> str:  # ReAct 루프: step.observation = str(result.data)
        return self._rendered


class ReActToolExecutor:
    """ReAct 루프 ↔ ToolRegistry 어댑터."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    # ── ReAct 루프가 쓰는 인터페이스 ─────────────────────────────

    def get_available_tools(self) -> list[str]:
        return self.registry.get_available_tools()

    def is_tool_available(self, tool_name: str) -> bool:
        return self.registry.is_tool_available(tool_name)

    async def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        result = await self.registry.execute(tool_name, params)
        if not result.success:
            return result
        data = dict(result.data or {})
        cards = len(data.get(EVIDENCE_KEY) or [])
        logger.debug(f"ReAct tool {tool_name}: {cards} evidence cards")
        return ToolResult(
            tool_name=result.tool_name,
            success=True,
            data=_Observation(data, render_tool_observation(result)),
            execution_time_ms=result.execution_time_ms,
        )


def build_react_tool_executor(registry: ToolRegistry) -> ReActToolExecutor:
    """ReAct 실행기를 만든다. 레지스트리는 Brain이 쓰는 것과 같은 객체여야 한다."""
    return ReActToolExecutor(registry)
