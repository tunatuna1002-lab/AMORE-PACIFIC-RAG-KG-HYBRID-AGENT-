"""
Decision Maker - LLM 의사결정 전담
==================================
UnifiedBrain에서 분리된 LLM 기반 의사결정 컴포넌트

책임:
- 질문 분석
- 도구 선택 (네이티브 function calling)
- LLM 호출 및 응답 해석

도구 선택 방식 (트랙 4-A, 설계 E4)
----------------------------------
예전에는 LLM에게 ``{"tool": ..., "tool_params": {...}}`` JSON 문자열을 요구하고 그 문자열을
파싱했다. 모델이 코드블록·설명문을 섞으면 파싱이 실패했고, 실패는 "direct_answer, 신뢰도
0.3" 폴백으로 조용히 흡수돼 도구가 선택되지 않은 이유가 남지 않았다.

이제는 ``tool_registry``의 정의에서 생성한 function calling 스키마를 ``tools=``로 넘기고
``tool_choice="auto"``로 모델이 도구를 고르게 한다. 인자는 API가 JSON으로 돌려주므로
프롬프트 파싱이 없다. 도구 없이 답할 수 있으면 모델은 도구를 부르지 않고 본문만 낸다.

신뢰도
------
function calling에는 모델이 스스로 매기는 신뢰도 필드가 없다. 그래서 ``Decision.confidence``는
기본값 0.0(= 보고되지 않음)으로 둔다. 지어낸 숫자를 넣지 않는다 — 답변 신뢰도는
``ResponsePipeline``이 근거 카드로 계산한다.

관련 Protocol: DecisionMakerProtocol
"""

import json
import logging
from typing import Any

from litellm import acompletion

from src.shared.constants import DEFAULT_MODEL

from .models import Context, Decision
from .tool_registry import function_schemas

logger = logging.getLogger(__name__)

DIRECT_ANSWER = "direct_answer"


class DecisionMaker:
    """
    LLM 기반 의사결정

    모든 판단을 LLM이 담당합니다 (LLM-First).

    Usage:
        decision_maker = DecisionMaker()
        decision = await decision_maker.decide(query, context, system_state)
    """

    DECISION_PROMPT = """당신은 Amazon 마켓 분석 시스템의 자율 에이전트입니다.

## 현재 시스템 상태
{system_state}

## 수집된 컨텍스트
{context_summary}

## 사용자 질문
{query}

## 지시사항
1. 시스템 상태와 컨텍스트를 분석하세요
2. 질문에 답하려면 무엇이 더 필요한지 판단하세요
3. 컨텍스트만으로 답할 수 있으면 도구를 호출하지 말고, 답변에 쓸 핵심 포인트를
   "- "로 시작하는 줄로 적으세요
4. 근거가 부족하면 제공된 도구 중 하나를 호출하세요 (모두 읽기 전용 조회입니다)
5. 도구를 호출할 때는 왜 그 도구가 필요한지 한 줄로 적으세요"""

    MODE_PROMPTS = {
        "high": """
## 모드: HIGH 신뢰도
컨텍스트가 충분합니다. 도구를 호출하지 말고 핵심 포인트만 정리하세요.""",
        "medium": """
## 모드: MEDIUM 신뢰도
컨텍스트가 부분적으로 있습니다. 부족한 근거가 분명할 때만 도구를 호출하세요.""",
        "low": """
## 모드: LOW 신뢰도
컨텍스트가 부족합니다. 적절한 도구를 호출해 근거(수치·관계·문서)를 먼저 확보하세요.""",
        "unknown": """
## 모드: UNKNOWN 신뢰도
질문 의도가 불명확합니다. 도구를 호출하지 말고
명확화를 위해 되물을 내용을 "- " 줄로 적으세요.""",
    }

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.1, max_tokens: int = 500):
        """
        Args:
            model: LLM 모델
            temperature: 생성 온도 (낮을수록 결정적)
            max_tokens: 최대 토큰 수
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._decision_count = 0
        self._tracer = None  # Set externally via set_tracer()

    def set_tracer(self, tracer) -> None:
        """ExecutionTracer 설정"""
        self._tracer = tracer

    async def decide(
        self,
        query: str,
        context: Context,
        system_state: dict[str, Any],
        confidence_level: str = "medium",
    ) -> Decision:
        """
        LLM 기반 의사결정 (네이티브 function calling)

        Args:
            query: 사용자 질문
            context: 수집된 컨텍스트
            system_state: 시스템 상태 (``available_tools``가 호출 가능한 도구 목록)
            confidence_level: 신뢰도 레벨 ("high", "medium", "low", "unknown")

        Returns:
            Decision 객체:
                - tool: 모델이 호출한 도구명 (호출이 없으면 "direct_answer")
                - tool_params: 도구 인자 (function calling이 준 JSON)
                - reason: 모델 본문 (도구 선택 이유 또는 직접 답변 근거)
                - confidence: 0.0 — function calling은 자기보고 신뢰도를 주지 않는다
                - key_points: 본문의 "- " 줄
        """
        self._decision_count += 1
        logger.debug(f"Decision #{self._decision_count} with confidence_level={confidence_level}")

        available = [str(name) for name in system_state.get("available_tools") or []]
        tools = function_schemas(available)

        try:
            prompt = self.DECISION_PROMPT.format(
                system_state=self._format_system_state(system_state),
                context_summary=context.summary or "컨텍스트 없음",
                query=query,
            )
            prompt += self.MODE_PROMPTS.get(confidence_level, self.MODE_PROMPTS["medium"])

            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            # LLM 트레이싱
            if self._tracer and self._tracer.get_current_trace_id():
                with self._tracer.llm_span(
                    "decision_llm",
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ) as span:
                    response = await acompletion(**kwargs)
                    # 토큰 사용량 기록
                    if hasattr(response, "usage") and response.usage:
                        span.attributes["llm.prompt_tokens"] = getattr(
                            response.usage, "prompt_tokens", 0
                        )
                        span.attributes["llm.completion_tokens"] = getattr(
                            response.usage, "completion_tokens", 0
                        )
                        span.attributes["llm.total_tokens"] = getattr(
                            response.usage, "total_tokens", 0
                        )
            else:
                response = await acompletion(**kwargs)

            return self._read_decision(response, available)

        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return self._fallback_decision(str(e))

    # =========================================================================
    # 응답 해석
    # =========================================================================

    def _read_decision(self, response: Any, available: list[str]) -> Decision:
        """function calling 응답 → Decision."""
        message = response.choices[0].message
        content = (getattr(message, "content", None) or "").strip()
        tool_calls = list(getattr(message, "tool_calls", None) or [])

        if not tool_calls:
            return Decision(
                tool=DIRECT_ANSWER,
                tool_params={},
                reason=content or "컨텍스트로 응답",
                key_points=self._key_points(content),
                confidence=0.0,
            )

        if len(tool_calls) > 1:
            # Decision은 도구 하나를 담는다. 남은 호출은 다음 턴에 다시 요청될 수 있다.
            logger.info(f"LLM returned {len(tool_calls)} tool calls; using the first one")

        call = tool_calls[0]
        name = getattr(getattr(call, "function", None), "name", None)
        raw_arguments = getattr(getattr(call, "function", None), "arguments", None) or "{}"

        if name not in available:
            return self._fallback_decision(f"호출 불가 도구 선택: {name}")

        try:
            arguments = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning(f"tool_call arguments are not valid JSON: {raw_arguments!r}")
            return self._fallback_decision(f"도구 인자 파싱 실패: {name}")

        if not isinstance(arguments, dict):
            return self._fallback_decision(f"도구 인자가 객체가 아닙니다: {name}")

        return Decision(
            tool=name,
            tool_params=arguments,
            reason=content or f"{name} 호출",
            key_points=self._key_points(content),
            confidence=0.0,
        )

    @staticmethod
    def _key_points(content: str) -> list[str]:
        """본문의 "- " 줄만 핵심 포인트로 쓴다 (형식이 틀리면 포인트가 없을 뿐이다)."""
        points = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "* ")):
                points.append(stripped[2:].strip())
        return [p for p in points if p]

    def _fallback_decision(self, reason: str) -> Decision:
        """폴백 의사결정"""
        return Decision(
            tool=DIRECT_ANSWER,
            tool_params={},
            reason=f"LLM 오류: {reason}",
            confidence=0.3,
            key_points=[],
        )

    def _format_system_state(self, state: dict[str, Any]) -> str:
        """시스템 상태 포맷"""
        lines = [
            f"- 데이터 상태: {state.get('data_status', '알 수 없음')}",
            f"- 동작 모드: {state.get('mode', '알 수 없음')}",
            f"- 사용 가능 도구: {', '.join(state.get('available_tools', []))}",
        ]
        failed_tools = state.get("failed_tools", [])
        if failed_tools:
            lines.append(f"- 실패 도구: {', '.join(failed_tools)}")
        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return {"decision_count": self._decision_count, "model": self.model}
