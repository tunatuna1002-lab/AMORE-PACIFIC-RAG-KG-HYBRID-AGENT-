"""
ReAct Self-Reflection Agent
============================
Reasoning + Acting 패턴으로 복잡한(2홉 이상) 질문을 처리한다.

루프
----
1. 모델이 **네이티브 function calling**으로 도구를 고른다 (``tools=``/``tool_choice="auto"``)
2. 도구를 실행하고 관찰(증거 카드 렌더링)을 ``role="tool"`` 메시지로 되돌려 준다
3. 모델이 ``final_answer``를 부르거나 도구 없이 본문만 내면 루프가 끝난다
4. Self-Reflection으로 품질 점수를 매긴다

왜 function calling인가 (트랙 5-C)
----------------------------------
예전에는 프롬프트로 ``{"thought", "action", "action_input"}`` JSON 문자열을 요구하고 그
문자열을 직접 파싱했다. 모델이 설명문을 섞거나 코드블록을 빠뜨리면 파싱이 실패했고, 실패한
스텝은 ``thought``에 원문만 남긴 채 아무 도구도 부르지 않아 루프가 헛돌았다. 도구 스키마는
이미 ``tool_registry``가 만들고 있었고 DecisionMaker는 그것으로 function calling을 쓰는데,
ReAct만 자체 파서를 들고 있어 같은 도구를 두 가지 방식으로 부르고 있었다.

또 대화가 매 스텝 프롬프트 재조립이었기 때문에 관찰이 문자열로 잘려 들어갔다. 이제는
messages에 assistant(tool_calls) → tool(observation)을 그대로 쌓으므로 모델이 관찰 원문을
본다.

한도
----
- ``max_iterations``: 도구 루프 반복 상한 (기존)
- ``max_total_tokens``: **질문당 토큰 예산**. 루프 앞에서 누적 사용량을 보고 넘으면 멈춘다.
  마무리 답변·self-reflection 호출은 답을 내기 위한 것이라 예산으로 막지 않지만, 사용량은
  ``ReActResult.token_usage``에 함께 집계된다. 예산에 걸렸는지는 ``budget_exceeded``에 남는다.

보안
----
- ``ALLOWED_ACTIONS``: 레지스트리 도구 5종 + 루프 제어 2종만 실행한다. 모델이 다른 이름을
  부르면 레지스트리까지 가지 않고 관찰에 ``Security Error``를 남긴다 (읽기 전용 원칙).
- 도구 인자는 레지스트리 정의가 다시 검증한다.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from litellm import acompletion

from src.shared.constants import DEFAULT_MODEL

from .models import ToolResult
from .tool_registry import (
    TOOL_DEFINITIONS,
    TOOL_NAMES,
    TOOL_SEARCH_DOCS,
    function_schemas,
    tool_evidence,
)

logger = logging.getLogger(__name__)


# _format_steps가 마무리 프롬프트에 싣는 관찰 길이 상한
OBSERVATION_CHARS = 1500

# 질문 하나에 허용하는 기본 토큰 예산 (ReAct 루프). 5스텝 × 프롬프트+응답을 넉넉히 덮는다.
DEFAULT_TOKEN_BUDGET = 12000

# 루프 제어 액션 (도구가 아니다): 답을 내거나(final_answer) 추가 조회를 요청한다(refine_search)
FINAL_ANSWER_ACTION = "final_answer"
REFINE_SEARCH_ACTION = "refine_search"

# refine_search가 실행하는 도구 — IRCoT의 "추가 검색"은 문서 검색이다 (트랙 4-A).
REFINE_SEARCH_TOOL = TOOL_SEARCH_DOCS

# Security: 허용된 액션 목록 — 도구는 단일 레지스트리(tool_registry)에서 온다
ALLOWED_ACTIONS: frozenset[str] = frozenset(
    {*TOOL_NAMES, FINAL_ANSWER_ACTION, REFINE_SEARCH_ACTION}
)

# Security: 각 액션별 허용 파라미터 스키마 (도구 파라미터는 레지스트리 정의에서 생성)
ACTION_SCHEMAS: dict[str, dict[str, type]] = {
    **{d.name: d.parameter_types() for d in TOOL_DEFINITIONS},
    FINAL_ANSWER_ACTION: {
        "answer": str,
        "confidence": float,
    },
    REFINE_SEARCH_ACTION: {
        "refined_query": str,
        "reason": str,
        "focus_entities": list,
    },
}

# 루프 제어용 function calling 스키마. 레지스트리 도구가 아니므로 여기서 직접 쓴다
# (레지스트리는 "읽기 전용 조회 도구"만 담는다 — 제어 액션을 섞지 않는다).
CONTROL_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": FINAL_ANSWER_ACTION,
            "description": (
                "관찰에 근거해 최종 답변을 낸다. 답변의 수치는 관찰 카드에서만 가져오고 "
                "문장 끝에 그 카드 id([M-xxxxxx] 등)를 인용한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "한국어 최종 답변"},
                    "confidence": {
                        "type": "number",
                        "description": "0.0~1.0 자기 평가 신뢰도",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": REFINE_SEARCH_ACTION,
            "description": (
                "이전 관찰로는 부족할 때 질의를 다듬어 문서를 추가 검색한다 (IRCoT). "
                "이전 관찰과 새 결과를 함께 돌려준다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "refined_query": {"type": "string", "description": "다듬은 검색 질의"},
                    "reason": {"type": "string", "description": "추가 검색이 필요한 이유"},
                    "focus_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "함께 검색할 엔티티",
                    },
                },
                "required": ["refined_query"],
                "additionalProperties": False,
            },
        },
    },
)


def react_tool_schemas(available: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """모델에 넘길 도구 스키마: 레지스트리 조회 도구 + 루프 제어 2종."""
    return [*function_schemas(available), *CONTROL_TOOL_SCHEMAS]


def validate_action(action: str, action_input: dict[str, Any] | None) -> tuple[bool, str]:
    """
    액션 및 파라미터 검증

    Returns:
        (is_valid, error_message)
    """
    # 1. 허용된 액션인지 확인
    if action not in ALLOWED_ACTIONS:
        return False, f"Action '{action}' is not allowed. Allowed: {list(ALLOWED_ACTIONS)}"

    # 2. action_input 검증
    if action_input is None:
        return True, ""

    # 3. 스키마 검증 (존재하는 경우)
    schema = ACTION_SCHEMAS.get(action, {})
    for key, value in action_input.items():
        # 허용되지 않은 파라미터 확인
        if key not in schema:
            logger.warning(f"Unknown parameter '{key}' for action '{action}'")
            # 엄격 모드가 아니므로 warning만
            continue

        # 타입 검증
        expected_type = schema[key]
        if not isinstance(value, expected_type):
            return (
                False,
                f"Parameter '{key}' must be {expected_type.__name__}, got {type(value).__name__}",
            )

    return True, ""


@dataclass
class _RunState:
    """실행 한 번의 누적값. 에이전트 인스턴스는 Brain 수명 동안 공유되므로
    토큰 사용량·증거 카드를 인스턴스에 두면 동시 질의끼리 섞인다."""

    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    cards: list[Any] = field(default_factory=list)


@dataclass
class ReActStep:
    """ReAct 단계 기록"""

    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    reflection: str | None = None


@dataclass
class ReActResult:
    """ReAct 실행 결과"""

    final_answer: str
    steps: list[ReActStep] = field(default_factory=list)
    iterations: int = 0
    confidence: float = 0.0
    needs_improvement: bool = False
    hop_count: int = 0
    # 도구 관찰에서 모은 출처 라벨. ``Response.sources``·``BrainChatResponse.sources``가
    # ``list[str]``이므로 여기서도 문자열 목록이다 (트랙 5-C: 예전에는 그래프가 rag_docs
    # dict 목록을 그대로 실어 API 응답 모델에서 검증 오류가 났다).
    sources: list[str] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    budget_exceeded: bool = False


class ReActAgent:
    """ReAct Self-Reflection Agent (네이티브 function calling)"""

    SYSTEM_PROMPT = """당신은 Amazon 마켓 분석 시스템의 분석 에이전트입니다.

제공된 도구는 모두 읽기 전용 조회입니다. 다음 원칙을 지키세요:

1. 복잡한 질문은 단계로 나눠 푸세요. 엔티티 확정 → 관계 조회 → 수치 조회 → 판정 순서가
   기본입니다. 한 번에 한 도구만 부르고, 관찰을 본 뒤 다음 도구를 고르세요.
2. 관찰에는 근거 카드 id가 [M-xxxxxx]처럼 붙습니다. 최종 답변의 수치는 그 카드에서만
   가져오고 문장 끝에 같은 id를 인용하세요.
3. 근거가 부족하면 지어내지 말고 refine_search로 추가 검색하거나, 확인되지 않았다고 쓰세요.
4. 답할 준비가 되면 final_answer를 부르세요."""

    REFLECTION_PROMPT = """## 자체 평가
다음 응답의 품질을 평가하세요:

질문: {query}
응답: {answer}

평가 기준:
1. 질문에 완전히 답변했는가?
2. 누락된 중요 정보가 있는가?
3. 데이터/근거가 충분한가?

JSON으로 응답:
```json
{{
    "quality_score": 0.0-1.0,
    "missing_info": ["누락1", "누락2"],
    "needs_improvement": true/false,
    "improvement_suggestion": "개선 방향"
}}
```"""

    # IRCoT: 추가 검색이 필요함을 나타내는 키워드
    IRCOT_KEYWORDS: tuple[str, ...] = (
        "정보 부족",
        "확인 필요",
        "추가 검색",
        "더 알아",
        "need more",
        "insufficient",
        "unknown",
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 5,
        min_confidence: float = 0.7,
        max_hops: int = 2,
        ircot_enabled: bool = True,
        max_total_tokens: int = DEFAULT_TOKEN_BUDGET,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence
        self.max_hops = max_hops
        self.ircot_enabled = ircot_enabled
        self.max_total_tokens = max_total_tokens
        self.tool_executor = None  # 외부 주입

    def set_tool_executor(self, executor) -> None:
        """도구 실행기 설정"""
        self.tool_executor = executor

    # ── 스키마 ───────────────────────────────────────────────────────

    def tool_schemas(self) -> list[dict[str, Any]]:
        """이번 실행에서 모델에 광고할 도구. 실행기가 못 부르는 도구는 빼지 않으면
        모델이 존재하지 않는 근거를 기대한다."""
        available: Sequence[str] | None = None
        getter = getattr(self.tool_executor, "get_available_tools", None)
        if callable(getter):
            try:
                available = [str(name) for name in getter()]
            except Exception:  # 실행기가 목록을 못 내도 루프는 돌아야 한다
                logger.warning("tool executor could not list tools", exc_info=True)
        return react_tool_schemas(available)

    # ── 루프 ─────────────────────────────────────────────────────────

    async def run(
        self, query: str, context: str, initial_data: dict[str, Any] | None = None
    ) -> ReActResult:
        """ReAct 루프 실행"""
        steps: list[ReActStep] = []
        iterations = 0
        final_answer = ""
        hop_count = 0
        budget_exceeded = False
        run = _RunState()

        tools = self.tool_schemas()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"## 컨텍스트\n{context}\n\n## 질문\n{query}"},
        ]

        while iterations < self.max_iterations:
            if run.usage["total_tokens"] >= self.max_total_tokens:
                budget_exceeded = True
                logger.info(
                    f"ReAct token budget exhausted: "
                    f"{run.usage['total_tokens']}/{self.max_total_tokens}"
                )
                break

            iterations += 1

            content, call = await self._next_move(messages, tools, run)
            step = ReActStep(
                thought=content,
                action=call[0] if call else None,
                action_input=call[1] if call else None,
            )
            steps.append(step)

            # 1. 도구 호출이 없다 — 본문이 곧 답이다 (IRCoT 조건이면 추가 검색으로 돌린다)
            if call is None:
                if (
                    self.ircot_enabled
                    and content
                    and self._needs_retrieval(content)
                    and hop_count < self.max_hops
                ):
                    step.action = REFINE_SEARCH_ACTION
                    step.action_input = {"refined_query": content, "reason": "IRCoT auto-inject"}
                elif content:
                    final_answer = content
                    break
                else:
                    messages.append(
                        {"role": "user", "content": "도구를 부르거나 final_answer로 답하세요."}
                    )
                    continue

            # 2. final_answer — 답은 인자로 온다
            if step.action == FINAL_ANSWER_ACTION:
                final_answer = str((step.action_input or {}).get("answer") or "").strip()
                if final_answer:
                    break
                step.observation = "Error: final_answer에는 answer 인자가 필요합니다"
                self._append_exchange(messages, step)
                continue

            # 3. IRCoT: thought이 "정보 부족"이면 고른 도구를 추가 검색으로 돌린다
            if (
                self.ircot_enabled
                and step.action != REFINE_SEARCH_ACTION
                and step.thought
                and self._needs_retrieval(step.thought)
                and hop_count < self.max_hops
            ):
                logger.info("IRCoT: auto-injecting refine_search based on thought")
                step.action = REFINE_SEARCH_ACTION
                step.action_input = {
                    "refined_query": step.thought,
                    "reason": "IRCoT auto-inject: information insufficient",
                    "focus_entities": [],
                }

            # 4. 실행 (Security: 허용 액션·파라미터 검증 후)
            if not self.tool_executor:
                step.observation = "Error: 도구 실행기가 연결되지 않았습니다"
                self._append_exchange(messages, step)
                continue

            is_valid, error_msg = validate_action(step.action, step.action_input)
            if not is_valid:
                logger.warning(f"Action validation failed: {error_msg}")
                step.observation = f"Security Error: {error_msg}"
                self._append_exchange(messages, step)
                continue

            if step.action == REFINE_SEARCH_ACTION and hop_count < self.max_hops:
                hop_count += 1
                step.observation = await self._execute_multihop_search(step, steps, query, run)
            else:
                step.observation = await self._execute_tool(
                    step.action, step.action_input or {}, run
                )

            self._append_exchange(messages, step)

        # 5. 반복 한도·예산에 걸려 final_answer가 없으면 관찰을 근거로 한 번 더 요청한다
        if not final_answer:
            final_answer = await self._force_final_answer(query, context, steps, run)

        # 6. Self-Reflection
        reflection_result = await self._reflect(query, final_answer, run)

        return ReActResult(
            final_answer=final_answer,
            steps=steps,
            iterations=iterations,
            confidence=reflection_result.get("quality_score", 0.5),
            needs_improvement=reflection_result.get("needs_improvement", False),
            hop_count=hop_count,
            sources=self._collected_sources(run),
            token_usage=dict(run.usage),
            budget_exceeded=budget_exceeded,
        )

    # ── LLM 호출 ─────────────────────────────────────────────────────

    async def _next_move(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        run: _RunState | None = None,
    ) -> tuple[str, tuple[str, dict[str, Any], str] | None]:
        """한 스텝의 모델 응답 → (본문, (도구명, 인자, call_id) | None)."""
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=500,
                temperature=0.2,
            )
        except Exception as e:
            logger.error(f"ReAct step failed: {e}")
            return f"Error: {e}", None

        self._account(response, run)
        message = response.choices[0].message
        content = (getattr(message, "content", None) or "").strip()
        calls = list(getattr(message, "tool_calls", None) or [])
        if not calls:
            return content, None

        if len(calls) > 1:
            # 한 번에 한 도구만 실행한다 — 관찰을 보고 다음 도구를 고르는 게 ReAct다
            logger.info(f"ReAct got {len(calls)} tool calls; using the first one")

        call = calls[0]
        function = getattr(call, "function", None)
        name = getattr(function, "name", None) or ""
        raw_arguments = getattr(function, "arguments", None) or "{}"
        call_id = str(getattr(call, "id", None) or "call_0")

        try:
            arguments = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning(f"tool_call arguments are not valid JSON: {raw_arguments!r}")
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        return content, (name, arguments, call_id)

    @staticmethod
    def _account(response: Any, run: _RunState | None) -> None:
        """토큰 사용량 집계. 값이 숫자가 아니면(usage를 주지 않는 응답 등) 0으로 센다."""
        if run is None:
            return
        usage = getattr(response, "usage", None)
        for key in run.usage:
            value = getattr(usage, key, None)
            if isinstance(value, int | float) and not isinstance(value, bool):
                run.usage[key] += int(value)

    @staticmethod
    def _append_exchange(messages: list[dict[str, Any]], step: ReActStep) -> None:
        """assistant(tool_calls) → tool(observation)을 대화에 쌓는다.

        관찰을 프롬프트 문자열로 다시 붙이지 않는다 — 모델이 도구 결과를 원문으로 본다.
        """
        call_id = str((step.action_input or {}).get("__call_id") or f"call_{len(messages)}")
        arguments = {k: v for k, v in (step.action_input or {}).items() if k != "__call_id"}
        messages.append(
            {
                "role": "assistant",
                "content": step.thought or None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": step.action,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": step.action,
                "content": step.observation or "",
            }
        )

    # ── 도구 ─────────────────────────────────────────────────────────

    async def _execute_tool(self, action: str, action_input: dict[str, Any], run: _RunState) -> str:
        """도구 하나 실행 → 관찰 문자열. 증거 카드는 출처 집계용으로 모아 둔다."""
        try:
            result = await self.tool_executor.execute(action, action_input)
        except Exception as e:
            return f"Error: {e}"
        self._collect_cards(result, run)
        return str(result.data) if result.success else (result.error or "")

    @staticmethod
    def _collect_cards(result: ToolResult, run: _RunState) -> None:
        try:
            run.cards.extend(tool_evidence(result))
        except Exception:  # 카드가 아닌 결과(테스트용 가짜 도구 등)는 그냥 건너뛴다
            logger.debug("tool result carried no evidence cards", exc_info=True)

    @staticmethod
    def _collected_sources(run: _RunState) -> list[str]:
        if not run.cards:
            return []
        from src.rag.evidence_assembly import evidence_source_labels

        return evidence_source_labels(run.cards)

    def _needs_retrieval(self, thought: str) -> bool:
        """IRCoT: thought에서 추가 검색 필요 여부 판단"""
        thought_lower = thought.lower()
        return any(keyword in thought_lower for keyword in self.IRCOT_KEYWORDS)

    async def _execute_multihop_search(
        self,
        current_step: ReActStep,
        all_steps: list[ReActStep],
        original_query: str,
        run: _RunState,
    ) -> str:
        """Multi-hop refine_search 실행: 이전 관찰과 새 검색 결과를 결합"""
        action_input = current_step.action_input or {}
        refined_query = action_input.get("refined_query", original_query)
        focus_entities = action_input.get("focus_entities", [])

        previous_observations = [
            step.observation for step in all_steps if step.observation and step is not current_step
        ]

        # refine_search는 문서 검색(search_docs)으로 실행한다 — IRCoT의 "추가 검색" (트랙 4-A)
        query_parts = [str(refined_query)] + [str(e) for e in focus_entities]
        search_params: dict[str, Any] = {"query": " ".join(p for p in query_parts if p)}
        new_observation = await self._execute_tool(REFINE_SEARCH_TOOL, search_params, run)

        combined_parts = []
        if previous_observations:
            combined_parts.append(f"[Hop 이전 결과] {'; '.join(previous_observations[:3])}")
        combined_parts.append(f"[Hop 새 결과] {new_observation}")

        return " | ".join(combined_parts)

    # ── 마무리 ───────────────────────────────────────────────────────

    async def _force_final_answer(
        self, query: str, context: str, steps: list[ReActStep], run: _RunState | None = None
    ) -> str:
        """final_answer 없이 루프가 끝났을 때 지금까지의 관찰로 최종 답을 만든다."""
        prompt = (
            "지금까지의 컨텍스트와 도구 관찰만 근거로 질문에 한국어로 답하세요. "
            "근거가 없는 내용은 추측하지 말고 확인되지 않았다고 쓰세요.\n\n"
            f"## 컨텍스트\n{context}\n\n## 질문\n{query}\n\n"
            f"## 단계\n{self._format_steps(steps) or '없음'}"
        )
        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700,
                temperature=0.2,
            )
            self._account(response, run)
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"ReAct forced final answer failed: {e}")
            return ""

    async def _reflect(
        self, query: str, answer: str, run: _RunState | None = None
    ) -> dict[str, Any]:
        """Self-Reflection 실행"""
        prompt = self.REFLECTION_PROMPT.format(query=query, answer=answer)

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            self._account(response, run)

            content = response.choices[0].message.content
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0:
                return json.loads(content[json_start:json_end])
        except Exception as e:
            logger.error(f"Reflection failed: {e}")

        return {"quality_score": 0.5, "needs_improvement": False}

    def _format_steps(self, steps: list[ReActStep]) -> str:
        """마무리 프롬프트용 단계 요약"""
        if not steps:
            return ""

        lines = []
        for i, step in enumerate(steps, 1):
            lines.append(f"### Step {i}")
            lines.append(f"Thought: {step.thought}")
            if step.action:
                lines.append(f"Action: {step.action}")
            if step.observation:
                # 관찰이 잘리면 LLM이 최종 답에 수치를 옮길 수 없다
                observation = step.observation
                if len(observation) > OBSERVATION_CHARS:
                    observation = observation[:OBSERVATION_CHARS] + "..."
                lines.append(f"Observation: {observation}")

        return "\n".join(lines)


# 싱글톤
_react_agent: ReActAgent | None = None


def get_react_agent() -> ReActAgent:
    """ReActAgent 싱글톤"""
    global _react_agent
    if _react_agent is None:
        _react_agent = ReActAgent()
    return _react_agent
