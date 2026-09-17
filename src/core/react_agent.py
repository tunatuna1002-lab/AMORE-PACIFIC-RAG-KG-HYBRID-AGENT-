"""
ReAct Self-Reflection Agent
============================
Reasoning + Acting 패턴으로 복잡한 질문 처리

ReAct Loop:
1. Thought - 현재 상황 분석
2. Action - 도구 선택
3. Observation - 결과 관찰
4. Reflection - 자체 평가

Security:
- ALLOWED_ACTIONS: 허용된 도구만 실행 가능
- ACTION_SCHEMAS: 각 도구별 파라미터 스키마 검증

Usage:
    agent = ReActAgent()
    result = await agent.run(query, context)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from litellm import acompletion

from src.shared.constants import DEFAULT_MODEL

from .tool_registry import TOOL_DEFINITIONS, TOOL_NAMES, TOOL_SEARCH_DOCS

logger = logging.getLogger(__name__)


# _format_steps가 프롬프트에 싣는 관찰 길이 상한
OBSERVATION_CHARS = 1500

# 루프 제어 액션 (도구가 아니다): 답을 내거나(final_answer) 추가 조회를 요청한다(refine_search)
FINAL_ANSWER_ACTION = "final_answer"
REFINE_SEARCH_ACTION = "refine_search"

# refine_search가 실행하는 도구 — IRCoT의 "추가 검색"은 문서 검색이다 (트랙 4-A).
# 예전에는 지금은 없는 query_data로 실행했다.
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

# 프롬프트에 싣는 도구 목록 (설명·파라미터 서명은 레지스트리 정의가 정본)
AVAILABLE_ACTIONS_TEXT = "\n".join(f"- {d.signature()}: {d.description}" for d in TOOL_DEFINITIONS)


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


class ReActAgent:
    """ReAct Self-Reflection Agent"""

    REACT_PROMPT = """당신은 분석적 사고를 하는 AI입니다.

## 현재 컨텍스트
{context}

## 질문
{query}

## 지금까지의 단계
{steps}

## 지시사항
ReAct 패턴으로 사고하세요:

1. **Thought**: 현재 상황을 분석하세요. 무엇을 알고 있고, 무엇이 필요한가요?
2. **Action**: 필요한 행동을 선택하세요
3. **Action Input**: 행동에 필요한 파라미터 (JSON)

사용 가능한 행동 (모두 읽기 전용):
__AVAILABLE_ACTIONS__
- refine_search: 이전 관찰을 바탕으로 문서를 추가 검색. {{"refined_query": "...", "reason": "..."}}
- final_answer: 최종 답변. {{"answer": "관찰에 근거한 답변", "confidence": 0.0-1.0}}

Action Input은 위 서명에 있는 파라미터만 쓰세요 (모르는 파라미터는 거부됩니다).
관찰에는 근거 카드 id가 [M-xxxxxx]처럼 붙습니다. 최종 답변의 수치는 그 카드에서만 가져오고
문장 끝에 같은 id를 인용하세요.

**Multi-hop 추론**: 복잡한 질문은 여러 단계로 나눠서 해결하세요.
- 1단계: 핵심 엔티티 정보 수집
- 2단계: 관련 엔티티로 확장 (경쟁사, 카테고리 등)
- 3단계: 수집된 정보를 종합하여 최종 답변
`refine_search`를 사용하면 이전 관찰 결과를 바탕으로 검색을 정제할 수 있습니다.

**IRCoT (Interleaved Retrieval Chain-of-Thought)**:
정보가 부족하면 refine_search를 사용하여 추가 검색하세요.
"정보 부족", "확인 필요", "추가 검색" 등의 상황에서는 바로 final_answer하지 말고
refine_search로 추가 정보를 수집한 뒤 답변하세요.

JSON 형식으로 응답:
```json
{{
    "thought": "현재 상황 분석...",
    "action": "action_name",
    "action_input": {{}}
}}
```

"final_answer" action을 선택하면 루프가 종료됩니다.""".replace(
        "__AVAILABLE_ACTIONS__", AVAILABLE_ACTIONS_TEXT
    )

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
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence
        self.max_hops = max_hops
        self.ircot_enabled = ircot_enabled
        self.tool_executor = None  # 외부 주입

    def set_tool_executor(self, executor) -> None:
        """도구 실행기 설정"""
        self.tool_executor = executor

    async def run(
        self, query: str, context: str, initial_data: dict[str, Any] | None = None
    ) -> ReActResult:
        """ReAct 루프 실행"""
        steps: list[ReActStep] = []
        iterations = 0
        final_answer = ""
        hop_count = 0

        while iterations < self.max_iterations:
            iterations += 1

            # 1. ReAct 단계 실행
            step = await self._execute_step(query, context, steps)
            steps.append(step)

            # 2. Final Answer 체크 — 답은 LLM이 action_input.answer에 쓴다.
            # (_parse_step은 observation을 채우지 않으므로 observation에서 읽으면 항상 빈 답)
            if step.action == "final_answer":
                final_answer = str((step.action_input or {}).get("answer") or "").strip()
                if final_answer:
                    break
                step.observation = "Error: final_answer에는 action_input.answer가 필요합니다"
                continue

            # 3. IRCoT: 자동 refine_search 주입
            if (
                self.ircot_enabled
                and step.action != "refine_search"
                and step.thought
                and self._needs_retrieval(step.thought)
                and hop_count < self.max_hops
            ):
                logger.info("IRCoT: auto-injecting refine_search based on thought")
                step.action = "refine_search"
                step.action_input = {
                    "refined_query": step.thought,
                    "reason": "IRCoT auto-inject: information insufficient",
                    "focus_entities": [],
                }

            # 4. 도구 실행 - Security: 허용 액션 및 파라미터 검증
            if step.action and not self.tool_executor:
                step.observation = "Error: 도구 실행기가 연결되지 않았습니다"
                continue

            if step.action and self.tool_executor:
                # Security: 액션 검증
                is_valid, error_msg = validate_action(step.action, step.action_input)
                if not is_valid:
                    logger.warning(f"Action validation failed: {error_msg}")
                    step.observation = f"Security Error: {error_msg}"
                    continue

                # Multi-hop: refine_search 처리
                if step.action == "refine_search" and hop_count < self.max_hops:
                    hop_count += 1
                    observation = await self._execute_multihop_search(step, steps, query)
                    step.observation = observation
                else:
                    try:
                        result = await self.tool_executor.execute(
                            step.action, step.action_input or {}
                        )
                        step.observation = str(result.data) if result.success else result.error
                    except Exception as e:
                        step.observation = f"Error: {e}"

        # 5. 반복 한도에 걸려 final_answer가 없으면 관찰을 근거로 답을 한 번 더 요청한다
        if not final_answer:
            final_answer = await self._force_final_answer(query, context, steps)

        # 6. Self-Reflection
        reflection_result = await self._reflect(query, final_answer)

        return ReActResult(
            final_answer=final_answer,
            steps=steps,
            iterations=iterations,
            confidence=reflection_result.get("quality_score", 0.5),
            needs_improvement=reflection_result.get("needs_improvement", False),
            hop_count=hop_count,
        )

    async def _force_final_answer(self, query: str, context: str, steps: list[ReActStep]) -> str:
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
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"ReAct forced final answer failed: {e}")
            return ""

    def _needs_retrieval(self, thought: str) -> bool:
        """IRCoT: thought에서 추가 검색 필요 여부 판단"""
        thought_lower = thought.lower()
        return any(keyword in thought_lower for keyword in self.IRCOT_KEYWORDS)

    async def _execute_multihop_search(
        self,
        current_step: ReActStep,
        all_steps: list[ReActStep],
        original_query: str,
    ) -> str:
        """Multi-hop refine_search 실행: 이전 관찰과 새 검색 결과를 결합"""
        action_input = current_step.action_input or {}
        refined_query = action_input.get("refined_query", original_query)
        focus_entities = action_input.get("focus_entities", [])

        # 이전 관찰 수집
        previous_observations = []
        for step in all_steps:
            if step.observation and step is not current_step:
                previous_observations.append(step.observation)

        # refine_search는 문서 검색(search_docs)으로 실행한다 — IRCoT의 "추가 검색" (트랙 4-A).
        # 이전에는 query_data로 실행했는데 그 도구는 레지스트리에서 사라졌다.
        query_parts = [str(refined_query)] + [str(e) for e in focus_entities]
        search_params: dict[str, Any] = {"query": " ".join(p for p in query_parts if p)}

        try:
            result = await self.tool_executor.execute(REFINE_SEARCH_TOOL, search_params)
            new_observation = str(result.data) if result.success else (result.error or "")
        except Exception as e:
            new_observation = f"Error: {e}"

        # 이전 관찰과 새 결과를 결합
        combined_parts = []
        if previous_observations:
            combined_parts.append(f"[Hop 이전 결과] {'; '.join(previous_observations[:3])}")
        combined_parts.append(f"[Hop 새 결과] {new_observation}")

        return " | ".join(combined_parts)

    async def _execute_step(
        self, query: str, context: str, previous_steps: list[ReActStep]
    ) -> ReActStep:
        """단일 ReAct 단계 실행"""
        steps_text = self._format_steps(previous_steps)

        prompt = self.REACT_PROMPT.format(context=context, query=query, steps=steps_text or "없음")

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2,
            )

            return self._parse_step(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"ReAct step failed: {e}")
            return ReActStep(thought=f"Error: {e}")

    async def _reflect(self, query: str, answer: str) -> dict[str, Any]:
        """Self-Reflection 실행"""
        prompt = self.REFLECTION_PROMPT.format(query=query, answer=answer)

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )

            content = response.choices[0].message.content
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0:
                return json.loads(content[json_start:json_end])
        except Exception as e:
            logger.error(f"Reflection failed: {e}")

        return {"quality_score": 0.5, "needs_improvement": False}

    def _format_steps(self, steps: list[ReActStep]) -> str:
        """단계 포맷"""
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

    def _parse_step(self, content: str) -> ReActStep:
        """응답 파싱"""
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0:
                data = json.loads(content[json_start:json_end])
                return ReActStep(
                    thought=data.get("thought", ""),
                    action=data.get("action"),
                    action_input=data.get("action_input"),
                )
        except Exception:
            logger.warning("ReAct 스텝 JSON 파싱 실패", exc_info=True)

        return ReActStep(thought=content)


# 싱글톤
_react_agent: ReActAgent | None = None


def get_react_agent() -> ReActAgent:
    """ReActAgent 싱글톤"""
    global _react_agent
    if _react_agent is None:
        _react_agent = ReActAgent()
    return _react_agent
