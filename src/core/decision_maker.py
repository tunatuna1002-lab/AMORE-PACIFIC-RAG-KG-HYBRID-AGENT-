"""
Decision Maker - LLM 의사결정 전담
==================================
UnifiedBrain에서 분리된 LLM 기반 의사결정 컴포넌트

책임:
- 질문 분석
- 도구 선택 결정
- 신뢰도 판단
- LLM 호출 및 응답 파싱

관련 Protocol: DecisionMakerProtocol
"""

import json
import logging
from typing import Any

from litellm import acompletion

from .models import Context
from .tools import AGENT_TOOLS

logger = logging.getLogger(__name__)


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

## 사용 가능한 도구
{tools_description}

## 수집된 컨텍스트
{context_summary}

## 사용자 질문
{query}

## 지시사항
1. 시스템 상태와 컨텍스트를 분석하세요
2. 질문에 답하기 위해 필요한 것을 파악하세요
3. 컨텍스트만으로 답변 가능하면 "direct_answer"를 선택하세요
4. 추가 데이터가 필요하면 적절한 도구를 선택하세요
5. 데이터가 오래됐으면 크롤링을 권장하세요

반드시 다음 JSON 형식으로만 응답하세요:
```json
{{
    "tool": "도구명 또는 direct_answer",
    "tool_params": {{}},
    "reason": "선택 이유",
    "confidence": 0.0~1.0,
    "key_points": ["핵심 포인트1", "핵심 포인트2"]
}}
```"""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1, max_tokens: int = 500):
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

    async def decide(
        self, query: str, context: Context, system_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        LLM 기반 의사결정

        Args:
            query: 사용자 질문
            context: 수집된 컨텍스트
            system_state: 시스템 상태

        Returns:
            의사결정 결과 dict:
                - tool: 선택된 도구명 (또는 "direct_answer")
                - tool_params: 도구 파라미터
                - reason: 선택 이유
                - confidence: 신뢰도 (0.0~1.0)
                - key_points: 핵심 포인트 목록
        """
        self._decision_count += 1

        try:
            prompt = self.DECISION_PROMPT.format(
                system_state=self._format_system_state(system_state),
                tools_description=self._format_tools_description(system_state),
                context_summary=context.summary or "컨텍스트 없음",
                query=query,
            )

            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            return self._parse_decision(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return self._fallback_decision(str(e))

    def _parse_decision(self, response_text: str) -> dict[str, Any]:
        """LLM 응답 파싱"""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                decision = json.loads(response_text[json_start:json_end])
                # 필수 필드 검증
                if "tool" not in decision:
                    decision["tool"] = "direct_answer"
                if "confidence" not in decision:
                    decision["confidence"] = 0.5
                if "reason" not in decision:
                    decision["reason"] = ""
                if "key_points" not in decision:
                    decision["key_points"] = []
                if "tool_params" not in decision:
                    decision["tool_params"] = {}
                return decision

        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM decision JSON")

        return self._fallback_decision("파싱 실패")

    def _fallback_decision(self, reason: str) -> dict[str, Any]:
        """폴백 의사결정"""
        return {
            "tool": "direct_answer",
            "tool_params": {},
            "reason": f"LLM 오류: {reason}",
            "confidence": 0.3,
            "key_points": [],
        }

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

    def _format_tools_description(self, state: dict[str, Any]) -> str:
        """도구 설명 포맷"""
        available = state.get("available_tools", [])
        lines = []
        for name, tool in AGENT_TOOLS.items():
            if name in available:
                lines.append(f"- {name}: {tool.description}")
        lines.append("- direct_answer: 컨텍스트만으로 직접 답변")
        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return {"decision_count": self._decision_count, "model": self.model}
