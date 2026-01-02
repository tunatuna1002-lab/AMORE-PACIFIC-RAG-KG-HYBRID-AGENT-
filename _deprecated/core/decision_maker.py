"""
의사결정 엔진 (Decision Maker)
==============================
Level 4 Autonomous Agent의 의사결정 로직

역할:
1. 규칙 기반 빠른 판단 (Rule-based Fast Path)
2. LLM 기반 복잡한 판단 (LLM Slow Path)
3. 권한 및 동의 확인
4. 의사결정 로깅 및 감사

의사결정 흐름:
┌─────────────────────────────────────────────────────────────┐
│                      Query/Event                            │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              1. 권한 확인 (Permission Check)                 │
│              - 금지 액션 체크                                │
│              - 동의 필요 여부 확인                           │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              2. 규칙 기반 판단 (Fast Path)                   │
│              - 키워드 매칭                                   │
│              - 패턴 매칭                                     │
│              - 임계값 확인                                   │
│              ✓ 매칭 시: 즉시 반환 (LLM 호출 안 함)          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼ (규칙 매칭 실패)
┌─────────────────────────────────────────────────────────────┐
│              3. LLM 기반 판단 (Slow Path)                    │
│              - 컨텍스트 분석                                 │
│              - 도구 선택                                     │
│              - 신뢰도 평가                                   │
└─────────────────────────────────────────────────────────────┘

Usage:
    decision_maker = DecisionMaker(rules_engine)
    decision_maker.set_client(openai_client)

    decision = await decision_maker.decide(
        query="LANEIGE 순위 알려줘",
        context=context,
        system_state=state
    )
"""

import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .rules_engine import RulesEngine
from .models import Context, Decision

logger = logging.getLogger(__name__)


# =============================================================================
# 타입 정의
# =============================================================================

class DecisionPath(Enum):
    """의사결정 경로"""
    RULE_BASED = "rule_based"     # 규칙 기반 (빠름)
    LLM_BASED = "llm_based"       # LLM 기반 (유연)
    HYBRID = "hybrid"             # 규칙 + LLM 조합
    BLOCKED = "blocked"           # 권한 부족으로 차단


class DecisionConfidence(Enum):
    """의사결정 신뢰도"""
    HIGH = "high"        # 0.8 이상
    MEDIUM = "medium"    # 0.5 ~ 0.8
    LOW = "low"          # 0.5 미만


@dataclass
class DecisionAudit:
    """의사결정 감사 로그"""
    timestamp: datetime
    query: str
    path: DecisionPath
    decision: Decision
    rule_matches: List[str]
    llm_called: bool
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "query": self.query[:100],
            "path": self.path.value,
            "tool": self.decision.tool,
            "confidence": self.decision.confidence,
            "rule_matches": self.rule_matches,
            "llm_called": self.llm_called,
            "processing_time_ms": self.processing_time_ms
        }


# =============================================================================
# 키워드 패턴 정의
# =============================================================================

# 도구별 키워드 매핑
TOOL_KEYWORDS: Dict[str, List[str]] = {
    "crawl_amazon": [
        "크롤링", "수집", "데이터 수집", "새로고침", "업데이트해줘",
        "refresh", "crawl", "scrape", "다시 수집"
    ],
    "calculate_metrics": [
        "계산", "지표 계산", "분석해줘", "메트릭", "통계",
        "calculate", "metrics", "analyze"
    ],
    "generate_insight": [
        "인사이트", "분석 결과", "요약해줘", "리포트", "보고서",
        "insight", "report", "summary"
    ],
    "query_knowledge_graph": [
        "관계", "연결", "네트워크", "그래프",
        "어떤 브랜드", "경쟁사", "유사한"
    ],
}

# 직접 응답 가능 키워드 (도구 불필요)
DIRECT_ANSWER_KEYWORDS: List[str] = [
    "뭐야", "알려줘", "어때", "얼마야", "몇 등",
    "what", "tell me", "how is", "what is"
]


# =============================================================================
# 의사결정 엔진
# =============================================================================

class DecisionMaker:
    """
    의사결정 엔진

    규칙 + LLM 하이브리드 방식으로 최적의 의사결정을 수행합니다.

    원칙:
    1. 규칙으로 해결 가능하면 LLM 호출 안 함 (비용 절감)
    2. 금지된 액션은 절대 수행 안 함
    3. 동의가 필요한 액션은 사전 확인
    4. 모든 결정을 감사 로그에 기록
    """

    # LLM 프롬프트
    DECISION_PROMPT = """당신은 Amazon 마켓 분석 시스템의 의사결정 엔진입니다.

## 현재 상태
{system_state}

## 사용 가능한 도구
{tools_description}

## 컨텍스트
{context_summary}

## 질문
{query}

## 규칙
1. 컨텍스트만으로 답변 가능하면 "direct_answer" 선택
2. 데이터가 오래됐거나 없으면 적절한 도구 선택
3. 확신이 낮으면 솔직히 confidence 낮게 설정

JSON 형식으로 응답:
```json
{{
    "tool": "도구명 또는 direct_answer",
    "tool_params": {{}},
    "reason": "선택 이유",
    "confidence": 0.0~1.0
}}
```"""

    def __init__(
        self,
        rules_engine: RulesEngine,
        model: str = "gpt-4o-mini"
    ):
        """
        Args:
            rules_engine: 규칙 엔진
            model: LLM 모델
        """
        self.rules_engine = rules_engine
        self.model = model
        self.client = None

        # 감사 로그
        self._audit_log: List[DecisionAudit] = []
        self._max_audit_size = 1000

        # 통계
        self._stats = {
            "total_decisions": 0,
            "rule_based": 0,
            "llm_based": 0,
            "blocked": 0,
            "avg_confidence": 0.0
        }

    def set_client(self, client: Any) -> None:
        """OpenAI 클라이언트 설정"""
        self.client = client

    # =========================================================================
    # 메인 의사결정
    # =========================================================================

    async def decide(
        self,
        query: str,
        context: Context,
        system_state: Dict[str, Any],
        requested_action: Optional[str] = None
    ) -> Decision:
        """
        의사결정 수행

        Args:
            query: 사용자 질문
            context: 컨텍스트
            system_state: 시스템 상태
            requested_action: 명시적 요청 액션 (있으면 권한만 확인)

        Returns:
            Decision 객체
        """
        start_time = datetime.now()
        self._stats["total_decisions"] += 1
        path = DecisionPath.RULE_BASED
        rule_matches = []
        llm_called = False

        try:
            # 1. 명시적 액션 요청인 경우
            if requested_action:
                decision = self._check_explicit_action(requested_action)
                if decision:
                    path = DecisionPath.BLOCKED if not decision.tool else DecisionPath.RULE_BASED
                    return self._finalize_decision(
                        decision, query, path, rule_matches, llm_called, start_time
                    )

            # 2. 권한 사전 확인
            blocked_decision = self._check_forbidden_patterns(query)
            if blocked_decision:
                path = DecisionPath.BLOCKED
                self._stats["blocked"] += 1
                return self._finalize_decision(
                    blocked_decision, query, path, rule_matches, llm_called, start_time
                )

            # 3. 규칙 기반 판단 (Fast Path)
            rule_decision, matched_rules = self._try_rule_based(query, context, system_state)
            rule_matches = matched_rules

            if rule_decision:
                self._stats["rule_based"] += 1
                return self._finalize_decision(
                    rule_decision, query, path, rule_matches, llm_called, start_time
                )

            # 4. LLM 기반 판단 (Slow Path)
            path = DecisionPath.LLM_BASED
            llm_called = True
            self._stats["llm_based"] += 1
            llm_decision = await self._llm_decide(query, context, system_state)

            return self._finalize_decision(
                llm_decision, query, path, rule_matches, llm_called, start_time
            )

        except Exception as e:
            logger.error(f"Decision failed: {e}")
            return Decision(
                tool="direct_answer",
                reason=f"의사결정 오류: {e}",
                confidence=0.3
            )

    def _finalize_decision(
        self,
        decision: Decision,
        query: str,
        path: DecisionPath,
        rule_matches: List[str],
        llm_called: bool,
        start_time: datetime
    ) -> Decision:
        """결정 마무리 및 감사 로그 기록"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # 감사 로그 기록
        audit = DecisionAudit(
            timestamp=datetime.now(),
            query=query,
            path=path,
            decision=decision,
            rule_matches=rule_matches,
            llm_called=llm_called,
            processing_time_ms=processing_time
        )
        self._add_audit(audit)

        # 평균 신뢰도 업데이트
        total = self._stats["total_decisions"]
        current_avg = self._stats["avg_confidence"]
        self._stats["avg_confidence"] = (
            (current_avg * (total - 1) + decision.confidence) / total
        )

        return decision

    def _add_audit(self, audit: DecisionAudit) -> None:
        """감사 로그 추가"""
        self._audit_log.append(audit)
        if len(self._audit_log) > self._max_audit_size:
            self._audit_log = self._audit_log[-500:]

    # =========================================================================
    # 권한 확인
    # =========================================================================

    def _check_explicit_action(self, action: str) -> Optional[Decision]:
        """명시적 액션 요청 시 권한 확인"""
        if not self.rules_engine.is_action_allowed(action):
            return Decision(
                tool=None,
                reason=f"권한 없음: {action}은 금지된 액션입니다",
                confidence=1.0
            )

        if self.rules_engine.requires_consent(action):
            return Decision(
                tool=action,
                reason="사용자 동의 필요",
                confidence=1.0,
                requires_consent=True
            )

        return Decision(
            tool=action,
            reason="명시적 요청",
            confidence=1.0
        )

    def _check_forbidden_patterns(self, query: str) -> Optional[Decision]:
        """금지된 패턴 확인"""
        query_lower = query.lower()

        # 삭제 요청 패턴
        delete_patterns = [
            r"삭제", r"지워", r"제거해", r"없애",
            r"delete", r"remove", r"drop", r"truncate"
        ]

        for pattern in delete_patterns:
            if re.search(pattern, query_lower):
                return Decision(
                    tool=None,
                    reason="데이터 삭제는 금지된 작업입니다",
                    confidence=1.0
                )

        # 시스템 설정 변경 패턴
        system_patterns = [
            r"설정.*변경", r"config.*수정", r"시스템.*바꿔"
        ]

        for pattern in system_patterns:
            if re.search(pattern, query_lower):
                return Decision(
                    tool=None,
                    reason="시스템 설정 변경은 금지된 작업입니다",
                    confidence=1.0
                )

        return None

    # =========================================================================
    # 규칙 기반 판단
    # =========================================================================

    def _try_rule_based(
        self,
        query: str,
        context: Context,
        system_state: Dict[str, Any]
    ) -> Tuple[Optional[Decision], List[str]]:
        """
        규칙 기반 판단 시도

        Returns:
            (Decision or None, matched_rule_names)
        """
        matched_rules = []
        query_lower = query.lower()
        available_tools = system_state.get("available_tools", [])

        # 1. 도구별 키워드 매칭
        for tool_name, keywords in TOOL_KEYWORDS.items():
            if tool_name not in available_tools:
                continue

            for keyword in keywords:
                if keyword in query_lower:
                    matched_rules.append(f"keyword:{keyword}")
                    return (
                        Decision(
                            tool=tool_name,
                            reason=f"키워드 매칭: '{keyword}'",
                            confidence=0.9
                        ),
                        matched_rules
                    )

        # 2. 직접 응답 가능 여부
        if self._can_answer_directly(query, context):
            matched_rules.append("direct_answer:context_sufficient")
            return (
                Decision(
                    tool="direct_answer",
                    reason="컨텍스트 충분",
                    confidence=0.85
                ),
                matched_rules
            )

        # 3. 데이터 상태 기반 판단
        data_decision = self._decide_by_data_state(query, system_state, available_tools)
        if data_decision:
            matched_rules.append(f"data_state:{data_decision.reason}")
            return (data_decision, matched_rules)

        # 규칙 매칭 실패
        return (None, matched_rules)

    def _can_answer_directly(self, query: str, context: Context) -> bool:
        """직접 응답 가능 여부 판단"""
        # 컨텍스트가 충분한지 확인
        if not context.has_sufficient_context():
            return False

        # 직접 응답 키워드 확인
        query_lower = query.lower()
        for keyword in DIRECT_ANSWER_KEYWORDS:
            if keyword in query_lower:
                return True

        return False

    def _decide_by_data_state(
        self,
        query: str,
        system_state: Dict[str, Any],
        available_tools: List[str]
    ) -> Optional[Decision]:
        """데이터 상태 기반 의사결정"""
        data_status = system_state.get("data_status", "없음")

        # 데이터가 없거나 오래된 경우
        if "없음" in data_status or "오래됨" in data_status:
            # 크롤링 가능하면 크롤링 제안
            if "crawl_amazon" in available_tools:
                return Decision(
                    tool="crawl_amazon",
                    reason=f"데이터 상태: {data_status}",
                    confidence=0.75
                )

        return None

    # =========================================================================
    # LLM 기반 판단
    # =========================================================================

    async def _llm_decide(
        self,
        query: str,
        context: Context,
        system_state: Dict[str, Any]
    ) -> Decision:
        """LLM 기반 의사결정"""
        if not self.client:
            return Decision(
                tool="direct_answer",
                reason="LLM 미설정",
                confidence=0.5
            )

        try:
            prompt = self.DECISION_PROMPT.format(
                system_state=self._format_system_state(system_state),
                tools_description=self._format_tools(system_state),
                context_summary=context.summary or "컨텍스트 없음",
                query=query
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )

            return self._parse_llm_response(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return Decision(
                tool="direct_answer",
                reason=f"LLM 오류: {e}",
                confidence=0.3
            )

    def _parse_llm_response(self, response_text: str) -> Decision:
        """LLM 응답 파싱"""
        try:
            # JSON 추출
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                return Decision(
                    tool=data.get("tool", "direct_answer"),
                    tool_params=data.get("tool_params", {}),
                    reason=data.get("reason", "LLM 판단"),
                    confidence=data.get("confidence", 0.7)
                )

        except json.JSONDecodeError:
            pass

        return Decision(
            tool="direct_answer",
            reason="응답 파싱 실패",
            confidence=0.5
        )

    # =========================================================================
    # 포맷팅
    # =========================================================================

    def _format_system_state(self, state: Dict[str, Any]) -> str:
        """시스템 상태 포맷"""
        lines = [
            f"- 데이터 상태: {state.get('data_status', '알 수 없음')}",
            f"- 사용 가능 도구: {', '.join(state.get('available_tools', []))}",
        ]
        if state.get('failed_tools'):
            lines.append(f"- 실패 도구: {', '.join(state['failed_tools'])}")
        return "\n".join(lines)

    def _format_tools(self, state: Dict[str, Any]) -> str:
        """도구 설명 포맷"""
        from .tools import AGENT_TOOLS

        available = state.get("available_tools", [])
        lines = []
        for name, tool in AGENT_TOOLS.items():
            if name in available:
                lines.append(f"- {name}: {tool.description}")
        lines.append("- direct_answer: 추가 도구 없이 컨텍스트로 직접 응답")
        return "\n".join(lines)

    # =========================================================================
    # 통계 및 감사
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            **self._stats,
            "audit_log_size": len(self._audit_log)
        }

    def get_recent_audits(self, limit: int = 20) -> List[Dict[str, Any]]:
        """최근 감사 로그 반환"""
        return [a.to_dict() for a in self._audit_log[-limit:]]

    def get_decision_summary(self) -> str:
        """의사결정 요약"""
        total = self._stats["total_decisions"]
        if total == 0:
            return "아직 의사결정 없음"

        rule_pct = (self._stats["rule_based"] / total) * 100
        llm_pct = (self._stats["llm_based"] / total) * 100

        return (
            f"총 {total}회 | "
            f"규칙 {rule_pct:.1f}% | "
            f"LLM {llm_pct:.1f}% | "
            f"평균 신뢰도 {self._stats['avg_confidence']:.2f}"
        )
