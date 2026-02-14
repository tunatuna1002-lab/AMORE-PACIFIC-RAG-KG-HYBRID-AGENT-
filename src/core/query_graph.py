"""
Query Processing Graph
======================
LangGraph 패턴의 경량 자체 구현.

프레임워크 의존 없이 노드 함수 + 조건부 엣지로
쿼리 처리 파이프라인을 명시적 상태 그래프로 표현합니다.

Graph Structure:
    GUARD → CACHE_CHECK → GATHER_CONTEXT → ASSESS_CONFIDENCE
        → [HIGH] → GENERATE_RESPONSE
        → [UNKNOWN] → CLARIFICATION
        → [MEDIUM/LOW + complex] → REACT_AGENT → GENERATE_RESPONSE
        → [MEDIUM/LOW + simple] → DECIDE → EXECUTE_TOOL → GENERATE_RESPONSE
    GENERATE_RESPONSE → OUTPUT_GUARD → DONE
"""

from __future__ import annotations

import logging
from typing import Any

from .cache import ResponseCache
from .confidence import ConfidenceAssessor
from .context_gatherer import ContextGatherer
from .decision_maker import DecisionMaker
from .graph_state import QueryState
from .models import ConfidenceLevel, Context, Decision, Response
from .prompt_guard import PromptGuard
from .query_router import QueryRouter
from .response_pipeline import ResponsePipeline
from .tool_coordinator import ToolCoordinator

logger = logging.getLogger(__name__)


class QueryGraph:
    """
    쿼리 처리 상태 그래프

    LangGraph 패턴을 프레임워크 없이 구현.
    각 노드는 QueryState를 받아 수정하고 반환하며,
    라우팅 함수가 다음 노드를 결정합니다.

    Components (all injected via __init__):
        cache: 응답 캐시
        context_gatherer: 컨텍스트 수집기
        confidence_assessor: 신뢰도 평가기
        decision_maker: LLM 의사결정기
        tool_coordinator: 도구 실행 조율기
        response_pipeline: 응답 생성 파이프라인
        react_agent: ReAct 에이전트 (optional)
    """

    def __init__(
        self,
        cache: ResponseCache,
        context_gatherer: ContextGatherer,
        confidence_assessor: ConfidenceAssessor,
        decision_maker: DecisionMaker,
        tool_coordinator: ToolCoordinator,
        response_pipeline: ResponsePipeline,
        react_agent: Any | None = None,
    ):
        self._cache = cache
        self._context_gatherer = context_gatherer
        self._confidence_assessor = confidence_assessor
        self._decision_maker = decision_maker
        self._tool_coordinator = tool_coordinator
        self._response_pipeline = response_pipeline
        self._react_agent = react_agent

    # =========================================================================
    # Node Methods — each takes QueryState, returns QueryState
    # =========================================================================

    async def _node_guard(self, state: QueryState) -> QueryState:
        """PromptGuard 입력 검증 노드"""
        is_safe, block_reason, sanitized_query = PromptGuard.check_input(state.query)

        if not is_safe:
            logger.warning(f"PromptGuard blocked input: {block_reason}")
            state.is_blocked = True
            state.block_reason = block_reason
            state.response = Response(
                text=PromptGuard.get_rejection_message(block_reason),
                confidence_score=1.0,
                sources=[],
            )
            return state

        # out_of_scope 경고 시에도 처리는 계속
        if block_reason == "out_of_scope_warning":
            state.query = sanitized_query  # 원본 유지하되 플래그 기록

        return state

    async def _node_cache_check(self, state: QueryState) -> QueryState:
        """캐시 확인 노드"""
        if not state.skip_cache:
            cached = self._cache.get(state.query, "query")
            if cached:
                logger.info(f"Cache hit: {state.query[:30]}...")
                state.response = cached
                state.metadata["cache_hit"] = True

        return state

    async def _node_gather_context(self, state: QueryState) -> QueryState:
        """컨텍스트 수집 노드"""
        state.context = await self._context_gatherer.gather(
            query=state.query, current_metrics=state.current_metrics
        )
        return state

    def _node_assess_confidence(self, state: QueryState) -> QueryState:
        """신뢰도 평가 노드

        brain.py의 _assess_confidence_level 로직을 복제.
        컨텍스트 데이터 점수 + 쿼리 의도 명확성 점수를 합산하여
        ConfidenceAssessor에 위임합니다.
        """
        context = state.context
        if context is None:
            state.confidence_level = ConfidenceLevel.UNKNOWN
            return state

        # Build rule_result from context signals
        rule_result: dict[str, Any] = {
            "max_score": 0.0,
            "confidence": 0.0,
            "query_type": "unknown",
        }

        # --- 1) 컨텍스트 데이터 점수 ---
        score = 0.0
        if context.kg_facts:
            score += min(len(context.kg_facts), 3) * 1.5
        if context.rag_docs:
            score += min(len(context.rag_docs), 3) * 1.0
        if context.kg_inferences:
            score += min(len(context.kg_inferences), 2) * 2.0
        if context.entities:
            entity_count = sum(len(v) for v in context.entities.values() if isinstance(v, list))
            score += min(entity_count, 3) * 1.0

        # --- 2) 쿼리 의도 명확성 점수 (최소 바닥 보장) ---
        query = context.query if hasattr(context, "query") else ""
        query_intent_score = self._assess_query_intent(query)
        score += query_intent_score

        rule_result["max_score"] = score

        state.confidence_level = self._confidence_assessor.assess(rule_result, context)
        return state

    async def _node_decide(self, state: QueryState) -> QueryState:
        """LLM 의사결정 노드"""
        confidence_value = state.confidence_level.value if state.confidence_level else "medium"
        state.decision = await self._decision_maker.decide(
            state.query,
            state.context,
            state.system_state,
            confidence_level=confidence_value,
        )
        return state

    async def _node_execute_tool(self, state: QueryState) -> QueryState:
        """도구 실행 노드"""
        if state.decision and state.decision.requires_tool():
            state.tool_result = await self._tool_coordinator.execute(
                tool_name=state.decision.tool,
                params=state.decision.tool_params,
            )
        return state

    async def _node_react(self, state: QueryState) -> QueryState:
        """ReAct 에이전트 실행 노드"""
        if not self._react_agent:
            state.response = Response.fallback("ReAct 에이전트를 사용할 수 없습니다.")
            return state

        try:
            context = state.context
            react_result = await self._react_agent.run(
                query=state.query,
                context=context.summary or "컨텍스트 없음" if context else "컨텍스트 없음",
            )

            state.response = Response(
                text=react_result.final_answer,
                confidence_score=react_result.confidence,
                sources=context.rag_docs[:3] if context and context.rag_docs else [],
                tools_called=[step.action for step in react_result.steps if step.action],
            )

            if react_result.needs_improvement:
                logger.warning(
                    f"ReAct result needs improvement (confidence: {react_result.confidence:.2f})"
                )

        except Exception as e:
            logger.error(f"ReAct processing failed: {e}")
            state.response = Response.fallback(f"ReAct 처리 실패: {str(e)}")

        return state

    async def _node_generate_response(self, state: QueryState) -> QueryState:
        """응답 생성 노드"""
        if self._response_pipeline:
            state.response = await self._response_pipeline.generate(
                query=state.query,
                context=state.context,
                decision=state.decision,
                tool_result=state.tool_result,
            )
        else:
            # 폴백 응답 생성
            import json

            context = state.context
            decision = state.decision

            content = ""
            if state.tool_result and state.tool_result.success:
                content = (
                    f"도구 실행 결과:\n"
                    f"{json.dumps(state.tool_result.data, ensure_ascii=False, indent=2)}"
                )
            elif context and context.summary:
                content = context.summary
            else:
                content = "관련 정보를 찾을 수 없습니다."

            state.response = Response(
                text=content,
                confidence_score=decision.confidence if decision else 0.5,
                sources=(context.rag_docs[:3] if context and context.rag_docs else []),
                tools_called=(
                    [decision.tool] if decision and decision.tool != "direct_answer" else []
                ),
            )

        return state

    def _node_clarification(self, state: QueryState) -> QueryState:
        """명확화 요청 노드"""
        logger.info(f"UNKNOWN confidence - requesting clarification: {state.query[:50]}...")
        state.response = Response(
            text=(
                "질문을 더 구체적으로 해주시겠어요? "
                "예를 들어 특정 브랜드나 카테고리, "
                "분석 지표(SoS, HHI 등)를 포함해주세요."
            ),
            query_type="clarification",
            confidence_level=state.confidence_level,
            confidence_score=0.2,
            suggestions=[
                "LANEIGE의 Lip Care 카테고리 점유율은?",
                "최근 크롤링 데이터 기반 Top 10 브랜드 알려줘",
                "경쟁사 대비 LANEIGE 포지셔닝 분석해줘",
            ],
        )
        return state

    def _node_output_guard(self, state: QueryState) -> QueryState:
        """PromptGuard 출력 검증 노드"""
        if state.response and state.response.text:
            is_output_safe, sanitized_text = PromptGuard.check_output(state.response.text)
            if not is_output_safe:
                state.response.text = sanitized_text
        return state

    # =========================================================================
    # Routing Methods — determine next node based on state
    # =========================================================================

    def _route_after_guard(self, state: QueryState) -> str:
        """Guard 후 라우팅: 차단 시 done, 아니면 cache_check"""
        if state.is_blocked:
            return "done"
        return "cache_check"

    def _route_after_cache(self, state: QueryState) -> str:
        """캐시 후 라우팅: 캐시 히트 시 done, 아니면 gather_context"""
        if state.response is not None:
            return "done"
        return "gather_context"

    def _route_after_confidence(self, state: QueryState) -> str:
        """신뢰도 기반 라우팅

        Returns:
            "generate_response": HIGH confidence - 직접 응답
            "clarification": UNKNOWN - 명확화 요청
            "react": MEDIUM/LOW + 복잡한 질문 - ReAct 모드
            "decide": MEDIUM/LOW + 단순 질문 - DecisionMaker
        """
        if self._confidence_assessor.should_skip_llm_decision(state.confidence_level):
            return "generate_response"

        if self._confidence_assessor.should_request_clarification(state.confidence_level):
            return "clarification"

        # MEDIUM/LOW: 복잡도 판단
        if self._react_agent and self._is_complex_query(state.query, state.context):
            return "react"

        return "decide"

    def _route_after_decide(self, state: QueryState) -> str:
        """Decision 후 라우팅: 도구 필요 시 execute_tool, 아니면 generate_response"""
        if state.decision and state.decision.requires_tool():
            return "execute_tool"
        return "generate_response"

    # =========================================================================
    # Helper Methods (replicated from brain.py for self-containment)
    # =========================================================================

    @staticmethod
    def _assess_query_intent(query: str) -> float:
        """쿼리 자체의 의도 명확성 점수 반환

        UNKNOWN(< 1.5)은 의도 파악이 불가한 경우에만 해당.
        한국어/영어로 의미 있는 질문이면 최소 1.5점(LOW) 보장.

        Returns:
            0.0: 빈 쿼리 또는 의미 없는 문자열
            1.5: 일반적인 질문 (의도 파악 가능)
            2.5: 도메인 관련 질문 (브랜드, 지표, 분석 키워드 포함)
        """
        if not query or not query.strip():
            return 0.0

        stripped = query.strip()

        # 너무 짧은 무의미 입력 (1~2자)
        if len(stripped) <= 2:
            return 0.0

        score = 0.0

        # 도메인 키워드 (브랜드, 제품, 카테고리)
        domain_keywords = [
            "laneige",
            "라네즈",
            "lip",
            "립",
            "mask",
            "마스크",
            "sleeping",
            "슬리핑",
            "cream",
            "크림",
            "skin",
            "스킨",
            "beauty",
            "뷰티",
            "makeup",
            "메이크업",
            "powder",
            "파우더",
            "아모레",
            "amore",
            "설화수",
            "sulwhasoo",
            "이니스프리",
            "amazon",
            "아마존",
        ]
        if any(kw in stripped.lower() for kw in domain_keywords):
            score += 1.0

        # 분석/질문 의도 키워드
        intent_keywords = [
            "분석",
            "비교",
            "추천",
            "전략",
            "예측",
            "원인",
            "이유",
            "왜",
            "어떻게",
            "알려",
            "보여",
            "설명",
            "순위",
            "상승",
            "하락",
            "점유",
            "경쟁",
            "트렌드",
            "현황",
            "변화",
            "추이",
            "sos",
            "hhi",
            "cpi",
            "share",
            "rank",
            "top",
            "analyze",
            "compare",
            "explain",
            "show",
            "tell",
        ]
        if any(kw in stripped.lower() for kw in intent_keywords):
            score += 1.0

        # 의미 있는 질문이면 최소 LOW 바닥 보장 (1.5)
        has_meaningful_length = len(stripped) >= 3
        if has_meaningful_length and score == 0.0:
            score = 1.5

        # 도메인 또는 의도 키워드가 있으면 바닥 보장
        if score > 0.0 and score < 1.5:
            score = 1.5

        return score

    @staticmethod
    def _is_complex_query(query: str, context: Context | None) -> bool:
        """복잡한 질문인지 판단 (QueryRouter 통합)

        복잡한 질문의 특징:
        - 여러 단계 추론 필요
        - 다중 데이터 소스 필요
        - "왜", "어떻게", "비교" 등 분석적 질문
        - 컨텍스트가 불충분
        - 복합 쿼리 (A와 B 비교, 여러 분석 요청)
        """
        if context is None:
            return False

        # 복잡도 키워드
        complex_keywords = ["왜", "어떻게", "비교", "분석", "추천", "전략", "예측", "원인"]
        has_complex_keyword = any(keyword in query for keyword in complex_keywords)

        # 컨텍스트 부족
        has_kg_triples = hasattr(context, "kg_triples") and context.kg_triples
        low_context = not context.rag_docs or len(context.rag_docs) < 2 or not has_kg_triples

        # 다단계 질문 (여러 개의 의문사 또는 접속사)
        multi_step = query.count("?") > 1 or any(
            conj in query for conj in ["그리고", "또한", "하지만", "그러나"]
        )

        # QueryRouter 복합 쿼리 감지
        router = QueryRouter()
        is_compound = router.is_compound(query)

        return has_complex_keyword or (low_context and multi_step) or is_compound

    @staticmethod
    def _extract_key_points(context: Context | None) -> list[str]:
        """컨텍스트에서 핵심 포인트 추출"""
        if context is None:
            return []

        points: list[str] = []
        for fact in (context.kg_facts or [])[:3]:
            if hasattr(fact, "entity") and hasattr(fact, "fact_type"):
                points.append(f"{fact.entity}: {fact.fact_type}")
        for inf in (context.kg_inferences or [])[:2]:
            if isinstance(inf, dict) and "insight" in inf:
                points.append(inf["insight"])
        return points

    # =========================================================================
    # Graph Execution
    # =========================================================================

    async def run(self, state: QueryState) -> QueryState:
        """
        상태 그래프 실행

        Args:
            state: 초기 QueryState

        Returns:
            최종 QueryState (response 포함)
        """
        state.original_query = state.query

        # GUARD
        state = await self._node_guard(state)
        next_node = self._route_after_guard(state)
        if next_node == "done":
            return state

        # CACHE_CHECK
        state = await self._node_cache_check(state)
        next_node = self._route_after_cache(state)
        if next_node == "done":
            return state

        # GATHER_CONTEXT
        state = await self._node_gather_context(state)

        # ASSESS_CONFIDENCE
        state = self._node_assess_confidence(state)

        # ROUTE based on confidence
        next_node = self._route_after_confidence(state)

        if next_node == "generate_response":
            # HIGH confidence - direct answer (skip LLM decision)
            logger.info(f"HIGH confidence - skipping LLM decision for: {state.query[:50]}...")
            state.decision = Decision(
                tool="direct_answer",
                tool_params={},
                reason=(
                    f"HIGH confidence ({state.confidence_level.value}) - direct context answer"
                ),
                confidence=0.9,
                key_points=self._extract_key_points(state.context),
            )
        elif next_node == "clarification":
            state = self._node_clarification(state)
            state = self._node_output_guard(state)
            return state
        elif next_node == "react":
            logger.info(f"Complex query detected, using ReAct mode: {state.query[:50]}...")
            state = await self._node_react(state)
            state = self._node_output_guard(state)
            return state
        else:
            # DECIDE
            state = await self._node_decide(state)
            # ROUTE after decide
            next_node = self._route_after_decide(state)
            if next_node == "execute_tool":
                state = await self._node_execute_tool(state)

        # GENERATE_RESPONSE
        state = await self._node_generate_response(state)

        # OUTPUT_GUARD
        state = self._node_output_guard(state)

        return state
