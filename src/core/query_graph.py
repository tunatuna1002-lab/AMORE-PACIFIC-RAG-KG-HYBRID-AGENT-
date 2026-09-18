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
        → [MEDIUM/LOW + 2홉 이상] → REACT_AGENT → GENERATE_RESPONSE
        → [MEDIUM/LOW + 1홉] → DECIDE → EXECUTE_TOOL → GENERATE_RESPONSE
    GENERATE_RESPONSE → OUTPUT_GUARD → DONE

분기 구현은 ``stream()`` 하나뿐이다 (트랙 5-A). ``run()``은 그 제너레이터를 끝까지
소비하는 얇은 래퍼이고, ``UnifiedBrain.process_query_stream``은 같은 제너레이터가 내보내는
진행 이벤트를 SSE로 흘려보낸다. 예전에는 brain.py가 이 파이프라인을 통째로 복제해
스트림 경로에만 캐시·복합 질의 감지·route_trace가 빠져 있었다.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from .cache import ResponseCache
from .confidence import ConfidenceAssessor, legacy_count_score_to_fit, score_evidence_fit
from .context_gatherer import ContextGatherer
from .decision_maker import DecisionMaker
from .graph_state import QueryState
from .models import ConfidenceLevel, Context, Decision, Response
from .prompt_guard import PromptGuard
from .query_router import QueryRouter
from .response_pipeline import ResponsePipeline
from .router import HopRouter, RouteDecision
from .tool_coordinator import ToolCoordinator

logger = logging.getLogger(__name__)

# 진행 상태 문구 — SSE `status` 이벤트로 그대로 나간다 (v3 스트림 계약 유지).
# 분기마다 문구가 다르므로, 분기를 아는 그래프가 소유한다.
STATUS_GATHER = "컨텍스트 수집 중..."
STATUS_HIGH_CONFIDENCE = "높은 신뢰도 — 빠른 응답 생성 중..."
STATUS_CLARIFY = "질문 분석 중..."
STATUS_REACT = "복잡한 질문 감지 — ReAct 분석 모드 시작..."
STATUS_DECIDE = "분석 중..."
STATUS_GENERATE = "응답 생성 중..."


# ReAct 운용 모드 (트랙 5-C)
#   off     ReAct를 쓰지 않는다 (에이전트 미주입 또는 플래그 OFF)
#   on      2홉 이상 질문의 답을 ReAct가 만든다
#   shadow  답은 파이프라인이 만들고, ReAct는 같은 질문을 따로 돌려 기록만 남긴다
REACT_MODE_OFF = "off"
REACT_MODE_ON = "on"
REACT_MODE_SHADOW = "shadow"


def _status(content: str) -> dict[str, Any]:
    return {"type": "status", "content": content}


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
        router: HopRouter | None = None,
        react_mode: str | None = None,
        react_bypass_confidence: bool = False,
    ):
        self._cache = cache
        self._context_gatherer = context_gatherer
        self._confidence_assessor = confidence_assessor
        self._decision_maker = decision_maker
        self._tool_coordinator = tool_coordinator
        self._response_pipeline = response_pipeline
        self._react_agent = react_agent
        self._router = router or HopRouter()
        # 모드를 주지 않으면 에이전트가 있을 때 "on" — 기존 호출부(테스트 포함)의 동작 유지
        self._react_mode = react_mode or (REACT_MODE_ON if react_agent else REACT_MODE_OFF)
        # HIGH 신뢰도라도 홉 라우터가 2홉 이상이면 react로 보낼지 (트랙 6, 기본 False)
        self._react_bypass_confidence = react_bypass_confidence

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
                # 캐시에 저장된 원본 객체는 변경하지 않고 얕은 복사본에
                # route_trace만 "cache"로 덮어써서 반환한다 (다음 문항 오염 방지).
                cached_metadata = getattr(cached, "metadata", None) or {}
                trace = dict(cached_metadata.get("route_trace") or {})
                trace["route"] = "cache"

                response = copy.copy(cached)
                response.metadata = {**cached_metadata, "route_trace": trace}

                state.response = response
                state.metadata["cache_hit"] = True
                state.metadata["route_trace"] = trace

        return state

    async def _node_gather_context(self, state: QueryState) -> QueryState:
        """컨텍스트 수집 노드"""
        state.context = await self._context_gatherer.gather(
            query=state.query, current_metrics=state.current_metrics
        )
        return state

    def _node_assess_confidence(self, state: QueryState) -> QueryState:
        """신뢰도 평가 노드 (적합도 기반, 트랙 5-B — 유일한 구현. brain.py 복제본은 5-A에서 삭제)

        증거 카드가 이 질문에 맞는지를 재서 0~1 점수를 만들고 ConfidenceAssessor에
        위임한다 (``src.core.confidence.score_evidence_fit``). 예전처럼 컨텍스트 개수를
        더하지 않는다 — 검색이 질문마다 비슷한 양을 담아 오면 개수 합은 항상 높아져서
        233문항 중 230문항이 HIGH로 나왔다.

        카드가 하나도 없는 구형 경로(v1 retriever, 카드 미병합 컨텍스트)에서는 옛 개수
        점수로 폴백한 뒤 ``legacy_count_score_to_fit()``으로 같은 사다리에 올린다.
        """
        context = state.context
        if context is None:
            state.confidence_level = ConfidenceLevel.UNKNOWN
            return state

        query = context.query if hasattr(context, "query") else ""
        cards = getattr(context, "prompt_evidence", None) or getattr(context, "evidence", None)

        if cards:
            fit = score_evidence_fit(query, context.entities, cards)
            score = fit.score
            components: dict[str, Any] = fit.to_dict()
        else:
            # 폴백: 카드 파이프라인을 타지 않는 경로(v1 HybridContext, 최소 컨텍스트)에서
            # 신뢰도가 전부 UNKNOWN으로 무너지지 않게 옛 개수 점수를 같은 사다리에 올린다.
            counts: dict[str, float] = {
                "kg_facts": 0.0,
                "rag_docs": 0.0,
                "inferences": 0.0,
                "entities": 0.0,
                "query_intent": 0.0,
            }
            if context.kg_facts:
                counts["kg_facts"] = min(len(context.kg_facts), 3) * 1.5
            if context.rag_docs:
                counts["rag_docs"] = min(len(context.rag_docs), 3) * 1.0
            if context.kg_inferences:
                counts["inferences"] = min(len(context.kg_inferences), 2) * 2.0
            if context.entities:
                entity_count = sum(len(v) for v in context.entities.values() if isinstance(v, list))
                counts["entities"] = min(entity_count, 3) * 1.0
            counts["query_intent"] = self._assess_query_intent(query)

            raw_score = sum(counts.values())
            score = legacy_count_score_to_fit(raw_score)
            components = {
                "basis": "legacy",
                "legacy_count_score": raw_score,
                "legacy_counts": counts,
                "card_count": 0,
            }

        state.confidence_level = self._confidence_assessor.assess({"fit_score": score}, context)

        # 관측용 기록 (분기 로직에는 영향 없음). confidence_score는 0~1 적합도 눈금이다.
        state.metadata["confidence_score"] = score
        state.metadata["confidence_components"] = components
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

            if not react_result.final_answer:
                logger.warning("ReAct returned an empty final answer")
                state.response = Response.fallback("ReAct 분석이 답변을 만들지 못했습니다.")
                return state

            state.response = Response(
                text=react_result.final_answer,
                confidence_score=react_result.confidence,
                sources=self._react_sources(react_result, context),
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

    async def _node_route(self, state: QueryState) -> QueryState:
        """홉 수 라우터 판정 노드 (LLM 폴백이 켜졌을 때만 비동기 호출이 일어난다)."""
        state.metadata["router_decision"] = await self._router.route(state.query)
        return state

    async def _node_react_shadow(self, state: QueryState) -> QueryState:
        """섀도 ReAct: 답변은 그대로 두고 ReAct 결과만 기록한다 (트랙 5-C).

        파이프라인 답변을 **절대 바꾸지 않는다**. 실패해도 응답에 영향이 없도록 예외를
        삼키고 ``error``만 남긴다. 섀도가 쓴 토큰은 파이프라인 사용량과 섞이지 않도록
        ``token_usage``에 따로 적는다.
        """
        started = time.perf_counter()
        shadow: dict[str, Any] = {"ran": False, "error": None}

        try:
            context = state.context
            result = await self._react_agent.run(
                query=state.query,
                context=(context.summary or "컨텍스트 없음") if context else "컨텍스트 없음",
            )
            shadow.update(
                ran=True,
                answer=result.final_answer,
                steps=[
                    {
                        "thought": step.thought,
                        "action": step.action,
                        "observed": bool(step.observation),
                    }
                    for step in (result.steps or [])
                ],
                tools=[step.action for step in (result.steps or []) if step.action],
                iterations=result.iterations,
                hop_count=result.hop_count,
                confidence=result.confidence,
                token_usage=dict(getattr(result, "token_usage", None) or {}),
            )
        except Exception as e:  # 섀도 실패가 답변을 깨뜨리면 안 된다
            shadow["error"] = f"{type(e).__name__}: {e}"
            logger.warning(f"ReAct shadow run failed: {shadow['error']}")

        shadow["elapsed_ms"] = (time.perf_counter() - started) * 1000
        state.metadata["react_shadow"] = shadow

        if state.response is not None:
            if state.response.metadata is None:
                state.response.metadata = {}
            state.response.metadata["react_shadow"] = shadow

        return state

    @staticmethod
    def _react_sources(react_result: Any, context: Context | None) -> list[str]:
        """ReAct 답변의 출처 라벨 (``list[str]``).

        예전에는 ``context.rag_docs[:3]``(dict 목록)을 그대로 실었다. ``Response.sources``와
        ``BrainChatResponse.sources``는 ``list[str]``이라 ReAct 경로로 답한 질문은
        ``/api/v4/chat``에서 응답 모델 검증에 걸렸다 (트랙 5-C).

        1순위는 ReAct가 **실제로 본** 도구 관찰의 근거 카드다. 도구를 하나도 부르지 않았으면
        답변 프롬프트에 실린 검색 카드로 떨어진다 — 파이프라인 경로와 같은 기준이다.
        """
        sources = [str(s) for s in (getattr(react_result, "sources", None) or [])]
        if sources or context is None:
            return sources
        cards = getattr(context, "prompt_evidence", None) or getattr(context, "evidence", None)
        if not cards:
            return []
        from src.rag.evidence_assembly import evidence_source_labels

        return evidence_source_labels(cards)

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
        """신뢰도 → (MEDIUM/LOW면) 홉 수 기반 라우팅

        Returns:
            "clarification": UNKNOWN - 명확화 요청
            "react": react_bypass_confidence=True + react ON + 2홉 이상이면 HIGH도 포함
                     (트랙 6), 그 외에는 MEDIUM/LOW + 2홉 이상 (react_mode == "on"일 때만)
            "generate_response": HIGH confidence - 직접 응답
            "decide": MEDIUM/LOW + 1홉 - DecisionMaker
        """
        if self._confidence_assessor.should_request_clarification(state.confidence_level):
            return "clarification"

        # HIGH 신뢰도라도 홉 라우터가 2홉 이상이면 react로 보낸다 (opt-in, 트랙 6).
        # UNKNOWN은 위에서 이미 걸러졌으니 여기서부터는 HIGH/MEDIUM/LOW뿐이다.
        if (
            self._react_bypass_confidence
            and self._react_mode == REACT_MODE_ON
            and self._react_agent
        ):
            decision = self._router_decision(state)
            state.metadata["is_complex"] = decision.use_react
            if decision.use_react:
                return "react"

        if self._confidence_assessor.should_skip_llm_decision(state.confidence_level):
            return "generate_response"

        decision = self._router_decision(state)

        # is_complex는 관측 필드다 — ReAct를 쓸 수 있을 때만 채운다 (기존 계약 유지)
        if self._react_agent is not None:
            state.metadata["is_complex"] = decision.use_react

        if self._react_mode == REACT_MODE_ON and self._react_agent and decision.use_react:
            return "react"

        return "decide"

    def _router_decision(self, state: QueryState) -> RouteDecision:
        """이 질의의 라우터 판정. ``_node_route``가 미리 넣어 둔 값을 쓰고, 없으면 규칙만
        돌린다 (``_route_after_confidence``를 단독으로 부르는 테스트·호출부 때문)."""
        decision = state.metadata.get("router_decision")
        if decision is None:
            decision = self._router.analyze(state.query)
            state.metadata["router_decision"] = decision
        return decision

    def _route_after_decide(self, state: QueryState) -> str:
        """Decision 후 라우팅: 도구 필요 시 execute_tool, 아니면 generate_response"""
        if state.decision and state.decision.requires_tool():
            return "execute_tool"
        return "generate_response"

    # =========================================================================
    # Helper Methods (분기·판정의 유일한 구현 — 트랙 5-A에서 brain.py 복제본 삭제)
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
        """복잡한 질문인지 판단 — **경로 판정에서는 쓰지 않는다** (트랙 5-C).

        라우팅은 ``src/core/router.py``의 홉 수 판정으로 옮겼다. 이 헬퍼는 "비교·분석"
        같은 어조 키워드와 컨텍스트 부족을 섞어 쓰기 때문에, 서로 의존하지 않는 두 조회
        (예: 두 카테고리 HHI 비교)까지 ReAct로 보냈다. 복합 질의 감지 자체는 다른 곳에서
        참조하므로 남겨 둔다.

        (원래 설명)

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

    def _finalize_route_trace(self, state: QueryState, route: str) -> QueryState:
        """문항별 경로 관측 기록 (분기 로직에는 영향 없는 순수 관측 노드)

        state.metadata["route_trace"]와 (있다면) response.metadata["route_trace"]에
        동일한 dict를 기록한다.

        Args:
            state: 현재 QueryState
            route: "direct" | "clarify" | "decide" | "react" | "blocked" | "cache"

        라우터 판정(hops·router_stages·router_reason·router_basis·router_route)과 섀도 ReAct
        기록(react_shadow)이 있으면 같은 dict에 합쳐진다 (트랙 5-C).

        Returns:
            state (route_trace가 기록된 상태)
        """
        confidence_level = state.confidence_level
        tools_used: list[dict[str, Any]] = []
        decision_tool: str | None = None

        if route == "decide":
            decision_tool = state.decision.tool if state.decision else None
            if state.decision and state.decision.requires_tool():
                tools_used = [
                    {
                        "tool": state.decision.tool,
                        "executed": bool(state.tool_result and state.tool_result.success),
                    }
                ]
        elif route == "react" and state.response is not None:
            tools_used = [
                {"tool": action, "executed": True} for action in (state.response.tools_called or [])
            ]

        trace: dict[str, Any] = {
            "route": route,
            "confidence_level": confidence_level.value if confidence_level else None,
            "confidence_score": state.metadata.get("confidence_score"),
            "confidence_components": state.metadata.get("confidence_components"),
            "tools_used": tools_used,
            "decision_tool": decision_tool,
            "is_complex": state.metadata.get("is_complex"),
        }

        # 홉 수 라우터 판정 (hops·router_stages·router_reason·router_basis·router_route)
        router_decision = state.metadata.get("router_decision")
        if router_decision is not None:
            trace.update(router_decision.to_trace())

        # 섀도 ReAct 결과 (있을 때만)
        shadow = state.metadata.get("react_shadow")
        if shadow is not None:
            trace["react_shadow"] = shadow

        state.metadata["route_trace"] = trace

        if state.response is not None:
            if state.response.metadata is None:
                state.response.metadata = {}
            state.response.metadata["route_trace"] = trace

        return state

    # =========================================================================
    # Graph Execution
    # =========================================================================

    async def run(self, state: QueryState) -> QueryState:
        """
        상태 그래프 실행 (진행 이벤트를 버리는 얇은 래퍼)

        분기는 ``stream()`` 하나에만 있다. 여기서는 이벤트를 소비만 한다.

        Args:
            state: 초기 QueryState

        Returns:
            최종 QueryState (response 포함) — 인자로 받은 그 객체
        """
        async for _ in self.stream(state):
            pass
        return state

    async def stream(self, state: QueryState) -> AsyncIterator[dict[str, Any]]:
        """
        상태 그래프를 실행하면서 진행 이벤트를 순서대로 내보낸다

        이벤트는 SSE 청크와 같은 모양이다:
        ``{"type": "status"|"tool_call", "content": ...}``.
        최종 텍스트·완료 이벤트는 호출자(``UnifiedBrain.process_query_stream``)가
        ``state.response``로 조립한다 — 그래야 처리 시간·통계가 한곳에 남는다.

        모든 노드는 받은 state를 **제자리에서** 고치므로, 제너레이터를 끝까지 소비한
        뒤 호출자가 갖고 있는 state를 그대로 읽으면 된다.

        Args:
            state: 초기 QueryState (제자리에서 수정된다)
        """
        state.original_query = state.query

        # GUARD
        await self._node_guard(state)
        if self._route_after_guard(state) == "done":
            self._finalize_route_trace(state, "blocked")
            return

        # CACHE_CHECK
        await self._node_cache_check(state)
        if self._route_after_cache(state) == "done":
            # route_trace는 _node_cache_check에서 이미 "cache"로 기록됨
            return

        # GATHER_CONTEXT
        yield _status(STATUS_GATHER)
        await self._node_gather_context(state)

        # ASSESS_CONFIDENCE
        self._node_assess_confidence(state)

        # ROUTE — 홉 수 판정 (LLM 폴백이 켜졌을 때만 LLM을 부른다)
        await self._node_route(state)
        next_node = self._route_after_confidence(state)

        if next_node == "generate_response":
            # HIGH confidence - direct answer (skip LLM decision)
            yield _status(STATUS_HIGH_CONFIDENCE)
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
            route = "direct"
        elif next_node == "clarification":
            yield _status(STATUS_CLARIFY)
            self._node_clarification(state)
            self._node_output_guard(state)
            self._finalize_route_trace(state, "clarify")
            return
        elif next_node == "react":
            yield _status(STATUS_REACT)
            logger.info(f"Complex query detected, using ReAct mode: {state.query[:50]}...")
            await self._node_react(state)
            self._node_output_guard(state)
            self._finalize_route_trace(state, "react")
            return
        else:
            # DECIDE
            route = "decide"
            yield _status(STATUS_DECIDE)
            await self._node_decide(state)
            # ROUTE after decide
            if self._route_after_decide(state) == "execute_tool":
                yield {
                    "type": "tool_call",
                    "content": {"name": state.decision.tool, "status": "calling"},
                }
                await self._node_execute_tool(state)
            yield _status(STATUS_GENERATE)

        # GENERATE_RESPONSE
        await self._node_generate_response(state)

        # OUTPUT_GUARD
        self._node_output_guard(state)

        # SHADOW REACT — 답변은 위에서 이미 정해졌다. 기록만 남긴다.
        if (
            self._react_mode == REACT_MODE_SHADOW
            and self._react_agent is not None
            and self._router_decision(state).use_react
        ):
            await self._node_react_shadow(state)

        self._finalize_route_trace(state, route)
