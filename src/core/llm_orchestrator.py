"""
LLM 오케스트레이터
==================
LLM 기반 동적 라우팅 및 도구 선택

핵심 역할:
- 2단계 라우팅: Rule → LLM
- 신뢰도 기반 분기
- 도구 선택 및 실행
- 응답 생성 조율

연결 파일:
- core/models.py: Context, Response, Decision, ConfidenceLevel
- core/confidence.py: ConfidenceAssessor
- core/cache.py: ResponseCache
- core/state.py: OrchestratorState
- core/context_gatherer.py: ContextGatherer
- core/tools.py: ToolExecutor, AGENT_TOOLS
- core/response_pipeline.py: ResponsePipeline
- rag/router.py: RAGRouter (Rule 기반)
"""

import json
import logging
from datetime import datetime
from typing import Any

from src.shared.constants import DEFAULT_MODEL

from .cache import ResponseCache
from .confidence import ConfidenceAssessor
from .context_gatherer import ContextGatherer
from .models import ConfidenceLevel, Context, Decision, Response
from .response_pipeline import ResponsePipeline
from .state import OrchestratorState
from .tools import AGENT_TOOLS, ToolExecutor

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    LLM 기반 오케스트레이터

    동작 흐름:
    1. 캐시 확인
    2. Rule 기반 라우팅 (RAGRouter)
    3. 신뢰도 평가
    4. 신뢰도에 따른 분기:
       - HIGH: LLM 스킵, 바로 응답 생성
       - MEDIUM: LLM에게 도구 선택 위임
       - LOW: LLM에게 전체 판단 위임
       - UNKNOWN: 명확화 요청
    5. 도구 실행 (필요시)
    6. 응답 생성

    Usage:
        orchestrator = LLMOrchestrator()
        response = await orchestrator.process(query)
    """

    # LLM 판단 프롬프트
    DECISION_PROMPT = """당신은 Amazon 마켓 분석 시스템의 라우터입니다.

사용자 질문과 현재 컨텍스트를 분석하여 어떤 도구를 사용할지 결정하세요.

## 사용 가능한 도구

{tools_description}

## 현재 컨텍스트

{context_summary}

## 사용자 질문

{query}

## 지시사항

1. 질문을 분석하고 필요한 도구를 선택하세요
2. 컨텍스트만으로 답변 가능하면 "direct_answer"를 선택하세요
3. 데이터가 필요하면 적절한 도구를 선택하세요
4. 응답에 포함할 핵심 포인트를 정리하세요

반드시 다음 JSON 형식으로만 응답하세요:
```json
{{
    "tool": "도구명 또는 direct_answer",
    "tool_params": {{}},
    "reason": "선택 이유",
    "key_points": ["핵심 포인트1", "핵심 포인트2"]
}}
```"""

    def __init__(
        self,
        openai_client: Any | None = None,
        rag_router: Any | None = None,
        context_gatherer: ContextGatherer | None = None,
        tool_executor: ToolExecutor | None = None,
        response_pipeline: ResponsePipeline | None = None,
        cache: ResponseCache | None = None,
        state: OrchestratorState | None = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Args:
            openai_client: OpenAI 클라이언트
            rag_router: RAGRouter 인스턴스
            context_gatherer: 컨텍스트 수집기
            tool_executor: 도구 실행기
            response_pipeline: 응답 생성 파이프라인
            cache: 응답 캐시
            state: 오케스트레이터 상태
            model: LLM 모델
        """
        self.client = openai_client
        self.router = rag_router
        self.context_gatherer = context_gatherer or ContextGatherer()
        self.tool_executor = tool_executor or ToolExecutor()
        self.response_pipeline = response_pipeline or ResponsePipeline()
        self.cache = cache or ResponseCache()
        self.state = state or OrchestratorState()
        self.model = model

        # 컴포넌트
        self.confidence_assessor = ConfidenceAssessor()

        # 처리 통계
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "rule_handled": 0,
            "llm_decisions": 0,
            "tools_called": 0,
            "clarifications": 0,
        }

    # =========================================================================
    # 초기화 및 설정
    # =========================================================================

    def set_client(self, client: Any) -> None:
        """OpenAI 클라이언트 설정"""
        self.client = client
        self.response_pipeline.set_client(client)

    def set_router(self, router: Any) -> None:
        """RAGRouter 설정"""
        self.router = router

    def set_retriever(self, retriever: Any) -> None:
        """HybridRetriever 설정"""
        self.context_gatherer.set_retriever(retriever)

    async def initialize(self) -> None:
        """비동기 초기화"""
        await self.context_gatherer.initialize()
        logger.info("LLMOrchestrator initialized")

    # =========================================================================
    # 메인 처리 메서드
    # =========================================================================

    async def process(
        self,
        query: str,
        session_id: str | None = None,
        current_metrics: dict[str, Any] | None = None,
        skip_cache: bool = False,
    ) -> Response:
        """
        질문 처리 메인 엔트리

        Args:
            query: 사용자 질문
            session_id: 세션 ID
            current_metrics: 현재 계산된 지표
            skip_cache: 캐시 스킵 여부

        Returns:
            Response 객체
        """
        start_time = datetime.now()
        self._stats["total_queries"] += 1

        try:
            # 세션 설정
            if session_id:
                self.state.set_session(session_id)

            # 1. 캐시 확인
            if not skip_cache:
                cached = self.cache.get(query, "query")
                if cached:
                    self._stats["cache_hits"] += 1
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    return cached

            # 2. Rule 기반 라우팅
            route_result, entities = self._rule_based_routing(query)

            # 3. 컨텍스트 수집 (LLM 판단용)
            context = await self.context_gatherer.gather(
                query=query, entities=entities, current_metrics=current_metrics
            )

            # 4. 신뢰도 평가
            confidence = self.confidence_assessor.assess(route_result, context)

            logger.info(
                f"Query: {query[:50]}... | "
                f"Type: {route_result.get('query_type')} | "
                f"Confidence: {confidence.name}"
            )

            # 5. 신뢰도 기반 분기
            response = await self._route_by_confidence(
                query=query,
                context=context,
                route_result=route_result,
                confidence=confidence,
                current_metrics=current_metrics,
            )

            # 처리 시간 업데이트
            response.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            # 캐시 저장
            if not skip_cache and not response.is_fallback:
                self.cache.set(query, response, "query")

            return response

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return Response.fallback(f"처리 중 오류가 발생했습니다: {str(e)}")

    # =========================================================================
    # 라우팅
    # =========================================================================

    def _rule_based_routing(self, query: str) -> tuple[dict[str, Any], dict[str, list[str]]]:
        """
        Rule 기반 라우팅 (1단계)

        Args:
            query: 질문

        Returns:
            (라우팅 결과, 엔티티)
        """
        if self.router:
            route_result = self.router.route(query)
            entities = self.router.extract_entities(query)
        else:
            # 기본 라우팅
            route_result = {
                "query_type": "unknown",
                "confidence": 0.0,
                "requires_data": False,
                "requires_rag": True,
            }
            entities = {}

        return route_result, entities

    async def _route_by_confidence(
        self,
        query: str,
        context: Context,
        route_result: dict[str, Any],
        confidence: ConfidenceLevel,
        current_metrics: dict[str, Any] | None,
    ) -> Response:
        """
        신뢰도에 따른 분기 처리

        Args:
            query: 질문
            context: 컨텍스트
            route_result: Rule 라우팅 결과
            confidence: 신뢰도 레벨
            current_metrics: 지표

        Returns:
            Response
        """
        # HIGH: LLM 판단 스킵, 바로 응답
        if confidence == ConfidenceLevel.HIGH:
            self._stats["rule_handled"] += 1
            return await self._generate_direct_response(query, context)

        # UNKNOWN: 명확화 요청
        if confidence == ConfidenceLevel.UNKNOWN:
            self._stats["clarifications"] += 1
            return self._generate_clarification(query, context, route_result)

        # MEDIUM/LOW: LLM 판단
        return await self._llm_decision_flow(
            query=query,
            context=context,
            route_result=route_result,
            confidence=confidence,
            current_metrics=current_metrics,
        )

    # =========================================================================
    # LLM 판단 흐름
    # =========================================================================

    async def _llm_decision_flow(
        self,
        query: str,
        context: Context,
        route_result: dict[str, Any],
        confidence: ConfidenceLevel,
        current_metrics: dict[str, Any] | None,
    ) -> Response:
        """
        LLM 판단 기반 처리 흐름

        Args:
            query: 질문
            context: 컨텍스트
            route_result: Rule 결과
            confidence: 신뢰도
            current_metrics: 지표

        Returns:
            Response
        """
        self._stats["llm_decisions"] += 1

        # LLM 판단 요청
        decision = await self._get_llm_decision(query, context, route_result)

        # 도구 실행 필요 여부
        if decision.requires_tool():
            self._stats["tools_called"] += 1
            tool_result = await self.tool_executor.execute(decision.tool, decision.tool_params)

            # 도구 결과로 응답 생성
            return await self.response_pipeline.generate_with_tool_result(
                query=query, context=context, tool_result=tool_result
            )

        # direct_answer: 컨텍스트로 응답
        return await self.response_pipeline.generate(
            query=query, context=context, decision=decision
        )

    async def _get_llm_decision(
        self, query: str, context: Context, route_result: dict[str, Any]
    ) -> Decision:
        """
        LLM에게 도구 선택 판단 요청

        Args:
            query: 질문
            context: 컨텍스트
            route_result: Rule 결과

        Returns:
            Decision
        """
        if not self.client:
            # 클라이언트 없으면 기본 판단
            return self._default_decision(query, route_result)

        try:
            # 도구 설명 생성
            tools_desc = self._format_tools_description()

            # 프롬프트 구성
            prompt = self.DECISION_PROMPT.format(
                tools_description=tools_desc,
                context_summary=context.summary or "컨텍스트 없음",
                query=query,
            )

            # LLM 호출
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1,
            )

            # 응답 파싱
            return self._parse_decision_response(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return self._default_decision(query, route_result)

    def _format_tools_description(self) -> str:
        """도구 설명 포맷팅"""
        lines = []
        for name, tool in AGENT_TOOLS.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)

    def _parse_decision_response(self, response_text: str) -> Decision:
        """
        LLM 응답 파싱

        Args:
            response_text: LLM 응답

        Returns:
            Decision
        """
        try:
            # JSON 추출
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)

                return Decision(
                    tool=data.get("tool"),
                    tool_params=data.get("tool_params", {}),
                    reason=data.get("reason", ""),
                    key_points=data.get("key_points", []),
                    confidence=0.8,
                )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM decision: {e}")

        # 파싱 실패 시 기본값
        return Decision(
            tool="direct_answer", reason="LLM 응답 파싱 실패, 직접 응답", confidence=0.5
        )

    def _default_decision(self, query: str, route_result: dict[str, Any]) -> Decision:
        """
        기본 판단 로직 (LLM 없을 때)

        Args:
            query: 질문
            route_result: Rule 결과

        Returns:
            Decision
        """
        route_result.get("query_type", "unknown")

        # 크롤링 요청 감지
        if any(kw in query.lower() for kw in ["크롤링", "수집", "업데이트", "refresh"]):
            return Decision(tool="crawl_amazon", reason="크롤링 요청 감지", confidence=0.9)

        # 지표 계산 요청 감지
        if any(kw in query.lower() for kw in ["계산", "분석", "지표"]):
            return Decision(tool="calculate_metrics", reason="지표 계산 요청 감지", confidence=0.8)

        # 데이터 조회 필요
        if route_result.get("requires_data"):
            return Decision(
                tool="query_data",
                tool_params={"query_type": "brand_metrics"},
                reason="데이터 조회 필요",
                confidence=0.7,
            )

        # 기본: 직접 응답
        return Decision(tool="direct_answer", reason="컨텍스트로 응답 가능", confidence=0.6)

    # =========================================================================
    # 응답 생성
    # =========================================================================

    async def _generate_direct_response(self, query: str, context: Context) -> Response:
        """
        컨텍스트 기반 직접 응답 생성

        Args:
            query: 질문
            context: 컨텍스트

        Returns:
            Response
        """
        # RAG 문서가 있으면 문서 기반 응답 생성 (LLM 없이도 가능)
        if context.rag_docs:
            return self._generate_rag_based_response(query, context)

        return await self.response_pipeline.generate(query=query, context=context)

    def _generate_rag_based_response(self, query: str, context: Context) -> Response:
        """
        RAG 문서 기반 직접 응답 생성 (LLM 없이)

        Args:
            query: 질문
            context: 컨텍스트

        Returns:
            Response
        """
        # RAG 문서에서 내용 추출
        response_parts = []
        sources = []

        for doc in context.rag_docs[:2]:
            content = doc.get("content", "")
            title = doc.get("metadata", {}).get("title", "")

            if content:
                response_parts.append(content)
            if title and title not in sources:
                sources.append(title)

        response_text = (
            "\n\n".join(response_parts) if response_parts else "관련 정보를 찾을 수 없습니다."
        )

        # 질문 유형 추론
        query_type = self._infer_query_type_from_query(query)

        return Response(
            text=response_text,
            query_type=query_type,
            confidence_level=ConfidenceLevel.HIGH,
            confidence_score=5.0,
            sources=sources,
            entities=context.entities,
            suggestions=["더 궁금한 점이 있으신가요?", "다른 지표도 알아볼까요?"],
            processing_time_ms=0,
        )

    def _infer_query_type_from_query(self, query: str) -> str:
        """질문에서 유형 추론"""
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["뭐야", "무엇", "정의", "어떻게 계산"]):
            return "definition"
        elif any(kw in query_lower for kw in ["해석", "의미", "높으면", "낮으면"]):
            return "interpretation"
        elif any(kw in query_lower for kw in ["순위", "랭킹", "현재"]):
            return "data_query"
        elif any(kw in query_lower for kw in ["분석", "비교"]):
            return "analysis"
        return "general"

    def _generate_clarification(
        self, query: str, context: Context, route_result: dict[str, Any]
    ) -> Response:
        """
        명확화 요청 응답 생성

        Args:
            query: 질문
            context: 컨텍스트
            route_result: Rule 결과

        Returns:
            Response
        """
        # 기본 명확화 메시지
        message = route_result.get("fallback_message", "질문의 의도를 정확히 파악하지 못했습니다.")

        suggestions = [
            "어떤 브랜드/제품을 분석할까요?",
            "SoS 정의가 궁금하신가요?",
            "현재 순위를 조회할까요?",
        ]

        return Response.clarification(message, suggestions)

    # =========================================================================
    # 상태 및 통계
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """오케스트레이터 통계"""
        return {**self._stats, "cache_stats": self.cache.get_stats(), "state": self.state.to_dict()}

    def reset_stats(self) -> None:
        """통계 초기화"""
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "rule_handled": 0,
            "llm_decisions": 0,
            "tools_called": 0,
            "clarifications": 0,
        }

    def get_state_summary(self) -> str:
        """상태 요약"""
        return self.state.to_context_summary()
