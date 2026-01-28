"""
Query Processor - 사용자 질문 처리 전담
======================================
UnifiedBrain에서 분리된 질문 처리 컴포넌트

책임:
- 사용자 질문 전처리
- 의사결정 -> 도구 실행 -> 응답 생성 파이프라인 조율
- 캐시 관리
- 통계 추적

관련 Protocol: QueryProcessorProtocol
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .cache import ResponseCache
from .context_gatherer import ContextGatherer
from .decision_maker import DecisionMaker
from .models import Context, Response
from .response_pipeline import ResponsePipeline
from .tool_coordinator import ToolCoordinator

if TYPE_CHECKING:
    from .state import OrchestratorState

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    사용자 질문 처리

    의사결정 -> 도구 실행 -> 응답 생성 파이프라인을 조율합니다.

    Usage:
        processor = QueryProcessor(
            decision_maker=decision_maker,
            tool_coordinator=tool_coordinator,
            response_pipeline=response_pipeline,
            context_gatherer=context_gatherer
        )
        response = await processor.process(query, context)
    """

    def __init__(
        self,
        decision_maker: DecisionMaker,
        tool_coordinator: ToolCoordinator,
        response_pipeline: ResponsePipeline,
        context_gatherer: ContextGatherer | None = None,
        cache: ResponseCache | None = None,
        state: "OrchestratorState | None" = None,
    ):
        """
        Args:
            decision_maker: LLM 의사결정 컴포넌트
            tool_coordinator: 도구 실행 조율 컴포넌트
            response_pipeline: 응답 생성 파이프라인
            context_gatherer: 컨텍스트 수집기
            cache: 응답 캐시
            state: 오케스트레이터 상태
        """
        self.decision_maker = decision_maker
        self.tool_coordinator = tool_coordinator
        self.response_pipeline = response_pipeline
        self.context_gatherer = context_gatherer
        self.cache = cache or ResponseCache()
        self.state = state

        # 통계
        self._stats = {"total_queries": 0, "cache_hits": 0, "tool_calls": 0, "errors": 0}

    async def process(
        self,
        query: str,
        context: Context | None = None,
        session_id: str | None = None,
        current_metrics: dict[str, Any] | None = None,
        skip_cache: bool = False,
    ) -> Response:
        """
        사용자 질문 처리

        Args:
            query: 사용자 질문
            context: 미리 수집된 컨텍스트 (없으면 수집)
            session_id: 세션 ID
            current_metrics: 현재 지표 데이터
            skip_cache: 캐시 스킵 여부

        Returns:
            Response 객체
        """
        start_time = datetime.now()
        self._stats["total_queries"] += 1

        try:
            # 1. 세션 설정
            if session_id and self.state:
                self.state.set_session(session_id)

            # 2. 캐시 확인
            if not skip_cache:
                cached = self.cache.get(query, "query")
                if cached:
                    self._stats["cache_hits"] += 1
                    logger.info(f"Cache hit: {query[:30]}...")
                    return cached

            # 3. 컨텍스트 수집 (없으면)
            if context is None:
                if self.context_gatherer:
                    context = await self.context_gatherer.gather(
                        query=query, current_metrics=current_metrics
                    )
                else:
                    # 기본 컨텍스트
                    context = Context(query=query)

            # 4. 시스템 상태 수집
            system_state = self._get_system_state(current_metrics)

            # 5. LLM 의사결정
            decision = await self.decision_maker.decide(query, context, system_state)

            # 6. 도구 실행 (필요시)
            tool_result = None
            if decision.get("tool") and decision["tool"] != "direct_answer":
                self._stats["tool_calls"] += 1
                tool_result = await self.tool_coordinator.execute(
                    tool_name=decision["tool"], params=decision.get("tool_params", {})
                )

            # 7. 응답 생성
            response = await self.response_pipeline.generate(
                query=query, context=context, decision=decision, tool_result=tool_result
            )

            # 8. 처리 시간 기록
            response.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            # 9. 캐시 저장
            if not skip_cache and not response.is_fallback:
                self.cache.set(query, response, "query")

            return response

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Query processing failed: {e}")
            return Response.fallback(f"처리 중 오류가 발생했습니다: {str(e)}")

    def _get_system_state(self, current_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        """시스템 상태 수집"""
        data_status = "없음"
        data_date = None

        if current_metrics:
            metadata = current_metrics.get("metadata", {})
            data_date = metadata.get("data_date")
            if data_date:
                today = datetime.now().strftime("%Y-%m-%d")
                data_status = "최신" if data_date == today else f"오래됨 ({data_date})"

        available_tools = self.tool_coordinator.get_available_tools()
        failed_tools = self.tool_coordinator.get_failed_tools()

        return {
            "data_status": data_status,
            "data_date": data_date,
            "available_tools": available_tools,
            "failed_tools": failed_tools,
            "mode": "responding",
            "cache_stats": self.cache.get_stats(),
        }

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return {
            **self._stats,
            "decision_maker": self.decision_maker.get_stats(),
            "tool_coordinator": self.tool_coordinator.get_stats(),
        }
