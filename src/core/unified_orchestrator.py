"""
통합 오케스트레이터
==================
LLM 기반 단일 오케스트레이터 - 질문을 받아 상황 판단 후 에이전트 호출

핵심 역할:
1. 질문 수신 → 현재 상황/상태 파악
2. LLM이 어떤 에이전트를 호출할지 결정
3. 에이전트 실행 및 에러 처리
4. 응답 생성

에러 전파 전략:
- RETRY: 재시도 (최대 2회)
- FALLBACK: 캐시된 데이터 사용
- SKIP: 건너뛰고 다음 단계
- ABORT: 전체 중단 + 사용자 알림
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field

from .models import Context, Response, Decision, ToolResult, ConfidenceLevel
from .confidence import ConfidenceAssessor
from .cache import ResponseCache
from .state import OrchestratorState
from .context_gatherer import ContextGatherer
from .tools import ToolExecutor, AGENT_TOOLS
from .response_pipeline import ResponsePipeline

logger = logging.getLogger(__name__)


# =============================================================================
# 에러 전략 정의
# =============================================================================

class ErrorStrategy(Enum):
    """에러 발생 시 처리 전략"""
    RETRY = "retry"         # 재시도 (최대 2회)
    FALLBACK = "fallback"   # 캐시/대체 데이터 사용
    SKIP = "skip"           # 건너뛰고 계속
    ABORT = "abort"         # 중단 + 사용자 알림


@dataclass
class AgentError:
    """에이전트 에러 정보"""
    agent_name: str
    error_message: str
    error_type: str  # timeout, connection, validation, unknown
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "message": self.error_message,
            "type": self.error_type,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count
        }


# 에이전트별 에러 전략 매핑
AGENT_ERROR_STRATEGIES: Dict[str, ErrorStrategy] = {
    "crawl_amazon": ErrorStrategy.FALLBACK,      # 크롤링 실패 → 캐시 데이터 사용
    "calculate_metrics": ErrorStrategy.RETRY,     # 계산 실패 → 재시도
    "query_data": ErrorStrategy.FALLBACK,        # 조회 실패 → 캐시 사용
    "query_knowledge_graph": ErrorStrategy.SKIP,  # KG 실패 → 없이 진행
    "generate_insight": ErrorStrategy.RETRY,      # 인사이트 실패 → 재시도
}


# =============================================================================
# 통합 오케스트레이터
# =============================================================================

class UnifiedOrchestrator:
    """
    통합 오케스트레이터

    모든 질문을 LLM이 분석하고, 필요한 에이전트를 호출하는 단일 진입점.

    동작 흐름:
    1. 질문 수신
    2. 자기 상태 검토 (데이터 신선도, 사용 가능 에이전트 등)
    3. LLM이 상황 판단 → 에이전트 선택
    4. 에이전트 실행 (에러 시 전략에 따라 처리)
    5. 응답 생성

    Usage:
        orchestrator = UnifiedOrchestrator()
        await orchestrator.initialize()
        response = await orchestrator.process("LANEIGE SoS가 뭐야?")
    """

    # LLM 판단 프롬프트
    ORCHESTRATION_PROMPT = """당신은 Amazon 마켓 분석 시스템의 오케스트레이터입니다.

## 현재 시스템 상태
{system_state}

## 사용 가능한 도구
{tools_description}

## 수집된 컨텍스트
{context_summary}

## 사용자 질문
{query}

## 지시사항
1. 시스템 상태를 확인하고 질문에 답하기 위해 필요한 것을 파악하세요
2. 컨텍스트만으로 답변 가능하면 "direct_answer"를 선택하세요
3. 데이터가 오래됐거나 없으면 적절한 도구를 선택하세요
4. 에러가 발생한 도구는 피하세요

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
        rag_router: Optional[Any] = None,
        context_gatherer: Optional[ContextGatherer] = None,
        tool_executor: Optional[ToolExecutor] = None,
        response_pipeline: Optional[ResponsePipeline] = None,
        cache: Optional[ResponseCache] = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 2
    ):
        """
        Args:
            rag_router: RAG 라우터
            context_gatherer: 컨텍스트 수집기
            tool_executor: 도구 실행기
            response_pipeline: 응답 생성 파이프라인
            cache: 응답 캐시
            model: LLM 모델
            max_retries: 최대 재시도 횟수
        """
        self.client = None  # OpenAI 클라이언트 (외부 설정)
        self.router = rag_router
        self.context_gatherer = context_gatherer or ContextGatherer()
        self.tool_executor = tool_executor or ToolExecutor()
        self.response_pipeline = response_pipeline or ResponsePipeline()
        self.cache = cache or ResponseCache()
        self.model = model
        self.max_retries = max_retries

        # 상태 관리
        self.state = OrchestratorState()
        self.confidence_assessor = ConfidenceAssessor()

        # 에러 히스토리 (최근 에러 기록)
        self._error_history: List[AgentError] = []
        self._failed_agents: Dict[str, datetime] = {}  # 최근 실패한 에이전트

        # 통계
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "llm_decisions": 0,
            "tools_called": 0,
            "errors_handled": 0,
            "retries": 0
        }

    # =========================================================================
    # 초기화
    # =========================================================================

    def set_client(self, client: Any) -> None:
        """OpenAI 클라이언트 설정"""
        self.client = client
        self.response_pipeline.set_client(client)

    def set_router(self, router: Any) -> None:
        """RAG 라우터 설정"""
        self.router = router

    async def initialize(self) -> None:
        """비동기 초기화"""
        await self.context_gatherer.initialize()
        logger.info("UnifiedOrchestrator initialized")

    # =========================================================================
    # 메인 처리
    # =========================================================================

    async def process(
        self,
        query: str,
        session_id: Optional[str] = None,
        current_metrics: Optional[Dict[str, Any]] = None,
        skip_cache: bool = False
    ) -> Response:
        """
        질문 처리 메인 엔트리

        Args:
            query: 사용자 질문
            session_id: 세션 ID
            current_metrics: 현재 지표 데이터
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
                    logger.info(f"Cache hit: {query[:50]}...")
                    return cached

            # 2. 시스템 상태 점검
            system_state = self._get_system_state(current_metrics)

            # 3. 컨텍스트 수집
            entities = self._extract_entities(query)
            context = await self.context_gatherer.gather(
                query=query,
                entities=entities,
                current_metrics=current_metrics
            )

            # 4. LLM 판단 요청
            decision = await self._get_llm_decision(
                query=query,
                context=context,
                system_state=system_state
            )

            # 5. 도구 실행 (필요시)
            tool_result = None
            if decision.requires_tool():
                tool_result = await self._execute_with_error_handling(
                    tool_name=decision.tool,
                    params=decision.tool_params
                )

            # 6. 응답 생성
            if tool_result and tool_result.success:
                response = await self.response_pipeline.generate_with_tool_result(
                    query=query,
                    context=context,
                    tool_result=tool_result
                )
            else:
                response = await self.response_pipeline.generate(
                    query=query,
                    context=context,
                    decision=decision
                )

            # 처리 시간 기록
            response.processing_time_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            # 캐시 저장
            if not skip_cache and not response.is_fallback:
                self.cache.set(query, response, "query")

            return response

        except Exception as e:
            logger.error(f"Process failed: {e}")
            return Response.fallback(f"처리 중 오류가 발생했습니다: {str(e)}")

    # =========================================================================
    # 시스템 상태 관리
    # =========================================================================

    def _get_system_state(self, current_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        현재 시스템 상태 수집

        Returns:
            상태 정보 딕셔너리
        """
        # 데이터 신선도 판단
        data_status = "없음"
        data_date = None

        if current_metrics:
            metadata = current_metrics.get("metadata", {})
            data_date = metadata.get("data_date")
            if data_date:
                # 오늘 데이터인지 확인
                today = datetime.now().strftime("%Y-%m-%d")
                if data_date == today:
                    data_status = "최신 (오늘)"
                else:
                    data_status = f"오래됨 ({data_date})"
            else:
                data_status = "날짜 불명"

        # 사용 가능한 도구
        available_tools = self.tool_executor.get_available_tools()

        # 최근 실패한 도구 제외
        failed_recently = [
            name for name, time in self._failed_agents.items()
            if (datetime.now() - time).seconds < 300  # 5분 이내 실패
        ]

        return {
            "data_status": data_status,
            "data_date": data_date,
            "available_tools": [t for t in available_tools if t not in failed_recently],
            "failed_tools": failed_recently,
            "cache_stats": self.cache.get_stats(),
            "recent_errors": len(self._error_history)
        }

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """엔티티 추출"""
        if self.router:
            return self.router.extract_entities(query)
        return {}

    # =========================================================================
    # LLM 판단
    # =========================================================================

    async def _get_llm_decision(
        self,
        query: str,
        context: Context,
        system_state: Dict[str, Any]
    ) -> Decision:
        """
        LLM에게 어떤 도구를 사용할지 판단 요청

        Args:
            query: 질문
            context: 컨텍스트
            system_state: 시스템 상태

        Returns:
            Decision
        """
        self._stats["llm_decisions"] += 1

        if not self.client:
            return self._rule_based_decision(query, context, system_state)

        try:
            # 프롬프트 구성
            prompt = self.ORCHESTRATION_PROMPT.format(
                system_state=self._format_system_state(system_state),
                tools_description=self._format_tools_description(system_state),
                context_summary=context.summary or "컨텍스트 없음",
                query=query
            )

            # LLM 호출
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )

            # 응답 파싱
            return self._parse_decision(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return self._rule_based_decision(query, context, system_state)

    def _format_system_state(self, state: Dict[str, Any]) -> str:
        """시스템 상태 포맷팅"""
        lines = [
            f"- 데이터 상태: {state['data_status']}",
            f"- 사용 가능 도구: {', '.join(state['available_tools'])}",
        ]
        if state['failed_tools']:
            lines.append(f"- 최근 실패 도구 (사용 불가): {', '.join(state['failed_tools'])}")
        return "\n".join(lines)

    def _format_tools_description(self, state: Dict[str, Any]) -> str:
        """사용 가능한 도구 설명"""
        available = state.get("available_tools", [])
        lines = []
        for name, tool in AGENT_TOOLS.items():
            if name in available:
                lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)

    def _parse_decision(self, response_text: str) -> Decision:
        """LLM 응답 파싱"""
        try:
            # JSON 추출
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                return Decision(
                    tool=data.get("tool"),
                    tool_params=data.get("tool_params", {}),
                    reason=data.get("reason", ""),
                    key_points=data.get("key_points", []),
                    confidence=0.8
                )
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM decision")

        return Decision(tool="direct_answer", reason="파싱 실패", confidence=0.5)

    def _rule_based_decision(
        self,
        query: str,
        context: Context,
        system_state: Dict[str, Any]
    ) -> Decision:
        """
        LLM 없이 규칙 기반 판단 (폴백)
        """
        query_lower = query.lower()

        # 크롤링 요청
        if any(kw in query_lower for kw in ["크롤링", "수집", "업데이트", "refresh"]):
            if "crawl_amazon" in system_state.get("available_tools", []):
                return Decision(tool="crawl_amazon", reason="크롤링 요청 감지", confidence=0.9)

        # 계산 요청
        if any(kw in query_lower for kw in ["계산", "분석", "지표"]):
            if "calculate_metrics" in system_state.get("available_tools", []):
                return Decision(tool="calculate_metrics", reason="지표 계산 요청", confidence=0.8)

        # 컨텍스트가 충분하면 직접 응답
        if context.has_sufficient_context():
            return Decision(tool="direct_answer", reason="컨텍스트 충분", confidence=0.7)

        return Decision(tool="direct_answer", reason="기본 응답", confidence=0.5)

    # =========================================================================
    # 에러 처리가 포함된 도구 실행
    # =========================================================================

    async def _execute_with_error_handling(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> ToolResult:
        """
        에러 전략이 적용된 도구 실행

        Args:
            tool_name: 도구 이름
            params: 파라미터

        Returns:
            ToolResult
        """
        self._stats["tools_called"] += 1
        strategy = AGENT_ERROR_STRATEGIES.get(tool_name, ErrorStrategy.ABORT)
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                result = await self.tool_executor.execute(tool_name, params)

                if result.success:
                    # 성공 시 실패 기록 제거
                    self._failed_agents.pop(tool_name, None)
                    return result

                # 실패 처리
                error = AgentError(
                    agent_name=tool_name,
                    error_message=result.error or "Unknown error",
                    error_type=self._classify_error(result.error),
                    retry_count=retry_count
                )

                return await self._handle_error(error, strategy, params)

            except asyncio.TimeoutError:
                error = AgentError(
                    agent_name=tool_name,
                    error_message="Timeout",
                    error_type="timeout",
                    retry_count=retry_count
                )

                if strategy == ErrorStrategy.RETRY and retry_count < self.max_retries:
                    retry_count += 1
                    self._stats["retries"] += 1
                    logger.warning(f"Retrying {tool_name} ({retry_count}/{self.max_retries})")
                    await asyncio.sleep(1)  # 1초 대기 후 재시도
                    continue

                return await self._handle_error(error, strategy, params)

            except Exception as e:
                error = AgentError(
                    agent_name=tool_name,
                    error_message=str(e),
                    error_type="unknown",
                    retry_count=retry_count
                )

                if strategy == ErrorStrategy.RETRY and retry_count < self.max_retries:
                    retry_count += 1
                    self._stats["retries"] += 1
                    logger.warning(f"Retrying {tool_name} ({retry_count}/{self.max_retries})")
                    await asyncio.sleep(1)
                    continue

                return await self._handle_error(error, strategy, params)

        # 최대 재시도 초과
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"최대 재시도 횟수 초과 ({self.max_retries}회)"
        )

    def _classify_error(self, error_msg: Optional[str]) -> str:
        """에러 유형 분류"""
        if not error_msg:
            return "unknown"
        error_lower = error_msg.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "connection" in error_lower or "network" in error_lower:
            return "connection"
        elif "validation" in error_lower or "invalid" in error_lower:
            return "validation"
        else:
            return "unknown"

    async def _handle_error(
        self,
        error: AgentError,
        strategy: ErrorStrategy,
        params: Dict[str, Any]
    ) -> ToolResult:
        """
        에러 전략에 따른 처리

        Args:
            error: 에러 정보
            strategy: 처리 전략
            params: 원래 파라미터

        Returns:
            ToolResult (fallback 또는 실패)
        """
        self._stats["errors_handled"] += 1
        self._error_history.append(error)
        self._failed_agents[error.agent_name] = error.timestamp

        # 히스토리 크기 제한
        if len(self._error_history) > 100:
            self._error_history = self._error_history[-50:]

        logger.warning(
            f"Error in {error.agent_name}: {error.error_message} | "
            f"Strategy: {strategy.value}"
        )

        if strategy == ErrorStrategy.FALLBACK:
            # 캐시된 데이터 사용 시도
            cached = self.cache.get_tool_result(error.agent_name)
            if cached:
                logger.info(f"Using cached result for {error.agent_name}")
                return ToolResult(
                    tool_name=error.agent_name,
                    success=True,
                    data={**cached, "_from_cache": True}
                )
            # 캐시 없으면 실패 반환
            return ToolResult(
                tool_name=error.agent_name,
                success=False,
                error=f"{error.error_message} (캐시 없음)"
            )

        elif strategy == ErrorStrategy.SKIP:
            # 건너뛰기 - 빈 성공 반환
            logger.info(f"Skipping {error.agent_name} due to error")
            return ToolResult(
                tool_name=error.agent_name,
                success=True,
                data={"_skipped": True, "_reason": error.error_message}
            )

        else:  # ABORT
            return ToolResult(
                tool_name=error.agent_name,
                success=False,
                error=error.error_message
            )

    # =========================================================================
    # 상태 및 통계
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            **self._stats,
            "cache_stats": self.cache.get_stats(),
            "error_count": len(self._error_history),
            "failed_agents": list(self._failed_agents.keys())
        }

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 에러 목록"""
        return [e.to_dict() for e in self._error_history[-limit:]]

    def reset_failed_agents(self) -> None:
        """실패한 에이전트 목록 초기화"""
        self._failed_agents.clear()
        logger.info("Failed agents list cleared")

    def get_state_summary(self) -> str:
        """상태 요약 문자열"""
        return self.state.to_context_summary()


# =============================================================================
# 팩토리 함수
# =============================================================================

_orchestrator_instance: Optional[UnifiedOrchestrator] = None


def get_unified_orchestrator() -> UnifiedOrchestrator:
    """
    통합 오케스트레이터 싱글톤 반환

    Usage:
        orchestrator = get_unified_orchestrator()
        response = await orchestrator.process(query)
    """
    global _orchestrator_instance

    if _orchestrator_instance is None:
        from rag.router import RAGRouter
        from rag.hybrid_retriever import HybridRetriever
        from ontology.knowledge_graph import KnowledgeGraph
        from ontology.reasoner import OntologyReasoner

        # 컴포넌트 초기화
        kg = KnowledgeGraph(persist_path="./data/knowledge_graph.json")
        reasoner = OntologyReasoner(kg)
        hybrid_retriever = HybridRetriever(kg, reasoner)

        context_gatherer = ContextGatherer(
            hybrid_retriever=hybrid_retriever,
            orchestrator_state=OrchestratorState()
        )

        _orchestrator_instance = UnifiedOrchestrator(
            rag_router=RAGRouter(),
            context_gatherer=context_gatherer,
            tool_executor=ToolExecutor(),
            response_pipeline=ResponsePipeline(),
            cache=ResponseCache()
        )

        logger.info("UnifiedOrchestrator singleton created")

    return _orchestrator_instance


def reset_orchestrator() -> None:
    """싱글톤 인스턴스 리셋 (테스트용)"""
    global _orchestrator_instance
    _orchestrator_instance = None
