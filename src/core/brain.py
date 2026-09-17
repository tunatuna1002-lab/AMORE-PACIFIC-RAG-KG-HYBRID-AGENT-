"""
통합 두뇌 (Unified Brain) - Facade Pattern
==========================================
Level 4 Autonomous Agent의 핵심 두뇌

역할:
1. 모든 에이전트를 통제하는 단일 중앙 제어
2. LLM 기반 의사결정 (LLM-First 방식)
3. 자율 작업과 사용자 요청 우선순위 관리
4. 상태 관리 및 이벤트 처리

아키텍처 (SRP 분해):
- DecisionMaker: LLM 의사결정 (decision_maker.py)
- ToolCoordinator: 도구 실행 조율 (tool_coordinator.py)
- AlertManager: 알림 처리 (alert_manager.py)
- ContextGatherer: RAG + KG 컨텍스트 수집 (context_gatherer.py)
- ResponsePipeline: 응답 생성 (response_pipeline.py)
- UnifiedBrain: Facade (위 컴포넌트 조율)

동작 모드:
- 자율 모드: 스케줄 기반 자동 작업 (크롤링, 분석, 알림)
- 대화 모드: 사용자 질문 처리 (최우선)
- 알림 모드: 이벤트 기반 알림 생성

주요 책임:
- Query Processing: process_query(), process_query_stream() - 사용자 질문 처리
- Autonomous Scheduling: run_autonomous_cycle() - 자율 작업 실행
- Event Management: emit_event() - 이벤트 발생 및 핸들러 호출
- Component Coordination: Facade 패턴으로 내부 컴포넌트 조율
- ReAct Integration: 복잡한 질문 감지 및 ReAct 모드 라우팅
- KG Sync: Knowledge Graph 동기화
- Market Intelligence: collect_market_intelligence() - 시장 정보 수집
- Newsletter: _send_morning_brief() - 아침 브리핑 이메일 발송

Usage:
    brain = UnifiedBrain()
    await brain.initialize()

    # 사용자 질문 처리
    response = await brain.process_query("LANEIGE 순위 알려줘")

    # 자율 작업 실행
    await brain.run_autonomous_cycle()
"""

import asyncio
import heapq
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

# Resolve data directory: Railway volume at /data/ vs local ./data/
_DATA_DIR = "/data" if os.path.isdir("/data") else "./data"

# SRP 분해된 컴포넌트
from .alert_manager import AlertManager
from .cache import ResponseCache
from .confidence import ConfidenceAssessor
from .context_gatherer import ContextGatherer
from .decision_maker import DecisionMaker
from .models import Response
from .query_graph import REACT_MODE_OFF, REACT_MODE_ON, REACT_MODE_SHADOW, QueryGraph
from .response_pipeline import ResponsePipeline
from .router import HopRouter
from .scheduler import AutonomousScheduler
from .state import OrchestratorState
from .tool_coordinator import ToolCoordinator
from .tool_registry import ToolRegistry

# Type checking imports (순환 참조 방지)
if TYPE_CHECKING:
    from ..tools.intelligence.market_intelligence import MarketIntelligenceEngine
    from .graph_state import QueryState

# AlertAgent는 TYPE_CHECKING에서만 임포트 (순환 import 방지)
from src.shared.constants import DEFAULT_MODEL
from src.tools.calculators.metric_calculator import (
    calculate_hhi_from_counts,
    count_brands,
    hhi_to_points,
)

from ..core.state_manager import StateManager

logger = logging.getLogger(__name__)

# QueryGraph의 경로(route_trace.route) → SSE done 이벤트의 ``mode``.
# v3부터의 계약이 ReAct 여부만 구분하므로 나머지는 모두 "direct"다
# (세부 경로는 done.metadata.route_trace에 그대로 실린다).
_STREAM_MODE_BY_ROUTE = {"react": "react", "blocked": "blocked"}


# =============================================================================
# 타입 정의
# =============================================================================


class BrainMode(Enum):
    """두뇌 동작 모드"""

    IDLE = "idle"  # 대기
    AUTONOMOUS = "autonomous"  # 자율 작업 중
    RESPONDING = "responding"  # 사용자 응답 중
    EXECUTING = "executing"  # 에이전트 실행 중
    ALERTING = "alerting"  # 알림 처리 중


class TaskPriority(Enum):
    """작업 우선순위"""

    USER_REQUEST = 0  # 사용자 요청 (최우선)
    CRITICAL_ALERT = 1  # 중요 알림
    SCHEDULED = 2  # 예약 작업
    BACKGROUND = 3  # 백그라운드 작업


@dataclass
class BrainTask:
    """두뇌가 처리할 작업"""

    id: str
    type: str  # query, scheduled, alert, autonomous
    priority: TaskPriority
    payload: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any | None = None
    error: str | None = None

    def __lt__(self, other):
        """우선순위 기반 정렬"""
        return self.priority.value < other.priority.value


# =============================================================================
# 통합 두뇌 (Facade Pattern)
# =============================================================================


class UnifiedBrain:
    """
    Level 4 Autonomous Agent의 통합 두뇌 (LLM-First)

    Facade 패턴으로 구현되어 내부 컴포넌트들을 조율합니다.

    핵심 원칙:
    1. 사용자 요청은 항상 최우선
    2. 모든 판단은 LLM이 담당 (LLM-First)
    3. 자율 스케줄링으로 데이터 자동 수집
    4. 이벤트 기반 알림 시스템

    내부 컴포넌트 (SRP):
    - DecisionMaker: LLM 의사결정 전담
    - ToolCoordinator: 도구 실행 조율
    - AlertManager: 알림 처리 전담
    """

    # LLM 판단 프롬프트 (DecisionMaker와 공유, 하위호환성 유지)
    DECISION_PROMPT = DecisionMaker.DECISION_PROMPT

    def __init__(
        self,
        context_gatherer: ContextGatherer | None = None,
        tool_executor: ToolRegistry | None = None,
        response_pipeline: ResponsePipeline | None = None,
        cache: ResponseCache | None = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 2,
    ):
        """
        Args:
            context_gatherer: 컨텍스트 수집기
            tool_executor: 도구 레지스트리 (읽기 전용 도구 5종)
            response_pipeline: 응답 파이프라인
            cache: 응답 캐시
            model: LLM 모델
            max_retries: 최대 재시도 횟수
        """
        # 공유 컴포넌트
        self.cache = cache or ResponseCache()
        self.confidence_assessor = ConfidenceAssessor()
        self.state = OrchestratorState()

        # LLM 설정
        self.model = model
        self.max_retries = max_retries

        # 외부 주입 또는 기본 컴포넌트
        self._context_gatherer = context_gatherer
        # 읽기 전용 도구 레지스트리 — DecisionMaker·ToolCoordinator·ReAct가 같은 객체를 본다
        self._tool_executor = tool_executor or ToolRegistry()
        self._response_pipeline = response_pipeline

        # SRP 분해된 내부 컴포넌트 (lazy init)
        self._decision_maker: DecisionMaker | None = None
        self._tool_coordinator: ToolCoordinator | None = None
        self._alert_manager: AlertManager | None = None

        # 자율 스케줄러
        self.scheduler = AutonomousScheduler()

        # 모드 및 상태
        self.mode = BrainMode.IDLE
        self._current_task: BrainTask | None = None

        # 작업 큐 (우선순위 힙)
        self._task_queue: list[BrainTask] = []
        self._task_history: list[BrainTask] = []

        # 이벤트 콜백
        self._event_handlers: dict[str, list[Callable]] = {}

        # 에이전트 (lazy init)
        self._workflow_agent = None
        self._alert_agent = None
        self._react_agent = None
        # 홉 수 라우터 — LLM 폴백 캐시를 공유하려고 Brain 수명 동안 하나만 둔다
        self._router = HopRouter()
        self._react_mode = REACT_MODE_OFF

        # Market Intelligence Engine (lazy init)
        self._market_intelligence: MarketIntelligenceEngine | None = None

        # Query processing graph (initialized in initialize())
        self._query_graph: QueryGraph | None = None

        # 통계
        self._stats = {
            "total_queries": 0,
            "llm_decisions": 0,
            "cache_hits": 0,
            "autonomous_tasks": 0,
            "alerts_generated": 0,
            "errors": 0,
        }

        # 선택 컴포넌트 활성 상태 (/api/v4/brain/status로 노출)
        self._component_status: dict[str, dict[str, Any]] = {
            name: {"enabled": False, "active": False, "error": None} for name in ("react_agent",)
        }

        # 초기화 플래그
        self._initialized = False

    # =========================================================================
    # Properties - SRP 컴포넌트 접근 (lazy init)
    # =========================================================================

    @property
    def context_gatherer(self) -> ContextGatherer | None:
        """컨텍스트 수집기"""
        return self._context_gatherer

    @context_gatherer.setter
    def context_gatherer(self, value: ContextGatherer | None) -> None:
        """컨텍스트 수집기 설정"""
        self._context_gatherer = value

    @property
    def tool_executor(self) -> ToolRegistry:
        """도구 레지스트리 (읽기 전용 도구 5종)"""
        return self._tool_executor

    @property
    def response_pipeline(self) -> ResponsePipeline | None:
        """응답 파이프라인"""
        return self._response_pipeline

    @property
    def decision_maker(self) -> DecisionMaker:
        """의사결정 컴포넌트 (lazy init)"""
        if self._decision_maker is None:
            self._decision_maker = DecisionMaker(model=self.model)
        return self._decision_maker

    @property
    def tool_coordinator(self) -> ToolCoordinator:
        """도구 조율 컴포넌트 (lazy init)"""
        if self._tool_coordinator is None:
            self._tool_coordinator = ToolCoordinator(
                tool_executor=self._tool_executor,
                state=self.state,
                cache=self.cache,
                max_retries=self.max_retries,
            )
        return self._tool_coordinator

    @property
    def alert_manager(self) -> AlertManager:
        """알림 관리 컴포넌트 (lazy init)"""
        if self._alert_manager is None:
            self._alert_manager = AlertManager()
        return self._alert_manager

    # =========================================================================
    # 초기화
    # =========================================================================

    async def initialize(self) -> None:
        """비동기 초기화"""
        if self._initialized:
            return

        # Context Gatherer 초기화
        if self._context_gatherer:
            await self._context_gatherer.initialize()
        else:
            # 기본 Context Gatherer 생성
            from ..ontology.knowledge_graph import KnowledgeGraph

            # auto-detects Railway volume vs local path.
            # 서버는 KG 읽기 전용: 정식 기록자는 daily_crawl(exporter)이며,
            # 장시간 상주하는 서버가 stale 메모리 상태로 파일을 덮어쓰는 것을 방지
            kg = KnowledgeGraph(auto_save=False)
            self._knowledge_graph = kg

            from ..ontology.reasoner import OntologyReasoner
            from ..rag.hybrid_retriever import HybridRetriever

            reasoner = OntologyReasoner(kg)
            hybrid_retriever = HybridRetriever(knowledge_graph=kg, reasoner=reasoner)

            logger.info("UnifiedBrain: HybridRetriever initialized")

            self._context_gatherer = ContextGatherer(
                hybrid_retriever=hybrid_retriever, orchestrator_state=self.state
            )
            await self._context_gatherer.initialize()

        # 초기 KG 동기화 (대시보드 데이터 있으면)
        try:
            import json as _json

            data_path = os.environ.get("DASHBOARD_DATA_PATH", f"{_DATA_DIR}/dashboard_data.json")
            with open(data_path, encoding="utf-8") as f:
                initial_data = _json.load(f)
            self._sync_knowledge_graph(initial_data)
        except FileNotFoundError:
            logger.debug("No dashboard data for initial KG sync")
        except Exception as e:
            logger.warning(f"Initial KG sync failed: {e}")

        # Response Pipeline 초기화
        if not self._response_pipeline:
            self._response_pipeline = ResponsePipeline()

        # AlertAgent 초기화 (지연 임포트로 순환 import 방지)
        try:
            from ..agents.alert_agent import AlertAgent

            state_manager = StateManager()
            self._alert_agent = AlertAgent(state_manager)
            logger.info("AlertAgent initialized successfully")
        except Exception as e:
            logger.warning(f"AlertAgent initialization failed: {e}")
            self._alert_agent = None

        # AlertManager 초기화
        await self.alert_manager.initialize()

        # 도구 레지스트리 배선 (ReAct 실행기가 같은 레지스트리를 쓰므로 먼저 한다)
        self._bind_tool_registry()

        # ReActAgent 초기화 (피처 플래그)
        self._init_react_agent()

        # crawl_complete 이벤트 시 KG 동기화
        async def _on_crawl_complete(event_data: dict[str, Any]) -> None:
            try:
                import json as _json

                data_path = os.environ.get(
                    "DASHBOARD_DATA_PATH", f"{_DATA_DIR}/dashboard_data.json"
                )
                with open(data_path, encoding="utf-8") as f:
                    data = _json.load(f)
                self._sync_knowledge_graph(data)
            except Exception as e:
                logger.warning(f"KG sync on crawl_complete failed: {e}")

        self.on_event("crawl_complete", _on_crawl_complete)

        # Initialize query processing graph (컴포넌트 배선이 끝난 뒤 새로 만든다)
        self._query_graph = None
        self._ensure_query_graph()

        self._initialized = True
        logger.info("UnifiedBrain initialized (LLM-First mode, SRP components)")

    def _bind_tool_registry(self) -> None:
        """읽기 전용 도구 레지스트리를 검색기에 연결한다 (트랙 4-A).

        예전에는 여기서 대시보드 JSON(dashboard_data.json)을 읽는 도구 5종
        (get_brand_status·get_product_info·get_competitor_analysis·get_category_info·
        get_action_items)을 등록했다. 그 JSON은 날짜가 없는 캐시라 수치 근거가 될 수 없었고
        (E2와 같은 이유), ReAct는 또 다른 도구 3종을 따로 갖고 있어 경로마다 도구가 달랐다.
        이제 도구는 ``tool_registry``의 5종뿐이고, 백엔드는 검색과 같은 HybridRetriever다.
        """
        retriever = getattr(self._context_gatherer, "retriever", None)
        if retriever is None:
            logger.warning("Tool registry not bound: context gatherer has no retriever")
            return
        if isinstance(self._tool_executor, ToolRegistry):
            self._tool_executor.bind(retriever)
            logger.info(
                f"Tool registry bound: {', '.join(self._tool_executor.get_available_tools())}"
            )

    def _init_react_agent(self) -> None:
        """ReActAgent를 공용 도구 레지스트리(읽기 전용 5종)와 함께 연결한다.

        이전 코드는 존재하지 않는 `..agents.react_agent`를 import해 추가된 날(a965437)부터
        항상 None이었고, 예외는 debug 로그로 삼켜졌다.

        섀도 모드(``agents.react_shadow_mode``, 트랙 5-C)에서도 에이전트는 만들어야 한다 —
        답변은 파이프라인이 내지만 ReAct를 같이 돌려 기록을 남기기 때문이다. 이때 모드는
        ``shadow``이고 QueryGraph는 ReAct 경로로 분기하지 않는다.
        """
        from ..infrastructure.feature_flags import FeatureFlags

        flags = FeatureFlags.get_instance()
        use_react = bool(flags.use_react_agent())
        shadow = bool(flags.react_shadow_mode()) and not use_react
        self._react_mode = (
            REACT_MODE_ON if use_react else (REACT_MODE_SHADOW if shadow else REACT_MODE_OFF)
        )

        status = self._component_status["react_agent"]
        status.update(enabled=use_react, active=False, error=None, mode=self._react_mode)
        self._react_agent = None
        if not (use_react or shadow):
            return

        try:
            from .react_agent import ALLOWED_ACTIONS, ReActAgent
            from .react_tools import build_react_tool_executor

            executor = build_react_tool_executor(self._tool_executor)
            missing = (ALLOWED_ACTIONS - {"final_answer", "refine_search"}) - set(
                executor.get_available_tools()
            )
            if missing:
                logger.warning(f"ReActAgent: allowed actions without executor: {sorted(missing)}")

            # 싱글톤을 쓰지 않는다: 도구 실행기는 이 Brain의 레지스트리(=검색기)에 묶여 있다
            agent = ReActAgent()
            agent.set_tool_executor(executor)
            self._react_agent = agent
            status["active"] = True
            logger.info("ReActAgent initialized with read-only tools")
        except Exception as e:
            status["error"] = f"{type(e).__name__}: {e}"
            logger.warning(
                f"ReActAgent enabled by flag but failed to initialize: {status['error']}"
            )

    def get_component_status(self) -> dict[str, dict[str, Any]]:
        """선택 컴포넌트(ReAct)의 플래그·활성·오류 상태."""
        return {name: dict(status) for name, status in self._component_status.items()}

    # =========================================================================
    # 이벤트 시스템
    # =========================================================================

    def on_event(self, event_name: str, handler: Callable) -> None:
        """이벤트 핸들러 등록"""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)

    async def emit_event(self, event_name: str, data: dict[str, Any] | None = None) -> None:
        """이벤트 발생"""
        data = data or {}
        logger.debug(f"Event emitted: {event_name}")

        # 등록된 핸들러 호출
        handlers = self._event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

        # 알림 조건 체크 (AlertManager에 위임)
        if event_name in ["crawl_complete", "crawl_failed", "metrics_calculated", "rank_changed"]:
            alerts = await self.alert_manager.check_conditions(event_name, data)
            for alert in alerts:
                await self._process_alert(alert)

    async def _process_alert(self, alert: dict[str, Any]) -> None:
        """알림 처리 (AlertManager에 위임)"""
        self._stats["alerts_generated"] += 1
        await self.emit_event("alert_generated", alert)
        await self.alert_manager.process_alert(alert)

    # =========================================================================
    # 사용자 질문 처리 (최우선)
    # =========================================================================

    def _ensure_query_graph(self) -> QueryGraph:
        """질의 처리 그래프 (분기 구현은 이 그래프 하나뿐)

        테스트가 ``_initialized=True``만 직접 설정하는 경우를 위해 지연 생성도 지원한다.
        """
        if self._query_graph is None:
            self._query_graph = QueryGraph(
                cache=self.cache,
                context_gatherer=self._context_gatherer,
                confidence_assessor=self.confidence_assessor,
                decision_maker=self.decision_maker,
                tool_coordinator=self.tool_coordinator,
                response_pipeline=self._response_pipeline,
                react_agent=self._react_agent,
                router=self._router,
                react_mode=self._react_mode,
            )
        return self._query_graph

    def _new_query_state(
        self,
        query: str,
        session_id: str | None,
        current_metrics: dict[str, Any] | None,
        skip_cache: bool,
    ) -> "QueryState":
        """두 경로가 공유하는 초기 QueryState 생성 (+ 세션 설정)"""
        from .graph_state import QueryState

        if session_id:
            self.state.set_session(session_id)

        return QueryState(
            query=query,
            session_id=session_id,
            current_metrics=current_metrics,
            skip_cache=skip_cache,
            system_state=self._get_system_state(current_metrics),
        )

    def _finalize_query(self, state: "QueryState", query: str, start_time: datetime) -> Response:
        """그래프 실행 결과 → 최종 Response (통계·처리 시간·캐시 기록 공통 처리)"""
        if state.response:
            if state.metadata.get("cache_hit"):
                self._stats["cache_hits"] += 1
            if state.decision and state.decision.tool != "direct_answer":
                self._stats["llm_decisions"] += 1

        response = state.response or Response.fallback("처리 결과가 없습니다.")
        response.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        if not state.skip_cache and not response.is_fallback:
            self.cache.set(query, response, "query")

        return response

    async def process_query(
        self,
        query: str,
        session_id: str | None = None,
        current_metrics: dict[str, Any] | None = None,
        skip_cache: bool = False,
    ) -> Response:
        """
        사용자 질문 처리 (최우선)

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

        # 초기화 확인
        if not self._initialized:
            await self.initialize()

        # 사용자 요청 이벤트 발생
        await self.emit_event("user_request", {"query": query})

        # 모드 전환: 사용자 응답 최우선
        previous_mode = self.mode
        self.mode = BrainMode.RESPONDING

        try:
            state = self._new_query_state(query, session_id, current_metrics, skip_cache)
            await self._ensure_query_graph().run(state)
            return self._finalize_query(state, query, start_time)

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Query processing failed: {e}")
            await self.emit_event("error_occurred", {"error": str(e), "query": query})
            return Response.fallback(f"처리 중 오류가 발생했습니다: {str(e)}")

        finally:
            self.mode = previous_mode

    # =========================================================================
    # 스트리밍 응답 (Phase 5: v3에서 포팅)
    # =========================================================================

    async def process_query_stream(
        self,
        query: str,
        session_id: str | None = None,
        current_metrics: dict[str, Any] | None = None,
    ):
        """
        SSE 스트리밍 방식으로 질문 처리

        ``process_query``와 **같은 QueryGraph**를 실행하고, 그래프가 노드마다 내보내는
        진행 이벤트를 그대로 흘려보낸다. 분기(가드 → 캐시 → 컨텍스트 → 신뢰도 → 도구 →
        생성)는 그래프에만 있다 — 예전처럼 여기에 복제해 두지 않는다.

        토큰 단위 스트리밍은 아니다: 답변은 ``generate``가 끝난 뒤 한 덩어리로 나간다.

        Yields:
            dict: {"type": "status"|"tool_call"|"text"|"done"|"error", "content": ...}
        """
        start_time = datetime.now()
        self._stats["total_queries"] += 1

        # 초기화 확인
        if not self._initialized:
            await self.initialize()

        # 모드 전환
        previous_mode = self.mode
        self.mode = BrainMode.RESPONDING

        try:
            state = self._new_query_state(query, session_id, current_metrics, skip_cache=False)

            async for event in self._ensure_query_graph().stream(state):
                if event["type"] == "tool_call":
                    self.mode = BrainMode.EXECUTING
                yield event

            response = self._finalize_query(state, query, start_time)
            route = (state.metadata.get("route_trace") or {}).get("route")

            # 텍스트 청크로 yield (출력 가드는 그래프의 output_guard 노드가 이미 적용)
            yield {"type": "text", "content": response.text}

            # 완료 이벤트 (content는 dict - chat 라우트에서 json.dumps 처리)
            yield {"type": "done", "content": self._stream_done_content(response, route)}

        except Exception as e:
            logger.error(f"Stream processing failed: {e}", exc_info=True)
            self._stats["errors"] += 1
            yield {"type": "error", "content": str(e)}
            # 에러 후에도 done 이벤트 전송 → 프론트엔드 로딩 해제
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            yield {
                "type": "done",
                "content": {
                    "confidence": 0.0,
                    "sources": [],
                    "tools_used": [],
                    "suggestions": ["다시 질문해주세요"],
                    "processing_time_ms": round(processing_time, 1),
                    "mode": "error",
                    "confidence_level": "unknown",
                    "metadata": {},
                },
            }

        finally:
            self.mode = previous_mode

    @staticmethod
    def _stream_done_content(response: Response, route: str | None) -> dict[str, Any]:
        """``done`` 이벤트 내용 (v3부터의 SSE 계약 유지 + metadata 추가)

        ``mode``는 예전처럼 ReAct 여부만 구분한다 — 세부 경로는 ``metadata.route_trace``에
        있다. 차단 응답만 신뢰도·출처가 의미 없어 고정값을 쓴다.
        """
        trace = (response.metadata or {}).get("route_trace") or {}

        if route == "blocked":
            content: dict[str, Any] = {
                "confidence": 0.0,
                "sources": [],
                "tools_used": [],
                "suggestions": ["다른 질문을 해주세요"],
                "confidence_level": "unknown",
            }
        else:
            content = {
                "confidence": response.confidence_score,
                "sources": response.sources[:5] if response.sources else [],
                "tools_used": response.tools_called,
                "suggestions": response.suggestions[:3] if response.suggestions else [],
                "confidence_level": trace.get("confidence_level") or "medium",
            }

        content["processing_time_ms"] = round(response.processing_time_ms or 0.0, 1)
        content["mode"] = _STREAM_MODE_BY_ROUTE.get(route, "direct")
        # 관측 정보 (route_trace·numeric_verification). 대시보드는 무시해도 안전하다.
        content["metadata"] = response.metadata or {}
        return content

    # =========================================================================
    # KG 동기화 (Phase 4: v3에서 포팅)
    # =========================================================================

    def _sync_knowledge_graph(self, data: dict[str, Any]) -> None:
        """
        크롤링 데이터 → KG 동기화

        대시보드 데이터의 브랜드 메트릭을 KnowledgeGraph 엔티티 메타데이터로
        동기화합니다.

        Args:
            data: 대시보드 JSON 데이터 (brand.competitors 포함)
        """
        if not hasattr(self, "_knowledge_graph") or self._knowledge_graph is None:
            return

        try:
            # 브랜드 메트릭 → KnowledgeGraph
            brand_metrics = data.get("brand", {}).get("competitors", [])
            for brand_info in brand_metrics:
                brand_name = brand_info.get("brand")
                if brand_name:
                    self._knowledge_graph.set_entity_metadata(
                        entity=brand_name,
                        metadata={
                            "type": "brand",
                            "sos": brand_info.get("sos", 0) / 100,
                            "avg_rank": brand_info.get("avg_rank"),
                            "product_count": brand_info.get("products", 0),
                        },
                    )

            logger.info(f"KG synced: {len(brand_metrics)} brands")
        except Exception as e:
            logger.warning(f"KG sync failed: {e}")

    # =========================================================================
    # 자율 작업 (Autonomous)
    # =========================================================================

    async def run_autonomous_cycle(self) -> dict[str, Any]:
        """
        자율 작업 사이클 실행

        스케줄된 작업을 확인하고 실행합니다.
        사용자 요청이 들어오면 즉시 중단하고 우선 처리합니다.
        """
        if self.mode == BrainMode.RESPONDING:
            return {"status": "skipped", "reason": "사용자 응답 중"}

        self._stats["autonomous_tasks"] += 1
        self.mode = BrainMode.AUTONOMOUS

        results = {
            "started_at": datetime.now().isoformat(),
            "tasks_executed": [],
            "status": "completed",
        }

        try:
            # 1. 스케줄된 작업 확인
            due_tasks = self.scheduler.get_due_tasks(datetime.now())

            for task in due_tasks:
                # 사용자 요청 모드면 중단
                if self.mode == BrainMode.RESPONDING:
                    break

                task_result = await self._execute_scheduled_task(task)
                results["tasks_executed"].append(task_result)

                # 완료 마킹
                self.scheduler.mark_completed(task["id"])

            # 2. 대기 중인 작업 큐 처리
            await self._process_task_queue()

        except Exception as e:
            logger.error(f"Autonomous cycle error: {e}")
            results["status"] = "error"
            results["error"] = str(e)

        finally:
            self.mode = BrainMode.IDLE
            results["completed_at"] = datetime.now().isoformat()

        return results

    async def _execute_scheduled_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """스케줄된 작업 실행"""
        action = task.get("action")
        task_name = task.get("name", action)

        logger.info(f"Executing scheduled task: {task_name}")

        try:
            if action == "crawl_workflow":
                # BatchWorkflow 호출
                if not self._workflow_agent:
                    from src.application.workflows.batch_workflow import BatchWorkflow

                    self._workflow_agent = BatchWorkflow()

                result = await self._workflow_agent.run_daily_workflow()

                # crawl_complete 발화는 BatchWorkflow 완료 지점에서 한다 (D4).
                # 여기서 또 쏘면 스케줄러 경로에서만 이벤트가 두 번 나간다.

                # Market Intelligence 데이터 수집
                mi_result = await self.collect_market_intelligence()
                result["market_intelligence"] = mi_result

                return {
                    "task": task_name,
                    "status": "completed",
                    "result": result.get("summary"),
                    "market_intelligence": mi_result,
                }

            elif action == "check_data":
                needs_crawl = self.state.is_crawl_needed()
                return {"task": task_name, "status": "completed", "needs_crawl": needs_crawl}

            elif action == "check_integrity":
                return await self._run_integrity_check(task_name)

            else:
                return {
                    "task": task_name,
                    "status": "skipped",
                    "reason": f"Unknown action: {action}",
                }

        except Exception as e:
            logger.error(f"Scheduled task failed: {e}")
            return {"task": task_name, "status": "failed", "error": str(e)}

    async def _run_integrity_check(self, task_name: str) -> dict[str, Any]:
        """데이터 정합성 검사 실행. CRITICAL이면 알림까지 발화한다 (§3.3).

        run_full_check()는 구현돼 있었으나 __main__ 외 호출처가 0건이었다.
        """
        from src.tools.utilities.data_integrity_checker import check_data_integrity

        result = await check_data_integrity()
        severity = result.get("severity", "OK")
        logger.info(f"Data integrity check: severity={severity}")

        if severity == "CRITICAL":
            recommendations = result.get("recommendations", [])
            await self._process_alert(
                {
                    "type": "data_integrity",
                    "severity": "critical",
                    "message": (
                        f"데이터 정합성 CRITICAL: "
                        f"누락 {len(result.get('missing_dates', []))}일, "
                        f"gap {result.get('sync_status', {}).get('gap', 0)}건"
                    ),
                    "details": "\n".join(recommendations),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            "task": task_name,
            "status": "completed",
            "severity": severity,
            "result": result,
        }

    async def _process_task_queue(self) -> None:
        """작업 큐 처리"""
        while self._task_queue:
            if self.mode == BrainMode.RESPONDING:
                break

            task = heapq.heappop(self._task_queue)
            await self._execute_queued_task(task)

    async def _execute_queued_task(self, task: BrainTask) -> None:
        """큐 작업 실행"""
        task.started_at = datetime.now()
        self._current_task = task

        try:
            if task.type == "alert":
                await self._process_alert(task.payload)
                task.result = {"processed": True}

            task.completed_at = datetime.now()
            self._task_history.append(task)

        except Exception as e:
            task.error = str(e)
            logger.error(f"Queued task failed: {e}")

        finally:
            self._current_task = None

    def add_task(self, task: BrainTask) -> None:
        """작업 큐에 추가"""
        heapq.heappush(self._task_queue, task)

    # =========================================================================
    # 알림 처리 (AlertManager에 위임)
    # =========================================================================

    async def collect_market_intelligence(self) -> dict[str, Any]:
        """Market Intelligence 데이터 수집"""
        logger.info("Starting Market Intelligence collection...")

        try:
            if not self._market_intelligence:
                from ..tools.intelligence.market_intelligence import MarketIntelligenceEngine

                self._market_intelligence = MarketIntelligenceEngine()
                await self._market_intelligence.initialize()

            layer_data = await self._market_intelligence.collect_all_layers()
            self._market_intelligence.save_data()

            result = {
                "status": "success",
                "layers_collected": list(layer_data.keys()),
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Market Intelligence collection completed: {result['layers_collected']}")
            await self.emit_event("market_intelligence_collected", result)

            return result

        except Exception as e:
            logger.error(f"Market Intelligence collection failed: {e}")
            return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}

    async def check_alerts(self, metrics_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        지표 데이터에서 알림 조건 확인 (AlertManager에 위임)

        Args:
            metrics_data: 지표 데이터

        Returns:
            발생한 알림 목록
        """
        return await self.alert_manager.check_metrics_alerts(metrics_data)

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def _get_system_state(self, current_metrics: dict | None = None) -> dict[str, Any]:
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
            "mode": self.mode.value,
            "cache_stats": self.cache.get_stats(),
        }

    def _format_system_state(self, state: dict[str, Any]) -> str:
        """시스템 상태 포맷"""
        lines = [
            f"- 데이터 상태: {state['data_status']}",
            f"- 동작 모드: {state['mode']}",
            f"- 사용 가능 도구: {', '.join(state['available_tools'])}",
        ]
        if state["failed_tools"]:
            lines.append(f"- 실패 도구: {', '.join(state['failed_tools'])}")
        return "\n".join(lines)

    # =========================================================================
    # 상태 및 통계
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return {
            **self._stats,
            "mode": self.mode.value,
            "queue_size": len(self._task_queue),
            "scheduled_tasks": len(self.scheduler.schedules),
            "components": {
                "decision_maker": self.decision_maker.get_stats() if self._decision_maker else {},
                "tool_coordinator": self.tool_coordinator.get_stats()
                if self._tool_coordinator
                else {},
                "alert_manager": self.alert_manager.get_stats() if self._alert_manager else {},
            },
        }

    def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """최근 에러 목록"""
        return self.tool_coordinator.get_recent_errors(limit)

    def get_state_summary(self) -> str:
        """상태 요약"""
        return f"{self.mode.value} | {self.state.to_context_summary()}"

    def reset_failed_agents(self) -> None:
        """실패한 에이전트 목록 초기화"""
        self.tool_coordinator.reset_failed_tools()
        logger.info("Failed agents list cleared")

    # =========================================================================
    # Morning Brief (뉴스레터)
    # =========================================================================

    async def _send_morning_brief(self) -> None:
        """
        Morning Brief 뉴스레터 생성 및 발송

        매일 아침 8시 KST에 자동 실행됩니다.
        전날 크롤링 데이터를 기반으로 시장 현황을 요약합니다.

        인사이트 리포트 이메일도 함께 발송합니다.
        """

        logger.info("Generating Morning Brief...")

        try:
            # 1. 최신 크롤링 데이터 가져오기
            from src.tools.intelligence.market_intelligence import MarketIntelligenceEngine

            mi = MarketIntelligenceEngine()
            latest_data = await mi.get_latest_data()

            products = latest_data.get("products", [])
            crawl_data = {
                "products": products,
                "category": "All Categories",
            }

            # 2. Morning Brief 생성
            from src.tools.intelligence.morning_brief import (
                MorningBriefGenerator,
                render_morning_brief_html,
            )

            generator = MorningBriefGenerator()
            brief_data = await generator.generate(
                crawl_data=crawl_data,
                metrics_data=latest_data.get("metrics"),
            )

            # 3. HTML 렌더링
            html_content = render_morning_brief_html(brief_data)

            # 4. 이메일 발송
            from src.tools.notifications.email_sender import EmailSender

            sender = EmailSender()

            if not sender.is_enabled():
                logger.warning("Email sender is disabled, Morning Brief not sent")
                return

            # 수신자 목록 (환경변수에서)
            recipients_str = os.getenv("ALERT_RECIPIENTS", "")
            recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]

            if not recipients:
                logger.warning("No recipients configured for Morning Brief")
                return

            # 기존 Morning Brief 발송
            result = await sender.send_morning_brief(
                recipients=recipients,
                html_content=html_content,
                date_str=brief_data.date,
            )

            if result.success:
                logger.info(f"Morning Brief sent successfully to {len(result.sent_to)} recipients")
            else:
                logger.error(f"Morning Brief send failed: {result.message}")

            # 5. 인사이트 리포트 이메일도 발송
            await self._send_insight_report_email(products, recipients, sender)

        except Exception as e:
            logger.error(f"Morning Brief generation failed: {e}")
            self._stats["errors"] += 1

    async def _send_insight_report_email(
        self, products: list, recipients: list[str], sender
    ) -> None:
        """
        인사이트 리포트 이메일 발송 (자동)

        Morning Brief와 함께 발송되는 상세 인사이트 리포트입니다.
        """
        from datetime import datetime

        try:
            logger.info("Sending Insight Report email...")

            # KPI 계산
            laneige_products = [p for p in products if p.get("brand") == "LANEIGE"]
            avg_rank = (
                sum(p.get("rank", 100) for p in laneige_products) / len(laneige_products)
                if laneige_products
                else 0
            )

            # SoS 계산 (Top 100 기준)
            top100 = products[:100]
            laneige_in_top100 = len([p for p in top100 if p.get("brand") == "LANEIGE"])
            sos = (laneige_in_top100 / len(top100) * 100) if top100 else 0

            # HHI 계산 (정본 0-1 → 이메일 표시는 0-10000 포인트)
            hhi = hhi_to_points(calculate_hhi_from_counts(count_brands(top100)))

            # 인사이트 생성 (HybridInsightAgent 사용)
            insight_content = "<p>현재 생성된 인사이트가 없습니다.</p>"
            try:
                from src.agents.hybrid_insight_agent import HybridInsightAgent

                insight_agent = HybridInsightAgent()
                insight_result = await insight_agent.generate_insight(
                    {"products": products[:50], "category": "All Categories"}
                )
                if insight_result and insight_result.get("insight"):
                    raw_insight = insight_result["insight"]
                    insight_content = raw_insight.replace("\n\n", "</p><p>").replace("\n", "<br>")
                    insight_content = f"<p>{insight_content}</p>"
            except Exception as e:
                logger.warning(f"Failed to generate insight for email: {e}")

            # Top 10 제품 데이터
            top10_products = []
            for i, p in enumerate(products[:10]):
                top10_products.append(
                    {
                        "rank": i + 1,
                        "name": p.get("title", "N/A"),
                        "brand": p.get("brand", "Unknown"),
                        "change": p.get("rank_change", 0),
                    }
                )

            # 브랜드별 변동
            brand_changes = []
            for brand in ["LANEIGE", "e.l.f.", "Maybelline", "Summer Fridays", "COSRX"]:
                brand_products = [p for p in products if p.get("brand") == brand]
                if brand_products:
                    avg_change = sum(p.get("rank_change", 0) for p in brand_products) / len(
                        brand_products
                    )
                    if avg_change > 0:
                        brand_changes.append(
                            {
                                "brand": brand,
                                "change_text": f"평균 ▲{avg_change:.1f} 상승",
                                "color": "#28a745",
                            }
                        )
                    elif avg_change < 0:
                        brand_changes.append(
                            {
                                "brand": brand,
                                "change_text": f"평균 ▼{abs(avg_change):.1f} 하락",
                                "color": "#dc3545",
                            }
                        )

            # 리포트 날짜
            report_date = datetime.now().strftime("%Y년 %m월 %d일")

            # 대시보드 URL (Railway 자동 감지)
            def get_base_url() -> str:
                if url := os.getenv("DASHBOARD_URL"):
                    return url.rstrip("/")
                if railway_domain := os.getenv("RAILWAY_PUBLIC_DOMAIN"):
                    return f"https://{railway_domain}"
                return f"http://localhost:{os.getenv('PORT', '8001')}"

            dashboard_url = get_base_url() + "/dashboard"

            # 이메일 발송
            result = await sender.send_insight_report(
                recipients=recipients,
                report_date=report_date,
                avg_rank=avg_rank,
                sos=sos,
                hhi=hhi,
                insight_content=insight_content,
                top10_products=top10_products,
                brand_changes=brand_changes,
                dashboard_url=dashboard_url,
            )

            if result.success:
                logger.info(f"Insight Report sent successfully to {len(result.sent_to)} recipients")
            else:
                logger.error(f"Insight Report send failed: {result.message}")

        except Exception as e:
            logger.error(f"Insight Report email failed: {e}")

    # =========================================================================
    # 스케줄러 관리
    # =========================================================================

    async def start_scheduler(self) -> None:
        """자율 스케줄러 시작"""
        if self.scheduler.running:
            logger.info("Scheduler already running")
            return

        async def _handle_scheduled_task(task: dict[str, Any]):
            """스케줄된 작업 처리"""
            action = task.get("action")
            logger.info(f"Executing scheduled task: {task['name']} ({action})")

            try:
                if action == "crawl_workflow":
                    from src.core.crawl_manager import get_crawl_manager

                    crawl_manager = await get_crawl_manager()

                    if not crawl_manager.is_crawling():
                        await crawl_manager.start_crawl()
                        logger.info("Scheduled crawl started")

                        brain = await get_brain()
                        asyncio.create_task(brain.collect_market_intelligence())
                        logger.info("Market Intelligence collection queued")
                    else:
                        logger.info("Crawl already in progress, skipping")

                elif action == "send_morning_brief":
                    # Morning Brief 뉴스레터 발송
                    await self._send_morning_brief()

                elif action == "check_data":
                    from src.core.crawl_manager import get_crawl_manager

                    crawl_manager = await get_crawl_manager()

                    if crawl_manager.needs_crawl():
                        logger.info("Data is stale, triggering crawl")
                        if not crawl_manager.is_crawling():
                            await crawl_manager.start_crawl()
                    else:
                        logger.info("Data is fresh")

                self._stats["autonomous_tasks"] += 1

            except Exception as e:
                logger.error(f"Scheduled task error: {action} - {e}")
                self._stats["errors"] += 1

                # 크롤 관련 스케줄 태스크 실패는 CRITICAL 알림 대상
                if action in ("crawl_workflow", "check_data"):
                    try:
                        await self.emit_event(
                            "crawl_failed",
                            {"error": str(e), "details": f"스케줄 태스크 '{action}' 실패"},
                        )
                    except Exception as notify_error:
                        logger.error(f"crawl_failed 알림 발화 실패: {notify_error}")

        await self.scheduler.start(_handle_scheduled_task)
        self.mode = BrainMode.AUTONOMOUS
        logger.info("Autonomous scheduler started (KST 22:00 daily crawl = UTC 13:00)")

    def stop_scheduler(self) -> None:
        """자율 스케줄러 중지"""
        self.scheduler.stop()
        self.mode = BrainMode.IDLE
        logger.info("Autonomous scheduler stopped")


# =============================================================================
# 싱글톤 (스레드 안전)
# =============================================================================

_brain_instance: UnifiedBrain | None = None
_brain_lock = asyncio.Lock()


async def get_brain() -> UnifiedBrain:
    """통합 두뇌 싱글톤 반환 (스레드 안전)"""
    global _brain_instance

    async with _brain_lock:
        if _brain_instance is None:
            _brain_instance = UnifiedBrain()
            logger.info("UnifiedBrain singleton created (LLM-First, SRP)")

    return _brain_instance


async def get_initialized_brain() -> UnifiedBrain:
    """초기화된 통합 두뇌 싱글톤 반환"""
    brain = await get_brain()
    if not brain._initialized:
        await brain.initialize()
    return brain


def reset_brain() -> None:
    """싱글톤 리셋 (테스트용)"""
    global _brain_instance
    _brain_instance = None
