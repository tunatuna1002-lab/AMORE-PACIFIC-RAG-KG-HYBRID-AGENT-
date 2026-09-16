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
- KG/OWL Sync: Knowledge Graph 및 OWL Ontology 동기화
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
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

# Resolve data directory: Railway volume at /data/ vs local ./data/
_DATA_DIR = "/data" if os.path.isdir("/data") else "./data"

# SRP 분해된 컴포넌트
from .alert_manager import AlertManager
from .brain_scheduler import BrainTask, BrainTaskQueue, TaskHandler, TaskPriority
from .cache import ResponseCache
from .confidence import ConfidenceAssessor
from .context_gatherer import ContextGatherer
from .decision_maker import DecisionMaker
from .graph_state import QueryState
from .models import Response
from .query_graph import QueryGraph
from .response_pipeline import ResponsePipeline
from .scheduler import AutonomousScheduler
from .tool_coordinator import ToolCoordinator
from .tools import ToolExecutor

# Type checking imports (순환 참조 방지)
if TYPE_CHECKING:
    from ..tools.intelligence.market_intelligence import MarketIntelligenceEngine

# AlertAgent는 TYPE_CHECKING에서만 임포트 (순환 import 방지)
from src.domain.brand import is_target_brand
from src.shared.constants import DEFAULT_MODEL

from ..core.state_manager import StateManager, get_state_manager

logger = logging.getLogger(__name__)


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


# BrainTask / TaskPriority 의 정의는 ``src.core.brain_scheduler`` 에 있다 (Phase 4).
# 위에서 import 해 두었으므로 기존 ``from src.core.brain import BrainTask`` 도 그대로 동작한다.


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
        tool_executor: ToolExecutor | None = None,
        response_pipeline: ResponsePipeline | None = None,
        cache: ResponseCache | None = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 2,
    ):
        """
        Args:
            context_gatherer: 컨텍스트 수집기
            tool_executor: 도구 실행기
            response_pipeline: 응답 파이프라인
            cache: 응답 캐시
            model: LLM 모델
            max_retries: 최대 재시도 횟수
        """
        # 공유 컴포넌트
        self.cache = cache or ResponseCache()
        self.confidence_assessor = ConfidenceAssessor()
        # 단일 시스템 상태 (F7): BatchWorkflow가 기록, ContextGatherer/ToolCoordinator가 읽음
        self.state: StateManager = get_state_manager()

        # LLM 설정
        self.model = model
        self.max_retries = max_retries

        # 외부 주입 또는 기본 컴포넌트
        self._context_gatherer = context_gatherer
        self._tool_executor = tool_executor or ToolExecutor()
        self._response_pipeline = response_pipeline

        # SRP 분해된 내부 컴포넌트 (lazy init)
        self._decision_maker: DecisionMaker | None = None
        self._tool_coordinator: ToolCoordinator | None = None
        self._alert_manager: AlertManager | None = None

        # 자율 스케줄러
        self.scheduler = AutonomousScheduler()

        # 모드 및 상태
        self.mode = BrainMode.IDLE
        self._tasks = BrainTaskQueue()

        # 작업 큐 (우선순위 힙)

        # 이벤트 콜백
        self._event_handlers: dict[str, list[Callable]] = {}

        # 에이전트 (lazy init)
        self._workflow_agent = None
        self._alert_agent = None
        self._react_agent = None

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
    def tool_executor(self) -> ToolExecutor:
        """도구 실행기"""
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
                state_manager=self.state,
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

            # HybridRetriever with optional OWL strategy
            from ..infrastructure.feature_flags import FeatureFlags
            from ..ontology.reasoner import OntologyReasoner
            from ..rag.hybrid_retriever import HybridRetriever

            flags = FeatureFlags.get_instance()
            reasoner = OntologyReasoner(kg)
            owl_strategy = None
            self._owl_reasoner = None

            if flags.use_owl_strategy():
                try:
                    from ..ontology.owl_reasoner import OWLREADY2_AVAILABLE, OWLReasoner

                    if OWLREADY2_AVAILABLE:
                        from ..rag.retrieval_strategy import OWLRetrievalStrategy

                        self._owl_reasoner = OWLReasoner()
                        owl_strategy = OWLRetrievalStrategy(
                            knowledge_graph=kg,
                            owl_reasoner=self._owl_reasoner,
                            docs_path="./docs",
                        )
                        logger.info("UnifiedBrain: OWL strategy enabled")
                except Exception as e:
                    logger.info(f"UnifiedBrain: OWL strategy unavailable ({e})")

            hybrid_retriever = HybridRetriever(
                knowledge_graph=kg,
                reasoner=reasoner,
                owl_strategy=owl_strategy,
            )
            logger.info("UnifiedBrain: HybridRetriever initialized")

            self._context_gatherer = ContextGatherer(
                hybrid_retriever=hybrid_retriever, state_manager=self.state
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

            self._alert_agent = AlertAgent(self.state)
            logger.info("AlertAgent initialized successfully")
        except Exception as e:
            logger.warning(f"AlertAgent initialization failed: {e}")
            self._alert_agent = None

        # AlertManager 초기화
        await self.alert_manager.initialize()

        # ReActAgent 초기화 (ENABLE_REACT_AGENT 플래그로 활성화)
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

        # v3 대시보드 도구 등록
        self._register_dashboard_tools()

        # Initialize query processing graph
        self._query_graph = QueryGraph(
            cache=self.cache,
            context_gatherer=self._context_gatherer,
            confidence_assessor=self.confidence_assessor,
            decision_maker=self.decision_maker,
            tool_coordinator=self.tool_coordinator,
            response_pipeline=self._response_pipeline,
            react_agent=self._react_agent,
        )

        self._initialized = True
        logger.info("UnifiedBrain initialized (LLM-First mode, SRP components)")

    def _init_react_agent(self) -> None:
        """
        ReActAgent 초기화 (opt-in)

        ``ENABLE_REACT_AGENT`` 환경변수가 truthy("true"/"1"/"yes"/"on")일 때만
        ``src.core.react_agent.get_react_agent``를 통해 에이전트를 구성하고
        Brain의 tool executor를 연결합니다. 비활성 시 아무것도 임포트하지 않습니다.
        활성 상태에서 초기화가 실패하면 WARNING으로 기록합니다.
        """
        enabled = os.environ.get("ENABLE_REACT_AGENT", "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not enabled:
            logger.debug("ReActAgent disabled (ENABLE_REACT_AGENT not set)")
            self._react_agent = None
            return

        try:
            from . import react_agent as _react_module

            agent = _react_module.get_react_agent()
            agent.set_tool_executor(self._tool_executor)
            self._react_agent = agent
            logger.info("ReActAgent initialized")
        except Exception as e:
            logger.warning(f"ReActAgent initialization failed (ENABLE_REACT_AGENT=true): {e}")
            self._react_agent = None

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
        if event_name in ["crawl_complete", "metrics_calculated", "rank_changed"]:
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
        """QueryGraph lazy init (테스트에서 _initialized=True 직접 설정 시)"""
        if self._query_graph is None:
            self._query_graph = QueryGraph(
                cache=self.cache,
                context_gatherer=self._context_gatherer,
                confidence_assessor=self.confidence_assessor,
                decision_maker=self.decision_maker,
                tool_coordinator=self.tool_coordinator,
                response_pipeline=self._response_pipeline,
                react_agent=self._react_agent,
            )
        return self._query_graph

    def _build_query_state(
        self,
        query: str,
        session_id: str | None,
        current_metrics: dict[str, Any] | None,
        skip_cache: bool = False,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> QueryState:
        """QueryGraph 입력 상태 구성 (세션 설정 + 시스템 상태 스냅샷 + 이전 턴)"""
        if session_id:
            self.state.set_session(session_id)
        return QueryState(
            query=query,
            session_id=session_id,
            current_metrics=current_metrics,
            skip_cache=skip_cache,
            conversation_history=list(conversation_history or []),
            system_state=self._get_system_state(current_metrics),
        )

    def _record_query_stats(self, state: QueryState) -> None:
        if state.metadata.get("cache_hit"):
            self._stats["cache_hits"] += 1
        if state.route == "decide":
            self._stats["llm_decisions"] += 1

    async def process_query(
        self,
        query: str,
        session_id: str | None = None,
        current_metrics: dict[str, Any] | None = None,
        skip_cache: bool = False,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> Response:
        """
        사용자 질문 처리 (최우선) — ``QueryGraph.run`` 에 위임

        Args:
            query: 사용자 질문
            session_id: 세션 ID
            current_metrics: 현재 지표 데이터
            skip_cache: 캐시 스킵 여부
            conversation_history: 같은 세션의 이전 턴 (API 라우트가 세션 메모리에서 전달, F7)

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
            state = self._build_query_state(
                query, session_id, current_metrics, skip_cache, conversation_history
            )
            state = await self._ensure_query_graph().run(state)
            self._record_query_stats(state)

            response = state.response or Response.fallback("처리 결과가 없습니다.")
            response.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            return response

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Query processing failed: {e}")
            await self.emit_event("error_occurred", {"error": str(e), "query": query})
            return Response.fallback(f"처리 중 오류가 발생했습니다: {str(e)}")

        finally:
            self.mode = previous_mode

    # =========================================================================
    # 스트리밍 응답
    # =========================================================================

    async def process_query_stream(
        self,
        query: str,
        session_id: str | None = None,
        current_metrics: dict[str, Any] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ):
        """
        SSE 스트리밍 방식으로 질문 처리 — ``QueryGraph.run_stream`` 에 위임

        비스트리밍 ``process_query`` 와 같은 그래프(캐시·가드·복합 쿼리 판정 포함)를
        실행하며, 그래프가 emit 하는 토큰/이벤트를 SSE 청크로 yield 합니다.

        Yields:
            dict: {"type": "status"|"tool_call"|"text"|"done"|"error", "content": ...}
        """
        start_time = datetime.now()
        self._stats["total_queries"] += 1

        # 초기화 확인
        if not self._initialized:
            await self.initialize()

        await self.emit_event("user_request", {"query": query})

        # 모드 전환
        previous_mode = self.mode
        self.mode = BrainMode.RESPONDING

        queue: asyncio.Queue = asyncio.Queue()
        _END = object()

        async def on_token(text: str) -> None:
            await queue.put({"type": "text", "content": text})

        async def on_event(event: dict[str, Any]) -> None:
            if event.get("type") == "tool_call":
                self.mode = BrainMode.EXECUTING
            await queue.put(event)

        task: asyncio.Task | None = None
        try:
            state = self._build_query_state(
                query, session_id, current_metrics, conversation_history=conversation_history
            )
            graph = self._ensure_query_graph()
            task = asyncio.create_task(
                graph.run_stream(state, on_token=on_token, on_event=on_event)
            )
            task.add_done_callback(lambda _t: queue.put_nowait(_END))

            while True:
                item = await queue.get()
                if item is _END:
                    break
                yield item

            state = await task  # 그래프 예외는 여기서 재발생
            self._record_query_stats(state)
            response = state.response or Response.fallback("처리 결과가 없습니다.")

            if state.metadata.get("output_sanitized"):
                # 토큰이 이미 전송된 뒤 출력 가드가 텍스트를 정제함:
                # 프론트엔드는 error 이벤트로 본문을 교체하므로 정제된 전문을 보낸다.
                yield {"type": "error", "content": response.text}

            # 완료 이벤트 (content는 dict - 라우트에서 json.dumps 처리)
            yield {
                "type": "done",
                "content": self._stream_done_payload(state, response, start_time),
            }

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
                },
            }

        finally:
            if task is not None and not task.done():
                task.cancel()
            self.mode = previous_mode

    @staticmethod
    def _stream_done_payload(
        state: QueryState, response: Response, start_time: datetime
    ) -> dict[str, Any]:
        """스트리밍 ``done`` 이벤트 페이로드"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        if state.is_blocked:
            mode = "blocked"
            suggestions = ["다른 질문을 해주세요"]
            confidence_level = "unknown"
        else:
            mode = {"react": "react", "cache_hit": "cache"}.get(state.route or "", "direct")
            suggestions = response.suggestions[:3] if response.suggestions else []
            if state.confidence_level:
                confidence_level = state.confidence_level.value
            elif response.confidence_level and response.confidence_level.value != "unknown":
                confidence_level = response.confidence_level.value
            else:
                confidence_level = "medium"
        return {
            "confidence": response.confidence_score,
            "sources": response.sources[:5] if response.sources else [],
            "tools_used": response.tools_called,
            "suggestions": suggestions,
            "processing_time_ms": round(processing_time, 1),
            "mode": mode,
            "confidence_level": confidence_level,
        }

    # =========================================================================
    # KG + OWL 동기화 (Phase 4: v3에서 포팅)
    # =========================================================================

    def _sync_knowledge_graph(self, data: dict[str, Any]) -> None:
        """
        크롤링 데이터 → KG + OWL Ontology 동기화

        대시보드 데이터의 브랜드 메트릭을 KnowledgeGraph 엔티티 메타데이터와
        OWL Ontology의 Brand 인스턴스로 동기화합니다.

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

            # OWL Ontology에도 동기화
            if hasattr(self, "_owl_reasoner") and self._owl_reasoner:
                for brand_info in brand_metrics[:20]:  # 상위 20개 브랜드
                    brand_name = brand_info.get("brand")
                    if brand_name:
                        self._owl_reasoner.add_brand(
                            name=brand_name,
                            sos=brand_info.get("sos", 0) / 100,
                            avg_rank=brand_info.get("avg_rank"),
                            product_count=brand_info.get("products", 0),
                        )

                # 시장 포지션 추론
                self._owl_reasoner.infer_market_positions()

            logger.info(f"KG & OWL Ontology synced: {len(brand_metrics)} brands")
        except Exception as e:
            logger.warning(f"KG/Ontology sync failed: {e}")

    # =========================================================================
    # v3 대시보드 도구 등록 (Phase 3)
    # =========================================================================

    def _register_dashboard_tools(self) -> None:
        """v3 대시보드 조회 도구를 ToolCoordinator에 등록"""

        data_path = os.environ.get("DASHBOARD_DATA_PATH", f"{_DATA_DIR}/dashboard_data.json")

        def _load_dashboard_data() -> dict:
            """대시보드 데이터 로드"""
            try:
                with open(data_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load dashboard data: {e}")
                return {}

        async def exec_brand_status(**kwargs: Any) -> dict:
            data = _load_dashboard_data()
            brand = data.get("brand", {})
            kpis = brand.get("kpis", {})
            return {
                "brand": "LANEIGE",
                "sos": kpis.get("sos", 0),
                "sos_change": kpis.get("sos_delta", "N/A"),
                "top10_products": kpis.get("top10_count", 0),
                "avg_rank": kpis.get("avg_rank", 0),
                "hhi": kpis.get("hhi", 0),
                "total_products": data.get("metadata", {}).get("laneige_products", 0),
            }

        async def exec_product_info(**kwargs: Any) -> dict:
            data = _load_dashboard_data()
            product_name = kwargs.get("product_name", "")
            products = data.get("products", {})

            # ASIN으로 검색
            if product_name.upper().startswith("B0"):
                product = products.get(product_name.upper())
                if product:
                    return product

            # 제품명으로 검색
            for _asin, prod in products.items():
                name = prod.get("name", "").lower()
                if product_name.lower() in name:
                    return prod

            # 모든 LANEIGE 제품 목록
            laneige_products = [
                {"asin": k, "name": v.get("name", "")[:50], "rank": v.get("rank")}
                for k, v in products.items()
            ]
            return {
                "message": f"'{product_name}' 제품을 찾을 수 없습니다.",
                "available_products": laneige_products,
            }

        async def exec_competitor_analysis(**kwargs: Any) -> dict:
            data = _load_dashboard_data()
            brand_name = kwargs.get("brand_name")
            competitors = data.get("brand", {}).get("competitors", [])

            if brand_name:
                for comp in competitors:
                    if brand_name.lower() in comp.get("brand", "").lower():
                        return comp
                return {"message": f"'{brand_name}' 브랜드를 찾을 수 없습니다."}

            return {
                "competitors": competitors[:10],
                "laneige_rank": next(
                    (i + 1 for i, c in enumerate(competitors) if is_target_brand(c.get("brand"))),
                    "N/A",
                ),
            }

        async def exec_category_info(**kwargs: Any) -> dict:
            data = _load_dashboard_data()
            category = kwargs.get("category")
            categories = data.get("categories", {})

            if category:
                cat_data = categories.get(category) or categories.get(category.lower())
                if cat_data:
                    return cat_data
                return {"message": f"'{category}' 카테고리를 찾을 수 없습니다."}

            return categories

        async def exec_action_items(**kwargs: Any) -> dict:
            data = _load_dashboard_data()
            home = data.get("home", {})
            return {
                "status": home.get("status", {}),
                "action_items": home.get("action_items", []),
            }

        # ToolCoordinator의 ToolExecutor에 등록
        executor = self.tool_coordinator.tool_executor
        executor.register_executor("get_brand_status", exec_brand_status)
        executor.register_executor("get_product_info", exec_product_info)
        executor.register_executor("get_competitor_analysis", exec_competitor_analysis)
        executor.register_executor("get_category_info", exec_category_info)
        executor.register_executor("get_action_items", exec_action_items)

        logger.info("Registered 5 v3 dashboard tools in ToolCoordinator")

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
                    from src.infrastructure.container import Container

                    # 단일 배치 파이프라인 (F1) + 단일 시스템 상태 (F7)
                    self._workflow_agent = Container.get_batch_workflow(state_manager=self.state)

                result = await self._workflow_agent.run_daily_workflow()

                # 크롤링 완료 이벤트
                await self.emit_event("crawl_complete", {"result": result})

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

            else:
                return {
                    "task": task_name,
                    "status": "skipped",
                    "reason": f"Unknown action: {action}",
                }

        except Exception as e:
            logger.error(f"Scheduled task failed: {e}")
            return {"task": task_name, "status": "failed", "error": str(e)}

    @property
    def _task_queue(self) -> list[BrainTask]:
        """대기 중인 작업 (heapq 리스트) — BrainTaskQueue 위임"""
        return self._tasks.pending

    @property
    def _task_history(self) -> list[BrainTask]:
        """완료된 작업 이력 — BrainTaskQueue 위임"""
        return self._tasks.history

    @property
    def _current_task(self) -> BrainTask | None:
        """실행 중인 작업 — BrainTaskQueue 위임"""
        return self._tasks.current

    @property
    def _task_handlers(self) -> dict[str, TaskHandler]:
        """작업 타입 → 처리기. 없는 타입은 결과 없이 완료 처리된다."""
        return {"alert": self._process_alert}

    async def _process_task_queue(self) -> None:
        """작업 큐 처리 (사용자 응답 중이면 중단)"""
        while self._tasks.pending:
            if self.mode == BrainMode.RESPONDING:
                break
            await self._execute_queued_task(self._tasks.pop())

    async def _execute_queued_task(self, task: BrainTask) -> None:
        """큐 작업 실행"""
        await self._tasks.execute(task, self._task_handlers)

    def add_task(self, task: BrainTask) -> None:
        """작업 큐에 추가"""
        self._tasks.add(task)

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
        """Morning Brief + 인사이트 리포트 이메일 발송 (매일 08:00 KST 스케줄).

        생성·발송 자체는 ``src.tools.intelligence.newsletter`` 가 담당한다 (Phase 4).
        brain 은 실패를 자신의 통계에 기록하는 일만 한다.
        """
        from src.tools.intelligence.newsletter import send_morning_brief

        if not await send_morning_brief():
            self._stats["errors"] += 1

    # =========================================================================
    # 스케줄러 관리
    # =========================================================================

    # 스케줄된 크롤링 완료 대기 상한 (CrawlManager stale-lock 기준 2시간과 동일)
    CRAWL_WAIT_TIMEOUT_SEC: float = 2 * 60 * 60

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

                        # D13: start_crawl()은 백그라운드 태스크만 생성한다.
                        # 스케줄러가 완료 마킹을 크롤링 종료 이후에 하도록 실제 완료를 대기.
                        finished = await crawl_manager.wait_for_completion(
                            timeout=self.CRAWL_WAIT_TIMEOUT_SEC
                        )
                        if finished:
                            logger.info("Scheduled crawl finished")
                        else:
                            logger.warning(
                                "Scheduled crawl did not complete cleanly: "
                                f"{crawl_manager.get_status_message()}"
                            )
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


__all__ = [
    "BrainMode",
    # brain_scheduler 재수출 (기존 ``from src.core.brain import BrainTask`` 호환)
    "BrainTask",
    "TaskPriority",
    "UnifiedBrain",
    "get_brain",
    "get_initialized_brain",
    "reset_brain",
]
