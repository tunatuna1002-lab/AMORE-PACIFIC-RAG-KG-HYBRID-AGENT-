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
- QueryProcessor: 사용자 질문 처리
- DecisionMaker: LLM 의사결정
- ToolCoordinator: 도구 실행 조율
- AlertManager: 알림 처리
- UnifiedBrain: Facade (위 컴포넌트 조율)

동작 모드:
- 자율 모드: 스케줄 기반 자동 작업 (크롤링, 분석, 알림)
- 대화 모드: 사용자 질문 처리 (최우선)
- 알림 모드: 이벤트 기반 알림 생성

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
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


# SRP 분해된 컴포넌트
from .alert_manager import AlertManager
from .cache import ResponseCache
from .confidence import ConfidenceAssessor
from .context_gatherer import ContextGatherer
from .decision_maker import DecisionMaker
from .models import Context, Response, ToolResult
from .query_processor import QueryProcessor
from .response_pipeline import ResponsePipeline
from .scheduler import AutonomousScheduler
from .state import OrchestratorState
from .tool_coordinator import ErrorStrategy, ToolCoordinator
from .tools import AGENT_TOOLS, ToolExecutor

# Type checking imports (순환 참조 방지)
if TYPE_CHECKING:
    from ..tools.market_intelligence import MarketIntelligenceEngine

# AlertAgent는 TYPE_CHECKING에서만 임포트 (순환 import 방지)
from ..core.state_manager import StateManager

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
    - QueryProcessor: 사용자 질문 처리 전담
    """

    # LLM 판단 프롬프트 (DecisionMaker와 공유, 하위호환성 유지)
    DECISION_PROMPT = DecisionMaker.DECISION_PROMPT

    def __init__(
        self,
        context_gatherer: ContextGatherer | None = None,
        tool_executor: ToolExecutor | None = None,
        response_pipeline: ResponsePipeline | None = None,
        cache: ResponseCache | None = None,
        model: str = "gpt-4o-mini",
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
        self.state = OrchestratorState()

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
        self._query_processor: QueryProcessor | None = None

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

        # Market Intelligence Engine (lazy init)
        self._market_intelligence: MarketIntelligenceEngine | None = None

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

    @property
    def query_processor(self) -> QueryProcessor:
        """질문 처리 컴포넌트 (lazy init)"""
        if self._query_processor is None:
            self._query_processor = QueryProcessor(
                decision_maker=self.decision_maker,
                tool_coordinator=self.tool_coordinator,
                response_pipeline=self._response_pipeline or ResponsePipeline(),
                context_gatherer=self._context_gatherer,
                cache=self.cache,
                state=self.state,
            )
        return self._query_processor

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
            from ..ontology.reasoner import OntologyReasoner
            from ..rag.hybrid_retriever import HybridRetriever

            kg = KnowledgeGraph(persist_path="./data/knowledge_graph.json")
            reasoner = OntologyReasoner(kg)
            hybrid_retriever = HybridRetriever(kg, reasoner)

            self._context_gatherer = ContextGatherer(
                hybrid_retriever=hybrid_retriever, orchestrator_state=self.state
            )
            await self._context_gatherer.initialize()

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

        # ReActAgent 초기화 (있으면)
        try:
            from ..agents.react_agent import get_react_agent

            self._react_agent = get_react_agent()
            self._react_agent.set_tool_executor(self._tool_executor)
            logger.info("ReActAgent initialized")
        except Exception as e:
            logger.debug(f"ReActAgent not available: {e}")
            self._react_agent = None

        self._initialized = True
        logger.info("UnifiedBrain initialized (LLM-First mode, SRP components)")

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

    async def _check_alert_conditions(self, event_name: str, data: dict[str, Any]) -> None:
        """알림 조건 체크 (하위호환성 유지, AlertManager에 위임)"""
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
            # 세션 설정
            if session_id:
                self.state.set_session(session_id)

            # 1. 캐시 확인
            if not skip_cache:
                cached = self.cache.get(query, "query")
                if cached:
                    self._stats["cache_hits"] += 1
                    logger.info(f"Cache hit: {query[:30]}...")
                    return cached

            # 2. 시스템 상태 수집
            system_state = self._get_system_state(current_metrics)

            # 3. 컨텍스트 수집
            context = await self._context_gatherer.gather(
                query=query, current_metrics=current_metrics
            )

            # 4. 복잡도 판단 및 ReAct 모드 활성화
            if self._react_agent and self._is_complex_query(query, context):
                logger.info(f"Complex query detected, using ReAct mode: {query[:50]}...")
                response = await self._process_with_react(query, context)
            else:
                # 4. LLM 의사결정 (DecisionMaker에 위임)
                decision = await self.decision_maker.decide(query, context, system_state)
                self._stats["llm_decisions"] += 1

                # 5. 도구 실행 (필요시, ToolCoordinator에 위임)
                tool_result = None
                if decision.get("tool") and decision["tool"] != "direct_answer":
                    self.mode = BrainMode.EXECUTING
                    tool_result = await self.tool_coordinator.execute(
                        tool_name=decision["tool"], params=decision.get("tool_params", {})
                    )

                # 6. 응답 생성
                response = await self._generate_response(
                    query=query, context=context, decision=decision, tool_result=tool_result
                )

            # 처리 시간
            response.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            # 캐시 저장
            if not skip_cache and not response.is_fallback:
                self.cache.set(query, response, "query")

            return response

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Query processing failed: {e}")
            await self.emit_event("error_occurred", {"error": str(e), "query": query})
            return Response.fallback(f"처리 중 오류가 발생했습니다: {str(e)}")

        finally:
            self.mode = previous_mode

    # =========================================================================
    # LLM 의사결정 (하위호환성 유지, DecisionMaker에 위임)
    # =========================================================================

    async def _make_llm_decision(
        self, query: str, context: Context, system_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        LLM 기반 의사결정 (DecisionMaker에 위임)

        하위호환성을 위해 유지합니다.
        """
        self._stats["llm_decisions"] += 1
        return await self.decision_maker.decide(query, context, system_state)

    def _parse_decision(self, response_text: str) -> dict[str, Any]:
        """LLM 응답 파싱 (하위호환성)"""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM decision")

        return {"tool": "direct_answer", "reason": "파싱 실패", "confidence": 0.5, "key_points": []}

    # =========================================================================
    # 도구 실행 (하위호환성 유지, ToolCoordinator에 위임)
    # =========================================================================

    async def _execute_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """
        도구 실행 (ToolCoordinator에 위임)

        하위호환성을 위해 유지합니다.
        """
        return await self.tool_coordinator.execute(tool_name, params)

    async def _handle_error(
        self, error: Any, strategy: ErrorStrategy, params: dict[str, Any]
    ) -> ToolResult:
        """
        에러 처리 (하위호환성)

        ToolCoordinator가 내부적으로 처리합니다.
        """
        self._stats["errors"] += 1
        return ToolResult(
            tool_name=error.agent_name if hasattr(error, "agent_name") else "unknown",
            success=False,
            error=str(error),
        )

    # =========================================================================
    # 응답 생성
    # =========================================================================

    async def _generate_response(
        self,
        query: str,
        context: Context,
        decision: dict[str, Any],
        tool_result: ToolResult | None = None,
    ) -> Response:
        """응답 생성"""
        if self._response_pipeline:
            return await self._response_pipeline.generate(
                query=query, context=context, decision=decision, tool_result=tool_result
            )

        # 폴백 응답 생성
        content = ""
        if tool_result and tool_result.success:
            content = (
                f"도구 실행 결과:\n{json.dumps(tool_result.data, ensure_ascii=False, indent=2)}"
            )
        elif context.summary:
            content = context.summary
        else:
            content = "관련 정보를 찾을 수 없습니다."

        return Response(
            text=content,
            confidence_score=decision.get("confidence", 0.5),
            sources=context.rag_docs[:3] if context.rag_docs else [],
            tools_called=[decision.get("tool")] if decision.get("tool") != "direct_answer" else [],
        )

    # =========================================================================
    # ReAct 처리
    # =========================================================================

    def _is_complex_query(self, query: str, context: Context) -> bool:
        """
        복잡한 질문인지 판단

        복잡한 질문의 특징:
        - 여러 단계 추론 필요
        - 다중 데이터 소스 필요
        - "왜", "어떻게", "비교" 등 분석적 질문
        - 컨텍스트가 불충분
        """
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

        return has_complex_keyword or (low_context and multi_step)

    async def _process_with_react(self, query: str, context: Context) -> Response:
        """
        ReAct 모드로 질문 처리

        Args:
            query: 사용자 질문
            context: 수집된 컨텍스트

        Returns:
            Response 객체
        """
        if not self._react_agent:
            return Response.fallback("ReAct 에이전트를 사용할 수 없습니다.")

        try:
            # ReAct 실행
            react_result = await self._react_agent.run(
                query=query, context=context.summary or "컨텍스트 없음"
            )

            # 응답 생성
            response = Response(
                text=react_result.final_answer,
                confidence_score=react_result.confidence,
                sources=context.rag_docs[:3] if context.rag_docs else [],
                tools_called=[step.action for step in react_result.steps if step.action],
            )

            # 개선 필요 시 로깅
            if react_result.needs_improvement:
                logger.warning(
                    f"ReAct result needs improvement (confidence: {react_result.confidence:.2f})"
                )

            return response

        except Exception as e:
            logger.error(f"ReAct processing failed: {e}")
            return Response.fallback(f"ReAct 처리 실패: {str(e)}")

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
                    from .batch_workflow import BatchWorkflow

                    self._workflow_agent = BatchWorkflow()

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
                from ..tools.market_intelligence import MarketIntelligenceEngine

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

    def _format_tools_description(self, state: dict[str, Any]) -> str:
        """도구 설명 포맷"""
        available = state.get("available_tools", [])
        lines = []
        for name, tool in AGENT_TOOLS.items():
            if name in available:
                lines.append(f"- {name}: {tool.description}")
        lines.append("- direct_answer: 컨텍스트만으로 직접 답변")
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
        import os

        logger.info("Generating Morning Brief...")

        try:
            # 1. 최신 크롤링 데이터 가져오기
            from src.tools.market_intelligence import MarketIntelligenceEngine

            mi = MarketIntelligenceEngine()
            latest_data = await mi.get_latest_data()

            products = latest_data.get("products", [])
            crawl_data = {
                "products": products,
                "category": "All Categories",
            }

            # 2. Morning Brief 생성
            from src.tools.morning_brief import MorningBriefGenerator, render_morning_brief_html

            generator = MorningBriefGenerator()
            brief_data = await generator.generate(
                crawl_data=crawl_data,
                metrics_data=latest_data.get("metrics"),
            )

            # 3. HTML 렌더링
            html_content = render_morning_brief_html(brief_data)

            # 4. 이메일 발송
            from src.tools.email_sender import EmailSender

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
        import os
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

            # HHI 계산
            brand_counts = {}
            for p in top100:
                brand = p.get("brand", "Unknown")
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
            hhi = (
                sum((count / len(top100) * 100) ** 2 for count in brand_counts.values())
                if top100
                else 0
            )

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

            # 대시보드 URL
            dashboard_url = os.getenv("DASHBOARD_URL", "http://localhost:8001") + "/dashboard"

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
