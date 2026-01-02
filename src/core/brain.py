"""
통합 두뇌 (Unified Brain)
=========================
Level 4 Autonomous Agent의 핵심 두뇌

역할:
1. 모든 에이전트를 통제하는 단일 중앙 제어
2. LLM 기반 의사결정 (LLM-First 방식)
3. 자율 작업과 사용자 요청 우선순위 관리
4. 상태 관리 및 이벤트 처리

동작 모드:
- 자율 모드: 스케줄 기반 자동 작업 (크롤링, 분석, 알림)
- 대화 모드: 사용자 질문 처리 (최우선)
- 알림 모드: 이벤트 기반 알림 생성

의사결정 흐름 (LLM-First):
1. 컨텍스트 수집 (RAG + KG)
2. LLM이 상황 분석 및 에이전트 선택
3. 에이전트 실행
4. 응답 생성

Usage:
    brain = UnifiedBrain()
    await brain.initialize()

    # 사용자 질문 처리
    response = await brain.process_query("LANEIGE 순위 알려줘")

    # 자율 작업 실행
    await brain.run_autonomous_cycle()
"""

import logging
import json
import asyncio
from datetime import datetime, time, timedelta
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass, field
import heapq

from litellm import acompletion

from .models import Context, Response, ToolResult, ConfidenceLevel
from .confidence import ConfidenceAssessor
from .cache import ResponseCache
from .state import OrchestratorState
from .context_gatherer import ContextGatherer
from .tools import ToolExecutor, AGENT_TOOLS
from .response_pipeline import ResponsePipeline

# Type checking imports (순환 참조 방지)
if TYPE_CHECKING:
    from ..agents.query_agent import QueryAgent
    from ..agents.workflow_agent import WorkflowAgent
    from ..agents.alert_agent import AlertAgent

logger = logging.getLogger(__name__)


# =============================================================================
# 타입 정의
# =============================================================================

class BrainMode(Enum):
    """두뇌 동작 모드"""
    IDLE = "idle"                    # 대기
    AUTONOMOUS = "autonomous"        # 자율 작업 중
    RESPONDING = "responding"        # 사용자 응답 중
    EXECUTING = "executing"          # 에이전트 실행 중
    ALERTING = "alerting"            # 알림 처리 중


class TaskPriority(Enum):
    """작업 우선순위"""
    USER_REQUEST = 0      # 사용자 요청 (최우선)
    CRITICAL_ALERT = 1    # 중요 알림
    SCHEDULED = 2         # 예약 작업
    BACKGROUND = 3        # 백그라운드 작업


class ErrorStrategy(Enum):
    """에러 처리 전략"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    NOTIFY_USER = "notify_user"


@dataclass
class BrainTask:
    """두뇌가 처리할 작업"""
    id: str
    type: str                       # query, scheduled, alert, autonomous
    priority: TaskPriority
    payload: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def __lt__(self, other):
        """우선순위 기반 정렬"""
        return self.priority.value < other.priority.value


@dataclass
class AgentError:
    """에이전트 에러 정보"""
    agent_name: str
    error_message: str
    error_type: str
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


# 에이전트별 에러 전략
AGENT_ERROR_STRATEGIES: Dict[str, ErrorStrategy] = {
    "crawl_amazon": ErrorStrategy.FALLBACK,
    "calculate_metrics": ErrorStrategy.RETRY,
    "query_data": ErrorStrategy.FALLBACK,
    "query_knowledge_graph": ErrorStrategy.SKIP,
    "generate_insight": ErrorStrategy.RETRY,
    "send_alert": ErrorStrategy.RETRY,
    "workflow": ErrorStrategy.RETRY,
}


# =============================================================================
# 자율 스케줄러
# =============================================================================

class AutonomousScheduler:
    """
    자율 작업 스케줄러

    설정된 스케줄에 따라 자동 작업을 트리거합니다.
    """

    def __init__(self):
        self.schedules: List[Dict[str, Any]] = []
        self._last_run: Dict[str, datetime] = {}
        self._load_default_schedules()

    def _load_default_schedules(self):
        """기본 스케줄 로드"""
        self.schedules = [
            {
                "id": "daily_crawl",
                "name": "일일 크롤링",
                "action": "crawl_workflow",
                "schedule_type": "daily",
                "hour": 9,
                "minute": 0,
                "enabled": True
            },
            {
                "id": "check_data_freshness",
                "name": "데이터 신선도 체크",
                "action": "check_data",
                "schedule_type": "interval",
                "interval_hours": 1,
                "enabled": True
            }
        ]

    def get_due_tasks(self, current_time: datetime) -> List[Dict[str, Any]]:
        """실행해야 할 작업 목록 반환"""
        due_tasks = []

        for schedule in self.schedules:
            if not schedule.get("enabled", True):
                continue

            schedule_id = schedule["id"]
            last_run = self._last_run.get(schedule_id)

            if schedule["schedule_type"] == "daily":
                # 매일 특정 시간에 실행
                scheduled_time = current_time.replace(
                    hour=schedule["hour"],
                    minute=schedule["minute"],
                    second=0,
                    microsecond=0
                )

                # 오늘 아직 실행 안 했고, 시간이 됐으면
                if (last_run is None or last_run.date() < current_time.date()) and \
                   current_time >= scheduled_time:
                    due_tasks.append(schedule)

            elif schedule["schedule_type"] == "interval":
                # 일정 간격으로 실행
                interval = timedelta(hours=schedule.get("interval_hours", 1))
                if last_run is None or (current_time - last_run) >= interval:
                    due_tasks.append(schedule)

        return due_tasks

    def mark_completed(self, schedule_id: str):
        """작업 완료 마킹"""
        self._last_run[schedule_id] = datetime.now()

    def add_schedule(self, schedule: Dict[str, Any]):
        """스케줄 추가"""
        self.schedules.append(schedule)

    def remove_schedule(self, schedule_id: str):
        """스케줄 제거"""
        self.schedules = [s for s in self.schedules if s["id"] != schedule_id]


# =============================================================================
# 통합 두뇌
# =============================================================================

class UnifiedBrain:
    """
    Level 4 Autonomous Agent의 통합 두뇌 (LLM-First)

    모든 에이전트를 통제하고, 자율 작업과 사용자 요청을 관리합니다.

    핵심 원칙:
    1. 사용자 요청은 항상 최우선
    2. 모든 판단은 LLM이 담당 (LLM-First)
    3. 자율 스케줄링으로 데이터 자동 수집
    4. 이벤트 기반 알림 시스템
    """

    # LLM 판단 프롬프트
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

    def __init__(
        self,
        context_gatherer: Optional[ContextGatherer] = None,
        tool_executor: Optional[ToolExecutor] = None,
        response_pipeline: Optional[ResponsePipeline] = None,
        cache: Optional[ResponseCache] = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 2
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
        # 핵심 컴포넌트
        self.context_gatherer = context_gatherer
        self.tool_executor = tool_executor or ToolExecutor()
        self.response_pipeline = response_pipeline
        self.cache = cache or ResponseCache()
        self.confidence_assessor = ConfidenceAssessor()

        # LLM 설정
        self.model = model
        self.max_retries = max_retries

        # 상태 관리
        self.state = OrchestratorState()
        self.mode = BrainMode.IDLE
        self._current_task: Optional[BrainTask] = None

        # 자율 스케줄러
        self.scheduler = AutonomousScheduler()

        # 작업 큐 (우선순위 힙)
        self._task_queue: List[BrainTask] = []
        self._task_history: List[BrainTask] = []

        # 에러 추적
        self._error_history: List[AgentError] = []
        self._failed_agents: Dict[str, datetime] = {}

        # 이벤트 콜백
        self._event_handlers: Dict[str, List[Callable]] = {}

        # 에이전트 (lazy init)
        self._query_agent = None
        self._workflow_agent = None
        self._alert_agent = None

        # 통계
        self._stats = {
            "total_queries": 0,
            "llm_decisions": 0,
            "cache_hits": 0,
            "autonomous_tasks": 0,
            "alerts_generated": 0,
            "errors": 0
        }

        # 초기화 플래그
        self._initialized = False

    # =========================================================================
    # 초기화
    # =========================================================================

    async def initialize(self) -> None:
        """비동기 초기화"""
        if self._initialized:
            return

        # Context Gatherer 초기화
        if self.context_gatherer:
            await self.context_gatherer.initialize()
        else:
            # 기본 Context Gatherer 생성
            from ..rag.hybrid_retriever import HybridRetriever
            from ..ontology.knowledge_graph import KnowledgeGraph
            from ..ontology.reasoner import OntologyReasoner

            kg = KnowledgeGraph(persist_path="./data/knowledge_graph.json")
            reasoner = OntologyReasoner(kg)
            hybrid_retriever = HybridRetriever(kg, reasoner)

            self.context_gatherer = ContextGatherer(
                hybrid_retriever=hybrid_retriever,
                orchestrator_state=self.state
            )
            await self.context_gatherer.initialize()

        # Response Pipeline 초기화
        if not self.response_pipeline:
            self.response_pipeline = ResponsePipeline()

        self._initialized = True
        logger.info("UnifiedBrain initialized (LLM-First mode)")

    # =========================================================================
    # 이벤트 시스템
    # =========================================================================

    def on_event(self, event_name: str, handler: Callable) -> None:
        """이벤트 핸들러 등록"""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)

    async def emit_event(self, event_name: str, data: Dict[str, Any] = None) -> None:
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

        # 알림 조건 체크
        if event_name in ["crawl_complete", "metrics_calculated", "rank_changed"]:
            await self._check_alert_conditions(event_name, data)

    async def _check_alert_conditions(self, event_name: str, data: Dict[str, Any]) -> None:
        """알림 조건 체크 및 알림 생성"""
        alerts = []

        if event_name == "rank_changed":
            product = data.get("product", {})
            change = data.get("change", 0)

            # 급락/급등 체크
            if abs(change) >= 10:
                alert_type = "rank_drop" if change > 0 else "rank_surge"
                alerts.append({
                    "type": alert_type,
                    "product": product.get("name"),
                    "change": change,
                    "message": f"{product.get('name')} 순위 {'급락' if change > 0 else '급등'}: {abs(change)}단계"
                })

        for alert in alerts:
            await self._process_alert(alert)

    async def _process_alert(self, alert: Dict[str, Any]) -> None:
        """알림 처리"""
        self._stats["alerts_generated"] += 1

        # 알림 이벤트 발생
        await self.emit_event("alert_generated", alert)

        # AlertAgent 호출 (구현되어 있으면)
        if self._alert_agent:
            await self._alert_agent.process_alert(alert)

    # =========================================================================
    # 사용자 질문 처리 (최우선)
    # =========================================================================

    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        current_metrics: Optional[Dict[str, Any]] = None,
        skip_cache: bool = False
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
            context = await self.context_gatherer.gather(
                query=query,
                current_metrics=current_metrics
            )

            # 4. LLM 의사결정 (LLM-First)
            decision = await self._make_llm_decision(query, context, system_state)

            # 5. 도구 실행 (필요시)
            tool_result = None
            if decision.get("tool") and decision["tool"] != "direct_answer":
                self.mode = BrainMode.EXECUTING
                tool_result = await self._execute_tool(
                    tool_name=decision["tool"],
                    params=decision.get("tool_params", {})
                )

            # 6. 응답 생성
            response = await self._generate_response(
                query=query,
                context=context,
                decision=decision,
                tool_result=tool_result
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
    # LLM 의사결정 (LLM-First)
    # =========================================================================

    async def _make_llm_decision(
        self,
        query: str,
        context: Context,
        system_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM 기반 의사결정 (LLM-First)

        모든 판단을 LLM이 담당합니다.
        """
        self._stats["llm_decisions"] += 1

        try:
            prompt = self.DECISION_PROMPT.format(
                system_state=self._format_system_state(system_state),
                tools_description=self._format_tools_description(system_state),
                context_summary=context.summary or "컨텍스트 없음",
                query=query
            )

            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )

            return self._parse_decision(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return {
                "tool": "direct_answer",
                "reason": f"LLM 오류: {e}",
                "confidence": 0.3,
                "key_points": []
            }

    def _parse_decision(self, response_text: str) -> Dict[str, Any]:
        """LLM 응답 파싱"""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM decision")

        return {
            "tool": "direct_answer",
            "reason": "파싱 실패",
            "confidence": 0.5,
            "key_points": []
        }

    # =========================================================================
    # 도구 실행
    # =========================================================================

    async def _execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> ToolResult:
        """에러 처리가 포함된 도구 실행"""
        strategy = AGENT_ERROR_STRATEGIES.get(tool_name, ErrorStrategy.NOTIFY_USER)
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                self.state.start_tool(tool_name)
                result = await self.tool_executor.execute(tool_name, params)
                self.state.end_tool(tool_name)

                if result.success:
                    self._failed_agents.pop(tool_name, None)
                    return result

                # 실패 처리
                error = AgentError(
                    agent_name=tool_name,
                    error_message=result.error or "Unknown error",
                    error_type="execution",
                    retry_count=retry_count
                )
                return await self._handle_error(error, strategy, params)

            except asyncio.TimeoutError:
                self.state.end_tool(tool_name)
                error = AgentError(
                    agent_name=tool_name,
                    error_message="Timeout",
                    error_type="timeout",
                    retry_count=retry_count
                )

                if strategy == ErrorStrategy.RETRY and retry_count < self.max_retries:
                    retry_count += 1
                    await asyncio.sleep(1)
                    continue

                return await self._handle_error(error, strategy, params)

            except Exception as e:
                self.state.end_tool(tool_name)
                error = AgentError(
                    agent_name=tool_name,
                    error_message=str(e),
                    error_type="exception",
                    retry_count=retry_count
                )

                if strategy == ErrorStrategy.RETRY and retry_count < self.max_retries:
                    retry_count += 1
                    await asyncio.sleep(1)
                    continue

                return await self._handle_error(error, strategy, params)

        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"최대 재시도 초과 ({self.max_retries}회)"
        )

    async def _handle_error(
        self,
        error: AgentError,
        strategy: ErrorStrategy,
        params: Dict[str, Any]
    ) -> ToolResult:
        """에러 전략에 따른 처리"""
        self._stats["errors"] += 1
        self._error_history.append(error)
        self._failed_agents[error.agent_name] = error.timestamp

        # 히스토리 크기 제한
        if len(self._error_history) > 100:
            self._error_history = self._error_history[-50:]

        logger.warning(f"Error in {error.agent_name}: {error.error_message}")

        # 에러 이벤트 발생
        await self.emit_event("error_occurred", {
            "agent": error.agent_name,
            "error": error.error_message,
            "strategy": strategy.value
        })

        if strategy == ErrorStrategy.FALLBACK:
            cached = self.cache.get_tool_result(error.agent_name)
            if cached:
                return ToolResult(
                    tool_name=error.agent_name,
                    success=True,
                    data={**cached, "_from_cache": True}
                )

        elif strategy == ErrorStrategy.SKIP:
            return ToolResult(
                tool_name=error.agent_name,
                success=True,
                data={"_skipped": True}
            )

        # NOTIFY_USER 또는 기타
        return ToolResult(
            tool_name=error.agent_name,
            success=False,
            error=error.error_message
        )

    # =========================================================================
    # 응답 생성
    # =========================================================================

    async def _generate_response(
        self,
        query: str,
        context: Context,
        decision: Dict[str, Any],
        tool_result: Optional[ToolResult] = None
    ) -> Response:
        """응답 생성"""
        if self.response_pipeline:
            return await self.response_pipeline.generate(
                query=query,
                context=context,
                decision=decision,
                tool_result=tool_result
            )

        # 폴백 응답 생성
        content = ""
        if tool_result and tool_result.success:
            content = f"도구 실행 결과:\n{json.dumps(tool_result.data, ensure_ascii=False, indent=2)}"
        elif context.summary:
            content = context.summary
        else:
            content = "관련 정보를 찾을 수 없습니다."

        return Response(
            content=content,
            confidence=decision.get("confidence", 0.5),
            sources=context.rag_docs[:3] if context.rag_docs else [],
            tools_called=[decision.get("tool")] if decision.get("tool") != "direct_answer" else []
        )

    # =========================================================================
    # 자율 작업 (Autonomous)
    # =========================================================================

    async def run_autonomous_cycle(self) -> Dict[str, Any]:
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
            "status": "completed"
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

    async def _execute_scheduled_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """스케줄된 작업 실행"""
        action = task.get("action")
        task_name = task.get("name", action)

        logger.info(f"Executing scheduled task: {task_name}")

        try:
            if action == "crawl_workflow":
                # WorkflowAgent 호출
                if not self._workflow_agent:
                    from ..agents.workflow_agent import WorkflowAgent
                    self._workflow_agent = WorkflowAgent()

                result = await self._workflow_agent.run_workflow()

                # 크롤링 완료 이벤트
                await self.emit_event("crawl_complete", {"result": result})

                return {
                    "task": task_name,
                    "status": "completed",
                    "result": result.get("summary")
                }

            elif action == "check_data":
                # 데이터 신선도 체크
                needs_crawl = self.state.is_crawl_needed()
                return {
                    "task": task_name,
                    "status": "completed",
                    "needs_crawl": needs_crawl
                }

            else:
                return {
                    "task": task_name,
                    "status": "skipped",
                    "reason": f"Unknown action: {action}"
                }

        except Exception as e:
            logger.error(f"Scheduled task failed: {e}")
            return {
                "task": task_name,
                "status": "failed",
                "error": str(e)
            }

    async def _process_task_queue(self) -> None:
        """작업 큐 처리"""
        while self._task_queue:
            # 사용자 요청 모드면 중단
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
    # 알림 처리
    # =========================================================================

    async def check_alerts(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        지표 데이터에서 알림 조건 확인

        Args:
            metrics_data: 지표 데이터

        Returns:
            발생한 알림 목록
        """
        alerts = []

        # 제품별 순위 변동 확인
        products = metrics_data.get("products", {})
        for asin, product in products.items():
            rank_delta = product.get("rank_delta", "")

            # 급락 체크 (순위가 올라가면 음수)
            if rank_delta:
                try:
                    change = int(rank_delta.replace("+", "").replace("-", ""))
                    if "-" in rank_delta:  # 순위 하락
                        change = -change

                    if abs(change) >= 10:
                        alerts.append({
                            "type": "rank_change",
                            "severity": "critical" if abs(change) >= 20 else "warning",
                            "product": product.get("name"),
                            "asin": asin,
                            "change": change,
                            "current_rank": product.get("rank"),
                            "message": f"{product.get('name')} 순위 {'급락' if change > 0 else '급등'}: {abs(change)}단계"
                        })
                except ValueError:
                    pass

        # SoS 변동 확인
        brand_kpis = metrics_data.get("brand", {}).get("kpis", {})
        sos_delta = brand_kpis.get("sos_delta", "")
        if sos_delta:
            try:
                sos_change = float(sos_delta.replace("+", "").replace("%", "").replace("p", ""))
                if sos_change <= -2:  # 2%p 이상 하락
                    alerts.append({
                        "type": "sos_drop",
                        "severity": "warning",
                        "change": sos_change,
                        "current_sos": brand_kpis.get("sos"),
                        "message": f"LANEIGE SoS {sos_change}%p 하락"
                    })
            except ValueError:
                pass

        return alerts

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def _get_system_state(self, current_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """시스템 상태 수집"""
        data_status = "없음"
        data_date = None

        if current_metrics:
            metadata = current_metrics.get("metadata", {})
            data_date = metadata.get("data_date")
            if data_date:
                today = datetime.now().strftime("%Y-%m-%d")
                data_status = "최신" if data_date == today else f"오래됨 ({data_date})"

        available_tools = self.tool_executor.get_available_tools()
        failed_recently = [
            name for name, time in self._failed_agents.items()
            if (datetime.now() - time).seconds < 300
        ]

        return {
            "data_status": data_status,
            "data_date": data_date,
            "available_tools": [t for t in available_tools if t not in failed_recently],
            "failed_tools": failed_recently,
            "mode": self.mode.value,
            "cache_stats": self.cache.get_stats()
        }

    def _format_system_state(self, state: Dict[str, Any]) -> str:
        """시스템 상태 포맷"""
        lines = [
            f"- 데이터 상태: {state['data_status']}",
            f"- 동작 모드: {state['mode']}",
            f"- 사용 가능 도구: {', '.join(state['available_tools'])}",
        ]
        if state['failed_tools']:
            lines.append(f"- 실패 도구: {', '.join(state['failed_tools'])}")
        return "\n".join(lines)

    def _format_tools_description(self, state: Dict[str, Any]) -> str:
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

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            **self._stats,
            "mode": self.mode.value,
            "queue_size": len(self._task_queue),
            "error_count": len(self._error_history),
            "failed_agents": list(self._failed_agents.keys()),
            "scheduled_tasks": len(self.scheduler.schedules)
        }

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 에러 목록"""
        return [e.to_dict() for e in self._error_history[-limit:]]

    def get_state_summary(self) -> str:
        """상태 요약"""
        return f"{self.mode.value} | {self.state.to_context_summary()}"

    def reset_failed_agents(self) -> None:
        """실패한 에이전트 목록 초기화"""
        self._failed_agents.clear()
        logger.info("Failed agents list cleared")


# =============================================================================
# 싱글톤
# =============================================================================

_brain_instance: Optional[UnifiedBrain] = None


def get_brain() -> UnifiedBrain:
    """통합 두뇌 싱글톤 반환"""
    global _brain_instance

    if _brain_instance is None:
        _brain_instance = UnifiedBrain()
        logger.info("UnifiedBrain singleton created (LLM-First)")

    return _brain_instance


async def get_initialized_brain() -> UnifiedBrain:
    """초기화된 통합 두뇌 싱글톤 반환"""
    brain = get_brain()
    if not brain._initialized:
        await brain.initialize()
    return brain


def reset_brain() -> None:
    """싱글톤 리셋 (테스트용)"""
    global _brain_instance
    _brain_instance = None
