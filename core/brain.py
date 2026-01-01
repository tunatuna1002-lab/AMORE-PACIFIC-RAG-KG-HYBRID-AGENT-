"""
통합 두뇌 (Unified Brain)
=========================
Level 4 Autonomous Agent의 핵심 두뇌

역할:
1. 모든 에이전트를 통제하는 단일 중앙 제어
2. 규칙 + LLM 하이브리드 의사결정
3. 자율 작업과 사용자 요청 우선순위 관리
4. 상태 관리 및 이벤트 처리

동작 모드:
- 자율 모드: 스케줄 기반 자동 작업 (크롤링, 분석, 알림)
- 대화 모드: 사용자 질문 처리 (최우선)

의사결정 흐름:
1. 규칙 엔진 먼저 확인 (빠름, 비용 없음)
2. 규칙 매칭 없으면 LLM 판단 (유연함)
3. 권한 확인 후 실행

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
from datetime import datetime, time
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

from .rules_engine import RulesEngine, RuleMatch
from .models import Context, Response, Decision, ToolResult, ConfidenceLevel
from .confidence import ConfidenceAssessor
from .cache import ResponseCache
from .state import OrchestratorState
from .context_gatherer import ContextGatherer
from .tools import ToolExecutor, AGENT_TOOLS
from .response_pipeline import ResponsePipeline

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


# 에이전트별 에러 전략
AGENT_ERROR_STRATEGIES: Dict[str, ErrorStrategy] = {
    "crawl_amazon": ErrorStrategy.FALLBACK,
    "calculate_metrics": ErrorStrategy.RETRY,
    "query_data": ErrorStrategy.FALLBACK,
    "query_knowledge_graph": ErrorStrategy.SKIP,
    "generate_insight": ErrorStrategy.RETRY,
    "send_alert": ErrorStrategy.RETRY,
}


# =============================================================================
# 통합 두뇌
# =============================================================================

class UnifiedBrain:
    """
    Level 4 Autonomous Agent의 통합 두뇌

    모든 에이전트를 통제하고, 자율 작업과 사용자 요청을 관리합니다.

    핵심 원칙:
    1. 사용자 요청은 항상 최우선
    2. 규칙으로 해결 가능하면 LLM 호출 안 함
    3. 금지된 액션은 절대 실행 안 함
    4. 에러 시 사용자에게 알림
    """

    # LLM 판단 프롬프트
    DECISION_PROMPT = """당신은 Amazon 마켓 분석 시스템의 자율 에이전트입니다.

## 현재 상태
{system_state}

## 사용 가능한 도구
{tools_description}

## 컨텍스트
{context_summary}

## 질문/상황
{query}

## 지시사항
1. 현재 상태와 컨텍스트를 분석하세요
2. 질문에 답하거나 상황을 해결할 최선의 방법을 결정하세요
3. 컨텍스트만으로 충분하면 "direct_answer"를 선택하세요
4. 도구가 필요하면 적절한 도구를 선택하세요

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
        rules_engine: Optional[RulesEngine] = None,
        context_gatherer: Optional[ContextGatherer] = None,
        tool_executor: Optional[ToolExecutor] = None,
        response_pipeline: Optional[ResponsePipeline] = None,
        cache: Optional[ResponseCache] = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 2
    ):
        """
        Args:
            rules_engine: 규칙 엔진
            context_gatherer: 컨텍스트 수집기
            tool_executor: 도구 실행기
            response_pipeline: 응답 파이프라인
            cache: 응답 캐시
            model: LLM 모델
            max_retries: 최대 재시도 횟수
        """
        # 핵심 컴포넌트
        self.rules_engine = rules_engine or RulesEngine()
        self.context_gatherer = context_gatherer or ContextGatherer()
        self.tool_executor = tool_executor or ToolExecutor()
        self.response_pipeline = response_pipeline or ResponsePipeline()
        self.cache = cache or ResponseCache()
        self.confidence_assessor = ConfidenceAssessor()

        # LLM 설정
        self.client = None
        self.model = model
        self.max_retries = max_retries

        # 상태 관리
        self.state = OrchestratorState()
        self.mode = BrainMode.IDLE
        self._current_task: Optional[BrainTask] = None

        # 작업 큐
        self._task_queue: List[BrainTask] = []
        self._task_history: List[BrainTask] = []

        # 에러 추적
        self._error_history: List[AgentError] = []
        self._failed_agents: Dict[str, datetime] = {}

        # 이벤트 콜백
        self._event_handlers: Dict[str, List[Callable]] = {}

        # 통계
        self._stats = {
            "total_queries": 0,
            "rule_decisions": 0,
            "llm_decisions": 0,
            "cache_hits": 0,
            "autonomous_tasks": 0,
            "errors": 0
        }

    # =========================================================================
    # 초기화
    # =========================================================================

    def set_client(self, client: Any) -> None:
        """OpenAI 클라이언트 설정"""
        self.client = client
        self.response_pipeline.set_client(client)

    async def initialize(self) -> None:
        """비동기 초기화"""
        await self.context_gatherer.initialize()
        logger.info(f"UnifiedBrain initialized | {self.rules_engine.to_summary()}")

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

        # 규칙 엔진에서 매칭 규칙 확인
        matches = self.rules_engine.evaluate_event(event_name, data)

        # 매칭된 규칙의 액션 실행
        for match in matches:
            await self._execute_rule_action(match)

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

    async def _execute_rule_action(self, match: RuleMatch) -> None:
        """규칙에 따른 액션 실행"""
        action = match.action

        # 권한 확인
        if not self.rules_engine.is_action_allowed(action):
            logger.warning(f"Action not allowed: {action}")
            return

        # 동의 필요 여부 확인
        if self.rules_engine.requires_consent(action):
            # TODO: 동의 확인 로직 (AlertAgent에서 처리)
            logger.info(f"Action requires consent: {action}")
            return

        logger.info(f"Executing rule action: {action} (rule: {match.rule.id})")

        # 작업 큐에 추가
        task = BrainTask(
            id=f"rule_{match.rule.id}_{datetime.now().timestamp()}",
            type="rule",
            priority=TaskPriority.SCHEDULED,
            payload={
                "action": action,
                "rule_id": match.rule.id,
                "context": match.context,
                "alert_type": match.rule.alert_type
            }
        )
        self._add_task(task)

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
            entities = self._extract_entities(query)
            context = await self.context_gatherer.gather(
                query=query,
                entities=entities,
                current_metrics=current_metrics
            )

            # 4. 의사결정 (규칙 → LLM)
            decision = await self._make_decision(query, context, system_state)

            # 5. 도구 실행 (필요시)
            tool_result = None
            if decision.requires_tool():
                self.mode = BrainMode.EXECUTING
                tool_result = await self._execute_tool(
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
    # 의사결정 (규칙 + LLM 하이브리드)
    # =========================================================================

    async def _make_decision(
        self,
        query: str,
        context: Context,
        system_state: Dict[str, Any]
    ) -> Decision:
        """
        하이브리드 의사결정

        1. 규칙 기반 판단 시도 (빠름, 비용 없음)
        2. 규칙 매칭 없으면 LLM 판단 (유연함)

        Args:
            query: 질문
            context: 컨텍스트
            system_state: 시스템 상태

        Returns:
            Decision
        """
        # 1. 규칙 기반 판단 시도
        rule_decision = self._try_rule_based_decision(query, context, system_state)
        if rule_decision:
            self._stats["rule_decisions"] += 1
            logger.info(f"Rule-based decision: {rule_decision.tool}")
            return rule_decision

        # 2. LLM 판단
        return await self._get_llm_decision(query, context, system_state)

    def _try_rule_based_decision(
        self,
        query: str,
        context: Context,
        system_state: Dict[str, Any]
    ) -> Optional[Decision]:
        """
        규칙 기반 의사결정 시도

        키워드 매칭으로 빠르게 판단 가능한 경우 사용
        """
        query_lower = query.lower()

        # 크롤링 요청
        if any(kw in query_lower for kw in ["크롤링", "수집", "업데이트", "새로고침", "refresh"]):
            if "crawl_amazon" in system_state.get("available_tools", []):
                return Decision(
                    tool="crawl_amazon",
                    reason="크롤링 요청 감지",
                    confidence=0.95
                )

        # 지표 계산 요청
        if any(kw in query_lower for kw in ["계산", "분석해", "지표 계산"]):
            if "calculate_metrics" in system_state.get("available_tools", []):
                return Decision(
                    tool="calculate_metrics",
                    reason="지표 계산 요청 감지",
                    confidence=0.9
                )

        # 인사이트 요청
        if any(kw in query_lower for kw in ["인사이트", "분석 결과", "요약"]):
            if "generate_insight" in system_state.get("available_tools", []):
                return Decision(
                    tool="generate_insight",
                    reason="인사이트 생성 요청",
                    confidence=0.85
                )

        # 컨텍스트가 충분하면 직접 응답
        if context.has_sufficient_context():
            return Decision(
                tool="direct_answer",
                reason="컨텍스트 충분",
                confidence=0.8
            )

        # 규칙 매칭 실패 → LLM 판단 필요
        return None

    async def _get_llm_decision(
        self,
        query: str,
        context: Context,
        system_state: Dict[str, Any]
    ) -> Decision:
        """LLM 기반 의사결정"""
        self._stats["llm_decisions"] += 1

        if not self.client:
            # LLM 없으면 기본 응답
            return Decision(tool="direct_answer", reason="LLM 미설정", confidence=0.5)

        try:
            prompt = self.DECISION_PROMPT.format(
                system_state=self._format_system_state(system_state),
                tools_description=self._format_tools_description(system_state),
                context_summary=context.summary or "컨텍스트 없음",
                query=query
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )

            return self._parse_decision(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return Decision(tool="direct_answer", reason=f"LLM 오류: {e}", confidence=0.3)

    def _parse_decision(self, response_text: str) -> Decision:
        """LLM 응답 파싱"""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                return Decision(
                    tool=data.get("tool", "direct_answer"),
                    tool_params=data.get("tool_params", {}),
                    reason=data.get("reason", ""),
                    confidence=data.get("confidence", 0.7)
                )
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM decision")

        return Decision(tool="direct_answer", reason="파싱 실패", confidence=0.5)

    # =========================================================================
    # 도구 실행
    # =========================================================================

    async def _execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> ToolResult:
        """에러 처리가 포함된 도구 실행"""
        # 권한 확인
        if not self.rules_engine.is_action_allowed(tool_name):
            logger.warning(f"Tool not allowed: {tool_name}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"권한 없음: {tool_name}"
            )

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
    # 자율 작업 (Autonomous)
    # =========================================================================

    async def run_autonomous_cycle(self) -> None:
        """
        자율 작업 사이클 실행

        스케줄된 작업과 대기 중인 작업을 처리합니다.
        사용자 요청이 들어오면 즉시 중단하고 우선 처리합니다.
        """
        self._stats["autonomous_tasks"] += 1
        self.mode = BrainMode.AUTONOMOUS

        try:
            # 스케줄 규칙 확인
            state_context = {"crawl_needed": self.state.is_crawl_needed()}
            scheduled = self.rules_engine.get_scheduled_actions(
                datetime.now(), state_context
            )

            for match in scheduled:
                await self._execute_rule_action(match)

            # 대기 중인 작업 처리
            await self._process_task_queue()

        finally:
            self.mode = BrainMode.IDLE

    async def _process_task_queue(self) -> None:
        """작업 큐 처리"""
        # 우선순위 순 정렬
        self._task_queue.sort()

        while self._task_queue:
            task = self._task_queue.pop(0)

            # 사용자 요청 모드면 중단
            if self.mode == BrainMode.RESPONDING:
                self._task_queue.insert(0, task)  # 다시 넣기
                break

            await self._execute_task(task)

    async def _execute_task(self, task: BrainTask) -> None:
        """개별 작업 실행"""
        task.started_at = datetime.now()
        self._current_task = task

        try:
            action = task.payload.get("action")

            if action == "crawl_workflow":
                result = await self.tool_executor.execute("crawl_amazon", {})
                if result.success:
                    await self.emit_event("crawl_complete", {"data": result.data})
                else:
                    await self.emit_event("crawl_failed", {"error": result.error})

            elif action == "update_dashboard":
                result = await self.tool_executor.execute("calculate_metrics", {})
                task.result = result

            elif action == "send_alert":
                # AlertAgent에서 처리 (추후 구현)
                pass

            elif action == "analyze_data":
                result = await self.tool_executor.execute("generate_insight", {})
                task.result = result

            task.completed_at = datetime.now()
            self._task_history.append(task)

        except Exception as e:
            task.error = str(e)
            logger.error(f"Task execution failed: {e}")

        finally:
            self._current_task = None

    def _add_task(self, task: BrainTask) -> None:
        """작업 큐에 추가"""
        self._task_queue.append(task)
        self._task_queue.sort()  # 우선순위 정렬

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

        # 순위 변동 확인
        products = metrics_data.get("products", [])
        for product in products:
            rank_change = product.get("rank_change", 0)

            # 급락/급등 확인
            matches = self.rules_engine.evaluate_threshold("rank_change", rank_change, {
                "product": product.get("name"),
                "brand": product.get("brand"),
                "current_rank": product.get("current_rank"),
                "previous_rank": product.get("previous_rank")
            })

            for match in matches:
                alerts.append({
                    "type": match.rule.alert_type,
                    "rule_id": match.rule.id,
                    "product": product.get("name"),
                    "message": f"{product.get('name')} 순위 변동: {rank_change}",
                    "context": match.context
                })

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

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """엔티티 추출"""
        # 간단한 패턴 매칭 (추후 RAG Router 연동)
        return {}

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
            "failed_agents": list(self._failed_agents.keys())
        }

    def get_state_summary(self) -> str:
        """상태 요약"""
        return f"{self.mode.value} | {self.state.to_context_summary()}"


# =============================================================================
# 싱글톤
# =============================================================================

_brain_instance: Optional[UnifiedBrain] = None


def get_brain() -> UnifiedBrain:
    """통합 두뇌 싱글톤 반환"""
    global _brain_instance

    if _brain_instance is None:
        from rag.router import RAGRouter
        from rag.hybrid_retriever import HybridRetriever
        from ontology.knowledge_graph import KnowledgeGraph
        from ontology.reasoner import OntologyReasoner

        kg = KnowledgeGraph(persist_path="./data/knowledge_graph.json")
        reasoner = OntologyReasoner(kg)
        hybrid_retriever = HybridRetriever(kg, reasoner)

        context_gatherer = ContextGatherer(
            hybrid_retriever=hybrid_retriever,
            orchestrator_state=OrchestratorState()
        )

        _brain_instance = UnifiedBrain(
            rules_engine=RulesEngine(),
            context_gatherer=context_gatherer,
            tool_executor=ToolExecutor(),
            response_pipeline=ResponsePipeline(),
            cache=ResponseCache()
        )

        logger.info("UnifiedBrain singleton created")

    return _brain_instance


def reset_brain() -> None:
    """싱글톤 리셋 (테스트용)"""
    global _brain_instance
    _brain_instance = None
