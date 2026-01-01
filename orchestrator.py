"""
WorkflowOrchestrator
====================
일일 워크플로우 전용 오케스트레이터 (Think-Act-Observe 루프)

역할:
- 일일 크롤링 워크플로우 실행
- 에이전트 순차 호출 (Crawl → Store → KG → Calculate → Insight → Export)
- Knowledge Graph 관리

참고:
- 챗봇/질의 처리는 core/unified_orchestrator.py 사용
- 이 파일은 배치 워크플로우 전용
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

from src.agents.crawler_agent import CrawlerAgent
from src.agents.storage_agent import StorageAgent
from src.agents.metrics_agent import MetricsAgent
from src.agents.insight_agent import InsightAgent
from src.agents.chatbot_agent import ChatbotAgent

# Hybrid Agents (신규)
from src.agents.hybrid_insight_agent import HybridInsightAgent
from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

# Ontology Components (신규)
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.business_rules import register_all_rules

from src.tools.dashboard_exporter import DashboardExporter

from src.memory.session import SessionManager
from src.memory.history import HistoryManager
from src.memory.context import ContextManager

from src.monitoring.logger import AgentLogger
from src.monitoring.tracer import ExecutionTracer
from src.monitoring.metrics import QualityMetrics

# 통합 오케스트레이터 (챗봇용)
from src.core.unified_orchestrator import get_unified_orchestrator


class WorkflowStep(Enum):
    """워크플로우 단계"""
    CRAWL = "crawl"
    STORE = "store"
    UPDATE_KG = "update_kg"  # 신규: Knowledge Graph 업데이트
    CALCULATE = "calculate"
    INSIGHT = "insight"
    EXPORT = "export"
    COMPLETE = "complete"


@dataclass
class ThinkResult:
    """Think 단계 결과"""
    next_action: str
    reasoning: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    should_continue: bool = True


@dataclass
class ActResult:
    """Act 단계 결과"""
    action: str
    success: bool
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ObserveResult:
    """Observe 단계 결과"""
    observations: List[str]
    state_updates: Dict[str, Any] = field(default_factory=dict)
    next_step: Optional[WorkflowStep] = None


class Orchestrator:
    """
    일일 워크플로우 전용 오케스트레이터 (Think-Act-Observe 루프)

    워크플로우:
    1. Crawl: Amazon 베스트셀러 크롤링
    2. Store: Google Sheets 저장
    3. Update KG: Knowledge Graph 업데이트
    4. Calculate: 지표 계산
    5. Insight: 하이브리드 인사이트 생성 (Ontology + RAG + LLM)
    6. Export: 대시보드 데이터 내보내기

    참고:
    - 챗봇/질의 처리는 chat() 메서드가 통합 오케스트레이터로 위임
    - 이 클래스는 배치 워크플로우 실행에 집중
    """

    def __init__(
        self,
        config_path: str = "./config/thresholds.json",
        spreadsheet_id: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        use_hybrid: bool = True,  # 신규: 하이브리드 모드 사용
        kg_persist_path: Optional[str] = "./data/knowledge_graph.json"  # 신규
    ):
        """
        Args:
            config_path: 설정 파일 경로
            spreadsheet_id: Google Sheets ID
            model: LLM 모델
            use_hybrid: 하이브리드 에이전트 사용 여부
            kg_persist_path: Knowledge Graph 영속화 경로
        """
        # 설정 로드
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.spreadsheet_id = spreadsheet_id
        self.model = model
        self.use_hybrid = use_hybrid

        # 모니터링 컴포넌트
        self.logger = AgentLogger("orchestrator")
        self.tracer = ExecutionTracer()
        self.metrics = QualityMetrics()

        # 메모리 컴포넌트
        self.session_manager = SessionManager()
        self.history_manager = HistoryManager()
        self.context_manager = ContextManager()

        # =========================================================================
        # Ontology 컴포넌트 (신규)
        # =========================================================================
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._reasoner: Optional[OntologyReasoner] = None
        self._kg_persist_path = kg_persist_path

        # 에이전트 (lazy initialization)
        self._crawler: Optional[CrawlerAgent] = None
        self._storage: Optional[StorageAgent] = None
        self._metrics_agent: Optional[MetricsAgent] = None
        self._insight: Optional[InsightAgent] = None
        self._chatbot: Optional[ChatbotAgent] = None
        self._dashboard_exporter: Optional[DashboardExporter] = None

        # Hybrid 에이전트 (신규)
        self._hybrid_insight: Optional[HybridInsightAgent] = None
        self._hybrid_chatbot: Optional[HybridChatbotAgent] = None

        # 현재 상태
        self._current_step = WorkflowStep.CRAWL
        self._state: Dict[str, Any] = {}
        self._session_id: Optional[str] = None

    def _load_config(self, path: str) -> Dict:
        """설정 로드"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config not found: {path}, using defaults")
            return {}

    # =========================================================================
    # Ontology 컴포넌트 (신규)
    # =========================================================================

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        """Knowledge Graph (공유 인스턴스)"""
        if self._knowledge_graph is None:
            self._knowledge_graph = KnowledgeGraph(
                persist_path=self._kg_persist_path
            )
            self.logger.info(f"Knowledge Graph initialized: {self._knowledge_graph}")
        return self._knowledge_graph

    @property
    def reasoner(self) -> OntologyReasoner:
        """Ontology Reasoner (공유 인스턴스)"""
        if self._reasoner is None:
            self._reasoner = OntologyReasoner(self.knowledge_graph)
            # 비즈니스 규칙 등록
            register_all_rules(self._reasoner)
            self.logger.info(f"Reasoner initialized with {len(self._reasoner.rules)} rules")
        return self._reasoner

    # =========================================================================
    # 기존 에이전트 초기화 (Lazy)
    # =========================================================================

    @property
    def crawler(self) -> CrawlerAgent:
        if self._crawler is None:
            self._crawler = CrawlerAgent(
                config_path=self.config_path,
                logger=AgentLogger("crawler"),
                tracer=self.tracer,
                metrics=self.metrics
            )
        return self._crawler

    @property
    def storage(self) -> StorageAgent:
        if self._storage is None:
            self._storage = StorageAgent(
                spreadsheet_id=self.spreadsheet_id,
                logger=AgentLogger("storage"),
                tracer=self.tracer,
                metrics=self.metrics
            )
        return self._storage

    @property
    def metrics_agent(self) -> MetricsAgent:
        if self._metrics_agent is None:
            self._metrics_agent = MetricsAgent(
                config_path=self.config_path,
                logger=AgentLogger("metrics"),
                tracer=self.tracer,
                metrics=self.metrics
            )
        return self._metrics_agent

    @property
    def insight(self) -> InsightAgent:
        """기존 InsightAgent (레거시)"""
        if self._insight is None:
            self._insight = InsightAgent(
                model=self.model,
                logger=AgentLogger("insight"),
                tracer=self.tracer,
                metrics=self.metrics
            )
        return self._insight

    @property
    def chatbot(self) -> ChatbotAgent:
        """기존 ChatbotAgent (레거시)"""
        if self._chatbot is None:
            self._chatbot = ChatbotAgent(
                model=self.model,
                logger=AgentLogger("chatbot"),
                tracer=self.tracer,
                metrics=self.metrics,
                context_manager=self.context_manager
            )
        return self._chatbot

    # =========================================================================
    # Hybrid 에이전트 초기화 (신규)
    # =========================================================================

    @property
    def hybrid_insight(self) -> HybridInsightAgent:
        """하이브리드 인사이트 에이전트"""
        if self._hybrid_insight is None:
            self._hybrid_insight = HybridInsightAgent(
                model=self.model,
                docs_dir=".",
                knowledge_graph=self.knowledge_graph,
                reasoner=self.reasoner,
                logger=AgentLogger("hybrid_insight"),
                tracer=self.tracer,
                metrics=self.metrics
            )
        return self._hybrid_insight

    @property
    def hybrid_chatbot(self) -> HybridChatbotAgent:
        """하이브리드 챗봇 에이전트"""
        if self._hybrid_chatbot is None:
            self._hybrid_chatbot = HybridChatbotAgent(
                model=self.model,
                docs_dir=".",
                knowledge_graph=self.knowledge_graph,
                reasoner=self.reasoner,
                logger=AgentLogger("hybrid_chatbot"),
                tracer=self.tracer,
                metrics=self.metrics,
                context_manager=self.context_manager
            )
        return self._hybrid_chatbot

    @property
    def dashboard_exporter(self) -> DashboardExporter:
        if self._dashboard_exporter is None:
            self._dashboard_exporter = DashboardExporter(
                spreadsheet_id=self.spreadsheet_id
            )
        return self._dashboard_exporter

    # =========================================================================
    # 워크플로우 실행
    # =========================================================================

    async def run_daily_workflow(
        self,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        일일 워크플로우 실행

        Args:
            categories: 크롤링할 카테고리 (None이면 전체)

        Returns:
            워크플로우 결과
        """
        # 세션 시작
        self._session_id = self.session_manager.create_session()
        self.tracer.start_trace(self._session_id)
        self.metrics.start_session()

        self.logger.info(
            f"Starting daily workflow - Session: {self._session_id}, "
            f"Hybrid: {self.use_hybrid}"
        )

        # 워크플로우 스텝 정의 (하이브리드 모드에 따라 다름)
        if self.use_hybrid:
            workflow_steps = [
                WorkflowStep.CRAWL.value,
                WorkflowStep.STORE.value,
                WorkflowStep.UPDATE_KG.value,  # 신규
                WorkflowStep.CALCULATE.value,
                WorkflowStep.INSIGHT.value,
                WorkflowStep.EXPORT.value
            ]
        else:
            workflow_steps = [
                WorkflowStep.CRAWL.value,
                WorkflowStep.STORE.value,
                WorkflowStep.CALCULATE.value,
                WorkflowStep.INSIGHT.value,
                WorkflowStep.EXPORT.value
            ]

        self.context_manager.start_workflow(workflow_steps)

        self._current_step = WorkflowStep.CRAWL
        self._state = {"categories": categories}

        results = {
            "session_id": self._session_id,
            "started_at": datetime.now().isoformat(),
            "steps": {},
            "status": "running",
            "hybrid_mode": self.use_hybrid
        }

        try:
            # Think-Act-Observe 루프
            while self._current_step != WorkflowStep.COMPLETE:
                step_name = self._current_step.value

                self.logger.workflow_step(step_name, "start")
                self.session_manager.start_agent(self._session_id, step_name)

                # Think
                think_result = await self._think()
                self.logger.debug(f"Think: {think_result.reasoning}")

                if not think_result.should_continue:
                    self.logger.warning(f"Workflow stopped at {step_name}")
                    break

                # Act
                act_result = await self._act(think_result)

                if act_result.success:
                    self.session_manager.complete_agent(
                        self._session_id, step_name, act_result.result
                    )
                    results["steps"][step_name] = {
                        "status": "completed",
                        "result": act_result.result
                    }
                else:
                    self.session_manager.fail_agent(
                        self._session_id, step_name, act_result.error
                    )
                    results["steps"][step_name] = {
                        "status": "failed",
                        "error": act_result.error
                    }

                # Observe
                observe_result = await self._observe(act_result)

                # 상태 업데이트
                self._state.update(observe_result.state_updates)

                # 다음 스텝
                if observe_result.next_step:
                    self._current_step = observe_result.next_step
                    self.context_manager.advance_workflow(act_result.result)
                else:
                    self._current_step = WorkflowStep.COMPLETE

                self.logger.workflow_step(step_name, "complete")

            # 완료
            results["status"] = "completed"
            results["completed_at"] = datetime.now().isoformat()

            # 최종 결과
            results["summary"] = self._generate_summary()

            # Knowledge Graph 저장
            if self.use_hybrid and self._knowledge_graph:
                self._knowledge_graph.save()
                self.logger.info("Knowledge Graph saved")

        except Exception as e:
            self.logger.error(f"Workflow failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)

        finally:
            # 세션 종료
            session_summary = self.session_manager.end_session(self._session_id)
            self.history_manager.add_execution(session_summary)

            trace_summary = self.tracer.end_trace()
            metrics_summary = self.metrics.end_session(results["status"])

            results["trace"] = trace_summary
            results["metrics"] = metrics_summary

            self.logger.info(f"Workflow completed - Status: {results['status']}")

        return results

    async def _think(self) -> ThinkResult:
        """
        Think 단계: 다음 행동 결정
        """
        step = self._current_step

        if step == WorkflowStep.CRAWL:
            return ThinkResult(
                next_action="crawl",
                reasoning="일일 크롤링 시작. Amazon 베스트셀러 Top 100 수집",
                parameters={"categories": self._state.get("categories")}
            )

        elif step == WorkflowStep.STORE:
            crawl_data = self._state.get("crawl_result")
            if not crawl_data or crawl_data.get("status") == "failed":
                return ThinkResult(
                    next_action="skip",
                    reasoning="크롤링 실패로 저장 단계 스킵",
                    should_continue=False
                )
            return ThinkResult(
                next_action="store",
                reasoning="크롤링 데이터를 Google Sheets에 저장",
                parameters={"crawl_data": crawl_data}
            )

        elif step == WorkflowStep.UPDATE_KG:
            # 신규: Knowledge Graph 업데이트
            crawl_data = self._state.get("crawl_result")
            return ThinkResult(
                next_action="update_kg",
                reasoning="Knowledge Graph에 크롤링 데이터 반영 (엔티티 관계 구축)",
                parameters={"crawl_data": crawl_data}
            )

        elif step == WorkflowStep.CALCULATE:
            crawl_data = self._state.get("crawl_result")
            if not crawl_data:
                return ThinkResult(
                    next_action="skip",
                    reasoning="크롤링 데이터 없음으로 지표 계산 스킵",
                    should_continue=False
                )
            return ThinkResult(
                next_action="calculate",
                reasoning="10개 전략 지표 계산 (SoS, HHI, CPI 등)",
                parameters={
                    "crawl_data": crawl_data,
                    "historical_data": self._state.get("historical_data")
                }
            )

        elif step == WorkflowStep.INSIGHT:
            metrics_data = self._state.get("metrics_result")
            if not metrics_data:
                return ThinkResult(
                    next_action="skip",
                    reasoning="지표 데이터 없음으로 인사이트 생성 스킵",
                    should_continue=False
                )

            if self.use_hybrid:
                return ThinkResult(
                    next_action="hybrid_insight",
                    reasoning="하이브리드 인사이트 생성 (Ontology 추론 + RAG + LLM)",
                    parameters={
                        "metrics_data": metrics_data,
                        "crawl_data": self._state.get("crawl_result"),
                        "crawl_summary": self._state.get("crawl_result", {}).get("summary")
                    }
                )
            else:
                return ThinkResult(
                    next_action="insight",
                    reasoning="일일 인사이트 생성 (LLM 기반)",
                    parameters={
                        "metrics_data": metrics_data,
                        "crawl_summary": self._state.get("crawl_result", {}).get("summary")
                    }
                )

        elif step == WorkflowStep.EXPORT:
            return ThinkResult(
                next_action="export",
                reasoning="Dashboard용 JSON 데이터 생성",
                parameters={}
            )

        return ThinkResult(
            next_action="complete",
            reasoning="모든 단계 완료",
            should_continue=False
        )

    async def _act(self, think_result: ThinkResult) -> ActResult:
        """
        Act 단계: 행동 실행
        """
        action = think_result.next_action
        params = think_result.parameters

        try:
            if action == "crawl":
                result = await self.crawler.execute(params.get("categories"))
                return ActResult(action=action, success=True, result=result)

            elif action == "store":
                result = await self.storage.execute(params.get("crawl_data"))
                return ActResult(action=action, success=True, result=result)

            elif action == "update_kg":
                # 신규: Knowledge Graph 업데이트
                crawl_data = params.get("crawl_data", {})
                added = self.knowledge_graph.load_from_crawl_data(crawl_data)
                kg_stats = self.knowledge_graph.get_stats()

                result = {
                    "relations_added": added,
                    "total_triples": kg_stats.get("total_triples", 0),
                    "unique_subjects": kg_stats.get("unique_subjects", 0),
                    "unique_objects": kg_stats.get("unique_objects", 0)
                }
                return ActResult(action=action, success=True, result=result)

            elif action == "calculate":
                result = await self.metrics_agent.execute(
                    params.get("crawl_data"),
                    params.get("historical_data")
                )
                return ActResult(action=action, success=True, result=result)

            elif action == "insight":
                # 레거시 인사이트 에이전트
                result = await self.insight.execute(
                    params.get("metrics_data"),
                    params.get("crawl_summary")
                )
                return ActResult(action=action, success=True, result=result)

            elif action == "hybrid_insight":
                # 신규: 하이브리드 인사이트 에이전트
                result = await self.hybrid_insight.execute(
                    metrics_data=params.get("metrics_data"),
                    crawl_data=params.get("crawl_data"),
                    crawl_summary=params.get("crawl_summary")
                )
                return ActResult(action=action, success=True, result=result)

            elif action == "export":
                await self.dashboard_exporter.initialize()
                result = await self.dashboard_exporter.export_dashboard_data(
                    "./data/dashboard_data.json"
                )
                return ActResult(action=action, success=True, result={
                    "exported": True,
                    "path": "./data/dashboard_data.json",
                    "products": result.get("metadata", {}).get("total_products", 0),
                    "laneige_count": result.get("metadata", {}).get("laneige_products", 0)
                })

            elif action == "skip":
                return ActResult(action=action, success=True, result={"skipped": True})

            else:
                return ActResult(
                    action=action,
                    success=False,
                    error=f"Unknown action: {action}"
                )

        except Exception as e:
            self.logger.error(f"Action {action} failed: {e}")
            return ActResult(action=action, success=False, error=str(e))

    async def _observe(self, act_result: ActResult) -> ObserveResult:
        """
        Observe 단계: 결과 관찰 및 상태 업데이트
        """
        observations = []
        state_updates = {}
        next_step = None

        if act_result.action == "crawl":
            result = act_result.result
            observations.append(
                f"크롤링 완료: {result.get('total_products', 0)} 제품, "
                f"LANEIGE {result.get('laneige_count', 0)}개"
            )
            state_updates["crawl_result"] = result
            next_step = WorkflowStep.STORE

            self.context_manager.update_crawl_data(
                categories=list(result.get("categories", {}).keys()),
                products_count=result.get("total_products", 0),
                laneige_products=result.get("laneige_products", [])
            )

        elif act_result.action == "store":
            result = act_result.result
            observations.append(
                f"저장 완료: {result.get('raw_records', 0)} 레코드, "
                f"{result.get('products_upserted', 0)} 제품"
            )
            state_updates["storage_result"] = result

            # 하이브리드 모드면 KG 업데이트로, 아니면 지표 계산으로
            if self.use_hybrid:
                next_step = WorkflowStep.UPDATE_KG
            else:
                next_step = WorkflowStep.CALCULATE

        elif act_result.action == "update_kg":
            # 신규: KG 업데이트 관찰
            result = act_result.result
            observations.append(
                f"Knowledge Graph 업데이트: {result.get('relations_added', 0)} 관계 추가, "
                f"총 {result.get('total_triples', 0)} 트리플"
            )
            state_updates["kg_result"] = result
            next_step = WorkflowStep.CALCULATE

        elif act_result.action == "calculate":
            result = act_result.result
            alert_count = len(result.get("alerts", []))
            observations.append(
                f"지표 계산 완료: {len(result.get('brand_metrics', []))} 브랜드, "
                f"{len(result.get('product_metrics', []))} 제품, "
                f"알림 {alert_count}건"
            )
            state_updates["metrics_result"] = result
            next_step = WorkflowStep.INSIGHT

            self.context_manager.set_metrics_calculated(True)

            # KG에 지표 데이터 반영
            if self.use_hybrid:
                self.knowledge_graph.load_from_metrics_data(result)

        elif act_result.action in ["insight", "hybrid_insight"]:
            result = act_result.result
            inferences_count = len(result.get("inferences", []))
            observations.append(
                f"인사이트 생성 완료: 액션 {len(result.get('action_items', []))}건, "
                f"하이라이트 {len(result.get('highlights', []))}건"
                + (f", 추론 {inferences_count}건" if inferences_count else "")
            )
            state_updates["insight_result"] = result
            next_step = WorkflowStep.EXPORT

            self.context_manager.set_insights_generated(True)

            # 챗봇 데이터 컨텍스트 설정
            if self.use_hybrid:
                self.hybrid_chatbot.set_data_context(
                    self._state.get("metrics_result", {})
                )
            else:
                self.chatbot.set_data_context(
                    self._state.get("metrics_result", {})
                )

        elif act_result.action == "export":
            result = act_result.result
            observations.append(
                f"Dashboard 데이터 내보내기 완료: {result.get('path', '')} "
                f"(제품 {result.get('products', 0)}개)"
            )
            state_updates["export_result"] = result
            next_step = WorkflowStep.COMPLETE

        elif act_result.action == "skip":
            observations.append("단계 스킵됨")
            next_step = WorkflowStep.COMPLETE

        # 에러 관찰
        if not act_result.success:
            observations.append(f"에러 발생: {act_result.error}")
            self.context_manager.record_workflow_error(
                act_result.action,
                act_result.error or "Unknown error"
            )

        for obs in observations:
            self.logger.info(f"Observe: {obs}")

        return ObserveResult(
            observations=observations,
            state_updates=state_updates,
            next_step=next_step
        )

    def _generate_summary(self) -> Dict[str, Any]:
        """최종 요약 생성"""
        crawl = self._state.get("crawl_result", {})
        metrics = self._state.get("metrics_result", {})
        insight = self._state.get("insight_result", {})
        export = self._state.get("export_result", {})
        kg = self._state.get("kg_result", {})

        summary = {
            "products_crawled": crawl.get("total_products", 0),
            "laneige_tracked": crawl.get("laneige_count", 0),
            "categories": list(crawl.get("categories", {}).keys()),
            "alerts": len(metrics.get("alerts", [])),
            "action_items": len(insight.get("action_items", [])),
            "daily_insight": insight.get("daily_insight", "")[:200] + "...",
            "dashboard_exported": export.get("exported", False),
            "dashboard_path": export.get("path", "")
        }

        # 하이브리드 모드 추가 정보
        if self.use_hybrid:
            summary["hybrid"] = {
                "kg_triples": kg.get("total_triples", 0),
                "inferences": len(insight.get("inferences", [])),
                "explanations": len(insight.get("explanations", []))
            }

        return summary

    # =========================================================================
    # 챗봇 인터페이스 (통합 오케스트레이터로 위임)
    # =========================================================================

    async def chat(self, message: str) -> Dict[str, Any]:
        """
        챗봇 질의 - 통합 오케스트레이터로 위임

        Args:
            message: 사용자 메시지

        Returns:
            챗봇 응답
        """
        orchestrator = get_unified_orchestrator()
        response = await orchestrator.process(
            query=message,
            session_id=self._session_id,
            current_metrics=self._state.get("metrics_result")
        )
        return response.to_dict()

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        질의 처리 - 통합 오케스트레이터로 위임

        Args:
            query: 사용자 질문

        Returns:
            Response 딕셔너리
        """
        orchestrator = get_unified_orchestrator()
        response = await orchestrator.process(
            query=query,
            session_id=self._session_id,
            current_metrics=self._state.get("metrics_result")
        )
        return response.to_dict()

    # =========================================================================
    # 유틸리티
    # =========================================================================

    async def cleanup(self) -> None:
        """리소스 정리"""
        if self._crawler:
            await self._crawler.close()

        # Knowledge Graph 저장
        if self._knowledge_graph:
            self._knowledge_graph.save()

        self.metrics.save()
        self.logger.info("Orchestrator cleanup completed")

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        status = {
            "session_id": self._session_id,
            "current_step": self._current_step.value if self._current_step else None,
            "workflow": self.context_manager.get_workflow_status(),
            "data": self.context_manager.get_data_status(),
            "hybrid_mode": self.use_hybrid
        }

        # 하이브리드 모드 추가 상태
        if self.use_hybrid and self._knowledge_graph:
            status["knowledge_graph"] = self._knowledge_graph.get_stats()

        if self.use_hybrid and self._reasoner:
            status["reasoner"] = self._reasoner.get_inference_stats()

        return status

    def get_history_stats(self) -> Dict[str, Any]:
        """히스토리 통계 조회"""
        return {
            "success_rate": self.history_manager.get_success_rate(),
            "avg_duration": self.history_manager.get_average_duration(),
            "recent_errors": self.history_manager.get_error_summary(days=7)
        }

    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Knowledge Graph 통계 조회"""
        if not self._knowledge_graph:
            return {"initialized": False}

        stats = self._knowledge_graph.get_stats()
        stats["initialized"] = True
        stats["most_connected"] = self._knowledge_graph.get_most_connected(5)

        return stats

    def get_inference_stats(self) -> Dict[str, Any]:
        """추론 통계 조회"""
        if not self._reasoner:
            return {"initialized": False}

        return self._reasoner.get_inference_stats()
