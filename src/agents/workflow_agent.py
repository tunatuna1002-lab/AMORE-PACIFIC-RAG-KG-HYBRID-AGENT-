"""
워크플로우 에이전트 (Workflow Agent)
=====================================
일일 배치 워크플로우 전담 에이전트

역할:
- 일일 크롤링 워크플로우 실행 (Think-Act-Observe 루프)
- 에이전트 순차 호출 (Crawl → Store → KG → Calculate → Insight → Export)
- Knowledge Graph 관리

이 에이전트는 core/brain.py에서 호출되어 작동합니다.

Usage:
    workflow_agent = WorkflowAgent()
    result = await workflow_agent.run_workflow(categories=["skincare"])
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from src.agents.crawler_agent import CrawlerAgent
from src.agents.storage_agent import StorageAgent
from src.agents.metrics_agent import MetricsAgent
from src.agents.hybrid_insight_agent import HybridInsightAgent

from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.business_rules import register_all_rules

from src.tools.dashboard_exporter import DashboardExporter

from src.monitoring.logger import AgentLogger
from src.monitoring.tracer import ExecutionTracer
from src.monitoring.metrics import QualityMetrics

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 워크플로우 단계 정의
# =============================================================================

class WorkflowStep(Enum):
    """워크플로우 단계"""
    CRAWL = "crawl"
    STORE = "store"
    UPDATE_KG = "update_kg"
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


# =============================================================================
# 워크플로우 에이전트
# =============================================================================

class WorkflowAgent:
    """
    워크플로우 전담 에이전트

    Think-Act-Observe 루프를 통해 일일 배치 워크플로우를 실행합니다.

    워크플로우 순서:
    1. Crawl: Amazon 베스트셀러 크롤링
    2. Store: Google Sheets 저장
    3. Update KG: Knowledge Graph 업데이트
    4. Calculate: 지표 계산
    5. Insight: 인사이트 생성
    6. Export: 대시보드 데이터 내보내기
    """

    def __init__(
        self,
        config_path: str = "./config/thresholds.json",
        spreadsheet_id: Optional[str] = None,
        model: str = "gpt-4o-mini",
        use_hybrid: bool = True,
        kg_persist_path: str = "./data/knowledge_graph.json"
    ):
        """
        Args:
            config_path: 설정 파일 경로
            spreadsheet_id: Google Sheets ID
            model: LLM 모델
            use_hybrid: 하이브리드 모드 사용 여부
            kg_persist_path: Knowledge Graph 영속화 경로
        """
        import os
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.spreadsheet_id = spreadsheet_id or os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID")
        self.model = model
        self.use_hybrid = use_hybrid
        self.kg_persist_path = kg_persist_path

        # 모니터링
        self.logger = AgentLogger("workflow_agent")
        self.tracer = ExecutionTracer()
        self.metrics = QualityMetrics()

        # 에이전트 (lazy init)
        self._crawler: Optional[CrawlerAgent] = None
        self._storage: Optional[StorageAgent] = None
        self._metrics_agent: Optional[MetricsAgent] = None
        self._hybrid_insight: Optional[HybridInsightAgent] = None
        self._dashboard_exporter: Optional[DashboardExporter] = None

        # Ontology 컴포넌트
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._reasoner: Optional[OntologyReasoner] = None

        # 상태
        self._current_step = WorkflowStep.CRAWL
        self._state: Dict[str, Any] = {}
        self._session_id: Optional[str] = None

        # 콜백
        self._on_step_complete: Optional[Callable] = None
        self._on_workflow_complete: Optional[Callable] = None

    def _load_config(self, path: str) -> Dict:
        """설정 로드"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    # =========================================================================
    # Lazy 초기화
    # =========================================================================

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        if self._knowledge_graph is None:
            self._knowledge_graph = KnowledgeGraph(persist_path=self.kg_persist_path)
        return self._knowledge_graph

    @property
    def reasoner(self) -> OntologyReasoner:
        if self._reasoner is None:
            self._reasoner = OntologyReasoner(self.knowledge_graph)
            register_all_rules(self._reasoner)
        return self._reasoner

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
    def hybrid_insight(self) -> HybridInsightAgent:
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
    def dashboard_exporter(self) -> DashboardExporter:
        if self._dashboard_exporter is None:
            self._dashboard_exporter = DashboardExporter(
                spreadsheet_id=self.spreadsheet_id
            )
        return self._dashboard_exporter

    # =========================================================================
    # 콜백 설정
    # =========================================================================

    def on_step_complete(self, callback: Callable) -> None:
        """단계 완료 콜백 설정"""
        self._on_step_complete = callback

    def on_workflow_complete(self, callback: Callable) -> None:
        """워크플로우 완료 콜백 설정"""
        self._on_workflow_complete = callback

    # =========================================================================
    # 워크플로우 실행
    # =========================================================================

    async def run_workflow(
        self,
        categories: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        일일 워크플로우 실행

        Args:
            categories: 크롤링할 카테고리
            session_id: 세션 ID

        Returns:
            워크플로우 결과
        """
        self._session_id = session_id or f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tracer.start_trace(self._session_id)
        self.metrics.start_session()

        logger.info(f"Starting workflow - Session: {self._session_id}, Hybrid: {self.use_hybrid}")

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
                logger.info(f"Step: {step_name}")

                # Think
                think_result = await self._think()

                if not think_result.should_continue:
                    logger.warning(f"Workflow stopped at {step_name}")
                    break

                # Act
                act_result = await self._act(think_result)

                results["steps"][step_name] = {
                    "status": "completed" if act_result.success else "failed",
                    "result": act_result.result if act_result.success else None,
                    "error": act_result.error
                }

                # Observe
                observe_result = await self._observe(act_result)
                self._state.update(observe_result.state_updates)

                # 콜백 호출
                if self._on_step_complete:
                    try:
                        if asyncio.iscoroutinefunction(self._on_step_complete):
                            await self._on_step_complete(step_name, act_result)
                        else:
                            self._on_step_complete(step_name, act_result)
                    except Exception as e:
                        logger.error(f"Step callback error: {e}")

                # 다음 단계
                if observe_result.next_step:
                    self._current_step = observe_result.next_step
                else:
                    self._current_step = WorkflowStep.COMPLETE

            # 완료
            results["status"] = "completed"
            results["completed_at"] = datetime.now().isoformat()
            results["summary"] = self._generate_summary()

            # KG 저장
            if self.use_hybrid and self._knowledge_graph:
                self._knowledge_graph.save()

            # 완료 콜백
            if self._on_workflow_complete:
                try:
                    if asyncio.iscoroutinefunction(self._on_workflow_complete):
                        await self._on_workflow_complete(results)
                    else:
                        self._on_workflow_complete(results)
                except Exception as e:
                    logger.error(f"Workflow callback error: {e}")

        except Exception as e:
            logger.error(f"Workflow failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)

        finally:
            trace_summary = self.tracer.end_trace()
            metrics_summary = self.metrics.end_session(results["status"])
            results["trace"] = trace_summary
            results["metrics"] = metrics_summary

        return results

    # =========================================================================
    # Think-Act-Observe
    # =========================================================================

    async def _think(self) -> ThinkResult:
        """Think: 다음 행동 결정"""
        step = self._current_step

        if step == WorkflowStep.CRAWL:
            return ThinkResult(
                next_action="crawl",
                reasoning="Amazon 베스트셀러 Top 100 수집",
                parameters={"categories": self._state.get("categories")}
            )

        elif step == WorkflowStep.STORE:
            crawl_data = self._state.get("crawl_result")
            if not crawl_data or crawl_data.get("status") == "failed":
                return ThinkResult(
                    next_action="skip",
                    reasoning="크롤링 실패로 저장 스킵",
                    should_continue=False
                )
            return ThinkResult(
                next_action="store",
                reasoning="크롤링 데이터를 Google Sheets에 저장",
                parameters={"crawl_data": crawl_data}
            )

        elif step == WorkflowStep.UPDATE_KG:
            crawl_data = self._state.get("crawl_result")
            return ThinkResult(
                next_action="update_kg",
                reasoning="Knowledge Graph 업데이트",
                parameters={"crawl_data": crawl_data}
            )

        elif step == WorkflowStep.CALCULATE:
            crawl_data = self._state.get("crawl_result")
            if not crawl_data:
                return ThinkResult(
                    next_action="skip",
                    reasoning="크롤링 데이터 없음",
                    should_continue=False
                )
            return ThinkResult(
                next_action="calculate",
                reasoning="10개 전략 지표 계산",
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
                    reasoning="지표 데이터 없음",
                    should_continue=False
                )

            return ThinkResult(
                next_action="hybrid_insight",
                reasoning="하이브리드 인사이트 생성 (Ontology + RAG + LLM)",
                parameters={
                    "metrics_data": metrics_data,
                    "crawl_data": self._state.get("crawl_result")
                }
            )

        elif step == WorkflowStep.EXPORT:
            return ThinkResult(
                next_action="export",
                reasoning="Dashboard 데이터 생성",
                parameters={}
            )

        return ThinkResult(
            next_action="complete",
            reasoning="모든 단계 완료",
            should_continue=False
        )

    async def _act(self, think_result: ThinkResult) -> ActResult:
        """Act: 행동 실행"""
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
                crawl_data = params.get("crawl_data", {})
                added = self.knowledge_graph.load_from_crawl_data(crawl_data)
                kg_stats = self.knowledge_graph.get_stats()
                result = {
                    "relations_added": added,
                    "total_triples": kg_stats.get("total_triples", 0)
                }
                return ActResult(action=action, success=True, result=result)

            elif action == "calculate":
                result = await self.metrics_agent.execute(
                    params.get("crawl_data"),
                    params.get("historical_data")
                )
                return ActResult(action=action, success=True, result=result)

            elif action == "hybrid_insight":
                result = await self.hybrid_insight.execute(
                    metrics_data=params.get("metrics_data"),
                    crawl_data=params.get("crawl_data")
                )
                return ActResult(action=action, success=True, result=result)

            elif action == "export":
                await self.dashboard_exporter.initialize()
                result = await self.dashboard_exporter.export_dashboard_data(
                    "./data/dashboard_data.json"
                )
                return ActResult(action=action, success=True, result={
                    "exported": True,
                    "path": "./data/dashboard_data.json"
                })

            elif action == "skip":
                return ActResult(action=action, success=True, result={"skipped": True})

            else:
                return ActResult(action=action, success=False, error=f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Action {action} failed: {e}")
            return ActResult(action=action, success=False, error=str(e))

    async def _observe(self, act_result: ActResult) -> ObserveResult:
        """Observe: 결과 관찰 및 상태 업데이트"""
        observations = []
        state_updates = {}
        next_step = None

        if act_result.action == "crawl":
            result = act_result.result
            observations.append(f"크롤링 완료: {result.get('total_products', 0)} 제품")
            state_updates["crawl_result"] = result
            next_step = WorkflowStep.STORE

        elif act_result.action == "store":
            observations.append("저장 완료")
            state_updates["storage_result"] = act_result.result
            next_step = WorkflowStep.UPDATE_KG if self.use_hybrid else WorkflowStep.CALCULATE

        elif act_result.action == "update_kg":
            result = act_result.result
            observations.append(f"KG 업데이트: {result.get('relations_added', 0)} 관계")
            state_updates["kg_result"] = result
            next_step = WorkflowStep.CALCULATE

        elif act_result.action == "calculate":
            observations.append("지표 계산 완료")
            state_updates["metrics_result"] = act_result.result
            next_step = WorkflowStep.INSIGHT

            if self.use_hybrid:
                self.knowledge_graph.load_from_metrics_data(act_result.result)

        elif act_result.action == "hybrid_insight":
            observations.append("인사이트 생성 완료")
            state_updates["insight_result"] = act_result.result
            next_step = WorkflowStep.EXPORT

        elif act_result.action == "export":
            observations.append("Dashboard 내보내기 완료")
            state_updates["export_result"] = act_result.result
            next_step = WorkflowStep.COMPLETE

        elif act_result.action == "skip":
            observations.append("단계 스킵됨")
            next_step = WorkflowStep.COMPLETE

        if not act_result.success:
            observations.append(f"에러: {act_result.error}")

        for obs in observations:
            logger.info(f"Observe: {obs}")

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
        kg = self._state.get("kg_result", {})

        summary = {
            "products_crawled": crawl.get("total_products", 0),
            "laneige_tracked": crawl.get("laneige_count", 0),
            "alerts": len(metrics.get("alerts", [])),
            "action_items": len(insight.get("action_items", []))
        }

        if self.use_hybrid:
            summary["hybrid"] = {
                "kg_triples": kg.get("total_triples", 0),
                "inferences": len(insight.get("inferences", []))
            }

        return summary

    # =========================================================================
    # 유틸리티
    # =========================================================================

    async def cleanup(self) -> None:
        """리소스 정리"""
        if self._crawler:
            await self._crawler.close()

        if self._knowledge_graph:
            self._knowledge_graph.save()

        self.metrics.save()
        logger.info("WorkflowAgent cleanup completed")

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        return {
            "session_id": self._session_id,
            "current_step": self._current_step.value if self._current_step else None,
            "hybrid_mode": self.use_hybrid
        }

    def get_state(self) -> Dict[str, Any]:
        """현재 상태 데이터 반환"""
        return self._state.copy()

    def get_metrics_data(self) -> Optional[Dict[str, Any]]:
        """지표 데이터 반환"""
        return self._state.get("metrics_result")
