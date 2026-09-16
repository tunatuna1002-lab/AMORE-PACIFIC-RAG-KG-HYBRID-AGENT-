"""
Batch Workflow Orchestrator
============================
배치 워크플로우 전용 오케스트레이터 (Think-Act-Observe 루프)

이 모듈은 **유일한** 배치 파이프라인 진입점입니다 (F1). 스케줄러(UnifiedBrain →
CrawlManager), 독립 실행 스크립트(scripts/daily_crawl.py), CLI(main.py)는 모두
``BatchWorkflow.run_daily_workflow()`` 를 호출합니다.

역할:
- 일일 크롤링 워크플로우 실행
- 에이전트 순차 호출 (Crawl → Store → KG → Calculate → Insight → Alert → Export)
- Knowledge Graph 관리
- 시스템 상태(StateManager) 기록 (F7: 단일 시스템 상태)

챗봇/질의 처리는 core/brain.py (UnifiedBrain)를 사용합니다.

Workflow Steps:
1. Crawl: Amazon 베스트셀러 크롤링 (+ 크롤링 스냅샷 JSON 덤프)
2. Store: Google Sheets 저장
3. Update KG: Knowledge Graph 업데이트 (하이브리드 모드)
4. Calculate: 지표 계산 (SoS, HHI, CPI 등)
5. Insight: 하이브리드 인사이트 생성 (Ontology + RAG + LLM)
6. Alert: 순위 변동 알림 생성 + 발송 (AlertAgent)
7. Export: 대시보드 데이터 내보내기

최종 상태:
- "completed": 실패 없음
- "partial": 비크리티컬 스텝(update_kg/calculate/insight/alert/export) 실패,
             크롤러가 status="partial" 을 반환, 또는 저장 오류 목록이 존재
- "failed": 크리티컬 스텝(crawl/store) 실패
``result["errors"]`` 에 모든 오류가 ``"<step>: <message>"`` 형태로 나열됩니다.

Usage:
    from src.application.workflows.batch_workflow import BatchWorkflow, run_full_workflow

    # 전체 워크플로우 실행
    results = await run_full_workflow()

    # 의존성 주입 (테스트 / 드라이런)
    deps = WorkflowDependencies(crawler=FakeCrawler())
    workflow = BatchWorkflow(deps=deps)
    results = await workflow.run_daily_workflow(categories=["lip_care"])
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from src.core.brain import get_brain
from src.core.state_manager import StateManager, get_state_manager
from src.domain.interfaces.agent import (
    CrawlerAgentProtocol,
    InsightAgentProtocol,
    MetricsAgentProtocol,
    StorageAgentProtocol,
)
from src.memory.context import ContextManager
from src.memory.history import HistoryManager
from src.memory.session import SessionManager
from src.monitoring.logger import AgentLogger
from src.monitoring.metrics import QualityMetrics
from src.monitoring.tracer import ExecutionTracer
from src.ontology.business_rules import register_all_rules
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner

if TYPE_CHECKING:
    from src.tools.exporters.dashboard_exporter import DashboardExporter

# 스텝별 진행률 콜백 시그니처: (step_name, step_payload)
ProgressCallback = Callable[[str, dict[str, Any]], Any]


def _default_data_dir() -> str:
    """Railway Volume(/data) 또는 로컬(./data) 데이터 디렉토리"""
    if os.environ.get("RAILWAY_ENVIRONMENT"):
        return "/data"
    return "./data"


def _json_default(obj: Any) -> Any:
    """datetime / Decimal 등 JSON 직렬화 불가 객체 처리"""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


# =========================================================================
# Result / DI types
# =========================================================================


class WorkflowStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class WorkflowResult:
    """워크플로우 실행 결과 (구조화 버전)"""

    status: WorkflowStatus
    started_at: datetime
    completed_at: datetime | None = None
    records_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    insights: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "records_count": self.records_count,
            "metrics": self.metrics,
            "insights": self.insights,
            "errors": self.errors,
        }


@dataclass
class WorkflowDependencies:
    """워크플로우 의존성 컨테이너 (DI)

    모든 컴포넌트는 선택 사항입니다. ``None`` 인 컴포넌트는 BatchWorkflow가 처음
    접근하는 시점에 ``from_container()`` 로 **하나씩** 지연 해석합니다.
    (application 레이어는 infrastructure를 import 시점에 알지 못합니다.)
    """

    COMPONENTS: ClassVar[tuple[str, ...]] = (
        "crawler",
        "storage",
        "metrics",
        "insight",
        "alert",
        "exporter",
        "chatbot",
    )

    crawler: CrawlerAgentProtocol | None = None
    storage: StorageAgentProtocol | None = None
    metrics: MetricsAgentProtocol | None = None
    insight: InsightAgentProtocol | None = None
    alert: Any | None = None
    exporter: Any | None = None
    chatbot: Any | None = None
    knowledge_graph: KnowledgeGraph | None = None
    categories: list[str] = field(default_factory=list)

    @classmethod
    def from_container(
        cls, only: tuple[str, ...] | None = None, **context: Any
    ) -> WorkflowDependencies:
        """infrastructure DI 컨테이너에서 의존성 해석 (지연 import)

        Args:
            only: 해석할 컴포넌트 이름 목록 (None이면 전체)
            **context: 컴포넌트 생성에 필요한 컨텍스트 (config_path, spreadsheet_id,
                model, tracer, metrics, knowledge_graph, reasoner, context_manager,
                state_manager ...)
        """
        from src.infrastructure.container import Container

        return Container.build_workflow_dependencies(only=only, **context)


# =========================================================================
# Think-Act-Observe types
# =========================================================================


class WorkflowStep(Enum):
    """워크플로우 단계"""

    CRAWL = "crawl"
    STORE = "store"
    UPDATE_KG = "update_kg"
    CALCULATE = "calculate"
    INSIGHT = "insight"
    ALERT = "alert"
    EXPORT = "export"
    COMPLETE = "complete"


@dataclass
class ThinkResult:
    """Think 단계 결과"""

    next_action: str
    reasoning: str
    parameters: dict[str, Any] = field(default_factory=dict)
    should_continue: bool = True


@dataclass
class ActResult:
    """Act 단계 결과"""

    action: str
    success: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ObserveResult:
    """Observe 단계 결과"""

    observations: list[str]
    state_updates: dict[str, Any] = field(default_factory=dict)
    next_step: WorkflowStep | None = None


# =========================================================================
# BatchWorkflow
# =========================================================================


class BatchWorkflow:
    """
    배치 워크플로우 전용 오케스트레이터 (Think-Act-Observe 루프)

    워크플로우:
    1. Crawl → 2. Store → 3. Update KG → 4. Calculate → 5. Insight → 6. Alert → 7. Export

    참고:
    - 챗봇/질의 처리는 chat() 메서드가 UnifiedBrain으로 위임
    - 크롤링/지표/KG 진행 상황은 StateManager(단일 시스템 상태)에 기록
    """

    # 실패 시 워크플로우 전체를 실패로 간주하는 스텝 (데이터 수집/저장)
    CRITICAL_STEPS: tuple[str, ...] = (WorkflowStep.CRAWL.value, WorkflowStep.STORE.value)

    # 의존성 이름 -> 내부 슬롯
    _DEP_SLOTS: dict[str, str] = {
        "crawler": "_crawler",
        "storage": "_storage",
        "metrics": "_metrics_agent",
        "insight": "_hybrid_insight",
        "alert": "_alert_agent",
        "exporter": "_dashboard_exporter",
        "chatbot": "_hybrid_chatbot",
    }

    def __init__(
        self,
        config_path: str = "./config/thresholds.json",
        spreadsheet_id: str | None = None,
        model: str = "gpt-4.1-mini",
        use_hybrid: bool = True,
        kg_persist_path: str | None = "./data/knowledge_graph.json",
        deps: WorkflowDependencies | None = None,
        state_manager: StateManager | None = None,
        data_dir: str | None = None,
    ):
        """
        Args:
            config_path: 설정 파일 경로
            spreadsheet_id: Google Sheets ID
            model: LLM 모델
            use_hybrid: 하이브리드 에이전트 사용 여부
            kg_persist_path: Knowledge Graph 영속화 경로
            deps: 주입할 의존성 (None인 항목은 Container에서 지연 해석)
            state_manager: 시스템 상태 (None이면 싱글톤)
            data_dir: 대시보드/크롤링 스냅샷 출력 디렉토리 (기본: ./data, Railway: /data)
        """
        # 모니터링 컴포넌트 (설정 로드보다 먼저 초기화 — _load_config에서 사용)
        self.logger = AgentLogger("batch_workflow")

        # 설정 로드
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.spreadsheet_id = spreadsheet_id
        self.model = model
        self.use_hybrid = use_hybrid
        self.data_dir = data_dir or _default_data_dir()
        self.tracer = ExecutionTracer()
        self.metrics = QualityMetrics()

        # 메모리 컴포넌트
        self.session_manager = SessionManager()
        self.history_manager = HistoryManager()
        self.context_manager = ContextManager()

        # 시스템 상태 (F7)
        self._state_manager = state_manager

        # Ontology 컴포넌트
        deps = deps or WorkflowDependencies()
        self._deps = deps
        self._knowledge_graph: KnowledgeGraph | None = deps.knowledge_graph
        self._reasoner: OntologyReasoner | None = None
        self._kg_persist_path = kg_persist_path

        # 에이전트 (주입 또는 lazy 해석)
        self._crawler = deps.crawler
        self._storage = deps.storage
        self._metrics_agent = deps.metrics
        self._hybrid_insight = deps.insight
        self._alert_agent = deps.alert
        self._dashboard_exporter: DashboardExporter | None = deps.exporter
        self._hybrid_chatbot = deps.chatbot

        # 현재 상태
        self._current_step = WorkflowStep.CRAWL
        self._state: dict[str, Any] = {}
        self._session_id: str | None = None

    def _load_config(self, path: str) -> dict:
        """설정 로드"""
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config not found: {path}, using defaults")
            return {}

    # =========================================================================
    # 시스템 상태 (F7)
    # =========================================================================

    @property
    def state_manager(self) -> StateManager:
        """단일 시스템 상태 (StateManager)"""
        if self._state_manager is None:
            self._state_manager = get_state_manager()
        return self._state_manager

    # =========================================================================
    # Ontology 컴포넌트
    # =========================================================================

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        """Knowledge Graph (공유 인스턴스)"""
        if self._knowledge_graph is None:
            self._knowledge_graph = KnowledgeGraph(persist_path=self._kg_persist_path)
            self.logger.info(f"Knowledge Graph initialized: {self._knowledge_graph}")
        return self._knowledge_graph

    @property
    def reasoner(self) -> OntologyReasoner:
        """Ontology Reasoner (공유 인스턴스)"""
        if self._reasoner is None:
            self._reasoner = OntologyReasoner(self.knowledge_graph)
            register_all_rules(self._reasoner)
            self.logger.info(f"Reasoner initialized with {len(self._reasoner.rules)} rules")
        return self._reasoner

    # =========================================================================
    # 의존성 지연 해석
    # =========================================================================

    def _dependency_context(self, name: str) -> dict[str, Any]:
        """컴포넌트 생성에 필요한 컨텍스트 (컴포넌트별 최소한만 전달)"""
        context: dict[str, Any] = {
            "config_path": self.config_path,
            "spreadsheet_id": self.spreadsheet_id,
            "model": self.model,
            "tracer": self.tracer,
            "metrics": self.metrics,
        }
        if name in ("insight", "chatbot"):
            context["knowledge_graph"] = self.knowledge_graph
            context["reasoner"] = self.reasoner
        if name == "chatbot":
            context["context_manager"] = self.context_manager
        if name == "alert":
            context["state_manager"] = self.state_manager
        return context

    def _resolve(self, name: str) -> Any:
        """누락된 의존성을 Container에서 하나만 해석하여 캐시"""
        slot = self._DEP_SLOTS[name]
        current = getattr(self, slot)
        if current is None:
            resolved = WorkflowDependencies.from_container(
                only=(name,), **self._dependency_context(name)
            )
            current = getattr(resolved, name)
            setattr(self, slot, current)
            setattr(self._deps, name, current)
        return current

    @property
    def crawler(self):
        return self._resolve("crawler")

    @property
    def storage(self):
        return self._resolve("storage")

    @property
    def metrics_agent(self):
        return self._resolve("metrics")

    @property
    def hybrid_insight(self):
        """하이브리드 인사이트 에이전트"""
        return self._resolve("insight")

    @property
    def alert_agent(self):
        """알림 에이전트 (AlertAgent)"""
        return self._resolve("alert")

    @property
    def hybrid_chatbot(self):
        """하이브리드 챗봇 에이전트"""
        return self._resolve("chatbot")

    @property
    def dashboard_exporter(self) -> DashboardExporter:
        return self._resolve("exporter")

    # =========================================================================
    # 후처리 헬퍼
    # =========================================================================

    async def _verify_unknown_brands(self) -> dict[str, Any]:
        """
        Unknown 브랜드 배치 검증 (크롤링 후 처리)

        크롤링 결과에서 brand="Unknown"인 제품들을 추출하여
        Amazon 상세 페이지 또는 웹검색으로 브랜드 확정
        """
        try:
            from src.tools.utilities.brand_resolver import get_brand_resolver

            crawl_result = self._state.get("crawl_result", {})
            all_products = []

            categories = crawl_result.get("categories", {})
            for _cat_id, cat_data in categories.items():
                products = cat_data.get("products", [])
                all_products.extend(products)

            if not all_products:
                self.logger.info("No products to verify brands")
                return {"verified_count": 0, "skipped": True}

            resolver = get_brand_resolver()
            result = await resolver.verify_unknown_brands(
                products=all_products,
                use_amazon=True,
                use_websearch=True,
                delay_seconds=3.0,  # Amazon 차단 방지
            )

            self.logger.info(
                f"Brand verification: {result['verified_count']} verified, "
                f"{result['failed_count']} failed, {result['skipped_count']} skipped"
            )

            return result

        except ImportError:
            self.logger.warning("BrandResolver not available")
            return {"verified_count": 0, "error": "BrandResolver not available"}
        except Exception as e:
            self.logger.error(f"Brand verification failed: {e}")
            return {"verified_count": 0, "error": str(e)}

    async def _sync_sheets_to_sqlite(self) -> dict[str, Any]:
        """
        Google Sheets → SQLite 자동 동기화 (크롤링 후 처리)

        크롤링 완료 후 Google Sheets 데이터를 SQLite에 동기화하여
        데이터 정합성을 보장합니다.
        """
        try:
            from src.tools.storage.sheets_writer import SheetsWriter
            from src.tools.storage.sqlite_storage import SQLiteStorage

            self.logger.info("Starting Google Sheets → SQLite auto-sync")

            sheets = SheetsWriter(spreadsheet_id=self.spreadsheet_id)
            if not await sheets.initialize():
                self.logger.warning("Google Sheets connection failed for sync")
                return {"synced_count": 0, "error": "Sheets connection failed"}

            sqlite = SQLiteStorage()
            if not await sqlite.initialize():
                self.logger.warning("SQLite connection failed for sync")
                return {"synced_count": 0, "error": "SQLite connection failed"}

            # 최근 7일 데이터만 동기화 (효율성)
            records = await sheets.get_rank_history(days=7)

            if not records:
                self.logger.info("No records to sync from Sheets")
                return {"synced_count": 0, "message": "No records"}

            result = await sqlite.append_rank_records(records)

            synced_count = result.get("rows_added", 0)
            self.logger.info(
                f"Google Sheets → SQLite sync complete: "
                f"{synced_count} records synced from {len(records)} fetched"
            )

            return {"synced_count": synced_count, "fetched_count": len(records), "success": True}

        except ImportError as e:
            self.logger.warning(f"Import error during sync: {e}")
            return {"synced_count": 0, "error": str(e)}
        except Exception as e:
            self.logger.error(f"Sheets → SQLite sync failed: {e}", exc_info=True)
            return {"synced_count": 0, "error": str(e)}

    def _dump_crawl_snapshot(self, result: dict[str, Any]) -> list[str]:
        """크롤링 원본을 JSON으로 덤프 (단일 위치, CrawlManager에서 이동)

        - ``<data_dir>/latest_crawl_result.json``: 전체 payload (Excel export용)
        - ``<data_dir>/raw_products/<snapshot_date>.json``: 카테고리별 제품을 평탄화한
          히스토리 (원본 payload는 변경하지 않음)
        """
        observations: list[str] = []
        data_dir = Path(self.data_dir)
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            crawl_json_path = data_dir / "latest_crawl_result.json"
            with open(crawl_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
            self.logger.info(f"Crawl result saved to {crawl_json_path}")
            observations.append(f"크롤링 원본 저장: {crawl_json_path}")
        except Exception as e:
            self.logger.error(f"Failed to save crawl result JSON: {e}")

        try:
            raw_products_dir = data_dir / "raw_products"
            raw_products_dir.mkdir(parents=True, exist_ok=True)
            snapshot_date = result.get("snapshot_date") or datetime.now().strftime("%Y-%m-%d")
            history_path = raw_products_dir / f"{snapshot_date}.json"

            all_products = []
            for cat_id, cat_data in result.get("categories", {}).items():
                for product in cat_data.get("products", []):
                    all_products.append({**copy.deepcopy(product), "category_id": cat_id})

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(all_products, f, ensure_ascii=False, indent=2, default=_json_default)
            self.logger.info(
                f"Historical data saved to {history_path} ({len(all_products)} products)"
            )
        except Exception as e:
            self.logger.error(f"Failed to save crawl history JSON: {e}")

        return observations

    # =========================================================================
    # 워크플로우 실행
    # =========================================================================

    def _workflow_steps(self) -> list[str]:
        steps = [WorkflowStep.CRAWL.value, WorkflowStep.STORE.value]
        if self.use_hybrid:
            steps.append(WorkflowStep.UPDATE_KG.value)
        steps.extend(
            [
                WorkflowStep.CALCULATE.value,
                WorkflowStep.INSIGHT.value,
                WorkflowStep.ALERT.value,
                WorkflowStep.EXPORT.value,
            ]
        )
        return steps

    async def run_daily_workflow(
        self,
        categories: list[str] | None = None,
        crawl_only: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """
        일일 워크플로우 실행

        Args:
            categories: 크롤링할 카테고리 (None이면 전체)
            crawl_only: True면 크롤링(+스냅샷 덤프) 후 종료
            progress_callback: 스텝 완료마다 호출 ``(step_name, step_payload)``

        Returns:
            워크플로우 결과 (``status``, ``steps``, ``errors``, ``summary`` ...)
        """
        self._session_id = self.session_manager.create_session()
        self.tracer.start_trace(self._session_id)
        self.metrics.start_session()

        self.logger.info(
            f"Starting daily workflow - Session: {self._session_id}, Hybrid: {self.use_hybrid}"
        )

        self.context_manager.start_workflow(self._workflow_steps())

        self._current_step = WorkflowStep.CRAWL
        self._state = {"categories": categories}

        results: dict[str, Any] = {
            "session_id": self._session_id,
            "started_at": datetime.now().isoformat(),
            "steps": {},
            "status": "running",
            "errors": [],
            "hybrid_mode": self.use_hybrid,
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
                        "result": act_result.result,
                    }
                else:
                    self.session_manager.fail_agent(self._session_id, step_name, act_result.error)
                    results["steps"][step_name] = {"status": "failed", "error": act_result.error}

                # Observe
                observe_result = await self._observe(act_result)
                self._state.update(observe_result.state_updates)

                self._notify_progress(progress_callback, step_name, results["steps"][step_name])

                # 다음 스텝
                if crawl_only and step_name == WorkflowStep.CRAWL.value:
                    self._current_step = WorkflowStep.COMPLETE
                elif observe_result.next_step:
                    self._current_step = observe_result.next_step
                    self.context_manager.advance_workflow(act_result.result)
                else:
                    self._current_step = WorkflowStep.COMPLETE

                self.logger.workflow_step(step_name, "complete")

            # 완료 상태 판정 (D27/D12)
            final_status, status_error, errors = self._resolve_final_status(results["steps"])
            results["status"] = final_status
            results["errors"] = errors
            if status_error:
                results["error"] = status_error
            results["completed_at"] = datetime.now().isoformat()

            results["summary"] = self._generate_summary()

            # Knowledge Graph 저장 및 자동 백업
            if self.use_hybrid and self._knowledge_graph and not crawl_only:
                self._knowledge_graph.save()
                self.logger.info("Knowledge Graph saved")

                try:
                    from src.tools.utilities.kg_backup import get_kg_backup_service

                    backup_path = get_kg_backup_service().auto_backup()
                    if backup_path:
                        self.logger.info(f"KG auto-backup created: {backup_path}")
                except Exception as e:
                    self.logger.warning(f"KG auto-backup failed (non-critical): {e}")

        except Exception as e:
            self.logger.error(f"Workflow failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
            results["errors"] = [*results.get("errors", []), f"workflow: {e}"]

        finally:
            session_summary = self.session_manager.end_session(self._session_id)
            self.history_manager.add_execution(session_summary)

            results["trace"] = self.tracer.end_trace()
            results["metrics"] = self.metrics.end_session(results["status"])

            self.logger.info(f"Workflow completed - Status: {results['status']}")

        return results

    def _notify_progress(
        self, callback: ProgressCallback | None, step_name: str, payload: dict[str, Any]
    ) -> None:
        if callback is None:
            return
        try:
            callback(step_name, payload)
        except Exception as e:
            self.logger.warning(f"progress_callback failed for {step_name} (ignored): {e}")

    def _resolve_final_status(
        self, steps: dict[str, dict[str, Any]]
    ) -> tuple[str, str | None, list[str]]:
        """최종 워크플로우 상태 판정 (D27 + D12)

        - "failed": 크롤 payload가 status="failed"이거나, 크리티컬 스텝(crawl, store) 실패
        - "partial": 비크리티컬 스텝 실패, 크롤 payload가 status="partial",
                     또는 저장 결과에 errors 목록 존재
        - "completed": 실패 없음

        Returns:
            (status, error_message | None, errors)
        """
        failed_steps = [name for name, step in steps.items() if step.get("status") == "failed"]

        crawl_result = self._state.get("crawl_result") or {}
        crawl_status = crawl_result.get("status")
        crawl_step = WorkflowStep.CRAWL.value
        store_step = WorkflowStep.STORE.value

        critical: list[str] = []
        if crawl_status == "failed":
            critical.append(
                f"{crawl_step}: "
                + str(crawl_result.get("error") or "crawler returned status=failed")
            )
        for name in failed_steps:
            if name in self.CRITICAL_STEPS and not (
                crawl_status == "failed" and name == crawl_step
            ):
                critical.append(f"{name}: {steps[name].get('error') or 'failed'}")

        partial: list[str] = []
        if crawl_status == "partial":
            crawl_errors = [str(e) for e in (crawl_result.get("errors") or [])]
            if not crawl_errors:
                crawl_errors = ["crawler reported partial result"]
            partial.extend(f"{crawl_step}: {e}" for e in crawl_errors)

        storage_result = self._state.get("storage_result") or {}
        partial.extend(f"{store_step}: {e}" for e in (storage_result.get("errors") or []))

        partial.extend(
            f"{name}: {steps[name].get('error') or 'failed'}"
            for name in failed_steps
            if name not in self.CRITICAL_STEPS
        )

        errors = critical + partial
        if critical:
            return "failed", "critical step(s) failed: " + "; ".join(critical), errors
        if partial:
            return "partial", "partial result: " + "; ".join(partial), errors
        return "completed", None, errors

    # =========================================================================
    # Think / Act / Observe
    # =========================================================================

    async def _think(self) -> ThinkResult:
        """Think 단계: 다음 행동 결정"""
        step = self._current_step

        if step == WorkflowStep.CRAWL:
            return ThinkResult(
                next_action="crawl",
                reasoning="일일 크롤링 시작. Amazon 베스트셀러 Top 100 수집",
                parameters={"categories": self._state.get("categories")},
            )

        elif step == WorkflowStep.STORE:
            crawl_data = self._state.get("crawl_result")
            if not crawl_data or crawl_data.get("status") == "failed":
                return ThinkResult(
                    next_action="skip",
                    reasoning="크롤링 실패로 저장 단계 스킵",
                    should_continue=False,
                )
            return ThinkResult(
                next_action="store",
                reasoning="크롤링 데이터를 Google Sheets에 저장",
                parameters={"crawl_data": crawl_data},
            )

        elif step == WorkflowStep.UPDATE_KG:
            return ThinkResult(
                next_action="update_kg",
                reasoning="Knowledge Graph에 크롤링 데이터 반영 (엔티티 관계 구축)",
                parameters={"crawl_data": self._state.get("crawl_result")},
            )

        elif step == WorkflowStep.CALCULATE:
            crawl_data = self._state.get("crawl_result")
            if not crawl_data:
                return ThinkResult(
                    next_action="skip",
                    reasoning="크롤링 데이터 없음으로 지표 계산 스킵",
                    should_continue=False,
                )
            return ThinkResult(
                next_action="calculate",
                reasoning="10개 전략 지표 계산 (SoS, HHI, CPI 등)",
                parameters={
                    "crawl_data": crawl_data,
                    "historical_data": self._state.get("historical_data"),
                },
            )

        elif step == WorkflowStep.INSIGHT:
            metrics_data = self._state.get("metrics_result")
            if not metrics_data:
                return ThinkResult(
                    next_action="skip",
                    reasoning="지표 데이터 없음으로 인사이트 생성 스킵",
                    should_continue=False,
                )
            return ThinkResult(
                next_action="hybrid_insight",
                reasoning="하이브리드 인사이트 생성 (Ontology 추론 + RAG + LLM)",
                parameters={
                    "metrics_data": metrics_data,
                    "crawl_data": self._state.get("crawl_result"),
                    "crawl_summary": (self._state.get("crawl_result") or {}).get("summary"),
                },
            )

        elif step == WorkflowStep.ALERT:
            metrics_data = self._state.get("metrics_result")
            if not metrics_data:
                return ThinkResult(
                    next_action="skip",
                    reasoning="지표 데이터 없음으로 알림 단계 스킵",
                    should_continue=False,
                )
            return ThinkResult(
                next_action="alert",
                reasoning="지표 기반 순위 변동 알림 생성 및 발송",
                parameters={"metrics_data": metrics_data},
            )

        elif step == WorkflowStep.EXPORT:
            return ThinkResult(
                next_action="export", reasoning="Dashboard용 JSON 데이터 생성", parameters={}
            )

        return ThinkResult(
            next_action="complete", reasoning="모든 단계 완료", should_continue=False
        )

    async def _run_crawler(self, categories: list[str] | None) -> dict[str, Any]:
        """크롤러 실행 (스크래퍼 브라우저 초기화/종료 포함)"""
        crawler = self.crawler
        scraper = getattr(crawler, "scraper", None)
        initialize = getattr(scraper, "initialize", None)
        if callable(initialize):
            await initialize()
        try:
            return await crawler.execute(categories)
        finally:
            close = getattr(scraper, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception as e:
                    self.logger.warning(f"Scraper close failed (ignored): {e}")

    async def _act(self, think_result: ThinkResult) -> ActResult:
        """Act 단계: 행동 실행"""
        action = think_result.next_action
        params = think_result.parameters

        try:
            if action == "crawl":
                result = await self._run_crawler(params.get("categories"))
                return ActResult(action=action, success=True, result=result)

            elif action == "store":
                result = await self.storage.execute(params.get("crawl_data"))
                return ActResult(action=action, success=True, result=result)

            elif action == "update_kg":
                crawl_data = params.get("crawl_data") or {}
                # F9: OntologyBuilder writes JSON KG + OWL A-Box from the same normalised
                # entities; materialize() runs the OWL reasoning offline and stores the
                # inferred facts (with provenance) in the KG for the chat path to read.
                from src.ontology.builder import OntologyBuilder
                from src.ontology.materializer import materialize

                result: dict[str, Any] = {"relations_added": 0, "inferred_added": 0}
                ontology_error: str | None = None
                try:
                    build = OntologyBuilder().from_snapshot(
                        crawl_data, metrics=None, kg=self.knowledge_graph
                    )
                    result["relations_added"] = build.stats["relations_added"]
                    result["owl_individuals"] = build.stats["owl_individuals"]
                    if build.owl is not None:
                        result["inferred_added"] = materialize(build.owl, self.knowledge_graph)
                except Exception as e:
                    # Recorded as a step error (update_kg is non-critical, so the run ends
                    # "partial"), never swallowed into a "completed" day.
                    ontology_error = f"{type(e).__name__}: {e}"
                    result["ontology_error"] = ontology_error
                    self.logger.error(f"Ontology build/materialize failed: {e}")
                    # keep the JSON KG usable even when the ontology step fails
                    if not result["relations_added"]:
                        result["relations_added"] = self.knowledge_graph.load_from_crawl_data(
                            crawl_data
                        )

                kg_stats = self.knowledge_graph.get_stats()
                result.update(
                    {
                        "total_triples": kg_stats.get("total_triples", 0),
                        "unique_subjects": kg_stats.get("unique_subjects", 0),
                        "unique_objects": kg_stats.get("unique_objects", 0),
                    }
                )
                return ActResult(
                    action=action,
                    success=ontology_error is None,
                    result=result,
                    error=ontology_error,
                )

            elif action == "calculate":
                result = await self.metrics_agent.execute(
                    params.get("crawl_data"), params.get("historical_data")
                )
                return ActResult(action=action, success=True, result=result)

            elif action == "hybrid_insight":
                result = await self.hybrid_insight.execute(
                    metrics_data=params.get("metrics_data"),
                    crawl_data=params.get("crawl_data"),
                    crawl_summary=params.get("crawl_summary"),
                )
                return ActResult(action=action, success=True, result=result)

            elif action == "alert":
                alerts = await self.alert_agent.process_metrics(params.get("metrics_data") or {})
                send_result = await self.alert_agent.send_pending_alerts() or {}
                # D6: AlertAgent는 "sent"/"failed" 키를 반환 (레거시 "*_count" 도 허용)
                result = {
                    "alerts_created": len(alerts or []),
                    "alerts_sent": send_result.get("sent", send_result.get("sent_count", 0)),
                    "alerts_failed": send_result.get("failed", send_result.get("failed_count", 0)),
                    "alerts_skipped": send_result.get("skipped", 0),
                }
                return ActResult(action=action, success=True, result=result)

            elif action == "export":
                export_path = f"{self.data_dir}/dashboard_data.json"
                await self.dashboard_exporter.initialize()
                result = await self.dashboard_exporter.export_dashboard_data(export_path)

                # Unknown 브랜드 배치 검증 (크롤링 후 처리)
                brand_verify_result = await self._verify_unknown_brands()

                # Google Sheets → SQLite 자동 동기화
                sync_result = await self._sync_sheets_to_sqlite()

                metadata = (result or {}).get("metadata", {})
                return ActResult(
                    action=action,
                    success=True,
                    result={
                        "exported": True,
                        "path": export_path,
                        "products": metadata.get("total_products", 0),
                        "laneige_count": metadata.get("laneige_products", 0),
                        "brands_verified": brand_verify_result.get("verified_count", 0),
                        "sqlite_synced": sync_result.get("synced_count", 0),
                    },
                )

            elif action == "skip":
                return ActResult(action=action, success=True, result={"skipped": True})

            else:
                return ActResult(action=action, success=False, error=f"Unknown action: {action}")

        except Exception as e:
            self.logger.error(f"Action {action} failed: {e}")
            return ActResult(action=action, success=False, error=str(e))

    async def _observe(self, act_result: ActResult) -> ObserveResult:
        """Observe 단계: 결과 관찰 및 상태 업데이트"""
        observations: list[str] = []
        state_updates: dict[str, Any] = {}
        next_step: WorkflowStep | None = None

        if act_result.action == "crawl":
            result = act_result.result

            # LLM 기반 Unknown 브랜드 검증 (크롤링 직후)
            try:
                all_products = []
                for cat_data in result.get("categories", {}).values():
                    all_products.extend(cat_data.get("products", []))

                unknown_count = sum(
                    1
                    for p in all_products
                    if not p.get("brand") or p.get("brand") in ("Unknown", "")
                )

                if unknown_count > 0:
                    self.logger.info(f"LLM 브랜드 검증 시작: {unknown_count}개 Unknown 브랜드 발견")

                    from src.tools.utilities.brand_resolver import get_brand_resolver

                    resolver = get_brand_resolver()
                    verify_result = await resolver.verify_brands_with_llm(
                        products=all_products, max_concurrent=5, delay_seconds=0.3
                    )

                    observations.append(
                        f"브랜드 검증: {verify_result.get('verified_count', 0)}개 확인, "
                        f"{verify_result.get('failed_count', 0)}개 실패"
                    )

                    updated_products = verify_result.get("updated_products", [])
                    if updated_products:
                        updated_by_asin = {p["asin"]: p for p in updated_products if p.get("asin")}

                        for cat_data in result.get("categories", {}).values():
                            for i, prod in enumerate(cat_data.get("products", [])):
                                asin = prod.get("asin")
                                if asin and asin in updated_by_asin:
                                    cat_data["products"][i]["brand"] = updated_by_asin[asin].get(
                                        "brand", prod.get("brand")
                                    )

            except Exception as e:
                self.logger.warning(f"LLM 브랜드 검증 실패 (크롤링 계속): {e}")

            observations.append(
                f"크롤링 완료: {result.get('total_products', 0)} 제품, "
                f"LANEIGE {result.get('laneige_count', 0)}개"
            )
            state_updates["crawl_result"] = result
            next_step = WorkflowStep.STORE

            self.context_manager.update_crawl_data(
                categories=list(result.get("categories", {}).keys()),
                products_count=result.get("total_products", 0),
                laneige_products=result.get("laneige_products", []),
            )

            if act_result.success and result.get("status") != "failed":
                # F7: 단일 시스템 상태에 크롤링 완료 기록
                self.state_manager.mark_crawled(
                    success=True, products_count=result.get("total_products", 0)
                )
                # 크롤링 스냅샷 덤프 (단일 위치)
                observations.extend(self._dump_crawl_snapshot(result))

        elif act_result.action == "store":
            result = act_result.result
            observations.append(
                f"저장 완료: {result.get('raw_records', 0)} 레코드, "
                f"{result.get('products_upserted', 0)} 제품"
            )
            state_updates["storage_result"] = result

            next_step = WorkflowStep.UPDATE_KG if self.use_hybrid else WorkflowStep.CALCULATE

        elif act_result.action == "update_kg":
            result = act_result.result
            observations.append(
                f"Knowledge Graph 업데이트: {result.get('relations_added', 0)} 관계 추가, "
                f"총 {result.get('total_triples', 0)} 트리플"
            )
            state_updates["kg_result"] = result
            next_step = WorkflowStep.CALCULATE

            if act_result.success:
                total = int(result.get("total_triples", 0) or 0)
                if self.state_manager.kg_initialized:
                    self.state_manager.update_kg_stats(total)
                else:
                    self.state_manager.mark_kg_initialized(total)

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
            if act_result.success:
                self.state_manager.mark_metrics_calculated()

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
            next_step = WorkflowStep.ALERT

            self.context_manager.set_insights_generated(True)

            # 챗봇 데이터 컨텍스트 설정
            if self.use_hybrid:
                self.hybrid_chatbot.set_data_context(self._state.get("metrics_result", {}))

        elif act_result.action == "alert":
            result = act_result.result
            observations.append(
                f"알림 처리 완료: 생성 {result.get('alerts_created', 0)}건, "
                f"발송 {result.get('alerts_sent', 0)}건"
            )
            state_updates["alert_result"] = result
            next_step = WorkflowStep.EXPORT

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

        if not act_result.success:
            observations.append(f"에러 발생: {act_result.error}")
            self.context_manager.record_workflow_error(
                act_result.action, act_result.error or "Unknown error"
            )

        for obs in observations:
            self.logger.info(f"Observe: {obs}")

        return ObserveResult(
            observations=observations, state_updates=state_updates, next_step=next_step
        )

    def _generate_summary(self) -> dict[str, Any]:
        """최종 요약 생성"""
        crawl = self._state.get("crawl_result", {})
        metrics = self._state.get("metrics_result", {})
        insight = self._state.get("insight_result", {})
        alert = self._state.get("alert_result", {})
        export = self._state.get("export_result", {})
        kg = self._state.get("kg_result", {})

        summary = {
            "products_crawled": crawl.get("total_products", 0),
            "laneige_tracked": crawl.get("laneige_count", 0),
            "categories": list(crawl.get("categories", {}).keys()),
            "alerts": len(metrics.get("alerts", [])),
            "alerts_sent": alert.get("alerts_sent", 0),
            "action_items": len(insight.get("action_items", [])),
            "daily_insight": insight.get("daily_insight", "")[:200] + "...",
            "dashboard_exported": export.get("exported", False),
            "dashboard_path": export.get("path", ""),
        }

        if self.use_hybrid:
            summary["hybrid"] = {
                "kg_triples": kg.get("total_triples", 0),
                "inferences": len(insight.get("inferences", [])),
                "explanations": len(insight.get("explanations", [])),
            }

        return summary

    # =========================================================================
    # 챗봇 인터페이스 (UnifiedBrain으로 위임)
    # =========================================================================

    async def chat(self, message: str) -> dict[str, Any]:
        """챗봇 질의 - UnifiedBrain으로 위임"""
        brain = get_brain()
        response = await brain.process_query(
            query=message,
            session_id=self._session_id,
            current_metrics=self._state.get("metrics_result"),
        )
        return response.to_dict() if hasattr(response, "to_dict") else response

    async def process_query(self, query: str) -> dict[str, Any]:
        """질의 처리 - UnifiedBrain으로 위임"""
        brain = get_brain()
        response = await brain.process_query(
            query=query,
            session_id=self._session_id,
            current_metrics=self._state.get("metrics_result"),
        )
        return response.to_dict() if hasattr(response, "to_dict") else response

    # =========================================================================
    # 유틸리티
    # =========================================================================

    async def cleanup(self) -> None:
        """리소스 정리"""
        close = getattr(self._crawler, "close", None) if self._crawler else None
        if callable(close):
            await close()

        if self._knowledge_graph:
            self._knowledge_graph.save()

        self.metrics.save()
        self.logger.info("Batch workflow cleanup completed")

    def get_status(self) -> dict[str, Any]:
        """현재 상태 조회"""
        status = {
            "session_id": self._session_id,
            "current_step": self._current_step.value if self._current_step else None,
            "workflow": self.context_manager.get_workflow_status(),
            "data": self.context_manager.get_data_status(),
            "hybrid_mode": self.use_hybrid,
        }

        if self.use_hybrid and self._knowledge_graph:
            status["knowledge_graph"] = self._knowledge_graph.get_stats()

        if self.use_hybrid and self._reasoner:
            status["reasoner"] = self._reasoner.get_inference_stats()

        return status

    def get_history_stats(self) -> dict[str, Any]:
        """히스토리 통계 조회"""
        return {
            "success_rate": self.history_manager.get_success_rate(),
            "avg_duration": self.history_manager.get_average_duration(),
            "recent_errors": self.history_manager.get_error_summary(days=7),
        }

    def get_knowledge_graph_stats(self) -> dict[str, Any]:
        """Knowledge Graph 통계 조회"""
        if not self._knowledge_graph:
            return {"initialized": False}

        stats = self._knowledge_graph.get_stats()
        stats["initialized"] = True
        stats["most_connected"] = self._knowledge_graph.get_most_connected(5)
        return stats

    def get_inference_stats(self) -> dict[str, Any]:
        """추론 통계 조회"""
        if not self._reasoner:
            return {"initialized": False}
        return self._reasoner.get_inference_stats()


# =========================================================================
# Backward Compatibility Aliases
# =========================================================================

Orchestrator = BatchWorkflow

CrawlResult = dict[str, Any]
MetricsResult = dict[str, Any]
InsightResult = dict[str, Any]
WorkflowState = dict[str, Any]


# =========================================================================
# Convenience Functions
# =========================================================================


async def run_full_workflow(
    categories: list[str] | None = None,
    config_path: str = "./config/thresholds.json",
    spreadsheet_id: str | None = None,
    use_hybrid: bool = True,
) -> dict[str, Any]:
    """전체 배치 워크플로우 실행 (편의 함수)"""
    workflow = BatchWorkflow(
        config_path=config_path, spreadsheet_id=spreadsheet_id, use_hybrid=use_hybrid
    )

    try:
        return await workflow.run_daily_workflow(categories=categories)
    finally:
        await workflow.cleanup()
