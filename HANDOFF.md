# HANDOFF.md — Phase 4-6 Worktree

> **워크트리**: `amore-phase4-arch`
> **브랜치**: `refactor/phase4-architecture`
> **범위**: Phase 4 (Batch Workflow 통합) + Phase 5 (DI Container) + Phase 6 (Config + 테스트)
> **마지막 업데이트**: 2026-02-16

---

## 1. 완료한 작업

### Phase 4: Batch Workflow 통합 (완료, PR #10)
- [x] `src/core/batch_workflow.py` → `src/application/workflows/batch_workflow.py` 이전 (1,059줄)
- [x] Import 경로 업데이트 (orchestrator.py, evaluate_golden.py, __init__.py)
- [x] `Container.get_batch_workflow()` 팩토리 메서드 추가
- [x] 검증 통과: ruff 0 err, pytest 453/454 (99.8%)

### Phase 5: DI Container 완성 (완료)
- [x] Container에 7개 컴포넌트 추가 등록 (11 → 18 메서드)
  - 팩토리: AlertAgent, MetricsAgent, StorageAgent
  - 싱글톤: SuggestionEngine, SourceProvider, ExternalSignalManager, MarketIntelligenceEngine
- [x] 주요 소비자 직접 import → Container DI 전환
  - `batch_workflow.py`: StorageAgent, MetricsAgent
  - `hybrid_chatbot_agent.py`: SuggestionEngine, SourceProvider, ExternalSignalManager
  - `crawl_manager.py`: CrawlerAgent, StorageAgent
- [x] 검증 통과: ruff 0 err, pytest 453/454 (99.8%)

---

## 2. Phase 4 분석 결과

### 두 버전 비교

| 항목 | `src/core/batch_workflow.py` (1,059줄) | `src/application/workflows/batch_workflow.py` (119줄) |
|------|----------------------------------------|------------------------------------------------------|
| 패턴 | Think-Act-Observe 루프 | 단순 순차 실행 |
| DI | 없음 (구체 클래스 직접 import) | Protocol 기반 DI (`WorkflowDependencies`) |
| 에이전트 | Lazy init (6개: crawler, storage, metrics, hybrid_insight, hybrid_chatbot, dashboard_exporter) | DI 주입 (4개: crawler, storage, metrics, insight) |
| KG 통합 | 있음 (KnowledgeGraph, OntologyReasoner, business_rules) | 없음 |
| 워크플로우 단계 | 6단계 (Crawl→Store→UpdateKG→Calculate→Insight→Export) | 4단계 (Crawl→Save→Calculate→Insight) |
| 브랜드 검증 | 있음 (`_verify_unknown_brands`, LLM 브랜드 검증) | 없음 |
| Sheets 동기화 | 있음 (`_sync_sheets_to_sqlite`) | 없음 |
| 챗봇 인터페이스 | 있음 (`chat()`, `process_query()` → UnifiedBrain 위임) | 없음 |
| 모니터링 | 있음 (AgentLogger, ExecutionTracer, QualityMetrics) | 없음 |
| 메모리 | 있음 (SessionManager, HistoryManager, ContextManager) | 없음 |
| Backward Compat | `Orchestrator = BatchWorkflow` 별칭, type aliases | 없음 |
| 배포 사용 | **실제 사용됨** | **미사용** |

### Import 의존관계

| 소비자 | 현재 import 경로 |
|--------|-----------------|
| `orchestrator.py` | `from src.core.batch_workflow import ...` |
| `main.py` | `from orchestrator import Orchestrator` |
| `scripts/evaluate_golden.py` | `from src.core.batch_workflow import BatchWorkflow` |
| `tests/run_hybrid_integration.py` | `from orchestrator import Orchestrator, WorkflowStep` |
| `src/application/__init__.py` | `from src.application.workflows.batch_workflow import BatchWorkflow` (이름 충돌!) |

### 통합 전략

**core/ 로직을 application/으로 이전 + DI 패턴 적용**:
1. application/ 버전에 core/의 전체 비즈니스 로직 이전
2. 구체 클래스 직접 import → Protocol 기반 DI 전환 (가능한 범위)
3. `orchestrator.py` import 경로를 `src.application.workflows.batch_workflow`로 변경
4. `src/core/batch_workflow.py` 삭제
5. Container에 BatchWorkflow 팩토리 등록

---

## 3. 시도했지만 실패한 접근법

- (아직 구현 단계 아님)

---

## 4. 수정한 파일 목록

| 파일 | 변경 유형 | 요약 |
|------|-----------|------|
| `src/infrastructure/container.py` | 수정 | 7개 컴포넌트 추가 (11→18 메서드) |
| `src/application/workflows/batch_workflow.py` | 수정 | Phase4: core/ 이전 + Phase5: Container DI 전환 |
| `src/agents/hybrid_chatbot_agent.py` | 수정 | 직접 import → Container DI 전환 |
| `src/core/crawl_manager.py` | 수정 | CrawlerAgent/StorageAgent → Container DI 전환 |
| `src/core/batch_workflow.py` | 삭제 | application/으로 이전 완료 |
| `orchestrator.py` | 수정 | import 경로 변경 (core → application) |
| `src/application/__init__.py` | 수정 | 새 export 항목 추가 |
| `scripts/evaluate_golden.py` | 수정 | import 경로 변경 |
| `HANDOFF.md` | 수정 | Phase 5 진행 상황 반영 |
| `TODO.md` | 수정 | Phase 5 체크리스트 완료 |

---

## 5. 참조 문서

- **전체 계획**: `~/.claude/plans/snoopy-floating-pie.md`
- **성공 기준**: ruff 0 errors + pytest 통과율 95%+ + 커버리지 60%+
- **주의**: `dashboard_api.py`는 Phase 3에서 작업 중이므로 건드리지 않음
