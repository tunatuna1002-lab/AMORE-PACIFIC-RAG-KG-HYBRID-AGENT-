# TODO.md — Phase 4-6 체크리스트

> **워크트리**: `amore-phase4-arch`
> **브랜치**: `refactor/phase4-architecture`

---

## Phase 4: Batch Workflow 통합 (P1, 2일, HIGH risk) — DONE

### 4-1. 준비 (분석 + 테스트 기준선)
- [x] 두 버전 기능 차이 분석
- [x] 기준선 테스트 실행 (453 passed, 1 failed pre-existing)
- [x] 기준선 ruff 확인 (All checks passed)

### 4-2. application/ 버전에 core/ 로직 이전
- [x] `WorkflowStep` enum 이전
- [x] `ThinkResult`, `ActResult`, `ObserveResult` dataclass 이전
- [x] `BatchWorkflow` 클래스 전체 이전 (Think-Act-Observe 루프)
- [x] Backward compatibility aliases 이전 (`Orchestrator = BatchWorkflow`, type aliases)
- [x] `run_full_workflow` 편의 함수 이전
- [x] 기존 application/ `WorkflowResult`, `WorkflowDependencies`, `WorkflowStatus` 보존

### 4-3. Import 경로 업데이트
- [x] `orchestrator.py` → `from src.application.workflows.batch_workflow import ...`
- [x] `scripts/evaluate_golden.py` → import 경로 변경
- [x] `src/application/__init__.py` → export 목록 업데이트
- [x] 기타 import 잔류 확인 (src/ 내 잔류 없음, docs/notes에만 잔존)

### 4-4. core/ 버전 삭제
- [x] `src/core/batch_workflow.py` 삭제

### 4-5. DI Container에 BatchWorkflow 등록
- [x] `src/infrastructure/container.py`에 `get_batch_workflow()` 팩토리 추가

### 4-6. 검증
- [x] `ruff check src/` → 0 errors
- [x] `python3 -m pytest tests/ -x --tb=short` → 99.8% (453/454, pre-existing 1 fail)
- [x] `from orchestrator import Orchestrator` → import 성공 (하위 호환)
- [x] `from src.application.workflows.batch_workflow import BatchWorkflow` → import 성공
- [x] Container `get_batch_workflow()` → 정상 동작

### 4-7. 커밋
- [x] Phase 4 커밋 생성 + PR #10

---

## Phase 5: DI Container 완성 + Clean Architecture 정리 (P2, 2일, MEDIUM risk)

### 5-1. 누락 컴포넌트 Container 등록
- [x] AlertAgent (`get_alert_agent`)
- [x] MetricsAgent (`get_metrics_agent`)
- [x] StorageAgent (`get_storage_agent`)
- [x] MarketIntelligenceEngine (`get_market_intelligence_engine`)
- [x] ExternalSignalManager (`get_external_signal_manager`)
- [x] SuggestionEngine (`get_suggestion_engine`)
- [x] SourceProvider (`get_source_provider`)
- [x] BatchWorkflow (Phase 4에서 추가 완료)

### 5-2. 주요 소비자 직접 import → Container DI 전환
- [x] `batch_workflow.py`: StorageAgent, MetricsAgent → Container
- [x] `hybrid_chatbot_agent.py`: SuggestionEngine, SourceProvider, ExternalSignalManager → Container
- [x] `crawl_manager.py`: CrawlerAgent, StorageAgent → Container

### 5-3. 검증
- [x] `ruff check src/` → 0 errors
- [x] `python3 -m pytest tests/ -x --tb=short` → 99.8% (453/454, pre-existing 1 fail)
- [x] Container 등록 컴포넌트 수 확인 (11개 → 18개 get_ 메서드)

---

## Phase 6: Config 정리 + 테스트 보강 (P2, 2일, LOW risk)

### 6-1. Config 통합
- [ ] `competitors.json` + `tracked_competitors.json` 병합
- [ ] `thresholds.json` → system settings / category URLs 분리
- [ ] Pydantic 스키마 기반 검증 추가

### 6-2. 테스트 추가
- [ ] `src/agents/metrics_agent.py` 테스트
- [ ] `src/agents/storage_agent.py` 테스트
- [ ] `src/agents/period_insight_agent.py` 테스트
- [ ] `src/core/query_processor.py` 테스트
- [ ] `src/tools/scrapers/deals_scraper.py` 테스트

### 6-3. 검증
- [ ] `ruff check src/` → 0 errors
- [ ] `python3 -m pytest tests/ --cov=src --cov-report=term-missing` → 커버리지 60%+

---

## 주의사항

- **dashboard_api.py 수정 금지** — Phase 3에서 별도 작업 중
- **롤백 기준**: 검증 실패시 `git revert` + 원인 분석
- **게이트**: Phase 4 PR main 머지 → Phase 5 시작 가능
