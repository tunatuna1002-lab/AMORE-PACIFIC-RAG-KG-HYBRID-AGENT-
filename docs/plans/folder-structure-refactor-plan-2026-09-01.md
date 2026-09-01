# 폴더 구조 리팩토링 계획 (2026-09-01)

> 기준 커밋: a46ce76 (main) / 분석 방법: 정적 import 그래프(AST) + 파일·라인 집계 + grep 실사용 확인
> 결론: 폴더 골격은 이미 정돈되어 있음(9세션 리팩토링 완료). 남은 문제는 **구조가 아니라 "선언과 실체의 불일치"** — 빈 레이어, 호환 shim, 중복 정의, 역방향 의존, 문서 드리프트.

## 목차

1. 현황 한눈에 보기
2. 기능별 문제와 조치
   - A. 아키텍처 레이어(Clean Architecture 정합성)
   - B. 오케스트레이션 코어(src/core)
   - C. 에이전트(src/agents)
   - D. RAG(src/rag)
   - E. 온톨로지(src/ontology)
   - F. 도구(src/tools)
   - G. API(src/api)
   - H. 메모리·모니터링·shared
   - I. 테스트(tests/)
   - J. 설정·빌드·문서·루트
3. 실행 순서(Phase 0~5)
4. 검증 방법
5. 하지 않을 것

---

## 1. 현황 한눈에 보기

| 항목 | 실측 | 비고 |
|------|------|------|
| src/ Python 파일 | 210개 / 74,681줄 | CLAUDE.md는 214개로 기재 |
| tests/ 파일 | 184개 (test_*.py 161개) | 루트 잔재 9개 포함 |
| 순환 의존(SCC) | 1개, 5모듈 | chatbot ↔ batch_workflow ↔ brain ↔ crawl_manager ↔ container |
| 레이어 역방향 import | 2건 | tools → api |
| 호환 shim 파일 | 8개 | adapters 2, ontology 3, core 1, application 1, tools 1 |
| 프로덕션 미참조 모듈 | 8개 | types.py, query_processor.py, llm_retry.py, session_crypto.py, brain_components.py 등 |
| 중복 클래스명 | 17쌍 | QueryIntent 3중, BrainMode/TaskPriority 2중 등 |
| 테스트 없는 src 모듈 | 39개 | api/routes 6, ontology/rules 6, notifications 2 등 |
| ruff 경고 | 16건 (UP042) | 전부 str-Enum |

### 패키지 간 의존 방향(정적 그래프 요약)

```
api ──25──> tools ──2──> api        ← 역방향(export_handlers)
application ──> agents/core/tools/infrastructure   ← Layer2가 Layer3·4를 봄
infrastructure ──> agents/rag/tools/core            ← DI 컨테이너, 정상
adapters ──> agents/rag (re-export만, 사용처 0)      ← 껍데기
```

---

## 2. 기능별 문제와 조치

### A. 아키텍처 레이어(Clean Architecture 정합성)

| # | 문제 | 근거 | 조치 | 위험 |
|---|------|------|------|------|
| A1 | `src/adapters/` 전체가 re-export 껍데기 | 4파일 75줄, 전부 `try: from src.agents ...`; src·tests·scripts 어디서도 import 없음 | 패키지 삭제. `src/adapters/AGENTS.md`도 삭제. CLAUDE.md §8 레이어 표에서 adapters 행 제거 또는 "미구현" 명시 | 없음 |
| A2 | `src/application/orchestrators/` 빈 패키지 | 2줄 주석만 | 삭제 | 없음 |
| A3 | `application/workflows/batch_workflow.py`가 Layer 2인데 core/agents/tools/infrastructure/memory/monitoring를 직접 import | fan-out 20, 순환 SCC의 중심. `_default_*` 팩토리가 지연 import로 구체 클래스 생성 | `WorkflowDependencies` 기본 생성 로직을 `infrastructure/container.py`(또는 `infrastructure/bootstrap.py`)로 이관. workflow는 Protocol만 참조 | 중 |
| A4 | `tools/exporters/export_handlers.py` → `api.dependencies.load_dashboard_data`, `api.routes.export._get_external_signals` 역참조 | 63행, 162행 (private 함수까지 참조) | 두 함수를 `application/services/export_service.py`로 내리고 api·tools 양쪽이 그것을 참조 | 낮 |
| A5 | `api/dependencies.py` 605줄에 인증·세션·감사로그·데이터로딩·제안생성·JWT·DI 게터 혼재 (fan-in 15) | 21개 top-level 함수 | `api/deps/{auth,session,audit,data,suggestions,providers}.py`로 분할, `api/dependencies.py`는 re-export만 남겨 1릴리스 유지 | 낮 |
| A6 | 5모듈 순환 SCC | hybrid_chatbot_agent → container(지연) → brain → batch_workflow → container… | A3 완료 후 `crawl_manager`가 `batch_workflow`를 지연 import하는 지점을 Protocol 주입으로 교체. 목표: SCC 0 | 중 |

### B. 오케스트레이션 코어(src/core, 30모듈 10.6k줄)

| # | 문제 | 근거 | 조치 | 위험 |
|---|------|------|------|------|
| B1 | `core/types.py` 미사용 + `brain.py`·`tool_coordinator.py`와 `BrainMode/BrainTask/TaskPriority/ErrorStrategy` 중복 정의 | fan-in 0 | brain·tool_coordinator가 `types.py`를 import하도록 단일화(권장) 또는 types.py 삭제 | 낮 |
| B2 | `query_processor.py` 프로덕션 미사용 | `QueryProcessor` 참조 = 테스트뿐. brain.py가 위임하지 않음 | brain에서 실제 위임하거나 삭제. 삭제 시 `tests/unit/core/test_query_processor.py` 동반 삭제 | 낮 |
| B3 | `llm_orchestrator.py` 레거시 오케스트레이터 잔존 | `LLMOrchestrator` 참조 = `core/__init__.py` re-export뿐. UnifiedBrain과 역할 중복 | `core/__init__` re-export 제거 → 테스트 이관 → 삭제. `confidence.py`/`tools.py` 주석 갱신 | 중 |
| B4 | `explainability.py` src 내 미사용 | eval/judge/llm.py에서만 사용 | 유지하되 eval 전용임을 docstring에 명시, 또는 `eval/`로 이동 | 낮 |
| B5 | `orchestrator.py` 호환 래퍼 | main.py, tests/run_hybrid_integration.py 2곳 | 두 호출부를 `src.application.workflows.batch_workflow`로 교체 후 삭제 | 낮 |
| B6 | `brain.py` 1,713줄 50메서드 god class | decision_maker/tool_coordinator/context_gatherer는 분리됐으나 query_processor는 미연결 | Phase 4: (1) 스케줄링(BrainTask 큐) → `core/brain_scheduler.py`, (2) 질의 파이프라인 → `query_processor.py` 실제 위임 | 높 |
| B7 | `core/__init__.py`가 9모듈 eager import + brain `__getattr__` 지연 로딩 hack | `import src.core.cache`만 해도 llm_orchestrator·scheduler 로드 | `__all__`은 유지하고 전부 `__getattr__` 지연 로딩으로 전환 | 낮 |
| B8 | 상태 모듈 3개 이름 혼동 (`state.py`, `state_manager.py`, `graph_state.py`) | 역할은 별개(오케스트레이터 상태 / 시스템·구독 상태 / 질의 그래프 상태) | 이름만 정리: `state.py` → `orchestrator_state.py`. 나머지 유지 | 낮 |

### C. 에이전트(src/agents)

| # | 문제 | 근거 | 조치 | 위험 |
|---|------|------|------|------|
| C1 | `agents/__init__.py` eager import → `import src.agents` 시 crawler → amazon_scraper → **playwright** 로드 | scrapers는 top-level에서 playwright import | `__getattr__` 지연 로딩 | 낮 |
| C2 | 직접 import 잔존(FUTURE_WORK §3) | hybrid_insight: ExternalSignalCollector·MarketIntelligenceEngine / period_insight: PeriodAnalyzer·InsightFormatter | container 경유 주입으로 전환 (A3와 같은 PR) | 중 |
| C3 | 구조 자체는 양호 | base_hybrid_agent 템플릿 존재, hybrid_insight ↔ period_insight 메서드 중복 없음 | 변경 없음 | - |

### D. RAG(src/rag, 16모듈 9.7k줄)

| # | 문제 | 근거 | 조치 | 위험 |
|---|------|------|------|------|
| D1 | `QueryIntent` 3중 정의 | `rag/hybrid_retriever.py`, `application/services/query_analyzer.py`, `core/intent.py`("single source of truth" 선언) | `core/intent.py`를 유일 정의로 두고 나머지 2곳은 alias import. 값 집합 차이는 매핑 테이블로 | 중 |
| D2 | `LinkedEntity` 중복 (`entity_linker` vs `confidence_fusion`), `InferenceResult` 중복 (`confidence_fusion` vs `domain/entities/relations`) | 동명 dataclass | confidence_fusion 쪽을 `FusedEntity`/`FusionInferenceResult`로 개명 또는 domain 것 재사용 | 낮 |
| D3 | `hybrid_retriever.py` 1,773줄 (4클래스) | `QueryIntent`·`HybridContext`·`EntityExtractor`·`HybridRetriever` 동거 | `rag/hybrid/{context.py, entity_extractor.py, retriever.py}` 분할, `hybrid_retriever.py`는 re-export | 중 |
| D4 | `retriever.py` 1,478줄 33메서드 단일 클래스 | 문서 로딩·인덱싱·검색·캐시 혼재 | `DocumentLoader`(로딩/청킹) / `IndexManager`(ChromaDB) / `DocumentRetriever`(검색) 3분할 | 높 |
| D5 | `rag/__init__.py` 12모듈 eager import | chromadb·sentence-transformers 간접 로드 | 지연 로딩 | 낮 |
| D6 | `query_enhancer.py` 테스트 없음 | - | Phase 5 | - |

### E. 온톨로지(src/ontology)

| # | 문제 | 근거 | 조치 | 위험 |
|---|------|------|------|------|
| E1 | `ontology/relations.py` DEPRECATED shim이 여전히 주 경로 | src 내 import edge 14건(절대경로 5 + `__init__` 재수출) | 호출부를 `src.domain.entities.relations`로 일괄 교체 → shim 삭제 | 낮 |
| E2 | `business_rules.py` 호환 레이어, `rules/*`는 이 파일을 통해서만 접근 | rules/ 6모듈의 유일 importer | `rules/__init__.py`가 `get_all_rules()` 등을 제공하고 `business_rules.py` 삭제 | 낮 |
| E3 | `schema.py` 호환 표시 | react_agent, ontology_knowledge_graph, scripts 3곳 사용 | 실사용 중이므로 "호환" 문구 제거하고 정식 모듈로 승격 | 없음 |
| E4 | `rules/*` 6모듈 테스트 없음 | - | Phase 5. 규칙당 최소 1 케이스 | - |
| E5 | `owl_reasoner.py` 1,271줄 30메서드 | 단일 클래스 | 우선순위 낮음. 유지 | - |

### F. 도구(src/tools)

| # | 문제 | 근거 | 조치 | 위험 |
|---|------|------|------|------|
| F1 | `tools/__init__.py` 100줄 eager import → playwright·matplotlib 등 전부 로드 (fan-in 9) | `from src.tools.calculators import X` 만으로 scrapers 로드. 삭제된 `apify_amazon_scraper` try/except 잔재 | 지연 로딩 + 잔재 제거 | 낮 |
| F2 | tools → api 역참조 | A4 | A4에서 처리 | - |
| F3 | 중복 클래스: `CircuitBreaker`(amazon_scraper 내부 vs core/circuit_breaker), `AlertType`(email_sender vs state_manager), `VerificationResult`(claim_verifier vs insight_verifier) | 동명 이의 | scraper는 core CircuitBreaker 재사용 검토; AlertType은 domain으로 승격 후 양쪽 참조; VerificationResult는 `ClaimVerification`/`InsightVerification`으로 개명 | 낮 |
| F4 | `report_generator.py` 1,4xx줄에 Docx/Pptx/Pdf 생성기 4클래스 | 포맷별 의존(python-docx, python-pptx, reportlab) 동시 로드 | `exporters/report/{design.py, docx.py, pptx.py, pdf.py, facade.py}` 분할 | 중 |
| F5 | `amazon_product_scraper.py`, `exchange_rate.py` `__init__` 재수출 외 미사용 | grep 실사용 0 | 사용 계획 확인 후 삭제 후보 (이미 a2a1c64에서 수집기 4종 삭제한 것과 동일 기준) | 낮 |
| F6 | 테스트 없는 모듈 9개 | email_sender, telegram_bot, sheets_writer, chart_generator, insight_formatter, tavily_search, google_trends_collector, exchange_rate, amazon_product_scraper | Phase 5 | - |

### G. API(src/api)

| # | 문제 | 근거 | 조치 | 위험 |
|---|------|------|------|------|
| G1 | Pydantic 모델 중복 정의 | `ExportRequest`·`AnalystReportRequest`(models.py vs routes/export.py), `LayerDataResponse`·`MarketIntelligenceStatusResponse`(models.py vs routes/market_intelligence.py) | routes 쪽 정의 삭제, `api/models.py` 단일화 | 낮 |
| G2 | `routes/export.py` 1,260줄, `routes/alerts.py` 1,067줄 — 라우트 파일에 비즈니스 로직 | export.py 함수 11개 중 대부분 데이터 가공. `_get_external_signals`는 tools에서 역참조 | `application/services/{export_service.py, alert_service.py}` 추출. 라우트는 요청 검증 + 서비스 호출만 | 중 |
| G3 | 라우트가 `src.tools` 직접 import 25회 | export 10, alerts 5 등 | 서비스 추출(G2) 후 자연 감소. 잔여는 `api/deps/providers.py` 게터 경유 | 낮 |
| G4 | 라우트 6개 테스트 없음 | alerts, analytics, competitors, health, signals, sync | Phase 5. `TestClient` 스모크 테스트 | - |
| G5 | `dashboard_api.py` lifespan 전환 | 이미 완료(85행) | FUTURE_WORK §4 첫 항목 체크 처리 | - |

### H. 메모리·모니터링·shared

| # | 문제 | 근거 | 조치 | 위험 |
|---|------|------|------|------|
| H1 | `ConversationTurn` 중복 (`memory/conversation_memory.py` vs `memory/context.py`), `AgentStatus` 중복 (`core/state_manager.py` vs `memory/session.py`) | 동명 클래스 | 하나로 통합, 다른 쪽은 alias | 낮 |
| H2 | `memory/session_crypto.py`, `shared/llm_retry.py` 프로덕션 미참조 | 테스트만 존재 | 연결 의도 확인 후 사용 또는 삭제 | 낮 |
| H3 | `shared` → `monitoring` 의존 1건 | shared는 최하위 레이어여야 함 | `llm_client.py`의 logger 의존을 표준 logging으로 교체 | 낮 |
| H4 | `domain/interfaces/brain_components.py` 미참조 (224줄) | fan-in 0 | brain 컴포넌트가 실제로 구현하도록 연결(B6) 또는 삭제 | 낮 |

### I. 테스트(tests/)

| # | 문제 | 근거 | 조치 | 위험 |
|---|------|------|------|------|
| I1 | tests 루트 잔재 9파일 | `run_*.py` 2개(실데이터·LLM 필요 수동 스크립트), `test_core_modules.py`·`test_dashboard_ontology.py`·`test_query_rewriter.py`는 pytest 함수 0개(스크립트), `test_confidence_fusion.py`·`test_query_rewriter.py`는 `tests/unit/rag/`와 중복 | 수동 스크립트 → `scripts/manual/`; 중복 → 삭제; 통합 성격(`test_ir_rag_integration`, `test_rag_integration`, `test_llm_integration`) → `tests/integration/` + `@pytest.mark.slow` | 낮 |
| I2 | `tests/unit/eval/`(1파일) vs `tests/eval/`(18파일) 이원화 | 동일 대상 | `tests/eval/`로 통합 | 없음 |
| I3 | 테스트 없는 src 모듈 39개 | 위 표 참조 | Phase 5. 우선순위: api/routes 6 → ontology/rules 6 → notifications 2 | - |
| I4 | 커버리지 게이트 없음 | pyproject `fail_under = 0`, 목표 60% | CI에서 `--cov-fail-under=55`부터 점진 상향 | 낮 |

### J. 설정·빌드·문서·루트

| # | 문제 | 근거 | 조치 | 위험 |
|---|------|------|------|------|
| J1 | CLAUDE.md 드리프트 | 존재하지 않는 `src/agents/true_hybrid_insight_agent.py`, `src/core/batch_workflow.py`, `docs/research/` 기재. 파일 수 214 → 210. `docs/experiments·ir·market·troubleshooting` 누락 | 구조 트리 재생성. 이 문서의 조치 반영 시마다 갱신 | 없음 |
| J2 | AGENTS.md 20개(2026-01-31 생성)가 CLAUDE.md와 내용 중복·노후 | `src/adapters/AGENTS.md`는 "구현 예정" 상태 그대로 | 루트 AGENTS.md 1개만 유지하고 하위 19개는 삭제하거나, 유지한다면 각 디렉토리 `README.md`로 개명해 목적을 분명히 | 낮 |
| J3 | docs 루트 26개 산재 | `architecture.md` vs `SYSTEM_ARCHITECTURE.md` vs `CORE_ARCHITECTURE_DEEP_DIVE.md`, `TRUE_*` 2개, `INSIGHT_SYSTEM_*` 3개 | `docs/architecture/`, `docs/guides/`, `docs/plans/`, `docs/reports/`로 분류. 대체된 문서는 `docs/archive/` | 없음 |
| J4 | 루트 메타 파일 | `.sisyphus/`(타 에이전트 산출물 6개), `PORTFOLIO_FACTS.md`(50KB 포트폴리오), `.claude/*.md` 요약 2개, `.claude/skills/railway-*` 11개 스킬이 reference 4파일씩 중복 보유 | `.sisyphus/` → `docs/archive/sisyphus/` 또는 gitignore; `PORTFOLIO_FACTS.md` → `docs/reports/`; `.claude/*.md` → `docs/`. railway 스킬은 사용자 결정 | 없음 |
| J5 | scripts 혼재 | `debug_amazon_page*.py` 3개, `fix_summer_brand.py`, `test_crawl_100.py`, `test_report_generator.py` | `scripts/debug/`, `scripts/oneoff/` 분류. `test_*` 접두사는 pytest 오인 방지 위해 `check_*`로 개명 | 없음 |
| J6 | pyproject 불일치 | `[project].dependencies = ["aiosqlite"]` 1개 vs requirements.txt 40개; `[tool.black]` 잔존(ruff format 사용) | dependencies를 `dynamic = ["dependencies"]` + `[tool.setuptools.dynamic] dependencies = {file = ["requirements.txt"]}`로; black 섹션 삭제 | 낮 |
| J7 | ruff UP042 16건 | `class X(str, Enum)` → `StrEnum` | `ruff check --fix --unsafe-fixes` 후 검토 | 낮 |
| J8 | FUTURE_WORK.md 노후 | 2026-02-16 기준. lifespan 항목 이미 완료, 순환 23 → 실측 1 | 이 문서 기준으로 갱신 | 없음 |

---

## 3. 실행 순서(Phase 0~5)

각 Phase는 독립 PR. 앞 Phase가 뒤 Phase의 안전망.

| Phase | 내용 | 항목 | 예상 변경 | 위험 |
|-------|------|------|-----------|------|
| **0. 무위험 정리** | 빈 패키지·죽은 파일·테스트 잔재·문서·루트 | A1 A2 B1 B2 B5 I1 I2 J1 J2 J3 J4 J5 J6 J7 J8 | 삭제 위주, 코드 로직 변경 없음 | 없음 |
| **1. 중복 정의 통합** | 동명 클래스 17쌍 | D1 D2 F3 G1 H1 | 파일 8~10개 | 낮 |
| **2. 호환 shim 제거 + 지연 로딩** | shim 삭제, `__init__` 경량화 | B3 B7 C1 D5 E1 E2 E3 F1 F5 H2 H4 | import 경로 교체 20~30곳 | 낮 |
| **3. 의존 방향 교정** | 역참조·순환 제거, 서비스 추출 | A3 A4 A5 A6 C2 G2 G3 H3 | container·workflow·routes·services | 중 |
| **4. 대형 파일 분할** | god class 분해 | B6 B8 D3 D4 F4 | brain, hybrid_retriever, retriever, report_generator | 높 |
| **5. 테스트 보강** | 미테스트 39모듈, 커버리지 게이트 | D6 E4 F6 G4 I3 I4 | tests/ 추가 | 없음 |

Phase 0~2는 이번 주기에 함께 진행 가능. Phase 3 이후는 각각 별도 사이클 권장.

---

## 4. 검증 방법

| 단계 | 명령 | 통과 기준 |
|------|------|-----------|
| 매 PR | `ruff check src/ tests/ scripts/ && ruff format --check src/` | 0 errors |
| 매 PR | `python3 -m pytest tests/ -m "not slow" -q` | 기존 통과 수 유지 |
| Phase 2·3 | 정적 import 그래프 재실행(이 문서 작성 시 사용한 AST 스크립트) | SCC 0, tools→api 0, application→agents/infrastructure 0(팩토리 이관 후) |
| Phase 2 | `python3 -c "import src.tools.calculators.metric_calculator"` 후 `sys.modules`에 `playwright` 없음 | 지연 로딩 확인 |
| Phase 4 | 골든셋 `python3 scripts/evaluate_golden.py` | baseline v8.1 대비 L1~L5 회귀 없음 |
| Phase 5 | `pytest --cov=src --cov-fail-under=55` | 통과 |

---

## 5. 하지 않을 것

- `src/agents/`, `src/rag/`를 `src/adapters/`로 물리 이동: 200+ import 경로 변경 대비 이득 없음. adapters 개념은 폐기.
- `src/core/` 를 `application/orchestrators/`로 이동: 동일 이유. core를 그대로 두고 문서에서 "core = 오케스트레이션 레이어"로 정의.
- `owl_reasoner.py`, `period_insight_agent.py`, `hybrid_insight_agent.py` 분할: 단일 책임이 명확하고 순환·중복이 없어 분할 이득이 낮음.
- `config/` JSON 스키마화: FUTURE_WORK §2대로 별도 PR.

> 한 줄 결론: 폴더를 옮길 일은 거의 없고, **껍데기(adapters·orchestrators)와 shim 8개를 걷어내고, 동명 클래스 17쌍을 합치고, tools→api·application→infrastructure 역방향 2축을 끊는 것**이 이번 리팩토링의 실체다.
