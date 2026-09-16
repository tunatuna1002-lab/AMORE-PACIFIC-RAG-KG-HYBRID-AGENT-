# 인계 프롬프트 — 리팩토링 잔여 작업 (Opus 5용, 2026-09-16)

아래 내용을 그대로 첫 메시지로 붙여 넣으면 됩니다.

---

당신은 이 저장소(AMORE Pacific RAG-KG Hybrid Agent, Python 3.11, FastAPI, RAG+KG+온톨로지)의 리팩토링을 이어받는 시니어 리팩토링 엔지니어입니다. 브랜치 `claude/folder-structure-refactor-plan-rduj0r`에서 작업하고, 이 브랜치에만 푸시합니다. 아래 "완료 정의"를 전부 만족할 때까지 끝까지 진행하세요. 질문으로 멈추지 말고 합리적 판단으로 진행하되, 판단 근거는 커밋 메시지와 최종 보고에 남기세요.

## 0. 먼저 읽을 것 (이 순서로)

1. `docs/plans/folder-structure-refactor-plan-2026-09-01.md` — 전체 계획(v2). §2 결함 목록, §4 기능별 계획(F1~F9), §5 실행 순서와 TDD 규칙, §6 결정, §9 진행 로그.
2. `docs/research/rag-kg-ontology-agent-canonical-architecture.md` — 온톨로지·KG·RAG 정석 구조와 이 프로젝트 매핑(F9 기준점).
3. `CLAUDE.md` — 프로젝트 컨벤션. 단, 구조 트리는 노후되어 있음(마지막에 갱신하는 것도 당신 작업).
4. `git log --oneline -20` — 완료된 커밋.

## 1. 환경 준비

```bash
python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt pytest pytest-asyncio pytest-cov httpx
# 시스템 python에 직접 설치하면 PyYAML 충돌로 실패함 → 반드시 venv
.venv/bin/python -m pytest tests/unit -m "not slow" -q --no-cov -p no:cacheprovider     # 약 4~5분
ruff check src tests scripts && ruff format --check src tests scripts
```

- 테스트는 `.env.test`만 로드하고 실제 `.env`는 로드하지 않습니다(`tests/conftest.py`). 싱글턴은 테스트마다 자동 리셋됩니다.
- `tests/characterization/`는 특성화 테스트(현재 동작 고정)입니다. 리팩토링 커밋은 이 테스트를 바꾸지 않고 통과해야 하며, 동작을 의도적으로 바꿀 때만 해당 핀을 근거 주석과 함께 갱신합니다.
- 네트워크가 막힌 환경입니다(LLM·RSS·모델 다운로드 403). 그 자체는 정상이며, 테스트는 오프라인으로 통과해야 합니다.
- 서브에이전트를 쓴다면 동시에 2개 이하로만 띄우세요. 3개 이상 병렬 실행 시 세션 한도로 중간에 끊긴 이력이 있습니다. 각 에이전트에 파일 소유권을 명시해 충돌을 막으세요.

## 2. 확정된 결정 (재논의하지 말 것)

- Q1: 스케줄 배치(22:00 KST)는 크롤→저장→KG 갱신→지표→인사이트→알림→내보내기 전체를 실행한다. (완료: `CrawlManager`가 `BatchWorkflow`에 위임)
- Q2: OWL 온톨로지(owlready2)는 유지하되 실효화한다. 추론은 배치에서 실행해 JSON KG에 물질화(provenance 포함)하고, 챗 경로는 물질화 결과만 읽으며 `owlready2`를 import하지 않는다.
- 단위 규약: `MetricCalculator.calculate_sos`와 `share_of_shelf`(저장·API)는 퍼센트, 온톨로지·KG 메타데이터·규칙은 분수(0~1). 변환은 `src/shared/units.py` 한 곳.
- ReAct는 유지하되 env `ENABLE_REACT_AGENT` 기본 off.
- 삭제 확정 완료: rules_engine·rules.json, unified_reasoner, ontology_knowledge_graph, query_processor, core/types, brain_components, session_crypto, llm_retry, amazon_product_scraper, exchange_rate, llm_orchestrator, core/orchestrator shim, core/state.py(OrchestratorState), crawl/insight/alert 소형 워크플로우, relations.py·schema.py shim, adapters/·orchestrators/ 빈 패키지, 디렉토리별 AGENTS.md.

## 3. 완료된 것 (커밋 기준)

| Phase | 내용 | 커밋 |
|-------|------|------|
| 0 | 테스트 격리(.env.test, autouse 싱글턴 리셋, KG_PERSIST_PATH), tests 루트 잔재 정리, 락파일, docs/루트 분류, 특성화 테스트 205개, 골든셋 record/replay 게이트(`eval/replay.py`, `scripts/record_golden_replay.py`) | c4797a0 3c0c664 d8b023a |
| 1 | 결함 D1~D27 수정(테스트 선행): 배치·코어 b0ddbfa, RAG·온톨로지 14164c7, API 39fd2d0, 코어 신뢰도 d8a49d0 | |
| 2 | F1 배치 파이프라인 단일화 + F7 상태 단일화(StateManager) 7ded067 / F2 QueryGraph 단일 챗 경로(run/run_stream) 63d5d89 | |
| 3(일부) | 사문 모듈 삭제 730ad55, relations shim 279c3d3, 패키지 `__init__` 지연 로딩·동명 클래스 정리 8cdf522, core 정리 8349d7a, AGENTS.md 정리 61cb8ab | |
| WIP | **c275e42**: F4/F9 온톨로지(`tbox.py`·`builder.py`·`materializer.py`·`thresholds.py`·`inference_context.py`·CQ 테스트) + F6(`src/api/deps/`, `application/services/{dashboard_data_service,date_range,category_names}.py`, `domain/brand.py`, `shared/parsing.py`, 메모리 단일화). **미완성 — 아래 §4를 먼저 끝내야 함** | |

## 4. 즉시 할 일: WIP 커밋(c275e42) 마무리

현재 21 실패 / 4 오류 / ruff 3건. 원인별로 묶으면:

A. 메모리 단일화 미완 (`src/api/deps/session.py` ↔ `src/memory/conversation_memory.py`): `test_dependencies.py::{TestCleanupExpiredSessions, TestGetConversationHistory, TestAddToMemory}`, `test_session_memory_char.py` 5건, `test_api_contracts_char.py::test_chat_memory_delete_requires_api_key`(`DELETE /api/chat/memory/{id}`가 `ConversationMemory`를 dict처럼 `in`으로 검사 → TypeError). `conversation_memory`가 이제 객체이므로 라우트와 래퍼 함수(`add_to_memory`, `get_conversation_history`, `cleanup_expired_sessions`, `clear_session`)의 계약을 맞추고 특성화 핀을 갱신하세요(예: cleanup이 매 add마다 실행되도록 바뀐 것은 의도된 변경).
B. deps 분할 부작용: `test_dependencies.py::{TestLoadDashboardData, TestLogChatInteraction, TestJWTHelpers}` — 재수출 누락 또는 지연 초기화로 인한 패치 대상 변경. `src/api/dependencies.py`가 기존 이름 전부를 재수출하는지 확인.
C. 임계값 단일 출처(F4): `test_fix_metrics_agent.py` 3건 — `MetricsAgent._check_alerts`가 `src/ontology/thresholds.py`를 읽도록 바뀌면서 Phase 1 테스트(레거시 `thresholds` 키 폴백)와 계약 충돌. 결정: `config/thresholds.json`의 `ranking.significant_drop/rise`가 단일 출처, 레거시 키 폴백은 제거하고 테스트를 그에 맞게 갱신.
D. update_kg 단계(F9 배선): `test_batch_workflow.py::TestAct::test_act_update_kg`, `test_batch_workflow_deps.py::test_run_marks_state_manager`, `test_batch_workflow_char.py` 2건 — `BatchWorkflow`의 `update_kg` 단계가 `OntologyBuilder`+`materialize`를 호출하도록 바뀐 뒤 픽스처(fake KG/owl)와 기대값이 어긋남. 빌더가 없는 환경(owlready2 미설치)에서도 JSON KG 갱신은 되어야 하고, 물질화 실패는 step 오류로 기록(삼키지 않음).
E. OWL 전략 챗 경로(F9-4): `test_retrieval_strategy.py::{test_initialize_with_owl_reasoner, test_owl_reasoner_fallback}` — 요청 시 owl_reasoner 호출 대신 `materializer.list_inferred(kg)`를 읽도록 바뀐 계약에 맞춰 테스트 갱신(플래그 폴백 유지).
F. ruff: `src/ontology/builder.py:433` zip strict, `src/ontology/owl_reasoner.py:81-82` 미사용 import.

마무리 기준: 위 전부 green, `tests/unit/rag/test_chat_path_no_owlready2.py` 통과, `tests/unit/ontology/test_competency_questions.py` CQ1~CQ12 통과, ruff clean. 그 다음 WIP를 정식 커밋 메시지로 정리(`git commit --amend` 금지, 새 커밋).

## 5. 남은 작업 (순서대로)

### Phase 2 잔여
- F6 마무리: 라우트에 `./data` 리터럴이 남아 있지 않은지 ast 테스트, `dependencies.load_dashboard_data`가 `DashboardDataService` 래퍼인지, 6변형 LANEIGE 판별이 `src/domain/brand.py::is_target_brand`로 수렴했는지, 날짜 기본값 7곳이 `resolve_date_range`로 수렴했는지 grep으로 검증하고 테스트 추가.
- F7 메모리: `/api/v4/chat/stream`이 세션의 이전 턴을 brain에 전달하는지(`_build_query_state`가 history를 받는지 읽고 최소 침습으로 연결) 테스트.
- `src/ontology/business_rules.py` shim 삭제: 사용처(`src/agents/base_hybrid_agent.py`, `src/rag/hybrid_retriever.py`, `src/tools/exporters/dashboard_exporter.py`, `src/application/workflows/batch_workflow.py`, `scripts/*`, `tests/characterization/conftest.py`, `tests/integration/*`)를 `src.ontology.rules`로 교체.

### Phase 3 잔여
- 동명 클래스: `AgentStatus`(core/state_manager vs memory/session), `AlertType`(tools/notifications/email_sender vs core/state_manager → domain으로 승격), `ConversationTurn`(memory/context vs conversation_memory — F6 WIP에서 처리 중), `QueryIntent`(rag/hybrid_retriever의 레거시 enum vs application/services/query_analyzer vs core/intent — `core/intent.py`를 유일 정의로, 나머지는 alias 또는 삭제), `InsightAgentProtocol`(domain/interfaces/agent.py vs insight.py — 하나로).
- `src/core/__init__.py`·`src/api/dependencies.py` 등 재수출 파사드에 남은 이름 정리, `src/tools/utilities/__init__.py`·`collectors/__init__.py` 지연 로딩 여부 확인.
- `print(` 63건을 logging으로(특히 `dashboard_exporter.py`, `data_integrity_checker.py`), `AgentLogger` vs `logging.getLogger` 혼용은 후자로 통일(AgentLogger는 구조화 로깅 목적이 있으면 유지, 없으면 어댑터로).
- 문서: `CLAUDE.md` 구조 트리·핵심 모듈 표·파일 수를 실제와 일치시키기(존재하지 않는 `true_hybrid_insight_agent.py`, `src/core/batch_workflow.py`, `docs/research/`→존재함, adapters/ 제거 등), `docs/dev/FUTURE_WORK.md`를 이 계획 기준으로 갱신, `docs/plans/folder-structure-refactor-plan-2026-09-01.md` §9 진행 로그에 커밋 추가.

### Phase 4 분할 (특성화 테스트가 전부 통과하는 상태에서만 시작)
- `src/rag/hybrid_retriever.py`(약 1,800줄) → `rag/selfrag_gate.py`, `rag/kg_facts.py`(KG 조회·엣지 정렬·추론 컨텍스트는 이미 `ontology/inference_context.py`로 이동됨), `rag/query_expansion.py`(QueryEnhancer 흡수), `rag/fusion/`(RRF 1개, weighted merge 1개 — 현재 RRF 구현 3개·BM25 검색 2개), `rag/context_render.py`(→ ContextBuilder로 수렴). `HybridRetriever`는 파사드.
- `src/rag/retriever.py`(약 1,500줄) → `document_registry.py`, `document_loader.py`, `section_chunker.py`, `vector_index.py`, `bm25_index.py`, `search_cache.py`; `DocumentRetriever`는 파사드. 분할 전에 "세 RRF 구현이 동일 입력에 동일 순서"를 증명하는 등가 테스트를 먼저 작성.
- `src/api/routes/export.py`(1,260줄)·`alerts.py`(1,067줄)·`data.py`·`analytics.py`의 순수 데이터 가공을 `application/services/{export_service,alert_service,analytics_service}.py`로 추출. `export_handlers.py`의 분석 보고서 중복 경로(동기/비동기가 다른 문서를 생성)를 하나로.
- `src/tools/exporters/report_generator.py` → `exporters/report/{design,docx,pptx,pdf,facade}.py`.
- `src/core/brain.py`(현재 약 1,440줄): 스케줄링(BrainTask 큐)을 `core/brain_scheduler.py`로, 뉴스레터·모닝브리프 생성을 `tools/intelligence/`로 이동, brain은 초기화·DI·파사드만.
- 각 분할은 "함수 이동만, 로직 변경 없음" 커밋과 "수렴" 커밋을 분리.

### Phase 5 계약 강화
- D16 Protocol↔구현 정렬: `src/domain/interfaces/agent.py`의 4개 Protocol을 실제 에이전트 시그니처(`execute` 기반)에 맞추거나 삭제, `insight.py`·`alert.py` 인자 순서 정렬, `runtime_checkable` + `isinstance` 테스트로 고정.
- 단위 테스트 네트워크 차단: autouse fixture로 소켓을 막고(예: `socket.socket` 패치), 현재 실제 RSS를 호출하는 `tests/unit/agents/test_hybrid_chatbot_agent.py`(테스트당 약 9초)를 fake collector로 교체.
- 커버리지 게이트: `pyproject.toml`의 `fail_under`를 55로 올리고 CI(`.github/workflows/test.yml`)에 `--cov-fail-under` 반영. 테스트 없는 모듈 우선순위: `api/routes` 6개 → `ontology/rules` 6개 → `notifications` 2개.
- 정적 import 그래프 재검증: 순환 SCC 0, tools→api 0, application→agents/infrastructure 상단 import 0 (계획 문서 §4 검증 표의 AST 스크립트 방식).
- 골든셋 회귀 게이트: `OPENAI_API_KEY`가 있으면 `python scripts/record_golden_replay.py`로 기록을 만들고 `tests/eval/test_golden_replay_gate.py`가 skip이 아니라 실행되게. 키가 없으면 그 사실만 보고.

## 6. 작업 규칙 (계획 §5 TDD 규칙과 동일)

1. 리팩토링 커밋은 테스트 파일을 바꾸지 않는다. 특성화 테스트가 그대로 통과해야 "동작 보존"이다.
2. 동작을 바꾸는 커밋은 실패하는 테스트를 먼저 포함하고, 특성화 핀은 같은 커밋에서 근거 주석과 함께 갱신한다.
3. 삭제 커밋은 "호출처 0" grep 근거를 커밋 메시지에 남긴다.
4. 새 테스트는 공개 진입점만 쓴다(TestClient, `BatchWorkflow.run_daily_workflow`, `QueryGraph.run`, `HybridRetriever.retrieve`, `OntologyReasoner`, `MetricCalculator`). private 메서드·내부 모듈 경로 patch 금지(LLM 경계 `litellm.acompletion` 예외).
5. 커밋은 작게, 영역별로. 각 커밋 전에 `ruff check && ruff format --check`와 영향 테스트 실행. Phase 종료마다 전체 `tests/unit tests/eval tests/adversarial -m "not slow"` 실행.
6. `git commit --amend`, force push 금지. 커밋 메시지 끝에 기존 커밋과 같은 Co-Authored-By 형식을 유지할 필요는 없다.

## 7. 완료 정의 (Definition of Done)

- §4 WIP 마무리 + §5 Phase 2~5 전부 완료.
- `pytest tests/unit tests/eval tests/adversarial -m "not slow"` 전부 통과(스킵은 네트워크·선택 의존성만), `tests/characterization` 통과, ruff clean.
- 정적 import 그래프: 순환 0, 역방향 0.
- `CLAUDE.md`·`FUTURE_WORK.md`·계획 문서 §9가 최종 상태와 일치.
- 최종 보고: 커밋 목록, 삭제한 모듈과 근거, 남긴 결정과 이유, 테스트 수 변화(현재 기준: unit 약 5,150 + characterization 205), 실행하지 못한 항목과 이유.

---
