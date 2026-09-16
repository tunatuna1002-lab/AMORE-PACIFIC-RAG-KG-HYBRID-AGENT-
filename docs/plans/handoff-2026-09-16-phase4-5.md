# 인계 프롬프트 — Phase 4 마무리 + Phase 5 + 문서 (2026-09-16)

> 이 파일 전체를 새 세션의 첫 메시지로 붙여넣으면 된다.

---

당신은 이 저장소(AMORE Pacific RAG-KG Hybrid Agent, Python 3.11, FastAPI, RAG+KG+온톨로지)의
리팩토링을 이어받는 시니어 리팩토링 엔지니어입니다. 브랜치
`claude/folder-structure-refactor-plan-rduj0r` 에서 작업하고 이 브랜치에만 푸시합니다.
아래 "완료 정의"를 전부 만족할 때까지 끝까지 진행하세요. 질문으로 멈추지 말고 합리적 판단으로
진행하되, 판단 근거는 커밋 메시지와 최종 보고에 남기세요.

## 0. 먼저 읽을 것 (이 순서로)

1. `docs/plans/folder-structure-refactor-plan-2026-09-01.md` — 전체 계획(v2).
   §2 결함 목록, §4 기능별 계획(F1~F9), §5 실행 순서와 TDD 규칙, §6 결정, §9 진행 로그.
2. `docs/research/rag-kg-ontology-agent-canonical-architecture.md` — F9 기준점.
3. `CLAUDE.md` — 프로젝트 컨벤션. **구조 트리·핵심 모듈 표·파일 수는 지금 실제와 다르다.
   갱신하는 것도 당신 작업이다(§3-C).**
4. `git log --oneline -12` — 완료된 커밋.

## 1. 환경 준비

```bash
python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt pytest pytest-asyncio pytest-cov httpx
.venv/bin/pip install ruff==0.13.3     # ruff 버전에 따라 지적 건수가 다르다. 이 버전 기준으로 clean 상태다
```
(시스템 python 에 직접 설치하면 PyYAML 충돌로 실패한다 → 반드시 venv)

```bash
.venv/bin/python -m pytest tests/unit tests/eval tests/adversarial tests/characterization \
  -m "not slow" -q --no-cov -p no:cacheprovider          # 약 4~5분
.venv/bin/ruff check src tests scripts && .venv/bin/ruff format --check src tests scripts
```

환경 사실(추측하지 말 것):
- 테스트는 `.env.test` 만 로드하고 실제 `.env` 는 로드하지 않는다(`tests/conftest.py`).
  단 **이 컨테이너에는 `.env` 도 `.env.test` 도 없다**(둘 다 gitignore). `.env.example` 만 추적된다.
- 싱글턴(Container, brain, FeatureFlags, StateManager, 대화 메모리)은 테스트마다 autouse 로 리셋된다.
- `tests/conftest.py` 가 `AMORE_OWL_REASONER=python`, `LITELLM_LOCAL_MODEL_COST_MAP=True` 를
  기본 설정하고 **아웃바운드 소켓을 차단**한다(루프백 허용, `@pytest.mark.allow_network` 로 해제).
  단위 테스트가 네트워크를 때리면 `NetworkBlockedError` 로 즉시 실패한다.
- `tests/characterization/` 은 특성화 테스트(현재 동작 고정)다. 리팩토링 커밋은 이 테스트를
  **바꾸지 않고** 통과해야 하며, 동작을 의도적으로 바꿀 때만 `CHANGED (Fx):` 주석과 함께 핀을 갱신한다.
- 네트워크가 막힌 환경이다. `OPENAI_API_KEY` 는 env 에 **없고**(`env | awk -F= '$2 ~ /^sk-/'` → 0건),
  `curl https://api.openai.com/v1/models` 는 `HTTP 000`(연결 실패)이다. 프록시 `noProxy` 에도 없다.
  그 자체는 정상이며 테스트는 오프라인으로 통과해야 한다.
- 서브에이전트를 쓴다면 **동시에 2개 이하**. 파일 소유권을 프롬프트에 명시해 충돌을 막을 것.
  이전 세션에서 에이전트 2개를 띄우고 그 사이 리드가 같은 패키지를 건드려 충돌이 났던 적이 있다.

## 2. 확정된 결정 (재논의하지 말 것)

- **Q1**: 스케줄 배치(22:00 KST)는 크롤→저장→KG 갱신→지표→인사이트→알림→내보내기 전체를 실행한다. (완료)
- **Q2**: OWL 온톨로지(owlready2)는 유지하되 실효화한다. 추론은 배치에서 실행해 JSON KG 에
  물질화(provenance 포함)하고, 챗 경로는 물질화 결과만 읽으며 owlready2 를 import 하지 않는다. (완료)
- **단위 규약**: `MetricCalculator.calculate_sos` 와 `share_of_shelf`(저장·API)는 퍼센트,
  온톨로지·KG 메타데이터·규칙은 분수(0~1). 변환은 `src/shared/units.py` 한 곳.
- **임계값**: `config/thresholds.json` 이 단일 출처. 프로세스 공용 `src/ontology/thresholds.py`
  (`get_thresholds()` / `set_thresholds()`)를 통해서만 읽는다. 레거시 키 폴백은 제거됐다.
- **ReAct**: 유지하되 env `ENABLE_REACT_AGENT` 기본 off.
- **삭제 확정 완료**: rules_engine·rules.json, unified_reasoner, ontology_knowledge_graph,
  query_processor, core/types, brain_components, session_crypto, llm_retry, amazon_product_scraper,
  exchange_rate, llm_orchestrator, core/orchestrator shim, core/state.py, crawl/insight/alert 소형
  워크플로우, relations.py·schema.py shim, adapters/·orchestrators/ 빈 패키지, 디렉토리별 AGENTS.md,
  **ontology/business_rules.py shim**, **rag/query_enhancer.py**.

## 3. 지금 상태

### 완료 (푸시됨)

| 커밋 | 내용 |
|------|------|
| `c4797a0` `3c0c664` `d8b023a` | Phase 0 안전망: 테스트 격리, 특성화 테스트, 골든셋 record/replay 게이트 |
| `d8a49d0` `b0ddbfa` `14164c7` `39fd2d0` | Phase 1 결함 D1~D27 수정 |
| `7ded067` `63d5d89` `8349d7a` `730ad55` `279c3d3` `8cdf522` `61cb8ab` | Phase 2 F1·F2 + Phase 3 일부 |
| `c275e42` | (WIP) F4/F9 온톨로지 + F6 서비스·deps 분할 — 미완성이었음 |
| **`f08b1ac`** | 위 WIP 마무리. 21 실패/4 오류 → 0 |
| **`fe99e5d`** | Phase 2 잔여: F6 데이터 접근·날짜·브랜드 수렴, F7 세션 메모리 연결, business_rules shim 삭제, D16 Protocol 정렬, import 그래프 테스트 |
| **`6ff6e88`** | Phase 4 일부(report_generator·brain 분할), Phase 3(AlertType 도메인 승격, tools 지연 로딩), Phase 5(오프라인 게이트, 커버리지 55 게이트) |

구체적으로 이미 있는 것:
- `src/ontology/`: `tbox.py` `builder.py` `materializer.py` `thresholds.py` `inference_context.py`
  + 역량 질문 테스트(CQ1~CQ12).
- `src/api/deps/`: `auth` `session` `audit` `data` `suggestions` `providers`.
  `src/api/dependencies.py` 는 전부 재수출하는 파사드.
- `src/application/services/`: `dashboard_data_service` `date_range` `category_names` `query_analyzer`.
- `src/domain/brand.py`(`is_target_brand`), `src/domain/entities/alert.py`(`AlertType`),
  `src/shared/parsing.py`, `src/shared/units.py`.
- `src/core/brain_scheduler.py`(BrainTask 큐), `src/tools/intelligence/newsletter.py`,
  `src/tools/exporters/report/`(design/docx/pptx/pdf/facade).
- 테스트: `tests/unit/test_import_graph.py`(순환 0·역방향 0),
  `tests/unit/domain/test_protocol_conformance.py`(D16),
  `tests/unit/domain/test_target_brand_convergence.py`,
  `tests/unit/api/test_f6_data_access.py`, `tests/unit/api/test_chat_session_memory.py`.

### 미완성 — 이것부터 하라 (§3-A)

직전 세션이 서브에이전트 2개로 Phase 4 분할을 진행하다가 **세션이 끝나 중단**됐다.
그 상태가 커밋 `<WIP_COMMIT>` 에 그대로 들어 있다. **먼저 이것을 완결해야 한다.**

**A-1. `src/rag/` 분할 (F3)**
- 새로 생긴 모듈: `selfrag_gate.py` `kg_facts.py` `kg_edges.py` `query_expansion.py`
  `context_render.py` `hybrid_context.py` `legacy_intent.py` `fusion/{rrf,weighted}.py`
  `document_registry.py` `document_loader.py` `section_chunker.py` `vector_index.py`
  `bm25_index.py` `search_cache.py`.
- `hybrid_retriever.py` 1,742 → 665줄, `retriever.py` 1,479 → 676줄 (파사드로 남기는 중).
- `rag/query_enhancer.py` 삭제됨(→ `query_expansion.py` 로 흡수).
- 새 테스트 `tests/unit/rag/test_fusion_equivalence.py` — RRF 3벌·BM25 2벌·렌더러 3벌의
  **현재 동작을 고정하는 등가 테스트**. 수렴 커밋은 이 테스트가 지킨다.
- **남은 일**: 테스트 전부 green 확인, `HybridRetriever`/`DocumentRetriever` 파사드가 기존
  import 이름을 모두 유지하는지 확인, 계획 F3 삭제 목록(`search_hybrid`/`search_hybrid_async`/
  `retrieve_for_entity`, 레거시 `QueryIntent`·`INTENT_DOC_TYPE_PRIORITY`,
  `retrieval_weights.json` weights 블록 **또는** `_INTENT_STRATEGY_MAP` 중 죽은 쪽) 처리 —
  각각 "호출처 0" grep 근거를 커밋 메시지에 남길 것.

**A-2. API 라우트 → 서비스 추출 (F6, Phase 4)**
- 새로 생긴 모듈: `src/application/services/` 의 `export_service` `alert_service`
  `analytics_service` `historical_service` `sos_trend_service` `brand_matrix`
  `external_signals_service` `sql_rows`.
- 라우트 축소: `export.py` 1,244 → 346, `analytics.py` 742 → 195, `data.py` 821 → 113,
  `alerts.py` 999 → 737줄.
- **남은 일**: `tests/unit/tools/test_export_handlers.py` 가 **8건 실패 중**이다
  (`export_handlers.py` 의 동기/비동기 분석 보고서 중복 경로를 하나로 합치는 작업이 중단된 상태).
  계획은 "비동기 경로로 통일, kpis 를 dict 로 가정해 항상 실패하는 simple docx 제거"이지만
  **코드를 먼저 확인하고 계획과 다르면 코드를 따르고 그 사실을 보고할 것.**
  그리고 `tests/unit/api/test_route_service_split.py` 를 포함해 전체 green 을 만들 것.
- 인증 회귀 금지: 옮긴 모든 POST/PUT/DELETE 라우트가 `Depends(verify_api_key)` 를 유지하고,
  키 없이 호출 시 401/403 을 반환하는 테스트가 있어야 한다.

**A-3. 두 분할의 공통 마무리**
- `.venv/bin/python -m pytest tests/unit/test_import_graph.py` 로 순환 0 유지 확인
  (`application/` 은 `src.api` 를 import 하면 안 된다).
- 분할 모듈은 각 400줄 이하를 목표로. 넘으면 더 쪼갤 것.
- 커밋은 "함수 이동만(테스트 무수정 green)" 과 "수렴(등가 테스트가 지킴)" 을 **분리**할 것.

### 남은 작업 (A 다음, 순서대로)

**B. Phase 4 잔여 — `src/api/routes/alerts.py`(737줄)**
계획상 가장 큰 라우트가 아직 덜 줄었다. 순수 데이터 가공을 `alert_service.py` 로 더 옮기고,
이메일 본문 생성은 `tools/notifications/` 로 보낼 것.

**C. 문서 (§7 완료 정의에 포함)**
- `CLAUDE.md`: 구조 트리·핵심 모듈 표·파일 수가 실제와 다르다. 실측값 기준(대략 src 238파일·74,910줄,
  tests 228파일)으로 갱신하고, **존재하지 않는 항목을 지울 것**:
  `true_hybrid_insight_agent.py`, `src/core/batch_workflow.py`, `adapters/`, `core/orchestrator.py`,
  `unified_reasoner.py`, `ontology_knowledge_graph.py`, `query_processor.py`, `config/rules.json`.
  새로 생긴 것 추가: `src/api/deps/`, `src/application/services/`, `src/ontology/{tbox,builder,
  materializer,thresholds,inference_context}.py`, `src/core/brain_scheduler.py`,
  `src/tools/exporters/report/`, `src/tools/intelligence/newsletter.py`, `src/domain/brand.py`,
  `src/domain/entities/alert.py`, A-1/A-2 에서 확정된 rag·services 모듈들.
  로깅 컨벤션도 한 줄 추가: **에이전트·워크플로우는 `AgentLogger`, 라이브러리 모듈은
  `logging.getLogger(__name__)`. CLI(`__main__`)와 docstring 예시의 `print` 는 유지한다.**
- `docs/dev/FUTURE_WORK.md`: 통째로 노후됐다(이미 해결된 순환 의존성 23건, 삭제된
  `business_rules.py` 1,540줄, 3,236줄짜리 `dashboard_api.py` 등). 현재 남은 것만 남기고 재작성.
- `docs/plans/folder-structure-refactor-plan-2026-09-01.md` §9 진행 로그에 `f08b1ac` `fe99e5d`
  `6ff6e88` 및 당신의 커밋 추가.

**D. Phase 5 잔여**
- **커버리지 게이트 실측**: `pyproject.toml` 의 `fail_under = 55` 와 CI(`.github/workflows/test.yml`)
  는 이미 넣었지만 **실제 커버리지를 측정하지 않았다.** 전체 오프라인 스위트로 한 번 재서
  55 를 넘는지 확인하고, 못 넘으면 테스트 없는 모듈부터 채울 것.
  우선순위: `api/routes` → `ontology/rules` → `tools/notifications`.
  넘을 수 없는 근거가 있으면 수치를 낮추지 말고 **왜 그런지 보고**할 것.
- **골든셋 회귀 게이트**: `tests/eval/test_golden_replay_gate.py` 는 기록 파일
  `eval/baselines/replay/subset_nokg.jsonl` 이 없어 skip 된다. 이 컨테이너에서는 만들 수 없다
  (키 없음 + api.openai.com 송신 차단, §1 참조). 키가 있고 송신이 열린 환경에서
  `python3 scripts/record_golden_replay.py` 를 한 번 돌려 기록을 커밋하면 이후 오프라인 CI 에서
  자동 동작한다. **이 사실을 그대로 최종 보고에 남기고, 우회하려 하지 말 것.**
- **정적 import 그래프**: `tests/unit/test_import_graph.py` 가 이미 순환 0 / 역방향 0 을 지키고 있다.
  A-1/A-2 분할 후에도 통과하는지만 확인.

## 4. 작업 규칙

1. 리팩토링 커밋은 테스트 파일을 바꾸지 않는다. 특성화 테스트가 그대로 통과해야 "동작 보존"이다.
2. 동작을 바꾸는 커밋은 실패하는 테스트를 먼저 포함하고, 특성화 핀은 같은 커밋에서
   `CHANGED (Fx):` 근거 주석과 함께 갱신한다.
3. 삭제 커밋은 "호출처 0" grep 근거를 커밋 메시지에 남긴다.
4. 새 테스트는 공개 진입점만 쓴다(TestClient, `BatchWorkflow.run_daily_workflow`, `QueryGraph.run`,
   `HybridRetriever.retrieve`, `DocumentRetriever.search`, `OntologyReasoner`, `MetricCalculator`).
   private 메서드·내부 모듈 경로 patch 금지(LLM 경계 `litellm.acompletion` 예외).
5. 기존 assertion 을 약화하거나 지워서 green 을 만들지 않는다.
6. 커밋은 작게, 영역별로. 각 커밋 전에 `ruff check && ruff format --check` 와 영향 테스트 실행.
   Phase 종료마다 전체 `tests/unit tests/eval tests/adversarial tests/characterization -m "not slow"` 실행.
7. `git commit --amend`, force push 금지.
8. **서브에이전트가 도는 동안 리드가 같은 파일을 건드리지 말 것.** 커밋도 하지 말 것
   (진행 중 파일이 깨진 중간 상태로 기록에 남는다).

## 5. 완료 정의

- §3-A(분할 2건) 완결 + §3-B~D 전부 완료.
- `pytest tests/unit tests/eval tests/adversarial tests/characterization -m "not slow"` 전부 통과
  (스킵은 네트워크·선택 의존성·골든셋 기록 부재만), `ruff check`·`ruff format --check` clean.
- 정적 import 그래프: 순환 0, 역방향 0.
- `CLAUDE.md`·`FUTURE_WORK.md`·계획 문서 §9 가 최종 상태와 일치.
- 최종 보고: 커밋 목록, 삭제한 모듈과 근거, 남긴 결정과 이유, 테스트 수 변화,
  실제 커버리지 수치, 실행하지 못한 항목과 이유.
