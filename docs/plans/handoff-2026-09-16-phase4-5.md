# 인계 프롬프트 — 문서 갱신 + Phase 5 마무리 (2026-09-16)

> 아래 `---` 아래 전체를 새 세션의 첫 메시지로 붙여넣으면 된다.

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
3. `CLAUDE.md` — 프로젝트 컨벤션. **구조 트리·핵심 모듈 표·파일 수가 지금 실제와 크게 다르다.
   이걸 고치는 게 당신의 첫 번째 작업이다(§3-A).**
4. `git log --oneline -14` — 완료된 커밋.

## 1. 환경 준비

```bash
python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt pytest pytest-asyncio pytest-cov httpx
.venv/bin/pip install ruff==0.13.3   # ruff 버전마다 지적 건수가 다르다. 이 버전 기준으로 clean 이다
```
시스템 python 에 직접 설치하면 PyYAML 충돌로 실패한다 → 반드시 venv.

```bash
.venv/bin/python -m pytest tests/unit tests/eval tests/adversarial tests/characterization \
  -m "not slow" -q --no-cov -p no:cacheprovider          # 약 3~4분, 현재 5,561 통과 / 9 스킵
.venv/bin/ruff check src tests scripts && .venv/bin/ruff format --check src tests scripts
```

**환경 사실 (추측하지 말 것)**
- 테스트는 `.env.test` 만 로드하고 실제 `.env` 는 로드하지 않는다(`tests/conftest.py`).
  단 이 컨테이너에는 `.env` 도 `.env.test` 도 없다(둘 다 gitignore). `.env.example` 만 추적된다.
- 싱글턴(Container, brain, FeatureFlags, StateManager, 대화 메모리)은 테스트마다 autouse 로 리셋된다.
- `tests/conftest.py` 가 `AMORE_OWL_REASONER=python`, `LITELLM_LOCAL_MODEL_COST_MAP=True` 를
  기본 설정하고 **아웃바운드 소켓을 차단**한다(루프백 허용, `@pytest.mark.allow_network` 로 해제).
- **`OPENAI_API_KEY` 는 이 컨테이너에 없다.** 확인 방법과 결과:
  `env | awk -F= '$2 ~ /^sk-/'` → 0건, `ls .env .env.test` → 없음,
  `curl https://api.openai.com/v1/models` → `HTTP 000`(연결 실패, 401 아님).
  프록시 `noProxy` 목록에도 `api.openai.com` 이 없다. 정상이며, 우회하려 하지 말 것.
- 서브에이전트는 **동시에 2개 이하**. 각 에이전트 프롬프트에 파일 소유권을 명시할 것.
  **에이전트가 도는 동안 리드는 같은 파일을 건드리지 말고, 커밋도 하지 말 것**
  (진행 중 파일이 깨진 중간 상태로 기록에 남는다 — 이전 세션에서 실제로 겪었다).

## 2. 확정된 결정 (재논의하지 말 것)

- **Q1**: 스케줄 배치(22:00 KST)는 크롤→저장→KG 갱신→지표→인사이트→알림→내보내기 전체 실행. (완료)
- **Q2**: OWL 온톨로지(owlready2) 유지하되 실효화. 추론은 배치에서 실행해 JSON KG 에
  물질화(provenance 포함)하고, 챗 경로는 물질화 결과만 읽으며 owlready2 를 import 하지 않는다. (완료)
- **단위 규약**: `MetricCalculator.calculate_sos` 와 `share_of_shelf`(저장·API)는 퍼센트,
  온톨로지·KG 메타데이터·규칙은 분수(0~1). 변환은 `src/shared/units.py` 한 곳.
- **임계값**: `config/thresholds.json` 이 단일 출처. 프로세스 공용 `src/ontology/thresholds.py`
  (`get_thresholds()` / `set_thresholds()`) 를 통해서만 읽는다. 레거시 키 폴백 제거됨.
- **ReAct**: 유지하되 env `ENABLE_REACT_AGENT` 기본 off.
- **삭제 확정 완료**: rules_engine·rules.json, unified_reasoner, ontology_knowledge_graph,
  query_processor, core/types, brain_components, session_crypto, llm_retry, amazon_product_scraper,
  exchange_rate, llm_orchestrator, core/orchestrator shim, core/state.py, crawl/insight/alert 소형
  워크플로우, relations.py·schema.py shim, adapters/·orchestrators/ 빈 패키지, 디렉토리별 AGENTS.md,
  ontology/business_rules.py shim, rag/query_enhancer.py,
  DocumentRetriever.search_hybrid(_async), HybridRetriever.retrieve_for_entity,
  export_handlers.handle_export_docx.

## 3. 지금 상태 — Phase 0~4 완료, 남은 건 문서와 Phase 5 일부

| 커밋 | 내용 |
|------|------|
| `c4797a0` `3c0c664` `d8b023a` | Phase 0 안전망: 테스트 격리, 특성화 테스트, 골든셋 record/replay 게이트 |
| `d8a49d0` `b0ddbfa` `14164c7` `39fd2d0` | Phase 1 결함 D1~D27 수정 |
| `7ded067` `63d5d89` `8349d7a` `730ad55` `279c3d3` `8cdf522` `61cb8ab` | Phase 2 F1·F2 + Phase 3 일부 |
| `c275e42` → `f08b1ac` | F4/F9 온톨로지 + F6 서비스·deps 분할 (WIP → 마무리, 21실패/4오류 → 0) |
| `fe99e5d` | Phase 2 잔여: F6 데이터 접근·날짜·브랜드 수렴, F7 세션 메모리 연결, business_rules shim 삭제, D16 Protocol 정렬, import 그래프 테스트 |
| `6ff6e88` | Phase 4 일부(report_generator·brain 분할) + Phase 3(AlertType 도메인 승격, tools 지연 로딩) + Phase 5(오프라인 게이트) |
| `a3705ad` | **Phase 4 완료**: F3 rag 스택 분할 + F6 API 라우트 → 서비스 추출 |

**현재 수치**: 테스트 5,561 통과 / 9 스킵, ruff clean, import 순환 0·역방향 0,
**커버리지 79.23%**, src 238파일·74,910줄, tests 228파일.

주요 분할 결과(문서에 반영해야 함):
- `src/rag/`: `hybrid_retriever.py` 1,742→665, `retriever.py` 1,479→676(둘 다 파사드). 신설
  `selfrag_gate` `kg_facts` `kg_edges` `query_expansion` `context_render` `hybrid_context`
  `legacy_intent` `fusion/{rrf,weighted,hybrid_search}` `document_registry` `document_loader`
  `section_chunker` `vector_index` `bm25_index` `search_cache`.
- `src/api/routes/`: `export` 1,244→346, `analytics` 742→195, `data` 821→113, `alerts` 999→737.
- `src/application/services/`: `dashboard_data_service` `date_range` `category_names`
  `query_analyzer` `export_service` `alert_service` `analytics_service` `sos_trend_service`
  `historical_service` `brand_matrix` `external_signals_service` `sql_rows`.
- `src/api/deps/`: `auth` `session` `audit` `data` `suggestions` `providers`
  (`src/api/dependencies.py` 는 전부 재수출하는 파사드).
- `src/ontology/`: `tbox` `builder` `materializer` `thresholds` `inference_context` + CQ 테스트.
- `src/core/brain_scheduler.py`, `src/tools/intelligence/newsletter.py`,
  `src/tools/exporters/report/{design,docx,pptx,pdf,facade}.py`,
  `src/domain/brand.py`, `src/domain/entities/alert.py`, `src/shared/{units,parsing}.py`.

## 4. 할 일

### A. 문서 갱신 (최우선, 완료 정의에 포함)

**A-1. `CLAUDE.md`** — §4 구조 트리와 §9 핵심 모듈 표가 실제와 다르다.
- **지울 것(존재하지 않음)**: `true_hybrid_insight_agent.py`, `src/core/batch_workflow.py`,
  `adapters/`, `core/orchestrator.py`, `unified_reasoner.py`, `ontology_knowledge_graph.py`,
  `query_processor.py`, `config/rules.json`, `business_rules.py`, `rag/query_enhancer.py`.
- **추가할 것**: §3 의 분할 결과 전부.
- **파일 수 갱신**: src 238파일 / 74,910줄, tests 228파일. (직접 재서 쓸 것 — 위 수치는
  2026-09-16 기준이고 당신 작업 뒤에 달라진다.)
- **컨벤션 한 줄 추가**: 로깅은 에이전트·워크플로우가 `AgentLogger`, 라이브러리 모듈이
  `logging.getLogger(__name__)`. CLI(`__main__`)와 docstring 예시의 `print` 는 유지한다.

**A-2. `docs/dev/FUTURE_WORK.md`** — 통째로 노후됐다. 이미 해결된 것(순환 의존성 23건,
`business_rules.py` 1,540줄, 3,236줄짜리 `dashboard_api.py`, Application Layer 120 LOC)을
전부 걷어내고 현재 남은 것만 재작성. 아래 §4-C 의 미해결 항목을 여기에 옮겨 적을 것.

**A-3. 계획 문서 §9 진행 로그** — `f08b1ac` `fe99e5d` `6ff6e88` `a3705ad` 및 당신 커밋 추가.

### B. Phase 5 잔여

**B-1. 커버리지 게이트** — 실측 79.23%. `pyproject.toml` 과 `.github/workflows/test.yml` 에
`fail_under = 75` 로 이미 설정돼 있다(실측 대비 4%p 여유). 당신 작업 후 다시 재서 여전히
75를 넘는지 확인만 하면 된다. 못 넘으면 **수치를 낮추지 말고** 테스트를 채울 것.
테스트가 얇은 순서: `api/routes` → `ontology/rules` → `tools/notifications`.

**B-2. 골든셋 회귀 게이트** — `tests/eval/test_golden_replay_gate.py` 는 기록 파일
`eval/baselines/replay/subset_nokg.jsonl` 이 없어 skip 된다. **이 컨테이너에서는 만들 수 없다**
(§1 참조). 키가 있고 OpenAI 송신이 열린 환경에서 `python3 scripts/record_golden_replay.py` 를
한 번 돌려 기록을 커밋하면 이후 오프라인 CI 에서 자동 동작한다. 최종 보고에 그대로 남길 것.

**B-3. `src/api/routes/alerts.py` (737줄)** — 남은 라우트 중 가장 크다. 이 중 약 390줄이
이메일 확인 페이지 2개의 인라인 HTML 이다. 템플릿 파일로 빼면 라우트가 ~350줄이 된다.

### C. 직전 세션이 근거를 남기고 **일부러 미룬** 것 (동작 변경이라 별도 커밋 필요)

각각 이유가 코드 docstring 에 적혀 있다. 하려면 RED 테스트 선행 + 골든셋 확인이 필요하다.

1. **RRF dedup 키 통일** (`src/rag/fusion/rrf.py`). 세 호출처의 id 폴백이 다르다
   (`""` / `str(rank)` / content 해시). 둘 다 잠복 결함이다 — `""` 는 id 없는 문서를 전부
   하나로 합치고, `str(rank)` 는 서로 다른 리스트의 같은 순위 문서를 잘못 병합한다.
   검색 결과가 바뀌므로 결함 커밋으로 다룰 것. 현재 차이는 등가 테스트가 고정하고 있다.
2. **KG 사실 렌더러 문구 통일** (`src/rag/context_render.py`). 수치는 같고 문구만 다르며
   각자의 테스트가 그 문구를 고정한다. 프롬프트 텍스트가 바뀌므로 골든셋 실행 동반 필요.
   승자는 `ContextBuilder`(출처를 등록하는 유일한 렌더러).
3. **BM25+RRF 이중 실행** (`src/rag/fusion/hybrid_search.py`). `DocumentRetriever.search` 가
   이미 RRF 융합을 한 뒤 `HybridRetriever._hybrid_search` 가 다른 점수 공간에서 또 한다.
   `test_hybrid_retriever_char.py` 의 `search_method == "hybrid_rrf"` 핀이 걸려 있다.
4. **`HybridRetriever` 의 doc-type 필터가 BM25 레그에 도달하지 않는다** (신규 발견, 테스트로 고정됨).
5. **검색 캐시 키에 임베딩 모델명 누락** (`src/rag/search_cache.py`).
6. **`config/retrieval_weights.json` 의 `weights` 블록은 죽었다** — 프로덕션은 항상
   `_INTENT_STRATEGY_MAP` 의 가중치를 넘긴다. 단 같은 파일의 `freshness` 와
   `max_context_items` 는 **살아 있다**(특성화 테스트가 `rag_chunks: 8` 을 이 파일에서
   가져오는 것을 고정). 파일을 지우지 말고 `weights` 블록만 정리할 것.
7. **`tests/integration/test_rag_integration.py`** 가 `src.rag.hybrid_retriever` 에서
   `QueryIntent`/`get_doc_type_filter` 를 import 한다. `src.core.intent` 로 돌리면
   `src/rag/legacy_intent.py` 를 삭제할 수 있다.
8. **`src/api/dashboard_shape.py`** 가 API 계층에 있는 순수 어댑터라 `application/` 도 `tools/` 도
   import 할 수 없다. `application/services/` 로 옮기면 뷰모델도 서비스에서 조립 가능해진다.
9. **계획 Q4 삭제 항목은 사용자 확인 대기 중이라 손대지 않았다**: 동기 `/api/export/docx`,
   `/api/v3/*`, `/api/chat/memory/*`.

> 주의: 직전 세션의 한 서브에이전트가 "`JobType.EXPORT_DOCX` 는 핸들러가 없으니 지워도 된다"고
> 보고했으나 **틀렸다.** `/api/export/async/start` 가 여전히 이 `job_type` 값을 받고 여러 테스트가
> 그것을 쓴다. 사라진 것은 핸들러뿐이다. 에이전트 보고는 반드시 직접 grep 으로 검증할 것.

## 5. 작업 규칙

1. 리팩토링 커밋은 테스트 파일을 바꾸지 않는다. 특성화 테스트가 그대로 통과해야 "동작 보존"이다.
2. 동작을 바꾸는 커밋은 실패하는 테스트를 먼저 포함하고, 특성화 핀은 같은 커밋에서
   `CHANGED (Fx):` 근거 주석과 함께 갱신한다.
3. 삭제 커밋은 "호출처 0" grep 근거를 커밋 메시지에 남긴다.
4. 새 테스트는 공개 진입점만 쓴다(TestClient, `BatchWorkflow.run_daily_workflow`, `QueryGraph.run`,
   `HybridRetriever.retrieve`, `DocumentRetriever.search`, `OntologyReasoner`, `MetricCalculator`).
   private 메서드·내부 모듈 경로 patch 금지(LLM 경계 `litellm.acompletion` 예외).
5. 기존 assertion 을 약화하거나 지워서 green 을 만들지 않는다.
6. 커밋은 작게, 영역별로. 각 커밋 전에 `ruff check && ruff format --check` 와 영향 테스트 실행.
7. `git commit --amend`, force push 금지.

## 6. 완료 정의

- §4-A(문서 3종) 전부 완료, §4-B 완료.
- `pytest tests/unit tests/eval tests/adversarial tests/characterization -m "not slow"` 전부 통과
  (스킵은 네트워크·선택 의존성·골든셋 기록 부재만), `ruff check`·`ruff format --check` clean.
- 정적 import 그래프: 순환 0, 역방향 0 (`tests/unit/test_import_graph.py`).
- 커버리지 75 게이트 통과(실측값을 보고에 기재).
- `CLAUDE.md`·`FUTURE_WORK.md`·계획 문서 §9 가 최종 상태와 일치.
- 최종 보고: 커밋 목록, 삭제한 모듈과 근거, 남긴 결정과 이유, 테스트 수 변화,
  실제 커버리지 수치, 실행하지 못한 항목과 이유.
