# 리팩토링 계획 v2 — 구조 + 의미론 검토 (2026-09-02)

> 기준 커밋: a46ce76 (main)
> v1(구조 검토)에 의미론 검토를 더해 전면 개정. 검증 라벨: [직접 확인] = 이 문서 작성 시 코드/실행으로 확인, [검토 보고] = 영역별 코드 리뷰 결과(라인 번호 제시), [추정] = 근거 부분적.
>
> **핵심 결론**: 문제의 본질은 폴더 배치가 아니라 **"문서·설계가 말하는 시스템"과 "실제로 실행되는 시스템"이 다르다**는 점이다. 배치·챗봇·추론 세 축 모두 설계된 경로의 절반이 실행되지 않거나(유령 경로), 두 벌 이상이 공존한다. 리팩토링은 "경로를 하나로 정하고 유령을 지우는 일"이 먼저이고, 폴더 정리는 그다음이다.

## 목차

1. 실행 경로 실사 요약 (설계 vs 실제)
2. 발견된 결함 목록 (수정이 리팩토링보다 먼저)
3. 테스트 안전망 진단 (TDD 관점)
4. 기능별 계획
   - F1 일일 배치 파이프라인
   - F2 챗봇 질의 파이프라인
   - F3 검색(RAG) 스택
   - F4 추론·규칙·지표 단위
   - F5 알림
   - F6 API·대시보드 데이터 계약
   - F7 상태·메모리·설정
   - F8 구조 정리 (v1 항목, 우선순위 재조정)
5. 실행 순서 (Phase 0~5) 와 TDD 절차
6. 사용자 결정이 필요한 항목
7. 하지 않을 것
8. 부록: 실측 수치

---

## 1. 실행 경로 실사 요약

| 축 | 문서/설계상 경로 | 실제 실행 경로 | 근거 |
|----|-----------------|----------------|------|
| 일일 배치 | 크롤 → 저장 → 지표(MetricsAgent) → 인사이트(HybridInsightAgent) → 알림(AlertAgent) → KG 갱신 | `scheduler → brain → CrawlManager._run_crawl` = 크롤 → 저장 → DashboardExporter **뿐**. 지표는 Exporter 자체 산술, 인사이트·알림·KG 갱신 없음. 전체 파이프라인(BatchWorkflow Think→Act→Observe)은 수동 `POST /api/v4/brain/*`, `main.py`, 골든셋 스크립트에서만 실행 | [직접 확인] `crawl_manager.py`가 import하는 에이전트는 DashboardExporter 뿐. [검토 보고] `brain.py:1216` vs `:1634` 동일 액션 문자열 `crawl_workflow`에 두 핸들러 |
| 챗봇 | `/api/v4/chat` → QueryGraph(Guard→Cache→Gather→Confidence→Decide→Tool→Respond), 복잡 질의는 ReAct | UI는 `/api/v4/chat/stream`만 호출 → `UnifiedBrain.process_query_stream`의 **인라인 복사본** 파이프라인(캐시 없음, compound 판정 없음). QueryGraph는 UI가 안 쓰는 비스트리밍 엔드포인트에서만. ReAct는 `from ..agents.react_agent` 경로가 존재하지 않아 항상 None | [직접 확인] `brain.py:375` import 경로, `src/agents/react_agent.py` 부재, 예외는 debug 로그로 삼켜짐 |
| 검색 | OWL 전략(feature flag 기본 true) + 레거시 전략 | OWL 전략은 생성자에 없는 `docs_path` 인자를 넘겨 TypeError → 삼켜짐 → 항상 레거시. 리랭커·관련성 판정은 flag false로 비활성 | [직접 확인] `retrieval_strategy.py:240-249` 시그니처, `brain.py:321`, `container.py:170` 호출 |
| 추론 | OWL + 비즈니스 규칙 + rules.json 엔진 통합 | 실제 인사이트는 `OntologyReasoner + rules/*` 하나. `core/rules_engine.py`+`config/rules.json`은 생성만 되고 호출 0. `unified_reasoner.py`는 생성조차 안 됨. `thresholds.json`의 임계값 블록은 아무도 읽지 않음 | [검토 보고] |
| 지표 | `MetricCalculator`가 SoS/HHI/CPI 단일 정의 | SoS는 routes/data·analytics·alerts, period_analyzer, dashboard_exporter, kg_enricher에서 각자 재계산. HHI는 세 가지 스케일(분수·%²/10000·0~10000) 공존 | [검토 보고] |

---

## 2. 발견된 결함 목록

리팩토링 전에 **각각 RED 테스트 → 최소 수정**으로 처리한다. 구조 변경과 섞지 않는다.

| # | 결함 | 영향 | 위치 | 검증 |
|---|------|------|------|------|
| D1 | `HybridInsightAgent.generate_insight()` 메서드가 존재하지 않는데 호출 → 예외 삼킴 | 모닝 브리프 이메일 인사이트가 항상 "없습니다" | `brain.py:1537` | [직접 확인] |
| D2 | SoS 단위 불일치: `calculate_sos`는 퍼센트(×100) 반환, 규칙은 분수(0.15) 기준 | `sos_above(0.15)`가 SoS 0.15%부터 발화, `category_entry_opportunity(<0.03)`는 영원히 미발화. KG metadata에 퍼센트/분수 두 스케일 혼재 | `metric_calculator.py:131`, `hybrid_retriever.py:1104`, `market_rules.py:17,44,73`, `kg_updater.py:144`(퍼센트) vs `brain.py:771`(÷100) | [직접 확인] |
| D3 | ReAct 에이전트 import 경로 오류 | 복잡 질의 분기 전부 사문화, 테스트는 `src.core.react_agent`를 import해 가림 | `brain.py:375` | [직접 확인] |
| D4 | OWL 전략 생성자 인자 불일치(`docs_path`) | OWL 검색 전략·ConfidenceFusion 랭킹·EntityLinker 필터 전부 미실행 | `retrieval_strategy.py:240`, `brain.py:321`, `container.py:170` | [직접 확인] |
| D5 | `metrics_agent.py:114` `product.get("product_asin")` — 레코드 키는 `asin` | 이력 항상 빈 리스트 → `rank_change_1d` 항상 None → 순위 급락 알림 발화 불가 | `metrics_agent.py:114` vs `crawler_agent.py:125` | [직접 확인] |
| D6 | `AlertAgent.send_pending_alerts` 반환 키 `sent` vs 워크플로우가 읽는 `sent_count` | 알림 발송 수 항상 0 보고 | `alert_agent.py:260`, `alert_workflow.py:107` | [직접 확인] |
| D7 | 대시보드 JSON 형태 불일치: Exporter는 `products`를 ASIN 키 dict로 쓰는데 export/alerts/competitors 라우트는 list로 읽음 | `/api/export/docx` 빈 보고서, `/api/alerts/send-insight-report` 500, `/api/competitors` LANEIGE 항상 빈 값 | `dashboard_exporter.py:735`, `export.py:492`, `alerts.py:962` | [직접 확인] |
| D8 | 변경 엔드포인트 인증 누락 | `PUT/DELETE /api/v4/alert-settings`(이메일만 알면 타인 설정 변경), `POST /api/data/refresh`, `/api/signals/*`, `/api/sync/upload`, `/api/export/*` | `alerts.py:443,485`, `data.py:192`, `signals.py:112-218`, `sync.py:160`, `export.py:359-1000` | [직접 확인] 데코레이터에 `verify_api_key` 없음 |
| D9 | `OrchestratorState` 기록자 부재 | `data_freshness` 영구 "unknown", LLM 프롬프트에 항상 "크롤링 기록 없음", 제안 질문에 항상 stale 문구 | `state.py` mark_* 호출처 0 | [직접 확인] |
| D10 | 신뢰도 스케일 혼합 `max(0~10, 0~1)` | API가 6.5와 0.2를 같은 필드로 반환, HIGH 결정(0.9)은 환각 검사(≥0.8 스킵) 항상 우회 | `response_pipeline.py:196` | [직접 확인] |
| D11 | `thresholds` 설정 키 불일치 (`config["thresholds"]["significant_rank_drop"]` 없음, 실제는 `ranking.significant_drop`) | 기본값 5 고정, 설정 무의미 | `metrics_agent.py:282` | [검토 보고] |
| D12 | 부분 실패를 성공으로 기록 (`status=="partial"`, 저장 실패 warning 후 COMPLETED) | 일부 카테고리 누락일에 재시도 억제 | `crawl_manager.py:321,379,400` | [검토 보고] |
| D13 | 스케줄러가 작업 시작 직후 `mark_completed` (create_task만 하고 완료 처리) | 크롤 중 크래시 시 그날 완료로 남음 | `scheduler.py:223`, `crawl_manager.py:291` | [검토 보고] |
| D14 | `_hybrid_search`: BM25 결과에 `id` 없음 → 특정 조건에서 KeyError → 빈 컨텍스트 반환 | 검색 결과가 조용히 사라짐 | `retriever.py:1338`, `hybrid_retriever.py:540,619` | [검토 보고] |
| D15 | 대화 메모리 이중 구현: 라우트가 `add_to_memory`(dict)로 쓰지만 읽는 곳 없음, ContextManager는 별도 | 스트리밍 챗봇에서 대화 맥락 미사용(추정) | `dependencies.py:70-121`, `chat.py:86` | [직접 확인] 쓰기만 존재 |
| D16 | Protocol과 구현체 시그니처 불일치 (`agent.py` 프로토콜 4종 모두, `insight.py` execute 인자, `alert.py` create_alert 인자 순서) | 소형 워크플로우(Crawl/Insight/Alert)는 실행 시 TypeError. 현재 미호출이라 잠복 | `domain/interfaces/agent.py:14-52`, `insight_workflow.py:102` | [검토 보고] |
| D17 | 규칙 입력 미생산: `churn_rate_7d` 항상 None, `ir_*`·`parent_group`·`discount_periods` 등 생산자 0 | IR 규칙 6개·인과 규칙 4개 영구 미발화, 감성 규칙은 shape 불일치로 예외→False | `metrics_agent.py:184`, `sentiment_rules.py:22` vs `kg_updater.py:476` | [검토 보고] |
| D18 | 일일 export가 `rank_change_7d=0, streak_days=7, rating_gap=0.1` 하드코딩 컨텍스트로 추론 | 매일 같은 규칙만 발화 | `dashboard_exporter.py:1477` | [검토 보고] |
| D19 | 타임존 혼용: snapshot_date는 KST, 나머지 naive `datetime.now()`(UTC), analytics 기본 end_date는 서버 UTC | Railway에서 하루 9시간 동안 당일 행 누락 | `analytics.py:61`, `storage_agent.py:169` 등 41곳 | [검토 보고] |
| D20 | 비동기 핸들러 내 동기 I/O (sqlite3, pandas, matplotlib, 12k줄 HTML 읽기) | 요청 중 이벤트 루프 블로킹 | `analytics.py`, `sync.py`, `deals.py`, `export.py:674,1022`, `job_queue.py:112`, `health.py:38` | [검토 보고] |

---

## 3. 테스트 안전망 진단 (TDD 관점)

| 항목 | 실측 |
|------|------|
| 샘플 37파일 분류 | 실제 협력자 사용 13 / 가벼운 fake 9 / **구현 결합(mock·private) 12** / smoke 2 / pytest 아님 6 |
| 리팩토링 시 무의미하게 깨질 patch 대상 상위 | `owl_reasoner.OWLREADY2_AVAILABLE` 20, `litellm.acompletion` 17, `builtins.open` 16, `query_rewriter.acompletion` 15, `query_graph.PromptGuard` 13, `hybrid_retriever.register_all_rules` 11, `brain.get_brain` 10 |
| private 접근 최다 | test_brain 121, test_batch_workflow 112, test_hybrid_insight_agent 93, test_amazon_scraper 84, test_dashboard_exporter 79 |
| 전무한 테스트 | E2E 챗(질의→의도→KG→RAG→답), 배치 E2E, **API 계약(TestClient 0개)**, KG 갱신 멱등성, 설정 파일 값이 실제 알림에 도달하는지 |
| 있는 테스트 | SoS/HHI/CPI 수치(`test_metric_calculator.py:129-340`) — 유지 가치 높음 |
| 환경 | `conftest.py`가 실제 `.env`를 로드, `.env.test` 부재 → 개발자 실제 API 키가 단위 테스트에 노출. `KnowledgeGraph()` 기본 `auto_load=True`로 3개 테스트가 `data/knowledge_graph.json` 실파일 의존 |
| 골든셋 | `eval/` 러너는 `StubJudge` 기본이라 L1~L4 결정적. 다만 `chat()`이 실제 LLM+실데이터 호출이라 오프라인 회귀 게이트로 쓰려면 record/replay 필요 |

> 판정: 현재 스위트는 "구현을 옮기면 깨지고, 동작이 틀려도 통과"하는 방향으로 기울어 있다. 리팩토링 전에 공개 진입점 기준 특성화 테스트(characterization test)를 먼저 깔아야 한다.

### 3.1 실제 실행 결과 (2026-09-02, 가상환경, `tests/unit tests/eval tests/adversarial -m "not slow"`)

| 결과 | 수치 |
|------|------|
| 통과 / 실패 / 스킵 | 5,141 / 9 / 7 (274초) |
| 실패 원인 A (8건) | `tests/unit/api/test_route_migration.py` — `route.path` 속성으로 라우터 등록을 검사하는데, 설치된 FastAPI 0.141은 include된 라우터를 `_IncludedRouter`(path 없음)로 지연 등록. TestClient로 확인하면 `/api/health` 200, OpenAPI 경로 71개로 **앱은 정상**. 즉 구현 결합 테스트가 프레임워크 버전에 깨진 사례 |
| 실패 원인 B (1건) | `tests/eval/test_semantic.py` — 문장 임베딩 모델 다운로드 403(네트워크). 오프라인 스킵 조건 부재 |
| 파생 발견 D21 | `requirements.txt`가 `fastapi>=0.104.0`처럼 하한만 고정 → 환경마다 다른 버전 설치. 락파일(`pip-compile` 또는 `uv lock`) 도입을 Phase 0에 추가 |

> 해석: 5,141개가 통과하지만 §2의 결함 D1~D20은 하나도 잡지 못했다. 통과 수가 안전망의 크기를 뜻하지 않는다는 실증이다.

---

## 4. 기능별 계획

각 기능은 **현재 → 문제 → 목표 → TDD 절차 → 삭제** 순으로 적는다. TDD 절차의 RED는 "현재 동작을 고정하는 특성화 테스트"와 "결함 수정용 실패 테스트" 두 종류다.

### F1. 일일 배치 파이프라인

- **현재**: 크롤→저장→내보내기가 네 벌(`crawl_manager`, `batch_workflow._act`, `scripts/daily_crawl.py`, `main.py`). 프로덕션은 CrawlManager, 지표·인사이트·알림·KG는 미실행. 레코드 매핑도 세 벌(`crawler_agent:113`, `storage_agent:81`, `daily_crawl:244`).
- **문제**: 설계된 파이프라인이 실제로 매일 돌지 않는다. `WorkflowDependencies`는 선언만 있고 인스턴스 0. Container는 DI가 아니라 service locator이며 우회 생성이 8곳.
- **목표**: 파이프라인 1개 + 작업 제어 1개.
  - `CrawlManager` = 작업 제어(락, 상태 파일, 진행률, 재시도)만.
  - `BatchWorkflow` = 단계 실행(crawl→store→metrics→insight→alert→kg→export). Think/Act/Observe 추상은 유지하되 각 단계는 Protocol 주입.
  - 스케줄러 → CrawlManager.start → BatchWorkflow.run. `daily_crawl.py`, `main.py`는 같은 함수 호출.
- **TDD 절차**:
  1. RED(특성화): `BatchWorkflow.run_daily_workflow`를 `tests/unit/application/conftest.py`의 fake(Scraper/Storage)+실제 `MetricCalculator`+fake insight로 실행 → 단계 순서, 최종 status, 저장된 지표 값 고정.
  2. RED(특성화): `CrawlManager._run_crawl` 현재 동작(부분 실패→COMPLETED)을 고정하는 테스트를 먼저 쓰고, D12 수정 시 기대값을 바꾼다.
  3. RED(결함): D5(`asin` 키), D13(완료 시점), D1(`generate_insight`).
  4. REFACTOR: CrawlManager에서 파이프라인 코드를 BatchWorkflow 호출로 교체. `daily_crawl.py`·`main.py`의 복사본 제거.
  5. GREEN 확인: 1번 테스트 + 새 "스케줄러 경로에서 metrics/insight/alert 단계가 호출된다" 테스트.
- **삭제**: `scripts/daily_crawl.py`의 자체 파이프라인(`_save_to_sqlite` 포함), `batch_workflow.py:840-856` 중복 JSON 덤프, `CrawlWorkflow/InsightWorkflow/AlertWorkflow`(호출 0, D16으로 실행 불가) — 단 그 테스트의 fake는 conftest로 보존.

### F2. 챗봇 질의 파이프라인

- **현재**: `process_query_stream`(라이브)과 `QueryGraph`(비스트리밍 전용)가 같은 로직의 복사본 8쌍(`_assess_confidence_level`=`_node_assess_confidence` 등, `query_graph.py:119,318`이 자인). `LLMOrchestrator`(죽음), `QueryProcessor`(죽음+Decision을 dict로 취급해 항상 예외).
- **문제**: 스트림 경로에 캐시·compound 판정 없음, ReAct 미연결(D3), 신뢰도 스케일 혼합(D10), 도구 SKIP을 success=True로 LLM에 전달, 컨텍스트 수집 실패 시 오류 문자열이 "컨텍스트"로 LLM에 들어감.
- **목표**: `QueryGraph` 하나가 스트림·비스트림 모두 처리(노드에 토큰 콜백 주입). brain은 파사드(초기화·DI·스케줄러)만.
- **TDD 절차**:
  1. RED(특성화): `QueryGraph.run(QueryState)`를 fake ContextGatherer/DecisionMaker/ResponsePipeline로 실행 → 인사말 스킵, HIGH→fast 경로, MEDIUM→도구 실행, guard 거절 세 시나리오의 최종 state 고정. 기존 `test_query_graph.py:122-337`(private 노드 직접 호출)은 이걸로 대체.
  2. RED(특성화): `/api/v4/chat/stream` TestClient — SSE 프레이밍과 최종 payload 키 고정.
  3. RED(결함): D3(import 경로), D10(스케일 통일: 0~1로 정규화하고 별도 `confidence_level` 필드), 도구 SKIP 처리.
  4. REFACTOR: `process_query_stream` 본문을 QueryGraph 호출로 교체, brain의 복사본 메서드 8개 삭제.
  5. GREEN 확인: 1·2번 통과 + `test_brain.py`의 private 상태 테스트 삭제(공개 `add_task` 등만 남김).
- **삭제**: `llm_orchestrator.py`, `query_processor.py`, `core/types.py`(brain·tool_coordinator가 types를 쓰도록 바꾼 뒤 중복 정의 삭제 — 둘 중 하나), `QueryRouter`의 미호출 메서드(classify/decompose/dispatch/synthesize) 또는 실제 연결.

### F3. 검색(RAG) 스택

- **현재**: 실제 경로는 레거시 `HybridRetriever.retrieve` 하나. OWL 전략 죽음(D4). BM25는 `DocumentRetriever.search` 안에서 이미 RRF 융합되는데 `HybridRetriever._hybrid_search`가 다시 BM25+RRF 수행. RRF 구현 3개, BM25 검색 2개(정규화 다름), KG 사실→프롬프트 렌더러 3개(500자/400자 절단 다름), Self-RAG 게이트 2개, 의도 분류 호출 3회/질의.
- **문제**: `HybridRetriever`가 게이트·의도·엔티티·KG 조회·추론 컨텍스트·확장·융합·프롬프트 렌더링 9개 책임. 점수 공간 혼합(코사인 유사도·BM25 raw/정규화·리랭커·RRF 첫 리스트 점수). 캐시 키가 임베딩 모델명을 포함하지 않음. 가중치 정의 4곳(`retrieval_weights.json`의 weights 블록은 실제로 죽음).
- **목표**: `rag/` 하위를 책임 단위로 분할하고 각 알고리즘 1개씩.
  - `selfrag_gate.py`, `kg_facts.py`(KG 조회·엣지 정렬·추론 컨텍스트), `query_expansion.py`(+QueryEnhancer 흡수), `fusion/`(RRF 1개, weighted merge 1개), `context_render.py`(ContextBuilder로 수렴), `document_registry.py`, `document_loader.py`, `section_chunker.py`, `vector_index.py`, `bm25_index.py`, `search_cache.py`. `HybridRetriever`·`DocumentRetriever`는 파사드.
- **TDD 절차**:
  1. RED(특성화, 검토 보고의 11개 케이스): 의도 분류 결과·doc_type 필터·가중치, 인사말 스킵 메타데이터, KG 엣지 정렬 규칙(≤12, 우선순위), `_weighted_merge` 순서·개수, `_combine_contexts` 정확한 문자열, 청커 ID 규칙, 검색 캐시 히트 조건, 세 RRF 구현의 동일 입력→동일 순서(수렴 전 등가 증명).
  2. RED(결함): D4(생성자), D14(BM25 id), 캐시 키에 모델명 포함.
  3. REFACTOR: 파일 분할(함수 이동만, 로직 변경 없음) → 특성화 통과 확인 → RRF/BM25/렌더러 수렴(등가 테스트가 지킴).
  4. GREEN: 리랭커·관련성 판정은 flag on 상태의 테스트 추가 후 활성 여부 결정.
- **삭제**: `retriever.search_hybrid/search_hybrid_async`, `retrieve_for_entity`(호출 0), `hybrid_retriever.QueryIntent`·`INTENT_DOC_TYPE_PRIORITY` 복사본, `retrieval_weights.json` weights 블록 또는 `_INTENT_STRATEGY_MAP` 중 하나.

### F4. 추론·규칙·지표 단위

- **현재**: 규칙 시스템 5개 중 1개만 살아 있음. 임계값이 코드 상수(0.15/0.20/0.25/110/150/±10/±5)로 6곳 이상 흩어져 있고 `thresholds.json`·`rules.json`은 읽히지 않음. SoS 퍼센트/분수 혼재(D2), HHI 3스케일, "Dominant" 정의 5개(0.10/0.15/0.20/0.30/Top-10 연속).
- **문제**: 같은 비즈니스 규칙이 다른 임계값·다른 단위로 여러 곳에 있어 어느 것이 정답인지 코드로 판별 불가.
- **목표**:
  - 지표: `MetricCalculator`가 유일한 SoS/HHI/CPI 정의. **단위 규약 명문화**: 내부 표현은 분수(0~1), 표시 계층에서만 ×100. HHI는 0~1.
  - 규칙: `OntologyReasoner + rules/*` 단일 엔진. `StandardConditions`가 `thresholds.json`을 읽어 설정이 실효(load-bearing)하도록.
  - 컨텍스트 빌더 1개: `build_inference_context(metrics) -> dict`(현재 `hybrid_retriever:1072`와 `dashboard_exporter:1440` 두 벌)를 `ontology/inference_context.py`로.
- **TDD 절차**:
  1. RED(특성화, 검토 보고 8케이스): 컨텍스트→발화 규칙명 집합 고정. 특히 `{sos: 2.0}`(퍼센트)에서 `market_dominance_*`가 발화하는 **현재의 잘못된 동작을 먼저 고정**해 두고, D2 수정 커밋에서 기대값을 바꾼다.
  2. RED(결함): D2, D11, D17(`churn_rate None` 예외 없음), D18(하드코딩 컨텍스트 제거).
  3. REFACTOR: 단위 규약 적용(생산자 `calculate_sos`는 분수 반환, 소비자 표시부 ×100), 임계값을 `thresholds.json` 단일 출처로.
  4. GREEN: `thresholds.json`을 tmp로 바꿔 넣으면 발화가 바뀌는 테스트.
- **삭제**: `core/rules_engine.py` + `config/rules.json`, `unified_reasoner.py`, `ontology_knowledge_graph.py`, `kg_iri.py`(사용 계획 없으면), `owl_reasoner.py`(유일 산출인 3단계 SoS 분류를 market_rules로 흡수; 사용자 결정 §6). 생산자 없는 IR·인과 규칙 10개는 생산자를 만들거나 제거.

### F5. 알림

- **현재**: 판정자 3개(`MetricsAgent._check_alerts` 5/10, `AlertManager` 10/2.0 하드코딩, `AlertAgent.process_metrics` ±10) + 발송자 2개(`AlertAgent`, deals 전용 `AlertService`). MetricsAgent 알림은 AlertAgent로 전달되지 않음. `emit_event("crawl_complete")`는 레거시 분기에서만 발화하고 `AlertManager._check_crawl_complete`가 찾는 `result.success`는 아무도 안 만듦.
- **목표**: 판정 1개(`ontology/rules/alert_rules.py`로 흡수, 임계값은 `thresholds.json.ranking`), 발송 1개(`AlertAgent`; AlertService의 Slack/SMTP 채널을 AlertAgent 채널로 편입). 배치 파이프라인 F1의 alert 단계가 유일한 호출자.
- **TDD 절차**: RED(특성화) `AlertAgent.process_metrics` 3케이스(±10, Top-10 진입) 고정 → RED(결함) D6, D11 → REFACTOR 판정 통합 → GREEN "`thresholds.json` rank_drop=7이면 8은 발화, 6은 미발화".
- **삭제**: `core/alert_manager.py`의 판정 로직(이벤트 구독 껍데기만 남기거나 삭제), 문자열 파싱 `rank_delta`.

### F6. API·대시보드 데이터 계약

- **현재**: 대시보드 데이터 로더 5개·형태 3종(D7). 데이터 디렉토리 해석이 `/data` vs `./data`로 갈려 Railway에서 JSON 폴백 불발. LANEIGE 판별 로직 6변형, 날짜 기본값 7곳, 카테고리 이름 맵 3곳, JWT·이메일검증·상태관리자 게터 각 2벌. 엔드포인트 52개 중 UI가 쓰는 것 17개. 분석 보고서 DOCX가 동기/비동기 두 경로에서 다른 문서를 생성. 인증 누락(D8).
- **목표**:
  - **계약 우선**: `api/models.py`를 유일한 스키마로, 라우트 내 Pydantic 정의 삭제. `DashboardData` Pydantic 모델을 정의하고 Exporter 출력·모든 소비자가 이 모델을 사용(D7 해소).
  - 데이터 접근 1개: `application/services/dashboard_data_service.py`(경로 해석 1곳, staleness 메타 1곳).
  - 라우트는 검증+서비스 호출만: `application/services/{export_service, alert_service, analytics_service}.py`. `_get_external_signals` 이동으로 tools→api 역참조 해소.
  - 도메인 유틸 1개: `domain/brand.py`에 `is_target_brand()`, 날짜 범위 파서, 카테고리 이름 맵은 `category_hierarchy.json` 단일 출처.
- **TDD 절차**:
  1. RED(특성화, TestClient 5개 우선): `GET /api/historical`, `/api/sos/category`, `/api/data` 3픽스처(정상/SQLite 폴백/빈 스켈레톤), `/api/category/kpi`, `_get_external_signals` 계층 분류. 현재 응답 키를 스냅샷.
  2. RED(결함): D7(export/alerts/competitors가 dict 형태로 정상 동작), D8(미인증 → 401/403), D19(end_date KST).
  3. REFACTOR: 서비스 추출(함수 이동) → 특성화 통과 → 중복 로직 수렴.
  4. GREEN: 인증 매트릭스 테스트(모든 POST/PUT/DELETE 라우트 × 키 유무).
- **삭제**: UI·외부가 쓰지 않는 엔드포인트 중 `/api/v3/*` 4개, `/api/chat/memory/*`, `/api/export/docx`(비동기 경로로 통일) — 사용자 확인 후. `export_handlers.py`의 "simple docx"(kpis dict 가정으로 항상 실패) 제거.

### F7. 상태·메모리·설정

- **현재**: 시스템 상태 파일 3개(`orchestrator_state.json` 무기록, `system_state.json` 무기록, `crawl_state.json`만 갱신). 대화 메모리 3벌(D15). `thresholds.json`을 11개 모듈이 각자 파싱하고 `AppConfig` 중앙 로더는 6곳만 사용. LLM 온도 env `LLM_TEMPERATURE_CHAT`은 읽는 곳 0. 하드코딩 프롬프트 13파일, registry 사용 2파일.
- **목표**: 상태 1개(`StateManager`, CrawlManager가 기록), 메모리 1개(`memory/context.ContextManager`를 세션 키로 확장, dependencies의 dict 제거), 설정 로더 1개(`AppConfig` 경유, 모듈별 json.load 제거), 프롬프트는 registry 경유.
- **TDD 절차**: RED "크롤 완료 후 `data_freshness`가 fresh" (현재 실패=D9) → REFACTOR 상태 통합 → RED "세션 A의 두 번째 질의 프롬프트에 첫 질의가 포함" → 메모리 통합.
- **삭제**: `core/state.py`(StateManager로 흡수), `memory/conversation_memory.py`·`session_crypto.py`(미사용) 또는 연결, `dependencies.py:70-121`.

### F8. 구조 정리 (v1 항목, 우선순위 재조정)

v1에서 제안한 항목은 유효하나 **F1~F7 뒤로** 미룬다. 유령 경로를 지우면 정리 대상 자체가 줄기 때문이다.

| v1 항목 | 판정 |
|---------|------|
| adapters/·orchestrators/ 빈 패키지 삭제 | 유지. Phase 0에서 즉시 |
| shim 8개 제거, `__init__` 지연 로딩 | 유지. F2~F4가 죽은 모듈을 지운 뒤 |
| 동명 클래스 17쌍 | 유지. 단 `QueryIntent`는 F2에서 의도 체계 자체를 정리하며 처리 |
| brain·hybrid_retriever·retriever 분할 | F2·F3로 흡수(특성화 테스트가 선행) |
| api/dependencies 분할 | F6로 흡수 |
| 문서·루트·scripts·pyproject 정리, ruff UP042 | 유지. Phase 0 |
| 테스트 루트 잔재 9개 | 유지. Phase 0 |

---

## 5. 실행 순서 (Phase 0~5) 와 TDD 절차

| Phase | 내용 | 프로덕션 동작 변화 | 산출물 |
|-------|------|-------------------|--------|
| **0. 안전망** | 의존성 락파일(D21), `.env.test` 생성 + conftest가 실제 `.env` 로드 중단, autouse fixture로 싱글턴(`Container`, `get_brain`, `FeatureFlags`, 모듈 dict) 리셋, `KnowledgeGraph` 테스트 기본 `auto_load=False`, 특성화 테스트 14종(§3·F1~F6의 RED 1단계), 골든셋 record/replay + `EvalRunner(StubJudge)` 회귀 게이트, tests 루트 잔재 정리, 빈 패키지·문서·루트 정리 | 없음 | 테스트만 추가. 이후 모든 Phase의 GREEN 기준 |
| **1. 결함 수정** | D1~D20 각각 RED→최소 수정, 1결함 1커밋. 단위 규약(D2)은 생산자·소비자를 한 커밋에 | 있음(의도된 수정) | 각 커밋에 실패→통과 테스트 |
| **2. 단일 경로** | F1 파이프라인 1개, F2 QueryGraph 1개, F4 규칙 엔진 1개, F6 데이터 로더 1개, F7 상태·메모리·설정 각 1개. 유령 모듈 삭제 | 있음(스케줄 배치가 지표·인사이트·알림까지 실행) | 삭제 목록 커밋, import 그래프 SCC 0 |
| **3. 구조 정리** | F8 전부(shim, `__init__` 지연, 동명 클래스, 문서 드리프트) | 없음 | v1 검증 항목 |
| **4. 분할** | F3 rag 분할, F6 서비스 추출, brain 파사드화 | 없음 | 특성화 테스트 전부 통과 |
| **5. 계약 강화** | Protocol↔구현 정렬(D16), `thresholds.json` 실효, 커버리지 게이트 55%→60%, Protocol 준수 테스트(`isinstance(impl, Proto)` runtime_checkable) | 없음 | CI 게이트 |

**TDD 규칙(모든 Phase 공통)**
1. 리팩토링 커밋은 테스트 파일을 변경하지 않는다(특성화 테스트가 그대로 통과해야 "동작 보존"이 증명됨).
2. 결함 수정 커밋은 실패하는 테스트를 먼저 포함하고, 그 테스트가 특성화 테스트의 기대값을 바꾸면 같은 커밋에서 바꾼다.
3. 삭제 커밋은 "호출처 0" 근거(grep 결과)를 커밋 메시지에 남긴다.
4. 새 테스트는 공개 진입점만 사용(TestClient, `BatchWorkflow.run_daily_workflow`, `QueryGraph.run`, `HybridRetriever.retrieve`, `OntologyReasoner.infer`, `MetricCalculator`). private·patch 문자열 금지.

---

## 6. 사용자 결정이 필요한 항목

| # | 질문 | 권고 |
|---|------|------|
| Q1 | 스케줄 배치가 매일 지표·인사이트·알림·KG 갱신까지 실행해야 하는가 (현재는 안 함) | 예. 문서·README가 그렇게 약속하고 있고, 안 하면 알림·인사이트 기능 전체가 수동 전용이 됨. LLM 비용은 인사이트 1회/일 |
| Q2 | OWL 추론기(owlready2·Pellet/Java)를 유지할지 | 제거. 실제 산출이 3단계 SoS 분류뿐이고 규칙으로 대체 가능. 포트폴리오 서사상 필요하면 "실험 모듈"로 격리하고 프로덕션 경로에서 분리 |
| Q3 | ReAct 에이전트를 연결할지 제거할지 | 연결(import 경로 1줄 수정) + feature flag 기본 off. 골든셋으로 효과 측정 후 on |
| Q4 | UI 미사용 엔드포인트 35개 중 삭제 범위 | `/api/v3/*`, `/api/chat/memory/*`, 동기 `/api/export/docx`는 삭제. `/api/v4/brain/*`, `/api/crawl/*`, `/api/sync/*`는 운영용이므로 유지 |
| Q5 | `kg_iri` 마이그레이션을 진행할지 폐기할지 | 폐기. 실제 문제는 IRI가 아니라 브랜드 대소문자 불일치이며 그것부터 고침 |

---

## 7. 하지 않을 것

- `src/agents`·`src/rag`를 `adapters/`로 물리 이동, `core`를 `application/orchestrators`로 이동 (v1과 동일).
- Think→Act→Observe 추상 자체의 제거: 단계 주입 구조로는 쓸 만하다. 복사본만 제거.
- `period_insight_agent.py`·`owl_reasoner.py` 내부 분할(Q2에서 제거되면 무의미).
- Phase 1 이전의 어떤 파일 이동도 하지 않는다.

---

## 8. 부록: 실측 수치

| 항목 | 값 |
|------|-----|
| src Python | 210파일 / 74,681줄 |
| tests Python | 184파일 (pytest 수집 대상 test_*.py 161) |
| 정적 import 순환 SCC | 1개(5모듈) |
| 레이어 역방향 import | tools→api 2건 |
| 호환 shim | 8파일 |
| 프로덕션 미참조 모듈 | 8 + 검토로 추가 확인된 유령(llm_orchestrator, query_processor, unified_reasoner, ontology_knowledge_graph, rules_engine, Crawl/Insight/Alert 소형 워크플로우, WorkflowDependencies) |
| 동명 클래스 | 17쌍 |
| 테스트 없는 src 모듈 | 39개 |
| `print(` in src | 63 (dashboard_exporter 11, data_integrity_checker 14 등) |
| 로깅 | `logging.getLogger` 94파일 vs `AgentLogger` 11파일 |
| ruff | UP042 16건 |

> 한 줄 결론: 이 코드베이스에서 폴더를 옮기는 것은 마지막 일이다. 먼저 특성화 테스트로 현재 동작을 고정하고(Phase 0), 확인된 결함 20건을 테스트 선행으로 고치고(Phase 1), 배치·챗봇·규칙·데이터 각 축에서 실행 경로를 하나로 정해 유령을 지운 뒤(Phase 2), 그제야 shim 제거와 파일 분할(Phase 3~4)을 한다.
