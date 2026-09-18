# Future Work

> Phase 1-6 리팩토링 완료 후 남은 작업 목록
> 최종 업데이트: 2026-02-16

---

## 1. 커버리지 개선 (43% → 60% 목표)

| 영역 | 현재 상태 | 우선순위 |
|------|----------|---------|
| `src/tools/` 모듈 테스트 | ~10% | HIGH |
| `src/agents/` 통합 테스트 | 일부만 | HIGH |
| `src/core/brain.py` 통합 시나리오 | 미작성 | MEDIUM |
| `src/rag/` 검색 파이프라인 E2E | 미작성 | MEDIUM |

---

## 2. Config 정리 (별도 PR 권장)

- [ ] `competitors.json` 미사용 dead config → 삭제 검토
- [ ] `thresholds.json` 분리 (system settings / category URLs) — 20+ 소비자로 HIGH risk
- [ ] Pydantic 스키마 기반 config 검증 추가

---

## 3. 남은 직접 import (DI 전환 후보)

| 파일 | 직접 import 대상 |
|------|-----------------|
| `hybrid_insight_agent.py` | ExternalSignalCollector, MarketIntelligenceEngine |
| `period_insight_agent.py` | PeriodAnalyzer, InsightFormatter |
| `api/routes/deals.py` | AlertAgent |
| `api/routes/signals.py` | ExternalSignalCollector |

---

## 4. 기술 부채

- [ ] `dashboard_api.py` `@app.on_event("startup")` → lifespan event handler 전환
- [ ] `knowledge_graph.json` 동시 쓰기 보호 (flaky test 원인)
- [ ] `TestResult` 클래스 이름 충돌 해결 (pytest 수집 경고)

---

## 5. 순환 의존성 해소 (23 cycles → 0 목표)

### P1: core ↔ agents
- brain.py → AgentProtocol (interface) 사용으로 전환
- 현재: brain.py가 concrete agent class 직접 import

### P2: tools ↔ agents
- export_handlers.py → PeriodInsightAgent 직접 import 제거
- Protocol 기반 DI로 전환

### P3: api ↔ tools/core
- Route handlers → service layer 분리

---

## 6. Application Layer 강화

현재 `src/application/`은 120 LOC (거의 비어있음). 실제 비즈니스 로직이 Layer 4에 산재.

- [ ] CrawlWorkflow use case 구현
- [ ] ChatWorkflow use case 구현
- [ ] InsightWorkflow use case 구현
- [ ] API routes가 application use cases 호출하도록 전환

---

## 7. 보안 개선 사항

SECURITY_AUDIT_REPORT.md (2026-01-28) 기준 미해결 항목:

| ID | Severity | 내용 |
|----|----------|------|
| VULN-005 | HIGH | Docker non-root user 추가 |
| VULN-006 | HIGH | API key timing-safe 비교 (hmac.compare_digest) |
| VULN-007 | HIGH | Chat endpoint 인증 추가 |
| VULN-008 | HIGH | Prompt injection 방어 강화 |
| VULN-011 | MEDIUM | Security headers (CSP, X-Frame-Options) |
| VULN-012 | MEDIUM | Session ID 암호학적 랜덤 생성 |

---

## 8. God Objects 분할 (13개 남음)

1,000줄 이상 파일 중 추가 분할 후보:
- `dashboard_api.py` (3,236줄) — 추가 라우트 분리
- `src/ontology/business_rules.py` (1,540줄) — 규칙 카테고리별 분리
- `src/ontology/knowledge_graph.py` (1,514줄) — CRUD/Query 분리
- `src/agents/hybrid_chatbot_agent.py` (1,353줄)
- `src/tools/scrapers/amazon_scraper.py` (1,321줄)

---

*Sources: TODO.md (Phase 4-6), REFACTOR_PLAN.md, HANDOFF.md §4 Future Work*

---

## 9. 2026-08-31 리팩토링에서 남긴 항목

`docs/plans/refactoring-plan-2026-08-31.md` 실행 중 "명시적 비범위"이거나
의도적으로 뒤로 미룬 항목들.

### 9.1 계획서의 명시적 비범위 (이번에 손대지 않음)
- BM25/리랭커/검색 품질 튜닝 — 별도 eval 사이클 영역
- `brand_resolver` 웹검색 실구현 — 이번에는 이름 정정(`_resolve_from_known_patterns`)과
  한계 문서화, Unknown 비율 로그만 추가. 하드코딩 패턴 14개로는 Unknown 28%를 해소할 수 없다.
- 골든셋 문항 추가 / IR 도메인 확장
- 대시보드 디자인 변경, 모바일 대응

### 9.2 죽은 엔드포인트 (삭제하지 않고 기록만 — 대시보드 외 소비자 가능성)
대시보드가 호출하지 않는 라우트. 삭제 전 외부 소비자 확인 필요.
- `GET /api/v3/alert-settings`, `POST /api/v3/alert-settings`,
  `POST /api/v3/alert-settings/revoke`, `GET /api/v3/alerts`
  → v4로 대체됨. 코드에 `deprecated=True` + docstring 표시만 해둠.
  계획서 §6.4는 삭제를 제안했으나 §6.7의 "엔드포인트는 삭제하지 말 것"을 따랐다.
  deprecation 기간을 정한 뒤 별도 커밋으로 제거할 것.

### 9.3 미구현 지표 (현재 `None`으로 방출)
상수를 실측처럼 내보내지 않도록 `None`으로 바꿨다. 실제 계산이 필요하다.
- `streak_days` (Top N 연속 체류일) — `dashboard_exporter._build_reasoning_context`
  및 대시보드 "순위고착도" KPI. 온톨로지 growth_rules가 이 값을 조건으로 쓴다.
- `rank_change_7d`, `rating_gap` (같은 함수)
- `new_competitors` (신규 경쟁자 수) — 시계열 비교 필요.
  현재는 `brand_count`(카테고리 내 고유 브랜드 수)만 제공하며 대시보드 라벨도 교정했다.
- `market_metrics.churn_rate` — 백필 스크립트에서도 `None` (전일 비교 필요)

### 9.4 Sheets 지표 백업
`StorageAgent.save_metrics()`를 삭제했다. 호출처가 0건이었고
`BrandMetrics`/`ProductMetrics` 엔티티에 없는 필드(`brand_name`, `share_of_shelf`,
`avg_rank`, `top10_count`)를 참조해 실제 엔티티로는 동작하지 않았다.
Sheets에 지표 백업이 필요하면 엔티티 필드에 맞춰 새로 구현할 것.
SQLite 영속화는 `BatchWorkflow`의 `STORE_METRICS` 스텝이 담당한다.

### 9.5 운영 확인 필요
- Railway 배포 시 `AUTO_START_SCHEDULER=true` 확인.
  `config_manager.py`의 기본값이 `False`라 스케줄 태스크(일일 크롤, 정합성 검사)가
  뜨지 않는다.
- `/api/health/integrity`가 로컬에서 CRITICAL을 반환한다
  (Sheets 0건 vs SQLite 35,328건). 운영 환경에서 Sheets 자격증명이 있는 상태로 재확인 필요.

### 9.6 `applyProductDateRange()` 호출처 없음
실제 차트 로더에 연결했으나 현재 이 함수를 부르는 UI 컨트롤이 없다
(`productStartDate`가 hidden input). Product View 기간 선택 UI를 되살릴지 결정 필요.

### 9.7 eval 프레임워크 결함 (Phase 4 검증 중 발견)
`docs/experiments/refactor_phase4_2026-08-31.md` 참조.
- **LLM 호출 타임아웃 없음** — `eval.cli run`이 135/172에서 53분 무기한 정지
  (프로세스 생존, TCP 4개 보유, CPU 0%). 판정 결과가 실행 끝에만 기록돼
  중단 시 부분 결과도 유실된다. 문항별 타임아웃 + 증분 저장 필요.
- **리포트 `metadata.requires_kg` 직렬화 유실** — 데이터셋이 `false`인 문항이
  리포트에는 `true`로 기록된다. `check_gating`이 이 값으로 L2/L3 검사를 갈라
  적용하므로, 리포트만으로는 게이팅을 재현할 수 없다.
- **기준선/현재 L1 `concept_map_f1` 임계값 불일치** — v8.1 기준선의 4문항
  (lg059·lg080·lg156·lg179)이 현재 로직으로는 통과하지 않는다.
  baseline 저장 시 임계값을 함께 고정할 것.

### 9.8 위험 지점 보완 작업(2026-09-17) 중 발견한 범위 밖 항목
- (해소 — 2026-09-18 트랙 0-E `c9db4e9`가 `acompletion`을 가짜로 둠) `tests/unit/core/test_react_agent.py::test_react_run`이 `acompletion`을 가짜로 두지 않아 전체 테스트마다 실제 OpenAI를 호출한다.
- (해소 — 2026-09-18 트랙 0-E `c9db4e9`가 가짜 시계(`_patch_clock`)로 결정적 재현으로 바꿈) `tests/unit/core/test_cache.py::test_cleanup_expired_removes_old`가 TTL 0초·동일 시각 비교에 의존해 간헐 실패한다.
- (해소 — 2026-09-18 증거 카드 재작업. `HybridRetriever._combine_contexts`가 이제 `render_for_prompt(context.prompt_evidence)`이고 `metric_facts`가 metric 카드로 변환돼 실린다) v4 Brain 경로의 답변 프롬프트(`HybridRetriever._combine_contexts` → `ResponsePipeline`)에는 크롤 DB 수치 사실(`metric_facts`, `3fce8e8`)이 실리지 않는다. v1 `ContextBuilder`만 렌더링한다. 평가 트레이스(judge 컨텍스트)에는 두 경로 모두 들어간다.
- `eval.cli ablation`은 `run`과 달리 데이터 시점(`AMORE_DATA_AS_OF`)을 고정하지 않는다.
- (해소 — 2026-09-18 트랙 0-C `f7d9bcd`가 litellm `model_cost` 우선 조회로 바꿈, `eval/cost_tracker.py`에 gpt-4.1-mini $0.40/$1.60 확인됨) `eval/cost_tracker.py:34`의 gpt-4.1-mini 단가가 $0.15/$0.60 per 1M(공시가 $0.40/$1.60)이라 리포트 비용이 약 1/2.67로 과소 집계된다. v1.0~v9.1 기준선의 비용 수치도 같다. **단, v1.0~v9.1 기준선 리포트 자체의 저장된 수치는 재계산하지 않았다** — `docs/experiments/evidence_pipeline_2026-09.md` 0단계의 "비용 재계산" 표(×2.667) 참고.
- (해소 — 2026-09-18 트랙 0-B `024ad89`·`f083519`가 핵심 검색 실패를 `context.metadata["retrieval_error"]`에 남기도록 바꿈) `HybridRetriever.retrieve`가 검색 예외를 삼키고 빈 컨텍스트로 답변을 계속 만들어, 평가 하니스가 인프라 실패로 분류하지 못한다(2026-09-17 Chroma 오류 실행 2건이 조용히 채점됨).
- (해소 — 2026-09-18 트랙 0-A가 `DocumentRetriever.initialize()`를 읽기 전용(`get_collection`만)으로 바꾸고 색인은 `python -m src.rag.build_index` 전용 CLI로 분리) 평가 프로세스 여러 개를 같은 `data/chroma`로 동시에 띄우면 한쪽의 색인 변경이 다른 쪽 검색을 깨뜨린다. 실행별 `CHROMA_PERSIST_DIR` 복사본이나 읽기 전용 모드가 필요하다.
- (해소 — 2026-09-18 트랙 4-C가 OWL 검색 전략을 삭제) `OWLRetrievalStrategy` 생성자 기본값이 시맨틱 청킹 `DocumentRetriever`를 새로 만들어 초기화 때 공유 컬렉션에 다른 청크를 추가 색인한다(`create_owl_strategy`는 `0669b75`에서 공유 검색기를 넘기게 했지만 생성자 기본값은 그대로).
- (해소 — 2026-09-18 트랙 4-C가 OWL 검색 전략을 삭제. 엔티티 신호는 색인 태그 기반 재정렬 가산점으로 대체(4-B)) `OWLRetrievalStrategy._matches_filters`가 엔티티 링커의 Chroma where 형식 필터(`$or`, brand·category 키)를 처리하지 못하고 문서 메타데이터에도 그 키가 없어, 엔티티가 연결된 질의는 문서를 0건 가져온다(130문항 중 124문항).
- (해소 — 2026-09-18 규칙이 증거 카드를 입력으로 받도록 재작업(`src/ontology/rule_contracts.py`) 이후 233문항 시험지 3단계 측정에서 규칙 발화 문항 비율 0.568, 규칙 정답 일치율 0.779로 나옴. 근거: `docs/experiments/evidence_pipeline_2026-09.md` 3단계) 규칙 추론(`OntologyReasoner`)이 v4 평가 130문항·전 실행에서 추론 0건이다. `_build_inference_context`가 읽는 키가 운영 데이터와 맞지 않는다(근거 문서 §5.3).
- (부분 해소 — 2026-09-18 트랙 5-B가 신뢰도 점수를 증거 적합도 기반으로 교체해 검색 점수 분포 하나가 HIGH를 막던 구조를 없앴다(`src/core/confidence.py`, 커밋 `bb0627f`·`450e9d1`·`5fb6030`). **다만 이 재작업 이후 172문항 재평가는 아직 없다** — `docs/experiments/evidence_pipeline_2026-09.md`는 트랙 4까지만 기록돼 있어, HIGH 편중이 실제로 줄었는지는 미측정이다) 신뢰도 점수가 v4 평가의 172/172문항을 HIGH로 분류해 DecisionMaker(LLM 도구 선택)와 ReAct 분기가 평가에서 한 번도 실행되지 않는다. 임계값·점수 구성을 재검토해야 이 경로들을 측정할 수 있다.

### 9.9 증거 카드·규칙 추론·ReAct 통합 작업(2026-09-17~) 중 발견한 범위 밖 항목
> 출처: `eval_output/evidence-2026-09/notes/0c_recalc.md`(트랙별 보고), `docs/experiments/evidence_pipeline_2026-09.md`. 이 작업에서 고치지 않았다.

**측정·비용**
- `eval/judge/llm.py`(`MODEL_PRICING`, 135~140·337행)에 옛 단가가 남아 있다(`get_usage` 호출처 0이라 현재 리포트에는 영향 없음).
- L2 임베딩(`src/rag/retriever.py`의 openai 직접 호출)과 v4 질의 확장(`DocumentRetriever.expand_query`)의 토큰·비용이 어느 리포트에도 집계되지 않는다(장부는 ×1.1로 보정).
- judge 호출 시간 초과(60초×3회)가 동시 실행 중 특정 시간대에 몰려 문항이 채점에서 빠진다(2단계 run2 `lg161`, run3 `lg155`). 재시도 간격(backoff) 없음.
- 골든 문서의 CPI 정의(<1.0 비율)와 코드 CPI(100 기준)가 다르다(lg043, lg056).
- (해소 — 2026-09-18 온톨로지 작업 O0·O7) 골든 엣지는 `ownedBy`인데 KG 술어는 `ownedByGroup`이라 엣지 Recall이 낮게 나올 수 있다(미검증). → 확인했다. 술어를 정식 이름으로 맞춘 canonical recall 필드를 따로 두었고(`eval/metrics/l3_kg.py`), 기존 raw 필드는 그대로다. 54문항 `ownedByGroup` canonical recall은 플래그 OFF 0.424, ON 0.909다. 남은 골드 레거시 표기(lg102·lg158)는 9.10에 적었다.

**데이터 파이프라인**
- 2026-08-31 지표 테이블(`brand_metrics`·`market_metrics`, 08-30 18:06 UTC 계산)이 같은 날짜 `raw_data`(08-31 13:03 UTC 교체)와 다른 크롤 상태로 계산됐다(예: lip_care HHI 0.0681 vs raw 0.0637, lip_makeup 21/21 브랜드 SoS 불일치). 지표 재계산 시점 결함.
- `raw_data.brand` 부분 문자열 오귀속 24/356행·14쌍(lip_care "Hera" ← "Therapy", "CHI" ← KimChiChic, "OPI", "Verb", "elf", skin_care "Fresh" 등). 골든 lip_care 순위·HHI와 KG `competesWith`(system 출처)에 유입.
- `raw_data.price` 2025-12~2026-01 714행이 1,411~86,377(KRW로 추정)인데 `price_currency=USD`.
- `raw_data.reviews_count` 8,769행이 빈 문자열(2025-12-16~2026-01-19).
- `badge` 필드에 평점 문자열("4.6")이 들어 있다 — 스크레이퍼 필드 매핑 결함 추정.
- 2026-08-31 lip_care SoS가 정수값(제품 수 기반)이다. 다른 날짜·카테고리와 계산 방식 일관성 확인 필요.

**KG·온톨로지**
- KG 수치 엣지 359개(`hasSoS` 169, price position 124, `hasHHI` 66)가 전부 날짜가 없다. lip_care `hasHHI`는 값이 15개이고 0~10000 스케일이 섞여 있다. 검색 증거에서는 제외했지만(E2) 정리·버전 부여는 하지 않았다. [2026-09-18 온톨로지 O5] 평가 스냅샷 마이그레이션 후에도 333건(`hasSoS` 151·`hasPricePosition` 116·`hasHHI` 66)이 날짜 없이 남는다. 날짜를 지어내지 않기로 했다(OA-8). 9.10 참고.
- (부분 해소 — 2026-09-18 온톨로지 O3·O5) KG 주어 대소문자가 술어마다 다르다(`competesWith` 주어 `laneige`, `ownedByGroup` 주어 `LANEIGE`). → 질의 경로는 ON에서 읽을 때 정식화한다(O3). 평가 스냅샷 사본은 마이그레이션으로 비정식 브랜드 1,029 → 0이 됐다(O5). **운영 KG에는 적용하지 않았다**(OA-11, 효과 없음). 매일 크롤은 기본 `warn`에서 예전 표기를 계속 쓴다.
- (부분 해소 — 2026-09-18 온톨로지 O5) KG `TATA HARPER acquiredIn='True'`. → 마이그레이션이 이 트리플을 지우고, `kg.write_validation=enforce`는 막는다. 원인인 `kg_updater` 버그는 그대로다(9.10).
- `KnowledgeGraph()` 기본 생성이 로드 중 자동 저장으로 KG 파일을 다시 쓴다(0-F의 근본 원인). 테스트는 `persist_path`·`auto_save=False`로 격리했지만 생성자 동작 자체는 그대로다.
- 규칙 37개 중 입력을 공급할 수 없는 규칙: sentiment 8개 전부, IR 5개, 이력(기간 비교)이 필요한 규칙.

**코드**
- `src/rag/context_builder.py:599` `churn_rate`를 `:.1f%`로 표시한다(정의는 0~1). 데이터가 전부 NULL이라 아직 드러나지 않음.
- `MetricFactsProvider`가 제품명을 60자로 자르고 ASIN을 넘기지 않아 제품 카드가 KG 제품(ASIN)과 연결되지 않는다.
- `HybridRetriever._query_knowledge_graph`의 `trend_keywords`가 주어(브랜드/MARKET)를 보존하지 않는다.
- 1-A가 시험지에서 뺀 23문항은 `gold_source=domain_expectation`(추정 골드)이라 정답 채점이 불가하다 — 골드 보강 필요.

#### 9.9.1 트랙 3-B·2-D·4-B에서 추가로 발견한 항목 (2026-09-18)
- (브랜드 인식은 해소 — 2026-09-18 온톨로지 O2, 플래그 `ontology.use_class_reasoning` ON에서만. nivea 과다 추출은 남음) 엔티티 추출기가 골든 문항의 브랜드 일부를 인식하지 못한다(IT Cosmetics, Jouer, Almay, Charlotte Tilbury, COVERGIRL). → 등록부 사전으로 인식한다. 골든 273개 질문에서 새 오탐 0건, 오프라인 규칙 일치율 25/32 → 29/32(골드 엔티티 상한과 같음), LLM 측정 rule 42문항 일치율 0.781 → 0.906(§O7). Lip Care 질문에서 nivea를 과다 추출한다(KG 제품 슬러그 역링크 경로, O2와 무관). 규칙 판단 정답 일치율이 오프라인 측정에서 25/32인데, 골드 브랜드를 직접 넣으면 29/32로 오른다 — 즉 남은 격차의 대부분이 이 결함이다.
- `HybridRetriever.retrieve`가 마지막에 `context.metadata`를 새로 할당해 `weighted_scores`·`fusion` 메타데이터를 지운다.
- 프롬프트에 싣는 추론 카드는 신뢰도 상위 5개라, 질문이 겨냥한 규칙이 발화했는데도 프롬프트에서 빠질 수 있다.
- `dashboard_exporter`는 여전히 `reasoner.infer`와 대시보드 JSON 컨텍스트를 쓴다(v4 검색 경로만 카드 입력으로 바뀜).
- ReAct 경로(`brain._process_with_react`)와 v1 경로(`HybridChatbotAgent`)는 `ResponsePipeline`을 거치지 않아 답변 수치 검증(2-D)이 적용되지 않는다.
- 스트림 `done` 이벤트에 `response.metadata`가 실리지 않아 대시보드에서는 `numeric_verification`을 볼 수 없다.
- 모델이 카드 값으로 계산한 수(`0.5%p`, `2배`, `3계단`)는 카드에 없어 `mismatch`로 분류된다 — 검증기 enforce를 기본값으로 올리기 전에 annotate 비율을 봐야 한다.
- `DocumentRetriever._search_cache`·`_cache_timestamps`가 클래스 속성이라 persist_dir·코퍼스가 달라도 프로세스 전역으로 공유된다(평가·테스트 오염 위험).
- `DocumentRetriever.search_bm25`는 `doc_type_filter`를 적용하지 않아 인텐트 문서유형 필터가 dense 검색에만 걸린다. `reciprocal_rank_fusion` 결과에는 최상위 `id`가 없어 BM25 출처 결과는 metadata로만 식별된다.
- `scripts/start.py`는 `build_index()`만 부르므로 이미 존재하는 미태깅 볼륨 색인은 배포해도 태그가 붙지 않는다(운영에서 `--retag` 1회 필요). `--retag`는 `--prune`과 함께 줘도 prune을 하지 않는다.
- 로컬 `ruff format`과 pre-commit 훅의 ruff가 일부 파일에서 서로 다른 스타일로 고친다.
- (트랙 4-C, 2026-09-18) `Container.get_unified_retriever`는 OWL 전략 주입이 존재 이유였는데 전략 삭제 후에도 서비스 호출처가 0건이다(사용하는 곳은 없고 `HybridRetriever` 싱글톤만 만든다). 삭제 여부는 Container 전반 정리와 함께 판단 필요.
- (해소 — 2026-09-18 문서 갱신 트랙) OWL 검색 전략·`llm_orchestrator`·`query_processor`·`unified_reasoner`·SPARQL 계층 삭제 후 남은 문서 표기: `CLAUDE.md`(모듈 표·디렉터리 트리), `README.md`(rdflib SPARQL·`use_owl_strategy`), `src/core/AGENTS.md`(QueryProcessor), `AGENTS.md`(루트), `src/rag/AGENTS.md`, `docs/architecture.md`·`docs/SYSTEM_ARCHITECTURE.md`를 코드 기준으로 고쳤다. **확인 결과**: `src/core/tools.py`는 이미 삭제돼 존재하지 않고, `src/core/confidence.py`에는 `llm_orchestrator.py`를 언급하는 주석이 없었다(grep 0건) — 이 항목의 원래 서술은 그 시점에도 부정확했을 수 있다. `src/rag/AGENTS.md`는 이미 갱신돼 있어 손대지 않았다.
- (트랙 4-C, 2026-09-18) SPARQL 계층 삭제로 `rdflib`를 import하는 `src/` 코드가 없어졌다(`requirements.txt`의 의존성 정리 후보).
- (2026-09-18, 문서 갱신 트랙 이어서) `docs/architecture.md`·`docs/CORE_ARCHITECTURE_DEEP_DIVE.md`·`docs/guides/react_agent_guide.md`·`docs/SYSTEM_ARCHITECTURE.md`·`docs/architecture/LLM_ORCHESTRATOR_DESIGN.md`(삭제 배너 추가)의 삭제 모듈·도구·활성화 서술만 고쳤다. 이전부터 낡은 서술이 남은 부분: `docs/architecture.md`의 신뢰도 점수 공식(2.5절, OWL 가산점 방식 — 실제로는 `confidence.py`가 엔티티 충족도 0.60+카드 종류 충족 0.40 적합도 식으로 재작성됨, 트랙 5-B)과 DecisionMaker MODE_PROMPTS 설명(2.6절, 실제로는 네이티브 function calling), `docs/CORE_ARCHITECTURE_DEEP_DIVE.md`의 TrueHybridRetriever 절(5.4, 그런 파일명 없음)과 같은 신뢰도 공식 서술, `docs/guides/react_agent_guide.md`의 모델 기본값·성능 최적화 예시(`gpt-4o-mini`/`gpt-4o` 등, 코드 기준 미확인) — 삭제 모듈 참조만 고침(2026-09-18).

#### 9.9.2 6단계(ReAct 비교·E8 판정)에서 발견한 항목 (2026-09-18)
> 근거: `docs/experiments/evidence_pipeline_2026-09.md` 6단계, 결정 S6-2~S6-5. 6단계에서는 고치지 않았다.

- **ReAct 진입이 신뢰도 관문 뒤에 있다.** `QueryGraph._route_after_confidence`가 HIGH를 홉 판정보다 먼저 봐서, 라우터가 2홉 이상으로 본 30문항 중 25문항이 ReAct를 건너뛴다. 신뢰도 변별력은 증거 선별(질의마다 카드 ~70장, 233문항 중 184문항이 충족도 만점)을 좁히기 전에는 오르지 않는다(5단계 게이트 미충족 항목과 같은 원인). 측정용 우회 플래그 `agents.react_bypass_confidence`(기본 OFF)가 있다.
- **ReAct 토큰 예산 12,000이 실제 사용량보다 작다.** 그림자 88회 평균 20,438토큰(85/88 초과), (d) 90회 중 73회가 예산 소진 후 강제 답변. 원인은 컨텍스트 요약이 매 단계 프롬프트에 다시 실려 한 단계가 ~7천 토큰이 되는 것. 예산을 올리기 전에 요약 크기부터 줄일 것.
- **ReAct 답은 카드 인용이 자주 빠지고(33/90) 수치 검증을 받지 않는다.** ReAct 경로(지금은 `QueryGraph._node_react`)는 `ResponsePipeline`을 거치지 않는다(9.9.1의 같은 항목이 경로 이름만 바뀐 채 유효).
- **ReAct 도구 선택**: "같은 그룹 브랜드" 질문(mh005·mh006·lg158)에서 `kg_neighbors`만 반복하다 예산에 닿는다. 그룹 소속 브랜드를 한 번에 펼치는 도구 인자나 예시가 없다.
- **그림자 모드가 동기로 돈다.** 답변 뒤에 같은 요청 안에서 ReAct를 기다려 지연 +67%. 운영에서 쓰려면 백그라운드 실행이 필요하다.
- **수치 검증기 인용 파싱 결함**: `[M-a], [M-b]`처럼 쉼표로 떨어진 인용 괄호는 첫 id만 인용으로 연결한다(`[M-a][M-b]`, `[M-a, M-b]`는 정상). mismatch의 25~43%가 이것이다. enforce의 선행 조건.
- **수치 검증기 계산값 처리**: 100 기준 지수의 환산(CPI 111.1 → "11.1% 높음"), 카드 3개 이상의 합, 반올림("3만 7천여"), 규칙 임계값("HHI < 0.15"), 제품명 속 숫자("96%")가 mismatch로 잡힌다. enforce의 선행 조건.
- **평가 측정**: 3회 측정에서 토큰 F1 노이즈 기준 0.01이 A/A 폭(+0.011 전체, +0.024 multihop)보다 좁다 — 분산이 큰 소수 문항 때문. 그림자 토큰(`route_trace.react_shadow.token_usage`)은 평가 리포트 비용 집계에 연결돼 있지 않다(리포트의 l5 비용에 그림자 비용이 섞인다). multihop·relation 시험지에는 `rule_gold`가 없어 규칙 일치율을 잴 수 없다.

### 9.10 온톨로지 작동 작업(O0~O7, 2026-09-18) 중 남긴 항목
> 근거: `docs/plans/ontology-activation-plan-2026-09-18.md`, 결정표 `docs/plans/ontology-activation-decisions-2026-09.md`, 실험 기록 `docs/experiments/ontology_activation_2026-09.md`. 트랙 O6에서 처음 적었고, §7 마무리(2026-09-18)에서 O7 결과로 보탰다.

**검토 보고서(`docs/analysis/ontology-review-2026-09-18.md`) 항목 중 해소된 것**

- (해소 — O6) OWL 모듈(`owl_reasoner.py`·`ontology_knowledge_graph.py`·`cosmetics_ontology.owl`·`scripts/migrate_kg_to_ontology.py`) 삭제, owlready2는 `requirements-dev.txt`로 이동. 호출처 확인표는 실험 기록 §O6-1.
- (부분 해소 — O1·O2) 브랜드 어휘가 5곳 이상에 흩어져 있던 문제(검토 보고서 §3.1). 브랜드 등록부 `config/ontology/brands.json`과 로더 `src/ontology/ontology.py`가 온톨로지의 단일 원본이 됐다(Pellet 교차 검증 불일치 0, 결정 OA-5). 연결기는 플래그 ON에서 등록부 사전을 **더해** 쓴다. 기존 `config/entities.json`·연결기 사전은 그대로 남아 있다("기존 인식은 잃지 않는다" 원칙).
- (해소 — O3, 플래그 ON) 그룹을 소속 브랜드로 전개하지 않던 문제, 세그먼트·원산지 술어가 `priority_preds`에서 버려지던 문제(§3.1). 54문항 canonical recall: `ownedByGroup` 0.424 → 0.909, `hasSegment` 0 → 0.429, `originatesFrom` 0 → 0.500, `siblingBrand` 0 → 1.000(§O7).
- (해소 — O3, 플래그 ON) 자매 브랜드 부정 판정(rl015·rl016, §3.4). 등록부를 닫힌 세계로 보고 `notOwnedByGroup`·`notSiblingBrand` 카드를 낸다. negative 부분집합 종합 0.744 → 0.849.
- (해소 — O0) L4 지표가 0.0/1.0으로 고정돼 있던 문제, L3 recall이 골드 엣지 없는 문항 때문에 부풀던 문제(§4). 새 필드(`rule_constraint_violation_rate`·`typed_consistency_rate`·`kg_edge_recall_gold_only`·canonical 필드)를 더했다. 기존 필드와 게이트는 그대로다.
- (해소 — O4) 소유 검증 규칙이 KG에 있는 원산지·세그먼트·인수 정보를 "입력 없음"으로 보던 문제(§3.3). 등록부 카드에서 읽는다(플래그 ON).
- (해소 — O5 이후) 실험 기록 §O5 4절 3항: `hasPricePosition`이 이제 `evidence_adapters.KG_NUMERIC_PREDICATES`와 `hybrid_retriever`의 정식 술어 우선순위에 모두 들어 있다(코드 확인).
- (해소 — `3421ef8`) ON에서 새로 발화한 가격 규칙의 `related_entities`에 빈 문자열이 들어가 L4 규칙 위반율이 0.116 → 0.228로 오르던 결함(OA-13). 오프라인 재채점 0.228 → 0.0. 수리 뒤 코드로 LLM 재측정은 하지 않았다.

**새로 남긴 항목 (O7, 2026-09-18)**

- **프롬프트 카드 수가 한계선 ~70장에 닿았다(OA-12).** 플래그 ON에서 54문항 평균 58.1 → 69.9장, 70장 초과 문항 25~26 → 32~33개(54문항), 102 → 131개(233문항). segment 부분집합은 34.5 → 72.2장. 파이프라인 비용 +9%, 프롬프트 토큰 +417/문항(54). 카드 상한·선별(증거 선별 좁히기, 9.9.2 첫 항목과 같은 원인)을 조정해야 한다.
- **골드 어휘 불일치 (기록만, 골드는 고치지 않음 — 사용자 결정 필요).**
  - lg102·lg158 골드가 레거시 술어 `ownedBy`를 쓴다. 플래그 ON은 정식 이름 `ownedByGroup`으로 내므로 raw L3에서 잃는다(canonical은 잃지 않는다). 종합 점수는 raw L3를 쓴다.
  - rl011 골드가 `cosrx -originatesFrom-> korea`인데 등록부 국가 id는 `south_korea`다. canonical에서도 맞지 않는다(나라 정규화 함수 없음). 그래서 `originatesFrom` recall이 1/2에서 멈춘다.
- **lg158: 그룹과 소속 브랜드를 함께 언급하면 전개하지 않는다.** "LANEIGE 모회사와 해당 기업의 다른 브랜드" 질문은 LANEIGE가 함께 나와 "그룹만 언급" 조건(`ontology_context`)을 채우지 못한다. 답변이 "다른 브랜드 현황은 데이터에 없음"이라고 한다. 54문항 중 ON에서 3회 모두 떨어진 유일한 문항이다((b) −0.133).
- **`kg.write_validation=enforce` 선행 조건(OA-7·OA-8).** ① `dashboard_exporter.py`와 `scripts/enrich_kg_from_crawl.py`가 `enrich_and_store(...)`에 `as_of`를 넘겨야 한다(아래 기존 항목). ② 평가 스냅샷 기준 날짜 없는 수치 엣지 333건(`hasSoS` 151·`hasPricePosition` 116·`hasHHI` 66)은 크롤 날짜와 함께 다시 쓰거나 그날 DB에서 다시 계산해야 한다. ③ 운영 KG 마이그레이션을 적용한다면 enforce를 같이 켜야 KG가 다시 섞이지 않는다. 다만 O7에서 마이그레이션 효과가 없어 적용은 권하지 않는다(OA-11).
- **CI가 `requirements-dev.txt`를 설치하지 않는다.** `.github/workflows/test.yml`은 `requirements.txt`만 설치하므로 Pellet 교차 검증 테스트(`tests/unit/ontology/test_ontology_owl_check.py`)가 skip된다(아래 기존 항목과 같다).
- **`is_target` 일반화 안 함(§O4).** "AP 그룹 브랜드"·"K-Beauty 브랜드" 클래스로 넓히면 rule 골드 32문항 중 15문항의 발화 규칙 집합이 바뀐다. rule 골드가 LANEIGE 기준이라 골드를 다시 정의하기 전에는 하지 않는다.
- **날짜 없는 수치 엣지 333건**(위 enforce 항목 ②). 증거 계층은 이 엣지를 증거 카드에서 뺀다.
- **233문항 1회 측정에서 판정을 보류한 것**: 유형 외 문항 수치 정확도 0.139 → 0.056(−0.083, 수치 채점 문항 소수). 반복 측정이 없다.
- **`3421ef8`·기본값 전환 이후 코드로 LLM 평가를 다시 돌리지 않았다.** L4는 오프라인 재채점만 했다.

**O6에서 남긴 항목**

- **스크레이퍼 브랜드 오귀속**: `src/tools/scrapers/amazon_scraper.py`가 일부 제품의 브랜드를 잘못 붙인다(가짜 브랜드 `unknown`·`fresh`·`chi` 등이 KG에 들어오는 원인). 매일 크롤에 영향을 주므로 계획 범위 밖으로 두었고, 조회 쪽에서 `is_placeholder` 브랜드만 거른다(O2).
- **`dashboard_exporter`가 `as_of`를 넘겨야 `kg.write_validation=enforce`로 올릴 수 있다.** 지금은 날짜 없는 수치 엣지(hasSoS·hasHHI·hasPricePosition)가 매일 쓰여 enforce에서 막힌다(결정 OA-7·OA-8).
- **`kg_updater.load_brand_ownership`이 인수 연도로 `"True"`를 쓴다.** 설정 값이 불리언 `True`이면 `isinstance(True, int)`가 참이라 `str(True)`="True"가 되고, 소문자 `"true"`만 거르는 조건을 통과한다(`src/ontology/kg_updater.py` 인수 연도 처리부).
- **`config/entities.json`의 별칭 "로드"(rhode)가 "로드맵" 같은 단어에 오탐한다.** 부분 문자열 매칭이라 한국어 일반어와 겹친다.
- **(선택) Docker 다단계 빌드에서 Pellet 검증**: JDK 25 단계에서 `scripts/check_ontology_owl.py`만 돌리고 런타임 이미지에는 Java를 넣지 않는 방식(설계 OE10). Dockerfile 변경은 사용자 승인 필요. 지금은 배포 전 로컬에서 수동 실행.
- **운영에서 매일 Pellet 실행(크롤 후 OWL 재분류)은 이번 계획 범위 밖.** 필요해지면 JRE 25 추가(+100~200MB)와 Railway 메모리·빌드 시간을 측정한 뒤 따로 결정한다(OE10).
- CI(`.github/workflows/test.yml`)는 `requirements.txt`만 설치하므로 owlready2가 없어 `tests/unit/ontology/test_ontology_owl_check.py`가 skip된다. CI에서도 돌리려면 `requirements-dev.txt` 설치와 Java 25가 필요하다.
