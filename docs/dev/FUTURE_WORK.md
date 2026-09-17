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
- `tests/unit/core/test_react_agent.py::test_react_run`이 `acompletion`을 가짜로 두지 않아 전체 테스트마다 실제 OpenAI를 호출한다.
- `tests/unit/core/test_cache.py::test_cleanup_expired_removes_old`가 TTL 0초·동일 시각 비교에 의존해 간헐 실패한다.
- v4 Brain 경로의 답변 프롬프트(`HybridRetriever._combine_contexts` → `ResponsePipeline`)에는 크롤 DB 수치 사실(`metric_facts`, `3fce8e8`)이 실리지 않는다. v1 `ContextBuilder`만 렌더링한다. 평가 트레이스(judge 컨텍스트)에는 두 경로 모두 들어간다.
- `eval.cli ablation`은 `run`과 달리 데이터 시점(`AMORE_DATA_AS_OF`)을 고정하지 않는다.
- `eval/cost_tracker.py:34`의 gpt-4.1-mini 단가가 $0.15/$0.60 per 1M(공시가 $0.40/$1.60)이라 리포트 비용이 약 1/2.67로 과소 집계된다. v1.0~v9.1 기준선의 비용 수치도 같다.
- `HybridRetriever.retrieve`가 검색 예외를 삼키고 빈 컨텍스트로 답변을 계속 만들어, 평가 하니스가 인프라 실패로 분류하지 못한다(2026-09-17 Chroma 오류 실행 2건이 조용히 채점됨).
- 평가 프로세스 여러 개를 같은 `data/chroma`로 동시에 띄우면 한쪽의 색인 변경이 다른 쪽 검색을 깨뜨린다. 실행별 `CHROMA_PERSIST_DIR` 복사본이나 읽기 전용 모드가 필요하다.
- `OWLRetrievalStrategy` 생성자 기본값이 시맨틱 청킹 `DocumentRetriever`를 새로 만들어 초기화 때 공유 컬렉션에 다른 청크를 추가 색인한다(`create_owl_strategy`는 `0669b75`에서 공유 검색기를 넘기게 했지만 생성자 기본값은 그대로).
- `OWLRetrievalStrategy._matches_filters`가 엔티티 링커의 Chroma where 형식 필터(`$or`, brand·category 키)를 처리하지 못하고 문서 메타데이터에도 그 키가 없어, 엔티티가 연결된 질의는 문서를 0건 가져온다(130문항 중 124문항).
- 규칙 추론(`OntologyReasoner`)이 v4 평가 130문항·전 실행에서 추론 0건이다. `_build_inference_context`가 읽는 키가 운영 데이터와 맞지 않는다(근거 문서 §5.3).
- 신뢰도 점수가 v4 평가의 172/172문항을 HIGH로 분류해 DecisionMaker(LLM 도구 선택)와 ReAct 분기가 평가에서 한 번도 실행되지 않는다. 임계값·점수 구성을 재검토해야 이 경로들을 측정할 수 있다.

### 9.9 증거 카드·규칙 추론·ReAct 통합 작업(2026-09-17~) 중 발견한 범위 밖 항목
> 출처: `eval_output/evidence-2026-09/notes/0c_recalc.md`(트랙별 보고), `docs/experiments/evidence_pipeline_2026-09.md`. 이 작업에서 고치지 않았다.

**측정·비용**
- `eval/judge/llm.py`(`MODEL_PRICING`, 135~140·337행)에 옛 단가가 남아 있다(`get_usage` 호출처 0이라 현재 리포트에는 영향 없음).
- L2 임베딩(`src/rag/retriever.py`의 openai 직접 호출)과 v4 질의 확장(`DocumentRetriever.expand_query`)의 토큰·비용이 어느 리포트에도 집계되지 않는다(장부는 ×1.1로 보정).
- judge 호출 시간 초과(60초×3회)가 동시 실행 중 특정 시간대에 몰려 문항이 채점에서 빠진다(2단계 run2 `lg161`, run3 `lg155`). 재시도 간격(backoff) 없음.
- 골든 문서의 CPI 정의(<1.0 비율)와 코드 CPI(100 기준)가 다르다(lg043, lg056).
- 골든 엣지는 `ownedBy`인데 KG 술어는 `ownedByGroup`이라 엣지 Recall이 낮게 나올 수 있다(미검증).

**데이터 파이프라인**
- 2026-08-31 지표 테이블(`brand_metrics`·`market_metrics`, 08-30 18:06 UTC 계산)이 같은 날짜 `raw_data`(08-31 13:03 UTC 교체)와 다른 크롤 상태로 계산됐다(예: lip_care HHI 0.0681 vs raw 0.0637, lip_makeup 21/21 브랜드 SoS 불일치). 지표 재계산 시점 결함.
- `raw_data.brand` 부분 문자열 오귀속 24/356행·14쌍(lip_care "Hera" ← "Therapy", "CHI" ← KimChiChic, "OPI", "Verb", "elf", skin_care "Fresh" 등). 골든 lip_care 순위·HHI와 KG `competesWith`(system 출처)에 유입.
- `raw_data.price` 2025-12~2026-01 714행이 1,411~86,377(KRW로 추정)인데 `price_currency=USD`.
- `raw_data.reviews_count` 8,769행이 빈 문자열(2025-12-16~2026-01-19).
- `badge` 필드에 평점 문자열("4.6")이 들어 있다 — 스크레이퍼 필드 매핑 결함 추정.
- 2026-08-31 lip_care SoS가 정수값(제품 수 기반)이다. 다른 날짜·카테고리와 계산 방식 일관성 확인 필요.

**KG·온톨로지**
- KG 수치 엣지 359개(`hasSoS` 169, price position 124, `hasHHI` 66)가 전부 날짜가 없다. lip_care `hasHHI`는 값이 15개이고 0~10000 스케일이 섞여 있다. 검색 증거에서는 제외했지만(E2) 정리·버전 부여는 하지 않았다.
- KG 주어 대소문자가 술어마다 다르다(`competesWith` 주어 `laneige`, `ownedByGroup` 주어 `LANEIGE`).
- KG `TATA HARPER acquiredIn='True'`.
- `KnowledgeGraph()` 기본 생성이 로드 중 자동 저장으로 KG 파일을 다시 쓴다(0-F의 근본 원인). 테스트는 `persist_path`·`auto_save=False`로 격리했지만 생성자 동작 자체는 그대로다.
- 규칙 37개 중 입력을 공급할 수 없는 규칙: sentiment 8개 전부, IR 5개, 이력(기간 비교)이 필요한 규칙.

**코드**
- `src/rag/context_builder.py:599` `churn_rate`를 `:.1f%`로 표시한다(정의는 0~1). 데이터가 전부 NULL이라 아직 드러나지 않음.
- `MetricFactsProvider`가 제품명을 60자로 자르고 ASIN을 넘기지 않아 제품 카드가 KG 제품(ASIN)과 연결되지 않는다.
- `HybridRetriever._query_knowledge_graph`의 `trend_keywords`가 주어(브랜드/MARKET)를 보존하지 않는다.
- 1-A가 시험지에서 뺀 23문항은 `gold_source=domain_expectation`(추정 골드)이라 정답 채점이 불가하다 — 골드 보강 필요.
