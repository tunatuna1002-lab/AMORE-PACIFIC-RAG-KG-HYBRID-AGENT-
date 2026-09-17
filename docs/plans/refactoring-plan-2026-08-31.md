# 리팩토링 계획 — 2026-08-31 전수 스윕 기반

> 작성: 2026-08-31, 기준 커밋 `1bfcd19` (sos_delta 실계산 커밋 직후)
> 근거: src/ 전수 스윕 + 대시보드/설정 전수 스윕 (파일:라인 전부 실검증됨)
> 원칙: **동작 변경은 테스트 먼저, Phase 단위 커밋, 계획 밖 리팩토링 금지**

## 이 문서를 읽는 에이전트에게

- 각 항목의 파일:라인은 2026-08-31 기준 실측이다. 코드가 이동했을 수 있으니 수정 전 반드시 현재 위치를 재확인하라.
- "결정 필요" 표시가 있는 항목은 구현 전에 이 문서의 §결정사항 권고안을 따르되, 코드 근거가 권고안과 충돌하면 근거를 남기고 판단하라.
- 골든셋 평가(160문항, 회당 ~$ 및 20분)는 **Phase 4 완료 후 1회만** 실행한다. 기준은 `eval/baselines/v7.1-2026-08-30/` (단, 다른 세션이 사이클 8을 진행 중이면 최신 baseline 확인). 실행: `.venv/bin/python -m eval.cli run --dataset eval/data/golden/laneige_golden_v2.jsonl --baseline v7.1-2026-08-30`
- 전체 테스트: `.venv/bin/python -m pytest tests/ -q --no-cov` (약 7분, 5,199+ 통과가 기준선). Phase마다 관련 테스트만 돌리고, 마지막에 전체 1회.
- 주의: `src/rag/retrieval_strategy.py`는 다른 세션이 작업 중(top_k 예산 정렬)이다. 커밋 전 `git status`로 타 세션 변경분이 섞이지 않게 파일 단위로 스테이징하라.

---

## Phase 0 — 안전망 (30분)

1. `git checkout -b refactor/data-integrity-2026-08-31`
2. 전체 테스트 1회 실행해 기준선 기록 (실패 수·스킵 수 메모)
3. `ruff check src/` 클린 확인

---## Phase 1 — P0: 지표 데이터 정합성 (가장 중요)

### 1.1 HHI 단일 구현으로 통합 — 현재 6곳, 2개 스케일 [P0]

현황 (전부 실검증):

| 위치 | 공식 | 스케일 | Unknown 처리 |
|---|---|---|---|
| `src/tools/calculators/metric_calculator.py:165` | Σ(count/total)² | 0–1 | 분자 제외·분모 포함 (버그, §1.2) |
| `src/tools/exporters/dashboard_exporter.py:636` | Σ(share×100)²/10000 | 0–1 | 포함 |
| `src/core/brain.py:1525` | Σ(share×100)² | 0–10000 | 포함 |
| `src/api/routes/alerts.py:980` | brain.py 복붙 | 0–10000 | 포함 |
| `src/ontology/kg_enricher.py:333` | round(Σs²×10000) | 0–10000 | 포함 |
| `src/tools/calculators/period_analyzer.py:311, :511` | Σ(share×100)² | 0–10000 | 포함 |

실피해: `src/rag/templates.py:186`이 `hhi >= 0.25`(0–1 기준)로 분기 → 0–10000 값이 오면 항상 "고집중 시장" 서사. `src/tools/intelligence/insight_verifier.py:291`은 `hhi > 10000`만 거부 → 0–1 값 검증 불가.

작업:
- `metric_calculator.calculate_hhi()`를 유일 구현으로. **정본 스케일은 0–1** (§결정사항 D1). 0–10000이 필요한 소비처(KG, 리포트)는 명시적 변환 헬퍼 `hhi_to_points(hhi) -> int` 사용.
- 나머지 5곳의 자체 계산 삭제 → 정본 호출로 교체. period_analyzer는 두 곳 모두.
- `templates.py:186`, `insight_verifier.py:291`의 분기·검증을 정본 스케일에 맞춤.
- 테스트: 스케일 고정 테스트(0≤hhi≤1), 변환 헬퍼 테스트, "두 스케일 혼용 회귀 방지" 테스트.

### 1.2 HHI 분모 버그 — Unknown이 분모에만 포함 [P0]

`metric_calculator.py:152-165`: Unknown/빈 브랜드를 분자 루프에서 `continue`하면서 `total = len(top_records)` 유지 → 점유율 합이 1 미만, HHI가 Unknown 비율의 제곱만큼 체계적 과소. Unknown이 ~28%인 현재 데이터에서 상시 발생.
작업: `total = sum(brand_counts.values())`로 교정 + 회귀 테스트 (Unknown 50% 데이터에서 기대값 검증).

### 1.3 SoS 분모 `max(total, 100)` 바닥 — 6곳 [P0]

`src/api/routes/data.py:121, 156, 473, 512, 537, 671` — 부분 수집일(예: 초기 60개/카테고리)에 SoS가 조용히 과소 계산됨.
작업: 실분모 사용 공통 헬퍼 1개 추출(`_sos(count, total, min_sample=50)` 형태), 표본 미달 시 `None` 반환하고 API 응답에 `insufficient_sample` 플래그. 6곳 교체 + 부분 수집 시나리오 테스트.

### 1.4 market_intelligence 가짜 지표 주입 [P0]

`src/api/routes/market_intelligence.py:200` — `amazon_data = {"sos": 5.2, "laneige_rank": 15}  # placeholder`. 198행에서 얻은 storage 핸들은 버려짐. 이 가짜 값이 LLM 인사이트의 근거로 그대로 나감.
작업: storage에서 실값 조회로 교체. 데이터 없으면 `amazon_data=None`으로 넘겨 엔진이 "Layer 1 데이터 없음"을 표기하게. 테스트로 placeholder 문자열 회귀 방지.

### 1.5 SQLite 폴백의 거짓 신선도 [P0]

`src/api/routes/data.py:139-141` — `"_cache_age_hours": 0, "_is_stale": False` 하드코딩. `latest_date`가 며칠 전이어도 신선하다고 표시됨.
작업: `latest_date` 기준으로 실계산 (`dependencies.py:197-209`와 같은 규칙). 신선도 판정 로직을 헬퍼로 공유.

### 1.6 가짜 성장 목표 [P1이지만 같은 파일권]

`src/agents/period_insight_agent.py:970` — `"target_sos": metrics.get("end_sos", 0) * 1.1  # +10% 목표 예시`. 임의 배수가 비즈니스 목표로 출력됨.
작업: 키 삭제(권장) 또는 config로 이동. 소비처 확인 후 처리.

**Phase 1 완료 기준**: 관련 유닛 테스트 통과 + `data.py`/`metric_calculator` 신규 테스트 + 대시보드 로컬 렌더에서 HHI·SoS 값 변화 눈검증(값이 바뀌는 게 정상 — 변화폭을 커밋 메시지에 기록).

---

## Phase 2 — P0: 대시보드 신뢰 (프론트, `dashboard/amore_unified_dashboard_v4.html`)

### 2.1 `loadAlertSettings` ReferenceError — 초기화 사슬 절단 [P0, 최우선]

`:13084`에서 존재하지 않는 함수 호출 (커밋 `515649b`가 정의만 삭제). `window.onload`에 try/catch가 없어 이후 전부 미실행: `:13085 initDateRangeSelectors()`, `:13093 loadInitialChartDataWithGlobalRange()`, `:13095 lucide.createIcons()`. "차트 빈 화면/날짜 피커 공백" 증상의 유력 원인.
작업: 13084행 삭제 + onload 본문 try/catch로 감싸기(개별 실패가 사슬을 끊지 않게). 브라우저 콘솔 에러 0 확인.

### 2.2 API_BASE 오리진/CSP 불일치 [P0]

`:8503-8506` — localhost/127.0.0.1일 때 `http://localhost:8001` 강제 → `127.0.0.1`로 열면 CSP `connect-src 'self'`에 전부 차단. 포트 8001도 하드코딩(PORT 환경변수 무시).
작업: `const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:8001' : window.location.origin;`. `:8756`, `:8776`의 localhost 링크도 `SERVER_URL` 상수 하나로. (참고: 별도 spawn_task 칩이 이미 존재 — 중복 작업 주의)

### 2.3 Category/Product 뷰 — 초기화 누락으로 가짜 KPI 전면 노출 [P0]

- Category: 바인딩 함수(`updateCategoryBadges :7902`, `selectCategory :7849`)는 전부 존재하나 `switchPage()`(`:7156-7225`)에 `category` 분기가 없어 첫 진입 시 하드코딩 시드(18.5% "Category Leader", #2 "1위 근접", 44점 "가격 경쟁력 하락", 18개 "경쟁 심화")가 실데이터처럼 표시.
- Product: 같은 문제 + `#productSelect`(`:4895-4899`)에 데모 옵션 3개 하드코딩. `updateProductList()`(`:7965`)는 category onchange에서만 호출됨. 시드 값 #2/12pt/4.5★/92pt 상시 노출.
작업: `switchPage`에 category 분기(`selectCategory('beauty', ...)`)와 product 분기에 `updateProductList()` 호출 추가. HTML 시드 값 전부 `—`/스켈레톤으로 교체, 데모 option 제거.

### 2.4 Brand 뷰 배지 3종 영구 정적 [P1]

`:4468 "+1 증가"`, `:4481 "▼ 2.3위 개선"`, `:4494 "중간 집중도"` — id 없음, 갱신 코드 없음. (sos_delta 배지는 오늘 수정 완료 — `#brand-sos-delta` 패턴을 그대로 따라할 것.)
작업: 각 배지에 id 부여, exporter kpis에 top10/avg_rank 전일 델타 추가(§1의 `_calculate_sos_delta` 패턴 재사용), HHI 밴드 라벨은 실 HHI로 계산. `:5838`의 타입 혼동 폴백(`kpis.hhi ?? '중간 집중도'`가 숫자 칸에 문자열)도 `'-'`로 교정.

### 2.5 거짓 성공 토스트 [P1]

`:8017` `applyProductDateRange()` — "기간이 적용되었습니다" 토스트 후 `// TODO: 실제 데이터 필터링 로직`. 필터링 없음.
작업: 실 필터 연결 또는 컨트롤 비활성+안내. 거짓 성공 UI는 남기지 않는다.

### 2.6 환율 API CSP 차단 [P1]

`:8546` frankfurter.app 호출이 CSP에 차단 → 항상 폴백 환율 1350 사용(표시는 됨). 작업: 백엔드 프록시 `/api/fx/rates` 신설(서버측 캐시 포함)로 교체. CSP 완화는 하지 않는다.

### 2.7 죽은 코드 [P2]

`:6142` `applyBrandMatrixDateRange()` return 뒤 ~35행 도달불가 블록 삭제.

**Phase 2 완료 기준**: Playwright로 4개 뷰 전부 로드 → 콘솔 에러 0, 하드코딩 시드 값 미노출, 127.0.0.1과 localhost 양쪽에서 데이터 로드 성공.

---

## Phase 3 — P1: 침묵 기능 배선 (알림·신선도·무결성)

### 3.1 이벤트 버스 이원화 정리 + crawl_failed 배선 [P1]

현황: `brain.on_event`는 전 코드베이스에서 1회만 사용(`brain.py:398`, KG 동기화 클로저). `AlertAgent.on_crawl_failed`(`alert_agent.py:359`)·`on_crawl_complete`·`on_error`는 **프로덕션 호출 0건**. 실제 알림은 `alert_manager.check_conditions()`(`brain.py:444-445`)만 타며, 하드코딩 3종 이벤트에 실패가 없음. 결과: CRITICAL 크롤 실패 이메일 알림이 완전 구현+완전 미도달.
작업 (§결정사항 D2): `alert_manager`를 정본 버스로 확정 → 크롤 예외 경로(`crawl_manager.py:425-430` 및 스케줄러 핸들러 `brain.py:1667-1669`)에서 `crawl_failed` 이벤트 발화 → alert_manager가 이메일/텔레그램 전송. `AlertAgent`의 미배선 핸들러 3종은 삭제(인터페이스 포함). 통합 테스트: 크롤 실패 모킹 → 알림 발송 함수 호출 검증.

### 3.2 data_freshness 상시 "unknown" [P1]

`mark_crawled()`/`mark_data_stale()`이 `src/core/state.py:83,96`과 `src/core/state_manager.py:191,207` **양쪽에 중복** 존재하며 호출은 테스트뿐. 파생 죽은 코드: `response_pipeline.py:622`(+1.0 신선도 보너스 도달불가), `:546`·`context_gatherer.py:329`(항상 참 분기).
작업: 크롤 완료 경로(batch_workflow)에서 `mark_crawled()` 호출 배선. state.py/state_manager.py 중 하나로 통합(중복 삭제). 파생 분기 3곳이 실제로 동작하게 됐는지 테스트.

### 3.3 data_integrity_checker 미배선 [P1]

`run_full_check`(`data_integrity_checker.py:233`)는 `__main__` 외 호출 0건. Sheets↔SQLite 드리프트 감지+CRITICAL 판정+권고가 전부 사장.
작업: 스케줄러 일일 태스크로 등록(크롤 후) + `/api/health/integrity` 라우트 노출. 결과 severity가 CRITICAL이면 alert_manager로 전달(§3.1과 연결).

### 3.4 crawl_complete 수동 전용 [P1]

`brain.py:1226`의 `crawl_complete` 발화는 `run_autonomous_cycle()` 전용 → 프로덕션 진입점은 수동 엔드포인트뿐. 스케줄러 기본값 `auto_start_scheduler=False`(`config_manager.py:62`).
작업 (§결정사항 D4): 이벤트 발화를 스케줄러 래퍼가 아니라 **워크플로우 완료 지점**으로 이동(어느 경로로 크롤되든 발화). Railway 배포는 `AUTO_START_SCHEDULER=true` 환경변수 확인.

---

## Phase 4 — P1: 신뢰도·가드레일 실효화 (챗봇 품질 — 골든셋 재측정 대상)

### 4.1 confidence가 `max()`로 상향만 됨 [P1]

`response_pipeline.py:196-198` — `final_confidence = max(calculated, decision.confidence)`. 근거 기반 점수가 바닥 올리기로만 쓰여, 낙관적 LLM 자신감이 항상 승리. 그라운딩 체크로서 정반대.
작업: `min()` 또는 곱셈 감쇠로 교체. 1월 실사용에서 확인된 "오답에 confidence 10.0" 사례가 재현 테스트 대상.

### 4.2 confidence 사다리 이중화 [P1]

`response_pipeline.py:591-627`이 `confidence.py:34-36`의 5.0/3.0/1.5 사다리를 다른 스코어러(10.0 캡)로 재구현 — RAG 3개+KG 3개면 자동 HIGH.
작업: `ConfidenceAssessor`로 위임 통합, 자체 사다리 삭제.

### 4.3 `apply_guardrails()` no-op [P1]

`src/rag/templates.py:268-276` — 루프 본문이 `pass`, 원문 그대로 반환. 호출처 2곳(`hybrid_chatbot_agent.py:595`, `hybrid_insight_agent.py:549`)은 가드된다고 믿는 중. 기존 테스트는 "문자열 반환"만 검증.
작업: FORBIDDEN_PHRASES 매칭 시 완곡 표현 치환(HEDGING_PHRASES 활용) 구현 + 금지 표현 입력이 실제로 바뀌는 테스트.

### 4.4 인사이트 few-shot의 가짜 수치 [P1]

`hybrid_insight_agent.py:462-490` — 출력 형식 예시가 실수치처럼 보이는 가짜 값 전체 세트(`SoS +2.1%p`, `+12.3%`, `+41% YoY`, `2.4M` 등)와 가짜 인용 `[1][2][3]`. 데이터가 빈약할 때 모델이 그대로 베낄 최악의 형태 (오늘 제거한 +2.1%p와 동일 값 잔존).
작업: `{{SOS_DELTA}}`, `[D{{n}}]` 형태 typed placeholder로 교체. **프롬프트 변경이므로 골든셋 영향 확인 필수** — n=30 서브셋으로 사전 A/B 후 적용 (`docs/experiments/prompt_exp_2026-08.md` 프로토콜 재사용).

### 4.5 소스 추출 이원화 [P2]

`response_pipeline.py:555-571`(bare 문자열 5개)과 `source_provider.py`(타입·URL·신뢰도 풀 메타) 병존 — 어느 경로가 응답하느냐에 따라 인용 품질이 다름.
작업: response_pipeline이 `Container.get_source_provider()` 소비하도록 통일.

### 4.6 brand_resolver 웹서치 스텁 [P1]

`brand_resolver.py:319-326` — 쿼리만 만들고 검색 안 함, ~12개 하드코딩 dict 폴백. Unknown 28%의 한 원인.
작업: 실구현은 범위 밖(별도 과제). 이번엔 이름을 `_resolve_from_known_patterns`로 정정하고 docstring에 한계 명시 + Unknown 비율 로그 추가.

**Phase 4 완료 기준**: 유닛 테스트 + **골든셋 160문항 1회 실행, v7.1(또는 최신) 대비 compare에서 회귀 없음** (groundedness·relevance 유지, 4.4의 A/B 기록 문서화).

---

## Phase 5 — 지표 영속화 결정 (brand_metrics/market_metrics 0행 문제)

현황 (전부 실검증): 테이블·인덱스는 정상 생성, 라이터 2개가 모두 미도달 —
- `SQLiteStorage.save_brand_metrics`(`sqlite_storage.py:516`)의 유일 호출처 `crawl_workflow.py:135`는 살아있는 경로가 아님 (`get_crawl_workflow` 호출 0건; 실경로는 `BatchWorkflow`). 게다가 전달 shape도 불일치(dict-keyed를 list로 래핑). `save_market_metrics`는 거기서도 미호출.
- `StorageAgent.save_metrics`(`storage_agent.py:287`)는 Sheets에만 쓰고 호출처 0건.
- 실경로 `batch_workflow.py:885-899`는 CALCULATE 결과를 state와 KG에만 넘기고 저장 스텝 없음 → `/api/data`가 매 요청 raw 재계산(`data.py:353`), 10초 안전 타임아웃(`:13065`)의 유력 원인.

작업 (§결정사항 D3 — 영속화 복원 권장):
1. `batch_workflow`에 CALCULATE→INSIGHT 사이 STORE_METRICS 스텝 추가, `MetricsAgent` 산출(flat list, shape 이미 정합)을 `save_brand_metrics`/`save_market_metrics`로 저장
2. `data.py:353` 재계산을 테이블 조회로 교체 (과거 날짜는 백필 스크립트 1회: raw_data → metrics 재계산 저장)
3. `CrawlWorkflow`와 `get_crawl_workflow` 삭제 또는 실경로 통합 (죽은 병렬 구현 제거)
4. `StorageAgent.save_metrics` — Sheets 백업이 여전히 필요하면 STORE_METRICS에서 함께 호출, 아니면 삭제
성능 검증: `/api/data` 응답 시간 before/after 기록.

---

## Phase 6 — P2 위생 (마지막, 일괄)

1. **config 죽은 키 12개** — `thresholds.json`의 `crawl_hour_utc`, `hhi_concentrated`, `sos_change_up/down`, `laneige_market_share_warning`, `cpi_premium`, `rating_gap_warning`, `new_product_watch_days`, `gap_alert`, `eviction_policy`, `system.rag.max_chunks`, `system.rag.rerank_enabled` 전부 리더 0. 특히 `sos_change_*: ±1.0`은 코드 상수 `SOS_CHANGE_THRESHOLD=2.0`(`alert_manager.py:41`)과 **값이 다름**. 대시보드 JS 매직넘버(`:7910` sos≥15, `:7941` cpi≥90, `:7954` newComp≥15)가 정확히 이 죽은 키들의 재구현. → 배지 임계값을 `/api/data` 페이로드로 서빙해 config 단일화, 나머지 키는 배선 또는 삭제.
2. **config 로더 이중화** — `rules_engine.py:374`/`config_manager.py:184` `get_threshold()` 둘 다 호출 0건, 폴백 의미도 다름(`config_manager.py:103`은 문서 전체를 바인딩). 하나로 통일, 하나 삭제.
3. **retrieval_weights 코드 기본값** — `hybrid_retriever.py:1322` `rag_chunks: 3` (파일은 8로 고쳐졌으나 코드 기본값은 사이클 2 버그 값 그대로). 파일 없는 배포 환경에서 침묵 회귀. → 8로 교정 + deep merge로 변경. **[P1로 취급]**
4. **죽은 코드 삭제** — `confidence.py:88-114` `_context_bonus()`(주석은 삭제됐다고 하는데 본문 잔존), `src/api/__init__.py:20-33` 죽은 api_router(라이브 등록과 프리픽스도 불일치), `alerts.py` v3 경로 절반(~19KB, v4로 대체 확인됨), `config/competitors.json`(리더 0).
5. **dashboard_exporter 상수 필드** — `:1514-1516` `rank_change_7d: 0, streak_days: 7, rating_gap: 0.1` "추후 계산 가능" — streak 7이 온톨로지 규칙을 상수로 트리거 가능. → `None` 방출 + reasoner가 결측 스킵. `:1009-1011` `new_competitors`가 실은 전체 브랜드 수인 것도 교정.
6. **예외 침묵 24곳** — 동일 문구 `"Suppressed Exception"` (hybrid_retriever 8곳 등). 작업명 포함 메시지로 구체화. 특히 `hybrid_retriever.py:1514-1520`의 bare `except: pass`(인텐트 분류 실패 → 무기록 weighted_sum 폴백)는 warning 로그 필수.
7. **죽은 엔드포인트 정리** — 대시보드 미호출 라우트 다수(위 스윕 표 참조)는 삭제하지 말고 목록만 `docs/dev/FUTURE_WORK.md`에 기록 (API 소비자가 대시보드만이 아닐 수 있음). 단 `/api/crawl/status`는 대시보드 신선도 배너에 연결(현재 metadata.generated_at보다 나은 소스).

---

## 결정사항 (구현 전 확정 필요 — 권고안 포함)

| ID | 결정 | 권고안 | 근거 |
|---|---|---|---|
| D1 | HHI 정본 스케일 | **0–1** | 규칙 엔진(`market_rules.py` 0.15/0.25 임계)과 대시보드 표시가 이미 0–1. KG·리포트만 ×10000 변환 헬퍼로 |
| D2 | 이벤트 버스 | **alert_manager 단일화, AlertAgent 핸들러 삭제** | on_event는 사실상 미사용, alert_manager가 실경로 |
| D3 | 지표 영속화 | **STORE_METRICS 복원** | 매 요청 raw 재계산이 대시보드 지연의 유력 원인, 기간 비교 기능(로드맵)도 테이블 필요 |
| D4 | crawl_complete 발화 위치 | **워크플로우 완료 지점** | 스케줄러/수동 어느 경로든 발화되도록 |

## 명시적 비범위 (이번에 하지 않는 것)

- BM25/리랭커/검색 품질 튜닝 (별도 eval 사이클 영역, 타 세션 진행 중)
- brand_resolver 웹서치 실구현 (이름 정정만)
- 골든셋 문항 추가/IR 도메인 확장
- 대시보드 디자인 변경, 모바일 대응
- `docs/portfolio/` 관련 일체

## 예상 규모

| Phase | 파일 수 | 난이도 | 예상 |
|---|---|---|---|
| 1 | ~10 | 중 (지표 값이 바뀜 — 변화폭 기록 필수) | 3–4h |
| 2 | 1 (13.8K줄 HTML) | 중 | 2–3h |
| 3 | ~8 | 중상 (배선 결정 포함) | 3–4h |
| 4 | ~7 | 상 (골든셋 검증 포함) | 4–5h |
| 5 | ~6 | 상 (백필 포함) | 3–4h |
| 6 | ~15 | 하 (기계적) | 2–3h |
