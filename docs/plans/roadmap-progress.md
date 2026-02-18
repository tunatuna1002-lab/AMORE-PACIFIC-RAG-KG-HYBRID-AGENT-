# AMORE RAG-KG Hybrid Agent — 로드맵 진행 상태

> 참조: `docs/plans/master-roadmap.md`
> 업데이트: 2026-02-18

---

## Sprint 0: 계획 영구 저장

- [x] 0-1. `docs/plans/master-roadmap.md` 저장
- [x] 0-2. `docs/plans/roadmap-progress.md` 생성 (이 파일)
- [x] 0-3. `MEMORY.md` 로드맵 참조 추가

---

## Sprint 1: 현재 작업 마무리 (1~2일)

- [x] 1-1. Phase 1B: 엔티티 추출 통합
  - [x] `src/rag/entity_linker.py` — 브랜드/카테고리/지표 맵 병합
  - [x] `src/rag/hybrid_retriever.py` — EntityExtractor → entity_linker 위임 (thin wrapper)
  - [x] `src/rag/router.py` — extract_entities() → entity_linker 위임
  - [x] `src/agents/hybrid_chatbot_agent.py` — hybrid_context.entities 경유 (직접 추출 없음)
  - [x] entity_linker 단위 테스트 추가 (38개 pytest)
- [x] 1-2. Ruff 잔여 수정
  - [x] `ruff check .` — All checks passed
- [x] 1-3. 전체 테스트 통과 + 커밋
  - [x] `python3 -m pytest tests/ -v` — 4,837개 전체 통과 (커버리지 71.99%)
  - [x] `ruff check . && ruff format --check .` — 린트 통과
  - [x] Phase 1A/1B/1C/2A/2B + entity consolidation 커밋 (e604b28)

---

## Sprint 2: 보안 수정 (3~4일)

- [x] 2-1. P0 Critical
  - [x] ~~2.1: API Key HTML 노출~~ (이미 수정됨)
  - [x] 2.2: API_KEY 미설정 시 Fail-Open → `src/api/dependencies.py:29-31`
  - [x] 2.3: JWT_SECRET_KEY 보안 미검증 → `src/api/dependencies.py:463-465`
  - [x] 2.4: InputValidator 채팅 엔드포인트 미적용 → `src/api/routes/chat.py`
- [x] 2-2. P1 High
  - [x] Docker non-root user → `Dockerfile`
  - [x] 보안 헤더 (CSP, HSTS 등) → `src/api/middleware.py`
  - [x] Rate Limiting 범위 확대 → `src/api/routes/crawl.py`, `export.py` 등 (11개 라우트)
  - [x] Session ID UUID 전체 사용 → `src/memory/session.py:78`
  - [x] KG 경로 검증 → `src/ontology/knowledge_graph.py:400-451`
  - [x] SSL 검증 활성화 → `src/tools/calculators/exchange_rate.py:135,158`

---

## Sprint 3: Dashboard API 분해 (3~5일)

- [x] 사전 확인: `src/api/routes/` 13개 라우터 include_router 등록 현황
  - [x] alerts, analytics, brain, chat, competitors, crawl, data, deals, export, health, market_intelligence, signals, sync
- [x] dashboard_api.py 인라인 엔드포인트 → 라우트 모듈로 이동
  - [x] 이전 작업에서 이미 완료됨 (13개 라우트 모듈 분리)
- [x] dashboard_api.py ~200줄로 축소 (시작 + 글로벌 예외 처리만)
  - [x] 현재 183줄 (app 생성 + exception handler + startup event + __main__)
- [x] 라우트 모듈 app_factory.py에서 등록 완성
  - [x] 13개 라우터 + Telegram optional 등록 완료

---

## Sprint 4: Golden QA 확장 + Eval 기초 (1주)

- [x] 4-1. Golden QA 확장 (40→200+)
  - [x] `eval/data/golden/laneige_golden_v2.jsonl` 생성 (160 QA pairs, lg041-lg200)
  - [x] 메트릭(30), 제품(30), 브랜드(25), 시장(25), 멀티홉(20), 엣지(15), 시간(15) QA
  - [x] 난이도 분포: easy=28, medium=89, hard=43 (56 eval tests pass)
- [x] 4-2. Eval Harness: Fuzzy Matching
  - [x] `eval/metrics/base.py` — fuzzy_match, resolve_alias, set_f1_fuzzy
  - [x] `eval/metrics/l1_query.py` — alias_map + fuzzy_threshold
  - [x] `eval/schemas.py` — CostBreakdown, RegressionComparison 스키마

---

## Sprint 5: Application Layer + 순환 의존성 (1주)

- [x] 5-1. 핵심 워크플로우 구현
  - [x] `src/application/workflows/chat_workflow.py` (기존 구현 활용)
  - [x] `src/application/workflows/crawl_workflow.py` (기존 구현 활용)
  - [x] `src/application/workflows/insight_workflow.py` (기존 구현 활용)
  - [x] API 라우트 → 워크플로우 호출 전환 (chat.py v1 → ChatWorkflow)
- [x] 5-2. 순환 의존성 해소 잔여
  - [x] `export_handlers.py` → PeriodInsightAgent import 제거 (Container.get_period_insight_agent())
  - [x] 라우트 핸들러 → application workflow 호출 (chat.py v1 → ChatWorkflow)
  - [x] `container.py` 와이어링 중앙화 (get_period_insight_agent, get_rag_router 추가)

---

## Sprint 6: 보안 P2 + 기술 부채 (3~5일)

- [x] 6-1. 보안 P2
  - [x] PromptGuard 유니코드 우회 수정 (NFKC 정규화)
  - [x] 출력 필터 패턴 보강 (AWS Key, GitHub PAT, Slack, Private Key)
  - [x] Embedding Cache MD5 → SHA-256
  - [x] RAG 문서 인덱싱 크기 제한 (10MB)
  - [x] 소셜 수집기 프록시 자격증명 마스킹 (mask_proxy_url)
  - [x] Query Router ReDoS 방어 (MAX_QUERY_LENGTH=5000)
- [x] 6-2. 기술 부채
  - [x] `@app.on_event("startup")` → lifespan context manager 전환
  - [x] `knowledge_graph.json` 동시 쓰기 보호 (asyncio.Lock)
  - [x] `TestResult` 클래스 이름 충돌 해결 (`__test__ = False`)
  - [x] Config 정리 (분석 완료: 모든 config 활발히 사용 중, public_apis.json만 미사용)

---

## Sprint 7: Eval Harness 고도화 (1주)

- [x] 7-1. Judge 구현
  - [x] `eval/judge/llm.py` — GPT-4.1-mini 기반 LLMJudge (324줄, RAGAS-style)
  - [x] `eval/judge/nli.py` — 오프라인 NLI 모델 NLIJudge (359줄, cross-encoder)
  - [x] `eval/judge/__init__.py` — create_judge() 팩토리 (lazy import + fallback)
- [x] 7-2. Cost Tracking + Regression
  - [x] `eval/cost_tracker.py` — 레이어별 토큰/USD 추적 (369줄)
  - [x] `eval/regression.py` — 기준선 비교, 회귀 감지 (592줄)
  - [x] CLI: `python -m eval compare`, `set-baseline`, `--baseline` 플래그 (609줄)
  - [x] `eval/report.py` — Cost Summary + Regression Analysis 섹션 추가 (557줄)
- [x] 7-3. 테스트
  - [x] `tests/eval/test_fuzzy_matching.py` (기존 완료)
  - [x] `tests/eval/test_judges.py` (54개 테스트, StubJudge/LLMJudge/NLIJudge/Factory/Interface)
  - [x] `tests/eval/test_regression.py` (기존 완료)
  - [x] `tests/eval/test_cost_tracker.py` (기존 완료)

---

## Sprint 8: 학술 기준 Ontology Track A+B (1~2주)

- [ ] 8-1. Track A: OWL 온톨로지 심화
  - [ ] A-1: OWL Class Restriction
  - [ ] A-2: inverseOf 선언
  - [ ] A-3: Disjointness 공리
  - [ ] A-4: Cardinality 제약
  - [ ] A-5: Hard Validation
- [ ] 8-2. Track B: RAG BM25+RRF
  - [ ] B-1: BM25 Sparse Retrieval (rank_bm25)
  - [ ] B-2: RRF 병합 구현 (k=60)
  - [ ] B-3: ConfidenceFusion에 RRF 전략 추가
  - [ ] B-4: Self-RAG 검색 필요성 판단

---

## Sprint 9: 학술 기준 Track C+D (1~2주)

- [ ] 9-1. Track C: Multi-hop + SPARQL
  - [ ] C-1: Multi-hop refine_search 액션
  - [ ] C-2: IRCoT 패턴
  - [ ] C-3: SPARQL 기본 지원
  - [ ] C-4: 문장별 인라인 인용 (AIS)
- [ ] 9-2. Track D: 통합 + 검증
  - [ ] D-1: IRI 체계 도입
  - [ ] D-2: KG IRI 마이그레이션
  - [ ] D-3: HybridRetriever Self-RAG 통합
  - [ ] D-4: OWL Consistency Check
  - [ ] D-5: 전체 통합 테스트

---

## Sprint 10: 마무리 + God Objects 분할 (1주)

- [ ] 10-1. God Objects 추가 분할
  - [ ] `src/ontology/business_rules.py` (1,540줄) → 규칙 카테고리별 분리
  - [ ] `src/ontology/knowledge_graph.py` (1,514줄) → CRUD/Query 분리
  - [ ] `src/agents/hybrid_chatbot_agent.py` (1,353줄) → base 상속 후 정리
- [ ] 10-2. 보안 P3
  - [ ] TrustedHost 미들웨어
  - [ ] CSRF 보호
  - [ ] 의존성 보안 스캔 자동화 (safety, pip-audit, bandit)
  - [ ] 세션 데이터 암호화
- [ ] 10-3. DI 전환 잔여
  - [ ] hybrid_insight_agent.py → ExternalSignalCollector DI
  - [ ] period_insight_agent.py → PeriodAnalyzer DI
  - [ ] api/routes/deals.py → AlertAgent DI

---

## 완료 현황 요약

| Sprint | 상태 | 완료율 |
|--------|------|--------|
| Sprint 0 | 완료 | 3/3 |
| Sprint 1 | **완료** | 3/3 |
| Sprint 2 | **완료** | 9/9 |
| Sprint 3 | **완료** | 4/4 |
| Sprint 4 | **완료** | 8/8 |
| Sprint 5 | **완료** | 8/8 |
| Sprint 6 | **완료** | 10/10 |
| Sprint 7 | **완료** | 12/12 |
| Sprint 8 | 대기 | 0/9 |
| Sprint 9 | 대기 | 0/9 |
| Sprint 10 | 대기 | 0/10 |
