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

- [ ] 1-1. Phase 1B: 엔티티 추출 통합
  - [ ] `src/rag/entity_linker.py` — 브랜드/카테고리/지표 맵 병합
  - [ ] `src/rag/hybrid_retriever.py` — EntityExtractor → entity_linker 위임
  - [ ] `src/rag/router.py` — extract_entities() → entity_linker 위임
  - [ ] `src/agents/hybrid_chatbot_agent.py` — 브랜드 추출 → entity_linker 위임
  - [ ] entity_linker 단위 테스트 추가
- [ ] 1-2. Ruff 잔여 4개 수정
  - [ ] `scripts/evaluate_golden.py:155-156` — C401: `set(...)` → `{...}`
  - [ ] `src/infrastructure/bootstrap.py:103,112` — F401: 미사용 import 제거
- [ ] 1-3. 전체 테스트 통과 + 커밋
  - [ ] `python3 -m pytest tests/ -v` — 4,816개 전체 통과
  - [ ] `ruff check . && ruff format --check .` — 린트 통과
  - [ ] Phase 1A/1B/1C/2A/2B + Ruff 일괄 커밋

---

## Sprint 2: 보안 수정 (3~4일)

- [ ] 2-1. P0 Critical
  - [x] ~~2.1: API Key HTML 노출~~ (이미 수정됨)
  - [ ] 2.2: API_KEY 미설정 시 Fail-Open → `src/api/dependencies.py:29-31`
  - [ ] 2.3: JWT_SECRET_KEY 보안 미검증 → `src/api/dependencies.py:463-465`
  - [ ] 2.4: InputValidator 채팅 엔드포인트 미적용 → `src/api/routes/chat.py`
- [ ] 2-2. P1 High
  - [ ] Docker non-root user → `Dockerfile`
  - [ ] 보안 헤더 (CSP, HSTS 등) → `src/api/middleware.py`
  - [ ] Rate Limiting 범위 확대 → `src/api/routes/crawl.py`, `export.py` 등
  - [ ] Session ID UUID 전체 사용 → `src/memory/session.py:78`
  - [ ] KG 경로 검증 → `src/ontology/knowledge_graph.py:400-451`
  - [ ] SSL 검증 활성화 → `src/tools/calculators/exchange_rate.py:135,158`

---

## Sprint 3: Dashboard API 분해 (3~5일)

- [ ] 사전 확인: `src/api/routes/` 12개 라우터 include_router 등록 현황
- [ ] dashboard_api.py 인라인 엔드포인트 → 라우트 모듈로 이동
- [ ] dashboard_api.py ~200줄로 축소 (시작 + 글로벌 예외 처리만)
- [ ] 라우트 모듈 12개 app_factory.py에서 등록 완성

---

## Sprint 4: Golden QA 확장 + Eval 기초 (1주)

- [ ] 4-1. Golden QA 확장 (40→200+)
  - [ ] `eval/data/golden/laneige_golden_v2.jsonl` 생성
  - [ ] 메트릭(30), 제품(30), 브랜드(25), 시장(25), 멀티홉(20), 엣지(15), 시간(15) QA
  - [ ] 난이도 분포: easy 30%, medium 50%, hard 20%
- [ ] 4-2. Eval Harness: Fuzzy Matching
  - [ ] `eval/metrics/base.py` — fuzzy_match, resolve_alias, set_f1_fuzzy
  - [ ] `eval/metrics/l1_query.py` — alias_map + fuzzy_threshold
  - [ ] `eval/schemas.py` — CostBreakdown, RegressionComparison 스키마

---

## Sprint 5: Application Layer + 순환 의존성 (1주)

- [ ] 5-1. 핵심 워크플로우 구현
  - [ ] `src/application/workflows/chat_workflow.py` (~300줄)
  - [ ] `src/application/workflows/crawl_workflow.py` (~300줄)
  - [ ] `src/application/workflows/insight_workflow.py` (~300줄)
  - [ ] API 라우트 → 워크플로우 호출 전환
- [ ] 5-2. 순환 의존성 해소 잔여
  - [ ] `export_handlers.py` → PeriodInsightAgent import 제거
  - [ ] 라우트 핸들러 → application workflow 호출
  - [ ] `container.py` 와이어링 중앙화

---

## Sprint 6: 보안 P2 + 기술 부채 (3~5일)

- [ ] 6-1. 보안 P2
  - [ ] PromptGuard 유니코드 우회 수정 (NFKC 정규화)
  - [ ] 출력 필터 패턴 보강 (AWS Key, GitHub PAT 등)
  - [ ] Embedding Cache MD5 → SHA-256
  - [ ] RAG 문서 인덱싱 크기 제한 (10MB)
  - [ ] 소셜 수집기 프록시 자격증명 마스킹
  - [ ] Query Router ReDoS 방어 (5,000자 제한)
- [ ] 6-2. 기술 부채
  - [ ] `@app.on_event("startup")` → lifespan 전환
  - [ ] `knowledge_graph.json` 동시 쓰기 보호
  - [ ] `TestResult` 클래스 이름 충돌 해결
  - [ ] Config 정리 (competitors.json dead config 등)

---

## Sprint 7: Eval Harness 고도화 (1주)

- [ ] 7-1. Judge 구현
  - [ ] `eval/judge/llm_judge.py` — GPT-4.1-mini 기반
  - [ ] `eval/judge/nli_judge.py` — 오프라인 NLI 모델
  - [ ] `eval/judge/__init__.py` — create_judge() 팩토리
- [ ] 7-2. Cost Tracking + Regression
  - [ ] `eval/cost_tracker.py` — 레이어별 토큰/USD 추적
  - [ ] `eval/regression.py` — 기준선 비교, 회귀 감지
  - [ ] CLI: `python -m eval compare`, `--baseline` 플래그
- [ ] 7-3. 테스트
  - [ ] `tests/eval/test_fuzzy_matching.py`
  - [ ] `tests/eval/test_judges.py`
  - [ ] `tests/eval/test_regression.py`

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
| Sprint 1 | 진행 중 | 0/3 |
| Sprint 2 | 대기 | 0/9 |
| Sprint 3 | 대기 | 0/4 |
| Sprint 4 | 대기 | 0/8 |
| Sprint 5 | 대기 | 0/8 |
| Sprint 6 | 대기 | 0/10 |
| Sprint 7 | 대기 | 0/9 |
| Sprint 8 | 대기 | 0/9 |
| Sprint 9 | 대기 | 0/9 |
| Sprint 10 | 대기 | 0/10 |
