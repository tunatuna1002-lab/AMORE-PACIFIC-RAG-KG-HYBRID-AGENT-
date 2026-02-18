# AMORE RAG-KG Hybrid Agent — 통합 장기 로드맵

> 작성일: 2026-02-18
> 목적: 12개 계획 파일을 하나의 실행 가능한 로드맵으로 통합

---

## Context

12개의 계획 파일이 산재하고 Karpathy 수정 계획이 Phase 1-2 중간에서 중단되었다. 보안, AI 품질, 코드 구조, 평가 프레임워크 등 모든 영역이 중요하므로, 의존성과 위험도를 고려한 단계별 실행 계획을 수립한다.

### 현재 상태 (2026-02-18 기준)

| 항목 | 수치 |
|------|------|
| 테스트 | 4,816개 |
| 커버리지 | 70.80% |
| git staged (미커밋) | Phase 1A/1C/2A/2B 완료 코드 |
| 보안 P0 | 4건 중 1건 이미 수정 (API Key HTML) |
| brain.py 순환 의존성 | 이미 정리됨 (Protocol DI) |

---

## Sprint 0: 계획 영구 저장 (실행 첫 단계)

> 목표: 이 로드맵을 영구 파일로 저장하여 세션이 끝나도 이어서 진행 가능하게

### 0-1. 마스터 로드맵 저장
- 이 계획을 `docs/plans/master-roadmap.md`로 복사 저장

### 0-2. 진행 상태 파일 생성
- `docs/plans/roadmap-progress.md` 생성 — 전체 체크리스트
- Sprint별 하위 작업 체크박스 포함

### 0-3. MEMORY.md 업데이트
- `~/.claude/projects/.../memory/MEMORY.md`에 로드맵 참조 추가
- 새 세션에서 자동으로 컨텍스트 로드됨

---

## Sprint 1: 현재 작업 마무리 (1~2일)

> 목표: 중단된 Karpathy Phase 1-2를 완결하고 커밋

### 1-1. Phase 1B 완료: 엔티티 추출 통합
- `src/rag/entity_linker.py` — 3곳의 브랜드/카테고리/지표 맵 병합
- `src/rag/hybrid_retriever.py` — EntityExtractor 인라인 코드 → entity_linker 위임
- `src/rag/router.py` — extract_entities() → entity_linker 위임
- `src/agents/hybrid_chatbot_agent.py` — 브랜드 추출 → entity_linker 위임
- 테스트: 기존 테스트 회귀 확인 + entity_linker 단위 테스트

### 1-2. Ruff 잔여 4개 수정
- `scripts/evaluate_golden.py:155-156` — C401: `set(...)` → `{...}` (2건)
- `src/infrastructure/bootstrap.py:103,112` — F401: 미사용 import 제거 (2건)

### 1-3. 전체 테스트 + 커밋
- `python3 -m pytest tests/ -v` — 4,816개 전체 통과 확인
- Phase 1A/1B/1C/2A/2B + Ruff 일괄 커밋

### 검증
```bash
python3 -m pytest tests/ -v --tb=short
ruff check . && ruff format --check .
```

---

## Sprint 2: 보안 수정 (3~4일)

> 목표: P0/P1 보안 취약점 해소 (docs/security/plan.md)

### 2-1. P0 Critical (3건 — 1건은 이미 수정됨)
| ID | 내용 | 파일 |
|----|------|------|
| ~~2.1~~ | ~~API Key HTML 노출~~ | ~~이미 수정됨 (DASHBOARD_READ_TOKEN)~~ |
| 2.2 | API_KEY 미설정 시 Fail-Open | `src/api/dependencies.py:29-31` |
| 2.3 | JWT_SECRET_KEY 보안 미검증 | `src/api/dependencies.py:463-465` |
| 2.4 | InputValidator 채팅 엔드포인트 미적용 | `src/api/routes/chat.py`, `dashboard_api.py` |

### 2-2. P1 High (6건)
| 내용 | 파일 |
|------|------|
| Docker non-root user | `Dockerfile` |
| 보안 헤더 (CSP, HSTS 등) | `src/api/middleware.py` |
| Rate Limiting 범위 확대 | `src/api/routes/crawl.py`, `export.py` 등 |
| Session ID UUID 전체 사용 | `src/memory/session.py:78` |
| KG 경로 검증 | `src/ontology/knowledge_graph.py:400-451` |
| SSL 검증 활성화 | `src/tools/calculators/exchange_rate.py:135,158` |

### 검증
```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/adversarial/ -v
curl -s http://localhost:8001/dashboard | grep -c "API_KEY"  # 0이어야 함
```

---

## Sprint 3: Dashboard API 분해 확인 + 정리 (3~5일)

> 목표: Karpathy Phase 4 — dashboard_api.py 모놀리스 해체
> 참조: `docs/plans/karpathy-modification-plan.md` Phase 4

### 사전 확인 (이미 모듈화 진행 상태일 수 있음)
- `src/api/routes/` 12개 라우터 모듈 → `include_router` 등록 확인
- `dashboard_api.py` 인라인 엔드포인트 잔여 수 확인
- `app_factory.py` 라우터 등록 완성도 확인

### 작업
- dashboard_api.py 인라인 엔드포인트 → 해당 라우트 모듈로 이동
- dashboard_api.py를 시작 + 글로벌 예외 처리만 남김 (~200줄 목표)
- 라우트 모듈 12개 모두 app_factory.py에서 등록

### 검증
```bash
uvicorn src.api.dashboard_api:app --port 8001  # 서버 기동
python3 -m pytest tests/unit/api/ -v
```

---

## Sprint 4: Golden QA 확장 + Eval 기초 (1주)

> 목표: Karpathy Phase 3 + Eval Harness Phase 1
> 참조: `karpathy-modification-plan.md` Phase 3, `.omc/plans/eval-harness-enhancement.md` Phase 1

### 4-1. Golden QA 확장 (40→200+)
- `eval/data/golden/laneige_golden_v2.jsonl` 신규 생성
- 160+ QA 쌍: 메트릭(30), 제품(30), 브랜드(25), 시장(25), 멀티홉(20), 엣지(15), 시간(15)
- 난이도 분포: easy 30%, medium 50%, hard 20%

### 4-2. Eval Harness: Fuzzy Matching
- `eval/metrics/base.py` — fuzzy_match, resolve_alias, set_f1_fuzzy 추가
- `eval/metrics/l1_query.py` — alias_map + fuzzy_threshold 통합
- `eval/schemas.py` — CostBreakdown, RegressionComparison 스키마 확장

### 검증
```bash
python3 -m pytest tests/eval/ -v
python3 scripts/evaluate_golden.py --verbose
```

---

## Sprint 5: Application Layer + 순환 의존성 (1주)

> 목표: Karpathy Phase 5 — Clean Architecture 강화
> 참조: `karpathy-modification-plan.md` Phase 5, `docs/dev/FUTURE_WORK.md` §5-6

### 5-1. 핵심 워크플로우 구현
- `src/application/workflows/chat_workflow.py` (~300줄)
- `src/application/workflows/crawl_workflow.py` (~300줄)
- `src/application/workflows/insight_workflow.py` (~300줄)
- API 라우트 → 워크플로우 호출로 전환

### 5-2. 순환 의존성 해소 (잔여 분)
- brain.py는 이미 정리됨 — 나머지 확인:
  - `export_handlers.py` → PeriodInsightAgent 직접 import 제거
  - 라우트 핸들러 → application workflow 호출로 전환
  - `container.py` 와이어링 중앙화

### 검증
```bash
python3 -c "import src.core.brain"  # 순환 import 없음
python3 -m pytest tests/ -v
```

---

## Sprint 6: 보안 P2 + 기술 부채 (3~5일)

> 목표: Karpathy Phase 6 + FUTURE_WORK 잔여
> 참조: `docs/security/plan.md` §4, `docs/dev/FUTURE_WORK.md`

### 6-1. 보안 P2 (6건)
- PromptGuard 유니코드 우회 수정 (NFKC 정규화)
- 출력 필터 패턴 보강 (AWS Key, GitHub PAT 등)
- Embedding Cache MD5 → SHA-256
- RAG 문서 인덱싱 크기 제한 (10MB)
- 소셜 수집기 프록시 자격증명 마스킹
- Query Router ReDoS 방어 (5,000자 제한)

### 6-2. 기술 부채
- `dashboard_api.py` `@app.on_event("startup")` → lifespan 전환
- `knowledge_graph.json` 동시 쓰기 보호
- `TestResult` 클래스 이름 충돌 해결
- Config 정리 (competitors.json dead config 등)

### 검증
```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/adversarial/ -v
```

---

## Sprint 7: Eval Harness 고도화 (1주)

> 목표: LLM Judge, Regression Testing, Cost Tracking
> 참조: `.omc/plans/eval-harness-enhancement.md` Phase 2-5

### 7-1. Judge 구현
- `eval/judge/llm_judge.py` — GPT-4.1-mini 기반 groundedness/relevance
- `eval/judge/nli_judge.py` — 오프라인 NLI 모델 (cross-encoder)
- `eval/judge/__init__.py` — create_judge() 팩토리

### 7-2. Cost Tracking + Regression
- `eval/cost_tracker.py` — 레이어별 토큰/USD 추적
- `eval/regression.py` — 기준선 비교, 회귀 감지
- CLI: `python -m eval compare`, `--baseline` 플래그

### 7-3. 테스트
- `tests/eval/test_fuzzy_matching.py`
- `tests/eval/test_judges.py`
- `tests/eval/test_regression.py`

### 검증
```bash
python3 -m pytest tests/eval/ -v
python3 scripts/evaluate_golden.py --verbose --baseline reports/baseline.json
```

---

## Sprint 8: 학술 기준 Ontology 업그레이드 — Track A+B (1~2주)

> 목표: OWL 2 심화 + BM25 Sparse Retrieval
> 참조: `docs/plans/academic-standards-ontology-rag-upgrade-plan.md`

### 8-1. Track A: OWL 온톨로지 심화
- A-1: OWL Class Restriction (DominantBrand ≡ Brand ⊓ ∃hasShareOfShelf[≥0.30])
- A-2: inverseOf 선언
- A-3: Disjointness 공리
- A-4: Cardinality 제약
- A-5: Hard Validation (T-Box 위반 시 거부)

### 8-2. Track B: RAG BM25+RRF
- B-1: BM25 Sparse Retrieval 추가 (rank_bm25)
- B-2: RRF 병합 구현 (k=60)
- B-3: ConfidenceFusion에 RRF 전략 추가
- B-4: Self-RAG 검색 필요성 판단

### 검증
```bash
python3 -m pytest tests/ -v
python3 scripts/evaluate_golden.py --verbose  # L1-L5 점수 비교
```

---

## Sprint 9: 학술 기준 — Track C+D (1~2주)

> 목표: Multi-hop + SPARQL + 통합

### 9-1. Track C: Multi-hop + SPARQL
- C-1: Multi-hop refine_search 액션
- C-2: IRCoT 패턴
- C-3: SPARQL 기본 지원
- C-4: 문장별 인라인 인용 (AIS)

### 9-2. Track D: 통합 + 검증
- D-1: IRI 체계 도입
- D-2: KG IRI 마이그레이션
- D-3: HybridRetriever Self-RAG 통합
- D-4: OWL Consistency Check
- D-5: 전체 통합 테스트

### 검증
```bash
python3 -m pytest tests/ -v
# OWL Consistency Check
# BM25+RRF recall@10 비교
# Multi-hop 2-hop 질문 5개
# AIS 문장별 출처 매핑률 80%+
```

---

## Sprint 10: 마무리 + God Objects 분할 (1주)

> 목표: 나머지 FUTURE_WORK + 보안 P3

### 10-1. God Objects 추가 분할
- `src/ontology/business_rules.py` (1,540줄) → 규칙 카테고리별 분리
- `src/ontology/knowledge_graph.py` (1,514줄) → CRUD/Query 분리
- `src/agents/hybrid_chatbot_agent.py` (1,353줄) → base 상속 후 추가 정리

### 10-2. 보안 P3 (4건)
- TrustedHost 미들웨어
- CSRF 보호
- 의존성 보안 스캔 자동화 (safety, pip-audit, bandit)
- 세션 데이터 암호화

### 10-3. DI 전환 잔여
- hybrid_insight_agent.py → ExternalSignalCollector DI
- period_insight_agent.py → PeriodAnalyzer DI
- api/routes/deals.py → AlertAgent DI

---

## 전체 타임라인 요약

| Sprint | 기간 | 내용 |
|--------|------|------|
| Sprint 0 | 즉시 | 계획 영구 저장 |
| Sprint 1 | 1~2일 | 현재 작업 마무리 + 커밋 |
| Sprint 2 | 3~4일 | 보안 P0/P1 |
| Sprint 3 | 3~5일 | Dashboard API 분해 |
| Sprint 4 | 1주 | Golden QA + Eval 기초 |
| Sprint 5 | 1주 | Application Layer + 순환 의존성 |
| Sprint 6 | 3~5일 | 보안 P2 + 기술 부채 |
| Sprint 7 | 1주 | Eval Harness 고도화 |
| Sprint 8 | 1~2주 | 학술 Ontology Track A+B |
| Sprint 9 | 1~2주 | 학술 Ontology Track C+D |
| Sprint 10 | 1주 | 마무리 + God Objects |

### 의존성 그래프

```
Sprint 1 (현재 작업) → Sprint 2 (보안) → Sprint 3 (API 분해)
                    │                              │
                    ├→ Sprint 4 (Golden QA) ────────┤
                    │                              │
                    └→ Sprint 5 (App Layer) ←───────┘
                                │
                    Sprint 6 (기술부채) ←── Sprint 5
                                │
                    Sprint 7 (Eval) ←── Sprint 4
                                │
                    Sprint 8-9 (학술) ←── Sprint 1 + Sprint 7
                                │
                    Sprint 10 (마무리) ←── 전부
```

### 출처 계획 파일 매핑

| Sprint | 출처 |
|--------|------|
| 1 | `docs/plans/karpathy-modification-plan.md` Phase 1B, `docs/plans/ruff-lint-cleanup-plan.md` |
| 2 | `docs/security/plan.md` §2-3 |
| 3 | `docs/plans/karpathy-modification-plan.md` Phase 4 |
| 4 | `docs/plans/karpathy-modification-plan.md` Phase 3, `.omc/plans/eval-harness-enhancement.md` Phase 1 |
| 5 | `docs/plans/karpathy-modification-plan.md` Phase 5, `docs/dev/FUTURE_WORK.md` §5-6 |
| 6 | `docs/plans/karpathy-modification-plan.md` Phase 6, `docs/security/plan.md` §4, `docs/dev/FUTURE_WORK.md` §4 |
| 7 | `.omc/plans/eval-harness-enhancement.md` Phase 2-5 |
| 8-9 | `docs/plans/academic-standards-ontology-rag-upgrade-plan.md` |
| 10 | `docs/dev/FUTURE_WORK.md` §8, `docs/security/plan.md` §5 |
