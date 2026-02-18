# AMORE RAG-KG-Ontology Agent: Karpathy 철학 기반 수정 계획

> **작성일**: 2026-02-18
> **기반 문서**: "Architecting the AMORE hybrid agent: from Karpathy minimalism to production-grade RAG-KG-Ontology systems"
> **상태**: 계획 수립 완료, 구현 대기

---

## 1. 배경 및 현황 분석

### 1.1 문서 제안 요약

Karpathy의 "이해는 압축이다" 철학을 적용하여:
- 코드베이스 ~90K → ~19K 라인 (79% 감소)
- Microsoft GraphRAG / OG-RAG / Neo4j 패턴 도입
- 7가지 안티패턴 제거
- Langfuse 모니터링 + 200+ Golden QA 세트

### 1.2 실제 코드베이스 정밀 탐색 결과

| 항목 | 문서 가정 | 실제 |
|------|----------|------|
| src/ 라인 수 | ~90,000 | **73,679** |
| 테스트 수 | 미언급 | **4,720개, 68,568줄** |
| 커버리지 | 미언급 | **70.80%** (목표 60% 달성) |
| Config 복잡도 | "5계층 우선순위" | **201줄 dataclass** (이미 단순) |
| Retriever 상속 | "6+파일, 깊은 상속" | **상속 없음, 합성 패턴** (이미 좋음) |
| Strategy Pattern | "도입 필요" | **이미 부분 구현** (`retrieval_strategy.py`) |
| 평가 프레임워크 | "기본부터 구축" | **5레벨 L1-L5 + LLM Judge + 회귀 테스트** (이미 강력) |
| brain.py | "God Object" | 1,706줄이지만 **SRP 8개 컴포넌트로 내부 분해 완료** |
| 19K 목표 | 가능 | **비현실적** (크롤링, SNS, 알림, 차트, 수출 등 실제 기능 미고려) |

### 1.3 문서가 맞는 부분 (진짜 문제)

1. **인텐트 분류 5중 구현** — 동일 기능이 5곳에 분산
2. **엔티티 추출 3중 구현** — 변경 시 3곳 동시 수정 필요
3. **에이전트 중복** — chatbot(1,667줄) + insight(1,344줄) = 3,011줄, 60% 겹침
4. **Golden QA 부족** — 40개 (권장 200+)
5. **죽은 Feature Flag** — 존재하지 않는 파일 참조

### 1.4 문서가 놓친 진짜 문제

1. **dashboard_api.py 모놀리스** — 3,900줄, 12개 라우트 모듈이 있지만 `include_router` 미호출
2. **Application Layer 부재** — `src/application/` 120줄, 비즈니스 로직이 Layer 4에 산재
3. **순환 의존성 23개** — core ↔ agents, tools ↔ agents, api ↔ tools/core
4. **미해결 보안 취약점 6개** — VULN-005 ~ VULN-012

---

## 2. 수정 계획 (6단계)

### Phase 1: 중복 통합 (최고 ROI) — 예상 2주

#### 1A. 인텐트 분류 통합 (5개 → 1개)

**현재 상태**: 5개의 독립적인 인텐트 분류 시스템이 존재

| # | 위치 | Enum | 타입들 |
|---|------|------|--------|
| 1 | `src/rag/hybrid_retriever.py` | `QueryIntent` | DIAGNOSIS, TREND, CRISIS, METRIC, GENERAL |
| 2 | `src/core/query_router.py` | `QueryCategory` | METRIC, TREND, COMPETITIVE, DIAGNOSTIC, GENERAL |
| 3 | `src/rag/router.py` | `QueryType` | DEFINITION, INTERPRETATION, COMBINATION, INSIGHT_RULE, DATA_QUERY, ANALYSIS |
| 4 | `src/core/decision_maker.py` | (암묵적) | LLM 기반 결정 로직 |
| 5 | `src/agents/hybrid_chatbot_agent.py` | (암묵적) | 인라인 분류 |

**작업 내용**:
```
1. src/core/intent.py 생성 (~150줄)
   - 통합 QueryIntent Enum (모든 5곳의 키워드 맵 병합)
   - classify_intent() 함수 (단일 진입점)
   - 기존 Enum → 새 Enum 매핑 테이블 (하위 호환)

2. 기존 파일에서 중복 분류 코드 제거
   - hybrid_retriever.py: classify_intent() 삭제 (lines 131-216, ~85줄)
   - query_router.py: classify() 중 키워드 매칭 부분을 intent.py 위임
   - router.py: classify_query() → intent.py 위임
   - hybrid_chatbot_agent.py: 인라인 분류 제거

3. 회귀 테스트 추가
   - tests/unit/core/test_intent.py (~100줄)
   - 기존 5곳의 키워드 → 동일 결과 매핑 검증
```

**수정 파일**: `src/core/intent.py` (신규), `src/rag/hybrid_retriever.py`, `src/core/query_router.py`, `src/rag/router.py`, `src/agents/hybrid_chatbot_agent.py`
**예상 변화**: -400줄 (중복 키워드 맵 + 분류 함수 제거)
**위험도**: 낮음 — 분류 출력을 매핑 가능, 회귀 테스트로 검증
**검증**: `python3 -m pytest tests/ -v` (4,720개 전체 통과)

#### 1B. 엔티티 추출 통합 (3개 → 1개)

**현재 상태**: 3개의 독립적인 엔티티 추출 구현

| # | 위치 | 라인 | 방식 |
|---|------|------|------|
| 1 | `src/rag/hybrid_retriever.py` (EntityExtractor) | ~300줄 | 키워드 매칭 + 정규식 |
| 2 | `src/rag/router.py` (RAGRouter.extract_entities) | ~80줄 | 키워드 매칭 |
| 3 | `src/agents/hybrid_chatbot_agent.py` | ~200줄 | 브랜드 별칭 + 키워드 |

**작업 내용**:
```
1. src/rag/entity_linker.py를 정규(canonical) 추출기로 확장
   - 이미 713줄, spaCy NER 지원
   - 3곳의 브랜드 별칭, 카테고리 맵, 지표 맵 병합
   - config/entities.json (이미 존재)에서 매핑 로드

2. 기존 파일에서 인라인 추출 제거
   - hybrid_retriever.py: EntityExtractor 클래스 삭제 → entity_linker 위임
   - router.py: extract_entities() → entity_linker 위임
   - hybrid_chatbot_agent.py: 브랜드 추출 로직 → entity_linker 위임

3. 검증
   - Golden QA L1(entity linking) 점수 비교: 변경 전/후
```

**수정 파일**: `src/rag/entity_linker.py`, `src/rag/hybrid_retriever.py`, `src/rag/router.py`, `src/agents/hybrid_chatbot_agent.py`
**예상 변화**: -500줄
**위험도**: 중간 — 엔티티 추출이 검색 품질에 직접 영향
**검증**: `python3 scripts/evaluate_golden.py --verbose` (L1 점수 회귀 없음)

#### 1C. 에이전트 공통 베이스 추출

**현재 상태**: chatbot(1,667줄) + insight(1,344줄) = 3,011줄, ~60% 코드 겹침

공유되는 파이프라인:
- 엔티티 추출 → KG 쿼리 → 온톨로지 추론 → RAG 검색 → 컨텍스트 융합 → 소스 귀속

**작업 내용**:
```
1. src/agents/base_hybrid_agent.py 생성 (~500줄)
   - 공유 파이프라인: extract → query_kg → reason → retrieve → fuse
   - Template Method 패턴: 하위 클래스가 특화 단계만 오버라이드

2. hybrid_chatbot_agent.py → 채팅 특화 로직만 (~400줄)
   - 대화 메모리, 후속 질문 생성, 스트리밍

3. hybrid_insight_agent.py → 배치 특화 로직만 (~400줄)
   - 메트릭 포맷팅, 리포트 생성, 내보내기
```

**수정 파일**: `src/agents/base_hybrid_agent.py` (신규), `src/agents/hybrid_chatbot_agent.py`, `src/agents/hybrid_insight_agent.py`
**예상 변화**: -1,300줄 (3,011 → ~1,300)
**위험도**: 중간 — 핵심 기능, 충분한 테스트 필요
**검증**: `python3 -m pytest tests/unit/agents/ -v`

---

### Phase 2: Retriever 전략 패턴 완성 — 예상 1주

#### 2A. 인텐트 기반 전략 선택

**현재 상태**: `retrieval_strategy.py`에 `RetrievalStrategy` Protocol이 정의되어 있고 Legacy/OWL 두 전략이 있지만, Feature Flag로만 선택됨 (인텐트 무시)

**작업 내용**:
```
1. retrieval_strategy.py 확장
   - 인텐트별 전략 매핑 (Phase 1A의 통합 인텐트 사용):
     * ENTITY_LOOKUP, RELATIONSHIP → Graph-heavy 전략
     * SEMANTIC, GENERAL → Vector-heavy 전략
     * COMPLIANCE → Ontology-grounded 전략 (OG-RAG 스타일)
     * TREND, COMPARISON, MULTI_HOP → Hybrid (Vector + Graph 병렬)

2. container.py에서 전략 DI 와이어링
   - get_retrieval_strategy(intent) → 적절한 Strategy 반환
```

**수정 파일**: `src/rag/retrieval_strategy.py`, `src/rag/hybrid_retriever.py`, `src/infrastructure/container.py`
**예상 변화**: +100줄 (전략 라우팅 로직)
**위험도**: 낮음 — 기존 전략 유지, 추가적 변경
**검증**: Golden QA 평가 + 각 타입별 대표 쿼리 수동 테스트

#### 2B. 죽은 Feature Flag 정리

**현재 상태**: `config/feature_flags.json`에 존재하지 않는 파일 참조
- `use_true_hybrid_retriever` → `true_hybrid_retriever.py` 없음
- `use_unified_retriever` → `unified_retriever.py` 없음

**작업 내용**:
```
1. config/feature_flags.json에서 죽은 플래그 제거
2. src/infrastructure/feature_flags.py에서 관련 코드 경로 제거
3. CLAUDE.md 참조 업데이트
```

**수정 파일**: `config/feature_flags.json`, `src/infrastructure/feature_flags.py`, `CLAUDE.md`
**예상 변화**: -30줄
**위험도**: 매우 낮음
**검증**: `python3 -m pytest tests/ -k "feature_flag" -v`

---

### Phase 3: Golden QA 확장 (40 → 200+) — 예상 2주

#### 3A. 데이터셋 확장

**현재 상태**: `eval/data/golden/laneige_golden_v1.jsonl` — 40개 항목
- metric: 9, product: 10, brand: 9, market: 9, general: 3
- 난이도: easy 15, medium 19, hard 6

**작업 내용**:
```
160+ QA 쌍 추가:
- 메트릭 (SoS, HHI, CPI): 30개 (easy/medium/hard)
- 제품 쿼리: 30개
- 브랜드 비교: 25개
- 시장 트렌드: 25개
- 멀티홉 추론: 20개 (예: "Lip Care에서 LANEIGE보다 SoS 높은 경쟁사 찾기")
- 엣지 케이스: 15개 (한영 혼용, 모호한 쿼리, 범위 밖 질문)
- 시간 쿼리: 15개 (주간 변동, 월간 트렌드)

난이도 분포: easy 30%, medium 50%, hard 20%
기존 JSONL 스키마 활용 (L1-L5 메타데이터 포함)
```

**수정 파일**: `eval/data/golden/laneige_golden_v2.jsonl` (신규)
**위험도**: 낮음 — 코드 변경 없음, 데이터 추가만
**검증**: `python3 -m pytest tests/eval/ -v` + `python3 scripts/evaluate_golden.py --verbose`

#### 3B. CI 평가 게이트 추가

**작업 내용**:
```
- CI에 Golden QA 평가 단계 추가
- 최소 임계값: L1 ≥ 0.7, L2 ≥ 0.6, L5 ≥ 0.5
- 회귀 감지: 기준선 대비 5% 이상 하락 시 경보
```

**수정 파일**: CI 설정, `scripts/evaluate_golden.py`
**예상 변화**: +50줄

---

### Phase 4: Dashboard API 분해 — 예상 1주

**현재 상태**: `src/api/dashboard_api.py`가 3,900줄 모놀리스. `src/api/routes/`에 12개 라우트 모듈이 존재하지만, `include_router`가 호출되지 않아 부분적으로 고아 상태.

**작업 내용**:
```
1. app_factory.py에서 12개 라우터 등록 확인/완성
2. dashboard_api.py의 인라인 엔드포인트 → 해당 라우트 모듈로 이동
3. dashboard_api.py를 시작 + 글로벌 예외 처리만 남김 (~200줄)
4. 중복된 인라인 엔드포인트 삭제
```

**라우트 모듈** (12개):
- `health.py`, `crawl.py`, `data.py`, `deals.py`, `alerts.py`
- `chat.py`, `brain.py`, `competitors.py`, `analytics.py`
- `sync.py`, `market_intelligence.py`, `signals.py`, `export.py`

**수정 파일**: `src/api/dashboard_api.py`, `src/api/app_factory.py`, `src/api/routes/*.py`
**예상 변화**: -3,000줄 (dashboard_api.py에서, 라우트 모듈로 재분배)
**위험도**: 높음 — 프로덕션 API; 모든 엔드포인트 테스트 필수
**검증**: 서버 기동 + 전체 엔드포인트 테스트
```bash
uvicorn src.api.dashboard_api:app --port 8001
python3 -m pytest tests/unit/api/ -v
# 수동: GET /api/health, GET /api/data, POST /api/v3/chat 등
```
**의존성**: 없음 (Phase 1-3과 독립)

---

### Phase 5: Application Layer 구현 — 예상 2주

#### 5A. 핵심 워크플로우 구현

**현재 상태**: `src/application/`이 120줄. 비즈니스 로직이 인프라 레이어에 산재.

**작업 내용**:
```
1. src/application/workflows/chat_workflow.py (~300줄)
   - classify → retrieve → reason → generate 오케스트레이션
   - Protocol 타입 의존성만 받음 (concrete class 아님)

2. src/application/workflows/crawl_workflow.py (~300줄)
   - scrape → store → update_kg → calculate_metrics 오케스트레이션

3. src/application/workflows/insight_workflow.py (~300줄)
   - gather_data → reason → generate_insights → format 오케스트레이션

4. API 라우트가 워크플로우 호출하도록 전환
   - routes/chat.py → ChatWorkflow.execute()
   - routes/crawl.py → CrawlWorkflow.execute()
```

**수정 파일**: `src/application/workflows/` (3개 파일), API 라우트 모듈, `src/infrastructure/container.py`
**예상 변화**: +900줄 (신규 워크플로우)
**위험도**: 중간 — DI 와이어링 주의
**검증**: `python3 -m pytest tests/ -v`
**의존성**: Phase 1C (공통 에이전트 베이스)가 선행되면 깔끔

#### 5B. 순환 의존성 해소 (23개 → 0개)

**작업 내용**:
```
1. brain.py: concrete agent import → AgentProtocol 사용
2. export_handlers.py: PeriodInsightAgent 직접 import 제거
3. 라우트 핸들러: infrastructure 직접 호출 → application workflow 호출
4. container.py에서 모든 와이어링 중앙화
```

**수정 파일**: ~15개 (core/, agents/, tools/, api/)
**위험도**: 중간 — 각 사이클 제거 후 테스트
**검증**: `python3 -c "import src.core.brain"` (순환 import 에러 없음)
**의존성**: Phase 5A 선행

---

### Phase 6: 보안 + 기술 부채 — 예상 1주

**현재 상태**: SECURITY_AUDIT_REPORT.md (2026-01-28) 기준 미해결 항목

| ID | Severity | 내용 | 수정 방법 |
|----|----------|------|----------|
| VULN-005 | HIGH | Docker non-root user | `Dockerfile`에 `USER nonroot` 추가 |
| VULN-006 | HIGH | API key 타이밍 공격 | `==` → `hmac.compare_digest()` |
| VULN-007 | HIGH | Chat 엔드포인트 미인증 | auth 미들웨어 추가 |
| VULN-008 | HIGH | Prompt injection 방어 | PromptGuard 강화 |
| VULN-011 | MEDIUM | Security headers 누락 | CSP, X-Frame-Options 추가 |
| VULN-012 | MEDIUM | Session ID 예측 가능 | `secrets.token_urlsafe()` 사용 |

**수정 파일**: `Dockerfile`, `src/api/app_factory.py`, `src/api/routes/chat.py`, `src/core/prompt_guard.py`, `src/memory/session.py`
**위험도**: 낮음~중간 (각 수정은 독립적)
**검증**: `python3 -m pytest tests/adversarial/ -v` + 보안 테스트

---

## 3. 실행 순서 및 의존성

```
Phase 1A (인텐트 통합) ──→ Phase 1B (엔티티 통합) ──→ Phase 1C (에이전트 베이스)
                                                              │
Phase 2A (전략 패턴) ← 1A 필요                                │
Phase 2B (죽은 플래그) ← 독립                                  │
                                                              ▼
Phase 3 (Golden QA) ← 독립                             Phase 5A (워크플로우)
Phase 4 (API 분해) ← 독립                                     │
Phase 6 (보안) ← 독립                                         ▼
                                                       Phase 5B (순환 의존성)
```

**권장 실행 순서**: `1A → 1B → 2B → 1C → 2A → 4 → 3 → 5A → 5B → 6`

병렬 가능한 조합:
- Phase 3 (Golden QA) + Phase 4 (API 분해) 동시 진행 가능
- Phase 6 (보안) 언제든 독립 실행 가능

---

## 4. 예상 최종 결과

| 지표 | Before | After | 변화 |
|------|--------|-------|------|
| src/ 라인 | 73,679 | ~68,000 | -8% |
| 인텐트 분류기 | 5개 | 1개 | -80% |
| 엔티티 추출기 | 3개 | 1개 | -67% |
| 에이전트 중복 | 3,011줄 (60% 겹침) | ~1,300줄 (0% 겹침) | -57% |
| dashboard_api.py | 3,900줄 | ~200줄 | -95% |
| Golden QA 항목 | 40개 | 200+개 | +400% |
| 순환 의존성 | 23개 | 0개 | -100% |
| 보안 취약점 | 6개 | 0개 | -100% |
| Application Layer | 120줄 | ~1,020줄 | +750% |

### 19K 라인이 아닌 이유

문서의 19K 목표는 클린룸 재작성을 가정하며 다음을 제외:
- Amazon 크롤링 인프라 (scraper 1,561줄, deals 624줄)
- SNS 수집기 (TikTok, Instagram, YouTube, Reddit — 4개 파일)
- Google Sheets 동기화 (721줄)
- 이메일/Telegram 알림 (927 + 별도)
- 차트 생성 (671줄), 리포트 생성 (1,257줄), 대시보드 내보내기 (1,521줄)
- IR 리포트 파싱 (741줄)
- 53개 도메인 비즈니스 규칙 (2,298줄)

이들은 **실제 프로덕션 기능**이지 **우발적 복잡성**이 아닙니다. 현실적 감소는 **~8%**이지만, **유지보수성과 코드 품질은 극적으로 개선**됩니다.

---

## 5. 각 Phase별 시작 체크리스트

### Phase 1A 시작 전 확인사항
```bash
# 현재 테스트 통과 확인
python3 -m pytest tests/ -v --tb=short

# 5개 인텐트 분류 위치 확인
grep -rn "class QueryIntent" src/
grep -rn "class QueryCategory" src/
grep -rn "class QueryType" src/
grep -rn "classify_intent\|classify_query\|classify(" src/core/ src/rag/ src/agents/
```

### Phase 1B 시작 전 확인사항
```bash
# 엔티티 추출 위치 확인
grep -rn "class EntityExtractor" src/
grep -rn "extract_entities" src/
grep -rn "BRAND_ALIASES\|brand_aliases" src/

# Golden QA L1 기준선 측정
python3 scripts/evaluate_golden.py --verbose 2>&1 | grep "L1"
```

### Phase 4 시작 전 확인사항
```bash
# dashboard_api.py 인라인 엔드포인트 확인
grep -n "@app\.\(get\|post\|put\|delete\)" src/api/dashboard_api.py | wc -l

# 라우트 모듈 등록 확인
grep -n "include_router" src/api/app_factory.py
grep -n "include_router" src/api/dashboard_api.py
```

---

## 6. 참고 문서

| 문서 | 위치 |
|------|------|
| 원본 아키텍처 제안 | `~/Desktop/바이브코딩/Architecting the AMORE hybrid agent...md` |
| 미래 작업 목록 | `docs/dev/FUTURE_WORK.md` |
| 보안 감사 결과 | `docs/security/SECURITY_AUDIT_REPORT.md` |
| 리팩토링 성과 | `docs/REFACTORING_RESULTS.md` |
| 프로젝트 구조 | `CLAUDE.md` §4 |
| Feature Flags | `config/feature_flags.json` |
| 평가 스키마 | `eval/schemas.py` |
| Golden QA | `eval/data/golden/laneige_golden_v1.jsonl` |
