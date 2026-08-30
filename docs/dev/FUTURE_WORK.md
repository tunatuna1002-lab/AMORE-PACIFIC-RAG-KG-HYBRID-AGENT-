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
