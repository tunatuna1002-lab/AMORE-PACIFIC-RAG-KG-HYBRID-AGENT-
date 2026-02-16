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
