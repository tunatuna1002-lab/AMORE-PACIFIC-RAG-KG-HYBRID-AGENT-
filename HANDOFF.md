# Phase 3: dashboard_api.py 분해 — HANDOFF

> **Branch**: `refactor/phase3-api-modularize`
> **Status**: Steps 1-7 완료, Step 8 최종 검증 진행 중
> **이전 Phase**: Phase 1-2 완료, PR #9 merged to main

---

## 1. 완료된 작업

### Phase 3 결과 요약

| 항목 | Before | After |
|------|--------|-------|
| `dashboard_api.py` | 3,235줄 (33 inline endpoints) | **186줄** (thin shell) |
| `app_factory.py` 라우터 | 8개 등록 | **13개 등록** (전체) |
| 인라인 엔드포인트 | 33개 `@app.*` | **0개** |
| Migration tests | 4/23 active | **23/23 passed** |
| ruff errors | 0 | **0** |

### Step 완료 현황

- [x] **Step 0**: 동기화 분석 — 13개 라우트 파일 1:1 비교 완료
- [x] **Step 1**: Health + Crawl — crawl_router 등록, inline 제거
- [x] **Step 2**: Data + Historical — data.py 전면 재작성 (SQLite-first), ~650줄 inline 제거
- [x] **Step 3**: Deals — deals.py 재작성 (bulk save, alert_service), 5개 endpoint 이전
- [x] **Step 4**: Alerts — alerts.py 전면 재작성 (16개 endpoint + JWT helpers + HTML pages)
- [x] **Step 5**: Chat — chat.py 재작성 (v1+memory+v4+stream 4개 endpoint)
- [x] **Step 6**: Export — dead inline 코드 제거 (이미 등록된 export_router가 우선)
- [x] **Step 7**: dashboard_api.py 축소 → 186줄 thin shell

### dashboard_api.py 최종 구조 (186줄)

```
L1-53:   docstring (아키텍처 다이어그램)
L54-64:  imports (asyncio, logging, os, dotenv, fastapi, app_factory, brain, crawl_manager)
L65-77:  app = create_app()
L78-103: global_exception_handler (Telegram 알림)
L104-178: startup_event (config 검증, 크롤링 체크, 스케줄러, job queue, telegram)
L179-186: if __name__ == "__main__": uvicorn.run(...)
```

### app_factory.py 등록 라우터 (13개)

```python
health, crawl, data, deals, alerts, chat,
brain, competitors, analytics, sync,
market_intelligence, signals, export
+ telegram (optional)
```

---

## 2. 다음에 해야 할 작업

### Step 8: 최종 검증 (남은 항목)

- [x] `ruff check src/ dashboard_api.py` → 0 errors
- [x] Migration tests → 23/23 passed
- [ ] `python3 -m pytest tests/ -x --tb=short` → 통과율 확인
  - Pre-existing failures: `test_ir_rag_integration`, `test_llm_integration`, `test_rag_integration`
- [ ] PR 생성

---

## 3. 수정한 파일 목록

| 파일 | 변경 유형 | 요약 |
|------|-----------|------|
| `dashboard_api.py` | 축소 | 3,235줄 → 186줄 (thin shell) |
| `src/api/app_factory.py` | 수정 | 5개 라우터 추가 등록 (alerts, chat, crawl, data, deals) |
| `src/api/routes/data.py` | 덮어쓰기 | SQLite-first + Sheets fallback + historical endpoint |
| `src/api/routes/deals.py` | 덮어쓰기 | bulk save_deals + alert_service 연동 |
| `src/api/routes/alerts.py` | 덮어쓰기 | 16개 endpoint (v3/v4 settings + email verification + insight) |
| `src/api/routes/chat.py` | 덮어쓰기 | v1 chat + memory + v4 brain + v4 stream |
| `tests/unit/api/test_route_migration.py` | 수정 | 모든 23개 테스트 활성화 |
| `TODO.md` | 수정 | Steps 1-7 체크 완료 |
| `HANDOFF.md` | 덮어쓰기 | 최종 상태 반영 |

---

## 4. 참조

- **Pre-existing test failures** (Phase 3 변경과 무관):
  - `tests/test_ir_rag_integration.py::test_ir_document_metadata` — AttributeError
  - `tests/test_llm_integration.py::test_context_builder_only` — FileNotFoundError
  - `tests/test_rag_integration.py` — similar issues
- **Coverage threshold**: 14.2% (fail-under=15%) — pre-existing, not caused by Phase 3
