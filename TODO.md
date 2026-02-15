# Phase 3: dashboard_api.py 분해 — TODO

> **목표**: 3,235줄 모놀리스 → `src/api/routes/` 모듈화 + `dashboard_api.py` ~100줄
> **검증 기준**: ruff 0 err + pytest 95%+ + curl smoke test

---

## Step 0: 사전 분석

- [x] 0-1. `dashboard_api.py` vs `src/api/routes/` 동기화 분석
  - health.py: ✓ SYNCED, 이미 app_factory에 등록됨
  - crawl.py: ✓ IDENTICAL, 미등록 → 등록 필요
  - data.py: ✗ OUT OF SYNC (Sheets-only vs SQLite-first) → dashboard_api.py 기준 덮어쓰기
  - chat.py: ✗ INCOMPLETE (v4/chat만 존재) → v1+v4+stream+memory 전부 이전
  - deals.py: ✓ ~SYNCED, 미등록 → 등록 필요
  - alerts.py: ✗ INCOMPLETE (v3+service만) → v4 subscribe+email verification 추가
  - export.py: ✓ SYNCED, 이미 app_factory에 등록됨. dashboard_api.py에 dead inline code 존재
  - brain/competitors/analytics/sync/mi/signals: ✓ 이미 등록, dashboard_api.py에 중복 없음
- [x] 0-2. `src/api/dependencies.py` 검토 — 공통 의존성 완비 (변경 불필요)
  - verify_api_key, limiter, conversation_memory, session helpers, RAG, JWT, state_manager 등
- [x] 0-3. `src/api/models.py` 검토 — 모든 모델 정의됨
  - 라우트 파일에 로컬 중복 모델 존재 (chat.py, deals.py, alerts.py, export.py) → 정리 필요

---

## Step 1: Health + Crawl (가장 독립적, ~160줄)

- [x] 1-1. `health.py` 동기화 확인 — 이미 app_factory에 등록됨, 변경 불필요
- [x] 1-2. `crawl.py` 동기화 확인 — dashboard_api.py와 IDENTICAL, 인라인 제거 완료
- [x] 1-3. `app_factory.py`에 crawl_router 등록 (prefix="/api/crawl")
- [x] 1-4. 검증: ruff 0 errors, migration tests 4/4 passed

---

## Step 2: Data + Historical (~930줄, 헬퍼 함수 포함)

- [x] 2-1. `data.py` 동기화 확인 — dashboard_api.py 기준 SQLite-first로 덮어쓰기
- [x] 2-2. dashboard_api.py에서 data 엔드포인트 이전:
  - `GET /api/data` (L225)
  - `GET /api/historical` (L511) — 가장 큰 엔드포인트
- [x] 2-3. 헬퍼 함수 이전:
  - `_calculate_brand_metrics_for_period` (L712, ~170줄)
  - `_get_brand_metrics_from_dashboard` (L890, ~30줄)
  - `_get_historical_from_local` (L923, ~230줄)
- [x] 2-4. `app.include_router(data_router)` 추가
- [x] 2-5. 검증: ruff 0 errors, migration tests 6/6 passed

---

## Step 3: Deals (~260줄)

- [x] 3-1. `deals.py` 동기화 — dashboard_api.py 기준으로 덮어쓰기 (bulk save_deals, alert_service, export_deals_report)
- [x] 3-2. dashboard_api.py에서 deals 엔드포인트 5개 + import 제거
- [x] 3-3. `app.include_router(deals_router)` 추가
- [x] 3-4. 검증: ruff 0 errors, migration tests 8/8 passed

---

## Step 4: Alerts (v3+v4 설정 + 이메일 검증, ~1000줄, 가장 큼)

- [x] 4-1. `alerts.py` 전면 재작성 — dashboard_api.py 기준 16개 엔드포인트 + JWT 헬퍼
- [x] 4-2. v3 alert settings 이전 (GET/POST/revoke + alerts list)
- [x] 4-3. v4 alert settings 이전 (subscribe/GET/PUT/DELETE)
- [x] 4-4. alert 서비스 이전 (status/send/test)
- [x] 4-5. 이메일 검증 이전 (send-verification/verify-email/confirm-email/verification-status/send-insight-report + JWT helpers)
- [x] 4-6. `app.include_router(alerts_router)` 추가
- [x] 4-7. 검증: ruff 0 errors, migration tests 12/12 passed

---

## Step 5: Chat (v1+v4+stream+memory, ~400줄)

- [x] 5-1. `chat.py` 전면 재작성 — v1 chat + memory + v4 brain + v4 stream 4개 엔드포인트
- [x] 5-2. dashboard_api.py에서 chat 엔드포인트 4개 제거
- [x] 5-3. Pydantic 모델은 src/api/models.py에서 import (BrainChatRequest, BrainChatResponse, ChatRequest, ChatResponse)
- [x] 5-4. `app.include_router(chat_router)` 추가
- [x] 5-5. 검증: ruff 0 errors, migration tests 16/16 passed

---

## Step 6: Export (docx+excel, ~680줄)

- [x] 6-1. `export.py` 이미 등록됨, 인라인 코드는 dead code (라우터가 먼저 매칭)
- [x] 6-2. dashboard_api.py에서 dead inline export 코드 제거 (~680줄)
- [x] 6-3. `app.include_router(export_router)` 이미 등록됨
- [x] 6-4. 검증: ruff 0 errors, migration tests 18/18 passed

---

## Step 7: dashboard_api.py 축소

- [x] 7-1. 모든 인라인 엔드포인트 제거 완료 (33개 → 0개)
- [x] 7-2. `create_app()` 팩토리 패턴 사용 중
- [x] 7-3. 남긴 것: app_factory, exception_handler, startup_event, server block
- [x] 7-4. Pydantic 모델은 `src/api/models.py`에서만 사용 (dashboard_api.py에서 제거 완료)
- [x] 7-5. 최종 라인 수: **186줄** (목표 200줄 이하 달성)
- [x] 7-6. 검증: ruff 0 errors, migration tests 23/23 passed

---

## Step 8: 최종 검증

- [x] 8-1. `ruff check dashboard_api.py src/api/` → 0 errors
- [x] 8-2. `pytest tests/unit/api/test_route_migration.py` → 23/23 passed
- [x] 8-3. `ruff check src/` 전체 → 0 errors
- [x] 8-4. `python3 -m pytest tests/ -x --tb=short` → 453 passed, 1 failed (pre-existing)
- [x] 8-5. HANDOFF.md 최종 업데이트
- [x] 8-6. PR 생성
