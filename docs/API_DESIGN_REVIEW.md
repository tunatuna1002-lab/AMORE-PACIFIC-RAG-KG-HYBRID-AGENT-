# API 설계 검토 보고서

> 검토일: 2026-02-14
> 검토 범위: `dashboard_api.py` + `src/api/` 전체 (59개 엔드포인트)
> 상태: 검토 완료, 코드 수정 미착수 (dashboard_api.py 분해 후 진행 예정)

---

## 목차

1. [아키텍처 현황](#1-아키텍처-현황)
2. [전체 엔드포인트 인벤토리](#2-전체-엔드포인트-인벤토리)
3. [발견된 문제](#3-발견된-문제)
4. [개선 계획](#4-개선-계획)
5. [우선순위 매트릭스](#5-우선순위-매트릭스)

---

## 1. 아키텍처 현황

### 1.1 API 레이어 구조

```
dashboard_api.py (3,900줄, 모놀리스)
  ├── create_app() 호출 → src/api/app_factory.py
  ├── 31개 엔드포인트 직접 정의
  └── 글로벌 예외 핸들러, 미들웨어

src/api/
  ├── app_factory.py          # 8개 라우터 등록
  ├── dependencies.py (547줄) # Auth + 세션 + RAG + JWT + 싱글톤
  ├── models.py (165줄)       # 공유 Pydantic 모델 (사실상 미사용)
  ├── middleware.py            # 보안 헤더
  ├── validators/
  │   └── input_validator.py   # 프롬프트 인젝션 방어
  └── routes/                  # 13개 라우트 파일
      ├── health.py            ✅ 등록됨
      ├── brain.py             ✅ 등록됨
      ├── competitors.py       ✅ 등록됨
      ├── analytics.py         ✅ 등록됨
      ├── sync.py              ✅ 등록됨
      ├── market_intelligence.py ✅ 등록됨
      ├── signals.py           ✅ 등록됨
      ├── export.py            ✅ 등록됨
      ├── chat.py              ❌ 미등록 (죽은 코드)
      ├── data.py              ❌ 미등록 (죽은 코드)
      ├── crawl.py             ❌ 미등록 (죽은 코드)
      ├── alerts.py            ❌ 미등록 (죽은 코드)
      └── deals.py             ❌ 미등록 (죽은 코드)
```

### 1.2 Clean Architecture 연결 상태

| 레이어 | 구현 상태 | API 연결 |
|--------|----------|----------|
| Domain Interfaces (24개 Protocol) | 잘 설계됨 | **미사용** |
| Application Workflows (5개) | 잘 설계됨 | **미사용** (어떤 라우트에서도 호출하지 않음) |
| DI Container (`infrastructure/container.py`) | 구현 완료 | **미사용** (라우트가 구체 클래스 직접 import) |
| API Routes | 동작 중 | 구체 구현체를 직접 import하여 CA 위반 |

### 1.3 라우터 등록 이중 경로

두 가지 등록 메커니즘이 공존:

- **`app_factory.py`**: health, brain, competitors, analytics, sync, market_intelligence, signals, export (8개) → 실제 동작
- **`routes/__init__.py`의 `include_routers()`**: chat, data, crawl, brain, export, deals, alerts (7개) → **호출되지 않음**

---

## 2. 전체 엔드포인트 인벤토리

### 2.1 dashboard_api.py 모놀리스 (31개)

| # | Method | Path | Auth | Rate Limit |
|---|--------|------|------|------------|
| 1 | GET | `/api/data` | - | - |
| 2 | POST | `/api/chat` | API Key | 10/min |
| 3 | DELETE | `/api/chat/memory/{session_id}` | - | - |
| 4 | GET | `/api/crawl/status` | - | - |
| 5 | POST | `/api/crawl/start` | API Key | - |
| 6 | GET | `/api/historical` | - | - |
| 7 | POST | `/api/export/docx` | - | - |
| 8 | POST | `/api/export/excel` | - | - |
| 9 | GET | `/api/v3/alert-settings` | - | - |
| 10 | POST | `/api/v3/alert-settings` | API Key | 5/min |
| 11 | POST | `/api/v3/alert-settings/revoke` | API Key | 5/min |
| 12 | POST | `/api/v4/subscribe` | - | 3/min |
| 13 | GET | `/api/v4/alert-settings` | - | - |
| 14 | PUT | `/api/v4/alert-settings` | - | 5/min |
| 15 | DELETE | `/api/v4/alert-settings` | - | 5/min |
| 16 | GET | `/api/v3/alerts` | - | - |
| 17 | POST | `/api/v4/chat` | API Key | 10/min |
| 18 | POST | `/api/v4/chat/stream` | **없음** | 10/min |
| 19 | GET | `/api/deals` | - | - |
| 20 | GET | `/api/deals/summary` | - | - |
| 21 | POST | `/api/deals/scrape` | API Key | - |
| 22 | GET | `/api/deals/alerts` | - | - |
| 23 | POST | `/api/deals/export` | - | - |
| 24 | GET | `/api/alerts/status` | - | - |
| 25 | POST | `/api/alerts/send` | **없음** | - |
| 26 | POST | `/api/alerts/test` | **없음** | - |
| 27 | POST | `/api/alerts/send-verification` | - | 3/min |
| 28 | POST | `/api/alerts/verify-email` | - | 10/min |
| 29 | GET | `/api/alerts/confirm-email` | - | - |
| 30 | GET | `/api/alerts/verification-status` | - | - |
| 31 | POST | `/api/alerts/send-insight-report` | **없음** | - |

### 2.2 등록된 라우트 파일 (28개)

| # | Method | Path | Auth | Rate Limit | 파일 |
|---|--------|------|------|------------|------|
| 32 | GET | `/` | - | - | health.py |
| 33 | GET | `/dashboard` | - | - | health.py |
| 34 | GET | `/api/health` | - | - | health.py |
| 35 | GET | `/api/health/deep` | - | - | health.py |
| 36 | GET | `/api/v4/brain/status` | - | - | brain.py |
| 37 | POST | `/api/v4/brain/scheduler/start` | API Key | - | brain.py |
| 38 | POST | `/api/v4/brain/scheduler/stop` | API Key | - | brain.py |
| 39 | POST | `/api/v4/brain/autonomous-cycle` | API Key | - | brain.py |
| 40 | POST | `/api/v4/brain/check-alerts` | **없음** | - | brain.py |
| 41 | GET | `/api/v4/brain/stats` | - | - | brain.py |
| 42 | POST | `/api/v4/brain/mode` | API Key | - | brain.py |
| 43 | GET | `/api/competitors` | - | - | competitors.py |
| 44 | GET | `/api/competitors/brands` | - | - | competitors.py |
| 45 | GET | `/api/category/kpi` | - | - | analytics.py |
| 46 | GET | `/api/sos/category` | - | - | analytics.py |
| 47 | GET | `/api/sos/brands` | - | - | analytics.py |
| 48 | GET | `/api/sos/trend` | - | - | analytics.py |
| 49 | GET | `/api/sos/trend/competitors-avg` | - | - | analytics.py |
| 50 | GET | `/api/sync/status` | - | - | sync.py |
| 51 | GET | `/api/sync/dates` | - | - | sync.py |
| 52 | GET | `/api/sync/download/{date}` | - | - | sync.py |
| 53 | POST | `/api/sync/upload` | Body key (비표준) | - | sync.py |
| 54-59 | ... | `/api/market-intelligence/*`, `/api/signals/*`, `/api/export/*` | 일부 API Key | - | 각 파일 |

### 2.3 미등록 라우트 파일 (죽은 코드, 5개 파일)

| 파일 | 엔드포인트 수 | 모놀리스와 차이 |
|------|-------------|----------------|
| `chat.py` | 1 | Auth 없음 (모놀리스는 있음), 응답 필드 2개 누락 |
| `data.py` | 2 | historical 응답 형식 상이 |
| `crawl.py` | 2 | prefix 없음 (경로 불일치) |
| `alerts.py` | 6 | rate limit 누락, get_app_state_manager 중복 |
| `deals.py` | 5 | 알림 처리 로직 상이 (AlertAgent vs alert_service) |

---

## 3. 발견된 문제

### 3.1 인증 보안 취약점 (CRITICAL)

59개 엔드포인트 중 **10개만** API Key 인증이 적용되어 있음. 파괴적/비용 발생 엔드포인트에 인증 누락:

| 엔드포인트 | 위험도 | 위치 |
|-----------|-------|------|
| `DELETE /api/signals/clear` | 전체 신호 데이터 삭제 | `signals.py:203` |
| `POST /api/alerts/send` | 실제 이메일/텔레그램 발송 | `dashboard_api.py:2544` |
| `POST /api/alerts/test` | 테스트 알림 발송 | `dashboard_api.py:2595` |
| `POST /api/alerts/send-insight-report` | 인사이트 리포트 발송 | `dashboard_api.py:3082` |
| `POST /api/export/docx` | LLM 비용 + 서버 부하 | `export.py:359` |
| `POST /api/export/analyst-report` | LLM 비용 + 외부 API | `export.py:605` |
| `POST /api/export/excel` | DB 전체 내보내기 | `export.py:996` |
| `POST /api/v4/chat/stream` | LLM 비용 (v4/chat은 인증 필요) | `dashboard_api.py:2210` |

추가 문제:
- `sync.py`의 upload 엔드포인트는 Body에 `api_key`를 받는 비표준 방식 사용
- `POST /api/v4/brain/check-alerts`는 POST인데 인증 없음 (같은 라우터의 다른 POST는 인증 있음)

### 3.2 에러 처리 비일관성 (HIGH)

4가지 서로 다른 에러 처리 패턴이 혼재:

**패턴 A — HTTPException (올바름)**
```
사용: health.py, sync.py, export.py (일부), market_intelligence.py
→ 적절한 HTTP 상태 코드 반환
```

**패턴 B — success: False + HTTP 200 (문제)**
```
사용: analytics.py (전체), deals.py (일부), alerts.py (일부), brain.py (일부)
→ 실패인데 200 OK 반환, 프론트엔드가 body를 파싱해야 에러 감지
```

**패턴 C — error 필드 반환 + HTTP 200 (문제)**
```
사용: brain.py (일부), competitors.py (일부)
→ {error: str(e)} 형태, 상태 코드 없음
```

**패턴 D — 성공 모델에 에러 삽입 + HTTP 200 (문제)**
```
사용: chat.py (BrainChatResponse with brain_mode="error")
→ 정상 응답 모델에 에러를 숨김
```

### 3.3 모놀리스 + 라우트 파일 중복 (HIGH)

16개 엔드포인트가 `dashboard_api.py`와 미등록 라우트 파일에 이중 정의:

| 경로 | 모놀리스 | 라우트 파일 | 주요 차이 |
|------|---------|-----------|----------|
| `POST /api/v4/chat` | Auth + 10/min | Auth 없음 + 30/min | 인증, rate limit, 응답 필드 |
| `GET /api/historical` | SQLite 우선 폴백 | Sheets 우선 | 데이터 소스 순서, 응답 형식 |
| `POST /api/export/docx` | 5섹션 단순 보고서 | 6섹션 + TOC + 외부 신호 | **완전히 다른 구현** |
| `POST /api/export/excel` | 450줄, 3가지 코드 경로 | `storage.export_to_excel()` 위임 | 구현 복잡도 |
| `POST /api/deals/scrape` | `alert_service` 사용 | `AlertAgent` 사용 | 알림 처리 |

### 3.4 Clean Architecture 완전 우회 (HIGH)

**Application Workflows 미사용**: `ChatWorkflow`, `CrawlWorkflow`, `InsightWorkflow`, `AlertWorkflow`, `BatchWorkflow` — 5개 모두 어떤 API 라우트에서도 import되지 않음.

**DI Container 미사용**: `src/infrastructure/container.py`가 제공하는 싱글톤/팩토리를 어떤 라우트도 사용하지 않음.

**구체 클래스 직접 import** (13개 라우트 중 13개 전부):
- `chat.py` → `from src.core.brain import get_initialized_brain`
- `brain.py` → `from src.core.brain import UnifiedBrain, BrainMode`
- `analytics.py` → `from src.tools.storage.sqlite_storage import get_sqlite_storage`
- `export.py` → 6개 구체 클래스 직접 import
- 기타 전부 동일 패턴

**도메인 예외 미번역**: 8개 도메인 예외 타입 중 `DataValidationError` → 400만 번역됨. 나머지는 모두 `except Exception as e` → 500으로 처리:
- `NetworkError` (url, status_code 메타데이터) → 502/504로 번역해야 함
- `LLMAPIError` (model, is_retryable 메타데이터) → 503으로 번역해야 함
- `ScraperError` (category, error_type 메타데이터) → 503으로 번역해야 함
- `ConfigurationError` → 500 + 구체적 메시지로 번역해야 함

### 3.5 Pydantic 모델 중복 (MEDIUM)

`src/api/models.py`에 정의된 모델이 라우트 파일에서 재정의되며, 필드 불일치 존재:

| 모델 | models.py | 라우트 파일 | 차이 |
|------|----------|-----------|------|
| `BrainChatResponse` | 10 필드 | `chat.py`: 8 필드 | `suggestions`, `query_type` 누락 |
| `ExportRequest` | 3 필드 | `export.py`: 4 필드 | `include_external_signals` 추가 |
| `AnalystReportRequest` | 2 필드 | `export.py`: 3 필드 | `include_external_signals` 추가 |
| `AlertSettingsRequest` | models.py 정의 | `alerts.py` 재정의 | 동일하나 중복 |
| `DealsRequest/Response` | models.py 정의 | `deals.py` 재정의 | 동일하나 중복 |
| `MI Status/Layer` | models.py 정의 | `market_intelligence.py` 재정의 | 동일하나 중복 |

→ `models.py`는 사실상 **죽은 코드** (등록된 라우트가 로컬 모델 사용)

### 3.6 Rate Limiting 부재 (MEDIUM)

| 엔드포인트 유형 | Rate Limit | 비용/위험 |
|---------------|-----------|----------|
| `POST /api/export/analyst-report` | 없음 | LLM + 외부 API |
| `POST /api/export/docx` | 없음 | LLM + DOCX 생성 |
| `POST /api/crawl/start` (모놀리스) | 없음 | Playwright 브라우저 세션 |
| `POST /api/deals/scrape` | 없음 | Playwright 브라우저 세션 |
| `POST /api/market-intelligence/collect` | 없음 | 외부 API 호출 |
| `POST /api/signals/fetch/rss` | 없음 | 외부 RSS 요청 |
| `POST /api/signals/fetch/reddit` | 없음 | Reddit API 호출 |

모놀리스의 chat 엔드포인트만 rate limiting 적용 (10/min).

### 3.7 입력 검증 불일치 (MEDIUM)

- `ChatRequest` (models.py): `max_length = 10,000`
- `InputValidator` (input_validator.py): `MAX_LENGTH = 2,000`
- **프로덕션 chat 엔드포인트** (dashboard_api.py): `InputValidator` **미사용**
- 프롬프트 인젝션 방어가 실제 프로덕션 채팅에 적용되지 않음

### 3.8 라우트 핸들러 내 Raw SQL (MEDIUM)

| 파일 | Raw SQL 수 | 위치 |
|------|-----------|------|
| `analytics.py` | 6개 | kpi, sos/category, sos/trend 등 전체 엔드포인트 |
| `sync.py` | 4개 + `ALTER TABLE` | status, dates, download, upload (스키마 마이그레이션 포함) |
| `deals.py` | 2개 | alerts 조회, export |
| `health.py` | 1개 | deep health check |

→ Repository 패턴으로 분리해야 함

### 3.9 API 버전 비일관성 (LOW)

| 버전 | 엔드포인트 |
|------|----------|
| 무버전 | `/api/data`, `/api/crawl/*`, `/api/deals/*`, `/api/export/*`, `/api/alerts/*`, `/api/competitors`, `/api/sos/*`, `/api/sync/*`, `/api/signals/*`, `/api/market-intelligence/*` |
| v3 | `/api/v3/alert-settings`, `/api/v3/alerts` |
| v4 | `/api/v4/chat`, `/api/v4/chat/stream`, `/api/v4/subscribe`, `/api/v4/alert-settings`, `/api/v4/brain/*` |

v1, v2는 존재하지 않음. v3는 alert 전용, v4는 chat+brain 전용.

### 3.10 export.py 비대 (LOW)

`src/api/routes/export.py`가 **1,251줄**이며, ~900줄이 DOCX 생성 비즈니스 로직:
- 섹션 생성, 차트 삽입, 신호 분류, 참고문헌 필터링
- 기존 `src/tools/exporters/report_generator.py`의 `DocxReportGenerator`와 중복
- 라우트 파일이 아닌 서비스 레이어에 있어야 할 로직

### 3.11 dependencies.py 비대 (LOW)

`src/api/dependencies.py` (547줄)에 과도한 책임 집중:
- Auth (적절) + Rate limiter (적절)
- 인메모리 세션 관리 → `src/memory/`로 이동해야 함
- RAG 컨텍스트 빌딩 → workflow/retriever로 이동해야 함
- 동적 후속 질문 생성 → `src/agents/suggestion_engine.py`에 이미 존재
- 감사 로깅 → `src/monitoring/`로 이동해야 함
- JWT 헬퍼 → 전용 auth 모듈로 이동해야 함
- 싱글톤 게터 → Container 사용해야 함

추가로 `load_dashboard_data()`와 `get_sheets_writer()`가 `data.py`에도 중복 정의,
JWT 함수가 `dashboard_api.py`에도 중복 정의.

---

## 4. 개선 계획

### Phase 1: 보안 & 일관성 (긴급, dashboard_api.py 분해와 병행 가능)

| # | 작업 | 대상 파일 | 위험도 |
|---|------|----------|--------|
| 1.1 | 파괴적 엔드포인트 8개에 `Depends(verify_api_key)` 추가 | signals.py, export.py, dashboard_api.py | LOW |
| 1.2 | `sync.py` Body api_key를 표준 헤더 인증으로 교체 | sync.py | LOW |
| 1.3 | `InputValidator`를 프로덕션 chat 엔드포인트에 적용, max_length 통일 (2,000) | dashboard_api.py, models.py | LOW |
| 1.4 | 비용 높은 엔드포인트에 rate limiting 추가 | export.py, signals.py, 각 라우트 | LOW |
| 1.5 | 미등록 5개 라우트 파일에 DEPRECATED 마킹 | chat/data/crawl/alerts/deals.py | LOW |
| 1.6 | `routes/__init__.py`의 미사용 `include_routers()` 제거 | `__init__.py` | LOW |

### Phase 2: 아키텍처 정렬 (dashboard_api.py 분해 완료 후)

| # | 작업 | 대상 파일 | 위험도 |
|---|------|----------|--------|
| 2.1 | 통일 에러 핸들러 도입 (도메인 예외 → HTTP 상태 코드) | 신규 error_handlers.py, app_factory.py | MEDIUM |
| 2.2 | `{success: False}` + HTTP 200 패턴을 HTTPException으로 점진적 교체 | analytics.py, deals.py, brain.py, alerts.py | MEDIUM |
| 2.3 | Pydantic 모델을 `models.py`로 통합 (Single Source of Truth) | models.py, 각 라우트 파일 | LOW |
| 2.4 | Raw SQL을 Repository 클래스로 추출 | 신규 repository 파일, analytics.py, sync.py, deals.py | LOW-MEDIUM |
| 2.5 | `export.py` DOCX 로직을 `src/tools/exporters/`로 분리 | export.py, 신규/기존 exporter 파일 | LOW |
| 2.6 | DI Container 연결: 라우트 → Container → Protocol 구현체 | container.py, 각 라우트 파일 | MEDIUM |
| 2.7 | Application Workflow 경유: 라우트 → Workflow → Domain | workflow 파일, 각 라우트 파일 | MEDIUM |

### Phase 3: 완성도 (장기)

| # | 작업 | 대상 파일 | 위험도 |
|---|------|----------|--------|
| 3.1 | 통일 응답 Envelope (`ApiResponse[T]`) 도입 | models.py, 전체 라우트 | MEDIUM |
| 3.2 | API 버전 전략 수립 (현재 무버전 → v1 alias) | 전체 라우트, app_factory.py | LOW |
| 3.3 | `dependencies.py` 분해 (Auth만 남기고 각 레이어로 이동) | dependencies.py, 각 대상 모듈 | MEDIUM |
| 3.4 | OpenAPI 문서 개선 (tags, summary, response_model 추가) | 전체 라우트 | LOW |

---

## 5. 우선순위 매트릭스

```
         높은 영향
            │
   Phase 1  │  Phase 2.1-2.2
   (보안)    │  (에러 핸들러)
            │
낮은 노력 ──┼── 높은 노력
            │
   Phase 1  │  Phase 2.6-2.7
   (정리)    │  (CA 연결)
            │
         낮은 영향
```

### 실행 순서 권장

```
1. dashboard_api.py 분해 완료 (별도 세션, 진행 중)
   ↓
2. Phase 1.1-1.4: 보안 수정 (인증, 검증, rate limit)
   ↓
3. Phase 1.5-1.6: 죽은 코드 정리
   ↓
4. Phase 2.1-2.2: 에러 핸들러 통일
   ↓
5. Phase 2.3-2.5: 모델 통합 + SQL 추출 + export 분리
   ↓
6. Phase 2.6-2.7: DI Container + Workflow 연결
   ↓
7. Phase 3: 응답 Envelope, 버전 전략, 문서화
```

---

## 부록: 인증 적용 현황 전체 표

| # | Method | Path | 현재 인증 | 권장 |
|---|--------|------|----------|------|
| 1 | GET | `/` | - | 불필요 |
| 2 | GET | `/dashboard` | - | 불필요 |
| 3 | GET | `/api/health` | - | 불필요 |
| 4 | GET | `/api/health/deep` | - | 불필요 |
| 5 | GET | `/api/data` | - | 불필요 |
| 6 | GET | `/api/historical` | - | 불필요 |
| 7 | POST | `/api/chat` | API Key | 유지 |
| 8 | DELETE | `/api/chat/memory/{session_id}` | - | **API Key 추가** |
| 9 | GET | `/api/crawl/status` | - | 불필요 |
| 10 | POST | `/api/crawl/start` | API Key | 유지 |
| 11 | POST | `/api/v4/chat` | API Key | 유지 |
| 12 | POST | `/api/v4/chat/stream` | **없음** | **API Key 추가** |
| 13 | POST | `/api/export/docx` | **없음** | **API Key 추가** |
| 14 | POST | `/api/export/analyst-report` | **없음** | **API Key 추가** |
| 15 | POST | `/api/export/excel` | **없음** | **API Key 추가** |
| 16 | POST | `/api/export/async/start` | **없음** | **API Key 추가** |
| 17 | POST | `/api/alerts/send` | **없음** | **API Key 추가** |
| 18 | POST | `/api/alerts/test` | **없음** | **API Key 추가** |
| 19 | POST | `/api/alerts/send-insight-report` | **없음** | **API Key 추가** |
| 20 | POST | `/api/v3/alert-settings` | API Key | 유지 |
| 21 | POST | `/api/v3/alert-settings/revoke` | API Key | 유지 |
| 22 | POST | `/api/deals/scrape` | API Key | 유지 |
| 23 | POST | `/api/deals/export` | **없음** | **API Key 추가** |
| 24 | POST | `/api/market-intelligence/collect` | API Key | 유지 |
| 25 | POST | `/api/signals/fetch/rss` | **없음** | **API Key 추가** |
| 26 | POST | `/api/signals/fetch/reddit` | **없음** | **API Key 추가** |
| 27 | POST | `/api/signals/manual` | **없음** | **API Key 추가** |
| 28 | POST | `/api/signals/trend-radar` | **없음** | **API Key 추가** |
| 29 | DELETE | `/api/signals/clear` | **없음** | **API Key 추가** |
| 30 | POST | `/api/sync/upload` | Body key | **헤더 인증으로 변경** |
| 31 | POST | `/api/v4/brain/scheduler/start` | API Key | 유지 |
| 32 | POST | `/api/v4/brain/scheduler/stop` | API Key | 유지 |
| 33 | POST | `/api/v4/brain/autonomous-cycle` | API Key | 유지 |
| 34 | POST | `/api/v4/brain/check-alerts` | **없음** | **API Key 추가** |
| 35 | POST | `/api/v4/brain/mode` | API Key | 유지 |
| - | GET | 나머지 GET 엔드포인트 | - | 불필요 (읽기 전용) |
