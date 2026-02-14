# REFACTOR_PLAN.md

> 코드베이스 구조 분석 결과 및 리팩토링 권고사항
> 생성일: 2026-02-14

---

## 1. Import 의존성 그래프 (src/ 패키지 간)

### 아키텍처 다이어그램

```
Entry Points: dashboard_api.py, main.py
         │
         ▼
    ┌─── api/ ◄──────────────────────────┐
    │     │                               │
    │     ▼                               │
    │   core/ ──────► agents/ ──────► rag/
    │     │             │    ◄────────  │
    │     │             │               │
    │     ▼             ▼               ▼
    │   memory/      ontology/ ◄──── tools/
    │     │             │               │
    │     ▼             ▼               │
    │   monitoring/  domain/ ◄──────────┘
    │                   ▲
    │                   │
    └── infrastructure/ ┘
         shared/ (독립)
```

### 의존성 매트릭스 (From → To, import 횟수)

| From \ To | core | agents | rag | ontology | tools | domain | application | adapters | infra | api | memory | monitoring | shared |
|-----------|:----:|:------:|:---:|:--------:|:-----:|:------:|:-----------:|:--------:|:-----:|:---:|:------:|:----------:|:------:|
| **core** | - | 8 | - | 3 | 8 | 1 | - | - | - | - | 3 | 3 | - |
| **agents** | - | - | 10 | 4 | 7 | 7 | - | - | 2 | - | 1 | 3 | 3 |
| **rag** | - | - | - | 4 | - | 3 | - | - | 3 | - | - | 1 | - |
| **ontology** | - | - | - | - | - | 9 | - | - | - | - | - | - | - |
| **tools** | - | 1 | - | 1 | 7 | 1 | - | - | - | 2 | - | - | - |
| **domain** | - | - | - | - | - | 5 | - | - | - | - | - | - | - |
| **application** | - | 2 | 1 | - | - | 8 | - | - | - | - | - | - | - |
| **adapters** | - | 4 | 4 | - | - | - | - | - | - | - | - | - | - |
| **infrastructure** | 2 | 3 | 3 | 4 | - | 4 | - | - | - | - | - | - | - |
| **api** | 4 | 2 | 2 | - | 9 | 1 | - | - | - | - | - | - | - |
| **shared** | - | - | - | - | - | - | - | - | - | - | - | 1 | - |
| **dashboard_api.py** | 3 | - | - | - | 6 | - | - | - | - | 4 | - | - | - |
| **main.py** | 1 | - | - | - | - | - | - | - | - | - | - | 1 | - |

### 핵심 발견

- **domain/** 외부 의존 0건 → Clean Architecture 핵심 원칙 완벽 준수 ✅
- **memory/**, **monitoring/** → src 내부 import 0건 (완전 독립 모듈)
- **agents/** → 가장 높은 coupling (9개 패키지에서 import, 총 37건)
- **순환 의존 3건** 발견 (아래 §4 참조)

---

## 2. 진입점 4개 역할 비교

### 비교표

| 항목 | main.py | orchestrator.py | start.py | dashboard_api.py |
|------|---------|-----------------|----------|------------------|
| **역할** | CLI 배치/챗봇 | 하위호환 래퍼 | Railway 배포 | FastAPI 웹서버 |
| **라인 수** | 282 | 47 | 29 | 3,236 |
| **실행 방법** | `python main.py [--chat]` | import only | `python start.py` | `uvicorn dashboard_api:app` |
| **src/ 의존** | core, monitoring | batch_workflow만 | 없음 (uvicorn만) | api, core, tools 등 12+ |
| **엔드포인트** | 0 (CLI) | 0 (래퍼) | 0 (프록시) | 38개 REST API |
| **Reload** | N/A | N/A | False (프로덕션) | True (개발) |
| **Workers** | Single async | N/A | 1 (Railway 제약) | Auto |
| **인증** | 없음 | N/A | 없음 | API Key (보호 엔드포인트) |
| **에러 처리** | try/except + logging | 없음 | 없음 | Global handler + Telegram |

### 실행 흐름

```
Railway 프로덕션:  start.py → uvicorn.run("dashboard_api:app", reload=False, workers=1)
로컬 개발:        uvicorn dashboard_api:app --host 0.0.0.0 --port 8001 --reload
CLI 배치 작업:    python main.py           → Orchestrator → crawl/metrics/insight
CLI 대화 모드:    python main.py --chat    → UnifiedBrain REPL
레거시 import:    from orchestrator import Orchestrator  → src.core.batch_workflow
```

### 각 진입점 상세

**main.py** (282줄)
- argparse 기반 CLI: `--chat`, `--categories`, `--spreadsheet-id`, `--dry-run`
- `run_daily_workflow()`: Orchestrator로 배치 실행 (크롤링 → 분석 → 알림)
- `run_chatbot()`: UnifiedBrain REPL (status, errors, help, exit 명령)
- 종료 코드: 0 (성공), 1 (실패)

**orchestrator.py** (47줄)
- `src.core.batch_workflow`의 순수 re-export 래퍼
- `BatchWorkflow` → `Orchestrator` 별칭 제공
- 9개 클래스 re-export (WorkflowStep, ThinkResult 등)
- `if __name__` 블록 없음 — 실행 불가

**start.py** (29줄)
- Railway 전용 시작 스크립트
- `PORT` 환경변수 읽기 (기본값: 8001)
- `reload=False`, `workers=1`, `log_level="info"`

**dashboard_api.py** (3,236줄)
- FastAPI 메인 서버 (모놀리스)
- startup 훅: config 검증 → 크롤링 체크 → 스케줄러 → 작업 큐 → Telegram
- 38개 엔드포인트 (data, chat, crawl, export, alerts, deals, insights)
- Lazy import 패턴 사용 (deals_scraper, alert_service 등)

---

## 3. Dead Code 파일 목록

### 확인된 미사용 파일 (3개, 674줄)

| # | 파일 경로 | 라인 수 | 설명 | 삭제 근거 |
|---|----------|---------|------|-----------|
| 1 | `src/core/explainability.py` | 265 | 응답 설명 엔진 (`ExplainabilityEngine`, `ExplanationTrace`) | 프로젝트 전체에서 import 0건. 기능은 brain.py에서 별도 구현됨 |
| 2 | `src/core/query_processor.py` | 187 | 쿼리 처리 파이프라인 (`QueryProcessor`) | 프로젝트 전체에서 import 0건. AGENTS.md에만 문서 참조 존재 |
| 3 | `src/domain/interfaces/brain_components.py` | 225 | Brain SRP 프로토콜 5개 (`QueryProcessorProtocol`, `DecisionMakerProtocol`, `ToolCoordinatorProtocol`, `AlertManagerProtocol`, `ResponseGeneratorProtocol`) | 프로젝트 전체에서 import 0건. 실제 구현체(`decision_maker.py` 등)가 이 인터페이스를 사용하지 않음 |

### 삭제 시 영향

- **런타임 영향**: 없음 (어디서도 import하지 않으므로)
- **테스트 영향**: 없음 (이 파일들의 테스트도 존재하지 않음)
- **커버리지**: 분모 674줄 감소 → 커버리지 소폭 개선
- **대체 구현**: 이미 존재
  - `explainability.py` → `src/core/brain.py` (UnifiedBrain 내부)
  - `query_processor.py` → `src/core/brain.py` + `src/core/response_pipeline.py`
  - `brain_components.py` → `src/core/decision_maker.py`, `tool_coordinator.py`, `alert_manager.py`

### 삭제 명령

```bash
rm src/core/explainability.py
rm src/core/query_processor.py
rm src/domain/interfaces/brain_components.py
```

---

## 4. Clean Architecture 위반 및 개선 권고

### ✅ 정상 (위반 없음)

| 레이어 | 상태 | 설명 |
|--------|------|------|
| `domain/` | ✅ 완벽 | 외부 레이어 의존 0건 |
| `memory/` | ✅ 독립 | src 내부 import 0건 |
| `monitoring/` | ✅ 독립 | src 내부 import 0건 |
| `ontology/` | ✅ 양호 | domain만 참조 (9건) |

### ⚠️ 순환 의존 (3건)

| # | 관계 | 원인 파일 | 심각도 |
|---|------|----------|--------|
| 1 | `tools/` ↔ `agents/` | `tools/exporters/` → `PeriodInsightAgent` import | HIGH |
| 2 | `tools/` ↔ `api/` | `tools/exporters/` → `api.dependencies` import | MEDIUM |
| 3 | `core/` ↔ `agents/` | 양방향 참조 (core가 agent 생성, agent가 core 콜백) | MEDIUM |

**권고**: 순환 의존을 Interface(Protocol)로 분리하거나, 이벤트 기반 패턴으로 전환

### ⚠️ 레이어 바이패스 (3건)

| # | 위반 | 현재 | 올바른 방향 |
|---|------|------|-------------|
| 1 | `application/` → `agents/` 직접 import | `HybridChatbotAgent` 직접 사용 | `adapters/` 경유 |
| 2 | `application/` → `rag/` 직접 import | `HybridRetriever` 직접 사용 | `adapters/` 경유 |
| 3 | `infrastructure/` → `agents/` 구체 클래스 import | 구체 클래스 직접 생성 | Protocol 기반 DI |

### 개선 우선순위

```
[P0] Dead Code 삭제 (3파일, 674줄) — 즉시 실행 가능, 리스크 없음
[P1] 순환 의존 #1 해소 (tools ↔ agents) — 런타임 import 오류 위험
[P2] application 레이어 바이패스 수정 — adapters 경유로 변경
[P3] agents 패키지 coupling 감소 — 9개 → 4~5개 패키지 의존으로
```

---

## 부록: 패키지별 파일 수 및 라인 수

| 패키지 | 파일 수 | 역할 |
|--------|---------|------|
| `src/core/` | 30+ | 오케스트레이션, 스케줄링, 라우팅 |
| `src/agents/` | 10 | AI 에이전트 (챗봇, 인사이트, 크롤러) |
| `src/rag/` | 12 | RAG 검색 (retriever, reranker, chunker) |
| `src/ontology/` | 15+ | Knowledge Graph + 추론 엔진 |
| `src/tools/` | 25+ | 스크래퍼, 계산기, 알림, 내보내기 |
| `src/domain/` | 15+ | 엔티티, 인터페이스, 예외 |
| `src/application/` | 8 | 유스케이스/워크플로우 |
| `src/adapters/` | 6 | 어댑터 |
| `src/infrastructure/` | 6 | DI 컨테이너, 부트스트랩 |
| `src/api/` | 15+ | FastAPI 라우트 |
| `src/memory/` | 4 | 대화 메모리 |
| `src/monitoring/` | 4 | 로깅, 메트릭, 트레이싱 |
| `src/shared/` | 2 | 상수, LLM 클라이언트 |
