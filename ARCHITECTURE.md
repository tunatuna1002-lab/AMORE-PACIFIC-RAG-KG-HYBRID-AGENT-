# AMORE RAG-Ontology Hybrid Agent Architecture

## Overview

Amazon US 베스트셀러 데이터 분석을 위한 하이브리드 AI 에이전트 시스템입니다.
RAG(Retrieval-Augmented Generation)와 Knowledge Graph + Ontology 추론을 결합하여
정확하고 맥락에 맞는 인사이트를 제공합니다.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Entry Points                                  │
├─────────────────────────────────────────────────────────────────┤
│  main.py           │  dashboard_api.py    │  start.py           │
│  (CLI Interface)   │  (FastAPI Server)    │  (Railway Deploy)   │
└────────┬───────────┴──────────┬───────────┴─────────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────────────────────┐
│  orchestrator.py │    │    unified_orchestrator.py      │
│  (Batch Workflow)│    │    (LLM-based Query Handler)    │
│                  │    │                                  │
│  Think-Act-Observe│    │  Query Analysis → Tool Selection │
│  Loop Pattern    │    │  → Agent Execution → Response    │
└────────┬─────────┘    └─────────────┬───────────────────┘
         │                            │
         └──────────┬─────────────────┘
                    │
    ┌───────────────┼───────────────────────┐
    │               │                       │
    ▼               ▼                       ▼
┌─────────┐   ┌───────────┐         ┌─────────────┐
│ Agents  │   │ Ontology  │         │    RAG      │
├─────────┤   ├───────────┤         ├─────────────┤
│Crawler  │   │Knowledge  │         │Hybrid       │
│Storage  │   │  Graph    │         │ Retriever   │
│Metrics  │   │Reasoner   │         │Context      │
│*Hybrid* │   │Relations  │         │ Builder     │
│ Insight │   │Business   │         │Router       │
│*Hybrid* │   │  Rules    │         │Templates    │
│ Chatbot │   │Schema     │         │             │
└────┬────┘   └─────┬─────┘         └──────┬──────┘
     │              │                      │
     └──────────────┴──────────────────────┘
                    │
    ┌───────────────┼───────────────────────┐
    │               │                       │
    ▼               ▼                       ▼
┌─────────┐   ┌───────────┐         ┌─────────────┐
│  Tools  │   │  Memory   │         │ Monitoring  │
├─────────┤   ├───────────┤         ├─────────────┤
│Scraper  │   │Session    │         │Logger       │
│Sheets   │   │History    │         │Tracer       │
│Email    │   │Context    │         │Metrics      │
│Calc     │   │           │         │             │
└─────────┘   └───────────┘         └─────────────┘
```

## Module Descriptions

### Entry Points

| File | Purpose | Usage |
|------|---------|-------|
| `main.py` | CLI 인터페이스 | `python main.py workflow` / `python main.py chat` |
| `dashboard_api.py` | FastAPI 서버 | 대시보드 백엔드 API |
| `orchestrator.py` | 배치 워크플로우 | 크롤링 → 저장 → 분석 → 인사이트 파이프라인 |
| `start.py` | Railway 배포 | 환경변수 PORT 처리 |

### Core Modules (`src/core/`)

| Module | Purpose | Status |
|--------|---------|--------|
| `unified_orchestrator.py` | LLM 기반 쿼리 처리 오케스트레이터 | **Active** |
| `llm_orchestrator.py` | 대체 LLM 오케스트레이터 | Active (Legacy) |
| `simple_chat.py` | 단순화된 채팅 서비스 | **Active** |
| `models.py` | 핵심 데이터 모델 | **Active** |
| `confidence.py` | 신뢰도 평가 | **Active** |
| `cache.py` | 응답 캐싱 | **Active** |
| `state.py` / `state_manager.py` | 상태 관리 | **Active** |
| `context_gatherer.py` | 컨텍스트 수집 | **Active** |
| `tools.py` | 도구 정의 | **Active** |
| `response_pipeline.py` | 응답 생성 | **Active** |
| `rules_engine.py` | 규칙 기반 결정 | **Active** |
| `crawl_manager.py` | 크롤링 상태 관리 | **Active** |

### Agent Modules (`src/agents/`)

| Module | Purpose | Status |
|--------|---------|--------|
| `hybrid_chatbot_agent.py` | KG+Ontology+RAG 챗봇 | **Active (Recommended)** |
| `hybrid_insight_agent.py` | KG+Ontology+RAG 인사이트 | **Active (Recommended)** |
| `crawler_agent.py` | Amazon 스크래핑 | **Active** |
| `storage_agent.py` | 데이터 저장 | **Active** |
| `metrics_agent.py` | 메트릭 계산 | **Active** |
| `alert_agent.py` | 알림 생성 | Active |
| `chatbot_agent.py` | 레거시 챗봇 | **Deprecated** → HybridChatbotAgent |
| `insight_agent.py` | 레거시 인사이트 | **Deprecated** → HybridInsightAgent |

### Ontology Modules (`src/ontology/`)

| Module | Purpose |
|--------|---------|
| `knowledge_graph.py` | Knowledge Graph 구현 (Neo4j 스타일 인메모리) |
| `reasoner.py` | Ontology 추론 엔진 |
| `relations.py` | 관계 타입 정의 |
| `schema.py` | Pydantic 스키마 |
| `business_rules.py` | 비즈니스 규칙 정의 |

### RAG Modules (`src/rag/`)

| Module | Purpose |
|--------|---------|
| `hybrid_retriever.py` | KG + RAG 하이브리드 검색 |
| `context_builder.py` | 컨텍스트 조립 |
| `retriever.py` | 문서 검색 |
| `router.py` | 쿼리 라우팅 |
| `templates.py` | 응답 템플릿 |

### Memory Modules (`src/memory/`)

| Module | Purpose |
|--------|---------|
| `session.py` | 세션 관리 |
| `history.py` | 실행 히스토리 |
| `context.py` | 컨텍스트 추적 |

### Monitoring Modules (`src/monitoring/`)

| Module | Purpose |
|--------|---------|
| `logger.py` | 구조화된 로깅 |
| `tracer.py` | 실행 추적 |
| `metrics.py` | 성능 메트릭 |

### Tools (`src/tools/`)

| Module | Purpose |
|--------|---------|
| `amazon_scraper.py` | Amazon 스크래핑 |
| `sheets_writer.py` | Google Sheets 통합 |
| `email_sender.py` | 이메일 발송 |
| `metric_calculator.py` | 메트릭 계산 |
| `dashboard_exporter.py` | 대시보드 내보내기 |

## Data Flow

### 1. Batch Workflow (orchestrator.py)

```
Crawl → Store → Update KG → Calculate Metrics → Generate Insights → Export
  │        │         │              │                  │              │
  ▼        ▼         ▼              ▼                  ▼              ▼
Amazon  JSON/     Knowledge    Brand KPIs,      Hybrid Insight   Google
 API    Sheets    Graph +      Product          Agent + RAG      Sheets
                  Ontology     Rankings                          + Email
```

### 2. Chat Query Flow (unified_orchestrator.py)

```
User Query → Query Analysis → Tool Selection → Agent Execution → Response
     │              │               │                │              │
     ▼              ▼               ▼                ▼              ▼
  Natural        RAG Router    Function        Hybrid Chatbot   Formatted
  Language      + Entity       Calling         + KG + Ontology    Answer
               Extraction                       Reasoning
```

## API Endpoints

### Dashboard API (dashboard_api.py)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/health` | GET | Railway health check |
| `/api/data` | GET | 대시보드 데이터 조회 |
| `/api/chat` | POST | v1 챗봇 (RAG 기반) |
| `/api/v2/chat` | POST | v2 챗봇 (Unified Orchestrator) |
| `/api/v3/chat` | POST | v3 챗봇 (Simple Chat) |
| `/api/crawl/status` | GET | 크롤링 상태 |
| `/api/crawl/start` | POST | 크롤링 시작 |
| `/api/export/docx` | POST | DOCX 리포트 생성 |
| `/api/v3/alert-settings` | GET/POST | 알림 설정 |
| `/dashboard` | GET | 대시보드 HTML 서빙 |

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
GOOGLE_SHEETS_SPREADSHEET_ID=...

# Optional (Railway)
PORT=8001
```

### Config Files

- `config/thresholds.json`: 메트릭 임계값 설정
- `railway.toml`: Railway 배포 설정
- `.env`: 환경 변수 (로컬)

## Deprecated Modules

아래 모듈들은 `_deprecated/` 폴더로 이동되었습니다:

| Module | Reason | Replacement |
|--------|--------|-------------|
| `brain.py` | Level 4 자율 시스템 설계 포기 | `unified_orchestrator.py` |
| `decision_maker.py` | 기능 통합됨 | `unified_orchestrator.py` |
| `query_agent.py` | 미사용 | `hybrid_chatbot_agent.py` |
| `workflow_agent.py` | 기능 중복 | `orchestrator.py` |

## Design Patterns

### 1. Think-Act-Observe Loop
`orchestrator.py`에서 사용하는 배치 워크플로우 패턴

### 2. LLM-based Orchestration
`unified_orchestrator.py`에서 사용하는 쿼리 처리 패턴

### 3. Hybrid Retrieval
`hybrid_retriever.py`에서 KG + RAG를 결합한 검색 패턴

### 4. Context Building
`context_builder.py`에서 Ontology 추론 + 현재 데이터 + 가이드라인을 조합하는 패턴

## Testing

```bash
# 전체 테스트
pytest tests/

# 특정 테스트
pytest tests/test_hybrid_integration.py -v

# 커버리지
pytest --cov=src tests/
```

## Development Guidelines

1. **새 에이전트 추가 시**: `HybridInsightAgent`/`HybridChatbotAgent` 패턴 참고
2. **Ontology 확장 시**: `src/ontology/schema.py`에 스키마 추가
3. **비즈니스 규칙 추가 시**: `src/ontology/business_rules.py` 수정
4. **새 도구 추가 시**: `src/tools/`에 모듈 추가 후 `src/core/tools.py`에 등록

## Version History

- **v1.0**: 기본 RAG 기반 챗봇
- **v2.0**: Ontology + Knowledge Graph 통합 (Hybrid)
- **v2.1**: Unified Orchestrator + 리팩토링
