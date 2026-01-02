# AMORE Pacific RAG-KG Hybrid Agent

> Amazon US 베스트셀러 분석을 위한 자율 AI 에이전트 시스템

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[한국어](#한국어) | [English](#english)

---

# 한국어

## 프로젝트 개요

AMORE Pacific LANEIGE 브랜드의 Amazon US 시장 경쟁력 분석을 위한 AI 에이전트 시스템입니다.

### 핵심 기능

| 기능 | 설명 |
|------|------|
| **자동 크롤링** | 매일 KST 06:00 Amazon Top 100 자동 수집 (5개 카테고리) |
| **KPI 분석** | SoS, HHI, CPI 등 전략 지표 계산 |
| **AI 챗봇** | Knowledge Graph + 키워드 RAG 기반 Q&A |
| **인사이트 생성** | LLM 기반 전략 인사이트 자동 생성 |

### 모니터링 카테고리

- Beauty & Personal Care
- Skin Care
- Lip Care
- Lip Makeup
- Face Powder

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn, Pydantic |
| **LLM** | OpenAI GPT-4.1-mini (via LiteLLM) |
| **Hybrid RAG** | HybridRetriever (KnowledgeGraph + OntologyReasoner + DocumentRetriever) |
| **크롤링** | Playwright (Chromium headless) |
| **데이터** | Pandas, Google Sheets API |
| **배포** | Docker, Railway |

### Hybrid RAG 시스템 상세

`HybridRetriever`는 3개 컴포넌트를 통합:
1. **KnowledgeGraph** - Triple Store 기반 지식 그래프 (브랜드/제품/카테고리 관계)
2. **OntologyReasoner** - 비즈니스 규칙 기반 추론 엔진
3. **DocumentRetriever** - 키워드 기반 가이드 문서 검색 (4개 MD 파일)

**참조 문서** (`docs/guides/`):
- Strategic Indicators Definition.md
- Metric Interpretation Guide.md
- Indicator Combination Playbook.md
- Home Page Insight Rules.md

> ChromaDB 벡터 검색은 코드에 존재하나 현재 **비활성화** 상태

---

## 핵심 모듈

| 모듈 | 파일 | 설명 |
|------|------|------|
| **UnifiedBrain** | `src/core/brain.py` | 자율 스케줄러, 에이전트 조율 |
| **HybridRetriever** | `src/rag/hybrid_retriever.py` | KG + Ontology + RAG 통합 검색 |
| **KnowledgeGraph** | `src/ontology/knowledge_graph.py` | Triple Store 지식 그래프 |
| **OntologyReasoner** | `src/ontology/reasoner.py` | 비즈니스 규칙 추론 엔진 |
| **AmazonScraper** | `src/tools/amazon_scraper.py` | Playwright 크롤러 |

### 자동 스케줄러

```python
# src/core/brain.py - AutonomousScheduler
# 한국시간(KST) 기준으로 동작
KST = timezone(timedelta(hours=9))

schedules = [
    {"id": "daily_crawl", "hour": 6, "minute": 0},  # KST 06:00 자동 크롤링
    {"id": "check_data_freshness", "interval_hours": 1}  # 1시간마다 데이터 신선도 체크
]
```

**상태 파일:** `data/scheduler_state.json` (서버 재시작 시에도 상태 유지)

---

## API 엔드포인트

### 주요 API

| Method | Endpoint | 설명 | 인증 |
|--------|----------|------|------|
| GET | `/api/health` | 헬스 체크 | - |
| GET | `/api/data` | 대시보드 데이터 | - |
| GET | `/dashboard` | 대시보드 UI | - |
| POST | `/api/chat` | v1 챗봇 | - |
| POST | `/api/v3/chat` | v3 챗봇 (권장) | - |
| POST | `/api/crawl/start` | 크롤링 시작 | API Key |
| GET | `/api/v4/brain/status` | 스케줄러 상태 | - |

### API Key 인증

```bash
curl -X POST "https://your-app.railway.app/api/crawl/start" \
  -H "X-API-Key: your-api-key"
```

---

## 설치 및 실행

```bash
# 1. 클론
git clone https://github.com/tunatuna1002-lab/AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-.git
cd AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-

# 2. 가상환경
python -m venv venv
source venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt
playwright install chromium

# 4. 환경 변수 (.env)
OPENAI_API_KEY=sk-...
API_KEY=your-api-key
AUTO_START_SCHEDULER=true

# 5. 실행
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001
```

**접속:** http://localhost:8001/dashboard

---

## 배포

### Railway

1. https://railway.app 에서 GitHub 연결
2. 환경 변수: `OPENAI_API_KEY`, `API_KEY`, `AUTO_START_SCHEDULER=true`
3. 도메인 생성

### Docker

```bash
docker build -t amore-agent .
docker run -p 8001:8001 -e OPENAI_API_KEY=sk-... amore-agent
```

---

## Audit Trail

챗봇 대화가 `./logs/chatbot_audit_YYYY-MM-DD.log`에 자동 기록됩니다.

---

## 트러블슈팅: 자동 크롤링 스케줄러

### 문제 1: 메서드명 불일치 (2026-01-03 해결)

**증상:** 서버 로그에 에러 발생
```
ERROR:src.core.brain:Scheduled task error: crawl_workflow - 'CrawlManager' object has no attribute 'run_full_crawl'
```

**원인:** `brain.py` 스케줄러에서 `crawl_manager.run_full_crawl()` 호출하지만, 실제 메서드는 `start_crawl()`

**해결:** `brain.py:1096, 1109`에서 `run_full_crawl()` → `start_crawl()`로 수정

---

### 문제 2: 시간대 불일치 (2026-01-03 해결)

**증상:** 한국시간 06:00 이후에도 크롤링이 실행되지 않음

**원인:**
- Railway 서버는 UTC 기준 동작
- `is_today_data_available()`이 서버 시간(UTC) 기준으로 "오늘" 판단
- UTC 1월 2일 22:00 = KST 1월 3일 07:00인데, 데이터 날짜가 1월 2일이면 "오늘 데이터 있음"으로 판단

**해결:**
1. `crawl_manager.py`에 `KST = timezone(timedelta(hours=9))` 추가
2. 모든 날짜 체크를 한국시간 기준으로 변경 (`get_kst_today()`)
3. 스케줄러도 KST 기준으로 작업 시간 판단

```python
# 변경 전 (UTC 기준)
today = date.today().isoformat()

# 변경 후 (KST 기준)
kst_today = datetime.now(KST).date().isoformat()
```

---

### 문제 3: 스케줄러 상태 초기화 (2026-01-03 해결)

**증상:** 서버 재시작 시 크롤링이 중복 실행되거나, 반대로 실행되지 않음

**원인:** `AutonomousScheduler._last_run`이 메모리에만 저장되어 서버 재시작 시 초기화

**해결:** 스케줄러 상태를 `data/scheduler_state.json`에 저장

```python
# brain.py - AutonomousScheduler
STATE_FILE = "./data/scheduler_state.json"

def _load_state(self):
    # 서버 시작 시 파일에서 last_run 복원

def _save_state(self):
    # 작업 완료 시 파일에 저장

def mark_completed(self, schedule_id: str):
    self._last_run[schedule_id] = self.get_kst_now()
    self._save_state()  # 즉시 저장
```

---

### 문제 4: Import 경로 오류 (2026-01-03 해결)

**증상:** 챗봇에서 크롤링 시작 기능이 작동하지 않음

**원인:** `simple_chat.py:492`에서 잘못된 import 경로 사용
```python
# 잘못됨
from core.crawl_manager import get_crawl_manager

# 올바름
from src.core.crawl_manager import get_crawl_manager
```

**해결:** 모든 import 경로를 `src.` 접두사로 통일

---

### 코드 연결 구조

```
dashboard_api.py (FastAPI 서버)
    ├── startup_event()
    │   └── brain.start_scheduler()  ← 서버 시작 시 스케줄러 시작
    │
    ├── /api/v3/chat
    │   └── SimpleChatService.chat()
    │       └── _tool_start_crawling()
    │           └── crawl_manager.start_crawl()  ← 챗봇에서 크롤링
    │
    └── /api/crawl/start
        └── crawl_manager.start_crawl()  ← API로 수동 크롤링

src/core/brain.py (자율 스케줄러)
    └── AutonomousScheduler
        └── _handle_scheduled_task()
            └── crawl_manager.start_crawl()  ← 매일 KST 06:00 자동 크롤링

src/core/crawl_manager.py (크롤링 관리)
    └── start_crawl() → _run_crawl()
        ├── CrawlerAgent.execute()  ← Amazon 크롤링
        ├── StorageAgent.execute()  ← Google Sheets 저장
        └── DashboardExporter.export_dashboard_data()  ← JSON 생성
```

---

# English

## Project Overview

AI agent system for analyzing AMORE Pacific LANEIGE brand competitiveness in Amazon US market.

### Key Features

| Feature | Description |
|---------|-------------|
| **Auto Crawling** | Daily Amazon Top 100 at KST 06:00 (5 categories) |
| **KPI Analysis** | Strategic metrics: SoS, HHI, CPI |
| **AI Chatbot** | Knowledge Graph + Keyword RAG based Q&A |
| **Insight Generation** | LLM-based strategic insights |

### Monitored Categories

- Beauty & Personal Care
- Skin Care
- Lip Care
- Lip Makeup
- Face Powder

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn, Pydantic |
| **LLM** | OpenAI GPT-4.1-mini (via LiteLLM) |
| **Hybrid RAG** | HybridRetriever (KnowledgeGraph + OntologyReasoner + DocumentRetriever) |
| **Crawling** | Playwright (Chromium headless) |
| **Data** | Pandas, Google Sheets API |
| **Deploy** | Docker, Railway |

### Hybrid RAG System Details

`HybridRetriever` integrates 3 components:
1. **KnowledgeGraph** - Triple Store based knowledge graph (brand/product/category relations)
2. **OntologyReasoner** - Business rules based inference engine
3. **DocumentRetriever** - Keyword-based guide document search (4 MD files)

**Reference Documents** (`docs/guides/`):
- Strategic Indicators Definition.md
- Metric Interpretation Guide.md
- Indicator Combination Playbook.md
- Home Page Insight Rules.md

> ChromaDB vector search exists in code but is currently **disabled**

---

## Core Modules

| Module | File | Description |
|--------|------|-------------|
| **UnifiedBrain** | `src/core/brain.py` | Autonomous scheduler, agent orchestration |
| **HybridRetriever** | `src/rag/hybrid_retriever.py` | KG + Ontology + RAG integrated search |
| **KnowledgeGraph** | `src/ontology/knowledge_graph.py` | Triple Store knowledge graph |
| **OntologyReasoner** | `src/ontology/reasoner.py` | Business rules inference engine |
| **AmazonScraper** | `src/tools/amazon_scraper.py` | Playwright crawler |

### Auto Scheduler

```python
# src/core/brain.py - AutonomousScheduler
# Operates on Korean Standard Time (KST)
KST = timezone(timedelta(hours=9))

schedules = [
    {"id": "daily_crawl", "hour": 6, "minute": 0},  # KST 06:00 auto crawl
    {"id": "check_data_freshness", "interval_hours": 1}  # Check data freshness hourly
]
```

**State file:** `data/scheduler_state.json` (persists across server restarts)

---

## API Endpoints

### Main APIs

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/health` | Health check | - |
| GET | `/api/data` | Dashboard data | - |
| GET | `/dashboard` | Dashboard UI | - |
| POST | `/api/chat` | v1 chatbot | - |
| POST | `/api/v3/chat` | v3 chatbot (recommended) | - |
| POST | `/api/crawl/start` | Start crawling | API Key |
| GET | `/api/v4/brain/status` | Scheduler status | - |

### API Key Authentication

```bash
curl -X POST "https://your-app.railway.app/api/crawl/start" \
  -H "X-API-Key: your-api-key"
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/tunatuna1002-lab/AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-.git
cd AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-

# 2. Virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 4. Environment variables (.env)
OPENAI_API_KEY=sk-...
API_KEY=your-api-key
AUTO_START_SCHEDULER=true

# 5. Run
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001
```

**Access:** http://localhost:8001/dashboard

---

## Deployment

### Railway

1. Connect GitHub at https://railway.app
2. Environment variables: `OPENAI_API_KEY`, `API_KEY`, `AUTO_START_SCHEDULER=true`
3. Generate domain

### Docker

```bash
docker build -t amore-agent .
docker run -p 8001:8001 -e OPENAI_API_KEY=sk-... amore-agent
```

---

## Audit Trail

Chatbot conversations are logged to `./logs/chatbot_audit_YYYY-MM-DD.log`.

---

## License

MIT License
