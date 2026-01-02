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
| **RAG** | 키워드 기반 문서 검색 (4개 MD 가이드 문서) |
| **크롤링** | Playwright (Chromium headless) |
| **데이터** | Pandas, Google Sheets API |
| **배포** | Docker, Railway |

### RAG 시스템 상세

현재 RAG는 **키워드 매칭 기반 검색**을 사용합니다:
- 4개 가이드 문서 (Strategic Indicators, Metric Interpretation, Indicator Combination, Home Insight Rules)
- ChromaDB 벡터 검색은 코드에 존재하나 **비활성화** 상태

---

## 핵심 모듈

| 모듈 | 파일 | 설명 |
|------|------|------|
| **UnifiedBrain** | `src/core/brain.py` | 자율 스케줄러, 에이전트 조율 |
| **SimpleChat** | `src/core/simple_chat.py` | v3 챗봇 (GPT-4.1-mini) |
| **KnowledgeGraph** | `src/ontology/knowledge_graph.py` | Triple Store 지식 그래프 |
| **AmazonScraper** | `src/tools/amazon_scraper.py` | Playwright 크롤러 |
| **DocumentRetriever** | `src/rag/retriever.py` | 키워드 기반 문서 검색 |

### 자동 스케줄러

```python
# src/core/brain.py - AutonomousScheduler
schedules = [
    {"id": "daily_crawl", "hour": 21, "minute": 0},  # UTC 21:00 = KST 06:00
    {"id": "check_data_freshness", "interval_hours": 1}
]
```

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
| **RAG** | Keyword-based document search (4 MD guide docs) |
| **Crawling** | Playwright (Chromium headless) |
| **Data** | Pandas, Google Sheets API |
| **Deploy** | Docker, Railway |

### RAG System Details

Current RAG uses **keyword matching based search**:
- 4 guide documents (Strategic Indicators, Metric Interpretation, Indicator Combination, Home Insight Rules)
- ChromaDB vector search exists in code but is **disabled**

---

## Core Modules

| Module | File | Description |
|--------|------|-------------|
| **UnifiedBrain** | `src/core/brain.py` | Autonomous scheduler, agent orchestration |
| **SimpleChat** | `src/core/simple_chat.py` | v3 chatbot (GPT-4.1-mini) |
| **KnowledgeGraph** | `src/ontology/knowledge_graph.py` | Triple Store knowledge graph |
| **AmazonScraper** | `src/tools/amazon_scraper.py` | Playwright crawler |
| **DocumentRetriever** | `src/rag/retriever.py` | Keyword-based document search |

### Auto Scheduler

```python
# src/core/brain.py - AutonomousScheduler
schedules = [
    {"id": "daily_crawl", "hour": 21, "minute": 0},  # UTC 21:00 = KST 06:00
    {"id": "check_data_freshness", "interval_hours": 1}
]
```

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
