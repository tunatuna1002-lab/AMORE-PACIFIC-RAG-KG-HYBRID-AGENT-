# AMORE Pacific RAG-KG Hybrid Agent

> **Level 4 Autonomous AI Agent** - RAG + Knowledge Graph + Ontology 기반 Amazon 베스트셀러 분석 플랫폼

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[한국어](#한국어) | [English](#english)

---

# 한국어

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [기술 스택](#기술-스택)
4. [핵심 모듈](#핵심-모듈)
5. [API 명세](#api-명세)
6. [설치 및 실행](#설치-및-실행)
7. [배포 가이드](#배포-가이드)

---

## 프로젝트 개요

### 목적

AMORE Pacific LANEIGE 브랜드의 Amazon US 시장 경쟁력 분석을 위한 **자율 AI 에이전트 시스템**입니다.

### 핵심 기능

| 기능 | 설명 |
|------|------|
| **자동 크롤링** | 매일 KST 06:00 Amazon Top 100 자동 수집 (5개 카테고리) |
| **KPI 분석** | SoS, HHI, CPI 등 10대 전략 지표 계산 |
| **AI 챗봇** | RAG + Knowledge Graph 기반 자연어 Q&A |
| **인사이트 생성** | LLM 기반 일일 전략 인사이트 자동 생성 |
| **알림 시스템** | 순위 급변동, SoS 변화 등 임계값 기반 알림 |

### 모니터링 대상 카테고리

- Beauty & Personal Care
- Skin Care
- Lip Care
- Lip Makeup
- Face Powder

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Level 4 Autonomous Agent                         │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                          UnifiedBrain                               │ │
│  │                      (LLM-First Decision)                           │ │
│  │  ┌──────────────┐  ┌────────────────┐  ┌────────────────────────┐  │ │
│  │  │ Priority Queue│  │ Auto Scheduler │  │   Event System         │  │ │
│  │  │ USER > ALERT │  │ KST 06:00 Crawl│  │   Alert/Log/Callback   │  │ │
│  │  └──────────────┘  └────────────────┘  └────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│         ┌──────────────────────────┼──────────────────────────┐         │
│         ▼                          ▼                          ▼         │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │  HybridChatbot  │     │  HybridInsight  │     │   AlertAgent    │   │
│  │  (RAG+KG Chat)  │     │  (Insight Gen)  │     │   (Alerts)      │   │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘   │
├──────────────────────────────────────────────────────────────────────────┤
│                              Core Components                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ ContextGatherer │  │ HybridRetriever │  │   ResponsePipeline      │  │
│  │                 │  │  ┌───────────┐  │  │  (Response/Cache)       │  │
│  │                 │  │  │    RAG    │  │  │                         │  │
│  │                 │  │  │ (ChromaDB)│  │  │                         │  │
│  │                 │  │  ├───────────┤  │  │                         │  │
│  │                 │  │  │ Knowledge │  │  │                         │  │
│  │                 │  │  │   Graph   │  │  │                         │  │
│  │                 │  │  └───────────┘  │  │                         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────────┤
│                             Execution Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Crawler     │  │ Storage     │  │ Metrics     │  │ SimpleChat      │ │
│  │ (Playwright)│  │ (G.Sheets)  │  │ (KPI Calc)  │  │ (v3 API)        │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │
├──────────────────────────────────────────────────────────────────────────┤
│                               Data Layer                                  │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────────────┐   │
│  │  ChromaDB   │  │   JSON Files     │  │     Google Sheets         │   │
│  │  (Vectors)  │  │  (Cache/State)   │  │     (Persistence)         │   │
│  └─────────────┘  └──────────────────┘  └───────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn, LiteLLM, Pydantic |
| **AI/ML** | OpenAI GPT-4.1-mini, Keyword-based RAG (ChromaDB optional) |
| **Crawling** | Playwright (Chromium headless) |
| **Data** | Pandas, NumPy, Google Sheets API, python-docx |
| **Infra** | Docker, Railway |

---

## 핵심 모듈

| 모듈 | 파일 | 설명 |
|------|------|------|
| **UnifiedBrain** | `src/core/brain.py` | Level 4 자율 에이전트 두뇌, 스케줄러 |
| **HybridRetriever** | `src/rag/hybrid_retriever.py` | RAG + KG 통합 검색 |
| **KnowledgeGraph** | `src/ontology/knowledge_graph.py` | Triple Store 기반 지식 그래프 |
| **AmazonScraper** | `src/tools/amazon_scraper.py` | Playwright 기반 크롤러 |
| **MetricsAgent** | `src/agents/metrics_agent.py` | 10대 KPI 계산 |

### 10대 전략 KPI

| KPI | 설명 |
|-----|------|
| **SoS** | Share of Shelf - Top 100 내 브랜드 제품 비율 |
| **HHI** | Herfindahl Index - 시장 집중도 |
| **CPI** | Competitive Position Index - 경쟁 포지션 지수 |
| **Volatility** | 7일 순위 변동 표준편차 |
| **Top3/5/10** | 상위권 진입 제품 수 |

---

## API 명세

### 주요 엔드포인트

| Method | Endpoint | 설명 | 인증 |
|--------|----------|------|------|
| GET | `/api/health` | 헬스 체크 | - |
| GET | `/api/data` | 대시보드 데이터 | - |
| GET | `/dashboard` | 대시보드 UI | - |
| POST | `/api/v3/chat` | AI 챗봇 | - |
| POST | `/api/crawl/start` | 크롤링 시작 | API Key |
| GET | `/api/v4/brain/status` | Brain 상태 | - |

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
API_KEY=your-secure-api-key
AUTO_START_SCHEDULER=true

# 5. 실행
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001
```

**접속:** http://localhost:8001/dashboard

---

## 배포 가이드

### Railway 배포

1. https://railway.app 에서 GitHub 연결
2. 환경 변수 설정: `OPENAI_API_KEY`, `API_KEY`, `AUTO_START_SCHEDULER=true`
3. 도메인 생성

### Docker

```bash
docker build -t amore-agent .
docker run -p 8001:8001 -e OPENAI_API_KEY=sk-... amore-agent
```

### Hobby 플랜 Sleep 모드 해결

[cron-job.org](https://cron-job.org)에서 매일 05:55 KST에 `/api/health` 호출하여 서버 깨우기

---

## Audit Trail

모든 챗봇 대화가 `./logs/chatbot_audit_YYYY-MM-DD.log`에 자동 기록됩니다.

---

# English

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Tech Stack](#tech-stack)
4. [Core Modules](#core-modules)
5. [API Reference](#api-reference)
6. [Installation](#installation)
7. [Deployment](#deployment)

---

## Project Overview

### Purpose

An **Autonomous AI Agent System** for analyzing AMORE Pacific's LANEIGE brand competitiveness in the Amazon US market.

### Key Features

| Feature | Description |
|---------|-------------|
| **Auto Crawling** | Daily Amazon Top 100 collection at KST 06:00 (5 categories) |
| **KPI Analysis** | 10 strategic metrics including SoS, HHI, CPI |
| **AI Chatbot** | Natural language Q&A based on RAG + Knowledge Graph |
| **Insight Generation** | LLM-based daily strategic insights |
| **Alert System** | Threshold-based alerts for rank changes, SoS fluctuations |

### Monitored Categories

- Beauty & Personal Care
- Skin Care
- Lip Care
- Lip Makeup
- Face Powder

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Level 4 Autonomous Agent                         │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                          UnifiedBrain                               │ │
│  │                      (LLM-First Decision)                           │ │
│  │  ┌──────────────┐  ┌────────────────┐  ┌────────────────────────┐  │ │
│  │  │ Priority Queue│  │ Auto Scheduler │  │   Event System         │  │ │
│  │  │ USER > ALERT │  │ KST 06:00 Crawl│  │   Alert/Log/Callback   │  │ │
│  │  └──────────────┘  └────────────────┘  └────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│         ┌──────────────────────────┼──────────────────────────┐         │
│         ▼                          ▼                          ▼         │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │  HybridChatbot  │     │  HybridInsight  │     │   AlertAgent    │   │
│  │  (RAG+KG Chat)  │     │  (Insight Gen)  │     │   (Alerts)      │   │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘   │
├──────────────────────────────────────────────────────────────────────────┤
│                              Core Components                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ ContextGatherer │  │ HybridRetriever │  │   ResponsePipeline      │  │
│  │                 │  │  ┌───────────┐  │  │  (Response/Cache)       │  │
│  │                 │  │  │    RAG    │  │  │                         │  │
│  │                 │  │  │ (ChromaDB)│  │  │                         │  │
│  │                 │  │  ├───────────┤  │  │                         │  │
│  │                 │  │  │ Knowledge │  │  │                         │  │
│  │                 │  │  │   Graph   │  │  │                         │  │
│  │                 │  │  └───────────┘  │  │                         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────────┤
│                             Execution Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Crawler     │  │ Storage     │  │ Metrics     │  │ SimpleChat      │ │
│  │ (Playwright)│  │ (G.Sheets)  │  │ (KPI Calc)  │  │ (v3 API)        │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │
├──────────────────────────────────────────────────────────────────────────┤
│                               Data Layer                                  │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────────────┐   │
│  │  ChromaDB   │  │   JSON Files     │  │     Google Sheets         │   │
│  │  (Vectors)  │  │  (Cache/State)   │  │     (Persistence)         │   │
│  └─────────────┘  └──────────────────┘  └───────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn, LiteLLM, Pydantic |
| **AI/ML** | OpenAI GPT-4.1-mini, Keyword-based RAG (ChromaDB optional) |
| **Crawling** | Playwright (Chromium headless) |
| **Data** | Pandas, NumPy, Google Sheets API, python-docx |
| **Infra** | Docker, Railway |

---

## Core Modules

| Module | File | Description |
|--------|------|-------------|
| **UnifiedBrain** | `src/core/brain.py` | Level 4 autonomous agent brain, scheduler |
| **HybridRetriever** | `src/rag/hybrid_retriever.py` | RAG + KG integrated search |
| **KnowledgeGraph** | `src/ontology/knowledge_graph.py` | Triple Store based knowledge graph |
| **AmazonScraper** | `src/tools/amazon_scraper.py` | Playwright-based crawler |
| **MetricsAgent** | `src/agents/metrics_agent.py` | 10 KPI calculations |

### 10 Strategic KPIs

| KPI | Description |
|-----|-------------|
| **SoS** | Share of Shelf - Brand product ratio in Top 100 |
| **HHI** | Herfindahl Index - Market concentration |
| **CPI** | Competitive Position Index |
| **Volatility** | 7-day rank standard deviation |
| **Top3/5/10** | Products in top tiers |

---

## API Reference

### Main Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/health` | Health check | - |
| GET | `/api/data` | Dashboard data | - |
| GET | `/dashboard` | Dashboard UI | - |
| POST | `/api/v3/chat` | AI Chatbot | - |
| POST | `/api/crawl/start` | Start crawling | API Key |
| GET | `/api/v4/brain/status` | Brain status | - |

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
API_KEY=your-secure-api-key
AUTO_START_SCHEDULER=true

# 5. Run
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001
```

**Access:** http://localhost:8001/dashboard

---

## Deployment

### Railway

1. Connect GitHub at https://railway.app
2. Set environment variables: `OPENAI_API_KEY`, `API_KEY`, `AUTO_START_SCHEDULER=true`
3. Generate domain

### Docker

```bash
docker build -t amore-agent .
docker run -p 8001:8001 -e OPENAI_API_KEY=sk-... amore-agent
```

### Hobby Plan Sleep Mode Solution

Use [cron-job.org](https://cron-job.org) to call `/api/health` daily at 05:55 KST to wake up the server before scheduled crawling.

---

## Audit Trail

All chatbot conversations are automatically logged to `./logs/chatbot_audit_YYYY-MM-DD.log`.

---

## License

MIT License

---

## Contact

- **GitHub Issues**: [Report Issues](https://github.com/tunatuna1002-lab/AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-/issues)
