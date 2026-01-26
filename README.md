# AMORE Pacific RAG-Ontology Hybrid Agent

> **Amazon US 시장에서 LANEIGE 브랜드 경쟁력을 분석하는 자율 AI 에이전트**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Quick Start

```bash
# 설치
git clone https://github.com/your-repo/AMORE-RAG-ONTOLOGY-HYBRID-AGENT.git
cd AMORE-RAG-ONTOLOGY-HYBRID-AGENT
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# 환경 변수 (.env)
OPENAI_API_KEY=sk-...
API_KEY=your-api-key
AUTO_START_SCHEDULER=true

# 실행
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001
```

**접속:** http://localhost:8001/dashboard

---

## 목차

1. [핵심 가치](#1-핵심-가치)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [주요 기능](#3-주요-기능)
4. [기술 스택](#4-기술-스택)
5. [API 레퍼런스](#5-api-레퍼런스)
6. [배포](#6-배포)
7. [테스트](#7-테스트)
8. [문서](#8-문서)

---

## 1. 핵심 가치

### 추론 기반 전략적 인사이트

| 기존 방식 | 이 에이전트 |
|----------|------------|
| "LANEIGE SoS 5.2%, COSRX 8.1%" | **"LANEIGE는 NicheBrand, COSRX는 StrongBrand. SoS 격차 10%p 이상 시 마케팅 강화 필요. 권고: Lip Sleeping Mask 집중"** |

### 4대 핵심 컴포넌트

| 컴포넌트 | 역할 |
|---------|------|
| **RAG** | 11개 문서 지식 검색 (docs/guides/, docs/market/) |
| **Knowledge Graph** | 브랜드-제품-카테고리 관계 (50K 트리플) |
| **OWL Ontology** | 도메인 규칙 자동 추론 (29+ 규칙) |
| **크롤링 데이터** | 실시간 Amazon 베스트셀러 (매일 22:00 KST) |

---

## 2. 시스템 아키텍처

```
Amazon Bestsellers (Top 100 × 5 categories)
         ↓
    CrawlerAgent (Playwright + Stealth)
         ↓
    StorageAgent (SQLite + Google Sheets)
         ↓
    KnowledgeGraph + OWL Ontology
         ↓
    HybridRetriever (RAG + KG + Ontology)
         ↓
    Dashboard + AI Chatbot
```

### 모니터링 카테고리

| 카테고리 | Amazon Node ID |
|----------|----------------|
| Beauty & Personal Care | beauty |
| Skin Care | 11060451 |
| Lip Care | 3761351 |
| Lip Makeup | 11059031 |
| Face Powder | 11058971 |

### 핵심 모듈

| 모듈 | 파일 | 역할 |
|------|------|------|
| UnifiedBrain | `src/core/brain.py` | 자율 스케줄러 |
| KnowledgeGraph | `src/ontology/knowledge_graph.py` | Triple Store |
| HybridRetriever | `src/rag/hybrid_retriever.py` | RAG + KG + Ontology 통합 |
| HybridChatbotAgent | `src/agents/hybrid_chatbot_agent.py` | AI 챗봇 |

---

## 3. 주요 기능

### 3.1 자동 크롤링 (22:00 KST)

- 5개 카테고리 × 100개 제품 = **500개 제품/일**
- Stealth 모드: playwright-stealth, browserforge, fake-useragent
- 예상 소요: ~80-90분

### 3.2 KPI 분석

| 지표 | 설명 |
|------|------|
| **SoS** | Share of Shelf (브랜드 점유율) |
| **HHI** | Herfindahl-Hirschman Index (시장 집중도) |
| **CPI** | Competitive Position Index (경쟁 포지션) |

### 3.3 AI 챗봇

- **API**: `POST /api/v3/chat`
- RAG + KG + Ontology 통합 컨텍스트
- 7-type 출처 추출 및 참고자료 표시

### 3.4 외부 신호 수집

- Tavily 뉴스 API (46개 신뢰 매체)
- RSS 피드 (Allure, Byrdie, WWD)
- Reddit API (r/SkincareAddiction, r/AsianBeauty)

---

## 4. 기술 스택

| 분류 | 기술 |
|------|------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| LLM | OpenAI GPT-4.1-mini (via LiteLLM) |
| RAG | ChromaDB + OpenAI Embeddings |
| Ontology | owlready2, Rule-based Reasoner |
| 크롤링 | Playwright, playwright-stealth |
| 데이터 | SQLite, Google Sheets, Pandas |
| 배포 | Docker, Railway |
| 테스트 | pytest, pytest-cov (60% 최소 커버리지) |

---

## 5. API 레퍼런스

| Method | Endpoint | 설명 | 인증 |
|--------|----------|------|------|
| GET | `/api/health` | 헬스 체크 | - |
| GET | `/api/data` | 대시보드 데이터 | - |
| GET | `/dashboard` | 대시보드 UI | - |
| POST | `/api/v3/chat` | AI 챗봇 | - |
| POST | `/api/crawl/start` | 크롤링 시작 | API Key |
| GET | `/api/v4/brain/status` | 스케줄러 상태 | - |
| POST | `/api/export/docx` | DOCX 리포트 | - |

---

## 6. 배포

### Railway

```bash
# 환경 변수 설정
OPENAI_API_KEY=sk-...
API_KEY=your-api-key
AUTO_START_SCHEDULER=true
GOOGLE_SHEETS_SPREADSHEET_ID=...
GOOGLE_SHEETS_CREDENTIALS_JSON=...
```

### Docker

```bash
docker build -t amore-agent .
docker run -p 8001:8001 -e OPENAI_API_KEY=sk-... amore-agent
```

### 로컬 데이터 동기화

```bash
python scripts/sync_from_railway.py        # Railway → 로컬 동기화
python scripts/sync_sheets_to_sqlite.py    # Sheets → SQLite 동기화
```

---

## 7. 테스트

```bash
# 전체 테스트 (커버리지 포함)
python -m pytest tests/ -v

# 커버리지 리포트
open coverage_html/index.html

# 골든셋 평가
python scripts/evaluate_golden.py --verbose

# KG 백업
python -m src.tools.kg_backup backup
python -m src.tools.kg_backup list
```

### 테스트 환경 분리

```bash
# .env.test 사용
ENV_FILE=.env.test python -m pytest tests/
```

---

## 8. 문서

| 문서 | 설명 |
|------|------|
| [`CLAUDE.md`](CLAUDE.md) | 개발 가이드 (Claude Code용) |
| [`docs/AuditReport.md`](docs/AuditReport.md) | E2E 통합 감사 보고서 |
| [`docs/TAVILY_NEWS_INTEGRATION.md`](docs/TAVILY_NEWS_INTEGRATION.md) | Tavily 뉴스 API 가이드 |
| [`docs/TRUE_RAG_ONTOLOGY_INTEGRATION_PLAN.md`](docs/TRUE_RAG_ONTOLOGY_INTEGRATION_PLAN.md) | RAG-Ontology 통합 계획 |

---

## 업데이트 히스토리

### 2026-01-27
- **TDD 권장안 구현**: KG Railway Volume 연결, 자동 백업 (7일), 커버리지 측정 환경
- **E2E 감사 완료**: 39개 이슈 발견, 로드맵 수립
- **Tavily 뉴스 API 통합**: 46개 신뢰 매체 실시간 수집

### 2026-01-26
- **SoS UI 개선**: 제품 개수 소수점 표시, 출현율 추가

### 2026-01-25
- **크롤링 최적화**: 22:00 KST 변경, Stealth 모드 적용
- **SQLite 동기화**: 데이터 정합성 검사 모듈 추가

### 2026-01-23
- **Clean Architecture**: Domain/Application/Adapters/Infrastructure 레이어 분리
- **TDD**: 164개 단위 테스트, 커스텀 예외 8종

---

## 라이선스

MIT License
