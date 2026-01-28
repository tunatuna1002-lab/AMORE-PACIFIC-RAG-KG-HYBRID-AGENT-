# AMORE Pacific RAG-Ontology Hybrid Agent

> **Amazon US 시장에서 LANEIGE 브랜드 경쟁력을 분석하는 자율 AI 에이전트**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Quick Start

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

## 📑 목차

1. [핵심 가치](#1-핵심-가치)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [주요 기능](#3-주요-기능)
4. [기술 스택](#4-기술-스택)
5. [API 레퍼런스](#5-api-레퍼런스)
6. [배포](#6-배포)
7. [테스트](#7-테스트)
8. [문서](#8-문서)
9. [업데이트 히스토리](#9-업데이트-히스토리)

---

## 1. 핵심 가치

### 추론 기반 전략적 인사이트

| 기존 방식 | 이 에이전트 |
|----------|------------|
| "LANEIGE SoS 5.2%, COSRX 8.1%" | **"LANEIGE는 K-Beauty 프리미엄 세그먼트 1위. SoS 2.8%로 3분기 연속 상승세. 권고: Prime Day 대비 재고 확보 및 Skin Care 카테고리 확장"** |

### 5대 핵심 컴포넌트

| 컴포넌트 | 역할 |
|---------|------|
| **RAG** | 문서 지식 검색 + Embedding 캐시 (API 비용 33%↓) |
| **Knowledge Graph** | 브랜드-제품-카테고리 관계 (50K+ 트리플) |
| **OWL Ontology** | 도메인 규칙 자동 추론 (29+ 규칙) |
| **ReAct Agent** | 복잡한 질문 자기반성 루프 (최대 3회) |
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
    ReAct Agent (복잡한 질문 자기반성)
         ↓
    Dashboard + AI Chatbot + IR-Style Report Export
```

### 모니터링 카테고리

| 카테고리 | Amazon Node ID | Level |
|----------|----------------|-------|
| Beauty & Personal Care | beauty | L0 |
| Skin Care | 11060451 | L1 |
| Lip Care | 3761351 | L2 |
| Lip Makeup | 11059031 | L2 |
| Face Powder | 11058971 | L3 |

### 핵심 모듈

| 모듈 | 파일 | 역할 |
|------|------|------|
| UnifiedBrain | `src/core/brain.py` | 자율 스케줄러 + ReAct 통합 |
| ReActAgent | `src/core/react_agent.py` | 복잡한 질문 자기반성 루프 |
| KnowledgeGraph | `src/ontology/knowledge_graph.py` | Triple Store |
| HybridRetriever | `src/rag/hybrid_retriever.py` | RAG + KG + Ontology 통합 |
| ReportGenerator | `src/tools/report_generator.py` | IR-Style DOCX/PPTX 리포트 |

---

## 3. 주요 기능

### 3.1 자동 크롤링 (22:00 KST)

- 5개 카테고리 × 100개 제품 = **500개 제품/일**
- Stealth 모드: playwright-stealth, browserforge, fake-useragent
- AWS WAF 대응: 지수 백오프, 디버그 스크린샷

### 3.2 KPI 분석

| 지표 | 설명 |
|------|------|
| **SoS** | Share of Shelf - 브랜드 점유율 |
| **HHI** | Herfindahl-Hirschman Index - 시장 집중도 |
| **CPI** | Competitive Position Index - 경쟁 포지션 |
| **TAM/SAM/SOM** | 시장 규모 분석 |

### 3.3 AI 챗봇

- **API**: `POST /api/v3/chat`
- RAG + KG + Ontology 통합 컨텍스트
- ReAct Self-Reflection: 복잡한 질문 자동 감지 및 자기반성 루프
- 7-type 출처 추출 및 참고자료 표시

### 3.4 IR-Style 리포트 생성 (NEW)

**AMOREPACIFIC 디자인 시스템 적용 전문 애널리스트 리포트**

| 기능 | 설명 |
|------|------|
| **표지** | AMOREPACIFIC 로고 + Pacific Blue 컬러 |
| **목차** | 자동 생성, 하이퍼링크 |
| **섹션** | Executive Summary, 심층 분석, 경쟁 환경, 시장 동향, 전략 제언 |
| **참고자료** | URL 포함 12개+ 소스 |
| **폰트** | 아리따 돋움 (제목), 아리따 부리 (본문) |

```bash
# 리포트 생성 테스트
python scripts/test_report_generator.py
```

**출력 포맷**: DOCX, PPTX (PDF 확장 예정)

### 3.5 외부 신호 수집

| 소스 | 기술 | 비용 |
|------|------|------|
| **Tavily 뉴스** | API | 월 1,000건 무료 |
| **GNews** | API | 일 100건 무료 |
| **RSS** | feedparser | 무료 |

### 3.6 소셜 미디어 수집

| 플랫폼 | 기술 | 수집 대상 |
|--------|------|----------|
| **TikTok** | Playwright | #laneige, #kbeauty |
| **Instagram** | Instaloader | #라네즈, #skincare |
| **YouTube** | yt-dlp | LANEIGE 리뷰 메타데이터 |
| **Reddit** | JSON API | r/AsianBeauty |
| **Google Trends** | trendspyg | 브랜드 검색 관심도 |

### 3.7 공공데이터 API

| API | 용도 |
|-----|------|
| **관세청 수출입통계** | 화장품 HS 3304 수출입 |
| **식약처 기능성화장품** | 신규 등록 현황 |

### 3.8 이메일 알림

- Gmail SMTP 연동
- 순위 변동 (±10위), SoS 급변동 시 자동 알림
- 담당자 다중 수신 지원

---

## 4. 기술 스택

| 분류 | 기술 |
|------|------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn |
| **LLM** | OpenAI GPT-4.1-mini (via LiteLLM) |
| **RAG** | ChromaDB + OpenAI Embeddings + MD5 캐시 |
| **Ontology** | owlready2, Rule-based Reasoner |
| **크롤링** | Playwright, playwright-stealth, browserforge |
| **리포트** | python-docx, python-pptx |
| **데이터** | SQLite, Google Sheets, Pandas |
| **배포** | Docker, Railway |
| **테스트** | pytest, pytest-cov (60% 최소 커버리지) |

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
| POST | `/api/export/docx` | DOCX 리포트 생성 | - |
| POST | `/api/export/pptx` | PPTX 리포트 생성 | - |

---

## 6. 배포

### Railway

```bash
# 필수 환경 변수
OPENAI_API_KEY=sk-...
API_KEY=your-api-key
AUTO_START_SCHEDULER=true

# Google Sheets (선택)
GOOGLE_SHEETS_SPREADSHEET_ID=...
GOOGLE_SHEETS_CREDENTIALS_JSON=...

# 뉴스 수집 (선택)
TAVILY_API_KEY=tvly-...         # 월 1,000건 무료
GNEWS_API_KEY=...               # 일 100건 무료

# 공공데이터 (선택)
DATA_GO_KR_API_KEY=...          # 관세청/식약처 API

# 이메일 알림 (선택)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your@gmail.com
SENDER_PASSWORD=xxxx xxxx xxxx xxxx  # Gmail 앱 비밀번호
ALERT_RECIPIENTS=alert@email.com
```

### Docker

```bash
docker build -t amore-agent .
docker run -p 8001:8001 -e OPENAI_API_KEY=sk-... amore-agent
```

### 로컬 데이터 동기화

```bash
python scripts/sync_from_railway.py        # Railway → 로컬
python scripts/sync_sheets_to_sqlite.py    # Sheets → SQLite
```

---

## 7. 테스트

```bash
# 전체 테스트
python -m pytest tests/ -v

# 커버리지 리포트
open coverage_html/index.html

# 골든셋 평가
python scripts/evaluate_golden.py --verbose

# KG 백업
python -m src.tools.kg_backup backup
python -m src.tools.kg_backup list

# 리포트 생성 테스트
python scripts/test_report_generator.py
```

### 테스트 환경 분리

```bash
ENV_FILE=.env.test python -m pytest tests/
```

---

## 8. 문서

| 문서 | 설명 |
|------|------|
| [`CLAUDE.md`](CLAUDE.md) | 개발 가이드 (Claude Code용) |
| [`docs/guides/react_agent_guide.md`](docs/guides/react_agent_guide.md) | ReAct Agent 가이드 |
| [`docs/embedding_cache_guide.md`](docs/embedding_cache_guide.md) | Embedding 캐시 가이드 |
| [`docs/AMOREPACIFIC_DESIGN_SYSTEM.md`](docs/AMOREPACIFIC_DESIGN_SYSTEM.md) | 디자인 시스템 가이드 |

---

## 9. 업데이트 히스토리

### 2026-01-28 (v4) - IR-Style Report Generator

- **전문 애널리스트 리포트**: AMOREPACIFIC 디자인 시스템 적용
- **아리따 폰트**: 돋움 (제목/목차), 부리 (본문) 적용
- **7개 섹션 템플릿**: Executive Summary, 심층 분석, 경쟁 환경, 시장 동향, 외부 신호, 리스크/기회, 전략 제언
- **12개+ 참고자료**: URL 포함, 소스별 용도 설명

### 2026-01-28 (v3) - ReAct Self-Reflection Agent

- **ReAct Loop**: Thought → Action → Observation → Reflection (최대 3회)
- **Self-Reflection**: 응답 품질 자체 평가
- **자동 활성화**: 복잡한 질문 감지 시 ReAct 모드 전환

### 2026-01-28 (v2) - Embedding 캐시

- **MD5 해시 기반 캐시**: 동일 텍스트 재임베딩 방지
- **FIFO Eviction**: 최대 1,000개 항목
- **비용 절감**: OpenAI API 호출 33%+ 절감

### 2026-01-28 (v1) - 카테고리 계층 구조

- **URL 형식 통일**: `zgbs/beauty/{node_id}`
- **계층 구조 정의**: `config/category_hierarchy.json`
- **AWS WAF 대응**: Stealth 컨텍스트, 지수 백오프

### 2026-01-27 (v3) - 이메일 알림

- **Gmail SMTP**: AlertAgent → EmailSender 통합
- **알림 조건**: 순위 ±10, SoS 급변동

### 2026-01-27 (v2) - 소셜 미디어 수집기

- **TikTok/Instagram/YouTube/Reddit**: 모두 무료
- **Google Trends**: trendspyg 지원

### 2026-01-27 (v1)

- **KG Railway Volume**: 자동 백업 (7일 보관)
- **테스트 환경 분리**: `.env.test`
- **골든셋 평가**: `scripts/evaluate_golden.py`

---

## 라이선스

MIT License
