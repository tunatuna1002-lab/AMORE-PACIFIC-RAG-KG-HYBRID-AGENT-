# AMORE Pacific RAG-KG Hybrid Agent

> **Level 4 Autonomous AI Agent** - RAG + Knowledge Graph + Ontology 기반 Amazon 베스트셀러 분석 플랫폼

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English Version](./README_EN.md)

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [기술 스택](#기술-스택)
4. [폴더 구조](#폴더-구조)
5. [핵심 모듈](#핵심-모듈)
6. [API 명세](#api-명세)
7. [전략적 KPI](#전략적-kpi)
8. [설치 및 실행](#설치-및-실행)
9. [배포 가이드](#배포-가이드)

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
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                          UnifiedBrain                               │ │
│  │                      (LLM-First 의사결정)                           │ │
│  │  ┌──────────────┐  ┌────────────────┐  ┌────────────────────────┐  │ │
│  │  │ 우선순위 큐   │  │ 자율 스케줄러   │  │   이벤트 시스템        │  │ │
│  │  │ USER > ALERT │  │ KST 06:00 크롤링│  │   알림/로깅/콜백       │  │ │
│  │  │ > SCHEDULED  │  │ 1시간 데이터체크 │  │                        │  │ │
│  │  └──────────────┘  └────────────────┘  └────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│         ┌──────────────────────────┼──────────────────────────┐         │
│         ▼                          ▼                          ▼         │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │  HybridChatbot  │     │  HybridInsight  │     │   AlertAgent    │   │
│  │  (RAG+KG 챗봇)  │     │  (인사이트 생성) │     │   (알림 생성)   │   │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘   │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                              Core Components                              │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ ContextGatherer │  │ HybridRetriever │  │   ResponsePipeline      │  │
│  │ (컨텍스트 수집) │  │  ┌───────────┐  │  │  (응답 생성/캐싱)       │  │
│  │                 │  │  │    RAG    │  │  │                         │  │
│  │                 │  │  │ (ChromaDB)│  │  │                         │  │
│  │                 │  │  ├───────────┤  │  │                         │  │
│  │                 │  │  │ Knowledge │  │  │                         │  │
│  │                 │  │  │   Graph   │  │  │                         │  │
│  │                 │  │  └───────────┘  │  │                         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                             Execution Layer                               │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Crawler     │  │ Storage     │  │ Metrics     │  │ SimpleChat      │ │
│  │ Agent       │  │ Agent       │  │ Agent       │  │ (v3 API)        │ │
│  │ (Playwright)│  │ (G.Sheets)  │  │ (KPI 계산)  │  │                 │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                               Data Layer                                  │
│                                                                          │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────────────┐   │
│  │  ChromaDB   │  │   JSON Files     │  │     Google Sheets         │   │
│  │  (Vectors)  │  │  (Cache/State)   │  │     (Persistence)         │   │
│  └─────────────┘  └──────────────────┘  └───────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 데이터 흐름

```
사용자 질문 → UnifiedBrain → ContextGatherer → HybridRetriever
                                                    │
                                    ┌───────────────┴───────────────┐
                                    ▼                               ▼
                              KnowledgeGraph                    RAG Search
                              (관계 추론)                       (문서 검색)
                                    │                               │
                                    └───────────────┬───────────────┘
                                                    ▼
                                             LLM 응답 생성
                                                    │
                                                    ▼
                                            ResponsePipeline
                                            (신뢰도 평가, 캐싱)
```

---

## 기술 스택

### Backend
| 기술 | 버전 | 용도 |
|------|------|------|
| Python | 3.11+ | 메인 언어 |
| FastAPI | 0.104+ | 비동기 API 서버 |
| Uvicorn | 0.24+ | ASGI 서버 |
| LiteLLM | 1.40+ | LLM 프로바이더 통합 |
| Pydantic | 2.0+ | 데이터 검증 |

### AI/ML
| 기술 | 용도 |
|------|------|
| OpenAI GPT-4o-mini | LLM 추론 |
| ChromaDB | 벡터 데이터베이스 |
| Sentence Transformers | 임베딩 모델 |

### Data & Integration
| 기술 | 용도 |
|------|------|
| Playwright | Amazon 크롤링 (브라우저 자동화) |
| Pandas/NumPy | 데이터 처리 |
| Google Sheets API | 데이터 영속성 |
| python-docx | DOCX 리포트 생성 |
| httpx | HTTP 클라이언트 |

### Infrastructure
| 기술 | 용도 |
|------|------|
| Docker | 컨테이너화 |
| Railway | 클라우드 배포 |

---

## 폴더 구조

```
AMORE-RAG-ONTOLOGY-HYBRID AGENT/
│
├── src/                              # 메인 소스 코드 (26,000+ LOC)
│   ├── core/                         # 핵심 오케스트레이션 (14개 모듈)
│   │   ├── brain.py                  # Level 4 자율 에이전트 두뇌
│   │   ├── unified_orchestrator.py   # 통합 오케스트레이터 (v2 API)
│   │   ├── simple_chat.py            # 단순화된 챗봇 (v3 API)
│   │   ├── context_gatherer.py       # RAG + KG 컨텍스트 수집
│   │   ├── response_pipeline.py      # 응답 생성 파이프라인
│   │   ├── crawl_manager.py          # 크롤링 상태 관리
│   │   ├── state_manager.py          # 알림 설정 관리
│   │   ├── cache.py                  # TTL 기반 응답 캐시
│   │   ├── confidence.py             # 신뢰도 평가
│   │   └── ...
│   │
│   ├── agents/                       # AI 에이전트 (10개 모듈)
│   │   ├── hybrid_chatbot_agent.py   # RAG+KG 하이브리드 챗봇
│   │   ├── hybrid_insight_agent.py   # 하이브리드 인사이트 생성
│   │   ├── crawler_agent.py          # Amazon 크롤링
│   │   ├── metrics_agent.py          # KPI 계산
│   │   ├── storage_agent.py          # Google Sheets 저장
│   │   ├── alert_agent.py            # 알림 생성
│   │   └── ...
│   │
│   ├── rag/                          # RAG 시스템 (5개 모듈)
│   │   ├── hybrid_retriever.py       # KG + RAG 통합 검색
│   │   ├── router.py                 # 쿼리 유형 분류
│   │   ├── retriever.py              # 문서 검색 (ChromaDB)
│   │   ├── context_builder.py        # 동적 컨텍스트 구성
│   │   └── templates.py              # 프롬프트 템플릿
│   │
│   ├── ontology/                     # Knowledge Graph (5개 모듈)
│   │   ├── knowledge_graph.py        # Triple Store 구현
│   │   ├── reasoner.py               # 온톨로지 추론 엔진
│   │   ├── business_rules.py         # 비즈니스 규칙
│   │   ├── relations.py              # 관계 유형 정의
│   │   └── schema.py                 # 엔티티 스키마
│   │
│   ├── tools/                        # 도구 모듈 (6개)
│   │   ├── amazon_scraper.py         # Playwright 기반 크롤러
│   │   ├── metric_calculator.py      # 10대 KPI 계산기
│   │   ├── sheets_writer.py          # Google Sheets API
│   │   ├── dashboard_exporter.py     # 대시보드 데이터 내보내기
│   │   └── email_sender.py           # 이메일 발송
│   │
│   ├── memory/                       # 대화 메모리 (3개 모듈)
│   └── monitoring/                   # 로깅/트레이싱 (3개 모듈)
│
├── dashboard/                        # 웹 UI
│   ├── amore_unified_dashboard_v4.html  # 메인 대시보드
│   └── test_chat.html                # 채팅 테스트
│
├── data/                             # 런타임 데이터
│   ├── dashboard_data.json           # 대시보드 상태
│   ├── knowledge_graph.json          # KG 데이터 (636KB)
│   ├── crawl_state.json              # 크롤링 상태
│   ├── ranking/                      # 날짜별 순위 데이터
│   ├── metrics/                      # KPI 이력
│   └── traces/                       # 실행 트레이스
│
├── docs/guides/                      # 비즈니스 가이드 문서
│   ├── Strategic Indicators Definition.md
│   ├── Metric Interpretation Guide.md
│   └── Indicator Combination Playbook.md
│
├── config/
│   └── thresholds.json               # 알림 임계값 및 카테고리 URL
│
├── logs/                             # 로그 파일
│   ├── chatbot_audit_*.log           # Audit Trail
│   ├── crawler_*.log                 # 크롤링 로그
│   └── ...
│
├── tests/                            # 테스트 코드
│
├── dashboard_api.py                  # FastAPI 메인 서버
├── start.py                          # Railway 배포용 시작 스크립트
├── orchestrator.py                   # CLI 배치 오케스트레이터
├── main.py                           # CLI 엔트리포인트
│
├── Dockerfile                        # Docker 설정
├── railway.toml                      # Railway 배포 설정
└── requirements.txt                  # Python 의존성
```

---

## 핵심 모듈

### 1. UnifiedBrain (`src/core/brain.py`)

Level 4 자율 에이전트의 핵심 두뇌. 모든 에이전트를 통제하고 스케줄링합니다.

**자율 스케줄러:**
```python
# 서버 시작 시 자동 실행 (AUTO_START_SCHEDULER=true)
schedules = [
    {
        "id": "daily_crawl",
        "name": "일일 크롤링",
        "schedule_type": "daily",
        "hour": 21,  # UTC 21:00 = KST 06:00
        "minute": 0
    },
    {
        "id": "check_data_freshness",
        "name": "데이터 신선도 체크",
        "schedule_type": "interval",
        "interval_hours": 1
    }
]
```

### 2. HybridRetriever (`src/rag/hybrid_retriever.py`)

RAG + Knowledge Graph 통합 검색기.

```
Query → EntityExtractor → ┬→ KnowledgeGraph (사실/관계 조회)
                          └→ RAG Search (문서 검색)
                                    ↓
                          Context Merge → LLM Prompt
```

### 3. Knowledge Graph (`src/ontology/knowledge_graph.py`)

Triple Store 기반 지식 그래프.

```python
# 관계 유형
RelationType = [
    HAS_PRODUCT,      # 브랜드 → 제품
    IN_CATEGORY,      # 제품 → 카테고리
    HAS_COMPETITOR,   # 브랜드 → 경쟁 브랜드
    HAS_RANK,         # 제품 → 순위 기록
    HAS_INSIGHT,      # 엔티티 → 인사이트
]
```

### 4. Amazon Scraper (`src/tools/amazon_scraper.py`)

Playwright 기반 Amazon 크롤러.

**특징:**
- Chromium 헤드리스 브라우저
- 봇 감지 우회 (`--disable-blink-features=AutomationControlled`)
- 랜덤 User-Agent
- 페이지당 50개, 총 100개 제품 수집

---

## API 명세

### Base URL
```
Production: https://your-app.railway.app
Local: http://localhost:8001
```

### 주요 엔드포인트

| Method | Endpoint | 설명 | 인증 |
|--------|----------|------|------|
| GET | `/` | 헬스 체크 | - |
| GET | `/api/health` | 상세 헬스 체크 | - |
| GET | `/api/data` | 대시보드 데이터 | - |
| GET | `/dashboard` | 대시보드 UI | - |
| POST | `/api/chat` | v1 챗봇 (RAG only) | - |
| POST | `/api/v2/chat` | v2 챗봇 (Unified Orchestrator) | - |
| POST | `/api/v3/chat` | v3 챗봇 (Simple Chat) | - |
| GET | `/api/crawl/status` | 크롤링 상태 | - |
| POST | `/api/crawl/start` | 크롤링 시작 | API Key |
| GET | `/api/v3/alert-settings` | 알림 설정 조회 | API Key |
| POST | `/api/v3/alert-settings` | 알림 설정 저장 | API Key |
| GET | `/api/v4/brain/status` | Brain 상태 | - |
| POST | `/api/v4/brain/scheduler/start` | 스케줄러 시작 | API Key |
| POST | `/api/v4/brain/scheduler/stop` | 스케줄러 중지 | API Key |
| POST | `/api/export/docx` | DOCX 리포트 생성 | API Key |

### API Key 인증

보호된 엔드포인트는 `X-API-Key` 헤더 필요:

```bash
curl -X POST "https://your-app.railway.app/api/crawl/start" \
  -H "X-API-Key: your-api-key"
```

### Chat API 예시

**Request:**
```json
POST /api/v3/chat
{
  "message": "LANEIGE 순위가 어떻게 되나요?",
  "session_id": "user_123"
}
```

**Response:**
```json
{
  "text": "현재 LANEIGE Lip Sleeping Mask는 Lip Care 카테고리에서 8위...",
  "confidence": 0.92,
  "sources": ["Dashboard Data", "Knowledge Graph"],
  "suggestions": ["경쟁사 대비 분석", "SoS 추이"]
}
```

---

## 전략적 KPI

### 10대 핵심 지표

| KPI | 명칭 | 설명 |
|-----|------|------|
| **SoS** | Share of Shelf | Top 100 내 브랜드 제품 비율 |
| **HHI** | Herfindahl Index | 시장 집중도 (0~1) |
| **CPI** | Competitive Position Index | 경쟁 포지션 지수 |
| **Volatility** | Rank Volatility | 7일 순위 변동 표준편차 |
| **Top3/5/10** | Tier Penetration | 상위권 진입 제품 수 |
| **Avg Rank** | Average Rank | 브랜드 평균 순위 |
| **Rating Gap** | Quality Indicator | 경쟁사 대비 평점 차이 |
| **MoM** | Month-over-Month | 월간 SoS 변화 |
| **YoY** | Year-over-Year | 연간 SoS 변화 |

### 해석 가이드

| 지표 | 좋음 | 보통 | 주의 |
|------|------|------|------|
| SoS | 20%+ | 10-20% | <10% |
| HHI | <0.15 | 0.15-0.25 | >0.25 |
| Volatility | <3 | 3-7 | >7 |

---

## 설치 및 실행

### 사전 요구사항
- Python 3.11+
- OpenAI API Key
- (선택) Google Cloud 서비스 계정

### 1. 저장소 클론
```bash
git clone https://github.com/tunatuna1002-lab/AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-.git
cd AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-
```

### 2. 가상환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt

# Playwright 브라우저 설치
playwright install chromium
```

### 4. 환경 변수 설정
```bash
# .env 파일 생성
OPENAI_API_KEY=sk-...
API_KEY=your-secure-api-key
AUTO_START_SCHEDULER=true

# (선택) Google Sheets 연동
GOOGLE_SHEETS_SPREADSHEET_ID=...
GOOGLE_CREDENTIALS_JSON={...}
```

### 5. 서버 실행
```bash
# 개발 모드
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001 --reload

# 프로덕션 모드
python start.py
```

### 6. 접속 확인
- 대시보드: http://localhost:8001/dashboard
- API 문서: http://localhost:8001/docs
- 헬스 체크: http://localhost:8001/api/health

---

## 배포 가이드

### Railway 배포

1. **Railway 계정 생성**: https://railway.app

2. **GitHub 연결 및 배포**
   - New Project → Deploy from GitHub repo
   - 저장소 선택

3. **환경 변수 설정**
   ```
   OPENAI_API_KEY=sk-...
   API_KEY=secure-random-string
   AUTO_START_SCHEDULER=true
   PORT=8001
   ```

4. **도메인 설정**
   - Settings → Domains → Generate Domain

### Docker 배포

```bash
# 빌드
docker build -t amore-agent .

# 실행
docker run -p 8001:8001 \
  -e OPENAI_API_KEY=sk-... \
  -e API_KEY=secure-key \
  amore-agent
```

### Railway Hobby 플랜 주의사항

Hobby 플랜은 비활성 시 Sleep 모드로 전환됩니다.

**해결책 - 외부 Cron 서비스:**

[cron-job.org](https://cron-job.org) 사용:
- URL: `https://your-app.railway.app/api/health`
- 실행 시간: `55 5 * * *` (매일 KST 05:55)
- 서버를 깨워 06:00 크롤링 정상 실행

```
05:55 KST          06:00 KST
    │                  │
    ▼                  ▼
cron-job.org  →  서버 Wake Up  →  자율 스케줄러 크롤링 실행
GET /api/health
```

---

## Audit Trail

모든 챗봇 대화는 자동으로 로깅됩니다.

**로그 위치:** `./logs/chatbot_audit_YYYY-MM-DD.log`

**기록 내용:**
```json
{
  "session_id": "user_123",
  "timestamp": "2026-01-03T06:00:00",
  "user_query": "LANEIGE 순위 알려줘",
  "ai_response": "현재 LANEIGE...",
  "query_type": "data_query",
  "confidence": 0.92,
  "response_time_ms": 1234.5
}
```

---

## 라이선스

MIT License

---

## 연락처

- **GitHub Issues**: [이슈 등록](https://github.com/tunatuna1002-lab/AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-/issues)
