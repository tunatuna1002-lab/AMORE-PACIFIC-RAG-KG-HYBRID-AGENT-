# AMORE Pacific RAG-Ontology Hybrid Agent

> **Amazon US 시장에서 LANEIGE 브랜드 경쟁력을 분석하는 자율 AI 에이전트**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 목차

1. [핵심 가치](#1-핵심-가치)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [주요 기능](#3-주요-기능)
4. [기술 스택](#4-기술-스택)
5. [설치 및 실행](#5-설치-및-실행)
6. [API 레퍼런스](#6-api-레퍼런스)
7. [배포](#7-배포)
8. [업데이트 히스토리](#8-업데이트-히스토리)

---

## 1. 핵심 가치

### 단순 데이터 조회가 아닌, **추론 기반 전략적 인사이트** 제공

| 구분 | 기존 방식 | 이 에이전트 |
|-----|----------|------------|
| **질문** | "LANEIGE vs COSRX 비교" | 동일 |
| **응답** | "LANEIGE SoS 5.2%, COSRX 8.1%" | **"LANEIGE는 NicheBrand (SoS 5.2%), COSRX는 StrongBrand (15.3%). 전략 가이드에 따르면 SoS 격차 10%p 이상 시 마케팅 강화 필요. 권고: Lip Sleeping Mask 집중, Top 10 확대"** |

### 4대 핵심 컴포넌트

```
사용자 질문
    │
    ├─→ 1. RAG (Vector Search)        : 문서 지식 검색
    │       docs/guides/, docs/market/ 11개 문서
    │
    ├─→ 2. Knowledge Graph            : 브랜드-제품-카테고리 관계
    │       LANEIGE ──competesWith──→ COSRX
    │       LANEIGE ──hasProduct──→ Lip Sleeping Mask
    │
    ├─→ 3. OWL Ontology               : 도메인 규칙 자동 추론
    │       Brand ─┬─ DominantBrand (SoS ≥ 30%)
    │              ├─ StrongBrand (15% ≤ SoS < 30%)
    │              └─ NicheBrand (SoS < 15%)
    │
    └─→ 4. 크롤링 데이터              : 실시간 Amazon 베스트셀러
            매일 KST 06:00 자동 수집
    │
    ▼
Confidence Fusion → LLM → 전략적 인사이트
```

---

## 2. 시스템 아키텍처

### 2.1 전체 흐름

```
Amazon Bestsellers (Top 100 × 5 categories)
         ↓
    CrawlerAgent (Playwright)
         ↓
    KnowledgeGraph + OWL Ontology 업데이트
         ↓
    MetricsAgent (SoS, HHI, CPI 계산)
         ↓
    Dashboard + AI Chatbot
```

### 2.2 모니터링 카테고리

| 카테고리 | Amazon Node ID | Level |
|----------|----------------|-------|
| Beauty & Personal Care | beauty | 0 |
| Skin Care | 11060451 | 1 |
| Lip Care | 3761351 | 2 |
| Lip Makeup | 11059031 | 2 |
| Face Powder | 11058971 | 2 |

### 2.3 핵심 모듈

| 모듈 | 파일 | 역할 |
|------|------|------|
| TrueHybridRetriever | `src/rag/true_hybrid_retriever.py` | RAG + KG + Ontology 통합 검색 |
| OWLReasoner | `src/ontology/owl_reasoner.py` | OWL 2 기반 자동 추론 |
| KnowledgeGraph | `src/ontology/knowledge_graph.py` | Triple Store 지식 그래프 |
| UnifiedBrain | `src/core/brain.py` | 자율 스케줄러 |

---

## 3. 주요 기능

### 3.1 자동 크롤링
- **매일 KST 22:00** Amazon Top 100 자동 수집 (UTC 13:00)
- 5개 카테고리 × 100개 제품 = 500개 제품 데이터
- 상태 파일: `data/scheduler_state.json`

#### 크롤링 시간 최적화 (22:00 KST)

| 시간 | 미국 동부 (EST) | 미국 서부 (PST) | 선택 이유 |
|------|----------------|----------------|----------|
| **22:00 KST** | 08:00 EST | 05:00 PST | 미국 새벽~아침 = 트래픽 최저 |

**왜 22:00 KST인가?**
- 미국 피크 타임 (7-10 PM EST) 판매가 BSR에 완전 반영된 후 수집
- 미국 트래픽 최저 시간대 → 차단 위험 최소
- 한국 09:00 출근 전 데이터 준비 완료

#### 안티봇 전략 (Stealth Only)

> **핵심 원칙**: "빠르게 하기보다 차단되지 않고 천천히 모두 수집"

| 기술 | 라이브러리 | 용도 |
|------|-----------|------|
| **Stealth 모드** | `playwright-stealth` | navigator.webdriver 제거, HeadlessChrome 숨김 |
| **핑거프린트** | `browserforge` | 실제 브라우저 핑거프린트 생성 |
| **User-Agent** | `fake-useragent` | 실제 UA 로테이션 |
| **행동 시뮬레이션** | 커스텀 | 랜덤 스크롤, 마우스 이동, 딜레이 |
| **회로 차단기** | 커스텀 | 연속 3회 실패 시 중단 + 백오프 |

**딜레이 설정 (보수적)**:
| 구간 | 기본 딜레이 | 랜덤 추가 |
|------|------------|----------|
| 상세 페이지 | 8초 | 0-4초 |
| 페이지 전환 | 12초 | 0-3초 |
| 카테고리 전환 | 45초 | 0-15초 |

**예상 크롤링 시간**: ~80-90분 (500개 제품)

### 3.2 KPI 분석
| 지표 | 설명 |
|------|------|
| **SoS** (Share of Shelf) | 브랜드 점유율 |
| **HHI** (Herfindahl-Hirschman Index) | 시장 집중도 |
| **CPI** (Competitive Position Index) | 경쟁 포지션 |

### 3.3 AI 챗봇
- **v3 API** (`/api/v3/chat`): 현재 프론트엔드 연결
- RAG + KG + Ontology 통합 컨텍스트
- Function Calling으로 실시간 데이터 조회

### 3.4 경쟁사 Deals 모니터링
- Lightning Deal, Deal of the Day 감지
- Slack/Email 자동 알림

---

## 4. 기술 스택

| 분류 | 기술 |
|------|------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| LLM | OpenAI GPT-4.1-mini (via LiteLLM) |
| RAG | ChromaDB, OpenAI Embeddings |
| Ontology | owlready2, Pellet Reasoner |
| 크롤링 | Playwright (Chromium), playwright-stealth, browserforge |
| 데이터 | Pandas, Google Sheets API |
| 배포 | Docker, Railway |

---

## 5. 설치 및 실행

```bash
# 1. 클론
git clone https://github.com/your-repo/AMORE-RAG-ONTOLOGY-HYBRID-AGENT.git
cd AMORE-RAG-ONTOLOGY-HYBRID-AGENT

# 2. 가상환경 및 의존성
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# 3. 환경 변수 (.env)
OPENAI_API_KEY=sk-...
API_KEY=your-api-key
AUTO_START_SCHEDULER=true

# 4. 실행
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001
```

**접속:** http://localhost:8001/dashboard

### 5.1 로컬 데이터 동기화

로컬 개발 시 Railway 서버의 최신 데이터를 동기화합니다:

```bash
# 누락된 데이터만 동기화
python scripts/sync_from_railway.py

# 전체 재동기화 (--force)
python scripts/sync_from_railway.py --force

# 확인만 (실제 동기화 없이)
python scripts/sync_from_railway.py --dry-run
```

> **Note**: 로컬 SQLite는 Railway와 자동 동기화되지 않습니다.
> 로컬 테스트 전에 수동으로 위 명령을 실행하세요.

---

## 6. API 레퍼런스

### 주요 엔드포인트

| Method | Endpoint | 설명 | 인증 |
|--------|----------|------|------|
| GET | `/api/health` | 헬스 체크 | - |
| GET | `/api/data` | 대시보드 데이터 | - |
| GET | `/dashboard` | 대시보드 UI | - |
| POST | `/api/v3/chat` | AI 챗봇 (권장) | - |
| POST | `/api/crawl/start` | 크롤링 시작 | API Key |
| GET | `/api/deals` | 경쟁사 Deals | - |

### 인증

```bash
curl -X POST "http://localhost:8001/api/crawl/start" \
  -H "X-API-Key: your-api-key"
```

---

## 7. 배포

### Railway

1. https://railway.app 에서 GitHub 연결
2. 환경 변수 설정:
   - `OPENAI_API_KEY`
   - `API_KEY`
   - `AUTO_START_SCHEDULER=true`
   - `GOOGLE_SHEETS_SPREADSHEET_ID`
   - `GOOGLE_SHEETS_CREDENTIALS_JSON`

### Docker

```bash
docker build -t amore-agent .
docker run -p 8001:8001 -e OPENAI_API_KEY=sk-... amore-agent
```

---

## 8. 업데이트 히스토리

### 2026-01-25: SQLite 동기화 누락 이슈 해결

**문제 발견:**
- 대시보드에서 기간 설정(01-07~01-25) 시 그래프가 01-21~01-25만 표시
- 원인: Google Sheets에는 데이터가 있으나 SQLite에 01-22~25일 데이터 누락

**원인 분석:**
- Railway 환경의 SQLite 볼륨 마운트 문제 또는 저장 오류
- 로컬 DB와 서버 DB 불일치

**해결 방안:**
| 구분 | 방법 |
|------|------|
| **즉시 해결** | `python scripts/sync_sheets_to_sqlite.py` 실행 |
| **자동 동기화** | 크롤링 완료 후 Sheets → SQLite 자동 동기화 추가 (`batch_workflow.py`) |
| **데이터 정합성 검사** | `src/tools/data_integrity_checker.py` 모듈 추가 |
| **로깅 강화** | SQLite 저장 실패 시 상세 에러 로깅 및 경고 메시지 |

**관련 파일:**
| 파일 | 역할 |
|------|------|
| `scripts/sync_sheets_to_sqlite.py` | 수동 동기화 스크립트 |
| `src/agents/storage_agent.py` | 이중 저장 에이전트 |
| `src/core/batch_workflow.py` | 자동 동기화 로직 추가 |
| `src/tools/data_integrity_checker.py` | 데이터 정합성 검사 모듈 |

**정합성 검사 사용법:**
```bash
# CLI 실행
python -m src.tools.data_integrity_checker

# 코드에서 사용
from src.tools.data_integrity_checker import check_data_integrity
result = await check_data_integrity()
print(f"Severity: {result['severity']}")
```

---

### 2026-01-25: 크롤링 시간 최적화 및 안티봇 전략

**크롤링 시간 변경**: KST 06:00 → **KST 22:00 (UTC 13:00)**
- 미국 트래픽 최저 시간대 크롤링으로 차단 위험 최소화
- US 피크 판매(7-10 PM EST) 반영된 BSR 데이터 수집
- 한국 09:00 업무 시작 전 데이터 준비 완료

**안티봇 Stealth 모드 적용**:
- `playwright-stealth`: navigator.webdriver 제거
- `browserforge`: 실제 브라우저 핑거프린트 생성
- `fake-useragent`: UA 로테이션
- 인간 행동 시뮬레이션 (스크롤, 마우스, 랜덤 딜레이)
- 회로 차단기 패턴 (연속 3회 실패 시 중단)

**할인 정보 수집 개선**:
- 상세 페이지 크롤링으로 list_price, coupon_text, promo_badges 수집
- 예상 수집률: list_price 70-90%, coupon_text 20-40%

### 2026-01-24: True RAG-Ontology 통합 계획

- `TrueHybridRetriever` v3 연결 계획 수립
- 벡터 검색 활성화 (ChromaDB + OpenAI Embeddings)
- OWL Ontology 추론 연결 (owlready2 + Pellet)
- 📄 상세 계획: [`docs/TRUE_RAG_ONTOLOGY_INTEGRATION_PLAN.md`](docs/TRUE_RAG_ONTOLOGY_INTEGRATION_PLAN.md)

### 2026-01-23: TDD 리팩토링 & 반응형 대시보드

**Clean Architecture 적용:**
- `src/domain/exceptions.py`: 커스텀 예외 8종
- `src/api/validators/input_validator.py`: 입력 검증 (프롬프트 인젝션 방어)
- `src/infrastructure/container.py`: DI 컨테이너
- 164개 단위 테스트

**대시보드 개선:**
- 모바일 반응형 디자인 (브레이크포인트: 1200px, 992px, 768px, 576px)
- slowapi 호환성 수정

### 2026-01-23: RAG 문서 통합

**11개 문서 체계화:**
| 유형 | 개수 | 위치 |
|------|------|------|
| 지표 가이드 | 4개 | `docs/guides/` |
| 시장 분석 | 7개 | `docs/market/` |

**QueryIntent 기반 검색:**
- DIAGNOSIS → 플레이북 우선
- TREND → 인텔리전스 우선
- CRISIS → 대응 가이드 우선
- METRIC → 지표 가이드 우선

### 2026-01-21: External Signal Collector

- RSS 피드: Allure, Byrdie, Refinery29
- Reddit API: r/SkincareAddiction, r/AsianBeauty
- 브랜드 인식 개선 (Multi-word 브랜드 우선 처리)

### 2026-01-20: 경쟁사 Deals 모니터링

- `src/tools/deals_scraper.py`: Amazon Deals 크롤러
- `src/tools/alert_service.py`: Slack/Email 알림
- 대시보드 Deals Monitor 페이지

### 2026-01-20: AI Customers Say 감성 분석

- Amazon AI 리뷰 요약 크롤링
- KnowledgeGraph 감성 관계 확장
- 8개 감성 기반 추론 규칙

### 2026-01-19: 카테고리 온톨로지

- Amazon 카테고리 계층 구조 (node_id, level)
- 순위 비교 시 카테고리 필터 적용
- 데이터 출처 명시 (Data Provenance)

### 2026-01-03: 자동 스케줄러 안정화

- KST 시간대 적용
- 스케줄러 상태 파일 저장
- Google Sheets 환경변수 credentials 지원

---

## 라이선스

MIT License

---

## 문서

| 문서 | 설명 |
|------|------|
| [`CLAUDE.md`](CLAUDE.md) | 개발 가이드 |
| [`docs/TRUE_RAG_ONTOLOGY_INTEGRATION_PLAN.md`](docs/TRUE_RAG_ONTOLOGY_INTEGRATION_PLAN.md) | RAG-Ontology 통합 계획 |
