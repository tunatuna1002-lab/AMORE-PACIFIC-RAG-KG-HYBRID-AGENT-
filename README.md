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
9. [리팩토링 & CI/CD 개선 (2026-02)](#9-리팩토링--cicd-개선-2026-02)
10. [업데이트 히스토리](#10-업데이트-히스토리)

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
| **테스트** | pytest, pytest-cov (현재 43%, 목표 60%) |

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
# 전체 테스트 (커버리지 포함)
python -m pytest tests/ -v

# 단위 테스트만
python -m pytest tests/unit/ -v --tb=short -x --timeout=60

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

### 테스트 현황

| 항목 | 수치 |
|------|------|
| 총 테스트 수 | 1,905개 |
| 통과율 | 100% (1,905 passed, 0 failed) |
| 커버리지 | 43.09% (목표 60%) |
| 테스트 구조 | `tests/unit/` (14개 서브디렉토리), `tests/eval/`, `tests/integration/`, `tests/adversarial/` |

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

## 9. 리팩토링 & CI/CD 개선 (2026-02)

2026-02-10 ~ 02-16, 6개 Phase에 걸쳐 코드 품질 및 테스트 인프라를 대폭 개선했습니다.
자세한 내용은 [`docs/REFACTORING_RESULTS.md`](docs/REFACTORING_RESULTS.md) 참조.

### 9.1 Before / After

| 지표 | Before (02-09) | After (02-16) | 변화 |
|------|----------------|---------------|------|
| src/ 총 코드 | ~97,000 lines | ~70,700 lines | **-27%** |
| Python 파일 수 | 155개 | 200개 | +29% (모듈 분할) |
| dashboard_api.py | 5,634줄 monolith | 3,236줄 + 12 route modules | **-43%** |
| 순환 의존성 | 23 cycles | 0 cycles | **완전 제거** |
| 테스트 수 | 238개 | 1,905개 | **+700%** |
| 테스트 커버리지 | 10.11% | 43.09% | **+33%p** |
| DI Container | 11 get_ 메서드 | 18 get_ 메서드 | +7 컴포넌트 |

### 9.2 Phase별 주요 변경

| Phase | 작업 | 문제 | 해결 |
|-------|------|------|------|
| **0** | Dead Code 삭제 | 미사용 코드 ~2,000줄 잔존 | 삭제 + 안전망 테스트 650개 작성 |
| **1-2** | Retriever 통합 | 4개 Retriever 분산, 순환 의존 | Strategy Pattern으로 2개로 통합, Domain Layer 순수성 확보 |
| **3** | dashboard_api 모듈화 | 5,634줄 monolith | `src/api/routes/` 12개 모듈로 분리 (-43%) |
| **4** | BatchWorkflow 이동 | core/에 위치한 Application 로직 | `src/application/workflows/`로 이동, 하위 호환 유지 |
| **5** | DI Container 완성 | 직접 import 의존 | Container 기반 DI 전환, 7개 컴포넌트 추가 등록 |
| **6** | 테스트 보강 | 238개, 10% 커버리지 | 5개 미테스트 모듈에 60개 테스트 추가, stale 4개 수정 |

### 9.3 CI/CD 파이프라인

GitHub Actions 워크플로우 (`.github/workflows/test.yml`)를 2-job 구조로 개선:

| Job | 내용 |
|-----|------|
| **test** | Ruff lint → 단위 테스트 (pytest + coverage) → 통합 테스트 (API key 있을 때만) |
| **security** | Bandit 보안 스캔 (`-ll -ii`) + pip-audit 취약 의존성 검사 |

주요 설정:
- Python 3.11, Playwright Chromium 설치 포함
- 커버리지: `--cov=src --cov-report=term-missing` (branch coverage 활성화)
- `fail_under = 0` (임시 — 안정화 후 점진적 상향 예정)

### 9.4 커버리지 로드맵 (진행 중)

현재 43% → 목표 60% 달성을 위한 6-Phase 계획 수립 완료.

| Phase | 대상 | 예상 커버리지 |
|-------|------|--------------|
| 1 | Low-hanging fruit (50-70% 파일 보강) | 45.2% |
| 2 | Core 비즈니스 로직 (brain, workflow, state) | 48.2% |
| 3 | 0% 커버리지 고영향 파일 | 49.9% |
| 4 | Core Agents (brain, chatbot, OWL) | 51.8% |
| 5 | Infrastructure & Tools (storage, scraper) | 54.1% |
| 6 | API 통합 테스트 | **59.8%** |

> 상세 분석: `.omc/autopilot/coverage-plan.md` 참조

---

## 10. 업데이트 히스토리

### 2026-02-10 - Amazon 크롤러 Top 100 수집 복구

**문제**: 카테고리당 100개가 아닌 60개만 수집 (전체 300개/500개)

**원인**: Amazon이 베스트셀러 페이지에 **lazy loading**을 도입하여 초기 로드 시 30개만 표시. 스크롤해야 나머지 20개가 추가 로드됨 (페이지당 50개). 기존 크롤러는 스크롤 없이 바로 파싱하여 30개 x 2페이지 = 60개만 수집.

**진단 과정**: Railway 환경에서 Playwright 디버그 스크립트를 실행하여 `[data-asin]` 카드 수, `span.zg-bdg-text` 순위 배지, 페이지네이션 구조를 확인. 스크롤 전 30개 → 스크롤 후 50개로 증가하는 것을 확인.

**해결**:
- `_scroll_to_load_all()`: 페이지 끝까지 스크롤하여 lazy-loaded 카드 전체 로드
- `_parse_bestseller_page()`: `[data-asin]` 순회 + 자체 rank 관리 대신 `span.zg-bdg-text` 순위 배지 기반 파싱으로 변경. 광고/스폰서 카드 자동 제외.
- `#zg-right-col` 컨테이너 내부만 파싱하여 정확도 향상

**결과**: 카테고리당 100개 (rank 1~100) 정상 수집 확인

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
