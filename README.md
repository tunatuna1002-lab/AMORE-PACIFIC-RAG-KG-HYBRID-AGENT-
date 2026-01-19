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
2. 환경 변수 설정:
   - `OPENAI_API_KEY`: OpenAI API 키 (sk-proj-...)
   - `API_KEY`: API 인증 키
   - `AUTO_START_SCHEDULER`: `true`
   - `GOOGLE_SHEETS_SPREADSHEET_ID`: Google Sheets 스프레드시트 ID
   - `GOOGLE_SHEETS_CREDENTIALS_JSON`: Google 서비스 계정 JSON (전체 내용)
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

### 문제 5: Google Sheets Credentials 파일 없음 (2026-01-03 해결)

**증상:** Railway 배포 후 크롤링은 성공하지만 데이터 저장 실패
```
Google Sheets 초기화 실패: [Errno 2] No such file or directory: './config/google_credentials.json'
ERROR:storage:Failed to save raw data: 'NoneType' object has no attribute 'spreadsheets'
```

**원인:**
- `config/google_credentials.json` 파일이 `.gitignore`에 포함되어 Railway에 배포되지 않음
- Google 서비스 계정 credentials 파일에는 민감한 정보가 포함되어 Git에 커밋하면 안 됨

**해결:** 환경 변수에서 credentials JSON 문자열을 직접 로드하도록 수정

```python
# src/tools/sheets_writer.py 수정
def _get_credentials(self) -> Credentials:
    if self.credentials_json:  # 환경 변수 우선
        credentials_info = json.loads(self.credentials_json)
        return Credentials.from_service_account_info(credentials_info, scopes=self.SCOPES)
    else:  # 파일에서 로드
        return Credentials.from_service_account_file(self.credentials_path, scopes=self.SCOPES)
```

**Railway 설정 방법:**

1. Google Cloud Console에서 서비스 계정 JSON 파일 내용 복사
2. Railway Variables에 `GOOGLE_SHEETS_CREDENTIALS_JSON` 추가
3. 값으로 JSON 전체 내용을 붙여넣기 (한 줄로)

```
# Railway Variables에 추가할 환경 변수
GOOGLE_SHEETS_CREDENTIALS_JSON={"type":"service_account","project_id":"...","private_key":"...","client_email":"...",...}
GOOGLE_SHEETS_SPREADSHEET_ID=your-spreadsheet-id
```

**로컬 개발 환경:**
- 기존처럼 `./config/google_credentials.json` 파일 사용
- `.env` 파일에 `GOOGLE_SHEETS_SPREADSHEET_ID` 설정

---

### 문제 6: Google Sheets API 할당량 초과 (2026-01-02 해결)

**증상:** 크롤링 중 Google Sheets 저장 단계에서 오류 발생
```
googleapiclient.errors.HttpError: Quota exceeded for quota metric 'Write requests'
and limit 'Write requests per minute per user'
```

**원인:** 매 제품마다 개별 API 호출 → 60개 제품 = 60회 API 호출 → 할당량 초과

**해결:** 배치 처리로 API 호출 최소화 (`sheets_writer.py`)

```python
# 변경 전: 제품마다 API 호출
for product in products:
    await self.sheets.upsert_product(product)  # 60회 API 호출

# 변경 후: 일괄 처리 (2회 API 호출로 축소)
await self.sheets.upsert_products_batch(products)  # 1. 기존 제품 조회, 2. 신규 제품 일괄 추가
```

**관련 파일:**
- `src/tools/sheets_writer.py`: `upsert_products_batch()` 메서드 추가
- `src/agents/storage_agent.py`: 배치 처리 호출로 변경

---

### 문제 7: Spreadsheet ID 파싱 오류 (2026-01-02 해결)

**증상:** Google Sheets 저장 실패, 스프레드시트를 찾을 수 없음
```
HttpError 404: Requested entity was not found.
```

**원인:**
1. 환경 변수에 전체 URL이 입력됨 (ID만 필요)
2. 환경 변수에 줄바꿈/공백이 포함됨

**해결:** URL에서 ID 추출 및 `.strip()` 처리

```python
# src/tools/sheets_writer.py
raw_spreadsheet_id = spreadsheet_id or os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID") or ""
self.spreadsheet_id = raw_spreadsheet_id.strip()  # 공백/줄바꿈 제거
```

**올바른 환경 변수 설정:**
```
# 잘못됨 (전체 URL)
GOOGLE_SHEETS_SPREADSHEET_ID=https://docs.google.com/spreadsheets/d/1cNr3E2WSSbO83XXh_9V92jwc6nfsxjAogcswlHcjV9w/edit

# 올바름 (ID만)
GOOGLE_SHEETS_SPREADSHEET_ID=1cNr3E2WSSbO83XXh_9V92jwc6nfsxjAogcswlHcjV9w
```

---

### 문제 8: 크롤링 데이터 날짜가 하루 전으로 저장됨 (2026-01-03 해결)

**증상:** 한국 시간 1월 3일에 크롤링했는데 Google Sheets에 1월 2일로 저장됨

**원인:** Railway 서버는 UTC 시간대, `date.today()`가 UTC 기준 날짜 반환

```
한국 시간: 2026-01-03 08:18 (KST)
서버 시간: 2026-01-02 23:18 (UTC)
→ date.today() = 2026-01-02 ❌
```

**영향받은 파일:**
- `src/tools/amazon_scraper.py` - `snapshot_date` 생성
- `src/agents/crawler_agent.py` - `RankRecord.snapshot_date` 생성
- `src/tools/dashboard_exporter.py` - `generated_at` 타임스탬프

**해결:** 모든 날짜/시간 생성에 KST 시간대 적용

```python
# 변경 전 (UTC)
snapshot_date = date.today().isoformat()
generated_at = datetime.now().isoformat()

# 변경 후 (KST)
from datetime import timezone, timedelta
KST = timezone(timedelta(hours=9))

snapshot_date = datetime.now(KST).date().isoformat()
generated_at = datetime.now(KST).isoformat()
```

**관련 커밋:**
- `838be10`: fix: use KST timezone for snapshot_date instead of UTC
- `4cb10f0`: fix: use KST timezone for dashboard generated_at timestamp

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
2. Set environment variables:
   - `OPENAI_API_KEY`: OpenAI API key (sk-proj-...)
   - `API_KEY`: API authentication key
   - `AUTO_START_SCHEDULER`: `true`
   - `GOOGLE_SHEETS_SPREADSHEET_ID`: Google Sheets spreadsheet ID
   - `GOOGLE_SHEETS_CREDENTIALS_JSON`: Google service account JSON (full content)
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

---

## Changelog (개선 타임라인)

### 2026-01-19: 카테고리 온톨로지 & 대시보드 UX 개선

#### 🏗️ 카테고리 계층 구조 온톨로지
- **Amazon 카테고리 node_id 추가**: `config/category_hierarchy.json`에 각 카테고리별 `amazon_node_id`, `level`, `parent_id` 추가
- **크롤러 개선**: `amazon_scraper.py`에서 제품별 `category_id`, `amazon_node_id`, `category_name`, `category_level` 수집
- **동일 ASIN 다중 카테고리 인식**: 하나의 제품이 여러 카테고리에서 다른 순위를 가질 수 있음을 시스템이 인식

#### 🐛 버그 수정
- **순위 비교 로직 수정** (`dashboard_exporter.py`):
  - `_calculate_rank_change()`에 `category_id` 필터 추가
  - 이전: ASIN만으로 비교 → "73위→4위 급상승" 오류 발생
  - 이후: 동일 카테고리 내에서만 순위 변동 비교
- **ASIN 중복 표시 제거**: `_generate_action_items()`에서 ASIN별 가장 좋은 순위 카테고리만 표시
- **평점 0.00 처리**: `rating > 0` 조건 추가로 데이터 없음을 "평점 하락"으로 오인하지 않음

#### 📊 데이터 출처 인용 (Data Provenance)
- **챗봇 응답**: 응답 상단에 `📅 데이터 기준: Amazon US Best Sellers {날짜} 수집` 표시
- **인사이트**: `data_source` 필드 추가 (platform, collected_at, snapshot_date, disclaimer)
- **대시보드 JSON**: `dashboard_data.json`에 `data_source` 섹션 추가

#### 🎨 대시보드 UX 개선
- **카테고리 포지션 툴팁**: "TOP 4" 카드에 마우스 hover 시 각 제품의 카테고리별 순위 표시
- **노출 상태 툴팁**: "Laneige 노출 상태" 카드에 hover 시 상태 판단 근거 (Top 10 제품 수, 순위 변동) 표시
- **차트 지표 설명 툴팁**: "SoS × Avg Rank" 차트 제목 hover 시 지표 의미 설명
- **차트 날짜 범위 선택**: 순위 추이, 제품 매트릭스, 할인 추이 차트에 1일/7일/14일/30일 선택 버튼 추가

#### 📅 차트 기간 선택 기능 (신규)
- **캘린더 Date Range Picker**: "SoS × Avg Rank" 차트에 시작일/종료일 캘린더 선택기 추가
- **분석 기간 표시**: 차트 툴팁에 현재 분석 기간 (예: "2026-01-12 ~ 2026-01-19") 표시
- **데이터 미존재 구간 알림**: 선택한 기간 내 데이터가 없는 날짜를 자동 감지하여 경고 표시
  - 연속된 날짜는 범위로 그룹화 (예: "2026-01-15 ~ 2026-01-17: 데이터 없음")
- **Historical API 개선**: `/api/historical` 응답에 `available_dates`, `brand_metrics` 필드 추가

#### 📁 변경된 파일
| 파일 | 변경 내용 |
|------|----------|
| `config/category_hierarchy.json` | 카테고리 계층 구조 v1.1 |
| `config/thresholds.json` | 카테고리별 amazon_node_id, level 추가 |
| `src/tools/amazon_scraper.py` | 카테고리 메타데이터 수집 |
| `src/tools/dashboard_exporter.py` | 카테고리 인식 순위 비교, 데이터 출처 |
| `src/agents/hybrid_chatbot_agent.py` | 데이터 출처 인용 |
| `src/agents/hybrid_insight_agent.py` | 데이터 출처 정보 |
| `dashboard/amore_unified_dashboard_v4.html` | 툴팁, 날짜 선택기, 기간 선택 UI, 데이터 갭 알림 |
| `dashboard_api.py` | Historical API에 brand_metrics, available_dates 추가 |

---

### 2026-01-18: AI 인사이트 & 할인 분석 강화

#### 🧠 AI 인사이트 개선
- 인사이트에 데이터 출처 명시
- 카테고리 계층 인식 프롬프트 추가
- 순위 언급 시 카테고리 명시 유도

#### 💰 할인/프로모션 분석
- 제품별 할인율, 쿠폰, Deal 정보 수집
- 경쟁사 할인 전략 비교 분석
- 할인 추이 차트 추가

---

### 2026-01-03: 자동 크롤링 스케줄러 안정화

#### 🔧 핵심 수정
- KST 시간대 적용 (UTC → KST)
- 스케줄러 상태 파일 저장 (`scheduler_state.json`)
- Google Sheets 환경변수 credentials 지원
- API 할당량 초과 방지 (배치 처리)

#### 📁 변경된 파일
| 파일 | 변경 내용 |
|------|----------|
| `src/core/brain.py` | KST 시간대, 상태 저장 |
| `src/core/crawl_manager.py` | KST 날짜 체크 |
| `src/tools/sheets_writer.py` | 환경변수 credentials, 배치 처리 |
| `src/tools/amazon_scraper.py` | KST snapshot_date |

---

### 2026-01-02: Google Sheets 통합 개선

#### 🔧 핵심 수정
- 배치 upsert로 API 호출 최소화
- Spreadsheet ID 파싱 오류 수정
- 환경변수 줄바꿈/공백 처리

---

### 초기 릴리스: Hybrid RAG 시스템

#### 🏗️ 핵심 아키텍처
- Knowledge Graph (Triple Store)
- Ontology Reasoner (비즈니스 규칙)
- Document Retriever (키워드 기반)
- LLM 통합 (GPT-4.1-mini)

#### 📊 대시보드
- 실시간 순위 모니터링
- SoS, HHI, CPI 지표
- AI 챗봇 통합
