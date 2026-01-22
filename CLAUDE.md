# CLAUDE.md

> Essential context for Claude Code when working with this codebase

## Project Overview

**AMORE Pacific RAG-KG Hybrid Agent** - An autonomous AI system that monitors LANEIGE brand competitiveness on Amazon US marketplace.

### Core Features
- **Daily Auto-Crawling**: Amazon Top 100 bestsellers across 5 categories (KST 06:00)
- **KPI Analysis**: SoS (Share of Shelf), HHI, CPI calculations
- **AI Chatbot**: Hybrid RAG-powered Q&A (Knowledge Graph + Ontology + Documents)
- **Insight Generation**: LLM-based strategic insights

### Monitored Categories
1. Beauty & Personal Care
2. Skin Care
3. Lip Care
4. Lip Makeup
5. Face Powder

---

## Design System - AMOREPACIFIC Color Palette

모든 UI 개발 시 아래 컬러 팔레트를 **반드시** 적용합니다.

> "달 그림자와 같은 부드럽고 온화한 느낌의 '아모레 블루'와 깊은 바다와 사막의 밤을 연상시키는 이성적이고 강렬한 느낌의 '퍼시픽 블루'가 부드럽게 조화"

| 색상명 | HEX | RGB | 용도 |
|--------|-----|-----|------|
| **Amore Blue** | `#1F5795` | R31 G87 B149 | 부드럽고 온화한 느낌, 보조 강조, LANEIGE 브랜드 컬러 |
| **Pacific Blue** | `#001C58` | R0 G28 B88 | 강렬하고 이성적인 느낌, 헤더/사이드바, 주요 강조 |
| **Black** | `#000000` | R0 G0 B0 | 기본 텍스트 |
| **Gray** | `#7D7D7D` | R125 G125 B125 | 보조 텍스트, 비활성 상태, 테두리 |
| **White** | `#FFFFFF` | R255 G255 B255 | 배경, 카드 |

### CSS 변수 매핑

```css
:root {
    --pacific-blue: #001C58;    /* 메인 헤더, 사이드바 */
    --amore-blue: #1F5795;      /* 강조, 버튼, 링크 */
    --text-primary: #000000;    /* 기본 텍스트 */
    --text-secondary: #7D7D7D;  /* 보조 텍스트 */
    --bg-white: #FFFFFF;        /* 배경 */
}
```

### 사용 원칙

1. **Pacific Blue (#001C58)**: 헤더, 사이드바, 주요 CTA 버튼
2. **Amore Blue (#1F5795)**: 차트 강조색, 링크, 보조 버튼
3. **Gray (#7D7D7D)**: 비활성 상태, placeholder, 테두리
4. **White (#FFFFFF)**: 카드 배경, 콘텐츠 영역

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Backend | Python 3.11+, FastAPI 0.104+, Uvicorn |
| LLM | OpenAI GPT-4.1-mini via LiteLLM |
| Scraping | Playwright (Chromium headless) |
| Storage | Google Sheets API, JSON files |
| Data | Pandas, Pydantic 2.0+ |
| Deployment | Docker, Railway |

---

## Project Structure

```
├── main.py                      # CLI interface
├── dashboard_api.py             # FastAPI server (main entry point)
├── orchestrator.py              # Batch workflow orchestrator
├── start.py                     # Railway deployment entry
│
├── src/
│   ├── core/                    # Orchestration & scheduling
│   │   ├── brain.py             # UnifiedBrain - autonomous scheduler
│   │   ├── unified_orchestrator.py
│   │   ├── simple_chat.py
│   │   └── crawl_manager.py
│   │
│   ├── agents/                  # AI agents
│   │   ├── hybrid_chatbot_agent.py  # Active: KG+Ontology+RAG chatbot
│   │   ├── hybrid_insight_agent.py  # Active: KG+Ontology+RAG insights
│   │   ├── crawler_agent.py
│   │   ├── storage_agent.py
│   │   ├── metrics_agent.py
│   │   └── alert_agent.py
│   │
│   ├── ontology/                # Knowledge Graph & reasoning
│   │   ├── knowledge_graph.py   # Triple store implementation
│   │   ├── reasoner.py          # Ontology inference engine
│   │   ├── business_rules.py    # Domain-specific rules
│   │   └── schema.py            # Pydantic data schemas
│   │
│   ├── rag/                     # Retrieval-Augmented Generation
│   │   ├── hybrid_retriever.py  # KG + RAG integrated search
│   │   ├── context_builder.py
│   │   ├── retriever.py         # Document/keyword search
│   │   └── router.py            # Query classification
│   │
│   ├── tools/                   # Operational tools
│   │   ├── amazon_scraper.py    # Playwright crawler
│   │   ├── sheets_writer.py     # Google Sheets integration
│   │   ├── metric_calculator.py # KPI calculations
│   │   └── dashboard_exporter.py
│   │
│   ├── memory/                  # State management
│   │   ├── session.py
│   │   └── history.py
│   │
│   └── monitoring/              # Logging & tracing
│       ├── logger.py
│       └── tracer.py
│
├── dashboard/                   # Frontend
│   ├── amore_unified_dashboard_v4.html  # Latest dashboard
│   └── test_chat.html
│
├── data/                        # Runtime data
│   ├── dashboard_data.json
│   ├── scheduler_state.json
│   ├── raw_products/
│   └── metrics/
│
├── config/                      # Configuration
│   ├── thresholds.json
│   ├── rules.json
│   └── google_credentials.json  # (not in git)
│
├── docs/guides/                 # RAG reference documents
│   ├── Strategic Indicators Definition.md
│   ├── Metric Interpretation Guide.md
│   ├── Indicator Combination Playbook.md
│   └── Home Page Insight Rules.md
│
└── logs/                        # Audit logs
    └── chatbot_audit_*.log
```

---

## Development Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install browser for scraping
playwright install chromium
```

### Run Server
```bash
# Development (with auto-reload)
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001 --reload

# Production
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001
```

### Docker
```bash
# Build
docker build -t amore-agent .

# Run
docker run -p 8001:8001 \
  -e OPENAI_API_KEY=sk-... \
  -e API_KEY=your-api-key \
  amore-agent
```

### Manual Crawl Trigger
```bash
curl -X POST http://localhost:8001/api/crawl/start \
  -H "X-API-Key: your-api-key"
```

---

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...              # OpenAI API key for LLM

# Optional
API_KEY=...                        # API authentication key
GOOGLE_SPREADSHEET_ID=...          # Google Sheets spreadsheet ID
AUTO_START_SCHEDULER=true          # Enable scheduler on startup (default: false)
PORT=8001                          # Server port (default: 8001)
```

---

## Architecture

### Hybrid RAG System

`HybridRetriever` integrates 3 components:

1. **KnowledgeGraph** (`src/ontology/knowledge_graph.py`)
   - In-memory triple store
   - Entities: Brands, Products, Categories
   - Relations: competitor_of, ranks_higher_than, belongs_to

2. **OntologyReasoner** (`src/ontology/reasoner.py`)
   - Business rule inference engine
   - Applies domain logic (e.g., "If SoS > 30%, product is dominant")

3. **DocumentRetriever** (`src/rag/retriever.py`)
   - Keyword-based search over 4 guide documents in `docs/guides/`
   - ChromaDB support exists but currently disabled

### Agent Pattern

Agents follow input/output contracts:
- `CrawlerAgent` → raw product data
- `StorageAgent` → persists to Google Sheets
- `MetricsAgent` → calculates KPIs
- `HybridInsightAgent` → generates strategic insights
- `HybridChatbotAgent` → answers user queries

### Autonomous Scheduler

`UnifiedBrain` in `src/core/brain.py`:
- KST timezone-aware scheduling
- Daily crawl at 06:00 KST
- Persistent state in `data/scheduler_state.json`
- Graceful recovery across restarts

### Workflow (Think-Act-Observe)

`orchestrator.py` follows:
1. **Think**: Plan next action
2. **Act**: Execute agent
3. **Observe**: Validate results

---

## Clean Architecture Guidelines (v2026.01.22)

이 프로젝트는 **Clean Architecture** 원칙을 따릅니다. 새 코드 작성 시 반드시 아래 규칙을 준수하세요.

### Layer Structure (의존성 방향: 안쪽으로만)

```
src/
├── domain/           # Layer 1: Entities (핵심 비즈니스 규칙)
│   ├── entities/     # Product, Brand, RankRecord, Metrics
│   ├── interfaces/   # Protocol 정의 (Repository, Agent, Scraper)
│   └── value_objects/
│
├── application/      # Layer 2: Use Cases (애플리케이션 비즈니스 규칙)
│   ├── workflows/    # BatchWorkflow (DI 패턴)
│   ├── services/     # ChatService, InsightService
│   └── orchestrators/
│
├── adapters/         # Layer 3: Interface Adapters
│   ├── agents/       # Agent 구현체
│   └── rag/          # RAG 구현체
│
├── infrastructure/   # Layer 4: Frameworks & Drivers
│   ├── config/       # AppConfig
│   ├── persistence/  # Repository 구현체
│   ├── external/     # 외부 API (Amazon, LLM)
│   └── bootstrap.py  # DI Container
│
└── api/              # FastAPI Routes
```

### Import Rules (절대 위반 금지)

| From → To | 허용 여부 |
|-----------|----------|
| domain → (nothing) | ✅ domain은 외부 의존 없음 |
| application → domain | ✅ |
| adapters → domain, application | ✅ |
| infrastructure → domain, application | ✅ |
| **infrastructure → adapters** | ❌ 금지 |
| **domain → application** | ❌ 금지 |
| **domain → infrastructure** | ❌ 금지 |

### New Code Placement Guide

| 코드 유형 | 위치 |
|----------|------|
| 새 Entity/Model | `src/domain/entities/` |
| 새 Protocol/Interface | `src/domain/interfaces/` |
| 비즈니스 로직/워크플로우 | `src/application/workflows/` |
| Agent 구현체 | `src/adapters/agents/` |
| 외부 API 연동 | `src/infrastructure/external/` |
| DB/Storage 구현 | `src/infrastructure/persistence/` |

### TDD Workflow (테스트 먼저)

새 기능 개발 시 반드시 TDD를 따릅니다:

1. **🔴 RED**: 테스트 먼저 작성 (`tests/unit/{layer}/test_*.py`)
2. **🟢 GREEN**: 최소한의 구현으로 테스트 통과
3. **🔵 REFACTOR**: 코드 정리 (테스트 유지)

```bash
# 테스트 실행
python -m pytest tests/unit/domain/ -v      # Domain 테스트
python -m pytest tests/unit/application/ -v # Application 테스트
python -m pytest tests/ -v                  # 전체 테스트
```

### Dependency Injection Pattern

새 서비스/워크플로우는 Protocol 기반 DI를 사용합니다:

```python
# ❌ Bad: 직접 의존
from src.agents.crawler_agent import CrawlerAgent
class MyWorkflow:
    def __init__(self):
        self.crawler = CrawlerAgent()  # 직접 생성

# ✅ Good: Protocol 기반 DI
from src.domain.interfaces.agent import CrawlerAgentProtocol
class MyWorkflow:
    def __init__(self, crawler: CrawlerAgentProtocol):
        self.crawler = crawler  # 주입받음
```

### Deprecated Import Paths

기존 경로는 호환성을 위해 유지되지만, 새 코드는 새 경로를 사용하세요:

| Deprecated | Use Instead |
|------------|-------------|
| `from src.ontology.schema import *` | `from src.domain.entities import *` |
| `from src.ontology.relations import *` | `from src.domain.entities.relations import *` |

---

## API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/` | Health check | - |
| GET | `/api/health` | Railway health probe | - |
| GET | `/api/data` | Dashboard data JSON | - |
| GET | `/dashboard` | Dashboard HTML UI | - |
| POST | `/api/chat` | Chatbot v1 (RAG) | - |
| POST | `/api/v2/chat` | Chatbot v2 (Unified) | - |
| POST | `/api/v3/chat` | Chatbot v3 (Simple) | - |
| POST | `/api/crawl/start` | Manual crawl trigger | API Key |
| GET | `/api/crawl/status` | Crawl status | - |
| GET | `/api/v4/brain/status` | Scheduler status | - |
| POST | `/api/export/docx` | DOCX report export | - |

---

## Key Modules Reference

| Module | File | Purpose |
|--------|------|---------|
| UnifiedBrain | `src/core/brain.py` | Autonomous scheduler, agent coordination |
| HybridRetriever | `src/rag/hybrid_retriever.py` | KG + Ontology + RAG search |
| KnowledgeGraph | `src/ontology/knowledge_graph.py` | Triple store for entities/relations |
| OntologyReasoner | `src/ontology/reasoner.py` | Business rule inference |
| AmazonScraper | `src/tools/amazon_scraper.py` | Playwright-based crawler |
| ExternalSignalCollector | `src/tools/external_signal_collector.py` | RSS/Reddit/SNS trend collection |
| HybridChatbotAgent | `src/agents/hybrid_chatbot_agent.py` | RAG-powered chatbot |
| HybridInsightAgent | `src/agents/hybrid_insight_agent.py` | Strategic insight generation |
| MetricCalculator | `src/tools/metric_calculator.py` | SoS, HHI, CPI calculations |

---

## Common Development Tasks

### Adding a New Category
1. Edit `src/tools/amazon_scraper.py` - add category URL
2. Update `config/thresholds.json` if needed
3. Rebuild knowledge graph entities

### Modifying Chatbot Behavior
- Query routing: `src/rag/router.py`
- Context building: `src/rag/context_builder.py`
- Response generation: `src/agents/hybrid_chatbot_agent.py`
- Reference docs: `docs/guides/`

### Adding Business Rules
- Rule definitions: `src/ontology/business_rules.py`
- Config: `config/rules.json`
- Reasoner logic: `src/ontology/reasoner.py`

### Modifying Metrics
- Calculator: `src/tools/metric_calculator.py`
- Thresholds: `config/thresholds.json`
- Display: `dashboard/amore_unified_dashboard_v4.html`

### Dashboard Changes
- Main dashboard: `dashboard/amore_unified_dashboard_v4.html`
- API data source: `dashboard_api.py` → `/api/data`

---

## Code Conventions

### Async-First
```python
async def crawl_category(self, category: str) -> List[Product]:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ...
```

### Pydantic Models
```python
class Product(BaseModel):
    asin: str
    title: str
    brand: str
    rank: int
    price: Optional[float] = None
```

### Singleton Patterns
```python
_brain_instance: Optional[UnifiedBrain] = None

def get_brain() -> UnifiedBrain:
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = UnifiedBrain()
    return _brain_instance
```

### Structured Logging
```python
from src.monitoring.logger import get_logger
logger = get_logger(__name__)
logger.info("crawl_started", category=category, timestamp=now)
```

### Type Hints
All functions should have type hints for parameters and return values.

---

## Data Flow

```
Amazon Bestsellers (Top 100 × 5 categories)
         ↓
    CrawlerAgent (Playwright)
         ↓
    StorageAgent (Google Sheets)
         ↓
    MetricsAgent (KPI calculations)
         ↓
    HybridInsightAgent (GPT-4.1-mini + KG + Ontology)
         ↓
    DashboardExporter (JSON generation)
         ↓
    FastAPI Server → Dashboard UI / Chatbot
```

---

## Troubleshooting

### Common Issues

**Playwright not installed**
```bash
playwright install chromium
```

**Google Sheets authentication**
- Ensure `config/google_credentials.json` exists
- Or set `GOOGLE_APPLICATION_CREDENTIALS` env var

**OpenAI API errors**
- Check `OPENAI_API_KEY` is set correctly
- Verify API quota

**Scheduler not starting**
- Set `AUTO_START_SCHEDULER=true`
- Check `data/scheduler_state.json` for corruption

### Log Locations
- Server logs: stdout/stderr
- Chatbot audit: `logs/chatbot_audit_YYYY-MM-DD.log`
- Execution traces: `data/traces/`

---

## External Signal Collector (v2026.01.21)

뷰티 전문 매체 및 SNS에서 트렌드 신호를 수집하는 모듈입니다.

### Signal Tiers

| Tier | Sources | Purpose | Method |
|------|---------|---------|--------|
| Tier 1 | TikTok, Instagram | 바이럴 감지 | Manual input (TikTok Creative Center) |
| Tier 2 | YouTube, Reddit | 검증/리뷰 | Reddit API (무료) |
| Tier 3 | Allure, WWD, People | 권위 있는 근거 | RSS 피드 (무료) |
| Tier 4 | X (Twitter) | PR/실시간 이슈 | Manual input |

### Usage

```python
from src.tools.external_signal_collector import ExternalSignalCollector

collector = ExternalSignalCollector()
await collector.initialize()

# RSS에서 뷰티 기사 수집 (무료)
articles = await collector.fetch_all_rss_feeds(["LANEIGE", "K-Beauty"])

# Reddit에서 트렌드 수집 (무료)
reddit_posts = await collector.fetch_reddit_trends(["SkincareAddiction"])

# 수동 입력 (주간 트렌드 레이더)
collector.add_manual_media_input({
    "source": "allure",
    "date": "2026-01-10",
    "title": "2026 Skincare Trends",
    "quotes": ["펩타이드가 2026년 트렌드"],
    "keywords": ["peptide"]
})

# 보고서 섹션 생성
report = collector.generate_report_section()
```

### 보고서 출력 형식

```
■ 전문 매체 근거:
• Allure (1월 10일): "Lipification of Beauty 현상 가속화"
• People (1월 12일): "LANEIGE가 글래스 스킨 트렌드 선도"

■ 소비자 트렌드:
• TikTok #LipBasting: 520만 조회 (1월 14일 기준)
• Reddit r/SkincareAddiction: 립마스크 추천글 2,400 업보트
```

### 유료 API (주석 처리됨)

다음 API는 코드에 구현되어 있으나 비용 문제로 주석 처리됨:
- **NewsAPI**: $449/월 비즈니스, 개발자 무료 (제한적)
- **Bing News API**: $3/1,000 transactions, 무료 1,000/월
- **YouTube Data API**: 무료 10,000 quota/day

환경변수 설정 후 주석 해제하여 활성화:
```bash
NEWSAPI_KEY=...
BING_NEWS_API_KEY=...
YOUTUBE_API_KEY=...
```

---

## Brand Recognition (v2026.01.21)

`AmazonScraper._extract_brand()`에서 인식하는 브랜드 목록:

### Multi-word Brands (우선 처리)
Summer Fridays, Rare Beauty, La Roche-Posay, Beauty of Joseon, Tower 28,
Drunk Elephant, Paula's Choice, The Ordinary, Glow Recipe, Youth To The People,
Tatcha, Fresh, Sunday Riley, Supergoop, First Aid Beauty, Charlotte Tilbury,
Too Faced, Urban Decay, Fenty Beauty, Huda Beauty, Anastasia Beverly Hills,
Benefit Cosmetics, MAC Cosmetics, NARS, Clinique, Estee Lauder, Lancome

### Single-word Brands
LANEIGE, COSRX, TIRTIR, e.l.f., NYX, Maybelline, L'Oreal, Neutrogena,
CeraVe, SKIN1004, Anua, MEDICUBE, BIODANCE, Innisfree, MISSHA, ETUDE,
Benton, Purito, Klairs, Heimish, Isntree, Rovectin, Torriden, mixsoon,
Numbuzin, Revlon, Covergirl, Milani, ColourPop, Morphe, Tarte, Smashbox,
Hourglass, Glossier, Cetaphil, Aveeno, Olay, Garnier, Nivea
