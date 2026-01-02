# AMORE Pacific RAG-KG Hybrid Agent

> **Level 4 Autonomous Agent System** - Amazon Bestseller Analytics Platform powered by RAG + Knowledge Graph + LLM-First Architecture

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[한국어 버전](./README.md)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [System Architecture](#system-architecture)
4. [Project Structure](#project-structure)
5. [Core Modules](#core-modules)
6. [API Reference](#api-reference)
7. [Strategic KPIs](#strategic-kpis)
8. [Installation](#installation)
9. [Deployment](#deployment)
10. [Development History](#development-history)

---

## Project Overview

### Background & Purpose

An **AI Agent System** developed for AMORE Pacific's LANEIGE brand to maintain competitiveness in the Amazon US market. It collects real-time bestseller ranking data and provides strategic insights through analysis.

### Core Values

| Value | Description |
|-------|-------------|
| **Automation** | Daily automatic data collection, analysis, and insight generation |
| **Intelligence** | Hybrid AI based on RAG + Knowledge Graph |
| **Autonomy** | Minimal human intervention with LLM-First decision making |
| **Real-time** | Immediate awareness of rank changes and competitor trends |

### Key Features

```
┌────────────────────────────────────────────────────────────────────┐
│                        Feature Overview                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📊 Autonomous Crawling    📈 KPI Analytics     💬 AI Chatbot      │
│  ├─ Daily 09:00 auto      ├─ SoS (Share)        ├─ Natural language│
│  ├─ Top 100 collection    ├─ HHI (Concentration)├─ RAG-based       │
│  └─ Category-wise         └─ CPI (Position)     └─ KG reasoning    │
│                                                                     │
│  🔔 Alert System           📋 Report Generation  🎯 Knowledge Graph │
│  ├─ Rank change detection ├─ DOCX export        ├─ Brand-Product   │
│  ├─ Email notifications   ├─ Daily insights     ├─ Ontology        │
│  └─ Threshold config      └─ Dashboard viz      └─ Business rules  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Why LLM-First?

**Limitations of rule-based systems:**
```
[Rule-Based]
if SoS > 15% and rank < 10:
    return "Good"
else:
    return "Warning"

→ Problem: What if SoS = 14.9%, rank = 11?
           Rules say "Warning", but it might actually be a good situation
```

**Advantages of LLM-First approach:**
```
[LLM-First]
context = {
    SoS: 14.9%,
    rank: 11,
    daily_change: +3%,
    competitor_SoS: 12%,
    market_trend: rising
}

LLM Analysis → "With SoS at 14.9% exceeding competitors (12%),
                and rank improving day-over-day, this is a positive situation."
```

- LLM analyzes context for optimal decisions in all situations
- Complex business logic handled in natural language
- Flexible response to new patterns

---

## Tech Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Main language |
| FastAPI | 0.104+ | Async API server |
| LiteLLM | 1.40+ | LLM provider integration |
| Uvicorn | 0.24+ | ASGI server |

### AI/ML
| Technology | Purpose |
|------------|---------|
| OpenAI GPT-4 | LLM inference |
| ChromaDB | Vector database |
| Sentence Transformers | Embedding model |
| Custom Knowledge Graph | Ontology reasoning |

### Data & Integration
| Technology | Purpose |
|------------|---------|
| Playwright | Amazon crawling |
| Pandas/NumPy | Data processing |
| Google Sheets API | Data persistence |
| python-docx | Report generation |

### Deployment
| Technology | Purpose |
|------------|---------|
| Docker | Containerization |
| Railway | Cloud deployment |

---

## System Architecture

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Level 4 Autonomous Agent                         │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        UnifiedBrain                              │    │
│  │                     (LLM-First Decision)                         │    │
│  │                                                                   │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐   │    │
│  │  │  Priority   │  │  Autonomous  │  │     Event System      │   │    │
│  │  │    Queue    │  │  Scheduler   │  │   (Alert/Callback)    │   │    │
│  │  │            │  │              │  │                        │   │    │
│  │  │ USER > ALERT│  │ 09:00 Crawl │  │ on_alert: send email   │   │    │
│  │  │  > SCHEDULED│  │ 30min check │  │ on_complete: log       │   │    │
│  │  └─────────────┘  └──────────────┘  └───────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                      │
│                    ┌───────────────┼───────────────┐                     │
│                    ▼               ▼               ▼                     │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐│
│  │     QueryAgent      │ │   WorkflowAgent     │ │     AlertAgent      ││
│  │                     │ │                     │ │                     ││
│  │ • Query analysis    │ │ • Think-Act-Observe │ │ • Threshold monitor ││
│  │ • RAG+KG search     │ │ • Batch execution   │ │ • Alert generation  ││
│  │ • LLM response      │ │ • Error recovery    │ │ • Email dispatch    ││
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘│
│                                    │                                      │
├────────────────────────────────────┴──────────────────────────────────────┤
│                              Core Components                               │
│                                                                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐│
│  │  ContextGatherer │  │  HybridRetriever │  │    ResponsePipeline      ││
│  │                  │  │                  │  │                          ││
│  │  Data collection │  │  ┌────────────┐  │  │  • Response generation   ││
│  │  • Metric load   │  │  │    RAG     │  │  │  • Confidence scoring    ││
│  │  • KG query      │  │  │ (Doc search)│  │  │  • Caching               ││
│  │  • History       │  │  ├────────────┤  │  │  • Formatting            ││
│  │                  │  │  │ Knowledge  │  │  │                          ││
│  │                  │  │  │   Graph    │  │  │                          ││
│  │                  │  │  │ (Reasoning)│  │  │                          ││
│  └──────────────────┘  │  └────────────┘  │  └──────────────────────────┘│
│                        └──────────────────┘                               │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                              Execution Layer                               │
│                                                                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│  │ CrawlerAgent │ │ StorageAgent │ │ MetricsAgent │ │ HybridChatbot    │ │
│  │              │ │              │ │              │ │                  │ │
│  │ Amazon crawl │ │ Data storage │ │ KPI calc     │ │ Interactive chat │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────────┘ │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                Data Layer                                  │
│                                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────────┐  │
│  │   ChromaDB   │  │     JSON     │  │         Google Sheets          │  │
│  │  (Vectors)   │  │   (Cache)    │  │         (Persistence)          │  │
│  └──────────────┘  └──────────────┘  └────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

### Data Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        User Query Processing Flow                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    "What's LANEIGE's current ranking?"                                  │
│                    │                                                     │
│                    ▼                                                     │
│    ┌──────────────────────────────┐                                     │
│    │         UnifiedBrain         │                                     │
│    │    1. Check cache (Hit/Miss) │                                     │
│    │    2. Set priority (USER)    │                                     │
│    └──────────────┬───────────────┘                                     │
│                   │                                                      │
│                   ▼                                                      │
│    ┌──────────────────────────────┐                                     │
│    │       ContextGatherer        │                                     │
│    │                              │                                     │
│    │  ┌────────────────────────┐  │                                     │
│    │  │ 1. Entity extraction   │  │                                     │
│    │  │    "LANEIGE" → Brand   │  │                                     │
│    │  │    "ranking" → Metric  │  │                                     │
│    │  └────────────────────────┘  │                                     │
│    │                              │                                     │
│    │  ┌────────────────────────┐  │                                     │
│    │  │ 2. Load current metrics│  │                                     │
│    │  │    rank: 8             │  │                                     │
│    │  │    rank_delta: +2      │  │                                     │
│    │  │    sos: 15.3%          │  │                                     │
│    │  └────────────────────────┘  │                                     │
│    └──────────────┬───────────────┘                                     │
│                   │                                                      │
│                   ▼                                                      │
│    ┌──────────────────────────────┐                                     │
│    │       HybridRetriever        │                                     │
│    │                              │                                     │
│    │  ┌──────────┐ ┌───────────┐  │                                     │
│    │  │   RAG    │ │    KG     │  │                                     │
│    │  │          │ │           │  │                                     │
│    │  │ Document │ │ Relation  │  │                                     │
│    │  │ search   │ │ reasoning │  │                                     │
│    │  │ "rank"   │ │ LANEIGE   │  │                                     │
│    │  │ definition│ │ →AMORE   │  │                                     │
│    │  └──────────┘ └───────────┘  │                                     │
│    └──────────────┬───────────────┘                                     │
│                   │                                                      │
│                   ▼                                                      │
│    ┌──────────────────────────────┐                                     │
│    │         LLM Call             │                                     │
│    │                              │                                     │
│    │  Context:                    │                                     │
│    │  - Current rank: #8 (↑2)    │                                     │
│    │  - SoS: 15.3%               │                                     │
│    │  - Ahead of competitors      │                                     │
│    │  - Rank definition doc       │                                     │
│    │                              │                                     │
│    │  → GPT-4 response generation │                                     │
│    └──────────────┬───────────────┘                                     │
│                   │                                                      │
│                   ▼                                                      │
│    ┌──────────────────────────────┐                                     │
│    │      ResponsePipeline        │                                     │
│    │                              │                                     │
│    │  • Confidence score: 0.92    │                                     │
│    │  • Cache storage             │                                     │
│    │  • Source attachment         │                                     │
│    └──────────────┬───────────────┘                                     │
│                   │                                                      │
│                   ▼                                                      │
│    ┌──────────────────────────────┐                                     │
│    │           Response           │                                     │
│    │                              │                                     │
│    │  "LANEIGE Lip Sleeping Mask  │                                     │
│    │   is currently ranked #8,    │                                     │
│    │   up 2 positions from        │                                     │
│    │   yesterday. With 15.3% SoS, │                                     │
│    │   it maintains an advantage  │                                     │
│    │   over competitors."         │                                     │
│    │                              │                                     │
│    │  Confidence: 92%             │                                     │
│    │  Source: Dashboard Data      │                                     │
│    └──────────────────────────────┘                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
AMORE-RAG-ONTOLOGY-HYBRID AGENT/
│
├── 📁 src/                          # Main source code
│   │
│   ├── 📁 core/                     # ⭐ Core orchestration
│   │   ├── brain.py                 # Level 4 autonomous brain (LLM-First)
│   │   ├── unified_orchestrator.py  # Unified orchestrator (v2 API)
│   │   ├── context_gatherer.py      # RAG + KG context collection
│   │   ├── response_pipeline.py     # Response generation pipeline
│   │   ├── confidence.py            # Confidence scoring
│   │   ├── cache.py                 # Response caching (TTL-based)
│   │   ├── state.py                 # Orchestrator state management
│   │   ├── tools.py                 # Agent tool definitions
│   │   ├── models.py                # Data models (Context, Response)
│   │   └── ...
│   │
│   ├── 📁 agents/                   # ⭐ Agent modules
│   │   ├── query_agent.py           # Query processing agent (Brain)
│   │   ├── workflow_agent.py        # Batch workflow agent
│   │   ├── alert_agent.py           # Alert generation agent
│   │   ├── crawler_agent.py         # Amazon crawling agent
│   │   ├── storage_agent.py         # Data storage agent
│   │   ├── metrics_agent.py         # KPI calculation agent
│   │   ├── hybrid_chatbot_agent.py  # Hybrid chatbot (RAG+KG)
│   │   └── ...
│   │
│   ├── 📁 rag/                      # ⭐ RAG system
│   │   ├── router.py                # Query type classification
│   │   ├── retriever.py             # Document retriever (ChromaDB)
│   │   ├── hybrid_retriever.py      # RAG + KG hybrid search
│   │   └── ...
│   │
│   ├── 📁 ontology/                 # ⭐ Knowledge Graph
│   │   ├── knowledge_graph.py       # KG implementation (triple store)
│   │   ├── reasoner.py              # Ontology reasoning engine
│   │   ├── schema.py                # Entity schema definitions
│   │   └── ...
│   │
│   ├── 📁 memory/                   # Conversation memory
│   ├── 📁 tools/                    # Utility tools
│   └── 📁 monitoring/               # Monitoring
│
├── 📁 dashboard/                    # Frontend
├── 📁 data/                         # Data storage
├── 📁 docs/                         # Documentation
├── 📁 config/                       # Configuration
├── 📁 tests/                        # Tests
│
├── 📄 dashboard_api.py              # ⭐ FastAPI server (main)
├── 📄 start.py                      # Server start script
├── 📄 Dockerfile                    # Docker config
├── 📄 railway.toml                  # Railway deployment
└── 📄 requirements.txt              # Python dependencies
```

---

## Core Modules

### 1. UnifiedBrain (`src/core/brain.py`)

**The central brain of Level 4 Autonomous Agent** - Controls all agents

```python
from src.core.brain import UnifiedBrain, get_initialized_brain, BrainMode, TaskPriority

# Get Brain instance (singleton)
brain = await get_initialized_brain()

# Process user query
response = await brain.process_query(
    query="What's LANEIGE's current ranking?",
    session_id="user_123",
    current_metrics=dashboard_data
)

# Start autonomous scheduler
await brain.start_scheduler()

# Run autonomous cycle manually
result = await brain.run_autonomous_cycle()

# Check alerts
alerts = await brain.check_alerts(metrics_data)
```

**Key Features:**

| Feature | Description |
|---------|-------------|
| **LLM-First Decision** | All decisions made by LLM, no rule-based fast path |
| **Priority Queue** | `USER_REQUEST(0) > CRITICAL_ALERT(1) > SCHEDULED(2) > BACKGROUND(3)` |
| **Autonomous Scheduler** | Daily crawl (09:00), periodic alert check (30min) |
| **Event System** | Callback handling for alerts, completion, errors |

**Operating Modes:**

| Mode | Description | Trigger |
|------|-------------|---------|
| `IDLE` | Standby | Initial state |
| `RESPONDING` | Processing user query | Query received |
| `AUTONOMOUS` | Executing autonomous task | Scheduler trigger |
| `EXECUTING` | Running agent | Tool call |
| `ALERTING` | Processing alert | Alert condition met |

---

### 2. QueryAgent (`src/agents/query_agent.py`)

**Dedicated agent for user query processing** - Accurate responses via RAG + KG hybrid search

```python
from src.agents import QueryAgent

query_agent = QueryAgent(
    model="gpt-4o-mini",
    kg_persist_path="./data/knowledge_graph.json",
    cache_ttl=3600
)
await query_agent.initialize()

# Process query
result = await query_agent.process("What is LANEIGE's SoS?")

print(result.response)       # Response text
print(result.confidence)     # Confidence score
print(result.sources)        # Sources
print(result.entities)       # Extracted entities
print(result.inferences)     # KG inference results
```

---

### 3. WorkflowAgent (`src/agents/workflow_agent.py`)

**Batch workflow execution agent** - Complex task execution with Think-Act-Observe pattern

```python
from src.agents import WorkflowAgent

workflow_agent = WorkflowAgent()
await workflow_agent.initialize()

# Run full workflow
result = await workflow_agent.run_workflow(
    categories=["Lip Care", "Skin Care"],
    session_id="daily_batch"
)
```

**Workflow Steps:**
```
CRAWL → STORE → UPDATE_KG → CALCULATE → INSIGHT → EXPORT
```

---

### 4. Knowledge Graph (`src/ontology/`)

**Ontology-based relationship modeling**

```python
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner

kg = KnowledgeGraph(persist_path="./data/knowledge_graph.json")

# Add triples (Subject - Predicate - Object)
kg.add_triple("LANEIGE", "belongsTo", "AMORE Pacific")
kg.add_triple("LANEIGE", "hasProduct", "Lip Sleeping Mask")

# Reasoning
reasoner = OntologyReasoner(kg)
inferences = reasoner.infer("LANEIGE")
```

---

## API Reference

### Base URL
```
Production: https://amore-pacific-rag-kg-hybrid-agent-production.up.railway.app
Local: http://localhost:8001
```

### Endpoint Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            API Endpoints                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📊 Data & Health                                                        │
│  ├─ GET  /                          Health check                         │
│  ├─ GET  /api/health                Detailed health check                │
│  └─ GET  /api/data                  Dashboard data                       │
│                                                                          │
│  💬 Chat API (by version)                                               │
│  ├─ POST /api/chat                  v1: Basic RAG chatbot               │
│  ├─ POST /api/v2/chat               v2: Unified orchestrator            │
│  ├─ POST /api/v3/chat               v3: Simplified LLM chatbot          │
│  └─ POST /api/v4/chat               v4: Level 4 Brain (⭐ Recommended)  │
│                                                                          │
│  🧠 Brain API (v4)                                                       │
│  ├─ GET  /api/v4/brain/status           Status                          │
│  ├─ POST /api/v4/brain/scheduler/start  Start scheduler                 │
│  ├─ POST /api/v4/brain/scheduler/stop   Stop scheduler                  │
│  ├─ POST /api/v4/brain/autonomous-cycle Manual autonomous cycle         │
│  ├─ POST /api/v4/brain/check-alerts     Check alerts                    │
│  ├─ GET  /api/v4/brain/stats            Statistics                      │
│  └─ POST /api/v4/brain/mode             Change mode                     │
│                                                                          │
│  🔔 Alert API                                                            │
│  📡 Crawl API                                                            │
│  📄 Export API                                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### v4 Chat API (Recommended)

**Request:**
```http
POST /api/v4/chat
Content-Type: application/json

{
  "message": "What's LANEIGE's current ranking?",
  "session_id": "user_123",
  "skip_cache": false
}
```

**Response:**
```json
{
  "text": "LANEIGE Lip Sleeping Mask is currently ranked #8 in Lip Care. It moved up 2 positions from yesterday with a 15.3% SoS, maintaining an advantage over competitors.",
  "confidence": 0.92,
  "sources": ["Dashboard Data", "Strategic Indicators Definition"],
  "reasoning": "Retrieved rank definition from RAG, inferred LANEIGE-Lip Care relationship from KG, combined with current metric data.",
  "tools_used": ["query_data", "query_knowledge_graph"],
  "processing_time_ms": 1234.5,
  "from_cache": false,
  "brain_mode": "responding"
}
```

---

## Strategic KPIs

### KPI Definitions

| KPI | Full Name | Description | Formula |
|-----|-----------|-------------|---------|
| **SoS** | Share of Shelf | Brand share | Brand products / Top 100 × 100% |
| **HHI** | Herfindahl-Hirschman Index | Market concentration | Σ(market share²) × 10,000 |
| **CPI** | Competitive Position Index | Competitive position | Weighted rank score (higher is better) |
| **Volatility** | Rank Volatility | Rank stability | Standard deviation of rank changes |
| **Top10 Count** | Top 10 Products | Premium visibility | Products in Top 10 |
| **Avg Rank** | Average Rank | Mean position | Average rank of brand products |

### KPI Interpretation Guide

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        KPI Interpretation Matrix                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SoS (Share of Shelf)                                                   │
│  ├─ 20%+ : 🟢 Market dominant position                                  │
│  ├─ 10-20%: 🟡 Competitive position                                     │
│  └─ <10% : 🔴 Need to expand share                                      │
│                                                                          │
│  HHI (Market Concentration)                                              │
│  ├─ <1,500  : 🟢 Fragmented market (competitive)                        │
│  ├─ 1,500-2,500: 🟡 Moderate concentration                              │
│  └─ >2,500  : 🔴 Highly concentrated (oligopoly)                        │
│                                                                          │
│  Volatility (Rank Volatility)                                            │
│  ├─ <3   : 🟢 Stable                                                    │
│  ├─ 3-7  : 🟡 Moderate                                                  │
│  └─ >7   : 🔴 Unstable (monitoring needed)                              │
│                                                                          │
│  Combined Interpretation Examples:                                       │
│  ├─ SoS↑ + Rank↓ = Low-price products increasing? Review premium strategy│
│  ├─ SoS↓ + Rank↑ = Core product focus successful                        │
│  └─ HHI↑ + SoS↓ = Competitor dominance, response strategy needed        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.11+
- pip
- (Optional) Docker
- OpenAI API Key
- (Optional) Google Cloud service account (for Sheets API)

### 1. Clone Repository
```bash
git clone https://github.com/tunatuna1002-lab/AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-.git
cd AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-
```

### 2. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

# Install Playwright browser (for crawling)
playwright install chromium
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Required
OPENAI_API_KEY=sk-...

# Optional (for Google Sheets)
GOOGLE_SHEETS_SPREADSHEET_ID=...
GOOGLE_APPLICATION_CREDENTIALS=./config/credentials.json

# Settings
DATA_PATH=./data/dashboard_data.json
LOG_LEVEL=INFO
```

### 5. Run Server
```bash
# Development mode (auto-reload)
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001 --reload

# Or direct execution
python dashboard_api.py
```

### 6. Verify Installation
- API Docs: http://localhost:8001/docs
- Dashboard: http://localhost:8001/dashboard
- Health Check: http://localhost:8001/api/health

---

## Deployment

### Railway Deployment (Recommended)

1. **Create Railway account**: https://railway.app

2. **Create new project**
   - Dashboard → "New Project"
   - Select "Deploy from GitHub repo"
   - Connect repository

3. **Configure environment variables**
   - Settings → Variables
   ```
   OPENAI_API_KEY=sk-...
   PORT=8001
   ```

4. **Configure domain**
   - Settings → Domains
   - Click "Generate Domain"
   - Or connect custom domain

5. **Automatic deployment**
   - Push to GitHub triggers auto-redeploy
   - Check status in Deployments tab

### Docker Deployment

```bash
# Build
docker build -t amore-agent .

# Run
docker run -p 8001:8001 \
  -e OPENAI_API_KEY=sk-... \
  -e PORT=8001 \
  amore-agent
```

---

## Development History

### Phase 1: Foundation
**Goal**: Amazon data collection and storage
- ✅ CrawlerAgent: Amazon Top 100 crawling
- ✅ StorageAgent: Google Sheets integration
- ✅ Basic metric calculation

### Phase 2: Analytics
**Goal**: Strategic KPI introduction
- ✅ MetricsAgent: SoS, HHI, CPI calculation
- ✅ Volatility analysis
- ✅ Competitor comparison

### Phase 3: AI Integration
**Goal**: Natural language interface
- ✅ InsightAgent: LLM-based insight generation
- ✅ ChatbotAgent: Interactive Q&A
- ✅ RAG System: Document search-based responses

### Phase 4: Knowledge Graph
**Goal**: Structured knowledge representation
- ✅ Knowledge Graph implementation
- ✅ Ontology schema design
- ✅ RAG + KG hybrid search

### Phase 5: Production Hardening
**Goal**: Production readiness
- ✅ Multi-version API (v1, v2, v3)
- ✅ Background crawl management
- ✅ Audit trail logging

### Phase 6: Level 4 Autonomous (Current)
**Goal**: Fully autonomous agent
- ✅ UnifiedBrain: LLM-First decision making
- ✅ QueryAgent: Query processing
- ✅ WorkflowAgent: Batch execution
- ✅ Autonomous scheduler
- ✅ Event-based alert system
- ✅ v4 API endpoints

### Architecture Decision Records (ADR)

| Decision | Choice | Alternative | Reason |
|----------|--------|-------------|--------|
| Decision making | LLM-First | Rule-First | Rules miss edge cases |
| Search method | RAG + KG Hybrid | RAG only | KG enables relationship reasoning |
| Agent structure | Brain-centric | Distributed | Single control point for consistency |
| Priority | User request first | FIFO | Better UX |
| Caching | TTL-based | Permanent | Data freshness |
| Error handling | Per-agent strategy | Uniform | Flexible recovery |

---

## License

MIT License

---

## Contact

- **GitHub Issues**: [Open Issue](https://github.com/tunatuna1002-lab/AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-/issues)
- **Documentation**: [Architecture Docs](./docs/architecture/)
