# AMORE Pacific RAG-KG Hybrid Agent

> An intelligent AI agent system that combines **RAG (Retrieval-Augmented Generation)** and **Knowledge Graph** technologies for Amazon marketplace analytics and insights.

[한국어 README](./README_KO.md)

---

## Overview

This project was developed to provide **AMORE Pacific's LANEIGE brand** with real-time competitive intelligence on Amazon US marketplace. The system crawls Amazon Best Seller rankings daily, calculates strategic KPIs, and generates actionable insights through a hybrid AI approach.

### Why Hybrid RAG + Knowledge Graph?

Traditional RAG systems retrieve relevant documents but lack **structured reasoning capabilities**. Pure Knowledge Graph approaches have strong reasoning but limited **natural language understanding**. Our hybrid approach combines both:

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Query                                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │    Query Router       │
          │  (Intent Classification)│
          └───────────┬───────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼───────┐         ┌────────▼────────┐
│  RAG Pipeline │         │ Knowledge Graph │
│               │         │                 │
│ • Document    │         │ • Entity        │
│   Retrieval   │         │   Relations     │
│ • Semantic    │         │ • Ontology      │
│   Search      │         │   Reasoning     │
│ • Context     │         │ • Business      │
│   Building    │         │   Rules         │
└───────┬───────┘         └────────┬────────┘
        │                           │
        └─────────────┬─────────────┘
                      │
          ┌───────────▼───────────┐
          │   LLM Orchestrator    │
          │  (Response Generation)│
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │   Structured Response │
          │  + Actionable Insights│
          └───────────────────────┘
```

---

## Key Features

### 1. Multi-Agent Architecture
- **Crawler Agent**: Scrapes Amazon Best Seller rankings (Top 100 per category)
- **Storage Agent**: Persists data to Google Sheets with versioning
- **Metrics Agent**: Calculates 10+ strategic KPIs (SoS, HHI, CPI, etc.)
- **Insight Agent**: Generates AI-powered daily insights
- **Chatbot Agent**: Interactive Q&A with conversation memory
- **Alert Agent**: Monitors thresholds and sends notifications

### 2. RAG System
- **Document Retriever**: Semantic search over business documentation
- **Query Router**: Intent classification (Definition, Interpretation, Analysis, etc.)
- **Context Builder**: Dynamic context assembly based on query type
- **Hybrid Retriever**: Combines keyword and semantic search

### 3. Knowledge Graph & Ontology
- **Entity Types**: Brand, Product, Category, Competitor
- **Metrics Ontology**: ProductMetrics, BrandMetrics, MarketMetrics
- **Business Rules Engine**: Configurable threshold-based rules
- **Reasoner**: Inference engine for deriving insights

### 4. Dashboard & API
- **FastAPI Backend**: RESTful API with multiple versions (v1, v2, v3)
- **Interactive Dashboard**: Real-time visualization of KPIs
- **DOCX Export**: Generate professional insight reports
- **Audit Trail**: Complete logging of all chatbot interactions

---

## Project Structure

```
AMORE-RAG-ONTOLOGY-HYBRID AGENT/
├── src/                          # Source code
│   ├── agents/                   # AI Agents
│   │   ├── crawler_agent.py      # Amazon scraping
│   │   ├── storage_agent.py      # Google Sheets integration
│   │   ├── metrics_agent.py      # KPI calculations
│   │   ├── insight_agent.py      # AI insight generation
│   │   ├── chatbot_agent.py      # Conversational AI
│   │   ├── alert_agent.py        # Threshold monitoring
│   │   ├── hybrid_insight_agent.py   # RAG+KG hybrid insights
│   │   └── hybrid_chatbot_agent.py   # RAG+KG hybrid chat
│   │
│   ├── core/                     # Core orchestration
│   │   ├── unified_orchestrator.py   # Main orchestrator
│   │   ├── llm_orchestrator.py   # LLM coordination
│   │   ├── brain.py              # Decision making
│   │   ├── rules_engine.py       # Business rules
│   │   ├── simple_chat.py        # Simplified chat service
│   │   └── crawl_manager.py      # Background crawl management
│   │
│   ├── rag/                      # RAG components
│   │   ├── retriever.py          # Document retrieval
│   │   ├── router.py             # Query classification
│   │   ├── context_builder.py    # Context assembly
│   │   ├── hybrid_retriever.py   # Hybrid search
│   │   └── templates.py          # Prompt templates
│   │
│   ├── ontology/                 # Knowledge Graph
│   │   ├── schema.py             # Entity definitions
│   │   ├── knowledge_graph.py    # Graph operations
│   │   ├── reasoner.py           # Inference engine
│   │   ├── business_rules.py     # Rule definitions
│   │   └── relations.py          # Relationship types
│   │
│   ├── memory/                   # Session & context
│   │   ├── session.py            # Session management
│   │   ├── history.py            # Execution history
│   │   └── context.py            # Context tracking
│   │
│   ├── monitoring/               # Observability
│   │   ├── logger.py             # Structured logging
│   │   ├── tracer.py             # Execution tracing
│   │   └── metrics.py            # Quality metrics
│   │
│   └── tools/                    # External integrations
│       ├── amazon_scraper.py     # Amazon API wrapper
│       ├── sheets_writer.py      # Google Sheets API
│       ├── dashboard_exporter.py # JSON export
│       ├── metric_calculator.py  # KPI formulas
│       └── email_sender.py       # Alert notifications
│
├── dashboard/                    # Frontend
│   ├── amore_unified_dashboard_v4.html
│   └── test_chat.html
│
├── docs/                         # Documentation
│   ├── architecture/             # System design
│   │   ├── LLM_ORCHESTRATOR_DESIGN.md
│   │   └── *.xml (diagrams)
│   └── guides/                   # Business guides
│       ├── Strategic Indicators Definition.md
│       ├── Metric Interpretation Guide.md
│       └── Indicator Combination Playbook.md
│
├── config/                       # Configuration
│   └── thresholds.json           # Alert thresholds
│
├── data/                         # Runtime data
│   ├── dashboard_data.json       # Dashboard state
│   └── knowledge_graph.json      # Persisted KG
│
├── main.py                       # CLI entry point
├── dashboard_api.py              # FastAPI server
├── orchestrator.py               # Workflow orchestrator
└── export_dashboard.py           # Data export script
```

---

## Development Journey

### Phase 1: Foundation (Agents & Crawling)
Started with a simple agent architecture to crawl Amazon Best Seller data. Each agent was designed with single responsibility:
- Crawler fetches raw HTML
- Parser extracts structured data
- Storage persists to Google Sheets

### Phase 2: Analytics (Metrics & KPIs)
Added strategic KPI calculations based on e-commerce analytics best practices:
- **SoS (Share of Shelf)**: Brand visibility in Top 100
- **HHI (Herfindahl-Hirschman Index)**: Market concentration
- **CPI (Competitive Position Index)**: Relative positioning
- **Volatility**: Rank stability over time

### Phase 3: AI Integration (LLM + RAG)
Integrated OpenAI GPT models for natural language insights:
- Daily insight generation from metrics
- Chatbot for interactive Q&A
- RAG for grounding responses in business documentation

### Phase 4: Knowledge Graph (Ontology)
Added structured knowledge representation:
- Entity-relationship modeling
- Business rule inference
- Hybrid reasoning combining RAG + KG

### Phase 5: Production Hardening
- Multi-version API (v1 legacy, v2 orchestrator, v3 simplified)
- Background crawl management
- Audit trail logging
- DOCX report generation

---

## Installation

### Prerequisites
- Python 3.10+
- Google Cloud credentials (for Sheets API)
- OpenAI API key

### Setup

```bash
# Clone repository
git clone https://github.com/tunatuna1002-lab/AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-.git
cd AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```env
OPENAI_API_KEY=...
GOOGLE_SHEETS_SPREADSHEET_ID=...
GOOGLE_APPLICATION_CREDENTIALS=./....json
```

---

## Usage

### Daily Workflow (Batch Processing)
```bash
# Run full daily workflow
python main.py

# Run specific categories only
python main.py --categories lip_care face_moisturizer

# Dry run (no Google Sheets write)
python main.py --dry-run
```

### Interactive Chatbot
```bash
python main.py --chat
```

### Dashboard API Server
```bash
# Start FastAPI server
python dashboard_api.py

# Or with uvicorn
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001 --reload
```

### Export Dashboard Data
```bash
python export_dashboard.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/data` | GET | Get dashboard data |
| `/api/chat` | POST | Chat (v1 - RAG only) |
| `/api/v2/chat` | POST | Chat (v2 - Unified Orchestrator) |
| `/api/v3/chat` | POST | Chat (v3 - Simplified) |
| `/api/v2/stats` | GET | Orchestrator statistics |
| `/api/crawl/status` | GET | Crawl status |
| `/api/crawl/start` | POST | Trigger manual crawl |
| `/api/export/docx` | POST | Generate DOCX report |

---

## Strategic KPIs

| KPI | Description | Formula |
|-----|-------------|---------|
| **SoS** | Share of Shelf | Brand products in Top 100 / 100 |
| **HHI** | Market Concentration | Σ(market share²) |
| **CPI** | Competitive Position | Weighted rank score |
| **Volatility** | Rank Stability | Std dev of rank changes |
| **Top10 Count** | Premium Visibility | Products in Top 10 |
| **Avg Rank** | Mean Position | Average rank of brand products |

---

## Tech Stack

- **Backend**: Python, FastAPI, asyncio
- **LLM**: OpenAI GPT-4.1-mini (via LiteLLM)
- **RAG**: Custom retriever with semantic search
- **Knowledge Graph**: Custom implementation with JSON persistence
- **Storage**: Google Sheets API
- **Scraping**: BeautifulSoup, httpx
- **Frontend**: HTML/CSS/JS (vanilla)

---

## Architecture Principles

1. **Agent-Based Design**: Each agent has single responsibility
2. **Think-Act-Observe Loop**: Inspired by ReAct pattern
3. **Hybrid Intelligence**: Combining structured (KG) and unstructured (RAG) knowledge
4. **Graceful Degradation**: Fallback strategies for each component
5. **Observability**: Comprehensive logging and tracing

---

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

---

## License

This project is proprietary software developed for AMORE Pacific.

---

