# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-31
**Commit:** c40efb5
**Branch:** main

## OVERVIEW

AMOREPACIFIC RAG-KG Hybrid Agent for monitoring LANEIGE brand competitiveness on Amazon US. Autonomous AI system with daily crawling, KPI analysis (SoS/HHI/CPI), hybrid RAG+KG+Ontology chatbot, and ReAct self-reflection for complex queries.

## STRUCTURE

```
./
├── dashboard_api.py      # FastAPI entry + static UI mount
├── orchestrator.py       # Compat shim → src/core/batch_workflow
├── main.py               # CLI entry for batch + interactive chat
├── src/
│   ├── core/             # Brain, ReAct, scheduler, orchestration
│   ├── agents/           # AI agents (chatbot, insight, crawler, alert)
│   ├── rag/              # RAG + hybrid retrieval + entity linking
│   ├── ontology/         # KG triple store + OWL reasoner
│   ├── tools/            # Scrapers, collectors, utilities
│   ├── domain/           # Clean Architecture Layer 1 (entities, interfaces)
│   ├── application/      # Clean Architecture Layer 2 (workflows)
│   ├── api/              # FastAPI routes (v3, v4)
│   ├── infrastructure/   # Persistence, config
│   ├── memory/           # Conversation memory
│   ├── monitoring/       # Logging, tracing, metrics
│   └── adapters/         # Interface adapters
├── dashboard/            # Static HTML UI
├── tests/                # pytest (60% min coverage)
├── scripts/              # Operational utilities
├── config/               # JSON-driven rules/thresholds
├── docs/                 # Architecture docs, guides
└── data/                 # SQLite, ChromaDB, KG backups
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add API endpoint | `src/api/routes/` | Version in path (v3, v4) |
| Modify chatbot | `src/agents/hybrid_chatbot_agent.py` | Uses HybridRetriever |
| Modify insight generation | `src/agents/hybrid_insight_agent.py` | Daily + period reports |
| Add scraper/collector | `src/tools/` | Follow async pattern |
| Change KG logic | `src/ontology/knowledge_graph.py` | Triple store ops |
| Modify ontology rules | `src/ontology/reasoner.py` | Rule-based inference |
| Add domain entity | `src/domain/entities/` | Pydantic models |
| Add workflow | `src/application/workflows/` | Clean Architecture |
| Modify brain behavior | `src/core/brain.py` | UnifiedBrain facade |
| Add ReAct tool | `src/core/react_agent.py` | Register in ALLOWED_ACTIONS |
| Configure thresholds | `config/thresholds.json` | Category/alert rules |

## KEY MODULES

| Module | File | Role |
|--------|------|------|
| UnifiedBrain | `src/core/brain.py` | Facade: scheduler + query + ReAct |
| ReActAgent | `src/core/react_agent.py` | Thought-Action-Observation loop (max 3) |
| HybridRetriever | `src/rag/hybrid_retriever.py` | RAG + KG + Ontology context |
| KnowledgeGraph | `src/ontology/knowledge_graph.py` | Triple store + persistence |
| HybridChatbotAgent | `src/agents/hybrid_chatbot_agent.py` | AI chatbot |
| AmazonScraper | `src/tools/amazon_scraper.py` | Playwright + stealth |
| BatchWorkflow | `src/application/workflows/batch_workflow.py` | Daily crawl pipeline |

## CONVENTIONS

### Deviations from Standard
- Multiple top-level entrypoints outside `src/` (dashboard_api.py, main.py)
- HTML dashboard served by FastAPI (not separate frontend build)
- Duplicate "numbered" files exist (e.g., `file 2.py`) - likely backups, ignore

### Code Style
- **Line length**: 100 (Black + Ruff)
- **Async-first**: All I/O operations use `async/await`
- **Type hints**: Required on all functions
- **Pydantic**: Domain entities use Pydantic models
- **DI pattern**: Inject via Protocol interfaces, not concrete classes

### Clean Architecture Import Rules
```
domain → (nothing)           ✅
application → domain         ✅
infrastructure → domain      ✅
domain → application         ❌
infrastructure → adapters    ❌
```

## ANTI-PATTERNS

- **NEVER** suppress type errors (`as any`, `@ts-ignore`, `@ts-expect-error`)
- **NEVER** commit without explicit request
- **NEVER** use sync I/O in async context (wrap with `ThreadPoolExecutor`)
- **NEVER** hardcode API keys (use env vars)
- **NEVER** import infrastructure in domain layer

## COMMANDS

```bash
# Dev server
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001 --reload

# Tests (60% min coverage)
python -m pytest tests/ -v

# Golden set evaluation
python scripts/evaluate_golden.py --verbose

# KG backup
python -m src.tools.kg_backup backup

# Sync data from Railway
python scripts/sync_from_railway.py
```

## ENVIRONMENT

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional - Server
API_KEY=...                        # API auth
AUTO_START_SCHEDULER=true          # Auto-start scheduler

# Optional - External
GOOGLE_SPREADSHEET_ID=...          # Sheets backup
TAVILY_API_KEY=tvly-...            # News (1k/mo free)
DATA_GO_KR_API_KEY=...             # Public data APIs

# Optional - Alerts
SMTP_SERVER=smtp.gmail.com
SENDER_EMAIL=...
ALERT_RECIPIENTS=...
```

## NOTES

- **Lip Care vs Lip Makeup**: Different categories (Skin Care vs Makeup hierarchy)
- **KG Backup**: Auto 7-day rolling in `data/backups/kg/`
- **ReAct activation**: Auto-triggered for complex queries (analysis keywords, context gaps)
- **Embedding cache**: MD5-keyed FIFO (max 1000) reduces API costs 33%+
- **AWS WAF**: Stealth context + exponential backoff in scraper

---

*See subdirectory AGENTS.md files for module-specific details.*
