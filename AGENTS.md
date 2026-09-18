# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-31
**Commit:** c40efb5
**Branch:** main

## OVERVIEW

AMOREPACIFIC RAG-KG Hybrid Agent for monitoring LANEIGE brand competitiveness on Amazon US. Daily crawling, KPI analysis (SoS/HHI/CPI), and a hybrid RAG+KG+rule-reasoning chatbot with one LLM tool-selection step (DecisionMaker, native function calling). ReAct loop is wired behind a feature flag that defaults to OFF (`agents.use_react_agent`); it shares the same 5-tool registry (`src/core/tool_registry.py`) as DecisionMaker. [post-2026-09] The OWL retrieval strategy was deleted (dead code, 0 callers) — OWL now serves only as category-hierarchy vocabulary, not a retrieval path.

## STRUCTURE

```
./
├── src/api/dashboard_api.py  # FastAPI entry + static UI mount
├── src/core/orchestrator.py  # Compat shim → src/application/workflows/batch_workflow
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
├── tests/                # pytest (60% coverage goal, not enforced: fail_under = 0)
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
| Add a tool (DecisionMaker + ReAct) | `src/core/tool_registry.py` | Add a `ToolDefinition`; both `DecisionMaker` (function calling) and `ReActToolExecutor`/`ALLOWED_ACTIONS` read from this one registry |
| Configure thresholds | `config/thresholds.json` | Category/alert rules |

## KEY MODULES

| Module | File | Role |
|--------|------|------|
| UnifiedBrain | `src/core/brain.py` | Facade: scheduler + query + ReAct |
| ReActAgent | `src/core/react_agent.py` | Thought-Action-Observation loop (max 5), flag `agents.use_react_agent` (default OFF) |
| HybridRetriever | `src/rag/hybrid_retriever.py` | RAG + KG + Ontology context |
| KnowledgeGraph | `src/ontology/knowledge_graph.py` | Triple store + persistence |
| HybridChatbotAgent | `src/agents/hybrid_chatbot_agent.py` | AI chatbot |
| AmazonScraper | `src/tools/scrapers/amazon_scraper.py` | Playwright + stealth |
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
uvicorn src.api.dashboard_api:app --host 0.0.0.0 --port 8001 --reload

# Tests
python -m pytest tests/ -v

# Golden set evaluation
python scripts/evaluate_golden.py --verbose

# KG backup
python -m src.tools.utilities.kg_backup backup

# Sync data from Railway
python scripts/sync_from_railway.py
```

## ENVIRONMENT

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional - Server
API_KEY=...                        # API auth
AUTO_START_SCHEDULER=true          # Auto-start scheduler (default false when unset)

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
- **ReAct activation**: Only when `agents.use_react_agent` is on; then MEDIUM/LOW-confidence queries that the hop-count router (`src/core/router.py`) judges as 2+ hops route to it. HIGH confidence always answers via the pipeline unless the measurement flag `agents.react_bypass_confidence` (default OFF) is on. [post-2026-09-18] Stage-6 comparison kept all three ReAct flags OFF (decision S6-3 in `docs/plans/evidence-react-ontology-decisions-2026-09.md`). Check `/api/v4/brain/status` → `components`
- **Embedding cache**: MD5-keyed FIFO (max 1000). No measured cost reduction on record
- **AWS WAF**: Stealth context + exponential backoff in scraper

---

*See subdirectory AGENTS.md files for module-specific details.*
