# Import Dependency Analysis Report

> **Generated:** 2026-02-10
> **Codebase:** 286 Python files, ~83,000 LOC across 13 top-level modules
> **Architecture:** Clean Architecture (Layers 1-4)

---

## 1. Project Overview

| Directory | Files | Lines | Primary Role |
|-----------|-------|-------|--------------|
| src/tools/ | 38 | 19,323 | Scrapers, Collectors, Formatters |
| src/core/ | 24 | 8,075 | Brain, Scheduler, ReAct, State Mgmt |
| src/rag/ | 13 | 6,207 | RAG Pipeline, Retrievers, Rerankers |
| src/ontology/ | 11 | 5,861 | KG, Business Rules, Reasoners |
| src/agents/ | 10 | 5,684 | Crawlers, Chatbots, Insight Agents |
| src/api/ | 13 | 2,603 | Routes, Endpoints, Validators |
| src/domain/ | 17 | 2,048 | Entities, Exceptions, Models |
| src/infrastructure/ | 8 | 998 | Container, Bootstrap, Repositories |
| src/monitoring/ | 4 | 865 | Logger, Metrics, Tracer |
| src/memory/ | 4 | 636 | Session, Context, History |
| src/shared/ | 3 | 387 | LLM Client, Constants |
| src/application/ | 5 | 120 | Use Cases (Minimal) |

**Total:** 286 files, ~83,000 LOC, 13 modules, 34 internal dependency edges, 100+ external packages

---

## 2. Module Dependency Graph

```
Entry Points: dashboard_api.py, main.py
         │
         ▼
    ┌─── api/ ◄──────────────────────────┐
    │     │                               │
    │     ▼                               │
    │   core/ ──────► agents/ ──────► rag/
    │     │             │    ◄────────  │
    │     │             │               │
    │     ▼             ▼               ▼
    │   memory/      ontology/ ◄──── tools/
    │     │             │               │
    │     ▼             ▼               │
    │   monitoring/  domain/ ◄──────────┘
    │                   ▲
    │                   │
    └── infrastructure/ ┘
         shared/ (독립)
```

### Dependency Matrix (From → To)

| From \ To | core | agents | rag | ontology | tools | domain | api | memory | monitoring |
|-----------|:----:|:------:|:---:|:--------:|:-----:|:------:|:---:|:------:|:----------:|
| **core** | - | 8 | - | 3 | 8 | 1 | - | 3 | 3 |
| **agents** | - | - | 10 | 4 | 7 | 7 | - | 1 | 3 |
| **rag** | - | - | - | 4 | - | 3 | - | - | 1 |
| **ontology** | - | - | - | - | - | 9 | - | - | - |
| **tools** | - | 1 | - | 1 | 7 | 1 | 2 | - | - |
| **api** | 4 | 2 | 2 | - | 9 | 1 | - | - | - |
| **infrastructure** | 2 | 3 | 3 | 4 | - | 4 | - | - | - |

---

## 3. Clean Architecture Layer Assessment

### Layer 1: Domain — EXCELLENT
- Files: 17, Lines: 2,048
- Dependencies: None (only Pydantic, dataclasses, enum)
- Status: Perfectly isolated, zero internal imports

### Layer 2: Application — Underdeveloped
- Files: 5, Lines: 120
- Dependencies: src.domain (correct)
- Status: Nearly empty — business logic resides in Layer 4

### Layer 3: Adapters — CLEAN
- **src/rag/** (13 files, 6,207 LOC): depends on domain, ontology ✓
- **src/ontology/** (11 files, 5,861 LOC): depends on domain only ✓
- **src/memory/** (4 files, 636 LOC): zero internal dependencies ✓
- **src/monitoring/** (4 files, 865 LOC): zero internal dependencies ✓
- **src/shared/** (3 files, 387 LOC): depends on monitoring only ✓

### Layer 4: Frameworks — Tightly Coupled
- **src/core/** (24 files, 8,075 LOC): circular with agents ⚠️
- **src/agents/** (10 files, 5,684 LOC): 8 internal deps, circular ⚠️
- **src/tools/** (38 files, 19,323 LOC): circular with agents, api ⚠️
- **src/api/** (13 files, 2,603 LOC): circular with core/agents ⚠️
- **src/infrastructure/** (8 files, 998 LOC): depends on Layer 4 ⚠️

---

## 4. Circular Dependencies (23 cycles)

Primary cycles:
- `src.core ↔ src.agents` — brain imports agents; agents import brain
- `src.agents ↔ src.tools` — agents use tools; tools reference agents
- `src.api ↔ src.core/agents/tools` — routes import logic; logic imports routes

### Impact
- Difficult to unit test, refactoring risky, hidden coupling
- Blocks isolated testing of core/agent modules

### Resolution Strategy
1. Extract protocols to `src/domain/interfaces/`
2. Use dependency injection instead of direct imports
3. Move route handlers to separate service layer

---

## 5. External Package Dependencies (Top 20)

1. litellm (LLM Orchestration)
2. openai (OpenAI API)
3. pydantic (Data Validation)
4. aiohttp (Async HTTP)
5. chromadb (Vector DB)
6. playwright (Browser Automation)
7. pandas (Data Processing)
8. aiosqlite (Async SQLite)
9. gspread (Google Sheets)
10. owlready2 (OWL Reasoning)
11. google-api-python-client (Google API)
12. yt-dlp (YouTube Download)
13. instaloader (Instagram Scraper)
14. pytrends (Google Trends)
15. tavily-python (News Search)
16. browserforge (Browser Emulation)
17. python-docx (Word Documents)
18. requests (HTTP Client)
19. sentence-transformers (Embeddings)
20. matplotlib (Charts)

### Version Pinning Recommendations
- **chromadb**: Pin for reproducibility (vector DB changes affect retrieval)
- **playwright**: Pin for scraper stability
- **owlready2**: Monitor for OWL standard updates
- **litellm**: Keep current (LLM model additions)

---

## 6. Entry Points

| File | Lines | Role | Dependencies |
|------|-------|------|-------------|
| dashboard_api.py | 3,236 | FastAPI server (38 endpoints) | 29 internal modules |
| main.py | 282 | CLI batch/chatbot | core, monitoring |
| orchestrator.py | 47 | Backward compat wrapper | batch_workflow only |
| start.py | 29 | Railway deploy | uvicorn only |

---

## 7. Module Health Summary

| Module | Files | LOC | Ext Deps | Int Deps | Health | Priority |
|--------|-------|-----|----------|----------|--------|----------|
| src.domain | 17 | 2,048 | 5 | 0 | ✓✓ | STABLE |
| src.memory | 4 | 636 | 9 | 0 | ✓✓ | STABLE |
| src.monitoring | 4 | 865 | 15 | 0 | ✓✓ | STABLE |
| src.shared | 3 | 387 | 5 | 1 | ✓✓ | STABLE |
| src.ontology | 11 | 5,861 | 13 | 1 | ✓ | MAINTAIN |
| src.rag | 13 | 6,207 | 23 | 2 | ✓ | MAINTAIN |
| src.application | 5 | 120 | 4 | 1 | ⚠️ | DEVELOP |
| src.infrastructure | 8 | 998 | 11 | 5 | ⚠️ | REFACTOR |
| src.core | 24 | 8,075 | 17 | 6 | ⚠️ | REFACTOR |
| src.agents | 10 | 5,684 | 14 | 8 | ⚠️⚠️ | REFACTOR |
| src.api | 13 | 2,603 | 17 | 5 | ⚠️ | REFACTOR |
| src.tools | 38 | 19,323 | 61 | 5 | ⚠️⚠️ | REFACTOR |

---

## 8. File-Level Import Map (Key Files)

### dashboard_api.py — 29 internal imports
src.agents.alert_agent, src.api.routes.*, src.core.brain, src.core.crawl_manager, src.core.state_manager, src.ontology.knowledge_graph, src.rag.retriever, src.rag.router, src.tools.* (11 modules)

### src/core/batch_workflow.py — 21 internal imports
src.agents.* (5), src.core.* (2), src.memory.* (3), src.monitoring.* (3), src.ontology.* (3), src.tools.* (5)

### src/agents/hybrid_chatbot_agent.py — 17 internal imports
src.core.verification_pipeline, src.domain.*, src.memory.*, src.monitoring.*, src.ontology.* (3), src.rag.* (5), src.shared.*, src.tools.*

### src/agents/hybrid_insight_agent.py — 18 internal imports
src.domain.*, src.monitoring.* (3), src.ontology.* (3), src.rag.* (4), src.shared.*, src.tools.* (5)

---

## 9. Refactoring Recommendations

### P0: Dead Code Deletion (3 files, 674 lines)
- `src/core/explainability.py` — import 0건, brain.py에서 대체 구현
- `src/core/query_processor.py` — import 0건, brain.py + response_pipeline.py 대체
- `src/domain/interfaces/brain_components.py` — import 0건, 구현체 미사용

### P1: Break Circular Dependencies
- core ↔ agents: Interface injection via src/domain/interfaces/agent
- tools ↔ agents: Protocol-based DI
- api ↔ core: Service layer extraction

### P2: Strengthen Application Layer
- Move business logic from brain.py → application/workflows/
- Create CrawlUseCase, ChatUseCase, InsightGenerationUseCase

### P3: Wrap External Packages
- Create adapters: ChromaDBAdapter, PlaywrightAdapter, OpenAIAdapter

---

*Consolidated from: DEPENDENCY_ANALYSIS.txt, DEPENDENCY_GRAPH.txt, DEPENDENCY_INDEX.md, DEPENDENCY_SUMMARY.txt, FILE_IMPORT_MAP.txt*
*Last Updated: 2026-02-16*
