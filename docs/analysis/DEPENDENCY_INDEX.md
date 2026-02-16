# AMORE RAG-KG Hybrid Agent - Import Dependency Analysis
## Executive Summary & Navigation Guide

**Generated:** 2026-02-10
**Codebase:** 286 Python files, ~83,000 LOC across 13 top-level modules
**Architecture:** Clean Architecture (Layers 1-4) with circular dependencies in Layer 4

---

## Quick Navigation

### Reports Generated
This analysis includes 4 comprehensive reports. Choose based on your need:

1. **[DEPENDENCY_SUMMARY.txt](DEPENDENCY_SUMMARY.txt)** ← START HERE
   - Executive overview (10 sections)
   - Clean architecture assessment
   - Circular dependency detection (23 cycles)
   - Refactoring priorities & action items
   - **Best for:** Understanding the big picture, planning refactoring

2. **[DEPENDENCY_ANALYSIS.txt](DEPENDENCY_ANALYSIS.txt)**
   - Directory-by-directory breakdown
   - File counts, line counts, internal/external imports per directory
   - Module-level dependency graph
   - External package summary
   - **Best for:** Quick reference by directory, audit purposes

3. **[DEPENDENCY_GRAPH.txt](DEPENDENCY_GRAPH.txt)**
   - Module-level dependency graph
   - Layer analysis (Clean Architecture compliance)
   - Circular dependency detection with cycles listed
   - Module importance ranking
   - **Best for:** Understanding module relationships and architecture violations

4. **[FILE_IMPORT_MAP.txt](FILE_IMPORT_MAP.txt)**
   - Complete file-by-file import listing
   - All internal and external imports per file
   - Organized by directory
   - **Best for:** Deep dive into specific files/modules

---

## Key Findings at a Glance

### Project Stats
| Metric | Value |
|--------|-------|
| **Total Files** | 286 Python files |
| **Total LOC** | ~83,000 lines |
| **Total Modules** | 13 top-level |
| **Internal Dependencies** | 34 module-level edges |
| **External Packages** | 100+ unique packages |
| **Test Coverage** | 10.11% (target: 60%) |
| **Circular Dependencies** | 23 cycles detected ⚠️ |

### Module Size Distribution
```
src/tools/              38 files    19,323 LOC  ████████████████████████ (23%)
src/core/               24 files     8,075 LOC  ██████████ (10%)
src/rag/                13 files     6,207 LOC  ███████ (7%)
src/ontology/           11 files     5,861 LOC  ███████ (7%)
src/agents/             10 files     5,684 LOC  ███████ (7%)
src/domain/             17 files     2,048 LOC  ██ (2%)
src/api/                13 files     2,603 LOC  ███ (3%)
src/infrastructure/      8 files       998 LOC  █ (1%)
src/memory/              4 files       636 LOC  █ (1%)
src/monitoring/          4 files       865 LOC  █ (1%)
src/shared/              3 files       387 LOC  (0%)
src/application/         5 files       120 LOC  (0%)
```

### Clean Architecture Assessment

#### Excellent (✓✓)
- **src/domain** - 0 internal dependencies, only std lib + pydantic
- **src/memory** - 0 internal dependencies
- **src/monitoring** - 0 internal dependencies, isolated
- **src/shared** - 1 dependency (monitoring only)

#### Good (✓)
- **src/ontology** - 1 dependency (domain) - correct layers
- **src/rag** - 2 dependencies (domain, ontology) - correct layers
- **src/application** - 1 dependency (domain) - but nearly empty

#### Problematic (⚠️)
- **src/core** - 6 internal dependencies, includes circular (agents, tools)
- **src/agents** - 8 internal dependencies, circular with core & tools
- **src/tools** - 5 internal dependencies, circular with agents & api
- **src/api** - 5 internal dependencies, circular with core & tools
- **src/infrastructure** - 5 internal dependencies, imports Layer 4

---

## Most Critical Issues

### 1. Circular Dependencies (23 cycles)
**Impact:** HIGH - Makes testing, refactoring, and maintenance difficult

Primary cycles:
- `src.core ↔ src.agents` (brain imports agents; agents import brain)
- `src.agents ↔ src.tools` (agents use tools; tools reference agents)
- `src.api ↔ src.core/agents/tools` (routes import logic; logic imports routes)

**Recommendation:** Priority 1 refactoring - break cycles with interfaces

### 2. Underdeveloped Application Layer
**Impact:** MEDIUM - Business logic scattered across Layer 4

Current state:
- `src/application/` has only 120 LOC across 5 files
- Real use cases are in `src/core/` (brain.py, etc.)
- API routes directly call core logic

**Recommendation:** Priority 2 refactoring - move workflows to application

### 3. Heavy src/tools Module
**Impact:** MEDIUM - 38 files, 19K LOC with complex dependencies

Issues:
- 61 external packages (highest in codebase)
- Circular imports with agents and api
- Contains mixtures of scrapers, formatters, utilities

**Recommendation:** Priority 3 refactoring - break into sub-packages

### 4. Low Test Coverage
**Impact:** HIGH - Only 10.11% (target: 60%)

Reasons:
- Circular dependencies block isolated testing
- Heavy external I/O (browser, APIs, databases)
- Layer 4 tight coupling

**Recommendation:** Fix circular deps first, then unit test domain/application

---

## Module Dependency Summary

### Most Depended Upon (Critical to Change)
1. **src.domain** ← imported by 8 modules
   - Reason: Core entity definitions
   - Risk: **ANY CHANGE BREAKS EVERYTHING**
   - Recommendation: Never break compatibility

2. **src.ontology** ← imported by 5 modules
   - Reason: KG, business rules, reasoning
   - Risk: Medium
   - Recommendation: Version carefully

3. **src.agents** ← imported by 5 modules
   - Reason: Autonomous agents
   - Risk: **CIRCULAR (core imports agents)**
   - Recommendation: Extract to interface (src/domain/interfaces/agent)

4. **src.tools** ← imported by 3 modules
   - Reason: Utilities, scrapers
   - Risk: **CIRCULAR (core, api import tools)**
   - Recommendation: Extract protocols to src/domain/interfaces/

### Safest to Change (Isolated)
1. **src.memory** - No internal dependencies
2. **src.monitoring** - No internal dependencies
3. **src.shared** - Only depends on monitoring

---

## External Package Dependencies

### Top 20 by Usage
1. litellm (LLM orchestration)
2. openai (OpenAI API)
3. pydantic (validation)
4. aiohttp (async HTTP)
5. chromadb (vector DB)
6. playwright (browser automation)
7. pandas (data processing)
8. sqlalchemy (ORM)
9. gspread (Google Sheets)
10. owlready2 (OWL reasoning)

### Version Pinning Recommendations
- **chromadb** - Pin version for reproducibility (vector DB changes affect retrieval)
- **playwright** - Pin version for scraper stability
- **owlready2** - Monitor for OWL/Ontology standard updates
- **litellm** - Keep reasonably current (LLM model additions)

---

## Entry Points Analysis

### dashboard_api.py (5,626 LOC) - PRIMARY
- **Framework:** FastAPI
- **Routes:** Health, Chat, Crawl, Brain Status, WebSocket
- **Internal Imports:** 29 modules (all layers)
- **Status:** Depends on nearly everything

### main.py (220 LOC) - CHATBOT CLI
- **Purpose:** Async chatbot interface
- **Internal Imports:** src.core.brain, src.monitoring.logger
- **Status:** Relatively clean

### orchestrator.py (47 LOC) - BATCH WORKFLOWS
- **Purpose:** Orchestrates batch jobs
- **Internal Imports:** src.core.batch_workflow
- **Status:** Minimal dependencies

### start.py (19 LOC) - SERVER LAUNCHER
- **Purpose:** Uvicorn server startup
- **Status:** Trivial

---

## Refactoring Roadmap

### Phase 1: Break Circular Dependencies (Priority 1)
**Effort:** Medium | **Impact:** HIGH | **Timeline:** 2-4 weeks

1. Create `src/domain/interfaces/` with protocol classes
   - `AgentProtocol` (for brain to depend on instead of concrete agents)
   - `ToolProtocol` (for core/api to depend on instead of concrete tools)

2. Update core modules to use interfaces
   - brain.py imports AgentProtocol (not concrete agents)
   - Agents implement protocol, depend on interfaces

3. Use dependency injection at entry points

### Phase 2: Move Business Logic to Application (Priority 2)
**Effort:** High | **Impact:** MEDIUM | **Timeline:** 1-2 months

1. Create use cases in `src/application/workflows/`
   - CrawlUseCase
   - ChatUseCase
   - InsightGenerationUseCase

2. Move orchestration from brain.py to application

3. API routes call application use cases

### Phase 3: Wrap External Packages (Priority 3)
**Effort:** Low | **Impact:** MEDIUM | **Timeline:** 2-3 weeks

1. Create adapters in `src/infrastructure/`
   - ChromaDBAdapter
   - PlaywrightAdapter
   - OpenAIAdapter (or extend shared.llm_client)

2. Use adapters instead of direct package imports

### Phase 4: Establish DI Container (Priority 4)
**Effort:** Medium | **Impact:** LOW | **Timeline:** 1-2 weeks

1. Use `dependency_injector` or similar
2. Define containers per layer
3. Inject at entry points

---

## Testing Strategy

### Current State
- **Coverage:** 10.11% (far below 60% target)
- **Main Blockers:** Circular deps, heavy I/O

### Path to 60% Coverage
1. **Fix circular dependencies** (enables isolated unit tests)
2. **Test domain layer first** (should be 40%+ of final coverage)
   - Fast, no mocks needed
3. **Test application layer** (should be 20%+ of coverage)
   - Fast, domain-only mocks
4. **Mock external I/O**
   - Playwright → mock browser
   - APIs → mock responses
   - ChromaDB → in-memory

### Recommended Test Structure
```
tests/unit/
├── domain/           → Fast (no setup)
├── application/      → Fast (minimal mocks)
├── ontology/         → Medium (KG setup)
├── rag/             → Medium (chromadb mocks)
├── memory/          → Fast
├── monitoring/      → Fast
├── core/            → SLOW (needs many mocks)
├── agents/          → SLOW (I/O heavy)
├── tools/           → VERY SLOW (all external)
└── api/             → SLOW (fixture heavy)
```

---

## File Reference Guide

### By Use Case

**I want to understand module relationships:**
→ Read: DEPENDENCY_GRAPH.txt (Module-level analysis)

**I want to refactor a specific module:**
→ Read: FILE_IMPORT_MAP.txt + DEPENDENCY_ANALYSIS.txt (find all dependencies)

**I want to increase test coverage:**
→ Read: DEPENDENCY_SUMMARY.txt section 8 (Testing implications)

**I want a quick audit of src/tools/:**
→ Read: DEPENDENCY_ANALYSIS.txt (directory summary)

**I want to understand circular dependencies:**
→ Read: DEPENDENCY_GRAPH.txt (cycle detection section)

**I want to plan refactoring priorities:**
→ Read: DEPENDENCY_SUMMARY.txt sections 6-10 (recommendations)

---

## Key Statistics

### Dependency Density
- **Average dependencies per module:** 2.6
- **Most coupled:** src.agents (8 internal deps)
- **Most isolated:** src.memory, src.monitoring (0 internal deps)

### Code Distribution
- **Layer 1 (Domain):** 2.5% of codebase (excellent isolation)
- **Layer 2 (Application):** 0.1% of codebase (underdeveloped)
- **Layer 3 (Adapters):** 20% of codebase (good)
- **Layer 4 (Frameworks):** 45% of codebase (tightly coupled)

### External Dependencies
- **Total unique external packages:** 100+
- **Most packages used by:** src.tools (61 packages)
- **Cleanest module:** src.domain (5 packages only)

---

## Quick Reference: Most Important Files

### Always Safe to Change
```
src/domain/                    (no internal dependencies)
src/memory/                    (no internal dependencies)
src/monitoring/                (no internal dependencies)
```

### Carefully (1-2 dependencies each)
```
src/ontology/                  (depends on domain)
src/rag/                       (depends on domain, ontology)
src/shared/                    (depends on monitoring)
```

### High Risk (circular dependencies)
```
src/core/                      (circular with agents, tools)
src/agents/                    (circular with core, tools)
src/tools/                     (circular with agents, api)
src/api/                       (circular with core, agents, tools)
```

### Entry Points (touch sparingly)
```
dashboard_api.py               (5,626 LOC, imports everything)
main.py                        (220 LOC, imports brain)
orchestrator.py                (47 LOC, imports batch_workflow)
```

---

## Next Steps

### For Architects
1. Review DEPENDENCY_SUMMARY.txt section 2 (Clean Architecture assessment)
2. Review DEPENDENCY_GRAPH.txt (layer compliance)
3. Plan Phase 1 refactoring (break circular deps)

### For Implementation Teams
1. Read DEPENDENCY_SUMMARY.txt section 7 (Refactoring recommendations)
2. Review FILE_IMPORT_MAP.txt (understand what needs changing)
3. Create src/domain/interfaces/ for protocols

### For QA/Testing
1. Read DEPENDENCY_SUMMARY.txt section 8 (Testing implications)
2. Review DEPENDENCY_ANALYSIS.txt (find what to mock)
3. Plan test structure per module

### For DevOps/Release
1. Review external packages (DEPENDENCY_SUMMARY.txt section 4)
2. Monitor chromadb, playwright versions
3. Keep litellm reasonably current

---

## Document Legend

| Symbol | Meaning |
|--------|---------|
| ✓ | Clean, isolated, no issues |
| ✓✓ | Excellent, model for others |
| ⚠️ | Attention needed |
| ⚠️⚠️ | Critical issue |
| → | Imports/depends on |
| ↔ | Circular dependency |

---

## Questions?

If you need clarification on:
- **Module relationships** → See DEPENDENCY_GRAPH.txt
- **Specific file imports** → See FILE_IMPORT_MAP.txt
- **Directory overview** → See DEPENDENCY_ANALYSIS.txt
- **Refactoring plan** → See DEPENDENCY_SUMMARY.txt

All reports are in the same directory as this index.

---

**Last Updated:** 2026-02-10
**Analysis Scope:** All Python files except __pycache__, /tests/, /.omc/
