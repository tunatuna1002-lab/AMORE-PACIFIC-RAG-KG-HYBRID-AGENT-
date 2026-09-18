# RAG-Ontology-KG Hybrid Agent Architecture

## 1. Overview

The AMORE Pacific LANEIGE brand monitoring system is an autonomous AI system designed for competitive intelligence on Amazon US. It combines three knowledge paradigms:

- **RAG (Retrieval-Augmented Generation)**: Document-based semantic search
- **Knowledge Graph**: Structured entity-relationship triples
- **OWL Ontology**: Formal logical reasoning with schema validation

### Key Capabilities

- Daily auto-crawling (Amazon Top 100 × 5 categories at 22:00 KST)
- KPI analysis (SoS, HHI, CPI)
- AI chatbot with hybrid retrieval
- Insight generation with external signal integration

---

## 2. System Architecture

### 2.1 Core Pipeline

```
User Query → UnifiedBrain → ContextGatherer → ConfidenceAssessor
                                                    ↓
                                            HIGH → Direct Response (skip LLM)
                                            MEDIUM → DecisionMaker → ResponsePipeline
                                            LOW → DecisionMaker (full) → Tool Execution → ResponsePipeline
                                            UNKNOWN → Clarification Request
```

The pipeline routes queries based on confidence scores, optimizing for both accuracy and latency.

### 2.2 Knowledge Systems

#### T-Box (Schema): OWL Ontology

- **File**: `src/ontology/owl_reasoner.py` + `cosmetics_ontology.owl`
- **Classes**: Brand, Product, Category, MarketSegment
- **Properties**: hasSoS, competesIn, belongsToCategory
- **Reasoner**: owlready2 + Pellet

The T-Box defines the conceptual schema: what types of entities exist and how they can relate.

#### A-Box (Instances): Knowledge Graph

- **File**: `src/ontology/knowledge_graph.py`
- **Triple store**: subject-predicate-object format
- **Content**: Entity metadata, relationships, market data

The A-Box contains concrete instances: specific brands, products, and their real-world relationships.

#### Bridge: OntologyKnowledgeGraph

- **File**: `src/ontology/ontology_knowledge_graph.py`
- **Validation**: Checks triples against OWL schema before adding to KG
- **Classification**: Auto-classifies entities (MarketLeader, Challenger, etc.)
- **Sync**: Propagates OWL inferences back to KG

This bridge ensures consistency between formal reasoning and practical data storage.

### 2.3 Reasoning

#### UnifiedReasoner — [post-2026-09] deleted

`src/ontology/unified_reasoner.py` had 0 callers (never wired into the live query path) and was deleted as dead code (track 4-C, `docs/experiments/evidence_pipeline_2026-09.md`). The rule engine actually used on the retrieval path is `src/ontology/reasoner.py` (`OntologyReasoner`); its inputs are now evidence cards via `src/ontology/rule_contracts.py`, not the OWL+rules cascade described below (that cascade describes the never-instantiated module).

<details>
<summary>Original text (kept for history, describes code that was never connected to a live path)</summary>

- **File**: `src/ontology/unified_reasoner.py`
- **Cascade**: OWL formal reasoning → Business rules → Result fusion
- **Output**: UnifiedInferenceResult with source attribution

The reasoner combines deductive logic (OWL) with inductive heuristics (business rules).

</details>

**Example reasoning chain:**

1. OWL: "LANEIGE is a Brand with hasSoS > 0.20 → MarketLeader"
2. Business rule: "SoS increased by 5%+ in 7 days → Rising"
3. Fusion: "LANEIGE is a Rising MarketLeader"

### 2.4 Retrieval

#### HybridRetriever

- **File**: `src/rag/hybrid_retriever.py` (this repo has no `true_hybrid_retriever.py` — that filename is not present; verify before citing)
- **Sources**: RAG (ChromaDB) + KG facts + rule inferences
- **Entity Linking**: Confidence-scored entity extraction
- **Context Generation**: Combined multi-source context

**Retrieval workflow:**

```
Query: "LANEIGE 경쟁력은?"
  ↓
Entity extraction: ["LANEIGE"]
  ↓
RAG search: Top 5 document chunks (cosine similarity)
  ↓
KG lookup: LANEIGE triples (brand, SoS, rank history)
  ↓
Rule inference: MarketLeader classification (OntologyReasoner, business rules — not OWL; OWL is category-hierarchy vocabulary only, [post-2026-09])
  ↓
Combined context: {documents: [...], facts: [...], inferences: [...]}
```

### 2.5 Confidence-Based Routing

- **File**: `src/core/confidence.py`
- **Tiers**: HIGH (5.0+) / MEDIUM (3.0-4.9) / LOW (1.5-2.9) / UNKNOWN (<1.5)
- **Scoring**: Rule-based + context bonus (RAG docs, KG facts, inferences)

**Scoring logic:**

| Factor | Score Contribution |
|--------|-------------------|
| Base rules | 0-3 points (greeting, fact retrieval, comparison) |
| RAG documents | +0.1 per doc (max 0.5) |
| KG facts | +0.2 per fact (max 0.4) |
| OWL inferences | +0.3 per inference (max 0.6) |

**Example:**

- Query: "LANEIGE 순위는?" → Base: 3.0 (fact retrieval) + RAG: 0.3 + KG: 0.4 + OWL: 0.6 = **4.3 (MEDIUM)**

### 2.6 Decision Making

- **File**: `src/core/decision_maker.py`
- **Approach**: LLM-first with mode-specific prompts
- **MODE_PROMPTS**: high, medium, low, unknown
- **Output**: Tool selection or direct_answer

The decision maker determines whether to answer directly or invoke specialized tools.

**Mode characteristics:**

| Mode | Behavior | Example |
|------|----------|---------|
| HIGH | Often direct_answer | "LANEIGE 순위는 3위입니다" |
| MEDIUM | Balanced | "최근 순위 데이터를 확인해볼게요" |
| LOW | Tool-heavy | "먼저 KPI를 계산하고, 외부 뉴스를 확인하겠습니다" |

### 2.7 Response Generation

- **File**: `src/core/response_pipeline.py`
- **Fast path**: HIGH confidence (lighter prompt, fewer tokens)
- **Standard path**: MEDIUM/LOW (full context, detailed reasoning)
- **Defensive coding**: None-safety, timeout handling

**Pipeline stages:**

1. Context validation
2. Prompt construction (mode-specific)
3. LLM generation
4. Formatting & validation
5. Explainability trace attachment

### 2.8 Explainability

- **File**: `src/core/explainability.py`
- **Tracking**: ExplanationTrace records sources, routing path, key evidence
- **Formats**: Human-readable (markdown) and machine-readable (JSON)

**Trace structure:**

```python
{
  "sources": ["RAG docs: 3", "KG facts: 5", "OWL inferences: 2"],
  "routing_path": ["ConfidenceAssessor", "DecisionMaker", "ResponsePipeline"],
  "key_evidence": ["LANEIGE SoS: 0.23", "Rank: 3"],
  "confidence_tier": "MEDIUM"
}
```

### 2.9 ReAct Agent — [2026-09 사후] 도구·활성화 방식 갱신

- **File**: `src/core/react_agent.py`
- **Loop**: Thought → Action(네이티브 function calling, `tools=`/`tool_choice="auto"`) → Observation → Self-Reflection (기본 `max_iterations=5`)
- **Activation**: 키워드 기반 자동 트리거가 아니라 ① 홉 카운트 라우터(`src/core/router.py`, `hops >= HOP_THRESHOLD(2)`)가 먼저 판단하고 ② 플래그 `agents.use_react_agent`(기본 OFF)가 켜져 있어야 실제로 ReAct 경로를 탄다. OFF 상태에서는 `agents.react_shadow_mode`(기본 OFF)로 그림자 실행만 가능하다. 단, 신뢰도가 HIGH면 홉 수와 무관하게 파이프라인이 답한다 — ReAct는 신뢰도 MEDIUM/LOW + 2홉 이상일 때만 탄다(측정용 플래그 `agents.react_bypass_confidence`, 기본 OFF, 로 이 관문을 건너뛸 수 있다). [2026-09-18 사후] 6단계 비교 결과 세 플래그 모두 기본 OFF 유지(`docs/plans/evidence-react-ontology-decisions-2026-09.md` S6-3).
- **Tools**: ReAct 전용 도구 3종·대시보드 JSON 도구 5종은 폐지되었다. 지금은 DecisionMaker와 공유하는 5종 레지스트리(`resolve_entity`·`kg_neighbors`·`get_metrics`·`apply_rules`·`search_docs`, `src/core/tool_registry.py`)만 호출하고, 결과는 증거 카드로 돌아온다.
- **Token budget**: 질문당 기본 12,000 토큰(`DEFAULT_TOKEN_BUDGET`), 초과 시 `ReActResult.budget_exceeded=True`

<details>
<summary>Original text (kept for history, describes the ReAct loop before the 2026-09 rework — tool names below no longer exist)</summary>

- **Loop**: Thought → Action → Observation → Reflection (max 3 iterations)
- **Activation**: Auto-triggered for complex queries (analysis keywords, multi-step)

**Complexity detection criteria:**

- Analysis keywords: "분석", "비교", "전략", "예측"
- Multi-step queries: "먼저 A를 확인하고 B를 분석해줘"
- Context insufficiency: Confidence < 2.0 after initial context gathering

**ReAct execution example:**

```
Query: "LANEIGE가 경쟁사 대비 어떤 위치에 있는지 분석해줘"

Thought 1: "경쟁사 SoS 데이터가 필요하다"
Action 1: calculate_market_metrics
Observation 1: {LANEIGE: 0.23, CeraVe: 0.18, ...}

Thought 2: "순위 추이도 확인이 필요하다"
Action 2: get_rank_history
Observation 2: {LANEIGE: [3, 4, 3], CeraVe: [5, 6, 7]}

Reflection: "충분한 데이터를 수집했다. 시장 위치를 분석할 수 있다"
→ Final Response
```

</details>

---

## 3. Data Flow

### 3.1 Query Processing

```
1. User query → UnifiedBrain.process_query()
2. Entity extraction (keywords, brand names)
3. Context gathering:
   - RAG: ChromaDB semantic search
   - KG: Entity lookup + relationship traversal
   - Ontology: OWL inference (classification, property reasoning)
4. Confidence assessment (rule-based + context scoring)
5. Routing decision:
   - HIGH: Direct response
   - MEDIUM: DecisionMaker → possible tool execution
   - LOW: Full DecisionMaker → multi-tool execution
   - UNKNOWN: Clarification request
6. Response generation (mode-specific prompt)
7. Explainability trace attachment
```

### 3.2 Data Ingestion

```
1. Amazon crawling:
   - Playwright + playwright-stealth + browserforge
   - Anti-detection: random delays, realistic mouse movement
   - Error handling: exponential backoff, WAF detection
2. KG population:
   - OntologyKnowledgeGraph validation (T-Box compliance)
   - Triple insertion: <LANEIGE, hasSoS, 0.23>
   - Auto-classification via OWL reasoning
3. RAG indexing:
   - Document chunking (500 tokens, 100 overlap)
   - OpenAI embeddings (text-embedding-3-small)
   - ChromaDB persistence
4. OWL inference sync:
   - Run Pellet reasoner
   - Extract inferred triples
   - Update KG with inferred relationships
```

---

## 4. Module Map

| Module | File | Responsibility |
|--------|------|---------------|
| **UnifiedBrain** | `src/core/brain.py` | Orchestration, routing, ReAct activation |
| **ConfidenceAssessor** | `src/core/confidence.py` | 4-tier confidence routing |
| **DecisionMaker** | `src/core/decision_maker.py` | LLM-based tool selection |
| **ResponsePipeline** | `src/core/response_pipeline.py` | Response generation (mode-specific) |
| **ContextGatherer** | `src/core/context_gatherer.py` | RAG+KG context collection |
| **ExplainabilityEngine** | `src/core/explainability.py` | Transparency & tracing |
| **ReActAgent** | `src/core/react_agent.py` | Complex query handling (self-reflection) |
| **OntologyKnowledgeGraph** | `src/ontology/ontology_knowledge_graph.py` | T-Box/A-Box bridge, validation |
| **UnifiedReasoner** | ~~`src/ontology/unified_reasoner.py`~~ | [post-2026-09] deleted, 0 callers. Live reasoning is `OntologyReasoner` (`src/ontology/reasoner.py`) |
| **HybridRetriever** | `src/rag/hybrid_retriever.py` | Hybrid retrieval (RAG+KG+rules). Note: `true_hybrid_retriever.py` does not exist in this repo — verify before citing |
| **KnowledgeGraph** | `src/ontology/knowledge_graph.py` | Triple store (subject-predicate-object) |
| **OWLReasoner** | `src/ontology/owl_reasoner.py` | Formal reasoning (owlready2 + Pellet) |

---

## 5. Testing Strategy

| Test Type | Location | Count | Purpose |
|-----------|----------|-------|---------|
| **Type Flow** | `tests/unit/core/test_type_flow.py` | 15 | Validate ConfidenceLevel, DecisionMakerMode enums |
| **Confidence Routing** | `tests/unit/core/test_confidence_routing.py` | 13 | Test 4-tier confidence scoring |
| **Ontology KG** | `tests/unit/ontology/test_ontology_kg.py` | 9 | Validate T-Box/A-Box consistency |
| **DecisionMaker Modes** | `tests/unit/core/test_decision_maker_modes.py` | 6 | Test mode-specific prompts |
| **ReAct Integration** | `tests/unit/core/test_react_integration.py` | 10 | Test ReAct loop, auto-activation |
| **Golden Set** | `tests/golden/chatbot_golden.jsonl` | 20 | End-to-end quality benchmark |

### Coverage Target

- **Minimum**: 60% overall
- **Critical modules**: 80%+ (brain.py, confidence.py, decision_maker.py)

### Running Tests

```bash
# All tests with coverage
python3 -m pytest tests/ -v

# Domain layer only
python3 -m pytest tests/unit/domain/ -v

# Golden set evaluation
python3 scripts/evaluate_golden.py --verbose

# Coverage report
open coverage_html/index.html
```

---

## 6. Deployment

### Railway Configuration

| Component | Configuration |
|-----------|--------------|
| **Platform** | Railway with Volume mount (`/data`) |
| **Storage** | SQLite (primary) + Google Sheets (backup) |
| **KG Backup** | Auto-backup (7-day rolling, `/data/backups/kg/`) |
| **Health Check** | `/api/health` (HTTP 200) |
| **Port** | Railway-injected `PORT` environment variable |

### Data Persistence

```
Railway Volume: /data
├── amore_data.db          # SQLite (source of truth)
├── knowledge_graph.json   # KG triples
└── backups/
    └── kg/
        ├── 2026-01-27_kg_backup.json
        ├── 2026-01-28_kg_backup.json
        └── ... (7-day rolling)
```

### Synchronization

```
Railway SQLite (Source of Truth)
      ↓
Google Sheets (Backup)
      ↓
Local SQLite (Dev)
```

**Sync commands:**

```bash
# Railway → Local
python3 scripts/sync_from_railway.py

# Sheets → SQLite
python3 scripts/sync_sheets_to_sqlite.py
```

### Monitoring

- **Health endpoint**: `/api/health`
- **Status endpoint**: `/api/v4/brain/status`
- **Telegram Admin Bot**: Webhook for logs, errors, system status
- **Email Alerts**: Gmail SMTP for rank changes, SoS variations

---

## 7. Design Principles

### 7.1 Confidence-First Routing

Avoid over-computation by routing low-confidence queries directly to clarification.

### 7.2 Defensive Context Handling

Always validate context before LLM calls. Return structured errors, not exceptions.

### 7.3 Explainability by Default

Every response includes a trace of sources and reasoning steps.

### 7.4 Clean Architecture

```
Domain (entities) ← Application (workflows) ← Adapters (tools) ← Infrastructure (FastAPI, DB)
```

No cross-layer imports. DI via Protocol interfaces.

### 7.5 Test-Driven Development

Tests before implementation. Mock minimally. Focus on integration tests.

---

## 8. Future Roadmap

| Feature | Priority | Status |
|---------|----------|--------|
| SHACL constraint validation | Low | Not started |
| Webhook signature verification | Medium | Not started |
| Document chunk_id tracking | Medium | Not started |
| Prompt injection defense | High | Planned |
| Amazon review sentiment analysis | Medium | Planned |
| Multi-brand comparison dashboard | High | In progress |

---

## Appendix: Key Algorithms

### A.1 Confidence Scoring

```python
def calculate_confidence(query: str, context: Dict) -> float:
    base_score = apply_rules(query)  # 0-3 points
    rag_bonus = min(len(context["documents"]) * 0.1, 0.5)
    kg_bonus = min(len(context["facts"]) * 0.2, 0.4)
    owl_bonus = min(len(context["inferences"]) * 0.3, 0.6)
    return base_score + rag_bonus + kg_bonus + owl_bonus
```

### A.2 Entity Linking

```python
def extract_entities(query: str) -> List[Tuple[str, float]]:
    # Simple keyword matching + fuzzy matching
    entities = []
    for brand in KNOWN_BRANDS:
        if brand.lower() in query.lower():
            entities.append((brand, 1.0))
    return entities
```

### A.3 OWL Reasoning

```python
def classify_entity(entity_uri: URIRef) -> List[URIRef]:
    # Run Pellet reasoner
    owlready2.sync_reasoner_pellet([ontology])
    # Get inferred classes
    inferred_classes = list(entity.is_a)
    return inferred_classes
```

---

**Document Version**: 1.0
**Last Updated**: 2026-02-09
**Maintainer**: AMORE Pacific Data Team
