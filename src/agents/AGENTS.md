# src/agents - AI Agents

## OVERVIEW

Domain-specific AI agents for chatbot, insight generation, crawling, and alerting.

## KEY AGENTS

| Agent | File | Role |
|-------|------|------|
| HybridChatbotAgent | `hybrid_chatbot_agent.py` | RAG+KG+Ontology Q&A |
| HybridInsightAgent | `hybrid_insight_agent.py` | Daily market insights |
| TrueHybridInsightAgent | `true_hybrid_insight_agent.py` | + OWL reasoning mode |
| PeriodInsightAgent | `period_insight_agent.py` | Period reports with sections |
| CrawlerAgent | `crawler_agent.py` | Amazon category crawling |
| AlertAgent | `alert_agent.py` | Rules-driven alert dispatch |

## AGENT LIFECYCLE

```python
# Standard pattern
agent = HybridChatbotAgent(
    retriever=retriever,        # DI: HybridRetriever
    kg=knowledge_graph,         # DI: KnowledgeGraph
    memory=conversation_memory  # DI: ConversationMemory
)
await agent.initialize()        # Async setup
response = await agent.chat(query)
```

## AGENT PATTERNS

### Dependency Injection
All agents accept interfaces via constructor, not concrete classes.

### Monitoring Hooks
- `AgentLogger`: Structured logging per agent
- `ExecutionTracer`: Span-based tracing
- `QualityMetrics`: Response quality tracking

### Result Caching
Agents cache results in `_results` dict with `get_*` accessors.

## INSIGHT PIPELINE

```
HybridInsightAgent flow:
  KG update → Hybrid retrieval → RAG-to-KG ingest
    → External signals → Market intelligence → LLM insight
    → Actions + Highlights extraction
```

## ALERT PATTERNS

```python
AlertAgent:
  Rules evaluation → Dedupe check → Consent gate (StateManager)
    → Email dispatch (EmailSender)
```

Triggers: rank_changed, sos_threshold, new_competitor

## ANTI-PATTERNS

- **NEVER** instantiate agents without DI (use interfaces)
- **NEVER** skip initialize() before chat/execute
- **NEVER** bypass monitoring hooks in production
- **NEVER** call LLM without context from retriever
