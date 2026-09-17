# src/core - Orchestration Layer

## OVERVIEW

Central orchestration: UnifiedBrain facade, ReAct self-reflection agent, autonomous scheduler, and batch workflow coordination.

## KEY MODULES

| Module | File | Role |
|--------|------|------|
| UnifiedBrain | `brain.py` | Facade: DecisionMaker + ToolCoordinator + AlertManager + ContextGatherer + ResponsePipeline (query classification lives in `query_graph.py`'s QueryGraph, not a separate QueryProcessor — that module was deleted, [post-2026-09]) |
| ReActAgent | `react_agent.py` | Thought-Action-Observation loop (max 5 iterations, `max_iterations` default). Shares `tool_registry.py`'s 5-tool registry with DecisionMaker via the `react_tools.py` adapter, [post-2026-09] |
| Scheduler | `scheduler.py` | AutonomousScheduler with persisted state |
| BatchWorkflow | `batch_workflow.py` | Daily crawl pipeline orchestration |

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add a tool (ReAct + DecisionMaker) | `tool_registry.py` | Add a `ToolDefinition`; `react_agent.py`'s `ALLOWED_ACTIONS`/`ACTION_SCHEMAS` and DecisionMaker's function-calling schema are both generated from it, [post-2026-09] |
| Modify confidence/route branching | `query_graph.py` | `QueryGraph`; tool-selection logic itself is `DecisionMaker.decide()` (`decision_maker.py`) |
| Change schedule times | `scheduler.py` | DEFAULT_SCHEDULES dict |
| Add brain event handler | `brain.py` | emit_event() supports sync/async handlers |

## PATTERNS

### Brain SRP Components
```
DecisionMaker     → Tool selection logic (native function calling, [post-2026-09])
ToolCoordinator   → Tool execution orchestration
AlertManager      → Event-driven alert dispatch
QueryGraph        → Query routing + confidence branching (query_graph.py; replaces the deleted QueryProcessor, [post-2026-09])
ResponsePipeline  → Final response generation
ContextGatherer   → KG + Ontology + RAG context
```

### ReAct Loop
```
Thought → Action → Observation → Reflection (quality check)
         ↓
    ALLOWED_ACTIONS whitelist validation
         ↓
    ACTION_SCHEMAS input validation
```

### Scheduler Patterns
- Pull-style: `run_autonomous_cycle` checks due tasks
- Push-style: `start_scheduler` background loop
- State persisted to `./data/scheduler_state.json`

## ANTI-PATTERNS

- **NEVER** add tool to ReAct without ACTION_SCHEMAS validation
- **NEVER** bypass DecisionMaker for tool selection
- **NEVER** block scheduler loop with sync I/O
- **NEVER** modify brain state outside asyncio.Lock
