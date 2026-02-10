# src/core - Orchestration Layer

## OVERVIEW

Central orchestration: UnifiedBrain facade, ReAct self-reflection agent, autonomous scheduler, and batch workflow coordination.

## KEY MODULES

| Module | File | Role |
|--------|------|------|
| UnifiedBrain | `brain.py` | Facade: DecisionMaker + ToolCoordinator + AlertManager + QueryProcessor + ResponsePipeline |
| ReActAgent | `react_agent.py` | Thought-Action-Observation loop (max 3 iterations) |
| Scheduler | `scheduler.py` | AutonomousScheduler with persisted state |
| BatchWorkflow | `batch_workflow.py` | Daily crawl pipeline orchestration |

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add new tool to ReAct | `react_agent.py` | Register in ALLOWED_ACTIONS + ACTION_SCHEMAS |
| Modify query routing | `brain.py` | DecisionMaker.decide() |
| Change schedule times | `scheduler.py` | DEFAULT_SCHEDULES dict |
| Add brain event handler | `brain.py` | emit_event() supports sync/async handlers |

## PATTERNS

### Brain SRP Components
```
DecisionMaker     → Tool selection logic
ToolCoordinator   → Tool execution orchestration  
AlertManager      → Event-driven alert dispatch
QueryProcessor    → Query classification + preprocessing
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
