# Orchestrator Architecture (After Refactoring)

## Directory Structure

```
/Users/leedongwon/Desktop/AMORE-RAG-ONTOLOGY-HYBRID AGENT/
├── orchestrator.py                          # Backward compatibility wrapper
│   └── imports from → src/core/batch_workflow.py
│
└── src/core/
    ├── batch_workflow.py                    # NEW: Batch workflow orchestrator
    │   └── class BatchWorkflow (alias: Orchestrator)
    │
    ├── brain.py                             # Autonomous scheduler
    │   └── class UnifiedBrain
    │
    └── unified_orchestrator.py              # Chatbot/query orchestrator
        └── class UnifiedOrchestrator
```

## Import Flow

```
Legacy Code:
  from orchestrator import Orchestrator
         ↓
  orchestrator.py (wrapper)
         ↓
  src/core/batch_workflow.py
         ↓
  class BatchWorkflow (as Orchestrator)

New Code (recommended):
  from src.core.batch_workflow import BatchWorkflow
         ↓
  src/core/batch_workflow.py
         ↓
  class BatchWorkflow
```

## Orchestrator Responsibilities

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator Ecosystem                  │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│   UnifiedBrain       │     │  BatchWorkflow       │     │ UnifiedOrchestrator  │
│  (brain.py)          │     │  (batch_workflow.py) │     │ (unified_orch.py)    │
├──────────────────────┤     ├──────────────────────┤     ├──────────────────────┤
│ - Scheduler          │     │ - Think-Act-Observe  │     │ - Query routing      │
│ - Timer management   │─────▶ - Workflow steps     │     │ - RAG retrieval      │
│ - Agent coordinator  │     │ - Agent orchestration│     │ - Context building   │
│ - KST timezone       │     │ - KG updates         │     │ - Response generation│
│                      │     │ - Metrics calculation│     │                      │
│ Calls BatchWorkflow  │     │                      │◀────│ Used by chatbot API  │
│ at scheduled times   │     │ Used for batch jobs  │     │ for user queries     │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

## Usage Examples

### Batch Workflow (Scheduled Tasks)

```python
from src.core.batch_workflow import BatchWorkflow, run_full_workflow

# Quick usage
results = await run_full_workflow(categories=["Beauty & Personal Care"])

# Custom usage
workflow = BatchWorkflow(
    config_path="./config/thresholds.json",
    use_hybrid=True
)
results = await workflow.run_daily_workflow()
```

### Brain (Autonomous Scheduler)

```python
from src.core.brain import get_brain

brain = get_brain()
await brain.start()  # Starts autonomous scheduling
status = brain.get_status()
```

### Unified Orchestrator (Chatbot)

```python
from src.core.unified_orchestrator import get_unified_orchestrator

orchestrator = get_unified_orchestrator()
response = await orchestrator.process(
    query="What is LANEIGE's market share?",
    session_id="user123"
)
```

## Key Changes Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Main file** | `orchestrator.py` (825 lines) | `src/core/batch_workflow.py` (825 lines) |
| **Root file** | Full implementation | Thin wrapper (47 lines) |
| **Class name** | `Orchestrator` | `BatchWorkflow` (Orchestrator is alias) |
| **Logger name** | "orchestrator" | "batch_workflow" |
| **Clarity** | Ambiguous role | Clear: "Batch Workflow Orchestrator" |
| **Documentation** | Minimal | Comprehensive module docstring |
