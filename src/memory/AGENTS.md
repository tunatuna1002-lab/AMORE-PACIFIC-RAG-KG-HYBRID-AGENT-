# src/memory - Conversation & Session State

## OVERVIEW

In-memory and file-backed state management for conversation context, session lifecycle, and execution history.

## KEY MODULES

| Module | File | Role |
|--------|------|------|
| ContextManager | `context.py` | Conversation turns + workflow/data context |
| SessionManager | `session.py` | Agent lifecycle + session status |
| HistoryManager | `history.py` | Execution history + statistics |

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Modify context building | `context.py` | `build_llm_context()` |
| Track session state | `session.py` | `update_agent_status()` |
| Query execution stats | `history.py` | `get_success_rate()`, `get_daily_stats()` |

## CONTEXT MANAGEMENT

### ContextManager
```python
ctx = ContextManager(session_id)
ctx.add_turn(role="user", content="query")
ctx.set_workflow_context({"step": "crawl"})
ctx.save_context()  # Persists to data/context/

# For LLM prompt
prompt_context = ctx.build_llm_context()
```

### Limits
- Max turns: 100 (FIFO eviction)
- Summary window: last 5 turns

### Persistence
- Location: `data/context/{session_id}_context.json`
- Contains: conversation, workflow, data, variables

## SESSION MANAGEMENT

### SessionManager
```python
session = SessionManager()
session.start_session(session_id)
session.update_agent_status(agent_id, "running")
session.end_session(session_id)
summary = session.get_session_summary()
```

### State Tracking
- Session start/end times
- Per-agent status and durations
- No disk persistence (in-memory only)

## HISTORY MANAGEMENT

### HistoryManager
```python
history = HistoryManager()
history.record_execution(workflow_id, status, duration, details)
stats = history.get_success_rate()
daily = history.get_daily_stats()
errors = history.get_error_summary()
```

### Persistence
- Location: `data/history/execution_history.json`
- Max records: 1000 (FIFO eviction)

## ANTI-PATTERNS

- **NEVER** exceed turn limits without explicit eviction
- **NEVER** store sensitive data in context (API keys, tokens)
- **NEVER** rely on SessionManager for persistence (memory-only)
- **NEVER** bypass HistoryManager for execution tracking
