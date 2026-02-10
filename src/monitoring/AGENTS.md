# src/monitoring - Observability Stack

## OVERVIEW

Structured logging, execution tracing, and quality metrics for agents and workflows.

## KEY MODULES

| Module | File | Role |
|--------|------|------|
| AgentLogger | `logger.py` | Console + file logging with sensitive data masking |
| ExecutionTracer | `tracer.py` | Span-based execution tracing |
| QualityMetrics | `metrics.py` | Counters, gauges, histograms, timers |

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add log category | `logger.py` | Add method to AgentLogger |
| Track new metric | `metrics.py` | Use counters/gauges/histograms |
| Add trace span | `tracer.py` | Use `start_span()` context manager |

## LOGGING PATTERNS

### AgentLogger
```python
logger = AgentLogger("agent_name")
logger.agent_start(agent_id, config)
logger.tool_call(tool_name, params, result)
logger.chat_request(session_id, query, context_size)
logger.chat_response(session_id, response, duration, sources)
```

### Output Locations
- Console: Structured format
- Daily logs: `logs/{name}_YYYY-MM-DD.log`
- Chat audit: `logs/chatbot_audit_YYYY-MM-DD.jsonl`

### Sensitive Data Filter
Masks API keys, tokens, passwords in log output.

## TRACING PATTERNS

### ExecutionTracer
```python
tracer = ExecutionTracer()
with tracer.start_span("operation_name", attributes={"key": "value"}):
    # Tracked operation
    tracer.add_event("checkpoint", {"detail": "value"})
```

### Trace Output
- Location: `data/traces/trace_*.json`
- Structure: spans with timing, status, attributes, events

## METRICS PATTERNS

### QualityMetrics
```python
metrics = QualityMetrics()
metrics.increment("requests_total")
metrics.gauge("active_sessions", value)
metrics.histogram("response_time", duration)
metrics.timer("operation").start() / .stop()
```

### Metric Snapshots
- Location: `data/metrics/metrics_YYYY-MM-DD.json`
- Includes: session, agent, LLM, crawl metrics

## ANTI-PATTERNS

- **NEVER** log sensitive data without SensitiveDataFilter
- **NEVER** skip tracing in production agents
- **NEVER** use print() instead of AgentLogger
- **NEVER** create metrics without snapshot persistence
