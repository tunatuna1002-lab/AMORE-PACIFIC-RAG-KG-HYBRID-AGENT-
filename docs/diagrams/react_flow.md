# ReAct Self-Reflection Flow Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         UnifiedBrain                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  process_query(query)                                    │  │
│  │                                                          │  │
│  │  1. Cache Check                                         │  │
│  │  2. Context Gathering (RAG + KG)                        │  │
│  │  3. Complexity Detection ◄───┐                          │  │
│  │                               │                          │  │
│  │     ┌─────────────────────────┴──────────────┐          │  │
│  │     │ Simple Query  │  Complex Query         │          │  │
│  │     ▼              │         ▼               │          │  │
│  │  Normal Mode       │     ReAct Mode          │          │  │
│  │  (Single LLM)      │     (Multi-Step)        │          │  │
│  │     │              │         │               │          │  │
│  │     └──────────────┴─────────┘               │          │  │
│  │                │                              │          │  │
│  │  4. Response Generation                      │          │  │
│  │  5. Cache Save                                │          │  │
│  └──────────────────────────────────────────────┘          │  │
│                                                             │  │
└─────────────────────────────────────────────────────────────┘
```

## Complexity Detection Logic

```
┌──────────────────────────────────────────────────────────┐
│  _is_complex_query(query, context)                       │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 1. Keyword Analysis                                │ │
│  │    ✓ "왜", "어떻게", "비교", "분석", "추천"          │ │
│  │    ✓ "전략", "예측", "원인"                         │ │
│  └────────────────────────────────────────────────────┘ │
│                       ▼                                  │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 2. Context Sufficiency                             │ │
│  │    ✗ RAG docs < 2                                  │ │
│  │    ✗ KG triples empty                              │ │
│  └────────────────────────────────────────────────────┘ │
│                       ▼                                  │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 3. Multi-Step Detection                            │ │
│  │    ✓ Multiple "?"                                  │ │
│  │    ✓ Conjunctions ("그리고", "하지만")              │ │
│  └────────────────────────────────────────────────────┘ │
│                       ▼                                  │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Decision: Complex if ANY condition true            │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## ReAct Loop Detail

```
┌─────────────────────────────────────────────────────────────────┐
│                   ReActAgent.run()                              │
│                                                                 │
│  Iteration 1                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Thought                                              │   │
│  │    "현재 LANEIGE 순위를 확인해야 합니다"                  │   │
│  │                                                         │   │
│  │ 2. Action Selection                                     │   │
│  │    action: "query_data"                                 │   │
│  │    params: {"query_type": "brand_metrics"}              │   │
│  │                                                         │   │
│  │ 3. Tool Execution                                       │   │
│  │    ToolExecutor.execute("query_data", {...})            │   │
│  │                                                         │   │
│  │ 4. Observation                                          │   │
│  │    {"brand": "LANEIGE", "rank": 5, "sos": 12.5}         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ▼                                      │
│  Iteration 2                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Thought                                              │   │
│  │    "경쟁사 정보도 필요합니다"                             │   │
│  │                                                         │   │
│  │ 2. Action Selection                                     │   │
│  │    action: "query_knowledge_graph"                      │   │
│  │    params: {"entity": "LANEIGE", "relation": "all"}     │   │
│  │                                                         │   │
│  │ 3. Tool Execution                                       │   │
│  │    ToolExecutor.execute("query_knowledge_graph", {...}) │   │
│  │                                                         │   │
│  │ 4. Observation                                          │   │
│  │    {"competitors": ["CeraVe", "Neutrogena"]}            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ▼                                      │
│  Iteration 3                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Thought                                              │   │
│  │    "충분한 정보를 수집했으니 답변 생성"                    │   │
│  │                                                         │   │
│  │ 2. Action Selection                                     │   │
│  │    action: "final_answer" ◄─── Loop Termination         │   │
│  │                                                         │   │
│  │ 3. Answer Generation                                    │   │
│  │    "LANEIGE는 5위이며, CeraVe, Neutrogena와 경쟁..."    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ▼                                      │
│  Self-Reflection                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Quality Assessment                                      │   │
│  │                                                         │   │
│  │ 1. Completeness Check                                   │   │
│  │    ✓ All questions answered                             │   │
│  │                                                         │   │
│  │ 2. Missing Info Check                                   │   │
│  │    ✗ 구체적인 경쟁 전략 누락                             │   │
│  │                                                         │   │
│  │ 3. Evidence Sufficiency                                 │   │
│  │    ✓ Data-backed answer                                 │   │
│  │                                                         │   │
│  │ Result:                                                 │   │
│  │    confidence: 0.75                                     │   │
│  │    needs_improvement: True                              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
User Query
    │
    ▼
┌───────────────┐
│ UnifiedBrain  │
└───────────────┘
    │
    ├─────────────────────────────────┐
    │                                 │
    ▼                                 ▼
┌─────────────┐              ┌──────────────┐
│ Normal Mode │              │ ReAct Mode   │
└─────────────┘              └──────────────┘
    │                                 │
    │                                 ▼
    │                        ┌──────────────┐
    │                        │ ReActAgent   │
    │                        └──────────────┘
    │                                 │
    │                                 ├─► Step 1: Thought
    │                                 │
    │                                 ├─► Step 2: Action
    │                                 │       │
    │                                 │       ▼
    │                                 │   ┌──────────────┐
    │                                 │   │ ToolExecutor │
    │                                 │   └──────────────┘
    │                                 │       │
    │                                 │       ▼
    │                                 ├─► Step 3: Observation
    │                                 │
    │                                 ├─► Step N: Final Answer
    │                                 │
    │                                 ▼
    │                        ┌──────────────────┐
    │                        │ Self-Reflection  │
    │                        │ - Confidence     │
    │                        │ - Quality Score  │
    │                        └──────────────────┘
    │                                 │
    └─────────────────┬───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │    Response   │
              │  - content    │
              │  - confidence │
              │  - metadata   │
              └───────────────┘
                      │
                      ▼
                  User
```

## Tool Integration

```
┌────────────────────────────────────────────────────────────┐
│                     ToolExecutor                           │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ query_data   │  │ query_kg     │  │ calculate_   │    │
│  │              │  │              │  │ metrics      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         ▲                  ▲                  ▲           │
│         │                  │                  │           │
│         └──────────────────┴──────────────────┘           │
│                            │                              │
└────────────────────────────┼──────────────────────────────┘
                             │
                             │ Tool Call
                             │
                    ┌────────▼────────┐
                    │   ReActAgent    │
                    │                 │
                    │ step.action     │
                    │ step.action_    │
                    │      input      │
                    └─────────────────┘
```

## Comparison: Normal vs ReAct

### Normal Mode (Simple Query)
```
Query: "LANEIGE 순위는?"
  │
  ▼
LLM Decision
  │
  ▼
query_data
  │
  ▼
Response
  │
Total: ~2 seconds
LLM Calls: 1
```

### ReAct Mode (Complex Query)
```
Query: "LANEIGE가 경쟁사 대비 어떤 위치인지 분석해줘"
  │
  ▼
Iteration 1: Thought + query_data
  │
  ▼
Iteration 2: Thought + query_knowledge_graph
  │
  ▼
Iteration 3: Thought + final_answer
  │
  ▼
Self-Reflection
  │
  ▼
Response
  │
Total: ~10 seconds
LLM Calls: 4-5
Quality: +25% confidence
```

## Error Handling Flow

```
┌──────────────────────────────────────────────────────────┐
│                  ReAct Execution                         │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Try: ReAct Loop                                    │ │
│  │                                                    │ │
│  │ ┌────────────────────────────────────────────────┐ │ │
│  │ │ Step Execution                                 │ │ │
│  │ │                                                │ │ │
│  │ │   Tool Call ───► Success ───► Observation      │ │ │
│  │ │       │                                        │ │ │
│  │ │       └──► Error ───► observation = "Error:..." │ │ │
│  │ └────────────────────────────────────────────────┘ │ │
│  │                                                    │ │
│  │ Continue to next iteration                         │ │
│  └────────────────────────────────────────────────────┘ │
│                       │                                  │
│                       ▼                                  │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Catch: Exception                                   │ │
│  │                                                    │ │
│  │ Fallback to Normal Mode                            │ │
│  │                                                    │ │
│  │ Response.fallback("ReAct 처리 실패: {error}")      │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## State Diagram

```
        ┌─────────┐
        │  IDLE   │
        └────┬────┘
             │ query
             ▼
    ┌────────────────┐
    │ Complexity     │
    │ Detection      │
    └───┬───────┬────┘
        │       │
    Simple  Complex
        │       │
        ▼       ▼
    ┌──────┐ ┌─────────────┐
    │Normal│ │ReAct Loop   │◄───┐
    │ Mode │ │(Iteration N)│    │ iterations < max
    └──┬───┘ └──────┬──────┘    │
       │            │  final_   │
       │            │  answer   │
       │            ▼           │
       │     ┌─────────────┐   │
       │     │Self-        │   │
       │     │Reflection   │   │
       │     └──────┬──────┘   │
       │            │           │
       └────────┬───┘           │
                │               │
                ▼               │
         ┌────────────┐         │
         │  Response  │         │
         └────────────┘         │
                                │
                        max iterations reached
                        or error
```
