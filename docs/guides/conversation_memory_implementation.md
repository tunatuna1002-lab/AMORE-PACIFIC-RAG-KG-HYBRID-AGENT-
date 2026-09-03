# ConversationMemory Implementation (Task 3.3)

## Overview

Implemented a `ConversationMemory` module that tracks conversation history per session, provides recent turns + summarization for LLM context, and tracks entities mentioned across turns.

## Features

### 1. Session-based Conversation Tracking
- Per-session conversation history storage
- Automatic timestamping of each turn
- Support for both user and assistant messages

### 2. LRU Session Management
- Configurable maximum sessions (default: 100)
- Least Recently Used (LRU) eviction policy
- Automatic session cleanup when limit exceeded

### 3. Recent Turns Management
- Configurable recent turns window (default: 6)
- Full history preservation
- Efficient context window for LLM prompts

### 4. Automatic Summarization
- Turns exceeding session limit are automatically summarized
- Summary length capped at 2000 characters
- Preserves conversation continuity while reducing token usage

### 5. Entity Tracking
- Automatic extraction of brands (LANEIGE, COSRX, TIRTIR, etc.)
- Automatic extraction of indicators (SoS, HHI, CPI)
- Support for Korean and English keywords
- Entity accumulation across conversation turns
- Deduplication of entities

### 6. LLM Prompt Generation
- `to_prompt_text()` method generates formatted context
- Includes summary, recent turns, and tracked entities
- Optimized for LLM consumption
- Automatic content truncation for long messages

## Files Created

```
src/memory/conversation_memory.py           # Main implementation
tests/unit/memory/test_conversation_memory.py  # Comprehensive tests
examples/conversation_memory_demo.py        # Usage examples
docs/conversation_memory_implementation.md  # This document
```

## Updated Files

```
src/memory/__init__.py                      # Added ConversationMemory export
```

## API Reference

### ConversationMemory

```python
memory = ConversationMemory(
    max_recent_turns=6,        # Recent turns to keep in context
    max_sessions=100,          # Maximum concurrent sessions
    max_turns_per_session=50   # Turns per session before summarization
)
```

#### Methods

**add_turn(session_id, role, content, entities=None)**
- Add a conversation turn
- `role`: "user" or "assistant"
- `content`: Message text
- `entities`: Optional pre-extracted entities

**get_context(session_id) -> ConversationContext**
- Get conversation context for LLM
- Returns recent turns, summary, and tracked entities

**get_history(session_id) -> list[dict]**
- Get full conversation history for a session

**clear_session(session_id)**
- Remove all data for a session

**get_stats() -> dict**
- Get memory statistics

### ConversationContext

```python
@dataclass
class ConversationContext:
    summary: str                              # Previous conversation summary
    recent_turns: list[dict]                  # Recent N turns
    total_turns: int                          # Total turn count
    tracked_entities: dict[str, list[str]]    # Extracted entities
    session_id: str                           # Session identifier
```

**to_prompt_text() -> str**
- Generate formatted text for LLM prompts

### ConversationTurn

```python
@dataclass
class ConversationTurn:
    role: str                                 # "user" or "assistant"
    content: str                              # Message content
    timestamp: datetime                       # When added
    entities: dict[str, list[str]]           # Extracted entities
    metadata: dict[str, Any]                 # Additional metadata
```

## Usage Examples

### Basic Usage

```python
from src.memory import ConversationMemory

memory = ConversationMemory()

# Add conversation turns
memory.add_turn("session1", "user", "LANEIGE 점유율 알려줘")
memory.add_turn("session1", "assistant", "현재 LANEIGE의 SoS는 12.5%입니다.")

# Get context for LLM
ctx = memory.get_context("session1")
print(f"Tracked brands: {ctx.tracked_entities['brands']}")
print(f"Recent turns: {len(ctx.recent_turns)}")
```

### LLM Prompt Integration

```python
memory = ConversationMemory()
# ... add turns ...

ctx = memory.get_context("session1")
prompt = f"""
{ctx.to_prompt_text()}

User query: {user_message}

Please respond based on the conversation history above.
"""
```

### Entity Tracking

```python
memory = ConversationMemory()
memory.add_turn("s1", "user", "LANEIGE와 COSRX 비교")
memory.add_turn("s1", "user", "SoS와 HHI로")

ctx = memory.get_context("s1")
# ctx.tracked_entities = {
#     "brands": ["LANEIGE", "COSRX"],
#     "indicators": ["SoS", "HHI"]
# }
```

## Entity Keywords

### Brands
- English: laneige, cosrx, tirtir, rare beauty, innisfree, sulwhasoo
- Korean: 라네즈, 코스알엑스, 티르티르, 설화수

### Indicators
- English: sos, share of shelf, hhi, cpi
- Korean: 점유율, 집중도, 가격지수

## Test Coverage

- **33 tests** covering all functionality
- Test classes:
  - `TestConversationTurn` (2 tests)
  - `TestConversationContext` (5 tests)
  - `TestConversationMemory` (26 tests)

### Test Categories
- Basic operations (add, get, clear)
- Recent turns limiting
- Entity extraction and tracking
- LRU eviction
- Summarization
- Multi-session independence
- Edge cases (empty sessions, duplicates, etc.)

## Performance Characteristics

### Memory Usage
- ~500 bytes per turn (typical)
- ~2KB per session summary (max)
- ~10KB per session (50 turns + summary)
- ~1MB for 100 sessions (typical)

### Time Complexity
- `add_turn()`: O(1) amortized
- `get_context()`: O(N) where N = recent_turns
- `clear_session()`: O(1)
- `_touch_session()`: O(S) where S = max_sessions

## Design Decisions

### LangGraph add_messages Pattern
The implementation follows LangGraph's add_messages reducer pattern:
- Messages are appended to history
- Recent window provides working context
- Old messages are summarized, not discarded
- Maintains conversation continuity

### LRU Eviction
Sessions are evicted based on access time:
- Every `add_turn()` or `get_context()` updates access time
- Oldest unused sessions are removed first
- Prevents unbounded memory growth

### Automatic Summarization
When turns exceed `max_turns_per_session`:
- Excess turns are summarized
- Summaries are concatenated
- Summary length is capped at 2000 chars
- Most recent turns are preserved verbatim

### Entity Extraction
Simple keyword-based extraction:
- Fast and deterministic
- No external dependencies
- Supports Korean and English
- Can be replaced with NER if needed

## Integration Points

### Current
- Standalone module in `src/memory/`
- No dependencies on other modules
- Ready for integration

### Future Integration (Not Implemented)
- `HybridChatbotAgent`: Use for conversation context
- `UnifiedBrain`: Session management
- `ReActAgent`: Multi-turn reasoning context

## Limitations

1. **Keyword-based Entity Extraction**
   - Only recognizes predefined brands/indicators
   - No NER or ML-based extraction
   - Limited to hardcoded keywords

2. **Simple Summarization**
   - Text truncation, not semantic summarization
   - No LLM-based summary generation
   - Fixed length limit (2000 chars)

3. **No Persistence**
   - In-memory only
   - Lost on process restart
   - No database backing

4. **No Thread Safety**
   - Not designed for concurrent access
   - Single-threaded use assumed

## Future Enhancements

### Phase 1 (Low Effort)
- [ ] Add more brand/indicator keywords
- [ ] Configurable entity extraction rules
- [ ] JSON export/import for sessions

### Phase 2 (Medium Effort)
- [ ] LLM-based summarization
- [ ] Named Entity Recognition (NER)
- [ ] SQLite persistence layer
- [ ] Session expiration (time-based)

### Phase 3 (High Effort)
- [ ] Semantic search over history
- [ ] Cross-session entity tracking
- [ ] Thread-safe concurrent access
- [ ] Distributed session storage

## Verification

### Test Results
```bash
$ python3 -m pytest tests/unit/memory/test_conversation_memory.py -v --no-cov
======================== 33 passed in 0.02s =========================
```

### Import Verification
```bash
$ python3 -c "from src.memory import ConversationMemory; print('✓')"
✓
```

### Demo Execution
```bash
$ python3 examples/conversation_memory_demo.py
✅ All demos completed successfully!
```

## Conclusion

The ConversationMemory module is **fully implemented and tested**. It provides:
- ✅ Session-based conversation tracking
- ✅ LRU session management
- ✅ Recent turns + summarization
- ✅ Automatic entity extraction
- ✅ LLM prompt generation
- ✅ Comprehensive test coverage (33 tests)
- ✅ Usage examples and documentation

**Status**: Ready for integration with chatbot and brain modules.
