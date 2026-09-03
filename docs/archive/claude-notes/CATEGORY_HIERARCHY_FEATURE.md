# Category Hierarchy Awareness Feature

## Quick Reference Guide

### What This Feature Does

The chatbot now understands that products can rank differently across category hierarchy levels and provides context automatically.

**Example**:
- A product can be **1st in Lip Care** (specific category)
- But **4th in Skin Care** (parent category)
- And **73rd in Beauty & Personal Care** (root category)

This is normal because each level has different competition scope!

---

## Usage Examples

### 1. Query Product Rankings Across Levels

**User**: "립케어 1위 제품의 전체 순위는?"

**Chatbot Response** (will include):
```
**LANEIGE Lip Sleeping Mask**의 카테고리별 순위:
- [Lip Care] 1위 (Level 2)
- [Skin Care] 4위 (Level 1)
- [Beauty & Personal Care] 73위 (Level 0)

Lip Care는 Skin Care의 하위 카테고리이며,
경쟁 범위가 좁아 상위 순위를 기록하고 있습니다.
```

### 2. Understand Category Structure

**User**: "립케어는 어떤 카테고리에 속하나요?"

**Chatbot Response** (will include):
```
**Lip Care** (Level 2)
- 전체 경로: Beauty & Personal Care > Skin Care > Lip Care
- 상위 카테고리: Skin Care
- 하위 카테고리: 없음
```

### 3. Compare Across Hierarchy Levels

**User**: "Why is this product ranked so differently in different categories?"

**Chatbot Response** (will explain):
```
제품의 순위는 각 카테고리 레벨의 경쟁 범위에 따라 다릅니다:

1. Lip Care (Level 2): 립케어 제품들만 경쟁 → 상위 순위
2. Skin Care (Level 1): 모든 스킨케어 제품과 경쟁 → 중위 순위
3. Beauty & Personal Care (Level 0): 전체 뷰티 제품과 경쟁 → 하위 순위

이는 자연스러운 현상입니다.
```

---

## How It Works

### Automatic Detection

The system automatically detects ranking-related queries using keywords:
- Korean: "순위", "위", "등수"
- English: "rank", "ranking", "position"

### Knowledge Graph Integration

```python
# The system uses these methods internally:

# 1. Get category hierarchy
hierarchy = kg.get_category_hierarchy("lip_care")
# Returns: name, level, path, ancestors[], descendants[]

# 2. Get product's rankings across categories
product_ctx = kg.get_product_category_context("B08XYZ1234")
# Returns: rankings in all category levels
```

### Context Building

When a ranking query is detected:
1. Extract mentioned categories and products
2. Query knowledge graph for hierarchy information
3. Build category hierarchy section with HIGH priority
4. Inject into LLM prompt before generation

---

## Category Hierarchy Structure

Current monitored categories:

```
beauty (Level 0) - Beauty & Personal Care
├── skin_care (Level 1) - Skin Care
│   ├── lip_care (Level 2) - Lip Care ✓ MONITORED
│   ├── face_skincare (Level 2) - Face
│   └── body_skincare (Level 2) - Body
└── makeup (Level 1) - Makeup
    ├── lip_makeup (Level 2) - Lips ✓ MONITORED
    └── face_makeup (Level 2) - Face
        └── face_powder (Level 3) - Face Powder ✓ MONITORED
```

---

## Technical Details

### Files Modified

1. **`src/agents/hybrid_chatbot_agent.py`**
   - Builds category hierarchy context
   - Injects into system and user prompts
   - Adds hierarchy to sources

2. **`src/rag/context_builder.py`**
   - Detects ranking queries
   - Builds dedicated hierarchy section
   - Token-efficient context building

3. **`src/rag/hybrid_retriever.py`**
   - Extracts product entities (ASIN)
   - Can find products by rank + category

### Data Source

Category hierarchy is defined in:
**`config/category_hierarchy.json`**

This file contains:
- Category names (Korean + English)
- Amazon node IDs
- Hierarchy levels (0-3)
- Parent-child relationships
- Full paths

---

## Prompting Best Practices

### For Users

**Good Queries** (will trigger hierarchy context):
- "립케어 1위 제품의 순위는?"
- "What's the rank of B08XYZ1234?"
- "Show rankings across all categories"
- "립케어는 어디에 속하나요?"

**Specific Queries** (will get specific answer):
- "립케어 카테고리에서 1위 제품은?" (only Lip Care level)
- "전체 뷰티에서 LANEIGE 순위는?" (only Beauty level)

### For Developers

When debugging, check:
1. **Entity extraction**: Are categories/products extracted correctly?
2. **Hierarchy loading**: Is `config/category_hierarchy.json` loaded?
3. **Knowledge graph**: Are PARENT_CATEGORY relations present?
4. **Context priority**: Is hierarchy section being included?

---

## Limitations

1. **Requires Knowledge Graph**:
   - If KG is not initialized, hierarchy info won't appear
   - Gracefully degrades (no errors)

2. **Token Budget**:
   - Hierarchy section uses ~200-500 tokens
   - Only included for ranking-related queries
   - May be excluded if total context exceeds max_tokens

3. **Product Discovery**:
   - Can only extract products mentioned explicitly (ASIN)
   - Or by rank + category combination
   - Cannot extract from vague descriptions

---

## Future Improvements

### Planned Enhancements

1. **Sibling Category Comparison**
   - "How does this product rank in Lip Makeup vs Lip Care?"

2. **Trend Analysis Across Levels**
   - "Is ranking improving faster in subcategory or parent?"

3. **Competitive Positioning**
   - "Who are the top competitors at each hierarchy level?"

4. **Visual Hierarchy**
   - Generate tree diagrams for complex queries

### Configuration Options

Potential future config:
```json
{
  "category_hierarchy": {
    "include_in_context": true,
    "max_depth": 3,
    "auto_detect_ranking_queries": true,
    "ranking_keywords": ["순위", "rank", "위"]
  }
}
```

---

## Troubleshooting

### Issue: Hierarchy info not showing up

**Check**:
1. Is query about rankings? (contains "순위", "rank", etc.)
2. Are categories/products extracted in entities?
3. Is knowledge_graph passed to context_builder?

**Debug**:
```python
# In chatbot code
logger.debug(f"Entities: {hybrid_context.entities}")
logger.debug(f"Knowledge graph: {self.kg is not None}")
```

### Issue: Wrong category hierarchy

**Check**:
1. Is `config/category_hierarchy.json` correct?
2. Did knowledge graph load the hierarchy?

**Verify**:
```python
kg.get_category_hierarchy("lip_care")
# Should return: name, level, ancestors, descendants
```

### Issue: Product not found

**Check**:
1. Is ASIN format correct? (B0XXXXXXXX)
2. Is product in knowledge graph?

**Verify**:
```python
kg.get_product_category_context("B08XYZ1234")
# Should return: product, categories[]
```

---

## API Integration

For API users accessing the chatbot:

### Request

```json
{
  "message": "립케어 1위 제품의 순위는?",
  "session_id": "user123",
  "include_reasoning": true
}
```

### Response (includes hierarchy in sources)

```json
{
  "response": "...",
  "sources": [
    {
      "type": "category_hierarchy",
      "name": "카테고리 계층 구조",
      "icon": "🗂️",
      "description": "카테고리 계층 관계 및 순위 컨텍스트",
      "details": "**Lip Care** (Level 2)\n- 상위 경로: Beauty & Personal Care > Skin Care > Lip Care\n..."
    },
    ...
  ]
}
```

---

## Summary

This feature enhances the chatbot's understanding of Amazon's category hierarchy, enabling it to:

✅ Recognize products belong to multiple category levels
✅ Always specify which category when mentioning rankings
✅ Explain ranking differences across hierarchy levels
✅ Provide full context automatically for ranking queries
✅ Use real knowledge graph data (not hallucinations)

**Result**: More accurate, contextual, and helpful responses about product rankings!
