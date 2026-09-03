# Category Hierarchy Awareness Implementation

## Overview
Added category hierarchy awareness to the hybrid chatbot agent to correctly understand and explain product rankings across different category levels (e.g., "4th in Lip Care but 73rd in overall Beauty & Personal Care").

## Problem Statement
From meeting notes: The chatbot didn't understand context like "a product ranking 4th in Lip Care but 73rd in overall Beauty category" - it lacked awareness of category hierarchies.

## Implementation Changes

### 1. Enhanced Chatbot Agent (`src/agents/hybrid_chatbot_agent.py`)

#### Added: `_build_category_hierarchy_context()` method
- **Purpose**: Generates category hierarchy context from extracted entities
- **Functionality**:
  - Extracts category hierarchy using `KnowledgeGraph.get_category_hierarchy()`
  - Shows parent-child relationships
  - Displays product rankings across multiple category levels
  - Uses `KnowledgeGraph.get_product_category_context()` for product-specific context

#### Modified: `_extract_sources()` method
- Added category hierarchy as a source type
- Icon: 🗂️
- Includes hierarchy details in sources

#### Modified: `_generate_response()` method
- **Enhanced System Prompt**: Added category hierarchy awareness instructions
  - Explains that products can belong to multiple category levels
  - Instructs to always specify which category when mentioning rankings
  - Notes that ranking differences across hierarchy levels are natural

- **Enhanced User Prompt**: Added dedicated category hierarchy section
  - Calls `_build_category_hierarchy_context()` to inject hierarchy info
  - Requirement added: "Always mention category when stating rankings"

### 2. Enhanced Context Builder (`src/rag/context_builder.py`)

#### Modified: `ContextBuilder.build()` method
- **New parameter**: `knowledge_graph` (optional)
- **New logic**: Detects ranking-related queries using keywords
  - Keywords: "순위", "rank", "위", "ranking", "등수"
  - If detected AND entities contain categories/products → builds hierarchy section

#### Added: `_build_category_hierarchy_section()` method
- **Priority**: HIGH (ensures inclusion in context)
- **Builds two types of info**:
  1. **Category hierarchy info**:
     - Current category level
     - Full path from root to current category
     - Child categories (up to 5)

  2. **Product ranking context**:
     - For each product (up to 5)
     - Shows ranking in each category level it belongs to
     - Sorted by hierarchy level (root → leaf)
     - Format: `[Category Name] Xth position (Level Y)`

#### Modified: `CompactContextBuilder.build()` method
- Added `knowledge_graph` parameter
- For ranking queries: Adds compact category ranking section
- Shows up to 3 products × 2 category levels each

### 3. Enhanced Entity Extractor (`src/rag/hybrid_retriever.py`)

#### Modified: `EntityExtractor.extract()` method
- **New parameter**: `knowledge_graph` (optional)
- **New entity type**: `"products"` list

**Product extraction methods**:
1. **ASIN pattern matching**: Regex `\bB0[A-Z0-9]{8}\b`
2. **Rank-based extraction**:
   - Patterns: "1위 제품", "top 1 product", "rank 1"
   - Queries knowledge graph for products at that rank in mentioned categories

#### Modified: `HybridRetriever.retrieve()` method
- Passes `knowledge_graph=self.kg` to entity extractor

## Data Flow

```
User Query: "립케어 1위 제품의 순위는?"
    ↓
EntityExtractor.extract(query, knowledge_graph)
    ↓ Extracts:
    - categories: ["lip_care"]
    - products: [extracted ASIN from rank 1]
    ↓
HybridRetriever.retrieve()
    ↓
ContextBuilder.build(knowledge_graph=kg)
    ↓ Detects "순위" keyword → builds hierarchy section
    ↓
get_product_category_context(asin)
    → Returns rankings in:
       - Lip Care (Level 2): 1위
       - Skin Care (Level 1): 4위
       - Beauty & Personal Care (Level 0): 73위
    ↓
LLM Response with full category context
```

## Example Output Context

```markdown
## 카테고리 계층 구조

### Lip Care
- 레벨: 2
- 전체 경로: Beauty & Personal Care > Skin Care > Lip Care
- 하위 카테고리: (없음)

### 제품별 카테고리 순위

**LANEIGE Lip Sleeping Mask** (ASIN: B08XYZ1234):
  - [Lip Care] 1위 (Level 2)
  - [Skin Care] 4위 (Level 1)
  - [Beauty & Personal Care] 73위 (Level 0)
```

## Integration with Knowledge Graph

The implementation leverages existing KnowledgeGraph methods:

1. **`get_category_hierarchy(category_id)`** (lines 912-977 in knowledge_graph.py)
   - Returns: name, level, path, ancestors[], descendants[]

2. **`get_product_category_context(product_asin)`** (lines 979-1021 in knowledge_graph.py)
   - Returns: product, categories[] with hierarchy info

3. **`load_category_hierarchy()`** (lines 834-910 in knowledge_graph.py)
   - Already loads from `config/category_hierarchy.json`
   - Creates PARENT_CATEGORY and HAS_SUBCATEGORY relations

## Configuration File

Category hierarchy defined in: **`config/category_hierarchy.json`**

Current hierarchy:
```
beauty (Level 0)
├── skin_care (Level 1)
│   ├── lip_care (Level 2)
│   ├── face_skincare (Level 2)
│   └── body_skincare (Level 2)
└── makeup (Level 1)
    ├── lip_makeup (Level 2)
    └── face_makeup (Level 2)
        └── face_powder (Level 3)
```

## Key Features

### 1. Graceful Degradation
- All methods check `if knowledge_graph:` before using it
- Returns empty string if KG not available
- No crashes if hierarchy data missing

### 2. Smart Context Inclusion
- Only includes hierarchy section for ranking-related queries
- Saves token budget for other queries
- Uses priority system (HIGH priority for hierarchy)

### 3. Multi-language Support
- Korean keywords: "순위", "위"
- English keywords: "rank", "ranking"
- Works with both Korean and English queries

### 4. Product Discovery
- Can extract products by ASIN
- Can find products by rank + category
- Supports queries like "Show me the #1 product in Lip Care"

## Testing Recommendations

### Test Cases

1. **Category hierarchy query**:
   ```
   Query: "립케어 카테고리는 어디에 속하나요?"
   Expected: Shows Lip Care → Skin Care → Beauty path
   ```

2. **Multi-level ranking query**:
   ```
   Query: "립케어 1위 제품의 전체 순위는?"
   Expected: Shows ranking in Lip Care, Skin Care, and Beauty levels
   ```

3. **Comparison across levels**:
   ```
   Query: "Why is a product 4th in Lip Care but 73rd in Beauty?"
   Expected: Explains hierarchy structure and competition scope
   ```

4. **ASIN-based query**:
   ```
   Query: "B08XYZ1234 제품은 몇 위인가요?"
   Expected: Shows rankings across all category levels
   ```

## Files Modified

1. **`src/agents/hybrid_chatbot_agent.py`**
   - Added `_build_category_hierarchy_context()` method
   - Modified `_extract_sources()` to include hierarchy source
   - Modified `_generate_response()` to inject hierarchy context

2. **`src/rag/context_builder.py`**
   - Modified `ContextBuilder.build()` to accept knowledge_graph
   - Added `_build_category_hierarchy_section()` method
   - Modified `CompactContextBuilder.build()` for ranking queries

3. **`src/rag/hybrid_retriever.py`**
   - Modified `EntityExtractor.extract()` to extract products
   - Modified `HybridRetriever.retrieve()` to pass knowledge_graph

## Benefits

1. **Contextual Awareness**: Chatbot now understands product belongs to multiple category levels
2. **Accurate Ranking Reporting**: Always specifies which category level when stating ranks
3. **Natural Explanations**: Can explain why rankings differ across hierarchy levels
4. **Better UX**: Users don't need to specify category level - chatbot provides all levels
5. **Data-Driven**: Uses actual knowledge graph data, not hallucinations

## Future Enhancements

1. **Sibling Category Comparison**: "How does this rank in sibling category Lip Makeup?"
2. **Trend Across Levels**: "Is ranking improving faster in subcategory or parent category?"
3. **Competitive Context**: "Who are competitors at each hierarchy level?"
4. **Visual Hierarchy**: Generate hierarchy tree diagrams for complex queries
