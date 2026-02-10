# src/ontology - Knowledge Management

## OVERVIEW

Triple store knowledge graph with rule-based and OWL reasoning for brand/product/category relationships.

## KEY MODULES

| Module | File | Role |
|--------|------|------|
| KnowledgeGraph | `knowledge_graph.py` | In-memory triple store + JSON persistence |
| Reasoner | `reasoner.py` | Rule-based forward-chaining inference |
| OWLReasoner | `owl_reasoner.py` | OWL 2 reasoning via owlready2 |

## REASONING APPROACHES

| Approach | Class | When to Use |
|----------|-------|-------------|
| Rule-based | Reasoner | Fast inference, known patterns |
| OWL | OWLReasoner | Complex ontology queries, class hierarchies |

## DATA MODEL

### Triple Structure
```python
Relation(subject, predicate, object)
# Example: Relation("LANEIGE", "competes_with", "COSRX")
```

### KG Indices
- `subject_index`: subject → [relations]
- `object_index`: object → [relations]
- `predicate_index`: predicate → [relations]
- `entity_metadata`: entity → {type, attributes}

### Persistence
- Format: JSON with triples + metadata
- Location: `data/knowledge_graph.json`
- Backup: `data/backups/kg/` (7-day rolling)

## RULE ENGINE

```python
RuleCondition(subject_type, predicate, object_type)
InferenceRule(name, conditions, conclusion, confidence)
InferenceResult(conclusion, evidence, confidence, recommendation)
```

QueryIntentDetector filters applicable rules by query intent.

## OWL CLASSES

- `Brand`: name, origin, segment
- `Product`: name, brand, category, price
- `Category`: name, parent, level

Reasoners: Pellet (default), HermiT (fallback)

## ANTI-PATTERNS

- **NEVER** modify KG without triggering backup check
- **NEVER** add triples without entity_metadata update
- **NEVER** use OWL reasoner for simple lookups (too slow)
- **NEVER** bypass importance scores for eviction
