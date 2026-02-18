# src/rag - Retrieval System

## OVERVIEW

Hybrid retrieval combining vector search (ChromaDB), knowledge graph facts, and ontology reasoning.

## KEY MODULES

| Module | File | Role |
|--------|------|------|
| DocumentRetriever | `retriever.py` | ChromaDB vector search + SHA-256 embedding cache |
| HybridRetriever | `hybrid_retriever.py` | RAG + KG facts + rule inference + OWL strategy |
| RetrievalStrategy | `retrieval_strategy.py` | OWL + intent-based strategy pattern |
| EntityLinker | `entity_linker.py` | Text → KG entity mapping (unified extractor) |
| ConfidenceFusion | `confidence_fusion.py` | Multi-source score fusion + conflict detection |

## RETRIEVAL STRATEGIES

| Strategy | Class | When to Use |
|----------|-------|-------------|
| Vector-only | DocumentRetriever | Simple keyword queries |
| Hybrid | HybridRetriever | Queries needing KG context |
| OWL Strategy | OWLRetrievalStrategy | Complex analysis requiring OWL reasoning |

Intent-based config (`IntentRetrievalConfig`) tunes weights, top_k, doc_type_filter, and fusion_strategy per query intent.

## PIPELINE FLOW

```
HybridRetriever.retrieve():
  Intent classify → Entity extract → KG lookup → Rule inference
    → Query expand → RAG search → Relevance grading
    → Weighted merge (intent-based weights)
    → ConfidenceFusion (overall confidence + conflict detection)
    → Combine contexts → HybridContext
```

## DATA STRUCTURES

| Class | File | Purpose |
|-------|------|---------|
| HybridContext | `hybrid_retriever.py` | entities, ontology_facts, inferences, rag_chunks |
| IntentRetrievalConfig | `retrieval_strategy.py` | weights, top_k, doc_type_filter, fusion_strategy |
| LinkedEntity | `entity_linker.py` | text, entity_type, concept_uri, confidence |
| FusedResult | `confidence_fusion.py` | Weighted multi-source fusion with conflict detection |

## CHROMADB CONFIG

- Client: `PersistentClient` at `CHROMA_PERSIST_DIR` (default: `./data/chroma`)
- Collection: `amore_docs`
- Distance: `hnsw:space=cosine`
- Embeddings: `text-embedding-3-small` with SHA-256 cache (max 1000 FIFO)

## ANTI-PATTERNS

- **NEVER** bypass embedding cache for repeated queries
- **NEVER** skip doc_type filtering when available
- **NEVER** call OpenAI embeddings without cache check first
