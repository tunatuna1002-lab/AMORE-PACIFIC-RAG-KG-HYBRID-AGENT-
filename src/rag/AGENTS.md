# src/rag - Retrieval System

## OVERVIEW

Hybrid retrieval combining vector search (ChromaDB), knowledge graph facts, and ontology reasoning.

## KEY MODULES

| Module | File | Role |
|--------|------|------|
| DocumentRetriever | `retriever.py` | ChromaDB vector search + MD5 embedding cache |
| HybridRetriever | `hybrid_retriever.py` | RAG + KG facts + rule inference |
| TrueHybridRetriever | `true_hybrid_retriever.py` | + OWL reasoning + entity linking + reranking |
| EntityLinker | `entity_linker.py` | Text → KG entity mapping |
| ConfidenceFusion | `confidence_fusion.py` | Multi-source score fusion |

## RETRIEVAL STRATEGIES

| Strategy | Class | When to Use |
|----------|-------|-------------|
| Vector-only | DocumentRetriever | Simple keyword queries |
| Hybrid | HybridRetriever | Queries needing KG context |
| True Hybrid | TrueHybridRetriever | Complex analysis requiring OWL reasoning |

## PIPELINE FLOW

```
HybridRetriever:
  Intent classify → Entity extract → KG lookup → Rule inference
    → Query expand → RAG search → Merge context → HybridContext

TrueHybridRetriever:
  Entity link → Ontology filters → Vector search → OWL reason
    → Rerank → Confidence fusion → HybridResult
```

## DATA STRUCTURES

| Class | File | Purpose |
|-------|------|---------|
| HybridContext | `hybrid_retriever.py` | entities, ontology_facts, inferences, rag_chunks |
| HybridResult | `true_hybrid_retriever.py` | documents, ontology_context, entity_links, confidence |
| LinkedEntity | `entity_linker.py` | text, entity_type, concept_uri, confidence |
| FusedResult | `confidence_fusion.py` | Weighted multi-source fusion |

## CHROMADB CONFIG

- Client: `PersistentClient` at `CHROMA_PERSIST_DIR` (default: `./data/chroma`)
- Collection: `amore_docs`
- Distance: `hnsw:space=cosine`
- Embeddings: `text-embedding-3-small` with MD5 cache (max 1000 FIFO)

## ANTI-PATTERNS

- **NEVER** bypass embedding cache for repeated queries
- **NEVER** skip doc_type filtering when available
- **NEVER** call OpenAI embeddings without cache check first
