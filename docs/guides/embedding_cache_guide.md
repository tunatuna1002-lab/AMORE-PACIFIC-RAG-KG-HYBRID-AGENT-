# Embedding Cache Implementation Guide

## Overview

The `DocumentRetriever` class now includes an embedding cache to prevent redundant API calls to OpenAI's embedding endpoint when the same text is embedded multiple times.

## Implementation Details

### Cache Structure

```python
# Instance variables
self._embedding_cache: dict[str, list[float]] = {}  # MD5 hash -> embedding vector
self._EMBEDDING_CACHE_MAX = 1000                     # Maximum cache size
self._embedding_cache_hits = 0                       # Hit counter
self._embedding_cache_misses = 0                     # Miss counter
```

### Key Features

1. **Hash-based Lookup**: Uses MD5 hash of text as cache key
2. **FIFO Eviction**: When cache is full, removes oldest entry
3. **Hit/Miss Tracking**: Monitors cache performance
4. **Transparent Integration**: No API changes required

## Usage

### Basic Usage

The cache is automatically used when creating embeddings:

```python
from src.rag.retriever import DocumentRetriever

# Initialize retriever
retriever = DocumentRetriever(docs_path="./docs")
await retriever.initialize()

# First search - cache MISS (embeddings created)
results1 = await retriever.search("LANEIGE ranking", top_k=3)

# Second search with same query - cache HIT (embeddings reused)
results2 = await retriever.search("LANEIGE ranking", top_k=3)
```

### Monitoring Cache Performance

```python
# Get cache statistics
stats = retriever.get_embedding_cache_stats()

print(f"Cache size: {stats['size']}")           # Current entries
print(f"Cache hits: {stats['hits']}")           # Successful lookups
print(f"Cache misses: {stats['misses']}")       # New embeddings created
print(f"Hit rate: {stats['hit_rate']:.2%}")     # Efficiency percentage
```

### Example Output

```
Cache size: 42
Cache hits: 18
Cache misses: 24
Hit rate: 42.86%
```

## Performance Benefits

### Before Cache

```
Query 1: "LANEIGE ranking"  → 3 API calls (3 texts to embed)
Query 2: "LANEIGE ranking"  → 3 API calls (same texts, but re-embedded)
Query 3: "SoS meaning"      → 3 API calls
Total: 9 API calls
```

### After Cache

```
Query 1: "LANEIGE ranking"  → 3 API calls (3 cache misses)
Query 2: "LANEIGE ranking"  → 0 API calls (3 cache hits)
Query 3: "SoS meaning"      → 3 API calls (3 cache misses)
Total: 6 API calls (33% reduction)
```

## Cache Behavior

### When Cache Hits

1. Text is hashed (MD5)
2. Hash is found in `_embedding_cache`
3. Cached embedding vector is returned
4. `_embedding_cache_hits` incremented
5. **No API call made**

### When Cache Misses

1. Text is hashed (MD5)
2. Hash is not found in `_embedding_cache`
3. OpenAI Embeddings API called
4. Result stored in cache (with eviction if full)
5. `_embedding_cache_misses` incremented

### Cache Eviction

When cache reaches `_EMBEDDING_CACHE_MAX` (1000 entries):

1. Oldest entry is removed (FIFO strategy)
2. New entry is added

## Testing

Run the test script to verify cache functionality:

```bash
python test_embedding_cache.py
```

Expected output:

```
============================================================
Embedding Cache Test
============================================================

1. Initializing DocumentRetriever...
✅ Retriever initialized

2. Initial cache stats:
   Size: 0
   Hits: 0
   Misses: 0
   Hit Rate: 0.00%

3. First search (cache MISS expected)...
   Query: 'LANEIGE Lip Sleeping Mask 순위'
   Results: 3 chunks
   Cache size: 3
   Cache hits: 0
   Cache misses: 3

4. Second search with same query (cache HIT expected)...
   Query: 'LANEIGE Lip Sleeping Mask 순위'
   Results: 3 chunks
   Cache size: 3
   Cache hits: 1
   Cache misses: 3
   Hit rate: 25.00%

5. Third search with different query (cache MISS expected)...
   Query: 'SoS 지표 의미는 무엇인가요?'
   Results: 3 chunks
   Cache size: 6
   Cache hits: 1
   Cache misses: 6
   Hit rate: 14.29%

============================================================
✅ Embedding Cache Test Complete
============================================================
```

## Configuration

### Adjusting Cache Size

Modify `_EMBEDDING_CACHE_MAX` in `__init__()`:

```python
# Default: 1000 entries
self._EMBEDDING_CACHE_MAX = 1000

# For larger caches (more memory usage)
self._EMBEDDING_CACHE_MAX = 5000

# For smaller caches (less memory usage)
self._EMBEDDING_CACHE_MAX = 500
```

### Memory Considerations

Each embedding vector (text-embedding-3-small):
- Dimensions: 1536 floats
- Memory: ~6 KB per entry
- 1000 entries ≈ 6 MB

## API Reference

### `_get_text_hash(text: str) -> str`

Generates MD5 hash of text for cache key.

**Parameters:**
- `text`: Text to hash

**Returns:**
- Hexadecimal MD5 hash string

### `_embed_texts(texts: list[str]) -> list[list[float]]`

Creates embeddings with caching.

**Parameters:**
- `texts`: List of texts to embed

**Returns:**
- List of embedding vectors (from cache or API)

**Behavior:**
- Checks cache for each text
- Only calls API for cache misses
- Stores new embeddings in cache
- Evicts oldest entry if cache full

### `get_embedding_cache_stats() -> dict`

Returns cache statistics.

**Returns:**
```python
{
    "size": int,        # Current cache size
    "hits": int,        # Total cache hits
    "misses": int,      # Total cache misses
    "hit_rate": float   # hits / (hits + misses)
}
```

## Best Practices

1. **Monitor Hit Rate**: Aim for >30% for significant savings
2. **Adjust Cache Size**: Increase if memory allows
3. **Reset Stats**: Restart application to reset counters
4. **Log Performance**: Track cache stats in production

## Troubleshooting

### Low Hit Rate

**Problem**: Hit rate < 10%

**Causes:**
- Queries are too diverse
- Cache size too small
- Query expansion generating unique variants

**Solutions:**
- Increase `_EMBEDDING_CACHE_MAX`
- Disable query expansion if not needed
- Analyze query patterns

### High Memory Usage

**Problem**: Application using too much RAM

**Causes:**
- Cache size too large

**Solutions:**
- Reduce `_EMBEDDING_CACHE_MAX`
- Monitor cache size with `get_embedding_cache_stats()`

## Future Enhancements

Potential improvements:

1. **LRU Eviction**: Replace FIFO with Least Recently Used
2. **Persistent Cache**: Save cache to disk between runs
3. **TTL Support**: Expire old entries automatically
4. **Cache Warmup**: Pre-populate with common queries
5. **Separate Stats**: Track hits/misses per session

## Related Files

- `src/rag/retriever.py`: Implementation
- `test_embedding_cache.py`: Test script
- `CLAUDE.md`: Project documentation

## Version History

- **2026-01-28**: Initial implementation (v1.0)
  - Hash-based caching
  - FIFO eviction
  - Hit/miss tracking
  - 1000 entry limit
