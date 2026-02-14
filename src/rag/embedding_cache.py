"""Embedding cache with Protocol interface and two backends.

- InMemoryEmbeddingCache: dict-based (existing behavior, wrapped in protocol)
- SQLiteEmbeddingCache: aiosqlite-backed, LRU eviction, max 10,000 entries
"""

from __future__ import annotations

import struct
import time
from typing import Protocol, runtime_checkable

import aiosqlite


@runtime_checkable
class EmbeddingCacheProtocol(Protocol):
    """Protocol for embedding cache backends."""

    async def get(self, key: str) -> list[float] | None:
        """Retrieve embedding from cache.

        Args:
            key: Cache key (text hash)

        Returns:
            Embedding vector or None if not found
        """
        ...

    async def put(self, key: str, embedding: list[float]) -> None:
        """Store embedding in cache.

        Args:
            key: Cache key (text hash)
            embedding: Embedding vector
        """
        ...

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with keys: size, max_size, hits, misses, hit_rate (optional)
        """
        ...

    async def close(self) -> None:
        """Clean up cache resources."""
        ...


class InMemoryEmbeddingCache:
    """Dict-based cache compatible with existing behavior. Max 1000, FIFO."""

    def __init__(self, max_size: int = 1000):
        """Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries (default: 1000)
        """
        self._cache: dict[str, list[float]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> list[float] | None:
        """Retrieve embedding from cache.

        Args:
            key: Cache key

        Returns:
            Embedding vector or None if not found
        """
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    async def put(self, key: str, embedding: list[float]) -> None:
        """Store embedding in cache with FIFO eviction.

        Args:
            key: Cache key
            embedding: Embedding vector
        """
        if len(self._cache) >= self._max_size:
            # FIFO: remove oldest (first) entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = embedding

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with size, max_size, hits, misses
        """
        total_requests = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, total_requests),
        }

    async def close(self) -> None:
        """Clear cache."""
        self._cache.clear()


class SQLiteEmbeddingCache:
    """SQLite-backed cache with LRU eviction and binary storage.

    - Uses struct.pack for efficient float storage
    - LRU eviction at max 10,000 entries
    - Persists across restarts
    """

    def __init__(self, db_path: str = "data/embedding_cache.db", max_size: int = 10000):
        """Initialize SQLite cache.

        Args:
            db_path: Path to SQLite database file
            max_size: Maximum number of entries (default: 10000)
        """
        self._db_path = db_path
        self._max_size = max_size
        self._db: aiosqlite.Connection | None = None
        self._hits = 0
        self._misses = 0

    async def _ensure_db(self) -> None:
        """Lazy init - create connection and table if needed."""
        if self._db is not None:
            return

        # Ensure parent directory exists
        import os

        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)

        # Create connection
        self._db = await aiosqlite.connect(self._db_path)

        # Create table
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                last_accessed REAL NOT NULL
            )
            """
        )

        # Create index for LRU eviction
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_last_accessed
            ON embeddings(last_accessed)
            """
        )

        await self._db.commit()

    async def get(self, key: str) -> list[float] | None:
        """Retrieve embedding from cache and update access time.

        Args:
            key: Cache key

        Returns:
            Embedding vector or None if not found
        """
        await self._ensure_db()

        if self._db is None:
            return None

        cursor = await self._db.execute(
            "SELECT embedding, dim FROM embeddings WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()

        if row is None:
            self._misses += 1
            return None

        # Update last_accessed timestamp
        await self._db.execute(
            "UPDATE embeddings SET last_accessed = ? WHERE key = ?", (time.time(), key)
        )
        await self._db.commit()

        # Deserialize
        embedding_blob, dim = row
        embedding = self._deserialize(embedding_blob, dim)
        self._hits += 1
        return embedding

    async def put(self, key: str, embedding: list[float]) -> None:
        """Store embedding in cache with LRU eviction.

        Args:
            key: Cache key
            embedding: Embedding vector
        """
        await self._ensure_db()

        if self._db is None:
            return

        # Serialize
        embedding_blob = self._serialize(embedding)
        dim = len(embedding)

        # INSERT OR REPLACE
        await self._db.execute(
            """
            INSERT OR REPLACE INTO embeddings (key, embedding, dim, last_accessed)
            VALUES (?, ?, ?, ?)
            """,
            (key, embedding_blob, dim, time.time()),
        )

        # Check if over max_size and evict oldest if needed
        cursor = await self._db.execute("SELECT COUNT(*) FROM embeddings")
        count_row = await cursor.fetchone()
        count = count_row[0] if count_row else 0

        if count > self._max_size:
            # Delete oldest entries (LRU)
            to_delete = count - self._max_size
            await self._db.execute(
                """
                DELETE FROM embeddings
                WHERE key IN (
                    SELECT key FROM embeddings
                    ORDER BY last_accessed ASC
                    LIMIT ?
                )
                """,
                (to_delete,),
            )

        await self._db.commit()

    @staticmethod
    def _serialize(embedding: list[float]) -> bytes:
        """Serialize embedding vector to binary format.

        Args:
            embedding: Embedding vector

        Returns:
            Binary representation
        """
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _deserialize(data: bytes, dim: int) -> list[float]:
        """Deserialize embedding vector from binary format.

        Args:
            data: Binary data
            dim: Embedding dimension

        Returns:
            Embedding vector
        """
        return list(struct.unpack(f"{dim}f", data))

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics (synchronous for compatibility).

        Note: Returns current stats. Size query requires async access.

        Returns:
            Dict with hits, misses, max_size
        """
        total_requests = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "max_size": self._max_size,
            "hit_rate": self._hits / max(1, total_requests),
        }

    async def close(self) -> None:
        """Close database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
