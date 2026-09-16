"""Vector index
============

The dense leg: availability probing, the ChromaDB collection, OpenAI
embeddings (with the embedding cache in front), incremental indexing and
similarity search.

``DocumentRetriever`` keeps ``collection`` / ``openai_client`` /
``_embedding_cache`` as its own attributes and passes them in, so nothing here
owns retriever state.

Moved verbatim out of ``retriever.py`` (F3 split).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "amore_docs"
INDEX_BATCH_SIZE = 100


def detect_availability() -> bool:
    """벡터 검색 가능 여부 탐지 (OpenAI Embeddings + ChromaDB + API key).

    The *cached* answer lives in ``retriever.VECTOR_SEARCH_AVAILABLE`` — that
    module-level flag is part of the retriever's surface and tests patch it, so
    the caching stays there and this stays a pure probe.
    """
    try:
        import importlib.util

        chromadb_spec = importlib.util.find_spec("chromadb")
        openai_spec = importlib.util.find_spec("openai")

        return bool(chromadb_spec and openai_spec and os.getenv("OPENAI_API_KEY"))
    except ImportError:
        return False
    except Exception:
        return False


def open_collection(embedding_model_name: str | None) -> tuple[Any, Any, Any, str]:
    """OpenAI 클라이언트 + ChromaDB 퍼시스턴트 컬렉션 준비.

    Returns:
        ``(openai_client, chroma_client, collection, embedding_model_name)``

    Raises:
        ImportError: 필수 패키지 미설치 시
        ValueError: OPENAI_API_KEY 미설정 시
    """
    import chromadb
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    openai_client = openai.OpenAI(api_key=api_key)
    model_name = os.getenv(
        "OPENAI_EMBEDDING_MODEL", embedding_model_name or DEFAULT_EMBEDDING_MODEL
    )

    # ChromaDB 초기화 (modern API - persistent client)
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    os.makedirs(persist_dir, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    return openai_client, chroma_client, collection, model_name


def text_hash(text: str) -> str:
    """텍스트 해시 생성"""
    return hashlib.sha256(text.encode()).hexdigest()


async def embed_texts(
    openai_client: Any,
    model: str,
    cache: Any,
    texts: list[str],
) -> list[list[float]]:
    """OpenAI Embeddings API로 텍스트 임베딩 생성 (캐시 적용)"""
    if not openai_client:
        return []

    results: list[Any] = []
    texts_to_embed: list[str] = []
    indices_to_embed: list[int] = []

    # 캐시 확인
    for i, text in enumerate(texts):
        cached = await cache.get(text_hash(text))
        if cached is not None:
            results.append(cached)
        else:
            results.append(None)  # placeholder
            texts_to_embed.append(text)
            indices_to_embed.append(i)

    # 캐시 미스 텍스트만 임베딩
    if texts_to_embed:
        response = openai_client.embeddings.create(model=model, input=texts_to_embed)

        for j, embedding_data in enumerate(response.data):
            embedding = embedding_data.embedding
            results[indices_to_embed[j]] = embedding
            await cache.put(text_hash(texts_to_embed[j]), embedding)

    return results


async def index_documents(collection: Any, openai_client: Any, chunks: list[dict], embed) -> None:
    """문서 벡터 인덱싱 (컬렉션에 없는 청크만 증분 색인).

    기존 조건(``count() == 0``)은 최초 1회만 색인했기 때문에, 나중에 DOCUMENTS에
    추가된 문서(IR 분기보고서 3종 = 1,971청크)가 영원히 색인되지 않고 검색에서
    침묵으로 누락됐다 (2026-08-30 사이클 4 발견).
    """
    if not collection or not openai_client:
        return

    try:
        already_indexed = set(collection.get(include=[])["ids"])
    except Exception:
        logger.warning("Failed to read indexed chunk ids", exc_info=True)
        already_indexed = set()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for chunk in chunks:
        if chunk["id"] in already_indexed:
            continue
        ids.append(chunk["id"])
        documents.append(chunk["content"])
        metadatas.append(
            {
                "doc_id": chunk["doc_id"],
                "doc_type": chunk.get("doc_type", "metric_guide"),
                "title": chunk["title"],
                "description": chunk["description"],
                "content_type": chunk.get("content_type", "text"),
                "source_filename": chunk.get("source_filename", ""),
            }
        )

    if not documents:
        return

    total_batches = (len(documents) - 1) // INDEX_BATCH_SIZE + 1
    for i in range(0, len(documents), INDEX_BATCH_SIZE):
        batch_docs = documents[i : i + INDEX_BATCH_SIZE]
        batch_ids = ids[i : i + INDEX_BATCH_SIZE]
        batch_metadatas = metadatas[i : i + INDEX_BATCH_SIZE]

        try:
            embeddings = await embed(batch_docs)
            if embeddings:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=embeddings,
                    metadatas=batch_metadatas,
                )
                logger.info(f"Indexed batch {i // INDEX_BATCH_SIZE + 1}/{total_batches}")
        except Exception as e:
            logger.error(f"Indexing batch failed: {e}")


def build_where_filter(doc_filter: str | None, doc_type_filter: list[str] | None) -> dict | None:
    """Chroma ``where`` 절 구성."""
    if doc_filter and doc_type_filter:
        return {"$and": [{"doc_id": doc_filter}, {"doc_type": {"$in": doc_type_filter}}]}
    if doc_filter:
        return {"doc_id": doc_filter}
    if doc_type_filter:
        return {"doc_type": {"$in": doc_type_filter}}
    return None


def format_hits(
    results: dict,
    chunk_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Chroma 질의 결과 → 검색 결과 dict 목록 (거리를 유사도로 변환)."""
    search_results: list[dict[str, Any]] = []
    if not (results["ids"] and results["ids"][0]):
        return search_results

    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        chunk = chunk_index.get(chunk_id)
        if chunk:
            metadata = {
                "doc_id": chunk["doc_id"],
                "doc_type": chunk.get("doc_type", "metric_guide"),
                "title": chunk.get("title", ""),
                "description": chunk.get("description", ""),
                "keywords": chunk.get("keywords", []),
                "content_type": chunk.get("content_type", "text"),
                "chunk_id": chunk_id,
                "source_filename": chunk.get("source_filename", ""),
                "target_brand": chunk.get("target_brand"),
                "brands_covered": chunk.get("brands_covered", []),
            }
        else:
            # Chroma 메타데이터 폴백 — chunk_id를 보존해야
            # 리랭킹 후 재조립(search)에서 id가 유실되지 않는다
            metadata = dict(results["metadatas"][0][i] or {})
            metadata.setdefault("chunk_id", chunk_id)

        search_results.append(
            {
                "id": chunk_id,
                "content": results["documents"][0][i],
                "metadata": metadata,
                "score": 1 - results["distances"][0][i],  # 거리를 유사도로 변환
            }
        )

    return search_results
