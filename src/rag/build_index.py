"""
Chroma 벡터 인덱스 빌더 (쓰기 전용 CLI)
=====================================
`DocumentRetriever.initialize()`는 읽기 전용이라 Chroma 컬렉션에 아무것도
쓰지 않는다 (트랙 0-A: 색인/조회 분리). 컬렉션을 만들거나 최신 문서 코퍼스와
동기화하려면 이 모듈을 실행한다.

Usage:
    python -m src.rag.build_index
    python -m src.rag.build_index --persist-dir ./data/chroma
    python -m src.rag.build_index --dry-run       # 쓰지 않고 어긋남만 보고
    python -m src.rag.build_index --prune         # 컬렉션에만 있는 orphan id 삭제
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


async def build_index(
    persist_dir: str | None = None,
    docs_path: str = "./docs",
    dry_run: bool = False,
    prune: bool = False,
    use_semantic_chunking: bool = False,
) -> dict[str, Any]:
    """문서 코퍼스를 로드하고 Chroma 컬렉션을 증분 색인한다.

    Args:
        persist_dir: Chroma persist 디렉터리. None이면 CHROMA_PERSIST_DIR
            환경변수, 그 다음 "./data/chroma" 순으로 기본값을 사용한다.
        docs_path: 문서 루트 경로 (DocumentRetriever와 동일한 규칙).
        dry_run: True면 아무것도 쓰지 않고 현재 어긋남 상태만 계산한다.
            컬렉션이 아직 없어도 생성하지 않는다.
        prune: True면 컬렉션에만 있고 현재 코퍼스에는 없는 id(orphan)를
            삭제한다. 기본값 False (안전).
        use_semantic_chunking: DocumentRetriever와 동일한 옵션 (기본값 동일).

    Returns:
        get_index_status()에 dry_run/wrote/pruned를 더한 상태 딕셔너리.
    """
    import chromadb
    import openai

    from src.rag.retriever import DocumentRetriever

    retriever = DocumentRetriever(docs_path=docs_path, use_semantic_chunking=use_semantic_chunking)
    await retriever._load_documents()

    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    retriever.openai_client = openai.OpenAI(api_key=api_key)
    retriever.embedding_model_name = os.getenv(
        "OPENAI_EMBEDDING_MODEL", retriever.embedding_model_name
    )

    client = chromadb.PersistentClient(path=persist_dir)

    if dry_run:
        # dry-run은 아무것도 만들지 않는다 — 컬렉션이 없으면 없는 대로 집계.
        try:
            collection = client.get_collection(name="amore_docs")
        except Exception:
            collection = None
    else:
        os.makedirs(persist_dir, exist_ok=True)
        collection = client.get_or_create_collection(
            name="amore_docs", metadata={"hnsw:space": "cosine"}
        )

    retriever.collection = collection
    retriever._persist_dir = persist_dir

    status = retriever.get_index_status()

    if dry_run:
        return {"dry_run": True, "wrote": 0, "pruned": 0, **status}

    wrote = status["missing_count"]
    if wrote > 0:
        await retriever._index_documents()

    pruned = 0
    if prune:
        try:
            indexed_ids = set(collection.get(include=[])["ids"])
        except Exception:
            logger.warning("Failed to read indexed ids for prune", exc_info=True)
            indexed_ids = set()

        expected_ids = {chunk["id"] for chunk in retriever.chunks}
        extra_ids = sorted(indexed_ids - expected_ids)
        if extra_ids:
            collection.delete(ids=extra_ids)
            pruned = len(extra_ids)

    final_status = retriever.get_index_status()
    return {"dry_run": False, "wrote": wrote, "pruned": pruned, **final_status}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/refresh the Chroma document index (write path)."
    )
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="Chroma persist directory (default: $CHROMA_PERSIST_DIR or ./data/chroma)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report drift without writing anything",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete collection ids that no longer exist in the source corpus",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)

    status = asyncio.run(
        build_index(
            persist_dir=args.persist_dir,
            dry_run=args.dry_run,
            prune=args.prune,
        )
    )

    for key, value in status.items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
