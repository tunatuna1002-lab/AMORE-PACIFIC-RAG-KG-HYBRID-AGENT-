"""
Chroma 벡터 인덱스 빌더 (쓰기 전용 CLI)
=====================================
`DocumentRetriever.initialize()`는 읽기 전용이라 Chroma 컬렉션에 아무것도
쓰지 않는다 (트랙 0-A: 색인/조회 분리). 컬렉션을 만들거나 최신 문서 코퍼스와
동기화하려면 이 모듈을 실행한다.

새로 색인하는 청크에는 엔티티 태그(브랜드·카테고리·지표 canonical id)를 메타데이터로
함께 저장한다 (트랙 4-B, 설계 E9). 이미 색인된 컬렉션은 `--retag`로 **재임베딩 없이**
메타데이터만 갱신한다 — 임베딩 API를 부르지 않으므로 비용이 0이고 OPENAI_API_KEY도
필요 없다.

Usage:
    python -m src.rag.build_index
    python -m src.rag.build_index --persist-dir ./data/chroma
    python -m src.rag.build_index --dry-run       # 쓰지 않고 어긋남만 보고
    python -m src.rag.build_index --prune         # 컬렉션에만 있는 orphan id 삭제
    python -m src.rag.build_index --retag         # 기존 id의 엔티티 태그만 갱신
    python -m src.rag.build_index --retag --dry-run  # 태그 커버리지만 계산 (쓰기 없음)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "amore_docs"
_RETAG_BATCH_SIZE = 100


def _retag_collection(collection: Any, dry_run: bool) -> dict[str, Any]:
    """컬렉션의 모든 id에 엔티티 태그 메타데이터를 다시 계산해 갱신한다.

    태그는 Chroma에 저장된 문서 본문 + 메타데이터 `title`에서 뽑는다 — 현재 코퍼스
    파일과 무관하게 컬렉션 자체만으로 동작한다 (동결 평가 스냅샷 복사본 태깅용).
    `collection.update(ids, metadatas)`만 호출하고 documents·embeddings는 넘기지 않으므로
    재임베딩이 일어나지 않는다. 기존 메타데이터 키는 모두 보존한다.
    """
    from src.rag.entity_tags import build_tag_metadata, read_tags

    got = collection.get(include=["documents", "metadatas"])
    ids: list[str] = list(got["ids"])
    documents = got["documents"] or [""] * len(ids)
    metadatas = got["metadatas"] or [{}] * len(ids)

    coverage = {"total": len(ids), "brand": 0, "category": 0, "metric": 0, "brand_or_category": 0}
    new_metadatas: list[dict[str, Any]] = []
    changed_ids: list[str] = []
    changed_metadatas: list[dict[str, Any]] = []

    for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=True):
        metadata = dict(metadata or {})
        merged = {**metadata, **build_tag_metadata(metadata.get("title", ""), document or "")}
        tags = read_tags(merged) or {}
        coverage["brand"] += bool(tags.get("brands"))
        coverage["category"] += bool(tags.get("categories"))
        coverage["metric"] += bool(tags.get("metrics"))
        coverage["brand_or_category"] += bool(tags.get("brands") or tags.get("categories"))
        new_metadatas.append(merged)
        if merged != metadata:
            changed_ids.append(chunk_id)
            changed_metadatas.append(merged)

    retagged = 0
    if not dry_run:
        for i in range(0, len(changed_ids), _RETAG_BATCH_SIZE):
            collection.update(
                ids=changed_ids[i : i + _RETAG_BATCH_SIZE],
                metadatas=changed_metadatas[i : i + _RETAG_BATCH_SIZE],
            )
        retagged = len(changed_ids)

    return {
        "retag": True,
        "retagged": retagged,
        "retag_pending": len(changed_ids),
        "tag_coverage": coverage,
    }


async def build_index(
    persist_dir: str | None = None,
    docs_path: str = "./docs",
    dry_run: bool = False,
    prune: bool = False,
    use_semantic_chunking: bool = False,
    retag: bool = False,
) -> dict[str, Any]:
    """문서 코퍼스를 로드하고 Chroma 컬렉션을 증분 색인한다.

    Args:
        persist_dir: Chroma persist 디렉터리. None이면 CHROMA_PERSIST_DIR
            환경변수, 그 다음 "./data/chroma" 순으로 기본값을 사용한다.
        docs_path: 문서 루트 경로 (DocumentRetriever와 동일한 규칙).
        dry_run: True면 아무것도 쓰지 않고 현재 어긋남 상태만 계산한다.
            컬렉션이 아직 없어도 생성하지 않는다. 임베딩을 하지 않으므로
            OPENAI_API_KEY가 없어도 된다.
        prune: True면 컬렉션에만 있고 현재 코퍼스에는 없는 id(orphan)를
            삭제한다. 기본값 False (안전).
        use_semantic_chunking: DocumentRetriever와 동일한 옵션 (기본값 동일).
        retag: True면 새 청크를 색인하지 않고, 기존 id 전부의 엔티티 태그
            메타데이터만 갱신한다 (재임베딩 없음, API 키 불필요). `dry_run`과 함께
            쓰면 태그 커버리지만 계산한다. 컬렉션이 없으면 RuntimeError.

    Returns:
        get_index_status()에 dry_run/wrote/pruned(retag면 retagged/tag_coverage)를 더한
        상태 딕셔너리.
    """
    import chromadb

    from src.rag.retriever import DocumentRetriever

    retriever = DocumentRetriever(docs_path=docs_path, use_semantic_chunking=use_semantic_chunking)
    await retriever._load_documents()

    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    client = chromadb.PersistentClient(path=persist_dir)
    retriever._persist_dir = persist_dir

    if retag:
        try:
            collection = client.get_collection(name=_COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Chroma collection '{_COLLECTION_NAME}' not found at '{persist_dir}'. "
                "--retag only updates an existing index."
            ) from e
        retriever.collection = collection
        retag_status = _retag_collection(collection, dry_run=dry_run)
        return {"dry_run": dry_run, **retag_status, **retriever.get_index_status()}

    if not dry_run:
        # 실제 색인(쓰기)만 임베딩 API를 쓴다
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        retriever.openai_client = openai.OpenAI(api_key=api_key)
        retriever.embedding_model_name = os.getenv(
            "OPENAI_EMBEDDING_MODEL", retriever.embedding_model_name
        )

    if dry_run:
        # dry-run은 아무것도 만들지 않는다 — 컬렉션이 없으면 없는 대로 집계.
        try:
            collection = client.get_collection(name=_COLLECTION_NAME)
        except Exception:
            collection = None
    else:
        os.makedirs(persist_dir, exist_ok=True)
        collection = client.get_or_create_collection(
            name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    retriever.collection = collection

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
    if not final_status["tagged"] and final_status["indexed_count"]:
        logger.warning(
            "%d/%d indexed chunks lack entity tags — run "
            "`python -m src.rag.build_index --retag` (metadata only, no re-embedding)",
            final_status["indexed_count"] - final_status["tagged_count"],
            final_status["indexed_count"],
        )
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
        "--docs-path",
        default="./docs",
        help="Document corpus root (default: ./docs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report drift (or tag coverage with --retag) without writing anything",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete collection ids that no longer exist in the source corpus",
    )
    parser.add_argument(
        "--retag",
        action="store_true",
        help="Update entity tag metadata of existing ids only (no re-embedding, no API key)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)

    status = asyncio.run(
        build_index(
            persist_dir=args.persist_dir,
            docs_path=args.docs_path,
            dry_run=args.dry_run,
            prune=args.prune,
            retag=args.retag,
        )
    )

    for key, value in status.items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
