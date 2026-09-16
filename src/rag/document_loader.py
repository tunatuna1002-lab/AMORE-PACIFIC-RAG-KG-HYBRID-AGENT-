"""Document loader
===============

Reads the markdown corpus off disk: finds each registered document in the
folders the registry lists, strips the base64 image payloads the PDF→markdown
conversion left behind, and hands the text to the chunker.

Moved verbatim out of ``retriever.py`` (F3 split). ``strip_embedded_binaries``
and ``_has_hangul`` are re-exported from ``retriever`` for existing importers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .document_registry import candidate_paths
from .section_chunker import chunk_size_for, split_into_chunks

_HANGUL_RE = re.compile(r"[가-힣]")


def has_hangul(text: str) -> bool:
    """한글이 포함된 질의인지 — 교차언어 확장 방향 결정용."""
    return bool(_HANGUL_RE.search(text or ""))


# PDF→마크다운 변환 산출물에 남는 이미지 data URI 링크 정의
#   `[image1]: <data:image/png;base64,iVBORw0KGgo...>`
_DATA_URI_DEF_RE = re.compile(r"^\[[^\]]+\]:\s*<?data:[^>\n]+>?\s*$", re.MULTILINE)
# 위 형태에 들어맞지 않는 잔여 base64 덩어리 (200자 이상 연속)
_LONG_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")


def strip_embedded_binaries(content: str) -> str:
    """문서에 인라인된 base64 이미지 데이터를 제거한다.

    IR 분기보고서 3종은 PDF→마크다운 변환물이라 본문 끝에 이미지가 data URI로
    통째로 박혀 있고, 그 분량이 파일의 97%에 달한다. 이를 그대로 청킹하면
    2,242청크 중 **1,877청크(84%)가 base64 덩어리**가 되어 임베딩 비용을 쓰고
    LLM 컨텍스트에 잡음으로 들어간다 (2026-08-30 사이클 6 실측).
    실제 IR 본문은 파일당 약 1.5만 자다.
    """
    cleaned = _DATA_URI_DEF_RE.sub("", content)
    return _LONG_BASE64_RE.sub("", cleaned)


def load_documents(
    docs_path: Path,
    registry: dict[str, dict[str, Any]],
    semantic_chunker: Any | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """MD 문서 로드 + 청킹.

    Args:
        docs_path: 문서 루트 폴더 (``./docs``)
        registry: 문서 메타데이터 테이블 (``document_registry.DOCUMENTS``)
        semantic_chunker: SemanticChunker 인스턴스 (없으면 섹션 청커 사용)

    Returns:
        ``(documents, chunks)`` — 문서 id → 본문, 그리고 전체 청크 리스트
    """
    documents: dict[str, str] = {}
    chunks: list[dict[str, Any]] = []

    for doc_id, doc_info in registry.items():
        for file_path in candidate_paths(docs_path, doc_info["filename"]):
            if not file_path.exists():
                continue
            with open(file_path, encoding="utf-8") as f:
                content = strip_embedded_binaries(f.read())
            documents[doc_id] = content
            chunks.extend(_chunk_document(content, doc_id, doc_info, semantic_chunker))
            break

    return documents, chunks


def _chunk_document(
    content: str,
    doc_id: str,
    doc_info: dict[str, Any],
    semantic_chunker: Any | None,
) -> list[dict[str, Any]]:
    """한 문서를 청크로 자른다 (Semantic Chunking 옵션)."""
    if semantic_chunker:
        return semantic_chunker.chunk_document(content, doc_info)
    doc_type = doc_info.get("doc_type", "metric_guide")
    return split_into_chunks(content, doc_id, doc_info, chunk_size_for(doc_type))


def index_chunks(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """청크 인덱스 (벡터 검색 결과 메타데이터 보강용)."""
    return {chunk["id"]: chunk for chunk in chunks}
