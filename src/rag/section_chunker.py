"""Section chunker
===============

Splits a markdown document into the chunks that get embedded and indexed:
tables become standalone chunks (so a table is never cut in half), the rest is
split on ``## `` sections and long sections are further split on paragraph then
sentence boundaries.

Chunk ids are positional and stable for a given document text:
``<doc_id>_table_<t>`` for tables, ``<doc_id>_<section>`` for whole sections and
``<doc_id>_<section>_<part>`` for split ones.

Moved verbatim out of ``retriever.py`` (F3 split).
"""

from __future__ import annotations

import re
from typing import Any

# 문서 유형별 청크 크기
CHUNK_SIZES: dict[str, int] = {
    "playbook": 800,  # Type A: 분석 플레이북 - 큰 청크
    "intelligence": 600,  # Type B: 시장 인텔리전스
    "knowledge_base": 600,  # Type B: 지식 베이스
    "response_guide": 500,  # Type C: 대응 가이드
    "metric_guide": 500,  # Type D: 기존 지표 가이드
    "ir_report": 700,  # Type E: IR 분기 실적 - 테이블 포함 큰 청크
}
DEFAULT_CHUNK_SIZE = 500

# 마크다운 표 (헤더 + 구분선 + 본문 행)
TABLE_PATTERN = r"(\|[^\n]+\|\n(?:\|[-:| ]+\|\n)?(?:\|[^\n]+\|\n)+)"


def chunk_size_for(doc_type: str) -> int:
    """문서 유형별 청크 크기 반환"""
    return CHUNK_SIZES.get(doc_type, DEFAULT_CHUNK_SIZE)


def split_into_chunks(
    content: str, doc_id: str, doc_info: dict, chunk_size: int = 500
) -> list[dict[str, Any]]:
    """
    문서를 청크로 분할

    - 표(Table)는 별도 청크로 분리하여 완전성 유지
    - 섹션 기반 분할 후 크기 초과 시 추가 분할
    """
    chunks = []
    doc_type = doc_info.get("doc_type", "metric_guide")
    source_filename = doc_info.get("filename", "")
    target_brand = doc_info.get("target_brand")
    brands_covered = doc_info.get("brands_covered", [])

    # 1. 표(Table) 추출 및 별도 청크 생성
    table_pattern = TABLE_PATTERN
    tables = re.findall(table_pattern, content)

    for t_idx, table in enumerate(tables):
        table_text = table.strip()
        if table_text:
            # 표 주변 컨텍스트 찾기 (표 바로 위의 제목)
            table_pos = content.find(table)
            context_before = content[:table_pos].strip()
            lines_before = context_before.split("\n")

            # 표 제목 추출 (### 또는 **로 시작하는 마지막 라인)
            table_title = ""
            for line in reversed(lines_before[-5:]):
                line_stripped = line.strip()
                if line_stripped.startswith("#") or line_stripped.startswith("**"):
                    table_title = line_stripped.replace("#", "").replace("*", "").strip()
                    break

            chunks.append(
                {
                    "id": f"{doc_id}_table_{t_idx}",
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "title": table_title or f"Table {t_idx + 1}",
                    "content": table_text,
                    "content_type": "table",
                    "source_filename": source_filename,
                    "target_brand": target_brand,
                    "brands_covered": brands_covered,
                    "keywords": doc_info["keywords"],
                    "description": doc_info["description"],
                }
            )

    # 2. 표를 플레이스홀더로 대체한 후 섹션 분할
    content_without_tables = re.sub(table_pattern, "\n[TABLE]\n", content)
    sections = content_without_tables.split("\n## ")

    for i, section in enumerate(sections):
        if not section.strip():
            continue

        # [TABLE] 플레이스홀더만 있는 섹션은 스킵
        if section.strip() == "[TABLE]":
            continue

        # 섹션 제목 추출
        lines = section.split("\n")
        title = lines[0].replace("#", "").strip() if lines else ""

        # 청크 생성
        text = section.strip()

        # [TABLE] 플레이스홀더 제거
        text = re.sub(r"\n*\[TABLE\]\n*", "\n", text).strip()

        if not text:
            continue

        if len(text) > chunk_size:
            # 긴 섹션은 추가 분할
            sub_chunks = smart_split(text, chunk_size)
            for k, sub_chunk in enumerate(sub_chunks):
                if sub_chunk.strip():
                    chunks.append(
                        {
                            "id": f"{doc_id}_{i}_{k}",
                            "doc_id": doc_id,
                            "doc_type": doc_type,
                            "title": title,
                            "content": sub_chunk,
                            "content_type": "text",
                            "source_filename": source_filename,
                            "target_brand": target_brand,
                            "brands_covered": brands_covered,
                            "keywords": doc_info["keywords"],
                            "description": doc_info["description"],
                        }
                    )
        else:
            chunks.append(
                {
                    "id": f"{doc_id}_{i}",
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "title": title,
                    "content": text,
                    "content_type": "text",
                    "source_filename": source_filename,
                    "target_brand": target_brand,
                    "brands_covered": brands_covered,
                    "keywords": doc_info["keywords"],
                    "description": doc_info["description"],
                }
            )

    return chunks


def smart_split(text: str, chunk_size: int) -> list[str]:
    """
    텍스트를 의미 단위로 분할

    - 단락(\n\n) 기준으로 우선 분할
    - 단락이 chunk_size보다 크면 문장 단위로 분할
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)

            if len(para) <= chunk_size:
                current_chunk = para
            else:
                # 단락이 chunk_size보다 크면 강제 분할
                for j in range(0, len(para), chunk_size):
                    chunks.append(para[j : j + chunk_size])
                current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
