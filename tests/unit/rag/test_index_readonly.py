"""
색인/조회 분리 (트랙 0-A) 테스트
================================
`DocumentRetriever.initialize()`는 Chroma 컬렉션을 **읽기만** 해야 한다.
쓰기(생성/add/upsert/delete)는 `python -m src.rag.build_index` 전용 경로로
분리한다 (F1).

가짜로 두는 것: OpenAI 임베딩 API 호출뿐이다 (네트워크 금지, 결정적 가짜
벡터로 대체). Chroma는 임시 디렉터리를 가리키는 **실제** PersistentClient로
검증한다 — mock으로 통과시키지 않는다.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.rag.retriever import DocumentRetriever

REPO_ROOT = Path(__file__).resolve().parents[3]


def _fake_embeddings(texts: list[str]) -> list[list[float]]:
    """결정적 가짜 임베딩 — OpenAI 네트워크 호출 없음."""
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        vectors.append([b / 255.0 for b in digest[:8]])
    return vectors


def _patch_embed():
    return patch.object(DocumentRetriever, "_embed_texts", AsyncMock(side_effect=_fake_embeddings))


def _write_doc(docs_path: Path, subdir: str, filename: str, body: str) -> None:
    target_dir = docs_path / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / filename).write_text(body, encoding="utf-8")


@pytest.fixture()
def chroma_dir(tmp_path: Path) -> str:
    return str(tmp_path / "chroma")


@pytest.fixture()
def one_doc_corpus(tmp_path: Path) -> Path:
    """DOCUMENTS 카탈로그의 실제 파일명 하나만 채운 최소 코퍼스 (청크 1개)."""
    docs_path = tmp_path / "docs"
    _write_doc(
        docs_path,
        "guides",
        "Strategic Indicators Definition.md",
        "## Section\nMinimal content for index sync test.\n",
    )
    return docs_path


def _add_second_doc(docs_path: Path) -> None:
    """코퍼스 증가를 흉내 — DOCUMENTS의 다른 카탈로그 항목 파일을 추가."""
    _write_doc(
        docs_path,
        "guides",
        "Metric Interpretation Guide.md",
        "## Section\nSecond document added later for sync test.\n",
    )


# ---------------------------------------------------------------------------
# initialize()는 읽기 전용
# ---------------------------------------------------------------------------


class TestReadOnlyInitialize:
    @pytest.mark.asyncio
    async def test_initialize_raises_clear_error_when_collection_missing(
        self, chroma_dir, one_doc_corpus, monkeypatch
    ):
        """컬렉션이 없으면 build_index 실행을 안내하는 예외를 낸다 (생성하지 않음)."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", chroma_dir)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")

        retriever = DocumentRetriever(docs_path=str(one_doc_corpus))
        with pytest.raises(RuntimeError, match="build_index"):
            await retriever.initialize()

    @pytest.mark.asyncio
    async def test_initialize_after_build_is_idempotent_and_read_only(
        self, chroma_dir, one_doc_corpus, monkeypatch
    ):
        """build_index로 1회 색인 후, initialize()를 반복해도 컬렉션이 그대로다."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", chroma_dir)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")

        from src.rag.build_index import build_index

        with _patch_embed():
            status = await build_index(persist_dir=chroma_dir, docs_path=str(one_doc_corpus))
        assert status["in_sync"] is True
        n = status["indexed_count"]
        assert n == 1

        for _ in range(2):
            retriever = DocumentRetriever(docs_path=str(one_doc_corpus))
            with patch.object(retriever, "_index_documents", AsyncMock()) as spy_index:
                await retriever.initialize()
            spy_index.assert_not_awaited()
            assert retriever.collection.count() == n
            assert retriever.get_index_status()["in_sync"] is True

    @pytest.mark.asyncio
    async def test_two_processes_read_only_initialize_stay_in_sync(
        self, chroma_dir, one_doc_corpus, monkeypatch
    ):
        """서로 다른 프로세스 두 개가 각각 initialize()해도 컬렉션 id 집합이 불변이다."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", chroma_dir)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")

        from src.rag.build_index import build_index

        with _patch_embed():
            status = await build_index(persist_dir=chroma_dir, docs_path=str(one_doc_corpus))
        n = status["indexed_count"]

        script = f"""
import asyncio, os, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
os.environ["CHROMA_PERSIST_DIR"] = {chroma_dir!r}
os.environ["OPENAI_API_KEY"] = "sk-test-dummy"  # pragma: allowlist secret
from src.rag.retriever import DocumentRetriever

async def main():
    r = DocumentRetriever(docs_path={str(one_doc_corpus)!r})
    await r.initialize()

asyncio.run(main())
"""
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, result.stderr

        retriever = DocumentRetriever(docs_path=str(one_doc_corpus))
        await retriever.initialize()
        assert retriever.collection.count() == n
        status_final = retriever.get_index_status()
        assert status_final["in_sync"] is True
        assert status_final["missing_count"] == 0
        assert status_final["extra_count"] == 0


# ---------------------------------------------------------------------------
# 코퍼스 증가 감지 (in_sync=False) — build_index 이후에만 동기화
# ---------------------------------------------------------------------------


class TestDriftDetection:
    @pytest.mark.asyncio
    async def test_growth_detected_but_not_written_until_build(
        self, chroma_dir, one_doc_corpus, monkeypatch
    ):
        monkeypatch.setenv("CHROMA_PERSIST_DIR", chroma_dir)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")

        from src.rag.build_index import build_index

        with _patch_embed():
            status1 = await build_index(persist_dir=chroma_dir, docs_path=str(one_doc_corpus))
        assert status1["in_sync"] is True
        assert status1["indexed_count"] == 1

        # 코퍼스 증가를 흉내
        _add_second_doc(one_doc_corpus)

        retriever = DocumentRetriever(docs_path=str(one_doc_corpus))
        await retriever.initialize()  # 읽기 전용이라 실패하지 않고 경고만
        status2 = retriever.get_index_status()
        assert status2["in_sync"] is False
        assert status2["missing_count"] == 1
        assert retriever.collection.count() == 1  # 쓰지 않았다

        with _patch_embed():
            status3 = await build_index(persist_dir=chroma_dir, docs_path=str(one_doc_corpus))
        assert status3["in_sync"] is True
        assert status3["indexed_count"] == 2


# ---------------------------------------------------------------------------
# build_index CLI 동작 (--dry-run / --prune)
# ---------------------------------------------------------------------------


class TestBuildIndexOptions:
    @pytest.mark.asyncio
    async def test_dry_run_does_not_write(self, chroma_dir, one_doc_corpus, monkeypatch):
        monkeypatch.setenv("CHROMA_PERSIST_DIR", chroma_dir)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")

        from src.rag.build_index import build_index

        status = await build_index(
            persist_dir=chroma_dir, docs_path=str(one_doc_corpus), dry_run=True
        )
        assert status["dry_run"] is True
        assert status["missing_count"] == 1
        assert status["indexed_count"] == 0

        # 컬렉션이 생성되지 않았어야 한다
        retriever = DocumentRetriever(docs_path=str(one_doc_corpus))
        with pytest.raises(RuntimeError, match="build_index"):
            await retriever.initialize()

    @pytest.mark.asyncio
    async def test_prune_removes_orphan_ids(self, chroma_dir, one_doc_corpus, monkeypatch):
        monkeypatch.setenv("CHROMA_PERSIST_DIR", chroma_dir)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")

        from src.rag.build_index import build_index

        with _patch_embed():
            await build_index(persist_dir=chroma_dir, docs_path=str(one_doc_corpus))

        # 코퍼스를 완전히 교체 (기존 문서 삭제 + 새 문서 추가) → orphan 1개
        (one_doc_corpus / "guides" / "Strategic Indicators Definition.md").unlink()
        _add_second_doc(one_doc_corpus)

        with _patch_embed():
            status = await build_index(
                persist_dir=chroma_dir, docs_path=str(one_doc_corpus), prune=True
            )

        assert status["pruned"] == 1
        assert status["in_sync"] is True
        assert status["indexed_count"] == 1
