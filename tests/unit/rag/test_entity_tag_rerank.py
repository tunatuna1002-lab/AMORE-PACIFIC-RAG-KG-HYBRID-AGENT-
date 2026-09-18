"""
색인 엔티티 태그 + 온톨로지 재정렬 보너스 (트랙 4-B, 설계 E9·E10, 결함 F9)
=========================================================================

가짜로 두는 것: OpenAI 임베딩 API 호출뿐이다 (결정적 키워드 벡터로 대체). Chroma는
임시 디렉터리의 **실제** PersistentClient, 검색은 실제 DocumentRetriever(BM25 포함)와
실제 HybridRetriever._hybrid_search/retrieve로 검증한다. KG·추론기·DB 수치 제공자만
문서 검색과 무관하므로 가짜로 둔다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import chromadb
import pytest

from src.rag.entity_tags import (
    TAG_VERSION_KEY,
    apply_entity_bonus,
    build_tag_metadata,
    decode_tags,
    encode_tags,
    read_tags,
)
from src.rag.retriever import DocumentRetriever

QUERY = "LANEIGE Lip Care market position"
TOP_K = 5
MATCHING_CHUNK = "strategic_indicators_3"  # 한국어 별칭으로만 LANEIGE·Lip Care를 언급

# 한국어 별칭 청크는 영어 질의와 BM25 토큰이 겹치지 않는다 — 온톨로지 canonical id
# (라네즈→laneige, 립케어→lip_care)만이 질의와 연결해 준다.
CORPUS = (
    "## Lip balm routine\nlip moisture and body care tips for winter\n"
    "\n## Lip gloss shine\nlip gloss shine and hair care basics\n"
    "\n## Care instructions\ncare instructions for lip liner storage\n"
    "\n## 라네즈 립케어 전략\n라네즈 립케어 립 슬리핑 마스크 position 메모\n"
    "\n## COSRX serum\nCOSRX snail serum review\n"
    "\n## Concentration metric\nmarket concentration index overview\n"
)

_KEYWORDS = ("lip", "care", "laneige", "market", "position", "라네즈", "립케어", "cosrx")


def _fake_embeddings(texts: list[str]) -> list[list[float]]:
    """결정적 키워드 벡터 — OpenAI 네트워크 호출 없음."""
    vectors = []
    for text in texts:
        lowered = text.lower()
        vectors.append([1.0] + [1.0 if kw in lowered else 0.0 for kw in _KEYWORDS])
    return vectors


def _clear_search_cache() -> None:
    """검색 캐시는 클래스 속성(인스턴스·코퍼스와 무관) — 같은 질의를 두 번 쓰기 전에 비운다."""
    DocumentRetriever._search_cache.clear()
    DocumentRetriever._cache_timestamps.clear()


def _patch_embed():
    return patch.object(DocumentRetriever, "_embed_texts", AsyncMock(side_effect=_fake_embeddings))


@pytest.fixture(autouse=True)
def _isolated_search_cache():
    """DocumentRetriever 검색 캐시는 클래스 속성이라 테스트 간에 새지 않게 비운다."""
    _clear_search_cache()
    yield
    _clear_search_cache()


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    docs_path = tmp_path / "docs"
    (docs_path / "guides").mkdir(parents=True)
    (docs_path / "guides" / "Strategic Indicators Definition.md").write_text(
        CORPUS, encoding="utf-8"
    )
    return docs_path


@pytest.fixture()
def chroma_dir(tmp_path: Path, monkeypatch) -> str:
    path = str(tmp_path / "chroma")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")  # pragma: allowlist secret
    return path


async def _build_untagged_index(chroma_dir: str, corpus: Path) -> None:
    """트랙 4-B 이전 형식(태그 없는 메타데이터)의 색인 — 동결 평가 스냅샷과 같은 모양."""
    retriever = DocumentRetriever(docs_path=str(corpus))
    await retriever._load_documents()
    collection = chromadb.PersistentClient(path=chroma_dir).get_or_create_collection(
        name="amore_docs", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[c["id"] for c in retriever.chunks],
        documents=[c["content"] for c in retriever.chunks],
        embeddings=_fake_embeddings([c["content"] for c in retriever.chunks]),
        metadatas=[
            {
                "doc_id": c["doc_id"],
                "doc_type": c["doc_type"],
                "title": c["title"],
                "description": c["description"],
                "content_type": c["content_type"],
                "source_filename": c["source_filename"],
            }
            for c in retriever.chunks
        ],
    )


async def _build_tagged_index(chroma_dir: str, corpus: Path) -> dict[str, Any]:
    from src.rag.build_index import build_index

    with _patch_embed():
        return await build_index(persist_dir=chroma_dir, docs_path=str(corpus))


def _hybrid(doc_retriever: DocumentRetriever):
    from src.rag.hybrid_retriever import HybridRetriever

    kg = MagicMock()
    kg.get_entity_metadata.return_value = {}
    kg.get_brand_products.return_value = []
    kg.get_competitors.return_value = []
    kg.get_neighbors.return_value = {"outgoing": [], "incoming": []}
    kg.query.return_value = []
    kg.get_category_brands.return_value = []
    kg.get_category_hierarchy.return_value = {}
    kg.get_product_sentiments.return_value = {}
    kg.get_brand_sentiment_profile.return_value = {}
    kg.find_products_by_sentiment.return_value = []
    kg.load_category_hierarchy.return_value = 0
    reasoner = MagicMock()
    reasoner.rules = ["dummy"]
    metric_facts = MagicMock()
    metric_facts.collect = AsyncMock(return_value=[])
    hybrid = HybridRetriever(
        knowledge_graph=kg,
        reasoner=reasoner,
        doc_retriever=doc_retriever,
        auto_init_rules=False,
        metric_facts_provider=metric_facts,
    )
    return hybrid


def _ids(results: list[dict[str, Any]]) -> list[str]:
    from src.rag.entity_tags import result_chunk_id

    return [result_chunk_id(r) for r in results]


# ---------------------------------------------------------------------------
# 태그 형식
# ---------------------------------------------------------------------------


class TestTagFormat:
    def test_encode_decode_roundtrip_with_exact_substring_match(self):
        encoded = encode_tags(["LANEIGE", "cosrx", "laneige", ""])
        assert encoded == "|cosrx|laneige|"
        assert decode_tags(encoded) == ["cosrx", "laneige"]
        assert "|laneige|" in encoded
        assert "|lane|" not in encoded  # 부분 문자열 오탐 없음
        assert encode_tags([]) == ""
        assert decode_tags("") == []
        assert decode_tags(None) == []

    def test_tags_use_evidence_adapter_canonical_ids(self):
        from src.rag.evidence_adapters import EvidenceAdapter

        adapter = EvidenceAdapter()
        meta = build_tag_metadata("라네즈 립케어", "LANEIGE Lip Sleeping Mask SoS 분석")
        tags = read_tags(meta)
        assert tags is not None
        assert adapter.normalize_brand("LANEIGE") in tags["brands"]
        assert adapter.normalize_brand("라네즈") in tags["brands"]
        assert adapter.normalize_category("Lip Care") in tags["categories"]
        assert tags["metrics"] == ["sos"]
        assert meta[TAG_VERSION_KEY] == 1

    def test_untagged_metadata_reads_as_none(self):
        assert read_tags({"doc_id": "x", "title": "t"}) is None
        assert read_tags(None) is None


# ---------------------------------------------------------------------------
# F9 재현 — 엔티티 필터는 문서를 0건 돌려준다
# ---------------------------------------------------------------------------


class TestF9EntityFilterReproduction:
    @pytest.mark.asyncio
    async def test_ontology_filter_returns_zero_documents_on_index(self, chroma_dir, corpus):
        """get_ontology_filters의 `$or` where 조건은 실제 색인에서 0건이다 (태그 유무 무관).

        조건 키(`brand`·`category`)가 색인 메타데이터에 없고, 값도 canonical id가 아니다
        (`LANEIGE`·`lip care`). 필터 방식은 폐기하고 보너스로 바꾼다 (E10).
        """
        from src.rag.entity_linker import EntityLinker

        linker = EntityLinker(use_spacy=False)
        filters = linker.get_ontology_filters(linker.link(QUERY))
        assert "$or" in filters

        await _build_untagged_index(chroma_dir, corpus)
        collection = chromadb.PersistentClient(path=chroma_dir).get_collection("amore_docs")
        assert collection.count() == 6
        assert collection.get(where=filters, include=[])["ids"] == []

        await _build_tagged_index(chroma_dir, corpus)
        assert collection.get(where=filters, include=[])["ids"] == []


# ---------------------------------------------------------------------------
# build_index: 색인 시 태깅 + --retag (메타데이터만 갱신)
# ---------------------------------------------------------------------------


class TestBuildIndexTagging:
    @pytest.mark.asyncio
    async def test_build_tags_every_chunk(self, chroma_dir, corpus):
        status = await _build_tagged_index(chroma_dir, corpus)
        assert status["indexed_count"] == 6
        assert status["tagged_count"] == 6
        assert status["tagged"] is True

        collection = chromadb.PersistentClient(path=chroma_dir).get_collection("amore_docs")
        got = collection.get(ids=[MATCHING_CHUNK], include=["metadatas"])
        tags = read_tags(got["metadatas"][0])
        assert tags == {"brands": ["laneige"], "categories": ["lip_care"], "metrics": []}

    @pytest.mark.asyncio
    async def test_retag_updates_metadata_only(self, chroma_dir, corpus):
        from src.rag.build_index import build_index

        await _build_untagged_index(chroma_dir, corpus)
        collection = chromadb.PersistentClient(path=chroma_dir).get_collection("amore_docs")
        before = collection.get(include=["documents", "embeddings", "metadatas"])
        assert all(read_tags(m) is None for m in before["metadatas"])

        # 임베딩 경로가 호출되면 실패하도록 막는다 — retag는 재임베딩하지 않는다
        with patch.object(
            DocumentRetriever, "_embed_texts", AsyncMock(side_effect=AssertionError("embed"))
        ):
            status = await build_index(persist_dir=chroma_dir, docs_path=str(corpus), retag=True)

        assert status["retag"] is True
        assert status["retagged"] == 6
        assert status["tag_coverage"]["brand_or_category"] == 2
        assert status["tagged_count"] == 6

        after = collection.get(include=["documents", "embeddings", "metadatas"])
        assert collection.count() == 6
        assert after["ids"] == before["ids"]
        assert after["documents"] == before["documents"]
        assert [list(e) for e in after["embeddings"]] == [list(e) for e in before["embeddings"]]
        for old, new in zip(before["metadatas"], after["metadatas"], strict=True):
            assert {k: new[k] for k in old} == old  # 기존 메타데이터 보존
            assert read_tags(new) is not None

    @pytest.mark.asyncio
    async def test_retag_dry_run_reports_coverage_without_writing(self, chroma_dir, corpus):
        from src.rag.build_index import build_index

        await _build_untagged_index(chroma_dir, corpus)
        status = await build_index(
            persist_dir=chroma_dir, docs_path=str(corpus), retag=True, dry_run=True
        )
        assert status["dry_run"] is True
        assert status["retagged"] == 0
        assert status["tag_coverage"]["total"] == 6
        assert status["tag_coverage"]["brand"] == 2  # 라네즈 청크, COSRX 청크
        assert status["tag_coverage"]["category"] == 2  # 립케어 청크, COSRX(serum→skin_care)
        assert status["tagged_count"] == 0

    @pytest.mark.asyncio
    async def test_initialize_twice_keeps_count_and_tags(self, chroma_dir, corpus):
        await _build_tagged_index(chroma_dir, corpus)
        for _ in range(2):
            retriever = DocumentRetriever(docs_path=str(corpus), use_query_expansion=False)
            await retriever.initialize()
            assert retriever.collection.count() == 6
            assert retriever.get_index_status()["tagged_count"] == 6

    @pytest.mark.asyncio
    async def test_untagged_index_status_reports_not_tagged(self, chroma_dir, corpus):
        await _build_untagged_index(chroma_dir, corpus)
        retriever = DocumentRetriever(docs_path=str(corpus), use_query_expansion=False)
        await retriever.initialize()
        status = retriever.get_index_status()
        assert status["in_sync"] is True
        assert status["tagged"] is False
        assert status["tagged_count"] == 0


# ---------------------------------------------------------------------------
# HybridRetriever 레거시 경로: 보너스는 재정렬만, 후보는 줄지 않는다 (E10)
# ---------------------------------------------------------------------------


class TestHybridRetrieverEntityBonus:
    @pytest.mark.asyncio
    async def test_tagged_matching_chunk_ranks_higher_and_k_kept(self, chroma_dir, corpus):
        await _build_tagged_index(chroma_dir, corpus)
        doc_retriever = DocumentRetriever(docs_path=str(corpus), use_query_expansion=False)
        hybrid = _hybrid(doc_retriever)
        hybrid._initialized = True
        await doc_retriever.initialize()
        entities = hybrid.entity_extractor.extract(QUERY)
        assert entities["brands"] == ["laneige"] and entities["categories"] == ["lip_care"]

        with _patch_embed():
            baseline, method0 = await hybrid._hybrid_search(QUERY, top_k=TOP_K)
            _clear_search_cache()
            stats: dict[str, Any] = {}
            boosted, method1 = await hybrid._hybrid_search(
                QUERY, top_k=TOP_K, entities=entities, rerank_stats=stats
            )

        assert method0 == method1 == "hybrid_rrf"
        assert len(baseline) == TOP_K
        assert len(boosted) == TOP_K

        baseline_ids = _ids(baseline)
        boosted_ids = _ids(boosted)
        # 후보 집합은 그대로고 순서만 바뀐다 (보너스는 필터가 아니다)
        assert sorted(boosted_ids) == sorted(baseline_ids)
        # 전제: 보너스 전에는 매칭 청크가 1위가 아니다 (아니면 이 테스트는 아무것도 증명 못 함)
        assert baseline_ids.index(MATCHING_CHUNK) > 0
        assert boosted_ids.index(MATCHING_CHUNK) < baseline_ids.index(MATCHING_CHUNK)
        assert boosted_ids[0] == MATCHING_CHUNK
        # 보너스 결과에는 근거 표기가 붙는다
        matched = next(r for r in boosted if _ids([r])[0] == MATCHING_CHUNK)
        assert matched["entity_bonus"] > 0
        assert matched["entity_tag_matches"] == {
            "brands": ["laneige"],
            "categories": ["lip_care"],
        }
        assert stats["bonus_candidates"] >= 1
        assert stats["tagged_candidates"] == stats["candidates"]

    @pytest.mark.asyncio
    async def test_untagged_index_returns_same_k_as_without_bonus(self, chroma_dir, corpus):
        await _build_untagged_index(chroma_dir, corpus)
        doc_retriever = DocumentRetriever(docs_path=str(corpus), use_query_expansion=False)
        await doc_retriever.initialize()
        hybrid = _hybrid(doc_retriever)
        hybrid._initialized = True
        entities = hybrid.entity_extractor.extract(QUERY)

        with _patch_embed():
            baseline, _ = await hybrid._hybrid_search(QUERY, top_k=TOP_K)
            _clear_search_cache()
            stats: dict[str, Any] = {}
            with_entities, _ = await hybrid._hybrid_search(
                QUERY, top_k=TOP_K, entities=entities, rerank_stats=stats
            )

        assert len(with_entities) == TOP_K
        assert _ids(with_entities) == _ids(baseline)
        assert all("entity_bonus" not in r for r in with_entities)
        assert stats["tagged_candidates"] == 0
        assert stats["bonus_candidates"] == 0

    @pytest.mark.asyncio
    async def test_retrieve_records_bonus_in_metadata(self, chroma_dir, corpus):
        await _build_tagged_index(chroma_dir, corpus)
        doc_retriever = DocumentRetriever(docs_path=str(corpus), use_query_expansion=False)
        hybrid = _hybrid(doc_retriever)

        with _patch_embed():
            context = await hybrid.retrieve(QUERY)

        assert "retrieval_error" not in context.metadata
        rerank = context.metadata["entity_rerank"]
        assert rerank["query_targets"] == {"brands": ["laneige"], "categories": ["lip_care"]}
        assert rerank["bonus_chunks"] >= 1
        assert rerank["bonus_chunks"] == sum(
            1 for r in context.rag_chunks if r.get("entity_bonus", 0) > 0
        )
        assert MATCHING_CHUNK in _ids(context.rag_chunks)

    @pytest.mark.asyncio
    async def test_retrieve_on_untagged_index_records_zero_bonus(self, chroma_dir, corpus):
        await _build_untagged_index(chroma_dir, corpus)
        doc_retriever = DocumentRetriever(docs_path=str(corpus), use_query_expansion=False)
        hybrid = _hybrid(doc_retriever)

        with _patch_embed():
            context = await hybrid.retrieve(QUERY)

        assert "retrieval_error" not in context.metadata
        rerank = context.metadata["entity_rerank"]
        assert rerank["bonus_chunks"] == 0
        assert rerank["tagged_candidates"] == 0
        assert len(context.rag_chunks) == 6  # 코퍼스 전체(6) < 의도 예산 → 줄지 않음


# ---------------------------------------------------------------------------
# apply_entity_bonus 순수 함수
# ---------------------------------------------------------------------------


class TestApplyEntityBonus:
    def _results(self) -> list[dict[str, Any]]:
        return [
            {"id": "a", "content": "a", "rrf_score": 0.0330},
            {"id": "b", "content": "b", "rrf_score": 0.0325},
            {"id": "c", "content": "c", "rrf_score": 0.0162},
        ]

    def test_bonus_reorders_but_never_drops(self):
        tags = {
            "a": {"brands": ["cosrx"], "categories": []},
            "b": {"brands": ["laneige"], "categories": ["lip_care"]},
            "c": {"brands": ["laneige"], "categories": []},
        }
        out, stats = apply_entity_bonus(
            self._results(),
            tags,
            {"brands": {"laneige"}, "categories": {"lip_care"}},
            {"brands": 0.001, "categories": 0.001},
        )
        assert [r["id"] for r in out] == ["b", "a", "c"]  # c(+0.001)는 a를 넘지 못함
        assert len(out) == 3
        assert stats == {"candidates": 3, "tagged_candidates": 3, "bonus_candidates": 2}

    def test_no_tags_keeps_input_order_and_objects(self):
        results = self._results()
        out, stats = apply_entity_bonus(results, {}, {"brands": {"laneige"}}, {"brands": 0.001})
        assert out is results
        assert stats["bonus_candidates"] == 0

    def test_zero_weight_disables_bonus(self):
        results = self._results()
        tags = {"c": {"brands": ["laneige"], "categories": []}}
        out, stats = apply_entity_bonus(results, tags, {"brands": {"laneige"}}, {"brands": 0.0})
        assert out is results
        assert stats["tagged_candidates"] == 1
        assert stats["bonus_candidates"] == 0
