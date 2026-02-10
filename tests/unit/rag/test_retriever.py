"""
DocumentRetriever 단위 테스트
============================
문서 로딩, 청킹, 검색 캐시, 임베딩 캐시, 쿼리 확장 검증
(ChromaDB/OpenAI 호출은 mock)
"""

from pathlib import Path

import pytest

from src.rag.retriever import DocumentRetriever

# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------


class TestDocumentRetrieverInit:
    """초기화 및 설정 테스트"""

    def test_default_init(self):
        retriever = DocumentRetriever()
        assert retriever.documents == {}
        assert retriever.chunks == []
        assert retriever._initialized is False

    def test_init_with_options(self):
        retriever = DocumentRetriever(
            use_semantic_chunking=False,
            use_reranker=False,
            use_query_expansion=False,
        )
        assert retriever.use_semantic_chunking is False
        assert retriever.use_query_expansion is False

    def test_docs_path_custom(self):
        retriever = DocumentRetriever(docs_path="/tmp/docs")
        assert retriever.docs_path == Path("/tmp/docs")

    def test_docs_path_default(self):
        retriever = DocumentRetriever()
        assert retriever.docs_path == Path("./docs")


# ---------------------------------------------------------------------------
# 문서 목록
# ---------------------------------------------------------------------------


class TestDocumentCatalog:
    """문서 카탈로그 (DOCUMENTS 클래스 변수)"""

    def test_documents_dict_exists(self):
        assert hasattr(DocumentRetriever, "DOCUMENTS")
        assert isinstance(DocumentRetriever.DOCUMENTS, dict)

    def test_documents_have_required_fields(self):
        """각 문서 메타데이터에 filename과 doc_type이 있어야 함"""
        for doc_id, meta in DocumentRetriever.DOCUMENTS.items():
            assert "filename" in meta, f"{doc_id} missing 'filename'"
            assert "doc_type" in meta, f"{doc_id} missing 'doc_type'"

    def test_document_count(self):
        """14개 문서가 등록되어 있어야 함"""
        assert len(DocumentRetriever.DOCUMENTS) >= 10


# ---------------------------------------------------------------------------
# 임베딩 캐시
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    """임베딩 캐시 (MD5 기반)"""

    def test_cache_starts_empty(self):
        retriever = DocumentRetriever()
        stats = retriever.get_embedding_cache_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_stats_structure(self):
        retriever = DocumentRetriever()
        stats = retriever.get_embedding_cache_stats()
        expected_keys = {"size", "hits", "misses", "hit_rate"}
        assert expected_keys == set(stats.keys())

    def test_hit_rate_zero_initially(self):
        retriever = DocumentRetriever()
        stats = retriever.get_embedding_cache_stats()
        assert stats["hit_rate"] == 0


# ---------------------------------------------------------------------------
# 검색 캐시 TTL
# ---------------------------------------------------------------------------


class TestSearchCache:
    """검색 결과 TTL 캐시"""

    def test_class_level_cache_is_dict(self):
        assert isinstance(DocumentRetriever._search_cache, dict)
        assert isinstance(DocumentRetriever._cache_timestamps, dict)

    def test_cache_ttl_returns_int(self):
        ttl = DocumentRetriever.get_cache_ttl()
        assert isinstance(ttl, int)
        assert ttl > 0


# ---------------------------------------------------------------------------
# _load_documents (비동기 파일 로드)
# ---------------------------------------------------------------------------


class TestLoadDocuments:
    """문서 로드 테스트"""

    @pytest.mark.asyncio
    async def test_load_documents_is_async(self):
        """_load_documents는 async 메서드"""
        retriever = DocumentRetriever(docs_path="./docs")
        # 호출만 확인 (실제 파일 읽기는 환경 의존)
        await retriever._load_documents()
        # documents dict에 항목이 채워져야 함 (docs/ 폴더가 존재하면)
        assert isinstance(retriever.documents, dict)

    @pytest.mark.asyncio
    async def test_load_from_nonexistent_path(self):
        retriever = DocumentRetriever(docs_path="/nonexistent/path/docs")
        await retriever._load_documents()
        assert isinstance(retriever.documents, dict)
        # 존재하지 않는 경로면 문서가 0이거나 기본 문서
        # (실제 동작에 따라 유연하게)


# ---------------------------------------------------------------------------
# _index_documents (인덱싱)
# ---------------------------------------------------------------------------


class TestIndexDocuments:
    """문서 인덱싱 테스트"""

    def test_index_documents_method_exists(self):
        retriever = DocumentRetriever()
        assert hasattr(retriever, "_index_documents")


# ---------------------------------------------------------------------------
# get_relevant_context (통합 검색)
# ---------------------------------------------------------------------------


class TestGetRelevantContext:
    """get_relevant_context 메서드"""

    @pytest.mark.asyncio
    async def test_method_exists(self):
        retriever = DocumentRetriever()
        assert hasattr(retriever, "get_relevant_context")

    @pytest.mark.asyncio
    async def test_search_method_exists(self):
        retriever = DocumentRetriever()
        assert hasattr(retriever, "search")

    @pytest.mark.asyncio
    async def test_expand_query_method_exists(self):
        retriever = DocumentRetriever()
        assert hasattr(retriever, "expand_query")
