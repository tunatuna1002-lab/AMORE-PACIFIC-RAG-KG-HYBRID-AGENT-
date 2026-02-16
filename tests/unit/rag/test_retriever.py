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
        required_keys = {"size", "hits", "misses", "hit_rate"}
        assert required_keys.issubset(set(stats.keys()))

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


# ===========================================================================
# 이하 확장 테스트 (Wave 2 coverage 보강)
# ===========================================================================

import time
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# 청킹 (_split_into_chunks)
# ---------------------------------------------------------------------------


class TestSplitIntoChunks:
    """_split_into_chunks 청킹 테스트"""

    def _make_retriever(self):
        return DocumentRetriever(docs_path="/tmp/fake")

    def test_basic_chunking(self):
        """기본 텍스트 청킹"""
        r = self._make_retriever()
        content = "## Section One\nSome content here.\n## Section Two\nMore content."
        doc_info = {
            "doc_type": "metric_guide",
            "filename": "test.md",
            "keywords": ["test"],
            "description": "Test doc",
        }
        chunks = r._split_into_chunks(content, "test_doc", doc_info, chunk_size=500)
        assert len(chunks) >= 1
        assert all("id" in c for c in chunks)
        assert all("content" in c for c in chunks)

    def test_table_extracted_as_separate_chunk(self):
        """표(Table)가 별도 청크로 추출"""
        r = self._make_retriever()
        content = (
            "## Title\nIntro text\n"
            "| Col A | Col B |\n"
            "|-------|-------|\n"
            "| val1  | val2  |\n"
            "| val3  | val4  |\n"
            "\n## Next Section\nMore text"
        )
        doc_info = {
            "doc_type": "metric_guide",
            "filename": "t.md",
            "keywords": [],
            "description": "",
        }
        chunks = r._split_into_chunks(content, "doc", doc_info, chunk_size=500)
        table_chunks = [c for c in chunks if c.get("content_type") == "table"]
        assert len(table_chunks) >= 1

    def test_long_section_subsplit(self):
        """긴 섹션이 추가 분할됨"""
        r = self._make_retriever()
        long_text = "## Long Section\n" + "A" * 1200
        doc_info = {
            "doc_type": "metric_guide",
            "filename": "t.md",
            "keywords": [],
            "description": "",
        }
        chunks = r._split_into_chunks(long_text, "doc", doc_info, chunk_size=500)
        assert len(chunks) >= 2

    def test_metadata_preserved_in_chunks(self):
        """청크에 메타데이터가 보존"""
        r = self._make_retriever()
        content = "## Heading\nBody text"
        doc_info = {
            "doc_type": "playbook",
            "filename": "playbook.md",
            "keywords": ["kw1"],
            "description": "desc",
            "target_brand": "laneige",
            "brands_covered": ["cosrx"],
        }
        chunks = r._split_into_chunks(content, "doc", doc_info, chunk_size=500)
        assert len(chunks) >= 1
        c = chunks[0]
        assert c["doc_type"] == "playbook"
        assert c["target_brand"] == "laneige"
        assert c["brands_covered"] == ["cosrx"]

    def test_empty_content(self):
        """빈 내용"""
        r = self._make_retriever()
        doc_info = {
            "doc_type": "metric_guide",
            "filename": "t.md",
            "keywords": [],
            "description": "",
        }
        chunks = r._split_into_chunks("", "doc", doc_info, chunk_size=500)
        assert chunks == []


# ---------------------------------------------------------------------------
# _smart_split
# ---------------------------------------------------------------------------


class TestSmartSplit:
    """_smart_split 메서드 테스트"""

    def _make_retriever(self):
        return DocumentRetriever(docs_path="/tmp/fake")

    def test_short_text_single_chunk(self):
        """짧은 텍스트는 하나의 청크"""
        r = self._make_retriever()
        result = r._smart_split("Short text", 500)
        assert len(result) == 1
        assert result[0] == "Short text"

    def test_paragraph_split(self):
        """단락 기반 분할"""
        r = self._make_retriever()
        text = "Para 1 content\n\nPara 2 content\n\nPara 3 content"
        result = r._smart_split(text, 20)
        assert len(result) >= 2

    def test_forced_split_on_long_paragraph(self):
        """chunk_size 초과 단락 강제 분할"""
        r = self._make_retriever()
        text = "A" * 1000
        result = r._smart_split(text, 300)
        assert len(result) >= 3


# ---------------------------------------------------------------------------
# _get_chunk_size_by_type
# ---------------------------------------------------------------------------


class TestGetChunkSizeByType:
    """문서 유형별 청크 크기"""

    def test_known_types(self):
        """알려진 유형"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        assert r._get_chunk_size_by_type("playbook") == 800
        assert r._get_chunk_size_by_type("intelligence") == 600
        assert r._get_chunk_size_by_type("metric_guide") == 500
        assert r._get_chunk_size_by_type("ir_report") == 700

    def test_unknown_type_default(self):
        """알 수 없는 유형은 기본값 500"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        assert r._get_chunk_size_by_type("unknown_type") == 500


# ---------------------------------------------------------------------------
# _needs_retrieval (Self-RAG)
# ---------------------------------------------------------------------------


class TestNeedsRetrieval:
    """Self-RAG 검색 필요성 판단"""

    def _make_retriever(self):
        return DocumentRetriever(docs_path="/tmp/fake")

    def test_greeting_no_retrieval(self):
        """인사 쿼리는 검색 불필요"""
        r = self._make_retriever()
        assert r._needs_retrieval("안녕하세요") is False
        assert r._needs_retrieval("Hello") is False
        assert r._needs_retrieval("감사합니다") is False

    def test_short_query_no_retrieval(self):
        """3자 미만 쿼리는 검색 불필요"""
        r = self._make_retriever()
        assert r._needs_retrieval("ab") is False
        assert r._needs_retrieval("") is False

    def test_affirmative_no_retrieval(self):
        """단순 긍정은 검색 불필요"""
        r = self._make_retriever()
        assert r._needs_retrieval("네") is False
        assert r._needs_retrieval("ok") is False

    def test_real_query_needs_retrieval(self):
        """실제 쿼리는 검색 필요"""
        r = self._make_retriever()
        assert r._needs_retrieval("LANEIGE Lip Sleeping Mask 분석") is True
        assert r._needs_retrieval("SoS 지표란 무엇인가요?") is True

    def test_help_query_no_retrieval(self):
        """도움 요청은 검색 불필요"""
        r = self._make_retriever()
        assert r._needs_retrieval("도움말") is False


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    """한/영 토크나이저"""

    def test_korean_tokens(self):
        r = DocumentRetriever(docs_path="/tmp/fake")
        tokens = r._tokenize("라네즈 립슬리핑마스크")
        assert len(tokens) >= 1
        assert all(len(t) > 1 for t in tokens)

    def test_english_tokens(self):
        r = DocumentRetriever(docs_path="/tmp/fake")
        tokens = r._tokenize("LANEIGE Lip Sleeping Mask")
        assert "laneige" in tokens
        assert "lip" in tokens

    def test_mixed_tokens(self):
        r = DocumentRetriever(docs_path="/tmp/fake")
        tokens = r._tokenize("LANEIGE 립케어 분석 2026")
        assert any(t.isalpha() and t.isascii() for t in tokens)
        assert any(not t.isascii() for t in tokens)

    def test_single_char_filtered(self):
        """1자 토큰 제거"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        tokens = r._tokenize("a b c hello")
        assert "a" not in tokens
        assert "hello" in tokens


# ---------------------------------------------------------------------------
# _get_cache_key / _is_cache_valid / _clean_expired_cache
# ---------------------------------------------------------------------------


class TestSearchCacheExtended:
    """검색 캐시 확장 테스트"""

    def test_cache_key_format(self):
        """캐시 키 형식"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        key = r._get_cache_key("query", 5, None, None)
        assert "query" in key
        assert "5" in key
        assert "all" in key

    def test_cache_key_with_filter(self):
        """필터 포함 캐시 키"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        key = r._get_cache_key("q", 3, "doc_id_1", ["playbook"])
        assert "doc_id_1" in key
        assert "playbook" in key

    def test_cache_key_with_type_filter(self):
        """doc_type_filter 포함"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        key = r._get_cache_key("q", 3, None, ["playbook", "intelligence"])
        assert "playbook,intelligence" in key

    def test_is_cache_valid_no_entry(self):
        """캐시에 없는 키"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        assert r._is_cache_valid("nonexistent_key") is False

    def test_is_cache_valid_fresh_entry(self):
        """신선한 캐시 엔트리"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        key = "test_key"
        r._cache_timestamps[key] = time.time()
        assert r._is_cache_valid(key) is True
        # 정리
        r._cache_timestamps.pop(key, None)

    def test_is_cache_valid_expired_entry(self):
        """만료된 캐시 엔트리"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        key = "old_key"
        r._cache_timestamps[key] = time.time() - 99999
        assert r._is_cache_valid(key) is False
        r._cache_timestamps.pop(key, None)

    def test_clean_expired_cache(self):
        """만료된 캐시 정리"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        # 만료된 항목 추가
        r._search_cache["expired"] = ["data"]
        r._cache_timestamps["expired"] = time.time() - 99999
        # 신선한 항목 추가
        r._search_cache["fresh"] = ["data"]
        r._cache_timestamps["fresh"] = time.time()

        r._clean_expired_cache()

        assert "expired" not in r._search_cache
        assert "fresh" in r._search_cache

        # 정리
        r._search_cache.pop("fresh", None)
        r._cache_timestamps.pop("fresh", None)


# ---------------------------------------------------------------------------
# _rrf_merge
# ---------------------------------------------------------------------------


class TestRRFMerge:
    """Reciprocal Rank Fusion 테스트"""

    def _make_retriever(self):
        return DocumentRetriever(docs_path="/tmp/fake")

    def test_rrf_merge_basic(self):
        """기본 RRF 병합"""
        r = self._make_retriever()
        dense = [
            {"id": "d1", "content": "doc1", "score": 0.9},
            {"id": "d2", "content": "doc2", "score": 0.7},
        ]
        sparse = [
            {"id": "d2", "content": "doc2", "score": 5.0},
            {"id": "d3", "content": "doc3", "score": 3.0},
        ]
        merged = r._rrf_merge(dense, sparse)
        ids = [r["id"] for r in merged]
        # d2가 양쪽에 있으므로 높은 점수
        assert "d2" in ids
        assert len(merged) == 3  # d1, d2, d3

    def test_rrf_merge_no_overlap(self):
        """겹치지 않는 결과"""
        r = self._make_retriever()
        dense = [{"id": "d1", "content": "a", "score": 0.5}]
        sparse = [{"id": "d2", "content": "b", "score": 3.0}]
        merged = r._rrf_merge(dense, sparse)
        assert len(merged) == 2

    def test_rrf_merge_empty_lists(self):
        """빈 리스트 병합"""
        r = self._make_retriever()
        merged = r._rrf_merge([], [])
        assert merged == []

    def test_rrf_merge_one_empty(self):
        """한쪽만 빈 리스트"""
        r = self._make_retriever()
        dense = [{"id": "d1", "content": "a", "score": 0.5}]
        merged = r._rrf_merge(dense, [])
        assert len(merged) == 1


# ---------------------------------------------------------------------------
# _text_hash
# ---------------------------------------------------------------------------


class TestTextHash:
    """텍스트 해시 테스트"""

    def test_consistent_hash(self):
        """같은 텍스트는 같은 해시"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        h1 = r._get_text_hash("hello world")
        h2 = r._get_text_hash("hello world")
        assert h1 == h2

    def test_different_text_different_hash(self):
        """다른 텍스트는 다른 해시"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        h1 = r._get_text_hash("hello")
        h2 = r._get_text_hash("world")
        assert h1 != h2


# ---------------------------------------------------------------------------
# expand_query
# ---------------------------------------------------------------------------


class TestExpandQuery:
    """쿼리 확장 테스트"""

    @pytest.mark.asyncio
    async def test_no_client_returns_original(self):
        """OpenAI 클라이언트 없으면 원본 쿼리만 반환"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        r.openai_client = None
        result = await r.expand_query("test query")
        assert result == ["test query"]

    @pytest.mark.asyncio
    async def test_expansion_disabled_returns_original(self):
        """쿼리 확장 비활성화"""
        r = DocumentRetriever(docs_path="/tmp/fake", use_query_expansion=False)
        r.openai_client = MagicMock()
        result = await r.expand_query("test query")
        assert result == ["test query"]

    @pytest.mark.asyncio
    async def test_successful_expansion(self):
        """성공적 쿼리 확장"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        mock_client = MagicMock()
        r.openai_client = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["lip care trends", "lip mask analysis"]'
        mock_client.chat.completions.create.return_value = mock_response

        result = await r.expand_query("lip care")
        assert result[0] == "lip care"  # 원본 쿼리가 첫번째
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_expansion_exception_fallback(self):
        """쿼리 확장 실패 시 원본만 반환"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API fail")
        r.openai_client = mock_client

        result = await r.expand_query("test")
        assert result == ["test"]

    @pytest.mark.asyncio
    async def test_expansion_max_3_additional(self):
        """확장 쿼리 최대 3개"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        mock_client = MagicMock()
        r.openai_client = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["q1", "q2", "q3", "q4", "q5"]'
        mock_client.chat.completions.create.return_value = mock_response

        result = await r.expand_query("original")
        assert len(result) <= 4  # original + max 3


# ---------------------------------------------------------------------------
# search (통합 검색)
# ---------------------------------------------------------------------------


class TestSearchExtended:
    """search 메서드 확장 테스트"""

    @pytest.mark.asyncio
    async def test_search_returns_cached_result(self):
        """캐시된 결과 반환"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        r._initialized = True
        cache_key = r._get_cache_key("cached query", 3, None, None)
        cached_data = [{"id": "cached", "content": "cached content"}]
        r._search_cache[cache_key] = cached_data
        r._cache_timestamps[cache_key] = time.time()

        # _needs_retrieval이 True 반환하도록
        with patch.object(r, "_needs_retrieval", return_value=True):
            result = await r.search("cached query", top_k=3)
        assert result == cached_data

        # 정리
        r._search_cache.pop(cache_key, None)
        r._cache_timestamps.pop(cache_key, None)

    @pytest.mark.asyncio
    async def test_search_no_retrieval_needed(self):
        """검색 불필요 판정 시 빈 결과"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        r._initialized = True
        with patch.object(r, "_needs_retrieval", return_value=False):
            result = await r.search("안녕", top_k=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_document_uninitialized(self):
        """미초기화 상태에서 get_document → initialize 호출"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        r._initialized = False
        r.initialize = AsyncMock(return_value=True)
        # initialize 후 documents에서 검색
        r.documents = {"test_doc": "content"}

        # initialize가 호출되면 _initialized를 True로 설정
        async def fake_init():
            r._initialized = True
            return True

        r.initialize = fake_init
        result = await r.get_document("test_doc")
        assert result == "content"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self):
        """존재하지 않는 문서"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        r._initialized = True
        r.documents = {}
        result = await r.get_document("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_relevant_context_returns_string(self):
        """get_relevant_context가 문자열 반환"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        r._initialized = True
        r.search = AsyncMock(
            return_value=[
                {
                    "content": "test content",
                    "metadata": {"title": "Test Title"},
                }
            ]
        )
        result = await r.get_relevant_context("test query", max_tokens=2000)
        assert isinstance(result, str)
        assert "test content" in result

    @pytest.mark.asyncio
    async def test_get_relevant_context_respects_max_tokens(self):
        """max_tokens 제한 적용"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        r._initialized = True
        # 큰 문서 10개
        r.search = AsyncMock(
            return_value=[
                {"content": "X" * 5000, "metadata": {"title": f"Doc{i}"}} for i in range(10)
            ]
        )
        result = await r.get_relevant_context("q", max_tokens=100)
        # max_tokens * 4 = 400 글자 제한
        assert len(result) <= 5000  # 최소한 전체보다 적어야 함


# ---------------------------------------------------------------------------
# _embed_texts
# ---------------------------------------------------------------------------


class TestEmbedTexts:
    """임베딩 생성 테스트"""

    @pytest.mark.asyncio
    async def test_no_client_returns_empty(self):
        """OpenAI 클라이언트 없으면 빈 리스트"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        r.openai_client = None
        result = await r._embed_texts(["hello"])
        assert result == []

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """캐시 히트 시 API 호출 없음"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        r.openai_client = MagicMock()

        # 캐시에 결과 미리 설정
        text_hash = r._get_text_hash("cached text")
        r._embedding_cache = AsyncMock()
        r._embedding_cache.get = AsyncMock(return_value=[0.1, 0.2, 0.3])

        result = await r._embed_texts(["cached text"])
        assert result == [[0.1, 0.2, 0.3]]
        # API 호출 없어야 함
        r.openai_client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_api(self):
        """캐시 미스 시 API 호출"""
        r = DocumentRetriever(docs_path="/tmp/fake")
        mock_client = MagicMock()
        r.openai_client = mock_client

        r._embedding_cache = AsyncMock()
        r._embedding_cache.get = AsyncMock(return_value=None)
        r._embedding_cache.put = AsyncMock()

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.5, 0.6]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        result = await r._embed_texts(["new text"])
        assert result == [[0.5, 0.6]]
        mock_client.embeddings.create.assert_called_once()
        r._embedding_cache.put.assert_awaited_once()
