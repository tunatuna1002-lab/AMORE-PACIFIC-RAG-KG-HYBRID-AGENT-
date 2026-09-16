"""
Document Retriever
RAG를 위한 문서 검색 모듈

14개 MD 파일 기반:

[Type D: 기존 지표 가이드] - docs/guides/
- Strategic Indicators Definition.md
- Metric Interpretation Guide.md
- Indicator Combination Playbook.md
- Home Page Insight Rules.md

[Type A: 분석 플레이북] - docs/market/
- 아마존 랭킹 급등 원인 역추적 보고서.md
- 아마존 랭킹 변동 원인 분석 가이드.md

[Type B: 시장 인텔리전스] - docs/market/
- (1) K-뷰티 초격차의 서막.md
- 미국 뷰티 트렌드 레이더.md
- 뷰티 트렌드 분석 및 판매 전략 제안.md

[Type C: 대응 가이드] - docs/market/
- 부정 이슈 조기경보 및 대응 프롬프트.md
- 인플루언서 맵 & 메시지 맵 생성.md

[Type E: IR 분기 실적 보고서] - docs/ir/
- AP_1Q25_EN.md (아모레퍼시픽 2025 Q1)
- AP_2Q25_EN.md (아모레퍼시픽 2025 Q2)
- AP_3Q25_EN.md (아모레퍼시픽 2025 Q3)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from . import bm25_index, document_loader, search_cache, vector_index
from . import fusion as _fusion
from .bm25_index import BM25_AVAILABLE
from .document_loader import strip_embedded_binaries
from .document_registry import CONFIG_PATH as _REGISTRY_CONFIG_PATH
from .document_registry import DEFAULT_CACHE_TTL, load_rag_config
from .document_registry import DOCUMENTS as _DOCUMENTS
from .section_chunker import chunk_size_for, split_into_chunks
from .selfrag_gate import needs_retrieval as _needs_retrieval_gate

try:
    from .reranker import get_reranker

    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

try:
    from .chunker import get_semantic_chunker

    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False

# Lazy import flag - 실제 사용 시 초기화
VECTOR_SEARCH_AVAILABLE = None

logger = logging.getLogger(__name__)

# Backwards-compatible re-export: several modules and tests import this helper
# from ``src.rag.retriever``. It lives in ``document_loader`` now.
_has_hangul = document_loader.has_hangul

__all__ = [
    "BM25_AVAILABLE",
    "RERANKER_AVAILABLE",
    "SEMANTIC_CHUNKER_AVAILABLE",
    "VECTOR_SEARCH_AVAILABLE",
    "DocumentRetriever",
    "strip_embedded_binaries",
]


class DocumentRetriever:
    """문서 검색 파사드 (TTL 캐싱 지원).

    조립만 하고, 각 책임은 별도 모듈이 갖는다:

    ======================  ==================================
    책임                     모듈
    ======================  ==================================
    문서 카탈로그·설정        :mod:`src.rag.document_registry`
    파일 로드·전처리          :mod:`src.rag.document_loader`
    섹션/표 청킹             :mod:`src.rag.section_chunker`
    벡터 색인·검색           :mod:`src.rag.vector_index`
    BM25 색인·검색           :mod:`src.rag.bm25_index`
    결과 캐시                :mod:`src.rag.search_cache`
    RRF 융합                 :mod:`src.rag.fusion`
    Self-RAG 게이트          :mod:`src.rag.selfrag_gate`
    ======================  ==================================
    """

    # 설정 파일 경로
    CONFIG_PATH = _REGISTRY_CONFIG_PATH

    # 검색 결과 캐시 (maxsize=100, TTL=5분) — 인스턴스 간 공유
    _search_cache: dict[str, Any] = {}
    _cache_timestamps: dict[str, float] = {}
    _CACHE_TTL = None  # 설정에서 로드

    # 문서 메타데이터 (src.rag.document_registry.DOCUMENTS)
    DOCUMENTS = _DOCUMENTS

    @classmethod
    def _load_config(cls) -> dict:
        """설정 파일에서 RAG 관련 설정 로드"""
        return load_rag_config()

    @classmethod
    def get_cache_ttl(cls) -> int:
        """캐시 TTL 반환 (설정 파일에서 로드, 기본 300초)"""
        if cls._CACHE_TTL is None:
            cls._CACHE_TTL = cls._load_config().get("ttl_seconds", DEFAULT_CACHE_TTL)
        return cls._CACHE_TTL

    def __init__(
        self,
        docs_path: str = "./docs",
        use_semantic_chunking: bool = False,
        use_reranker: bool = True,
        use_query_expansion: bool = True,
    ):
        """
        Args:
            docs_path: 문서 폴더 경로
            use_semantic_chunking: Semantic Chunker 사용 여부
            use_reranker: Reranker 사용 여부 (기본값: True)
            use_query_expansion: Query Expansion 사용 여부 (기본값: True)
        """
        self.docs_path = Path(docs_path)
        self.documents: dict[str, str] = {}
        self.chunks: list[dict[str, Any]] = []
        self._chunk_index: dict[str, dict[str, Any]] = {}
        self.embedding_model = None
        self.openai_client = None
        self.embedding_model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.collection = None
        self._initialized = False

        # 새 기능 플래그
        self.use_semantic_chunking = use_semantic_chunking
        self.use_reranker = use_reranker and RERANKER_AVAILABLE
        self.use_query_expansion = use_query_expansion

        # Reranker 인스턴스 (lazy loading)
        self._reranker = None

        # BM25 index (lazy init)
        self._bm25_index = None
        self._bm25_corpus_ids: list[str] = []

        # Embedding 캐시 (feature-flag-guarded)
        from src.infrastructure.feature_flags import FeatureFlags

        flags = FeatureFlags.get_instance()
        if flags.use_sqlite_embedding_cache():
            from src.rag.embedding_cache import SQLiteEmbeddingCache

            self._embedding_cache = SQLiteEmbeddingCache()
        else:
            from src.rag.embedding_cache import InMemoryEmbeddingCache

            self._embedding_cache = InMemoryEmbeddingCache(max_size=1000)

    def _check_vector_search(self) -> bool:
        """벡터 검색 가능 여부 확인 (OpenAI Embeddings + ChromaDB)"""
        global VECTOR_SEARCH_AVAILABLE
        if VECTOR_SEARCH_AVAILABLE is None:
            VECTOR_SEARCH_AVAILABLE = vector_index.detect_availability()
        return VECTOR_SEARCH_AVAILABLE

    async def initialize(self) -> bool:
        """
        문서 로드 및 벡터 인덱스 초기화

        Returns:
            초기화 성공 여부

        Raises:
            ValueError: 벡터 검색 초기화 실패 시
        """
        # 문서 로드
        await self._load_documents()

        # 벡터 검색 초기화 (필수)
        if not self._check_vector_search():
            raise ValueError(
                "Vector search is required but not available. "
                "Please install required packages: chromadb, openai"
            )

        await self._initialize_vector_search()

        if self.collection is None:
            raise ValueError("Vector search initialization failed")

        self._initialized = True
        return True

    async def _load_documents(self) -> None:
        """MD 문서 로드 (:mod:`src.rag.document_loader` 로 위임)"""
        # Semantic Chunker 초기화 (옵션)
        semantic_chunker = None
        if self.use_semantic_chunking and SEMANTIC_CHUNKER_AVAILABLE:
            semantic_chunker = get_semantic_chunker()

        documents, chunks = document_loader.load_documents(
            self.docs_path, self.DOCUMENTS, semantic_chunker
        )
        self.documents.update(documents)
        self.chunks.extend(chunks)

        # 청크 인덱스 갱신 (벡터 검색 결과 메타데이터 보강용)
        self._chunk_index = document_loader.index_chunks(self.chunks)

    def _get_chunk_size_by_type(self, doc_type: str) -> int:
        """문서 유형별 청크 크기 반환 (:mod:`src.rag.section_chunker` 로 위임)"""
        return chunk_size_for(doc_type)

    def _split_into_chunks(
        self, content: str, doc_id: str, doc_info: dict, chunk_size: int = 500
    ) -> list[dict[str, Any]]:
        """문서를 청크로 분할 (:mod:`src.rag.section_chunker` 로 위임)"""
        return split_into_chunks(content, doc_id, doc_info, chunk_size)

    def _smart_split(self, text: str, chunk_size: int) -> list[str]:
        """텍스트를 의미 단위로 분할 (:mod:`src.rag.section_chunker` 로 위임)"""
        from .section_chunker import smart_split

        return smart_split(text, chunk_size)

    async def _initialize_vector_search(self) -> None:
        """
        벡터 검색 초기화 (필수)

        Raises:
            ImportError: 필수 패키지 미설치 시
            ValueError: OPENAI_API_KEY 미설정 시
            Exception: 기타 초기화 실패 시
        """
        if not VECTOR_SEARCH_AVAILABLE:
            raise ImportError("Vector search dependencies not available. Install: chromadb, openai")

        (
            self.openai_client,
            self.client,
            self.collection,
            self.embedding_model_name,
        ) = vector_index.open_collection(self.embedding_model_name)

        # 문서 인덱싱 — 컬렉션에 없는 청크만 증분 색인한다.
        await self._index_documents()

        logger.info(f"ChromaDB initialized: {self.collection.count()} documents indexed")

    def _get_text_hash(self, text: str) -> str:
        """텍스트 해시 생성 (:mod:`src.rag.vector_index` 로 위임)"""
        return vector_index.text_hash(text)

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """OpenAI Embeddings API로 텍스트 임베딩 생성 (캐시 적용)"""
        return await vector_index.embed_texts(
            self.openai_client, self.embedding_model_name, self._embedding_cache, texts
        )

    def get_embedding_cache_stats(self) -> dict:
        """캐시 통계 반환"""
        return self._embedding_cache.get_stats()

    async def expand_query(self, query: str) -> list[str]:
        """
        LLM을 사용하여 쿼리 확장

        Args:
            query: 원본 쿼리

        Returns:
            확장된 쿼리 리스트 (원본 포함)
        """
        if not self.openai_client or not self.use_query_expansion:
            return [query]

        try:
            # 코퍼스는 이중언어다 — 한국어 가이드/시장자료 271청크 + 영문 IR
            # 분기보고서 1,971청크(전체의 88%). 동일 언어 패러프레이즈만 만들면
            # 한국어 질의는 영문 IR 청크에 영영 닿지 못하고, 오히려 확장된
            # 동일 언어 결과가 소수의 IR 청크를 top-k 밖으로 밀어낸다
            # (실측: 확장 off일 때 한국어 IR 회수 0.138 → 확장 on일 때 0.000).
            # 그래서 반대 언어 변형을 반드시 1개 포함시킨다.
            other_language = "English" if _has_hangul(query) else "Korean"
            prompt = f"""You are a search query expansion assistant. Given a user query, generate 2-3 alternative phrasings or related queries that would help retrieve relevant documents.

Original query: {query}

The document corpus is bilingual: Korean market/metric guides and English
AMOREPACIFIC quarterly IR reports.

Generate alternative queries that:
1. Use synonyms or related terms
2. Rephrase the question differently
3. MUST include exactly one variant translated into {other_language},
   preserving proper nouns as they appear in that language
   (아모레퍼시픽 → AMOREPACIFIC, 라네즈 → LANEIGE, 1분기 → 1Q)

Return ONLY a JSON array of strings, like: ["query1", "query2", "query3"]
Do not include any explanation."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a search query expansion system."},
                    {"role": "user", "content": prompt},
                ],
                # 평가 재현성: 확장 질의가 매 실행 달라지면 검색 결과가 흔들려
                # 작은 지표 변화를 노이즈와 구분할 수 없다. 실제로 코드·코퍼스가
                # 동일한 두 실행에서 160문항 중 98문항의 검색 결과가 달랐다
                # (2026-08-30 사이클 7 실측). 온도를 0으로 고정한다.
                temperature=0.0,
                max_tokens=200,
            )

            content = response.choices[0].message.content.strip()

            # JSON 파싱
            import json

            expanded = json.loads(content)

            # 원본 쿼리를 리스트 맨 앞에 추가
            if isinstance(expanded, list):
                return [query] + expanded[:3]  # 최대 3개까지
            else:
                return [query]

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]

    async def _index_documents(self) -> None:
        """문서 벡터 인덱싱 (:mod:`src.rag.vector_index` 로 위임)"""
        await vector_index.index_documents(
            self.collection, self.openai_client, self.chunks, self._embed_texts
        )

    def _get_cache_key(
        self,
        query: str,
        top_k: int,
        doc_filter: str | None,
        doc_type_filter: list[str] | None = None,
    ) -> str:
        """캐시 키 생성 (:mod:`src.rag.search_cache` 로 위임)"""
        return search_cache.make_key(query, top_k, doc_filter, doc_type_filter)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 확인 (TTL 체크)"""
        return search_cache.is_valid(self._cache_timestamps, cache_key, self.get_cache_ttl())

    def _clean_expired_cache(self) -> None:
        """만료된 캐시 항목 정리"""
        search_cache.clean_expired(self._search_cache, self._cache_timestamps, self.get_cache_ttl())

    async def search(
        self,
        query: str,
        top_k: int = 3,
        doc_filter: str | None = None,
        doc_type_filter: list[str] | None = None,
        use_query_expansion: bool | None = None,
        use_reranking: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        쿼리 기반 문서 검색 (TTL 캐싱 적용)

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            doc_filter: 특정 문서 ID로 필터링
            doc_type_filter: 문서 유형 필터링 (예: ["playbook", "intelligence"])
            use_query_expansion: Query Expansion 사용 여부 (None이면 생성자 설정 따름)
            use_reranking: Reranking 사용 여부 (None이면 생성자 설정 따름)

        Returns:
            검색 결과 리스트
        """
        if not self._initialized:
            await self.initialize()

        # Self-RAG: Check if retrieval is needed
        if not self._needs_retrieval(query):
            return []

        # 파라미터 기본값 설정
        if use_query_expansion is None:
            use_query_expansion = self.use_query_expansion
        if use_reranking is None:
            use_reranking = self.use_reranker and self._reranker_flag_enabled()

        # 캐시 키 생성 및 확인
        cache_key = self._get_cache_key(query, top_k, doc_filter, doc_type_filter)

        if self._is_cache_valid(cache_key):
            return self._search_cache[cache_key]

        # 만료된 캐시 정리 (주기적으로)
        if len(self._cache_timestamps) > 50:
            self._clean_expired_cache()

        # Query Expansion (옵션)
        queries = [query]
        if use_query_expansion:
            queries = await self.expand_query(query)

        # 각 쿼리에 대해 검색 수행
        # 확장 질의별 랭킹을 따로 모아 RRF로 융합한다. 이전에는 순차 append 후
        # all_results[:top_k]로 잘라서, 첫 질의(원본)의 결과만으로 top_k가 채워져
        # **확장 질의의 결과가 단 한 건도 반영되지 않았다** — LLM 호출 비용만
        # 지불하고 효과는 0이었다 (2026-08-30 사이클 6 실측).
        per_query_rankings: list[list[dict]] = []

        for q in queries:
            # 벡터 검색 (필수)
            if self.collection is not None and self.openai_client is not None:
                vector_results = await self._vector_search(
                    q,
                    top_k * 3 if use_reranking else top_k,  # reranking 시 더 많은 후보 검색
                    doc_filter,
                    doc_type_filter,
                )
            else:
                raise RuntimeError("Vector search is required but not available")

            # BM25 검색 (옵션 - RRF 융합용)
            bm25_results = []
            if BM25_AVAILABLE:
                try:
                    bm25_results = await self._bm25_search(
                        q,
                        top_k * 3 if use_reranking else top_k,
                        doc_filter,
                        doc_type_filter,
                    )
                except Exception as e:
                    logger.warning(f"BM25 search failed, falling back to vector-only: {e}")
                    bm25_results = []

            # RRF 융합
            if bm25_results:
                merged_results = self._rrf_merge(vector_results, bm25_results)
            else:
                merged_results = vector_results

            per_query_rankings.append(merged_results)

        all_results = self._fuse_query_rankings(per_query_rankings)

        # Reranking (옵션)
        if use_reranking and all_results:
            if self._reranker is None and RERANKER_AVAILABLE:
                self._reranker = get_reranker()

            if self._reranker:
                # Reranker 호출
                ranked = self._reranker.rerank(query, all_results, top_k=top_k)
                result = [
                    {
                        "id": doc.metadata.get("chunk_id") or doc.metadata.get("id", ""),
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": doc.score,
                    }
                    for doc in ranked
                ]
            else:
                result = all_results[:top_k]
        else:
            result = all_results[:top_k]

        # 결과 캐싱
        self._search_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

        return result

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        doc_filter: str | None,
        doc_type_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """벡터 유사도 검색 (:mod:`src.rag.vector_index` 로 위임)"""
        if not self.openai_client:
            return []
        query_embedding = await self._embed_texts([query])
        if not query_embedding:
            return []

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=vector_index.build_where_filter(doc_filter, doc_type_filter),
            include=["documents", "metadatas", "distances"],
        )
        return vector_index.format_hits(results, self._chunk_index)

    def _needs_retrieval(self, query: str) -> bool:
        """Self-RAG: 검색 필요성 판단 (:mod:`src.rag.selfrag_gate` 로 위임)"""
        return _needs_retrieval_gate(query)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer for Korean+English (:mod:`src.rag.bm25_index` 로 위임)"""
        return bm25_index.tokenize(text)

    def _build_bm25_index(self):
        """Build BM25 index from current chunks"""
        if not BM25_AVAILABLE:
            return
        index, corpus_ids = bm25_index.build(self.chunks)
        if index is not None:
            self._bm25_index = index
            self._bm25_corpus_ids = corpus_ids

    def _rank_bm25(
        self,
        query: str,
        top_k: int,
        doc_filter: str | None = None,
        doc_type_filter: list[str] | None = None,
    ) -> tuple[list[tuple[str, dict[str, Any], float]], float]:
        """BM25 채점 — 두 결과 형태가 공유하는 단일 검색."""
        if self._bm25_index is None:
            self._build_bm25_index()
        return bm25_index.rank(
            self._bm25_index,
            self._bm25_corpus_ids,
            self._chunk_index,
            query,
            top_k=top_k,
            doc_filter=doc_filter,
            doc_type_filter=doc_type_filter,
        )

    async def _bm25_search(
        self,
        query: str,
        top_k: int = 10,
        doc_filter: str | None = None,
        doc_type_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 sparse search — raw scores, vector-search-shaped metadata."""
        if not BM25_AVAILABLE:
            return []
        hits, _max_score = self._rank_bm25(query, top_k, doc_filter, doc_type_filter)
        return bm25_index.as_detailed_results(hits)

    def search_bm25(self, query: str, top_k: int = 10) -> list[dict]:
        """
        BM25 sparse retrieval (synchronous public API).

        Returns list of dicts with keys: id, content, score, metadata, source.
        Scores are normalized to 0-1 range.
        """
        if not BM25_AVAILABLE:
            return []
        hits, max_score = self._rank_bm25(query, top_k)
        return bm25_index.as_normalized_results(hits, max_score)

    @staticmethod
    def _reranker_flag_enabled() -> bool:
        """CrossEncoder 리랭킹 사용 여부 — `retriever.use_reranker` 플래그를 따른다.

        이 리랭커는 `cross-encoder/ms-marco-MiniLM-L-6-v2`(영어 전용)인데
        코퍼스의 88%가 영문 IR, 질의는 한국어라 **교차언어 매칭을 체계적으로
        강등**시킨다. 2026-08-30 사이클 6 실측(한국어 IR 질의 10개, top-8 중
        IR 청크 비중): 벡터검색만 0.600 → BM25 융합 후 0.500 → CrossEncoder
        적용 후 0.263 → 현행 전체 파이프라인 0.138. 골든셋 137문항의 골드 문서
        recall@8도 CrossEncoder 적용 시 0.766 → 0.620으로 떨어진다.

        사이클 1 ablation이 끈 것은 LLM 관련성 채점기(hybrid_retriever)였고
        이 CrossEncoder는 그 결정의 사각지대에 있었다. 두 리랭커를 같은
        플래그 아래로 통일한다.
        """
        try:
            from src.infrastructure.feature_flags import FeatureFlags

            return FeatureFlags.get_instance().use_reranker()
        except Exception:
            logger.debug("feature flag lookup failed, defaulting reranker off", exc_info=True)
            return False

    def _fuse_query_rankings(self, rankings: list[list[dict]], k: int = 60) -> list[dict]:
        """확장 질의별 랭킹을 RRF로 융합. 단일 랭킹이면 그대로 반환."""
        if not rankings:
            return []
        if len(rankings) == 1:
            return rankings[0]
        return _fusion.fuse(rankings, k=k, key=_fusion.key_by_id)

    def _rrf_merge(
        self, dense_results: list[dict], sparse_results: list[dict], k: int = 60
    ) -> list[dict]:
        """Reciprocal Rank Fusion: RRF_score(d) = Σ 1/(k + rank_i(d))"""
        return _fusion.fuse([dense_results, sparse_results], k=k, key=_fusion.key_by_id_or_rank)

    @staticmethod
    def reciprocal_rank_fusion(
        *ranked_lists: list[dict],
        k: int = 60,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion for merging multiple ranked result lists.

        RRF score(d) = sum 1/(k + rank_i(d))

        Args:
            ranked_lists: Variable number of ranked result lists.
            k: RRF constant (default 60).
            top_k: Number of results to return.

        Returns:
            Merged and re-ranked list of results, each carrying ``rrf_score``.
        """
        return _fusion.fuse(
            ranked_lists,
            k=k,
            top_k=top_k,
            key=_fusion.key_by_content_hash,
            annotate=True,
        )

    async def get_document(self, doc_id: str) -> str | None:
        """특정 문서 전체 반환"""
        if not self._initialized:
            await self.initialize()

        return self.documents.get(doc_id)

    async def get_relevant_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        쿼리에 관련된 컨텍스트 반환 (LLM 프롬프트용)

        Args:
            query: 사용자 쿼리
            max_tokens: 최대 토큰 수 (대략적)

        Returns:
            관련 컨텍스트 문자열
        """
        results = await self.search(query, top_k=5)

        context_parts = []
        total_length = 0

        for result in results:
            content = result["content"]
            metadata = result["metadata"]

            # 토큰 제한 체크 (대략 4자 = 1토큰)
            if total_length + len(content) > max_tokens * 4:
                break

            context_parts.append(f"[{metadata.get('title', 'Unknown')}]\n{content}")
            total_length += len(content)

        return "\n\n---\n\n".join(context_parts)
