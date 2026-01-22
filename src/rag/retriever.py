"""
Document Retriever
RAG를 위한 문서 검색 모듈 (Hybrid Search 지원)

4개 MD 파일 기반:
- Strategic Indicators Definition.md
- Metric Interpretation Guide.md
- Indicator Combination Playbook.md
- Home Page Insight Rules.md

## 검색 모드
1. Keyword Search (BM25): 기본 폴백
2. Vector Search: ChromaDB + SentenceTransformers
3. Hybrid Search: BM25 + Vector + RRF (Reciprocal Rank Fusion)

## 청킹 전략
- Semantic Chunking: Markdown 헤딩 기반 섹션 분할
- Overlap: 문맥 보존을 위한 50자 오버랩
- Min/Max: 100~800자 범위
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

# 로거 설정
logger = logging.getLogger(__name__)

# Lazy import flag - 실제 사용 시 초기화
VECTOR_SEARCH_AVAILABLE = None

# Reranker availability flag
RERANKER_AVAILABLE = None


class DocumentRetriever:
    """
    문서 검색 클래스 (Hybrid Search + Semantic Chunking)

    Features:
    - Semantic chunking (Markdown 헤딩 기반)
    - BM25 키워드 검색
    - 벡터 유사도 검색 (ChromaDB)
    - Hybrid search with RRF fusion
    - TTL 캐싱
    """

    # 검색 결과 캐시 (maxsize=100, TTL=5분)
    _search_cache: Dict[str, Any] = {}
    _cache_timestamps: Dict[str, float] = {}
    _CACHE_TTL = 300  # 5분

    # Hybrid Search 설정
    RRF_K = 60  # RRF 상수 (일반적으로 60 사용)
    BM25_WEIGHT = 0.4  # BM25 가중치
    VECTOR_WEIGHT = 0.6  # Vector 가중치

    # Semantic Chunking 설정
    MIN_CHUNK_SIZE = 100  # 최소 청크 크기
    MAX_CHUNK_SIZE = 800  # 최대 청크 크기
    CHUNK_OVERLAP = 50  # 청크 간 오버랩

    # 문서 메타데이터
    DOCUMENTS = {
        "strategic_indicators": {
            "filename": "Strategic Indicators Definition.md",
            "description": "지표 정의 및 산출식",
            "keywords": ["정의", "산출식", "SoS", "HHI", "CPI", "계산", "공식"]
        },
        "metric_interpretation": {
            "filename": "Metric Interpretation Guide.md",
            "description": "지표 해석 가이드",
            "keywords": ["해석", "의미", "높음", "낮음", "주의사항", "함께 봐야"]
        },
        "indicator_combination": {
            "filename": "Indicator Combination Playbook.md",
            "description": "지표 조합 해석 플레이북",
            "keywords": ["조합", "시나리오", "액션", "전략", "상승", "하락"]
        },
        "home_insight_rules": {
            "filename": "Home Page Insight Rules.md",
            "description": "인사이트 생성 규칙",
            "keywords": ["인사이트", "요약", "문구", "템플릿", "톤", "안전장치"]
        }
    }

    def __init__(
        self,
        docs_path: str = "./docs",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        enable_hybrid: bool = True
    ):
        """
        Args:
            docs_path: 문서 폴더 경로
            embedding_model_name: 임베딩 모델 (한국어/영어 지원)
            enable_hybrid: 하이브리드 검색 활성화
        """
        self.docs_path = Path(docs_path)
        self.documents: Dict[str, str] = {}
        self.chunks: List[Dict[str, Any]] = []
        self.embedding_model = None
        self.embedding_model_name = embedding_model_name
        self.collection = None
        self.enable_hybrid = enable_hybrid
        self._initialized = False

        # BM25 인덱스 (키워드 검색용)
        self._bm25_index: Optional[Any] = None
        self._chunk_ids: List[str] = []

        # Reranker (Cross-Encoder)
        self._reranker = None
        self._reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.enable_reranking = True  # Can be disabled

    def _check_vector_search(self) -> bool:
        """벡터 검색 가능 여부 확인 (lazy import)"""
        global VECTOR_SEARCH_AVAILABLE
        if VECTOR_SEARCH_AVAILABLE is None:
            try:
                import chromadb
                from sentence_transformers import SentenceTransformer
                VECTOR_SEARCH_AVAILABLE = True
                logger.info("Vector search available (ChromaDB + SentenceTransformers)")
            except ImportError as e:
                VECTOR_SEARCH_AVAILABLE = False
                logger.warning(f"Vector search not available: {e}")
            except Exception as e:
                VECTOR_SEARCH_AVAILABLE = False
                logger.warning(f"Vector search initialization failed: {e}")
        return VECTOR_SEARCH_AVAILABLE

    def _check_bm25_available(self) -> bool:
        """BM25 라이브러리 가능 여부 확인"""
        try:
            from rank_bm25 import BM25Okapi
            return True
        except ImportError:
            return False

    def _check_reranker_available(self) -> bool:
        """Reranker (Cross-Encoder) 가능 여부 확인"""
        global RERANKER_AVAILABLE
        if RERANKER_AVAILABLE is None:
            try:
                from sentence_transformers import CrossEncoder
                RERANKER_AVAILABLE = True
                logger.info("Reranker available (CrossEncoder)")
            except ImportError:
                RERANKER_AVAILABLE = False
                logger.warning("Reranker not available (CrossEncoder not installed)")
            except Exception as e:
                RERANKER_AVAILABLE = False
                logger.warning(f"Reranker check failed: {e}")
        return RERANKER_AVAILABLE

    async def initialize(self) -> bool:
        """
        문서 로드 및 검색 인덱스 초기화

        Returns:
            초기화 성공 여부
        """
        try:
            # 문서 로드 (Semantic Chunking 적용)
            await self._load_documents()

            # BM25 인덱스 초기화
            if self._check_bm25_available():
                self._initialize_bm25_index()

            # 벡터 검색 초기화 (ChromaDB + SentenceTransformers 설치 시 활성화)
            if self._check_vector_search():
                await self._initialize_vector_search()

            # Reranker 초기화 (Cross-Encoder)
            if self.enable_reranking and self._check_reranker_available():
                await self._initialize_reranker()

            self._initialized = True
            logger.info(
                f"DocumentRetriever initialized: {len(self.chunks)} chunks, "
                f"vector={self.collection is not None}, "
                f"bm25={self._bm25_index is not None}"
            )
            return True

        except Exception as e:
            logger.error(f"DocumentRetriever 초기화 실패: {e}")
            # 폴백: 키워드 검색만 사용
            self._initialized = True
            return True

    async def _load_documents(self) -> None:
        """MD 문서 로드"""
        # 프로젝트 루트에서 MD 파일 찾기
        root_path = self.docs_path.parent
        guides_path = self.docs_path / "guides"  # docs/guides/ 폴더

        for doc_id, doc_info in self.DOCUMENTS.items():
            # docs/guides 폴더, docs 폴더, 루트 폴더 순으로 검색
            possible_paths = [
                guides_path / doc_info["filename"],  # docs/guides/
                self.docs_path / doc_info["filename"],  # docs/
                root_path / doc_info["filename"]  # 프로젝트 루트
            ]

            for file_path in possible_paths:
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        self.documents[doc_id] = content

                        # 청크 분할
                        chunks = self._split_into_chunks(content, doc_id, doc_info)
                        self.chunks.extend(chunks)
                    break

    def _split_into_chunks(
        self,
        content: str,
        doc_id: str,
        doc_info: Dict,
        chunk_size: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Semantic Chunking - Markdown 구조 기반 문서 분할

        전략:
        1. Markdown 헤딩(##, ###)을 기준으로 섹션 분리
        2. 각 섹션이 MAX_CHUNK_SIZE 초과 시 문단/문장 단위로 분할
        3. MIN_CHUNK_SIZE 미만 섹션은 이전 섹션과 병합
        4. 청크 간 CHUNK_OVERLAP 적용으로 문맥 보존
        """
        chunks = []

        # 1. Markdown 헤딩 기반 섹션 분리
        sections = self._split_by_headings(content)

        for i, section in enumerate(sections):
            section_title = section.get("title", "")
            section_content = section.get("content", "").strip()
            section_level = section.get("level", 0)

            if not section_content:
                continue

            # 2. 섹션 크기에 따른 처리
            if len(section_content) <= self.MAX_CHUNK_SIZE:
                # 적절한 크기 - 단일 청크
                chunks.append({
                    "id": f"{doc_id}_{i}",
                    "doc_id": doc_id,
                    "title": section_title,
                    "content": section_content,
                    "keywords": doc_info["keywords"],
                    "description": doc_info["description"],
                    "heading_level": section_level,
                    "section_index": i
                })
            else:
                # 큰 섹션 - 문단/문장 단위 분할
                sub_chunks = self._split_large_section(
                    section_content,
                    section_title,
                    doc_id,
                    doc_info,
                    i
                )
                chunks.extend(sub_chunks)

        # 3. 작은 청크 병합 (MIN_CHUNK_SIZE 미만)
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _split_by_headings(self, content: str) -> List[Dict[str, Any]]:
        """Markdown 헤딩 기반 섹션 분리"""
        sections = []

        # 헤딩 패턴 (##, ###, ####)
        heading_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)

        # 모든 헤딩 위치 찾기
        matches = list(heading_pattern.finditer(content))

        if not matches:
            # 헤딩 없으면 전체를 하나의 섹션으로
            return [{"title": "", "content": content, "level": 0}]

        # 첫 헤딩 이전 내용
        if matches[0].start() > 0:
            intro_content = content[:matches[0].start()].strip()
            if intro_content:
                sections.append({
                    "title": "Introduction",
                    "content": intro_content,
                    "level": 0
                })

        # 각 헤딩 섹션 처리
        for i, match in enumerate(matches):
            level = len(match.group(1))  # # 개수
            title = match.group(2).strip()

            # 섹션 내용: 현재 헤딩부터 다음 헤딩(또는 끝)까지
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()

            # 헤딩 자체도 컨텍스트에 포함
            full_content = f"{match.group(0)}\n{section_content}"

            sections.append({
                "title": title,
                "content": full_content,
                "level": level
            })

        return sections

    def _split_large_section(
        self,
        content: str,
        title: str,
        doc_id: str,
        doc_info: Dict,
        section_idx: int
    ) -> List[Dict[str, Any]]:
        """큰 섹션을 문단/문장 단위로 분할"""
        chunks = []

        # 문단 단위 분리 (빈 줄 기준)
        paragraphs = re.split(r'\n\s*\n', content)

        current_chunk = ""
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 현재 청크에 추가 시 크기 체크
            potential_chunk = current_chunk + "\n\n" + para if current_chunk else para

            if len(potential_chunk) <= self.MAX_CHUNK_SIZE:
                current_chunk = potential_chunk
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append({
                        "id": f"{doc_id}_{section_idx}_{chunk_idx}",
                        "doc_id": doc_id,
                        "title": title,
                        "content": current_chunk,
                        "keywords": doc_info["keywords"],
                        "description": doc_info["description"],
                        "heading_level": 2,
                        "section_index": section_idx,
                        "chunk_index": chunk_idx
                    })
                    chunk_idx += 1

                    # 오버랩 적용 (이전 청크 마지막 부분 포함)
                    overlap = current_chunk[-self.CHUNK_OVERLAP:] if len(current_chunk) > self.CHUNK_OVERLAP else ""
                    current_chunk = overlap + para if overlap else para
                else:
                    # 문단 자체가 너무 큰 경우 - 문장 단위 분할
                    if len(para) > self.MAX_CHUNK_SIZE:
                        sentence_chunks = self._split_by_sentences(para)
                        for sc in sentence_chunks:
                            chunks.append({
                                "id": f"{doc_id}_{section_idx}_{chunk_idx}",
                                "doc_id": doc_id,
                                "title": title,
                                "content": sc,
                                "keywords": doc_info["keywords"],
                                "description": doc_info["description"],
                                "heading_level": 2,
                                "section_index": section_idx,
                                "chunk_index": chunk_idx
                            })
                            chunk_idx += 1
                    else:
                        current_chunk = para

        # 마지막 청크 저장
        if current_chunk:
            chunks.append({
                "id": f"{doc_id}_{section_idx}_{chunk_idx}",
                "doc_id": doc_id,
                "title": title,
                "content": current_chunk,
                "keywords": doc_info["keywords"],
                "description": doc_info["description"],
                "heading_level": 2,
                "section_index": section_idx,
                "chunk_index": chunk_idx
            })

        return chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """문장 단위 분할 (큰 문단용)"""
        # 한국어/영어 문장 종결 패턴
        sentence_endings = re.compile(r'([.!?。！？]\s+|[.!?。！？]$)')
        sentences = sentence_endings.split(text)

        # 문장 재조합
        result = []
        current = ""

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            ending = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sentence = sentence + ending

            if len(current) + len(full_sentence) <= self.MAX_CHUNK_SIZE:
                current += full_sentence
            else:
                if current:
                    result.append(current.strip())
                current = full_sentence

        if current:
            result.append(current.strip())

        return result

    def _merge_small_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """작은 청크 병합"""
        if not chunks:
            return chunks

        merged = []
        current = None

        for chunk in chunks:
            if current is None:
                current = chunk.copy()
                continue

            # 같은 문서 && 작은 청크 → 병합
            if (
                chunk["doc_id"] == current["doc_id"] and
                len(current["content"]) < self.MIN_CHUNK_SIZE
            ):
                # 병합
                current["content"] = current["content"] + "\n\n" + chunk["content"]
                current["title"] = current["title"] or chunk["title"]
            else:
                merged.append(current)
                current = chunk.copy()

        if current:
            merged.append(current)

        return merged

    def _initialize_bm25_index(self) -> None:
        """BM25 인덱스 초기화"""
        try:
            from rank_bm25 import BM25Okapi

            # 청크 텍스트 토큰화
            tokenized_corpus = []
            self._chunk_ids = []

            for chunk in self.chunks:
                # 간단한 토큰화 (한국어/영어 지원)
                text = chunk["content"].lower()
                # 공백, 구두점 기준 분리
                tokens = re.findall(r'\b\w+\b|[\uAC00-\uD7AF]+', text)
                tokenized_corpus.append(tokens)
                self._chunk_ids.append(chunk["id"])

            if tokenized_corpus:
                self._bm25_index = BM25Okapi(tokenized_corpus)
                logger.info(f"BM25 index initialized: {len(tokenized_corpus)} documents")

        except ImportError:
            logger.warning("rank_bm25 not installed, BM25 search disabled")
            self._bm25_index = None
        except Exception as e:
            logger.warning(f"BM25 initialization failed: {e}")
            self._bm25_index = None

    async def _initialize_vector_search(self) -> None:
        """벡터 검색 초기화 (다국어 임베딩 모델)"""
        if not VECTOR_SEARCH_AVAILABLE:
            return

        try:
            # Keras 호환성 이슈 우회를 위한 환경변수 설정
            os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

            from sentence_transformers import SentenceTransformer
            import chromadb

            # 임베딩 모델 로드
            # all-MiniLM-L6-v2: 작고 빠름, 영어 최적화
            # intfloat/multilingual-e5-small: 다국어 지원 (한국어 포함)
            logger.info(f"Loading embedding model: {self.embedding_model_name}")

            try:
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    trust_remote_code=True
                )
            except Exception as model_error:
                # 폴백: 기본 모델 사용
                logger.warning(f"Failed to load {self.embedding_model_name}: {model_error}")
                logger.info("Falling back to all-MiniLM-L6-v2")
                self.embedding_model_name = "all-MiniLM-L6-v2"
                self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # ChromaDB 초기화 (modern API - persistent client)
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
            os.makedirs(persist_dir, exist_ok=True)

            self.client = chromadb.PersistentClient(path=persist_dir)

            # 컬렉션 생성/로드 (모델 이름 포함하여 구분)
            collection_name = f"amore_docs_{self.embedding_model_name.replace('/', '_').replace('-', '_')}"
            self.collection = self.client.get_or_create_collection(
                name=collection_name[:63],  # ChromaDB 이름 길이 제한
                metadata={"hnsw:space": "cosine"}
            )

            # 문서 인덱싱 (컬렉션이 비어있거나 청크 수가 다를 때)
            if self.collection.count() != len(self.chunks):
                # 기존 컬렉션 삭제 후 재생성
                if self.collection.count() > 0:
                    self.client.delete_collection(collection_name[:63])
                    self.collection = self.client.get_or_create_collection(
                        name=collection_name[:63],
                        metadata={"hnsw:space": "cosine"}
                    )
                await self._index_documents()

            logger.info(f"ChromaDB initialized: {self.collection.count()} documents indexed")

        except Exception as e:
            logger.warning(f"Vector search initialization failed (fallback to BM25/keyword): {e}")
            self.collection = None
            self.embedding_model = None

    async def _initialize_reranker(self) -> None:
        """Cross-Encoder Reranker 초기화"""
        if not RERANKER_AVAILABLE:
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model: {self._reranker_model_name}")
            self._reranker = CrossEncoder(
                self._reranker_model_name,
                max_length=512
            )
            logger.info("Reranker initialized successfully")

        except Exception as e:
            logger.warning(f"Reranker initialization failed: {e}")
            self._reranker = None

    async def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Cross-Encoder를 사용한 결과 재순위화

        Args:
            query: 검색 쿼리
            results: 초기 검색 결과
            top_k: 최종 반환할 결과 수

        Returns:
            재순위화된 결과
        """
        if not self._reranker or not results:
            return results[:top_k]

        try:
            # Query-Document 쌍 생성
            pairs = [(query, r["content"][:512]) for r in results]  # 512자 제한

            # Cross-Encoder 점수 계산
            scores = self._reranker.predict(pairs)

            # 점수 추가 및 재정렬
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])

            # 재순위화
            reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

            logger.debug(f"Reranked {len(results)} results")
            return reranked[:top_k]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return results[:top_k]

    async def _index_documents(self) -> None:
        """문서 벡터 인덱싱 (E5 모델 프리픽스 적용)"""
        if not self.collection or not self.embedding_model:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in self.chunks:
            ids.append(chunk["id"])
            # E5 모델은 "passage: " 프리픽스 사용
            if "e5" in self.embedding_model_name.lower():
                documents.append(f"passage: {chunk['content']}")
            else:
                documents.append(chunk["content"])
            metadatas.append({
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "description": chunk["description"],
                "heading_level": chunk.get("heading_level", 0),
                "section_index": chunk.get("section_index", 0)
            })

        if documents:
            logger.info(f"Indexing {len(documents)} documents...")
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Indexed {len(documents)} documents")

    def _get_cache_key(self, query: str, top_k: int, doc_filter: Optional[str]) -> str:
        """캐시 키 생성"""
        return f"{query}:{top_k}:{doc_filter or 'all'}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 확인 (TTL 체크)"""
        if cache_key not in self._cache_timestamps:
            return False
        elapsed = time.time() - self._cache_timestamps[cache_key]
        return elapsed < self._CACHE_TTL

    def _clean_expired_cache(self) -> None:
        """만료된 캐시 항목 정리"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp >= self._CACHE_TTL
        ]
        for key in expired_keys:
            self._search_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

    async def search(
        self,
        query: str,
        top_k: int = 3,
        doc_filter: Optional[str] = None,
        search_mode: str = "auto"
    ) -> List[Dict[str, Any]]:
        """
        쿼리 기반 문서 검색 (Hybrid Search + TTL 캐싱)

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            doc_filter: 특정 문서 ID로 필터링
            search_mode: 검색 모드 ("auto", "hybrid", "vector", "bm25", "keyword")

        Returns:
            검색 결과 리스트
        """
        if not self._initialized:
            await self.initialize()

        # 캐시 키 생성 및 확인
        cache_key = self._get_cache_key(query, top_k, doc_filter) + f":{search_mode}"

        if self._is_cache_valid(cache_key):
            return self._search_cache[cache_key]

        # 만료된 캐시 정리 (주기적으로)
        if len(self._cache_timestamps) > 50:
            self._clean_expired_cache()

        # 검색 모드 결정
        if search_mode == "auto":
            # Auto: 가능한 최선의 방법 선택
            if self.enable_hybrid and self._bm25_index and self.collection:
                search_mode = "hybrid"
            elif self.collection:
                search_mode = "vector"
            elif self._bm25_index:
                search_mode = "bm25"
            else:
                search_mode = "keyword"

        # 검색 실행
        if search_mode == "hybrid" and self._bm25_index and self.collection:
            result = await self._hybrid_search(query, top_k, doc_filter)
        elif search_mode == "vector" and self.collection:
            result = await self._vector_search(query, top_k, doc_filter)
        elif search_mode == "bm25" and self._bm25_index:
            result = await self._bm25_search(query, top_k, doc_filter)
        else:
            result = await self._keyword_search(query, top_k, doc_filter)

        # Reranking 적용 (활성화된 경우)
        if self.enable_reranking and self._reranker and len(result) > top_k:
            result = await self._rerank_results(query, result, top_k)

        # 결과 캐싱
        self._search_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

        return result

    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        doc_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Search: BM25 + Vector + RRF Fusion

        RRF (Reciprocal Rank Fusion):
        score = sum(1 / (k + rank_i)) for each retriever i

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            doc_filter: 문서 필터

        Returns:
            RRF로 융합된 검색 결과
        """
        # 더 많은 후보 가져오기 (fusion을 위해)
        candidate_k = min(top_k * 3, len(self.chunks))

        # 1. BM25 검색
        bm25_results = await self._bm25_search(query, candidate_k, doc_filter)
        bm25_ranks = {r["id"]: i + 1 for i, r in enumerate(bm25_results)}

        # 2. Vector 검색
        vector_results = await self._vector_search(query, candidate_k, doc_filter)
        vector_ranks = {r["id"]: i + 1 for i, r in enumerate(vector_results)}

        # 3. RRF Fusion
        all_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        rrf_scores = {}

        for doc_id in all_ids:
            bm25_rank = bm25_ranks.get(doc_id, candidate_k + 1)
            vector_rank = vector_ranks.get(doc_id, candidate_k + 1)

            # RRF 점수 계산 (가중치 적용)
            bm25_rrf = self.BM25_WEIGHT / (self.RRF_K + bm25_rank)
            vector_rrf = self.VECTOR_WEIGHT / (self.RRF_K + vector_rank)
            rrf_scores[doc_id] = bm25_rrf + vector_rrf

        # 4. 점수순 정렬
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 5. 결과 조합
        results = []
        id_to_result = {r["id"]: r for r in bm25_results + vector_results}

        for doc_id in sorted_ids[:top_k]:
            if doc_id in id_to_result:
                result = id_to_result[doc_id].copy()
                result["rrf_score"] = rrf_scores[doc_id]
                result["bm25_rank"] = bm25_ranks.get(doc_id)
                result["vector_rank"] = vector_ranks.get(doc_id)
                result["search_mode"] = "hybrid"
                results.append(result)

        logger.debug(f"Hybrid search: {len(results)} results (BM25: {len(bm25_results)}, Vector: {len(vector_results)})")
        return results

    async def _bm25_search(
        self,
        query: str,
        top_k: int,
        doc_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """BM25 검색"""
        if not self._bm25_index:
            return await self._keyword_search(query, top_k, doc_filter)

        # 쿼리 토큰화
        query_lower = query.lower()
        query_tokens = re.findall(r'\b\w+\b|[\uAC00-\uD7AF]+', query_lower)

        # BM25 점수 계산
        scores = self._bm25_index.get_scores(query_tokens)

        # 점수와 청크 매핑
        scored_chunks = []
        for i, score in enumerate(scores):
            if score > 0:
                chunk_id = self._chunk_ids[i]
                chunk = next((c for c in self.chunks if c["id"] == chunk_id), None)
                if chunk:
                    if doc_filter and chunk["doc_id"] != doc_filter:
                        continue
                    scored_chunks.append({
                        "id": chunk["id"],
                        "content": chunk["content"],
                        "metadata": {
                            "doc_id": chunk["doc_id"],
                            "title": chunk["title"],
                            "description": chunk["description"]
                        },
                        "score": float(score),
                        "search_mode": "bm25"
                    })

        # 점수순 정렬
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        doc_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """벡터 유사도 검색 (E5 모델 프리픽스 적용)"""
        # E5 모델은 "query: " 프리픽스 사용
        if "e5" in self.embedding_model_name.lower():
            query_text = f"query: {query}"
        else:
            query_text = query

        query_embedding = self.embedding_model.encode(
            [query_text],
            convert_to_numpy=True
        ).tolist()

        where_filter = {"doc_id": doc_filter} if doc_filter else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            content = results["documents"][0][i]
            # E5 프리픽스 제거
            if content.startswith("passage: "):
                content = content[9:]

            search_results.append({
                "id": results["ids"][0][i],
                "content": content,
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # 거리를 유사도로 변환
                "search_mode": "vector"
            })

        return search_results

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        doc_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """키워드 기반 검색 (최종 폴백)"""
        query_lower = query.lower()
        scored_chunks = []

        for chunk in self.chunks:
            if doc_filter and chunk["doc_id"] != doc_filter:
                continue

            score = 0

            # 키워드 매칭 (문서별 키워드)
            for keyword in chunk.get("keywords", []):
                if keyword.lower() in query_lower:
                    score += 2

            # 내용 매칭 (단어 단위)
            content_lower = chunk["content"].lower()
            query_words = re.findall(r'\b\w+\b|[\uAC00-\uD7AF]+', query_lower)
            for word in query_words:
                if len(word) > 2 and word in content_lower:
                    score += 1

            # 제목 매칭 (높은 가중치)
            title_lower = chunk.get("title", "").lower()
            for word in query_words:
                if len(word) > 2 and word in title_lower:
                    score += 3

            if score > 0:
                scored_chunks.append({
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "metadata": {
                        "doc_id": chunk["doc_id"],
                        "title": chunk["title"],
                        "description": chunk["description"]
                    },
                    "score": score,
                    "search_mode": "keyword"
                })

        # 점수순 정렬
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        return scored_chunks[:top_k]

    async def get_document(self, doc_id: str) -> Optional[str]:
        """특정 문서 전체 반환"""
        if not self._initialized:
            await self.initialize()

        return self.documents.get(doc_id)

    async def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 2000,
        search_mode: str = "auto"
    ) -> str:
        """
        쿼리에 관련된 컨텍스트 반환 (LLM 프롬프트용)

        Args:
            query: 사용자 쿼리
            max_tokens: 최대 토큰 수 (대략적)
            search_mode: 검색 모드

        Returns:
            관련 컨텍스트 문자열
        """
        results = await self.search(query, top_k=5, search_mode=search_mode)

        context_parts = []
        total_length = 0

        for result in results:
            content = result["content"]
            metadata = result["metadata"]

            # 토큰 제한 체크 (대략 4자 = 1토큰)
            if total_length + len(content) > max_tokens * 4:
                break

            # 검색 모드 표시
            mode = result.get("search_mode", "unknown")
            score_info = ""
            if "rrf_score" in result:
                score_info = f" (RRF: {result['rrf_score']:.4f})"
            elif "score" in result:
                score_info = f" (Score: {result['score']:.4f})"

            context_parts.append(
                f"[{metadata.get('title', 'Unknown')}]{score_info}\n{content}"
            )
            total_length += len(content)

        return "\n\n---\n\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """검색기 통계 반환"""
        return {
            "initialized": self._initialized,
            "documents_count": len(self.documents),
            "chunks_count": len(self.chunks),
            "vector_search_enabled": self.collection is not None,
            "bm25_search_enabled": self._bm25_index is not None,
            "hybrid_search_enabled": self.enable_hybrid and self._bm25_index and self.collection,
            "embedding_model": self.embedding_model_name if self.embedding_model else None,
            "collection_count": self.collection.count() if self.collection else 0,
            "cache_size": len(self._search_cache),
            "reranker_enabled": self._reranker is not None,
            "reranker_model": self._reranker_model_name if self._reranker else None,
            "chunking_config": {
                "min_size": self.MIN_CHUNK_SIZE,
                "max_size": self.MAX_CHUNK_SIZE,
                "overlap": self.CHUNK_OVERLAP
            },
            "hybrid_config": {
                "rrf_k": self.RRF_K,
                "bm25_weight": self.BM25_WEIGHT,
                "vector_weight": self.VECTOR_WEIGHT
            }
        }
