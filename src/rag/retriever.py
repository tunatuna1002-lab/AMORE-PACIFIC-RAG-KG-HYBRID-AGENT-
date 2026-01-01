"""
Document Retriever
RAG를 위한 문서 검색 모듈

4개 MD 파일 기반:
- Strategic Indicators Definition.md
- Metric Interpretation Guide.md
- Indicator Combination Playbook.md
- Home Page Insight Rules.md
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Lazy import flag - 실제 사용 시 초기화
VECTOR_SEARCH_AVAILABLE = None


class DocumentRetriever:
    """문서 검색 클래스"""

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

    def __init__(self, docs_path: str = "./docs"):
        """
        Args:
            docs_path: 문서 폴더 경로
        """
        self.docs_path = Path(docs_path)
        self.documents: Dict[str, str] = {}
        self.chunks: List[Dict[str, Any]] = []
        self.embedding_model = None
        self.collection = None
        self._initialized = False

    def _check_vector_search(self) -> bool:
        """벡터 검색 가능 여부 확인 (lazy import)"""
        global VECTOR_SEARCH_AVAILABLE
        if VECTOR_SEARCH_AVAILABLE is None:
            try:
                import chromadb
                from sentence_transformers import SentenceTransformer
                VECTOR_SEARCH_AVAILABLE = True
            except ImportError:
                VECTOR_SEARCH_AVAILABLE = False
            except Exception:
                VECTOR_SEARCH_AVAILABLE = False
        return VECTOR_SEARCH_AVAILABLE

    async def initialize(self) -> bool:
        """
        문서 로드 및 벡터 인덱스 초기화

        Returns:
            초기화 성공 여부
        """
        try:
            # 문서 로드
            await self._load_documents()

            # 벡터 검색 초기화 (가능한 경우) - 키워드 검색 폴백으로 충분
            # if self._check_vector_search():
            #     await self._initialize_vector_search()

            self._initialized = True
            return True

        except Exception as e:
            print(f"DocumentRetriever 초기화 실패: {e}")
            # 폴백: 키워드 검색만 사용
            self._initialized = True
            return True

    async def _load_documents(self) -> None:
        """MD 문서 로드"""
        # 프로젝트 루트에서 MD 파일 찾기
        root_path = self.docs_path.parent

        for doc_id, doc_info in self.DOCUMENTS.items():
            # docs 폴더와 루트 폴더 모두 검색
            possible_paths = [
                self.docs_path / doc_info["filename"],
                root_path / doc_info["filename"]
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
        """문서를 청크로 분할"""
        chunks = []
        sections = content.split("\n## ")

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # 섹션 제목 추출
            lines = section.split("\n")
            title = lines[0].replace("#", "").strip() if lines else ""

            # 청크 생성
            text = section.strip()
            if len(text) > chunk_size:
                # 긴 섹션은 추가 분할
                sub_chunks = [text[j:j+chunk_size] for j in range(0, len(text), chunk_size)]
                for k, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        "id": f"{doc_id}_{i}_{k}",
                        "doc_id": doc_id,
                        "title": title,
                        "content": sub_chunk,
                        "keywords": doc_info["keywords"],
                        "description": doc_info["description"]
                    })
            else:
                chunks.append({
                    "id": f"{doc_id}_{i}",
                    "doc_id": doc_id,
                    "title": title,
                    "content": text,
                    "keywords": doc_info["keywords"],
                    "description": doc_info["description"]
                })

        return chunks

    async def _initialize_vector_search(self) -> None:
        """벡터 검색 초기화"""
        if not VECTOR_SEARCH_AVAILABLE:
            return

        # 임베딩 모델 로드
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # ChromaDB 초기화
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))

        # 컬렉션 생성/로드
        self.collection = self.client.get_or_create_collection(
            name="amore_docs",
            metadata={"hnsw:space": "cosine"}
        )

        # 문서 인덱싱
        if self.collection.count() == 0:
            await self._index_documents()

    async def _index_documents(self) -> None:
        """문서 벡터 인덱싱"""
        if not self.collection or not self.embedding_model:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in self.chunks:
            ids.append(chunk["id"])
            documents.append(chunk["content"])
            metadatas.append({
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "description": chunk["description"]
            })

        if documents:
            embeddings = self.embedding_model.encode(documents).tolist()
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

    async def search(
        self,
        query: str,
        top_k: int = 3,
        doc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        쿼리 기반 문서 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            doc_filter: 특정 문서 ID로 필터링

        Returns:
            검색 결과 리스트
        """
        if not self._initialized:
            await self.initialize()

        # 키워드 기반 검색 사용 (벡터 검색은 비활성화)
        return await self._keyword_search(query, top_k, doc_filter)

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        doc_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """벡터 유사도 검색"""
        query_embedding = self.embedding_model.encode([query]).tolist()

        where_filter = {"doc_id": doc_filter} if doc_filter else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # 거리를 유사도로 변환
            })

        return search_results

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        doc_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """키워드 기반 검색 (폴백)"""
        query_lower = query.lower()
        scored_chunks = []

        for chunk in self.chunks:
            if doc_filter and chunk["doc_id"] != doc_filter:
                continue

            score = 0

            # 키워드 매칭
            for keyword in chunk["keywords"]:
                if keyword.lower() in query_lower:
                    score += 2

            # 내용 매칭
            content_lower = chunk["content"].lower()
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 2 and word in content_lower:
                    score += 1

            if score > 0:
                scored_chunks.append({
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "metadata": {
                        "doc_id": chunk["doc_id"],
                        "title": chunk["title"],
                        "description": chunk["description"]
                    },
                    "score": score
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
        max_tokens: int = 2000
    ) -> str:
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
