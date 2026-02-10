"""
Semantic Chunker
================
의미 기반 문서 청킹 모듈

기존 고정 크기 청킹의 문제점:
- 문장 중간에서 끊김
- 의미적으로 관련된 문장이 분리됨
- 청크 경계에서 컨텍스트 손실

Semantic Chunking 접근법:
1. 문장 단위로 분할
2. 각 문장을 임베딩
3. 인접 문장 간 코사인 유사도 계산
4. 유사도 급감 지점을 청크 경계로 설정
5. 의미적으로 연관된 문장들을 하나의 청크로 그룹화
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """청크 데이터 클래스"""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    start_idx: int = 0
    end_idx: int = 0
    sentence_count: int = 0
    avg_similarity: float = 0.0


class SemanticChunker:
    """
    의미 기반 문서 청킹

    특징:
    - 문장 임베딩 기반 유사도 분석
    - 동적 청크 경계 결정
    - 최대/최소 청크 크기 제약
    - 코사인 유사도 기반 breakpoint 탐지

    사용 예:
        chunker = SemanticChunker()
        chunks = chunker.chunk(text, doc_id="doc_001")
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        similarity_threshold: float = 0.5,
        percentile_threshold: int = 25,
        overlap_sentences: int = 1,
    ):
        """
        Args:
            embedding_model: OpenAI 임베딩 모델명
            min_chunk_size: 최소 청크 크기 (문자 수)
            max_chunk_size: 최대 청크 크기 (문자 수)
            similarity_threshold: 절대 유사도 임계값
            percentile_threshold: 유사도 하위 백분위 (breakpoint 기준)
            overlap_sentences: 청크 간 중첩 문장 수
        """
        self.embedding_model = embedding_model
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.percentile_threshold = percentile_threshold
        self.overlap_sentences = overlap_sentences

        self.openai_client = None
        self._initialized = False

    def _initialize(self) -> bool:
        """OpenAI 클라이언트 초기화"""
        if self._initialized:
            return True

        try:
            import openai

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set, falling back to simple chunking")
                return False

            self.openai_client = openai.OpenAI(api_key=api_key)
            self._initialized = True
            return True
        except ImportError:
            logger.warning("openai package not installed, falling back to simple chunking")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            return False

    def _split_sentences(self, text: str) -> list[str]:
        """
        텍스트를 문장 단위로 분할

        한국어/영어 혼합 텍스트 지원
        """
        # 문장 종결 패턴: .!? 뒤에 공백 또는 줄바꿈
        # 단, 숫자.숫자 (예: 3.14), 약어 (예: Mr., Dr.) 제외
        sentence_pattern = r"(?<!\d)(?<![A-Z])([.!?])\s+"

        # 줄바꿈도 문장 구분자로 처리
        text = re.sub(r"\n+", "\n", text)

        # 문장 분할
        sentences = re.split(sentence_pattern, text)

        # 빈 문장 제거 및 정리
        result = []
        current = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if s in ".!?":
                current += s
                if current.strip():
                    result.append(current.strip())
                current = ""
            else:
                if current:
                    result.append(current.strip())
                current = s

        if current.strip():
            result.append(current.strip())

        # 줄바꿈으로 추가 분할
        final_result = []
        for sentence in result:
            parts = sentence.split("\n")
            for part in parts:
                part = part.strip()
                if part:
                    final_result.append(part)

        return final_result

    def _embed_sentences(self, sentences: list[str]) -> list[list[float]]:
        """문장들을 임베딩"""
        if not self.openai_client or not sentences:
            return []

        try:
            # 배치 임베딩
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, input=sentences
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """코사인 유사도 계산"""
        a_np = np.array(a)
        b_np = np.array(b)

        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a_np, b_np) / (norm_a * norm_b))

    def _find_breakpoints(self, embeddings: list[list[float]], sentences: list[str]) -> list[int]:
        """
        유사도 기반 청크 경계점 탐지

        Args:
            embeddings: 문장 임베딩 리스트
            sentences: 문장 리스트

        Returns:
            경계점 인덱스 리스트
        """
        if len(embeddings) < 2:
            return []

        # 인접 문장 간 유사도 계산
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        if not similarities:
            return []

        # 유사도 임계값 계산 (하위 percentile_threshold%)
        threshold = np.percentile(similarities, self.percentile_threshold)
        threshold = max(threshold, self.similarity_threshold)

        # 경계점 탐지 (유사도가 임계값 이하인 지점)
        breakpoints = []
        current_chunk_size = 0

        for i, sim in enumerate(similarities):
            sentence_len = len(sentences[i])
            current_chunk_size += sentence_len

            # 경계점 조건:
            # 1. 유사도가 임계값 이하
            # 2. 최소 청크 크기 충족
            # 3. 또는 최대 청크 크기 초과
            if (
                sim < threshold and current_chunk_size >= self.min_chunk_size
            ) or current_chunk_size >= self.max_chunk_size:
                breakpoints.append(i + 1)  # 다음 문장이 새 청크 시작
                current_chunk_size = 0

        return breakpoints

    def _create_chunks_from_breakpoints(
        self,
        sentences: list[str],
        embeddings: list[list[float]],
        breakpoints: list[int],
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """경계점 기반 청크 생성"""
        chunks = []

        # 경계점에 시작점과 끝점 추가
        all_points = [0] + breakpoints + [len(sentences)]

        for i in range(len(all_points) - 1):
            start_idx = all_points[i]
            end_idx = all_points[i + 1]

            # overlap: 이전 청크의 마지막 N문장을 현재 청크 시작에 추가
            if self.overlap_sentences > 0 and i > 0:
                overlap_start = max(0, start_idx - self.overlap_sentences)
                overlap_sents = sentences[overlap_start:start_idx]
                chunk_sentences = overlap_sents + sentences[start_idx:end_idx]
            else:
                chunk_sentences = sentences[start_idx:end_idx]
            chunk_content = " ".join(chunk_sentences)

            # 청크 임베딩 계산 (문장 임베딩의 평균)
            chunk_embedding = None
            avg_sim = 0.0
            if embeddings and start_idx < len(embeddings):
                chunk_embeddings = embeddings[start_idx:end_idx]
                if chunk_embeddings:
                    chunk_embedding = list(np.mean(chunk_embeddings, axis=0))

                    # 청크 내 평균 유사도
                    if len(chunk_embeddings) > 1:
                        sims = []
                        for j in range(len(chunk_embeddings) - 1):
                            sims.append(
                                self._cosine_similarity(
                                    chunk_embeddings[j], chunk_embeddings[j + 1]
                                )
                            )
                        avg_sim = float(np.mean(sims)) if sims else 0.0

            chunk = Chunk(
                id=f"{doc_id}_chunk_{i}",
                content=chunk_content,
                metadata={
                    **(metadata or {}),
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "chunking_method": "semantic",
                    "has_overlap": self.overlap_sentences > 0 and i > 0,
                    "overlap_sentences": self.overlap_sentences if i > 0 else 0,
                },
                embedding=chunk_embedding,
                start_idx=start_idx,
                end_idx=end_idx,
                sentence_count=len(chunk_sentences),
                avg_similarity=avg_sim,
            )
            chunks.append(chunk)

        return chunks

    def _simple_chunk(
        self, text: str, doc_id: str, metadata: dict[str, Any] | None = None
    ) -> list[Chunk]:
        """
        폴백: 단순 크기 기반 청킹

        임베딩 실패 시 사용
        """
        sentences = self._split_sentences(text)
        chunks = []

        current_chunk = []
        current_size = 0
        chunk_idx = 0
        prev_chunk_tail: list = []  # 이전 청크의 마지막 N문장

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_size + sentence_len > self.max_chunk_size and current_chunk:
                # 현재 청크 저장
                chunk_content = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        id=f"{doc_id}_chunk_{chunk_idx}",
                        content=chunk_content,
                        metadata={
                            **(metadata or {}),
                            "doc_id": doc_id,
                            "chunk_index": chunk_idx,
                            "chunking_method": "simple",
                            "has_overlap": len(prev_chunk_tail) > 0,
                        },
                        sentence_count=len(current_chunk),
                    )
                )
                chunk_idx += 1
                # overlap: 이전 청크의 마지막 N문장 보존
                if self.overlap_sentences > 0:
                    prev_chunk_tail = current_chunk[-self.overlap_sentences :]
                    current_chunk = list(prev_chunk_tail)
                    current_size = sum(len(s) + 1 for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_len + 1  # +1 for space

        # 마지막 청크
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{chunk_idx}",
                    content=chunk_content,
                    metadata={
                        **(metadata or {}),
                        "doc_id": doc_id,
                        "chunk_index": chunk_idx,
                        "chunking_method": "simple",
                        "has_overlap": len(prev_chunk_tail) > 0 and chunk_idx > 0,
                    },
                    sentence_count=len(current_chunk),
                )
            )

        return chunks

    def chunk(
        self, text: str, doc_id: str = "doc", metadata: dict[str, Any] | None = None
    ) -> list[Chunk]:
        """
        텍스트를 의미 단위로 청킹

        Args:
            text: 청킹할 텍스트
            doc_id: 문서 ID
            metadata: 청크에 추가할 메타데이터

        Returns:
            Chunk 리스트
        """
        if not text or not text.strip():
            return []

        # OpenAI 클라이언트 초기화 시도
        if not self._initialize():
            # 폴백: 단순 청킹
            logger.info("Using simple chunking (embedding not available)")
            return self._simple_chunk(text, doc_id, metadata)

        # 문장 분할
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            # 문장이 하나면 그대로 반환
            return [
                Chunk(
                    id=f"{doc_id}_chunk_0",
                    content=sentences[0],
                    metadata={
                        **(metadata or {}),
                        "doc_id": doc_id,
                        "chunk_index": 0,
                        "chunking_method": "semantic",
                    },
                    sentence_count=1,
                )
            ]

        # 문장 임베딩
        embeddings = self._embed_sentences(sentences)

        if not embeddings or len(embeddings) != len(sentences):
            # 임베딩 실패 시 폴백
            logger.warning("Embedding failed, falling back to simple chunking")
            return self._simple_chunk(text, doc_id, metadata)

        # 경계점 탐지
        breakpoints = self._find_breakpoints(embeddings, sentences)

        # 청크 생성
        chunks = self._create_chunks_from_breakpoints(
            sentences, embeddings, breakpoints, doc_id, metadata
        )

        logger.info(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")

        return chunks

    def chunk_document(self, content: str, doc_info: dict[str, Any]) -> list[dict[str, Any]]:
        """
        DocumentRetriever 호환 인터페이스

        Args:
            content: 문서 내용
            doc_info: 문서 메타데이터 (DOCUMENTS 딕셔너리의 값)

        Returns:
            기존 retriever 포맷의 청크 리스트
        """
        doc_id = doc_info.get("filename", "doc").replace(".md", "").replace(" ", "_")

        chunks = self.chunk(
            text=content,
            doc_id=doc_id,
            metadata={
                "doc_type": doc_info.get("doc_type", "metric_guide"),
                "description": doc_info.get("description", ""),
                "keywords": doc_info.get("keywords", []),
                "source_filename": doc_info.get("filename", ""),
            },
        )

        # 기존 포맷으로 변환
        return [
            {
                "id": chunk.id,
                "doc_id": doc_id,
                "doc_type": chunk.metadata.get("doc_type", "metric_guide"),
                "title": chunk.metadata.get("title", ""),
                "content": chunk.content,
                "content_type": "text",
                "source_filename": chunk.metadata.get("source_filename", ""),
                "keywords": chunk.metadata.get("keywords", []),
                "description": chunk.metadata.get("description", ""),
                "embedding": chunk.embedding,
                "sentence_count": chunk.sentence_count,
                "avg_similarity": chunk.avg_similarity,
            }
            for chunk in chunks
        ]


# 싱글톤 인스턴스
_chunker_instance: SemanticChunker | None = None


def get_semantic_chunker() -> SemanticChunker:
    """SemanticChunker 싱글톤 인스턴스 반환"""
    global _chunker_instance
    if _chunker_instance is None:
        _chunker_instance = SemanticChunker()
    return _chunker_instance
