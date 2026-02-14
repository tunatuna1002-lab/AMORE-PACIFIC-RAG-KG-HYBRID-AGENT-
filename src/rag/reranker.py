"""
Cross-Encoder Reranker
======================
검색 결과 재순위화 모듈

Bi-Encoder vs Cross-Encoder:
- Bi-Encoder: 쿼리와 문서를 독립적으로 임베딩 후 유사도 계산 (빠름, 정확도 낮음)
- Cross-Encoder: 쿼리-문서 쌍을 함께 입력하여 관련도 직접 예측 (느림, 정확도 높음)

Two-Stage Retrieval:
1. Stage 1 (Recall): Bi-Encoder로 빠르게 후보군 추출 (Top-100)
2. Stage 2 (Precision): Cross-Encoder로 정밀 재순위화 (Top-5)
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RankedDocument:
    """재순위화된 문서"""

    content: str
    score: float
    original_rank: int
    new_rank: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "original_rank": self.original_rank,
            "new_rank": self.new_rank,
            "metadata": self.metadata,
        }


class CrossEncoderReranker:
    """
    Cross-Encoder 기반 재순위화

    sentence-transformers의 CrossEncoder 또는
    OpenAI API를 활용한 재순위화 지원

    사용 예:
        reranker = CrossEncoderReranker()
        ranked = reranker.rerank(query, documents, top_k=5)
    """

    # 지원되는 Cross-Encoder 모델
    SUPPORTED_MODELS = {
        "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "ms-marco-TinyBERT-L-6": "cross-encoder/ms-marco-TinyBERT-L-6",
        "stsb-roberta-base": "cross-encoder/stsb-roberta-base",
        "bge-reranker-base": "BAAI/bge-reranker-base",
        "bge-reranker-large": "BAAI/bge-reranker-large",
    }

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-6-v2",
        use_openai: bool = False,
        openai_model: str = "gpt-4.1-mini",
        device: str = "cpu",
    ):
        """
        Args:
            model_name: Cross-Encoder 모델명 (sentence-transformers용)
            use_openai: OpenAI API 사용 여부 (기본 False - 로컬 Cross-Encoder 우선)
            openai_model: OpenAI 모델명
            device: PyTorch 디바이스 (cpu, cuda, mps)
        """
        self.model_name = model_name
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.device = device

        self.cross_encoder = None
        self.openai_client = None
        self._initialized = False

    def _initialize(self) -> bool:
        """모델 초기화 (로컬 Cross-Encoder 우선)"""
        if self._initialized:
            return True

        # 우선순위 1: 로컬 Cross-Encoder (use_openai=False일 때 우선)
        if not self.use_openai:
            try:
                from sentence_transformers import CrossEncoder

                model_path = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)

                logger.info(f"Initializing local CrossEncoder: {model_path}")
                logger.info("First-time model download may take a few minutes...")

                self.cross_encoder = CrossEncoder(model_path, device=self.device)
                self._initialized = True
                logger.info(f"Reranker initialized with local CrossEncoder: {model_path}")
                return True
            except ImportError:
                logger.warning("sentence-transformers not installed, falling back to OpenAI")
            except Exception as e:
                logger.warning(f"CrossEncoder initialization failed: {e}, falling back to OpenAI")

        # 우선순위 2: OpenAI (use_openai=True 또는 로컬 실패 시)
        try:
            import openai

            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                self._initialized = True
                logger.info("Reranker initialized with OpenAI")
                return True
        except ImportError:
            logger.warning("openai package not installed")
        except Exception as e:
            logger.warning(f"OpenAI initialization failed: {e}")

        # 최종 폴백: 로컬 Cross-Encoder 재시도
        if not self.cross_encoder:
            try:
                from sentence_transformers import CrossEncoder

                model_path = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)

                logger.info(f"Final attempt: Initializing local CrossEncoder: {model_path}")
                self.cross_encoder = CrossEncoder(model_path, device=self.device)
                self._initialized = True
                logger.info(f"Reranker initialized with local CrossEncoder: {model_path}")
                return True
            except Exception as e:
                logger.error(f"Final CrossEncoder initialization failed: {e}")

        return False

    def _rerank_with_openai(
        self, query: str, documents: list[str], top_k: int
    ) -> list[tuple[str, float]]:
        """
        OpenAI API를 사용한 재순위화

        LLM에게 관련도 점수를 직접 예측하게 함
        """
        if not self.openai_client:
            return [(doc, 0.0) for doc in documents]

        scored_docs = []

        # 배치 처리 (한 번에 모든 문서 점수화)
        prompt = f"""You are a relevance scoring system. Score each document's relevance to the query on a scale of 0-10.

Query: {query}

Documents to score:
"""
        for i, doc in enumerate(documents):
            # 문서 길이 제한 (토큰 절약)
            truncated_doc = doc[:500] if len(doc) > 500 else doc
            prompt += f"\n[Document {i + 1}]: {truncated_doc}\n"

        prompt += """
Return ONLY a JSON array of scores in order, like: [8.5, 6.2, 9.1, ...]
Do not include any explanation, just the JSON array."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a precise relevance scoring system."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=200,
            )

            # 응답 파싱
            content = response.choices[0].message.content.strip()

            # JSON 배열 추출
            import json
            import re

            # [숫자, 숫자, ...] 패턴 찾기
            match = re.search(r"\[[\d.,\s]+\]", content)
            if match:
                scores = json.loads(match.group())

                # 점수와 문서 매칭
                for doc, score in zip(documents, scores, strict=False):
                    scored_docs.append((doc, float(score) / 10.0))  # 0-1 정규화
            else:
                # 파싱 실패 시 원래 순서 유지
                logger.warning(f"Failed to parse scores from: {content}")
                scored_docs = [(doc, 1.0 - i * 0.1) for i, doc in enumerate(documents)]

        except Exception as e:
            logger.error(f"OpenAI reranking failed: {e}")
            scored_docs = [(doc, 1.0 - i * 0.1) for i, doc in enumerate(documents)]

        # 점수순 정렬
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]

    def _rerank_with_cross_encoder(
        self, query: str, documents: list[str], top_k: int, batch_size: int = 32
    ) -> list[tuple[str, float]]:
        """
        CrossEncoder 모델을 사용한 재순위화 (배치 처리 최적화)

        Args:
            query: 검색 쿼리
            documents: 문서 리스트
            top_k: 반환할 상위 K개
            batch_size: 배치 처리 크기 (기본 32)
        """
        if not self.cross_encoder:
            return [(doc, 0.0) for doc in documents]

        # 쿼리-문서 쌍 생성
        pairs = [(query, doc) for doc in documents]

        try:
            # Cross-Encoder 점수 계산 (배치 처리)
            scores = self.cross_encoder.predict(
                pairs, batch_size=batch_size, show_progress_bar=False
            )

            # 점수와 문서 매칭
            scored_docs = list(zip(documents, scores, strict=False))

            # 점수순 정렬
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return scored_docs[:top_k]

        except Exception as e:
            logger.error(f"CrossEncoder reranking failed: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]

    def rerank(
        self, query: str, documents: list[str] | list[dict[str, Any]], top_k: int = 5
    ) -> list[RankedDocument]:
        """
        검색 결과 재순위화

        Args:
            query: 검색 쿼리
            documents: 문서 리스트 (문자열 또는 딕셔너리)
            top_k: 반환할 상위 K개

        Returns:
            재순위화된 RankedDocument 리스트
        """
        if not documents:
            return []

        # 초기화
        if not self._initialized:
            self._initialize()

        # 문서 형식 정규화
        doc_contents = []
        doc_metadata = []

        for doc in documents:
            if isinstance(doc, dict):
                doc_contents.append(doc.get("content", str(doc)))
                doc_metadata.append(doc.get("metadata", doc))
            else:
                doc_contents.append(str(doc))
                doc_metadata.append({})

        # 재순위화 수행
        if self.openai_client:
            scored_docs = self._rerank_with_openai(query, doc_contents, top_k)
        elif self.cross_encoder:
            scored_docs = self._rerank_with_cross_encoder(query, doc_contents, top_k)
        else:
            # 폴백: 원래 순서 유지
            logger.warning("No reranker available, returning original order")
            scored_docs = [(doc, 1.0 - i * 0.1) for i, doc in enumerate(doc_contents)][:top_k]

        # RankedDocument 객체 생성
        results = []
        for new_rank, (content, score) in enumerate(scored_docs):
            # 원래 순위 찾기
            try:
                original_rank = doc_contents.index(content)
                metadata = doc_metadata[original_rank]
            except ValueError:
                original_rank = -1
                metadata = {}

            results.append(
                RankedDocument(
                    content=content,
                    score=float(score),
                    original_rank=original_rank,
                    new_rank=new_rank,
                    metadata=metadata,
                )
            )

        return results

    def rerank_with_scores(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
        score_field: str = "score",
    ) -> list[dict[str, Any]]:
        """
        기존 검색 점수와 재순위화 점수를 결합

        Args:
            query: 검색 쿼리
            documents: 검색 결과 (score 필드 포함)
            top_k: 반환할 상위 K개
            score_field: 기존 점수 필드명

        Returns:
            재순위화된 결과 (combined_score 추가)
        """
        if not documents:
            return []

        # 재순위화
        ranked = self.rerank(query, documents, top_k=min(len(documents), top_k * 2))

        # 기존 점수와 결합
        results = []
        for ranked_doc in ranked[:top_k]:
            original_idx = ranked_doc.original_rank
            if 0 <= original_idx < len(documents):
                original_doc = documents[original_idx].copy()
                original_score = original_doc.get(score_field, 0)

                # 결합 점수 (가중 평균)
                combined_score = 0.6 * ranked_doc.score + 0.4 * original_score

                original_doc["rerank_score"] = ranked_doc.score
                original_doc["combined_score"] = combined_score
                original_doc["new_rank"] = ranked_doc.new_rank

                results.append(original_doc)

        # combined_score 기준 정렬
        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        return results


# 싱글톤 인스턴스
_reranker_instance: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    """CrossEncoderReranker 싱글톤 인스턴스 반환"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
