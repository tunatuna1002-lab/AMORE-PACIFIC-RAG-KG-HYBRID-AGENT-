"""
Runtime RAG Metrics Collector
==============================
검색 품질 메트릭을 런타임에 수집합니다.

기존 tests/eval/ L2-L3 메트릭(MRR, precision@k)을 런타임에 적용.

수집 메트릭:
- retrieval_count: 총 검색 횟수
- avg_chunks_retrieved: 평균 청크 수
- precision_at_k: 관련 문서 비율 (relevance grading 결과 활용)
- mrr: Mean Reciprocal Rank
- avg_retrieval_time_ms: 평균 검색 시간
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalRecord:
    """단일 검색 기록"""

    query: str
    chunks_retrieved: int
    relevant_count: int
    total_count: int
    first_relevant_rank: int | None  # MRR 계산용 (1-indexed, None=없음)
    retrieval_time_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def precision(self) -> float:
        """Precision@k"""
        if self.total_count == 0:
            return 0.0
        return self.relevant_count / self.total_count

    @property
    def reciprocal_rank(self) -> float:
        """Reciprocal Rank"""
        if self.first_relevant_rank is None or self.first_relevant_rank == 0:
            return 0.0
        return 1.0 / self.first_relevant_rank


class RAGMetricsCollector:
    """
    런타임 RAG 검색 메트릭 수집기

    Usage:
        collector = RAGMetricsCollector()
        collector.record_retrieval(
            query="LANEIGE 분석",
            chunks=rag_chunks,
            relevance_grades={"relevant": [...], "irrelevant": [...]}
        )
        metrics = collector.get_metrics()
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: 메트릭 계산에 사용할 최근 기록 수
        """
        self._records: deque[RetrievalRecord] = deque(maxlen=window_size)
        self._total_retrievals: int = 0
        self._window_size = window_size

    def record_retrieval(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        relevant_chunks: list[dict[str, Any]] | None = None,
        irrelevant_chunks: list[dict[str, Any]] | None = None,
        retrieval_time_ms: float = 0.0,
    ) -> None:
        """
        검색 결과 기록

        Args:
            query: 검색 쿼리
            chunks: 전체 검색 결과
            relevant_chunks: 관련 문서 (RelevanceGrader 결과)
            irrelevant_chunks: 비관련 문서
            retrieval_time_ms: 검색 소요 시간 (ms)
        """
        self._total_retrievals += 1

        total_count = len(chunks)
        relevant_count = len(relevant_chunks) if relevant_chunks is not None else total_count

        # MRR: 첫 번째 관련 문서의 순위 찾기
        first_relevant_rank = None
        if relevant_chunks is not None:
            relevant_ids = {c.get("id") for c in relevant_chunks}
            for i, chunk in enumerate(chunks, 1):
                if chunk.get("id") in relevant_ids:
                    first_relevant_rank = i
                    break

        record = RetrievalRecord(
            query=query,
            chunks_retrieved=total_count,
            relevant_count=relevant_count,
            total_count=total_count,
            first_relevant_rank=first_relevant_rank,
            retrieval_time_ms=retrieval_time_ms,
        )

        self._records.append(record)

        logger.debug(
            f"RAG metric recorded: chunks={total_count}, "
            f"relevant={relevant_count}, precision={record.precision:.2f}"
        )

    def get_metrics(self) -> dict[str, Any]:
        """
        집계 메트릭 반환

        Returns:
            {
                "total_retrievals": int,
                "window_size": int,
                "records_in_window": int,
                "avg_chunks_retrieved": float,
                "avg_precision_at_k": float,
                "mrr": float,
                "avg_retrieval_time_ms": float,
                "recent_queries": list[str],
            }
        """
        if not self._records:
            return {
                "total_retrievals": self._total_retrievals,
                "window_size": self._window_size,
                "records_in_window": 0,
                "avg_chunks_retrieved": 0.0,
                "avg_precision_at_k": 0.0,
                "mrr": 0.0,
                "avg_retrieval_time_ms": 0.0,
                "recent_queries": [],
            }

        records = list(self._records)
        n = len(records)

        avg_chunks = sum(r.chunks_retrieved for r in records) / n
        avg_precision = sum(r.precision for r in records) / n
        mrr = sum(r.reciprocal_rank for r in records) / n
        avg_time = sum(r.retrieval_time_ms for r in records) / n

        return {
            "total_retrievals": self._total_retrievals,
            "window_size": self._window_size,
            "records_in_window": n,
            "avg_chunks_retrieved": round(avg_chunks, 2),
            "avg_precision_at_k": round(avg_precision, 4),
            "mrr": round(mrr, 4),
            "avg_retrieval_time_ms": round(avg_time, 2),
            "recent_queries": [r.query[:50] for r in records[-5:]],
        }

    def reset(self) -> None:
        """메트릭 초기화"""
        self._records.clear()
        self._total_retrievals = 0
