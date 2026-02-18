"""
응답 캐싱 모듈
==============
중복 LLM 호출 방지 및 비용 최적화를 위한 캐싱 레이어

캐시 유형:
- query: 동일 질문 캐시 (24시간 TTL)
- kg: KG 조회 결과 캐시 (1시간 TTL)
- crawl: 크롤링 결과 캐시 (당일)

Usage:
    cache = ResponseCache()

    # 캐시 확인
    cached = cache.get(cache.hash_query(query))
    if cached:
        return cached

    # 결과 캐싱
    cache.set(cache.hash_query(query), response)
"""

import hashlib
import logging
from datetime import datetime, timedelta
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    응답 캐시 관리자

    Thread-safe한 인메모리 캐시 구현.
    TTL(Time To Live) 기반 자동 만료 지원.
    """

    # =========================================================================
    # 캐시 유형별 TTL 설정
    # =========================================================================

    DEFAULT_TTL = {
        "query": timedelta(hours=24),  # 동일 질문: 24시간
        "kg": timedelta(hours=1),  # KG 조회: 1시간
        "crawl": timedelta(days=1),  # 크롤링: 당일
        "context": timedelta(minutes=30),  # 컨텍스트: 30분
    }

    def __init__(self, max_size: int = 1000, ttl_config: dict[str, timedelta] = None):
        """
        Args:
            max_size: 최대 캐시 항목 수
            ttl_config: 캐시 유형별 TTL 설정
        """
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self._max_size = max_size
        self._ttl = ttl_config or self.DEFAULT_TTL.copy()

        # 통계
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}

    # =========================================================================
    # 기본 CRUD 연산
    # =========================================================================

    def get(self, key: str, cache_type: str = "query") -> Any | None:
        """
        캐시 조회

        Args:
            key: 캐시 키
            cache_type: 캐시 유형 (TTL 결정용)

        Returns:
            캐시된 값 (없거나 만료 시 None)
        """
        with self._lock:
            cached = self._cache.get(key)

            if cached is None:
                self._stats["misses"] += 1
                return None

            # TTL 확인
            ttl = self._ttl.get(cache_type, self._ttl["query"])
            if datetime.now() - cached["timestamp"] > ttl:
                # 만료됨 - 삭제
                del self._cache[key]
                self._stats["misses"] += 1
                logger.debug(f"Cache expired: {key[:20]}...")
                return None

            self._stats["hits"] += 1
            logger.debug(f"Cache hit: {key[:20]}...")
            return cached["value"]

    def set(self, key: str, value: Any, cache_type: str = "query") -> None:
        """
        캐시 저장

        Args:
            key: 캐시 키
            value: 저장할 값
            cache_type: 캐시 유형
        """
        with self._lock:
            # 최대 크기 초과 시 오래된 항목 제거
            if len(self._cache) >= self._max_size:
                self._evict_oldest()

            self._cache[key] = {"value": value, "timestamp": datetime.now(), "type": cache_type}
            self._stats["sets"] += 1
            logger.debug(f"Cache set: {key[:20]}... (type={cache_type})")

    def delete(self, key: str) -> bool:
        """
        캐시 삭제

        Args:
            key: 캐시 키

        Returns:
            삭제 성공 여부
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def invalidate(self, pattern: str = None, cache_type: str = None) -> int:
        """
        캐시 무효화

        Args:
            pattern: 키 패턴 (None이면 전체)
            cache_type: 캐시 유형으로 필터링 (query, kg, crawl, context)

        Returns:
            삭제된 항목 수
        """
        with self._lock:
            # cache_type만 지정된 경우
            if cache_type is not None and pattern is None:
                return self._invalidate_by_type_internal(cache_type)

            if pattern is None and cache_type is None:
                count = len(self._cache)
                self._cache.clear()
                logger.info(f"Cache cleared: {count} items")
                return count

            # 패턴 + cache_type 필터링 삭제
            keys_to_delete = []
            for k, v in self._cache.items():
                if pattern is not None and pattern not in k:
                    continue
                if cache_type is not None and v.get("type") != cache_type:
                    continue
                keys_to_delete.append(k)

            for key in keys_to_delete:
                del self._cache[key]

            logger.info(
                f"Cache invalidated: {len(keys_to_delete)} items (pattern={pattern}, type={cache_type})"
            )
            return len(keys_to_delete)

    def _invalidate_by_type_internal(self, cache_type: str) -> int:
        """내부용: 특정 유형 캐시 무효화 (lock 없이)"""
        keys_to_delete = [k for k, v in self._cache.items() if v.get("type") == cache_type]
        for key in keys_to_delete:
            del self._cache[key]
        logger.info(f"Cache invalidated by type: {len(keys_to_delete)} items (type={cache_type})")
        return len(keys_to_delete)

    def invalidate_by_type(self, cache_type: str) -> int:
        """
        특정 유형의 캐시 전체 무효화

        Args:
            cache_type: 캐시 유형

        Returns:
            삭제된 항목 수
        """
        with self._lock:
            keys_to_delete = [k for k, v in self._cache.items() if v.get("type") == cache_type]
            for key in keys_to_delete:
                del self._cache[key]

            logger.info(f"Cache type invalidated: {len(keys_to_delete)} items (type={cache_type})")
            return len(keys_to_delete)

    # =========================================================================
    # 유틸리티
    # =========================================================================

    @staticmethod
    def hash_query(query: str) -> str:
        """
        질문을 캐시 키로 변환

        소문자 변환 및 공백 정규화 후 SHA-256 해시

        Args:
            query: 사용자 질문

        Returns:
            SHA-256 해시 문자열
        """
        normalized = query.lower().strip()
        # 연속 공백을 단일 공백으로
        normalized = " ".join(normalized.split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    @staticmethod
    def make_key(*parts: str) -> str:
        """
        여러 부분으로 캐시 키 생성

        Args:
            *parts: 키 구성 요소들

        Returns:
            조합된 캐시 키
        """
        combined = ":".join(str(p) for p in parts)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _evict_oldest(self) -> None:
        """가장 오래된 캐시 항목 제거 (LRU 방식)"""
        if not self._cache:
            return

        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
        del self._cache[oldest_key]
        self._stats["evictions"] += 1
        logger.debug(f"Cache evicted: {oldest_key[:20]}...")

    def cleanup_expired(self) -> int:
        """
        만료된 모든 캐시 항목 정리

        Returns:
            정리된 항목 수
        """
        with self._lock:
            now = datetime.now()
            expired_keys = []

            for key, cached in self._cache.items():
                cache_type = cached.get("type", "query")
                ttl = self._ttl.get(cache_type, self._ttl["query"])

                if now - cached["timestamp"] > ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.info(f"Cache cleanup: {len(expired_keys)} expired items removed")

            return len(expired_keys)

    # =========================================================================
    # 통계 및 상태
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        캐시 통계 조회

        Returns:
            통계 정보 dict
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": round(hit_rate, 4),
                "sets": self._stats["sets"],
                "evictions": self._stats["evictions"],
            }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache
