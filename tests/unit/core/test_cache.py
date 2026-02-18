"""
ResponseCache 단위 테스트
"""

from datetime import timedelta

from src.core.cache import ResponseCache

# =============================================================================
# 기본 생성 테스트
# =============================================================================


class TestResponseCacheInit:
    """ResponseCache 초기화 테스트"""

    def test_default_init(self):
        """기본값으로 초기화"""
        cache = ResponseCache()
        assert len(cache) == 0
        assert cache._max_size == 1000

    def test_custom_max_size(self):
        """커스텀 max_size"""
        cache = ResponseCache(max_size=50)
        assert cache._max_size == 50

    def test_custom_ttl_config(self):
        """커스텀 TTL 설정"""
        custom_ttl = {"query": timedelta(hours=1)}
        cache = ResponseCache(ttl_config=custom_ttl)
        assert cache._ttl["query"] == timedelta(hours=1)


# =============================================================================
# Get/Set 테스트
# =============================================================================


class TestResponseCacheGetSet:
    """ResponseCache get/set 테스트"""

    def test_set_and_get(self):
        """값 저장 후 조회"""
        cache = ResponseCache()
        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")
        assert result == {"data": "value1"}

    def test_get_nonexistent_key(self):
        """존재하지 않는 키 조회 시 None"""
        cache = ResponseCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_set_overwrites_existing(self):
        """같은 키에 덮어쓰기"""
        cache = ResponseCache()
        cache.set("key1", "old")
        cache.set("key1", "new")
        assert cache.get("key1") == "new"

    def test_set_with_cache_type(self):
        """다양한 cache_type으로 저장"""
        cache = ResponseCache()
        cache.set("k1", "v1", cache_type="query")
        cache.set("k2", "v2", cache_type="kg")
        cache.set("k3", "v3", cache_type="crawl")
        cache.set("k4", "v4", cache_type="context")
        assert cache.get("k1", cache_type="query") == "v1"
        assert cache.get("k2", cache_type="kg") == "v2"
        assert cache.get("k3", cache_type="crawl") == "v3"
        assert cache.get("k4", cache_type="context") == "v4"


# =============================================================================
# TTL 만료 테스트
# =============================================================================


class TestResponseCacheTTL:
    """ResponseCache TTL 만료 테스트"""

    def test_expired_entry_returns_none(self):
        """만료된 항목은 None 반환"""
        cache = ResponseCache(ttl_config={"query": timedelta(seconds=0)})
        cache.set("key1", "value1", cache_type="query")
        # TTL이 0초이므로 즉시 만료
        result = cache.get("key1", cache_type="query")
        assert result is None

    def test_non_expired_entry_returns_value(self):
        """만료되지 않은 항목은 값 반환"""
        cache = ResponseCache(ttl_config={"query": timedelta(hours=24)})
        cache.set("key1", "value1", cache_type="query")
        result = cache.get("key1", cache_type="query")
        assert result == "value1"

    def test_expired_entry_is_deleted(self):
        """만료된 항목 조회 시 삭제됨"""
        cache = ResponseCache(ttl_config={"query": timedelta(seconds=0)})
        cache.set("key1", "value1")
        cache.get("key1")  # 만료 → 삭제
        assert "key1" not in cache


# =============================================================================
# Delete 테스트
# =============================================================================


class TestResponseCacheDelete:
    """ResponseCache delete 테스트"""

    def test_delete_existing_key(self):
        """존재하는 키 삭제"""
        cache = ResponseCache()
        cache.set("key1", "value1")
        result = cache.delete("key1")
        assert result is True
        assert cache.get("key1") is None

    def test_delete_nonexistent_key(self):
        """존재하지 않는 키 삭제 시 False"""
        cache = ResponseCache()
        result = cache.delete("nonexistent")
        assert result is False


# =============================================================================
# Invalidate 테스트
# =============================================================================


class TestResponseCacheInvalidate:
    """ResponseCache invalidate 테스트"""

    def test_invalidate_all(self):
        """전체 캐시 무효화"""
        cache = ResponseCache()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        count = cache.invalidate()
        assert count == 3
        assert len(cache) == 0

    def test_invalidate_by_pattern(self):
        """패턴 기반 무효화"""
        cache = ResponseCache()
        cache.set("brand_laneige", "v1")
        cache.set("brand_cosrx", "v2")
        cache.set("metric_sos", "v3")
        count = cache.invalidate(pattern="brand")
        assert count == 2
        assert len(cache) == 1

    def test_invalidate_by_cache_type(self):
        """캐시 유형 기반 무효화"""
        cache = ResponseCache()
        cache.set("k1", "v1", cache_type="query")
        cache.set("k2", "v2", cache_type="kg")
        cache.set("k3", "v3", cache_type="query")
        count = cache.invalidate(cache_type="query")
        assert count == 2
        assert len(cache) == 1

    def test_invalidate_by_pattern_and_type(self):
        """패턴 + 유형 기반 무효화"""
        cache = ResponseCache()
        cache.set("brand_laneige", "v1", cache_type="query")
        cache.set("brand_cosrx", "v2", cache_type="kg")
        cache.set("metric_sos", "v3", cache_type="query")
        count = cache.invalidate(pattern="brand", cache_type="query")
        assert count == 1
        assert len(cache) == 2


class TestResponseCacheInvalidateByType:
    """ResponseCache invalidate_by_type 테스트"""

    def test_invalidate_by_type(self):
        """특정 유형 전체 무효화"""
        cache = ResponseCache()
        cache.set("k1", "v1", cache_type="kg")
        cache.set("k2", "v2", cache_type="kg")
        cache.set("k3", "v3", cache_type="query")
        count = cache.invalidate_by_type("kg")
        assert count == 2
        assert len(cache) == 1

    def test_invalidate_by_type_nonexistent(self):
        """존재하지 않는 유형 무효화 시 0"""
        cache = ResponseCache()
        cache.set("k1", "v1", cache_type="query")
        count = cache.invalidate_by_type("nonexistent")
        assert count == 0


# =============================================================================
# Eviction 테스트
# =============================================================================


class TestResponseCacheEviction:
    """ResponseCache 캐시 크기 제한 테스트"""

    def test_evicts_oldest_when_full(self):
        """최대 크기 초과 시 가장 오래된 항목 제거"""
        cache = ResponseCache(max_size=3)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        cache.set("k4", "v4")  # k1이 제거되어야 함
        assert len(cache) == 3
        assert cache.get("k1") is None
        assert cache.get("k4") == "v4"

    def test_eviction_stats(self):
        """eviction 통계 추적"""
        cache = ResponseCache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        stats = cache.get_stats()
        assert stats["evictions"] == 1


# =============================================================================
# Hash/Key 유틸리티 테스트
# =============================================================================


class TestResponseCacheHashQuery:
    """ResponseCache.hash_query 테스트"""

    def test_hash_query_deterministic(self):
        """동일 입력에 동일 해시"""
        h1 = ResponseCache.hash_query("LANEIGE 순위")
        h2 = ResponseCache.hash_query("LANEIGE 순위")
        assert h1 == h2

    def test_hash_query_case_insensitive(self):
        """대소문자 무시"""
        h1 = ResponseCache.hash_query("LANEIGE")
        h2 = ResponseCache.hash_query("laneige")
        assert h1 == h2

    def test_hash_query_whitespace_normalized(self):
        """공백 정규화"""
        h1 = ResponseCache.hash_query("LANEIGE  순위")
        h2 = ResponseCache.hash_query("LANEIGE 순위")
        assert h1 == h2

    def test_hash_query_strip(self):
        """앞뒤 공백 제거"""
        h1 = ResponseCache.hash_query("  test  ")
        h2 = ResponseCache.hash_query("test")
        assert h1 == h2


class TestResponseCacheMakeKey:
    """ResponseCache.make_key 테스트"""

    def test_make_key_deterministic(self):
        """동일 입력에 동일 키"""
        k1 = ResponseCache.make_key("brand", "LANEIGE")
        k2 = ResponseCache.make_key("brand", "LANEIGE")
        assert k1 == k2

    def test_make_key_different_parts(self):
        """다른 입력에 다른 키"""
        k1 = ResponseCache.make_key("brand", "LANEIGE")
        k2 = ResponseCache.make_key("brand", "COSRX")
        assert k1 != k2

    def test_make_key_single_part(self):
        """단일 파트"""
        key = ResponseCache.make_key("only_one")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex length


# =============================================================================
# Cleanup 테스트
# =============================================================================


class TestResponseCacheCleanup:
    """ResponseCache.cleanup_expired 테스트"""

    def test_cleanup_expired_removes_old(self):
        """만료된 항목 정리"""
        cache = ResponseCache(ttl_config={"query": timedelta(seconds=0)})
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        count = cache.cleanup_expired()
        assert count == 2
        assert len(cache) == 0

    def test_cleanup_expired_keeps_fresh(self):
        """신선한 항목 유지"""
        cache = ResponseCache(ttl_config={"query": timedelta(hours=24)})
        cache.set("k1", "v1")
        count = cache.cleanup_expired()
        assert count == 0
        assert len(cache) == 1


# =============================================================================
# Stats 테스트
# =============================================================================


class TestResponseCacheStats:
    """ResponseCache.get_stats 테스트"""

    def test_initial_stats(self):
        """초기 통계"""
        cache = ResponseCache()
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0
        assert stats["sets"] == 0
        assert stats["evictions"] == 0

    def test_stats_after_operations(self):
        """연산 후 통계"""
        cache = ResponseCache()
        cache.set("k1", "v1")
        cache.get("k1")  # hit
        cache.get("missing")  # miss
        stats = cache.get_stats()
        assert stats["sets"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_stats_max_size(self):
        """max_size 통계"""
        cache = ResponseCache(max_size=500)
        stats = cache.get_stats()
        assert stats["max_size"] == 500


# =============================================================================
# Dunder 메서드 테스트
# =============================================================================


class TestResponseCacheDunder:
    """ResponseCache __len__ / __contains__ 테스트"""

    def test_len(self):
        """__len__ 테스트"""
        cache = ResponseCache()
        assert len(cache) == 0
        cache.set("k1", "v1")
        assert len(cache) == 1
        cache.set("k2", "v2")
        assert len(cache) == 2

    def test_contains(self):
        """__contains__ 테스트"""
        cache = ResponseCache()
        cache.set("k1", "v1")
        assert "k1" in cache
        assert "k2" not in cache
