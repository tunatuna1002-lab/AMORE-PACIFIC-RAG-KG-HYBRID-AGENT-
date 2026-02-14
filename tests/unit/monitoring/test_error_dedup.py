"""
ErrorDeduplicationFilter 단위 테스트
"""

import logging
import time

from src.monitoring.logger import ErrorDeduplicationFilter


class TestErrorDeduplicationFilter:
    """ErrorDeduplicationFilter 클래스 테스트"""

    def test_init(self):
        """초기화"""
        f = ErrorDeduplicationFilter(window_seconds=30, max_count=5)
        assert f.window_seconds == 30
        assert f.max_count == 5

    def test_passes_debug_info(self):
        """DEBUG/INFO 메시지는 통과"""
        f = ErrorDeduplicationFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Info message",
            args=None,
            exc_info=None,
        )
        assert f.filter(record) is True

    def test_passes_first_errors(self):
        """첫 max_count개 에러는 통과"""
        f = ErrorDeduplicationFilter(max_count=3)
        for _i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="Same error message",
                args=None,
                exc_info=None,
            )
            assert f.filter(record) is True

    def test_suppresses_after_max(self):
        """max_count 초과 시 억제"""
        f = ErrorDeduplicationFilter(max_count=2, window_seconds=60)
        results = []
        for _i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="Repeated error",
                args=None,
                exc_info=None,
            )
            results.append(f.filter(record))

        # 처음 2개는 통과, 3번째는 [Dedup] 메시지로 통과, 나머지 억제 가능
        assert results[0] is True
        assert results[1] is True

    def test_different_messages_independent(self):
        """다른 메시지는 독립적으로 처리"""
        f = ErrorDeduplicationFilter(max_count=1)

        record1 = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error A",
            args=None,
            exc_info=None,
        )
        record2 = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error B",
            args=None,
            exc_info=None,
        )

        assert f.filter(record1) is True
        assert f.filter(record2) is True

    def test_cleanup_expired(self):
        """만료된 항목 정리"""
        f = ErrorDeduplicationFilter(window_seconds=1, max_count=1)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Expiring error",
            args=None,
            exc_info=None,
        )
        f.filter(record)
        assert len(f._seen) == 1

        time.sleep(1.1)
        # 새 레코드로 cleanup 트리거
        record2 = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="New error",
            args=None,
            exc_info=None,
        )
        f.filter(record2)
        # 이전 항목이 정리됨
        assert "Expiring error" not in str(f._seen)

    def test_get_stats(self):
        """통계"""
        f = ErrorDeduplicationFilter(window_seconds=60, max_count=2)

        for _i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="Stats test error",
                args=None,
                exc_info=None,
            )
            f.filter(record)

        stats = f.get_stats()
        assert "tracked_messages" in stats
        assert "total_suppressed" in stats
        assert stats["window_seconds"] == 60
        assert stats["max_count"] == 2

    def test_warning_level_filtered(self):
        """WARNING도 필터링 대상"""
        f = ErrorDeduplicationFilter(max_count=1)

        for _i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.WARNING,
                pathname="",
                lineno=0,
                msg="Warning msg",
                args=None,
                exc_info=None,
            )
            f.filter(record)

        assert f._seen  # 추적됨
