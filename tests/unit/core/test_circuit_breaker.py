"""
CircuitBreaker 단위 테스트
"""

import time

from src.core.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitState:
    """CircuitState 열거형 테스트"""

    def test_states(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreaker:
    """CircuitBreaker 클래스 테스트"""

    def test_initial_state_is_closed(self):
        """초기 상태 = CLOSED"""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.can_execute() is True

    def test_stays_closed_on_success(self):
        """성공 시 CLOSED 유지"""
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_stays_closed_below_threshold(self):
        """임계값 이하 실패 시 CLOSED 유지"""
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.can_execute() is True

    def test_opens_at_threshold(self):
        """임계값 도달 시 OPEN 전환"""
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert breaker.can_execute() is False

    def test_open_to_half_open_after_timeout(self):
        """recovery_timeout 후 HALF_OPEN 전환"""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.1)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.can_execute() is True

    def test_half_open_success_closes(self):
        """HALF_OPEN에서 성공 시 CLOSED 전환"""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.05)
        breaker.record_failure()
        breaker.record_failure()
        time.sleep(0.06)

        assert breaker.state == CircuitState.HALF_OPEN
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        """HALF_OPEN에서 실패 시 OPEN 재전환"""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.05)
        breaker.record_failure()
        breaker.record_failure()
        time.sleep(0.06)

        assert breaker.state == CircuitState.HALF_OPEN
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_half_open_max_calls(self):
        """HALF_OPEN에서 max_calls 초과 시 실행 거부"""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.05,
            half_open_max_calls=1,
        )
        breaker.record_failure()
        breaker.record_failure()
        time.sleep(0.06)

        assert breaker.can_execute() is True
        # 첫 호출 후 half_open_calls 증가 시뮬레이션
        breaker._half_open_calls = 1
        assert breaker.can_execute() is False

    def test_reset(self):
        """reset() 후 CLOSED 상태"""
        breaker = CircuitBreaker(name="test", failure_threshold=1)
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
        assert breaker.can_execute() is True

    def test_success_resets_failure_count(self):
        """성공 시 실패 카운터 리셋"""
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker._failure_count == 2

        breaker.record_success()
        assert breaker._failure_count == 0

    def test_get_stats(self):
        """통계 반환"""
        breaker = CircuitBreaker(name="test_tool", failure_threshold=5)
        breaker.record_success()
        breaker.record_failure()

        stats = breaker.get_stats()
        assert stats["name"] == "test_tool"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 1
        assert stats["success_count"] == 1
        assert stats["failure_threshold"] == 5

    def test_default_parameters(self):
        """기본 파라미터 확인"""
        breaker = CircuitBreaker()
        assert breaker.name == "default"
        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 60.0
        assert breaker.half_open_max_calls == 1
