"""
Circuit Breaker Pattern
=======================
도구 실행의 연쇄 실패를 방지하는 회로 차단기

상태:
- CLOSED: 정상 (실행 허용)
- OPEN: 차단 (실행 거부, recovery_timeout 후 HALF_OPEN)
- HALF_OPEN: 시험 (1회 실행 허용, 성공 시 CLOSED, 실패 시 OPEN)
"""

import logging
import time
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """회로 차단기 상태"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit Breaker 패턴 구현

    Usage:
        breaker = CircuitBreaker(name="crawl_amazon", failure_threshold=3)
        if breaker.can_execute():
            try:
                result = await do_work()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
        else:
            # 차단 중 - 폴백 처리

    Args:
        name: 회로 이름 (로깅용)
        failure_threshold: OPEN 전환 실패 횟수 (기본 3)
        recovery_timeout: OPEN → HALF_OPEN 대기 시간 초 (기본 60)
        half_open_max_calls: HALF_OPEN 상태에서 허용할 최대 호출 수 (기본 1)
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """현재 상태 (OPEN → HALF_OPEN 자동 전환 포함)"""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(f"CircuitBreaker[{self.name}]: OPEN → HALF_OPEN (after {elapsed:.1f}s)")
        return self._state

    def can_execute(self) -> bool:
        """실행 가능 여부"""
        current = self.state  # 자동 전환 트리거

        if current == CircuitState.CLOSED:
            return True
        elif current == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        else:  # OPEN
            return False

    def record_success(self) -> None:
        """성공 기록"""
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            logger.info(f"CircuitBreaker[{self.name}]: HALF_OPEN → CLOSED (success)")
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

        self._success_count += 1

    def record_failure(self) -> None:
        """실패 기록"""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(f"CircuitBreaker[{self.name}]: HALF_OPEN → OPEN (failure)")
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"CircuitBreaker[{self.name}]: CLOSED → OPEN (failures={self._failure_count})"
                )

    def reset(self) -> None:
        """상태 초기화"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        logger.info(f"CircuitBreaker[{self.name}]: Reset to CLOSED")

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }
