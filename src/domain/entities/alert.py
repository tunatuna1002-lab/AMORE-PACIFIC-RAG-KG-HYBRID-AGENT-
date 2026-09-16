"""
Alert Domain Entities
=====================
알림 유형의 단일 정의.

Phase 3 동명 클래스 정리: ``AlertType`` 이 ``src.core.state_manager`` 와
``src.tools.notifications.email_sender`` 두 곳에 따로 있었고 후자에만
``INSIGHT_REPORT`` 가 있었다. 구독 설정(state_manager)이 저장하는 값과 발송기
(email_sender)가 템플릿을 고르는 값이 같은 어휘여야 하므로 도메인으로 올린다.

``.value`` 문자열이 구독 설정 JSON(``alert_types``)에 그대로 저장되므로 값은 바뀌지
않는다.
"""

from __future__ import annotations

from enum import Enum


class AlertType(Enum):
    """알림 유형 (구독 설정과 이메일 템플릿이 공유하는 어휘)"""

    RANK_CHANGE = "rank_change"  # 순위 변동
    IMPORTANT_INSIGHT = "important_insight"  # 중요 인사이트
    CRAWL_COMPLETE = "crawl_complete"  # 크롤링 완료
    ERROR = "error"  # 에러
    DAILY_SUMMARY = "daily_summary"  # 일일 요약
    INSIGHT_REPORT = "insight_report"  # 인사이트 전체 리포트 (이메일 전용 템플릿)


#: 구독 기본값으로 쓰이는 유형 (전체 리포트는 명시 구독 시에만 발송)
DEFAULT_SUBSCRIPTION_ALERT_TYPES: tuple[AlertType, ...] = (
    AlertType.RANK_CHANGE,
    AlertType.IMPORTANT_INSIGHT,
    AlertType.ERROR,
)

__all__ = ["DEFAULT_SUBSCRIPTION_ALERT_TYPES", "AlertType"]
