"""
규칙 엔진 (Rules Engine)
========================
Level 4 Autonomous Agent의 규칙 기반 의사결정 엔진

역할:
- config/rules.json에서 규칙 로드
- 조건 평가 및 액션 트리거
- 예측 가능한 상황에서 빠른 의사결정 (LLM 호출 불필요)

규칙 유형:
1. schedule_rules: 시간 기반 스케줄 규칙
2. alert_rules: 임계값/이벤트 기반 알림 규칙
3. priority_rules: 우선순위 결정 규칙

Usage:
    engine = RulesEngine()

    # 이벤트 발생 시 매칭되는 규칙 찾기
    matched = engine.evaluate_event("crawl_complete", context)

    # 스케줄 규칙 확인
    scheduled = engine.get_scheduled_actions(current_time)
"""

import json
import logging
from datetime import datetime, time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ConditionType(Enum):
    """규칙 조건 유형"""
    SCHEDULE = "schedule"      # 시간 기반
    EVENT = "event"            # 이벤트 기반
    THRESHOLD = "threshold"    # 임계값 기반


class ActionType(Enum):
    """규칙 액션 유형"""
    CRAWL_WORKFLOW = "crawl_workflow"
    UPDATE_DASHBOARD = "update_dashboard"
    SEND_ALERT = "send_alert"
    RESPOND_IMMEDIATELY = "respond_immediately"
    ANALYZE_DATA = "analyze_data"
    GENERATE_INSIGHT = "generate_insight"
    UPDATE_KNOWLEDGE_GRAPH = "update_knowledge_graph"


@dataclass
class Rule:
    """규칙 정의"""
    id: str
    name: str
    condition: Dict[str, Any]
    action: str
    priority: int
    enabled: bool
    alert_type: Optional[str] = None  # 알림 규칙용

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        """딕셔너리에서 Rule 생성"""
        return cls(
            id=data["id"],
            name=data["name"],
            condition=data["condition"],
            action=data["action"],
            priority=data.get("priority", 5),
            enabled=data.get("enabled", True),
            alert_type=data.get("alert_type")
        )


@dataclass
class RuleMatch:
    """규칙 매칭 결과"""
    rule: Rule
    action: str
    priority: int
    context: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """우선순위 기반 정렬 (낮은 숫자가 높은 우선순위)"""
        return self.priority < other.priority


class RulesEngine:
    """
    규칙 기반 의사결정 엔진

    config/rules.json에서 규칙을 로드하고,
    조건을 평가하여 적절한 액션을 반환합니다.
    """

    def __init__(self, rules_path: Optional[Path] = None):
        """
        Args:
            rules_path: 규칙 파일 경로 (기본: config/rules.json)
        """
        if rules_path is None:
            rules_path = Path("./config/rules.json")

        self.rules_path = rules_path
        self.schedule_rules: List[Rule] = []
        self.alert_rules: List[Rule] = []
        self.priority_rules: List[Rule] = []
        self.thresholds: Dict[str, Any] = {}
        self.permissions: Dict[str, List[str]] = {}

        self._load_rules()

    # =========================================================================
    # 규칙 로드
    # =========================================================================

    def _load_rules(self) -> None:
        """규칙 파일에서 규칙 로드"""
        try:
            if not self.rules_path.exists():
                logger.warning(f"Rules file not found: {self.rules_path}")
                return

            with open(self.rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 스케줄 규칙
            for rule_data in data.get("schedule_rules", []):
                rule = Rule.from_dict(rule_data)
                if rule.enabled:
                    self.schedule_rules.append(rule)

            # 알림 규칙
            for rule_data in data.get("alert_rules", []):
                rule = Rule.from_dict(rule_data)
                if rule.enabled:
                    self.alert_rules.append(rule)

            # 우선순위 규칙
            for rule_data in data.get("priority_rules", []):
                rule = Rule.from_dict(rule_data)
                if rule.enabled:
                    self.priority_rules.append(rule)

            # 임계값
            self.thresholds = data.get("thresholds", {})

            # 권한
            self.permissions = data.get("permissions", {})

            total = len(self.schedule_rules) + len(self.alert_rules) + len(self.priority_rules)
            logger.info(f"Loaded {total} rules from {self.rules_path}")

        except Exception as e:
            logger.error(f"Failed to load rules: {e}")

    def reload_rules(self) -> None:
        """규칙 다시 로드"""
        self.schedule_rules = []
        self.alert_rules = []
        self.priority_rules = []
        self._load_rules()

    # =========================================================================
    # 이벤트 기반 규칙 평가
    # =========================================================================

    def evaluate_event(self, event_name: str, context: Optional[Dict[str, Any]] = None) -> List[RuleMatch]:
        """
        이벤트에 매칭되는 규칙 찾기

        Args:
            event_name: 이벤트 이름 (예: "crawl_complete", "user_request")
            context: 이벤트 컨텍스트 데이터

        Returns:
            매칭된 규칙 목록 (우선순위 순)
        """
        context = context or {}
        matches: List[RuleMatch] = []

        # 모든 규칙 검사
        all_rules = self.schedule_rules + self.alert_rules + self.priority_rules

        for rule in all_rules:
            if rule.condition.get("type") == "event":
                if rule.condition.get("event") == event_name:
                    matches.append(RuleMatch(
                        rule=rule,
                        action=rule.action,
                        priority=rule.priority,
                        context=context
                    ))

        # 우선순위 순 정렬
        matches.sort()
        return matches

    # =========================================================================
    # 임계값 기반 규칙 평가
    # =========================================================================

    def evaluate_threshold(self, metric: str, value: float, context: Optional[Dict[str, Any]] = None) -> List[RuleMatch]:
        """
        임계값 조건을 만족하는 규칙 찾기

        Args:
            metric: 지표 이름 (예: "rank_change")
            value: 지표 값
            context: 추가 컨텍스트

        Returns:
            매칭된 규칙 목록
        """
        context = context or {}
        context["metric"] = metric
        context["value"] = value
        matches: List[RuleMatch] = []

        for rule in self.alert_rules:
            if rule.condition.get("type") != "threshold":
                continue

            if rule.condition.get("metric") != metric:
                continue

            operator = rule.condition.get("operator", ">=")
            threshold = rule.condition.get("value", 0)

            # 조건 평가
            if self._evaluate_operator(value, operator, threshold):
                matches.append(RuleMatch(
                    rule=rule,
                    action=rule.action,
                    priority=rule.priority,
                    context=context
                ))

        matches.sort()
        return matches

    def _evaluate_operator(self, value: float, operator: str, threshold: float) -> bool:
        """연산자 평가"""
        if operator == ">=":
            return value >= threshold
        elif operator == ">":
            return value > threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False

    # =========================================================================
    # 스케줄 기반 규칙 평가
    # =========================================================================

    def get_scheduled_actions(self, current_time: Optional[datetime] = None,
                              state_context: Optional[Dict[str, Any]] = None) -> List[RuleMatch]:
        """
        현재 시간에 실행해야 할 스케줄 규칙 반환

        Args:
            current_time: 현재 시간 (기본: 현재)
            state_context: 상태 컨텍스트 (조건 확인용)

        Returns:
            실행해야 할 규칙 목록
        """
        if current_time is None:
            current_time = datetime.now()

        state_context = state_context or {}
        matches: List[RuleMatch] = []

        for rule in self.schedule_rules:
            if rule.condition.get("type") != "schedule":
                continue

            # 시간 조건 확인
            schedule_time_str = rule.condition.get("time", "00:00")
            try:
                hour, minute = map(int, schedule_time_str.split(":"))
                schedule_time = time(hour, minute)
            except ValueError:
                logger.warning(f"Invalid time format in rule {rule.id}: {schedule_time_str}")
                continue

            # 현재 시간이 스케줄 시간과 일치하는지 (분 단위)
            if current_time.hour == schedule_time.hour and current_time.minute == schedule_time.minute:
                # 추가 조건 확인 (예: no_today_data)
                check = rule.condition.get("check")
                if check and not self._evaluate_check(check, state_context):
                    continue

                matches.append(RuleMatch(
                    rule=rule,
                    action=rule.action,
                    priority=rule.priority,
                    context=state_context
                ))

        matches.sort()
        return matches

    def _evaluate_check(self, check: str, state_context: Dict[str, Any]) -> bool:
        """추가 조건 확인"""
        if check == "no_today_data":
            # 오늘 데이터가 없는 경우만 True
            return state_context.get("crawl_needed", True)

        logger.warning(f"Unknown check: {check}")
        return True

    # =========================================================================
    # 권한 확인
    # =========================================================================

    def is_action_allowed(self, action: str) -> bool:
        """
        액션이 허용되는지 확인

        Args:
            action: 액션 이름

        Returns:
            True면 허용
        """
        if action in self.permissions.get("forbidden", []):
            return False

        if action in self.permissions.get("allowed", []):
            return True

        if action in self.permissions.get("requires_consent", []):
            # 동의가 필요한 액션 - 별도 확인 필요
            return None  # None은 동의 필요를 의미

        # 기본적으로 허용하지 않음
        return False

    def requires_consent(self, action: str) -> bool:
        """액션이 사용자 동의를 필요로 하는지 확인"""
        return action in self.permissions.get("requires_consent", [])

    def get_forbidden_actions(self) -> List[str]:
        """금지된 액션 목록 반환"""
        return self.permissions.get("forbidden", [])

    # =========================================================================
    # 임계값 조회
    # =========================================================================

    def get_threshold(self, name: str, default: Any = None) -> Any:
        """
        임계값 조회

        Args:
            name: 임계값 이름
            default: 기본값

        Returns:
            임계값
        """
        return self.thresholds.get(name, default)

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def get_all_rules(self) -> Dict[str, List[Rule]]:
        """모든 규칙 반환"""
        return {
            "schedule": self.schedule_rules,
            "alert": self.alert_rules,
            "priority": self.priority_rules
        }

    def get_rule_by_id(self, rule_id: str) -> Optional[Rule]:
        """ID로 규칙 찾기"""
        all_rules = self.schedule_rules + self.alert_rules + self.priority_rules
        for rule in all_rules:
            if rule.id == rule_id:
                return rule
        return None

    def to_summary(self) -> str:
        """규칙 요약 생성"""
        return (
            f"Rules Engine: "
            f"{len(self.schedule_rules)} schedule, "
            f"{len(self.alert_rules)} alert, "
            f"{len(self.priority_rules)} priority rules"
        )
