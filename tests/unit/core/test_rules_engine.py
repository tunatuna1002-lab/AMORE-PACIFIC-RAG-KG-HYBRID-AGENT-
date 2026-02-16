"""
RulesEngine 단위 테스트
"""

import json
from datetime import datetime

from src.core.rules_engine import (
    ActionType,
    ConditionType,
    Rule,
    RuleMatch,
    RulesEngine,
)

# =============================================================================
# Enum 테스트
# =============================================================================


class TestConditionType:
    """ConditionType 열거형 테스트"""

    def test_values(self):
        assert ConditionType.SCHEDULE.value == "schedule"
        assert ConditionType.EVENT.value == "event"
        assert ConditionType.THRESHOLD.value == "threshold"


class TestActionType:
    """ActionType 열거형 테스트"""

    def test_values(self):
        assert ActionType.CRAWL_WORKFLOW.value == "crawl_workflow"
        assert ActionType.UPDATE_DASHBOARD.value == "update_dashboard"
        assert ActionType.SEND_ALERT.value == "send_alert"
        assert ActionType.RESPOND_IMMEDIATELY.value == "respond_immediately"
        assert ActionType.ANALYZE_DATA.value == "analyze_data"
        assert ActionType.GENERATE_INSIGHT.value == "generate_insight"
        assert ActionType.UPDATE_KNOWLEDGE_GRAPH.value == "update_knowledge_graph"


# =============================================================================
# Rule 테스트
# =============================================================================


class TestRule:
    """Rule 데이터클래스 테스트"""

    def test_from_dict_full(self):
        """전체 필드로 Rule 생성"""
        data = {
            "id": "rule1",
            "name": "Test Rule",
            "condition": {"type": "event", "event": "crawl_complete"},
            "action": "update_dashboard",
            "priority": 1,
            "enabled": True,
            "alert_type": "rank_change",
        }
        rule = Rule.from_dict(data)
        assert rule.id == "rule1"
        assert rule.name == "Test Rule"
        assert rule.condition["type"] == "event"
        assert rule.action == "update_dashboard"
        assert rule.priority == 1
        assert rule.enabled is True
        assert rule.alert_type == "rank_change"

    def test_from_dict_defaults(self):
        """기본값으로 Rule 생성"""
        data = {
            "id": "rule2",
            "name": "Default Rule",
            "condition": {"type": "event"},
            "action": "analyze_data",
        }
        rule = Rule.from_dict(data)
        assert rule.priority == 5  # default
        assert rule.enabled is True  # default
        assert rule.alert_type is None  # default


# =============================================================================
# RuleMatch 테스트
# =============================================================================


class TestRuleMatch:
    """RuleMatch 정렬 테스트"""

    def test_sort_by_priority(self):
        """우선순위로 정렬"""
        rule_low = Rule(id="r1", name="Low", condition={}, action="a", priority=10, enabled=True)
        rule_high = Rule(id="r2", name="High", condition={}, action="b", priority=1, enabled=True)
        match_low = RuleMatch(rule=rule_low, action="a", priority=10)
        match_high = RuleMatch(rule=rule_high, action="b", priority=1)

        sorted_matches = sorted([match_low, match_high])
        assert sorted_matches[0].priority == 1
        assert sorted_matches[1].priority == 10


# =============================================================================
# RulesEngine 초기화 테스트
# =============================================================================


class TestRulesEngineInit:
    """RulesEngine 초기화 테스트"""

    def test_init_with_nonexistent_file(self, tmp_path):
        """존재하지 않는 파일로 초기화"""
        engine = RulesEngine(rules_path=tmp_path / "nonexistent.json")
        assert engine.schedule_rules == []
        assert engine.alert_rules == []
        assert engine.priority_rules == []

    def test_init_with_valid_file(self, tmp_path):
        """유효한 규칙 파일로 초기화"""
        rules_data = {
            "schedule_rules": [
                {
                    "id": "s1",
                    "name": "Daily Crawl",
                    "condition": {"type": "schedule", "time": "22:00"},
                    "action": "crawl_workflow",
                    "priority": 1,
                    "enabled": True,
                }
            ],
            "alert_rules": [
                {
                    "id": "a1",
                    "name": "Rank Alert",
                    "condition": {
                        "type": "threshold",
                        "metric": "rank_change",
                        "operator": ">=",
                        "value": 5,
                    },
                    "action": "send_alert",
                    "priority": 2,
                    "enabled": True,
                }
            ],
            "priority_rules": [
                {
                    "id": "p1",
                    "name": "Priority Rule",
                    "condition": {"type": "event", "event": "user_request"},
                    "action": "respond_immediately",
                    "priority": 1,
                    "enabled": True,
                }
            ],
            "thresholds": {"rank_change": 5, "sos_drop": 10},
            "permissions": {
                "allowed": ["crawl", "analyze"],
                "forbidden": ["delete_data"],
                "requires_consent": ["send_email"],
            },
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(rules_data), encoding="utf-8")

        engine = RulesEngine(rules_path=rules_file)
        assert len(engine.schedule_rules) == 1
        assert len(engine.alert_rules) == 1
        assert len(engine.priority_rules) == 1
        assert engine.thresholds["rank_change"] == 5
        assert "delete_data" in engine.permissions["forbidden"]

    def test_init_disabled_rules_excluded(self, tmp_path):
        """비활성 규칙은 제외"""
        rules_data = {
            "schedule_rules": [
                {
                    "id": "s1",
                    "name": "Disabled",
                    "condition": {"type": "schedule"},
                    "action": "crawl",
                    "enabled": False,
                }
            ],
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(rules_data), encoding="utf-8")

        engine = RulesEngine(rules_path=rules_file)
        assert len(engine.schedule_rules) == 0

    def test_init_invalid_json(self, tmp_path):
        """잘못된 JSON 파일"""
        rules_file = tmp_path / "rules.json"
        rules_file.write_text("invalid json {{{", encoding="utf-8")
        engine = RulesEngine(rules_path=rules_file)
        assert engine.schedule_rules == []


# =============================================================================
# reload_rules 테스트
# =============================================================================


class TestRulesEngineReload:
    """RulesEngine.reload_rules 테스트"""

    def test_reload_clears_and_reloads(self, tmp_path):
        """reload 시 기존 규칙 지우고 다시 로드"""
        rules_data = {
            "schedule_rules": [
                {
                    "id": "s1",
                    "name": "Rule",
                    "condition": {"type": "schedule"},
                    "action": "crawl",
                    "enabled": True,
                }
            ],
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(rules_data), encoding="utf-8")

        engine = RulesEngine(rules_path=rules_file)
        assert len(engine.schedule_rules) == 1

        # 파일 업데이트
        rules_data["schedule_rules"].append(
            {
                "id": "s2",
                "name": "Rule2",
                "condition": {"type": "schedule"},
                "action": "analyze",
                "enabled": True,
            }
        )
        rules_file.write_text(json.dumps(rules_data), encoding="utf-8")

        engine.reload_rules()
        assert len(engine.schedule_rules) == 2


# =============================================================================
# evaluate_event 테스트
# =============================================================================


class TestRulesEngineEvaluateEvent:
    """RulesEngine.evaluate_event 테스트"""

    def _make_engine(self, tmp_path, rules_data):
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(rules_data), encoding="utf-8")
        return RulesEngine(rules_path=rules_file)

    def test_matching_event(self, tmp_path):
        """이벤트 매칭 성공"""
        rules_data = {
            "alert_rules": [
                {
                    "id": "a1",
                    "name": "Crawl Complete",
                    "condition": {"type": "event", "event": "crawl_complete"},
                    "action": "update_dashboard",
                    "priority": 1,
                    "enabled": True,
                }
            ],
        }
        engine = self._make_engine(tmp_path, rules_data)
        matches = engine.evaluate_event("crawl_complete")
        assert len(matches) == 1
        assert matches[0].action == "update_dashboard"

    def test_no_matching_event(self, tmp_path):
        """이벤트 매칭 실패"""
        rules_data = {
            "alert_rules": [
                {
                    "id": "a1",
                    "name": "Crawl Complete",
                    "condition": {"type": "event", "event": "crawl_complete"},
                    "action": "update_dashboard",
                    "priority": 1,
                    "enabled": True,
                }
            ],
        }
        engine = self._make_engine(tmp_path, rules_data)
        matches = engine.evaluate_event("unknown_event")
        assert len(matches) == 0

    def test_multiple_matches_sorted_by_priority(self, tmp_path):
        """여러 매칭 시 우선순위 순 정렬"""
        rules_data = {
            "alert_rules": [
                {
                    "id": "a1",
                    "name": "Low Priority",
                    "condition": {"type": "event", "event": "crawl_complete"},
                    "action": "analyze",
                    "priority": 10,
                    "enabled": True,
                },
                {
                    "id": "a2",
                    "name": "High Priority",
                    "condition": {"type": "event", "event": "crawl_complete"},
                    "action": "alert",
                    "priority": 1,
                    "enabled": True,
                },
            ],
        }
        engine = self._make_engine(tmp_path, rules_data)
        matches = engine.evaluate_event("crawl_complete")
        assert len(matches) == 2
        assert matches[0].priority == 1
        assert matches[1].priority == 10

    def test_event_with_context(self, tmp_path):
        """이벤트에 컨텍스트 전달"""
        rules_data = {
            "priority_rules": [
                {
                    "id": "p1",
                    "name": "User Request",
                    "condition": {"type": "event", "event": "user_request"},
                    "action": "respond",
                    "priority": 1,
                    "enabled": True,
                }
            ],
        }
        engine = self._make_engine(tmp_path, rules_data)
        ctx = {"user_id": "123"}
        matches = engine.evaluate_event("user_request", context=ctx)
        assert len(matches) == 1
        assert matches[0].context == ctx


# =============================================================================
# evaluate_threshold 테스트
# =============================================================================


class TestRulesEngineEvaluateThreshold:
    """RulesEngine.evaluate_threshold 테스트"""

    def _make_engine(self, tmp_path, alert_rules):
        rules_data = {"alert_rules": alert_rules}
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(rules_data), encoding="utf-8")
        return RulesEngine(rules_path=rules_file)

    def test_threshold_gte_match(self, tmp_path):
        """'>=' 연산자 매칭"""
        rules = [
            {
                "id": "a1",
                "name": "Rank Drop",
                "condition": {
                    "type": "threshold",
                    "metric": "rank_change",
                    "operator": ">=",
                    "value": 5,
                },
                "action": "send_alert",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        matches = engine.evaluate_threshold("rank_change", 10)
        assert len(matches) == 1

    def test_threshold_gte_no_match(self, tmp_path):
        """'>=' 연산자 미매칭"""
        rules = [
            {
                "id": "a1",
                "name": "Rank Drop",
                "condition": {
                    "type": "threshold",
                    "metric": "rank_change",
                    "operator": ">=",
                    "value": 5,
                },
                "action": "send_alert",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        matches = engine.evaluate_threshold("rank_change", 3)
        assert len(matches) == 0

    def test_threshold_gt(self, tmp_path):
        """'>' 연산자"""
        rules = [
            {
                "id": "a1",
                "name": "Test",
                "condition": {"type": "threshold", "metric": "sos", "operator": ">", "value": 50},
                "action": "alert",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        assert len(engine.evaluate_threshold("sos", 51)) == 1
        assert len(engine.evaluate_threshold("sos", 50)) == 0

    def test_threshold_lte(self, tmp_path):
        """'<=' 연산자"""
        rules = [
            {
                "id": "a1",
                "name": "Test",
                "condition": {"type": "threshold", "metric": "sos", "operator": "<=", "value": 10},
                "action": "alert",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        assert len(engine.evaluate_threshold("sos", 10)) == 1
        assert len(engine.evaluate_threshold("sos", 11)) == 0

    def test_threshold_lt(self, tmp_path):
        """'<' 연산자"""
        rules = [
            {
                "id": "a1",
                "name": "Test",
                "condition": {"type": "threshold", "metric": "sos", "operator": "<", "value": 5},
                "action": "alert",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        assert len(engine.evaluate_threshold("sos", 4)) == 1
        assert len(engine.evaluate_threshold("sos", 5)) == 0

    def test_threshold_eq(self, tmp_path):
        """'==' 연산자"""
        rules = [
            {
                "id": "a1",
                "name": "Test",
                "condition": {"type": "threshold", "metric": "rank", "operator": "==", "value": 1},
                "action": "alert",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        assert len(engine.evaluate_threshold("rank", 1)) == 1
        assert len(engine.evaluate_threshold("rank", 2)) == 0

    def test_threshold_neq(self, tmp_path):
        """'!=' 연산자"""
        rules = [
            {
                "id": "a1",
                "name": "Test",
                "condition": {"type": "threshold", "metric": "rank", "operator": "!=", "value": 1},
                "action": "alert",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        assert len(engine.evaluate_threshold("rank", 2)) == 1
        assert len(engine.evaluate_threshold("rank", 1)) == 0

    def test_threshold_unknown_operator(self, tmp_path):
        """알 수 없는 연산자"""
        rules = [
            {
                "id": "a1",
                "name": "Test",
                "condition": {"type": "threshold", "metric": "rank", "operator": "???", "value": 1},
                "action": "alert",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        assert len(engine.evaluate_threshold("rank", 1)) == 0

    def test_threshold_wrong_metric(self, tmp_path):
        """다른 지표는 매칭 안됨"""
        rules = [
            {
                "id": "a1",
                "name": "Test",
                "condition": {
                    "type": "threshold",
                    "metric": "rank_change",
                    "operator": ">=",
                    "value": 5,
                },
                "action": "alert",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        matches = engine.evaluate_threshold("sos_change", 10)
        assert len(matches) == 0

    def test_threshold_skips_event_rules(self, tmp_path):
        """event 유형 규칙은 threshold 평가에서 스킵"""
        rules = [
            {
                "id": "a1",
                "name": "Event Rule",
                "condition": {"type": "event", "event": "crawl_complete"},
                "action": "alert",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        matches = engine.evaluate_threshold("rank_change", 10)
        assert len(matches) == 0


# =============================================================================
# get_scheduled_actions 테스트
# =============================================================================


class TestRulesEngineScheduledActions:
    """RulesEngine.get_scheduled_actions 테스트"""

    def _make_engine(self, tmp_path, schedule_rules):
        rules_data = {"schedule_rules": schedule_rules}
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(rules_data), encoding="utf-8")
        return RulesEngine(rules_path=rules_file)

    def test_matching_time(self, tmp_path):
        """시간 매칭"""
        rules = [
            {
                "id": "s1",
                "name": "Daily Crawl",
                "condition": {"type": "schedule", "time": "22:00"},
                "action": "crawl_workflow",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        current = datetime(2026, 2, 17, 22, 0)
        matches = engine.get_scheduled_actions(current_time=current)
        assert len(matches) == 1

    def test_non_matching_time(self, tmp_path):
        """시간 미매칭"""
        rules = [
            {
                "id": "s1",
                "name": "Daily Crawl",
                "condition": {"type": "schedule", "time": "22:00"},
                "action": "crawl_workflow",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        current = datetime(2026, 2, 17, 10, 0)
        matches = engine.get_scheduled_actions(current_time=current)
        assert len(matches) == 0

    def test_schedule_with_check_condition_pass(self, tmp_path):
        """추가 조건(check) 통과"""
        rules = [
            {
                "id": "s1",
                "name": "Crawl if needed",
                "condition": {"type": "schedule", "time": "22:00", "check": "no_today_data"},
                "action": "crawl_workflow",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        current = datetime(2026, 2, 17, 22, 0)
        matches = engine.get_scheduled_actions(
            current_time=current, state_context={"crawl_needed": True}
        )
        assert len(matches) == 1

    def test_schedule_with_check_condition_fail(self, tmp_path):
        """추가 조건(check) 실패"""
        rules = [
            {
                "id": "s1",
                "name": "Crawl if needed",
                "condition": {"type": "schedule", "time": "22:00", "check": "no_today_data"},
                "action": "crawl_workflow",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        current = datetime(2026, 2, 17, 22, 0)
        matches = engine.get_scheduled_actions(
            current_time=current, state_context={"crawl_needed": False}
        )
        assert len(matches) == 0

    def test_invalid_time_format(self, tmp_path):
        """잘못된 시간 형식"""
        rules = [
            {
                "id": "s1",
                "name": "Bad Time",
                "condition": {"type": "schedule", "time": "invalid"},
                "action": "crawl",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        current = datetime(2026, 2, 17, 22, 0)
        matches = engine.get_scheduled_actions(current_time=current)
        assert len(matches) == 0

    def test_skips_non_schedule_rules(self, tmp_path):
        """schedule 유형이 아닌 규칙 스킵"""
        rules = [
            {
                "id": "s1",
                "name": "Event rule in schedule list",
                "condition": {"type": "event", "event": "test"},
                "action": "crawl",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        matches = engine.get_scheduled_actions(current_time=datetime(2026, 2, 17, 22, 0))
        assert len(matches) == 0

    def test_unknown_check_passes(self, tmp_path):
        """알 수 없는 check는 True 반환"""
        rules = [
            {
                "id": "s1",
                "name": "Unknown check",
                "condition": {"type": "schedule", "time": "22:00", "check": "unknown_check"},
                "action": "crawl",
                "priority": 1,
                "enabled": True,
            }
        ]
        engine = self._make_engine(tmp_path, rules)
        current = datetime(2026, 2, 17, 22, 0)
        matches = engine.get_scheduled_actions(current_time=current)
        assert len(matches) == 1


# =============================================================================
# 권한 테스트
# =============================================================================


class TestRulesEnginePermissions:
    """RulesEngine 권한 관련 테스트"""

    def _make_engine(self, tmp_path):
        rules_data = {
            "permissions": {
                "allowed": ["crawl", "analyze"],
                "forbidden": ["delete_data"],
                "requires_consent": ["send_email"],
            }
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(rules_data), encoding="utf-8")
        return RulesEngine(rules_path=rules_file)

    def test_allowed_action(self, tmp_path):
        """허용된 액션"""
        engine = self._make_engine(tmp_path)
        assert engine.is_action_allowed("crawl") is True

    def test_forbidden_action(self, tmp_path):
        """금지된 액션"""
        engine = self._make_engine(tmp_path)
        assert engine.is_action_allowed("delete_data") is False

    def test_consent_required_action(self, tmp_path):
        """동의 필요 액션은 None 반환"""
        engine = self._make_engine(tmp_path)
        assert engine.is_action_allowed("send_email") is None

    def test_unknown_action_denied(self, tmp_path):
        """알 수 없는 액션은 거부"""
        engine = self._make_engine(tmp_path)
        assert engine.is_action_allowed("unknown_action") is False

    def test_requires_consent(self, tmp_path):
        """requires_consent 확인"""
        engine = self._make_engine(tmp_path)
        assert engine.requires_consent("send_email") is True
        assert engine.requires_consent("crawl") is False

    def test_get_forbidden_actions(self, tmp_path):
        """금지 액션 목록"""
        engine = self._make_engine(tmp_path)
        forbidden = engine.get_forbidden_actions()
        assert "delete_data" in forbidden


# =============================================================================
# Threshold 조회 테스트
# =============================================================================


class TestRulesEngineGetThreshold:
    """RulesEngine.get_threshold 테스트"""

    def test_get_existing_threshold(self, tmp_path):
        """존재하는 임계값 조회"""
        rules_data = {"thresholds": {"rank_change": 5}}
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(rules_data), encoding="utf-8")
        engine = RulesEngine(rules_path=rules_file)
        assert engine.get_threshold("rank_change") == 5

    def test_get_nonexistent_threshold(self, tmp_path):
        """존재하지 않는 임계값"""
        engine = RulesEngine(rules_path=tmp_path / "nonexistent.json")
        assert engine.get_threshold("unknown") is None
        assert engine.get_threshold("unknown", default=99) == 99


# =============================================================================
# 유틸리티 테스트
# =============================================================================


class TestRulesEngineUtilities:
    """RulesEngine 유틸리티 메서드 테스트"""

    def _make_engine(self, tmp_path):
        rules_data = {
            "schedule_rules": [
                {"id": "s1", "name": "S1", "condition": {}, "action": "a", "enabled": True}
            ],
            "alert_rules": [
                {"id": "a1", "name": "A1", "condition": {}, "action": "b", "enabled": True}
            ],
            "priority_rules": [
                {"id": "p1", "name": "P1", "condition": {}, "action": "c", "enabled": True}
            ],
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(rules_data), encoding="utf-8")
        return RulesEngine(rules_path=rules_file)

    def test_get_all_rules(self, tmp_path):
        """모든 규칙 반환"""
        engine = self._make_engine(tmp_path)
        all_rules = engine.get_all_rules()
        assert "schedule" in all_rules
        assert "alert" in all_rules
        assert "priority" in all_rules
        assert len(all_rules["schedule"]) == 1

    def test_get_rule_by_id_found(self, tmp_path):
        """ID로 규칙 찾기"""
        engine = self._make_engine(tmp_path)
        rule = engine.get_rule_by_id("s1")
        assert rule is not None
        assert rule.name == "S1"

    def test_get_rule_by_id_not_found(self, tmp_path):
        """ID로 규칙 못찾기"""
        engine = self._make_engine(tmp_path)
        rule = engine.get_rule_by_id("nonexistent")
        assert rule is None

    def test_to_summary(self, tmp_path):
        """규칙 요약"""
        engine = self._make_engine(tmp_path)
        summary = engine.to_summary()
        assert "1 schedule" in summary
        assert "1 alert" in summary
        assert "1 priority" in summary
