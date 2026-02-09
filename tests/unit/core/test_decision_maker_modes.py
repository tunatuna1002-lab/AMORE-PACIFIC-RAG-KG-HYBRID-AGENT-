"""
DecisionMaker 모드 분기 테스트
"""

from src.core.decision_maker import DecisionMaker
from src.core.models import Decision


class TestDecisionMakerModes:
    """DecisionMaker 모드 분기 테스트"""

    def setup_method(self):
        self.dm = DecisionMaker()

    def test_has_mode_prompts(self):
        """MODE_PROMPTS 존재"""
        assert hasattr(self.dm, "MODE_PROMPTS")
        assert "high" in self.dm.MODE_PROMPTS
        assert "medium" in self.dm.MODE_PROMPTS
        assert "low" in self.dm.MODE_PROMPTS
        assert "unknown" in self.dm.MODE_PROMPTS

    def test_mode_prompt_content(self):
        """각 모드 프롬프트 내용 확인"""
        assert "direct_answer" in self.dm.MODE_PROMPTS["high"]
        assert "도구" in self.dm.MODE_PROMPTS["low"]

    def test_decide_signature_accepts_confidence_level(self):
        """decide()가 confidence_level 파라미터 수용"""
        import inspect

        sig = inspect.signature(self.dm.decide)
        assert "confidence_level" in sig.parameters

    def test_confidence_level_default(self):
        """기본값은 medium"""
        import inspect

        sig = inspect.signature(self.dm.decide)
        param = sig.parameters["confidence_level"]
        assert param.default == "medium"

    def test_fallback_decision_unchanged(self):
        """폴백 결정은 모드에 무관"""
        d = self.dm._fallback_decision("test error")
        assert isinstance(d, Decision)
        assert d.tool == "direct_answer"

    def test_parse_decision_unchanged(self):
        """파싱 로직은 모드에 무관"""
        json_str = '{"tool": "direct_answer", "tool_params": {}, "reason": "test", "confidence": 0.8, "key_points": []}'
        d = self.dm._parse_decision(json_str)
        assert isinstance(d, Decision)
        assert d.tool == "direct_answer"
