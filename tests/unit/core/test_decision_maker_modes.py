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
        """각 모드 프롬프트 내용 확인 (HIGH는 도구 호출 금지, LOW는 도구 사용 유도)"""
        assert "도구를 호출하지 말고" in self.dm.MODE_PROMPTS["high"]
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

    def test_key_points_parsing_unchanged(self):
        """본문의 "- " 줄만 핵심 포인트가 된다 (모드 무관)"""
        points = self.dm._key_points("근거 충분\n- SoS 2%\n설명 줄\n* HHI 0.07")
        assert points == ["SoS 2%", "HHI 0.07"]
