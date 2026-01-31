"""
프롬프트 컴포넌트 테스트

날짜 컨텍스트, 보안 규칙, 환각 방지 규칙 테스트
"""

from datetime import datetime

from prompts.components import (
    build_date_context,
    get_hallucination_prevention,
    get_security_rules,
)


class TestBuildDateContext:
    """날짜 컨텍스트 생성 테스트"""

    def test_default_uses_current_date(self):
        """기본값으로 현재 날짜 사용"""
        context = build_date_context()
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in context
        assert "시점 정보" in context

    def test_custom_data_date(self):
        """지정된 데이터 수집일 사용"""
        context = build_date_context(data_date="2026-01-15")
        assert "2026-01-15" in context
        assert "데이터 수집일: 2026-01-15" in context

    def test_custom_current_date(self):
        """지정된 현재 날짜 사용"""
        context = build_date_context(current_date="2026-01-31")
        assert "오늘 날짜: 2026-01-31" in context

    def test_custom_analysis_period(self):
        """지정된 분석 기간 사용"""
        context = build_date_context(start_date="2026-01-01", end_date="2026-01-31")
        assert "2026-01-31" in context

    def test_contains_required_rules(self):
        """필수 규칙 포함 확인"""
        context = build_date_context(data_date="2026-01-15")
        assert "현재" in context
        assert "미래 날짜" in context
        assert "절대 언급 금지" in context

    def test_contains_emoji_marker(self):
        """시각적 구분을 위한 이모지 포함"""
        context = build_date_context()
        assert "⏰" in context


class TestGetSecurityRules:
    """보안 규칙 테스트"""

    def test_returns_non_empty_string(self):
        """비어있지 않은 문자열 반환"""
        rules = get_security_rules()
        assert isinstance(rules, str)
        assert len(rules) > 0

    def test_contains_system_prompt_protection(self):
        """시스템 프롬프트 보호 규칙 포함"""
        rules = get_security_rules()
        assert "시스템 프롬프트" in rules
        assert "공개하지 마세요" in rules

    def test_contains_jailbreak_protection(self):
        """Jailbreak 방지 규칙 포함"""
        rules = get_security_rules()
        assert "jailbreak" in rules.lower() or "역할을 바꾸라는" in rules

    def test_contains_url_protection(self):
        """외부 URL 접근 방지 규칙 포함"""
        rules = get_security_rules()
        assert "URL" in rules

    def test_contains_emoji_marker(self):
        """시각적 구분을 위한 이모지 포함"""
        rules = get_security_rules()
        assert "🔒" in rules


class TestGetHallucinationPrevention:
    """환각 방지 규칙 테스트"""

    def test_returns_non_empty_string(self):
        """비어있지 않은 문자열 반환"""
        rules = get_hallucination_prevention()
        assert isinstance(rules, str)
        assert len(rules) > 0

    def test_contains_data_fabrication_prevention(self):
        """데이터 생성 방지 규칙 포함"""
        rules = get_hallucination_prevention()
        assert "수치" in rules
        assert "생성하지 마세요" in rules

    def test_contains_uncertainty_expression(self):
        """불확실성 표현 규칙 포함"""
        rules = get_hallucination_prevention()
        assert "불확실" in rules or "확인이 필요" in rules

    def test_contains_sales_estimation_prevention(self):
        """매출 추정 방지 규칙 포함"""
        rules = get_hallucination_prevention()
        assert "매출" in rules or "판매량" in rules

    def test_contains_emoji_marker(self):
        """시각적 구분을 위한 이모지 포함"""
        rules = get_hallucination_prevention()
        assert "⚠️" in rules


class TestContextBuilderIntegration:
    """ContextBuilder 통합 테스트"""

    def test_build_system_prompt_with_date(self):
        """날짜 포함 시스템 프롬프트 생성"""
        from src.rag.context_builder import ContextBuilder

        builder = ContextBuilder()
        prompt = builder.build_system_prompt(include_guardrails=True, data_date="2026-01-31")

        assert "시점 정보" in prompt
        assert "2026-01-31" in prompt
        assert "보안 규칙" in prompt
        assert "환각 방지" in prompt

    def test_build_system_prompt_without_guardrails(self):
        """안전장치 없는 시스템 프롬프트"""
        from src.rag.context_builder import ContextBuilder

        builder = ContextBuilder()
        prompt = builder.build_system_prompt(include_guardrails=False, data_date="2026-01-31")

        # 날짜 컨텍스트는 항상 포함
        assert "시점 정보" in prompt
        assert "2026-01-31" in prompt

        # 안전장치는 미포함
        assert "보안 규칙" not in prompt

    def test_build_system_prompt_default_date(self):
        """기본 날짜로 시스템 프롬프트 생성"""
        from src.rag.context_builder import ContextBuilder

        builder = ContextBuilder()
        prompt = builder.build_system_prompt(include_guardrails=True)

        today = datetime.now().strftime("%Y-%m-%d")
        assert today in prompt
