"""
ResponseTemplates 단위 테스트
============================
안전장치, 일일 인사이트, 액션 큐, 지표 해석 검증
"""

from src.rag.templates import ResponseTemplates

# ---------------------------------------------------------------------------
# Guardrails (안전장치)
# ---------------------------------------------------------------------------


class TestGuardrails:
    """안전장치 클래스 속성 검증"""

    def test_forbidden_phrases_exist(self):
        assert len(ResponseTemplates.FORBIDDEN_PHRASES) > 0

    def test_hedging_phrases_exist(self):
        assert len(ResponseTemplates.HEDGING_PHRASES) > 0

    def test_caution_notes_exist(self):
        assert len(ResponseTemplates.CAUTION_NOTES) > 0

    def test_apply_guardrails_returns_text(self):
        result = ResponseTemplates.apply_guardrails("정상적인 분석 결과입니다.")
        assert isinstance(result, str)
        assert "정상적인 분석 결과입니다." in result

    def test_add_caution_note(self):
        text = "분석 결과입니다."
        result = ResponseTemplates.add_caution_note(text, note_index=0)
        assert text in result
        assert ResponseTemplates.CAUTION_NOTES[0] in result

    def test_add_caution_note_wraps_index(self):
        """인덱스가 범위를 넘으면 순환"""
        text = "테스트"
        n = len(ResponseTemplates.CAUTION_NOTES)
        result = ResponseTemplates.add_caution_note(text, note_index=n)
        assert ResponseTemplates.CAUTION_NOTES[0] in result


# ---------------------------------------------------------------------------
# generate_daily_insight
# ---------------------------------------------------------------------------


class TestGenerateDailyInsight:
    """일일 인사이트 생성 테스트"""

    def test_rank_shock_alert(self):
        alerts = [{"type": "rank_shock"}, {"type": "rank_shock"}]
        result = ResponseTemplates.generate_daily_insight({}, alerts)
        assert "2개 상품" in result
        assert "순위 급변" in result

    def test_price_quality_mismatch(self):
        metrics = {"cpi": 120, "avg_rating_gap": -0.5}
        result = ResponseTemplates.generate_daily_insight(metrics, [])
        assert "가격" in result and "품질" in result

    def test_high_hhi_market(self):
        metrics = {"hhi": 0.30}
        result = ResponseTemplates.generate_daily_insight(metrics, [])
        assert "집중" in result or "고착" in result

    def test_low_hhi_market(self):
        metrics = {"hhi": 0.10}
        result = ResponseTemplates.generate_daily_insight(metrics, [])
        assert "분산" in result or "경쟁" in result

    def test_stable_fallback(self):
        result = ResponseTemplates.generate_daily_insight({}, [])
        assert "안정" in result

    def test_alert_priority_over_metrics(self):
        """알림이 있으면 지표보다 우선"""
        alerts = [{"type": "rank_shock"}]
        metrics = {"cpi": 120, "avg_rating_gap": -0.5}
        result = ResponseTemplates.generate_daily_insight(metrics, alerts)
        assert "순위 급변" in result


# ---------------------------------------------------------------------------
# generate_action_queue
# ---------------------------------------------------------------------------


class TestGenerateActionQueue:
    """액션 큐 생성 테스트"""

    def test_rank_shock_action(self):
        alerts = [{"type": "rank_shock"}, {"type": "rank_shock"}]
        actions = ResponseTemplates.generate_action_queue(alerts, {})
        assert len(actions) >= 1
        assert "2개" in actions[0]["title"]

    def test_price_quality_action(self):
        alerts = []
        metrics = {"cpi": 120, "avg_rating_gap": -0.5}
        actions = ResponseTemplates.generate_action_queue(alerts, metrics)
        assert any("가격" in a["title"] for a in actions)

    def test_churn_action(self):
        metrics = {"churn_rate": 0.3}
        actions = ResponseTemplates.generate_action_queue([], metrics)
        assert any("변동성" in a["title"] for a in actions)

    def test_no_action_fallback(self):
        actions = ResponseTemplates.generate_action_queue([], {})
        assert len(actions) == 1
        assert "경고 신호가 없습니다" in actions[0]["title"]

    def test_max_items_limit(self):
        alerts = [{"type": "rank_shock"}]
        metrics = {"cpi": 120, "avg_rating_gap": -0.5, "churn_rate": 0.3}
        actions = ResponseTemplates.generate_action_queue(alerts, metrics, max_items=2)
        assert len(actions) <= 2


# ---------------------------------------------------------------------------
# format_metric_interpretation
# ---------------------------------------------------------------------------


class TestFormatMetricInterpretation:
    """지표 해석 포맷팅 테스트"""

    def test_sos_high(self):
        result = ResponseTemplates.format_metric_interpretation("sos", 20)
        assert "노출" in result

    def test_sos_low(self):
        result = ResponseTemplates.format_metric_interpretation("sos", 5)
        assert "약한" in result

    def test_hhi_high(self):
        result = ResponseTemplates.format_metric_interpretation("hhi", 0.30)
        assert "집중" in result

    def test_hhi_low(self):
        result = ResponseTemplates.format_metric_interpretation("hhi", 0.10)
        assert "분산" in result

    def test_cpi_high(self):
        result = ResponseTemplates.format_metric_interpretation("cpi", 120)
        assert "프리미엄" in result

    def test_cpi_low(self):
        result = ResponseTemplates.format_metric_interpretation("cpi", 80)
        assert "가성비" in result or "저가" in result

    def test_unknown_indicator(self):
        result = ResponseTemplates.format_metric_interpretation("xyz", 42)
        assert "xyz" in result and "42" in result


# ---------------------------------------------------------------------------
# get_system_prompt
# ---------------------------------------------------------------------------


class TestGetSystemPrompt:
    """시스템 프롬프트 테스트"""

    def test_returns_string(self):
        prompt = ResponseTemplates.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_guardrail_rules(self):
        prompt = ResponseTemplates.get_system_prompt()
        assert "단정 금지" in prompt
        assert "가능성 표현" in prompt

    def test_contains_metric_guide(self):
        prompt = ResponseTemplates.get_system_prompt()
        assert "SoS" in prompt
        assert "HHI" in prompt
        assert "CPI" in prompt


# ---------------------------------------------------------------------------
# Templates data structures
# ---------------------------------------------------------------------------


class TestTemplateStructures:
    """템플릿 데이터 구조 검증"""

    def test_daily_insight_templates_keys(self):
        expected = {"alert_priority", "price_quality_mismatch", "market_context", "stable"}
        assert expected == set(ResponseTemplates.DAILY_INSIGHT_TEMPLATES.keys())

    def test_brand_health_templates_keys(self):
        expected = {"sos", "market_structure", "alerts"}
        assert expected == set(ResponseTemplates.BRAND_HEALTH_TEMPLATES.keys())

    def test_action_queue_templates_keys(self):
        expected = {"rank_shock", "price_quality", "churn_high", "no_action"}
        assert expected == set(ResponseTemplates.ACTION_QUEUE_TEMPLATES.keys())

    def test_analysis_report_template_exists(self):
        tmpl = ResponseTemplates.ANALYSIS_REPORT_TEMPLATE
        assert "{brand}" in tmpl
        assert "{summary}" in tmpl
        assert "주의" in tmpl
