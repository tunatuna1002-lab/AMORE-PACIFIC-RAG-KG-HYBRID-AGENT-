"""
API Dependencies 단위 테스트

테스트 대상: src/api/dependencies.py
Coverage target: 60%+
"""

import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

# Mock jwt and slowapi before importing the module
with patch.dict(
    "os.environ",
    {"API_KEY": "test-api-key-12345"},  # pragma: allowlist secret
    clear=False,
):
    from src.api.dependencies import (
        _extract_response_keywords,
        _generate_entity_suggestions,
        _generate_type_suggestions,
        add_to_memory,
        build_data_context,
        cleanup_expired_sessions,
        conversation_memory,
        generate_dynamic_suggestions,
        get_base_url,
        get_conversation_history,
        load_dashboard_data,
        log_chat_interaction,
        session_last_activity,
    )
    from src.rag.router import QueryType


class TestCleanupExpiredSessions:
    """cleanup_expired_sessions 테스트"""

    def setup_method(self):
        """각 테스트 전 세션 데이터 초기화"""
        conversation_memory.clear()
        session_last_activity.clear()

    def test_cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        now = datetime.now()
        # 만료된 세션 (2시간 전)
        session_last_activity["expired-1"] = now - timedelta(hours=2)
        conversation_memory["expired-1"].append({"role": "user", "content": "hi"})
        # 활성 세션
        session_last_activity["active-1"] = now
        conversation_memory["active-1"].append({"role": "user", "content": "hello"})

        cleaned = cleanup_expired_sessions()

        assert cleaned == 1
        assert "expired-1" not in session_last_activity
        assert "expired-1" not in conversation_memory
        assert "active-1" in session_last_activity

    def test_cleanup_no_expired(self):
        """만료된 세션이 없는 경우"""
        now = datetime.now()
        session_last_activity["active-1"] = now

        cleaned = cleanup_expired_sessions()

        assert cleaned == 0
        assert "active-1" in session_last_activity

    def test_cleanup_empty_sessions(self):
        """세션이 없는 경우"""
        cleaned = cleanup_expired_sessions()
        assert cleaned == 0


class TestGetConversationHistory:
    """get_conversation_history 테스트"""

    def setup_method(self):
        conversation_memory.clear()
        session_last_activity.clear()

    def test_get_history_basic(self):
        """기본 대화 기록 조회"""
        conversation_memory["sess-1"].append({"role": "user", "content": "Hello"})
        conversation_memory["sess-1"].append({"role": "assistant", "content": "Hi there"})

        history = get_conversation_history("sess-1")

        assert "[사용자]: Hello" in history
        assert "[AI]: Hi there" in history

    def test_get_history_empty(self):
        """대화 기록 없는 경우"""
        history = get_conversation_history("no-such-session")
        assert history == ""

    def test_get_history_with_limit(self):
        """제한된 대화 기록 조회"""
        for i in range(10):
            conversation_memory["sess-2"].append({"role": "user", "content": f"Message {i}"})

        history = get_conversation_history("sess-2", limit=3)

        # 마지막 3개만 포함되어야 함
        assert "Message 7" in history
        assert "Message 8" in history
        assert "Message 9" in history

    def test_get_history_truncates_long_content(self):
        """긴 내용 truncation"""
        long_content = "A" * 200
        conversation_memory["sess-3"].append({"role": "user", "content": long_content})

        history = get_conversation_history("sess-3")

        assert "..." in history


class TestAddToMemory:
    """add_to_memory 테스트"""

    def setup_method(self):
        conversation_memory.clear()
        session_last_activity.clear()

    def test_add_to_memory_basic(self):
        """기본 메모리 추가"""
        add_to_memory("sess-1", "user", "Hello")

        assert len(conversation_memory["sess-1"]) == 1
        assert conversation_memory["sess-1"][0]["role"] == "user"
        assert conversation_memory["sess-1"][0]["content"] == "Hello"
        assert "timestamp" in conversation_memory["sess-1"][0]
        assert "sess-1" in session_last_activity

    def test_add_to_memory_trims(self):
        """MAX_MEMORY_TURNS * 2 초과 시 오래된 항목 제거"""
        # MAX_MEMORY_TURNS = 10, so > 20 entries triggers trim
        for i in range(25):
            add_to_memory("sess-trim", "user", f"Msg {i}")

        assert len(conversation_memory["sess-trim"]) == 20


class TestLoadDashboardData:
    """load_dashboard_data 테스트"""

    def test_load_valid_json(self, tmp_path):
        """유효한 JSON 파일 로드"""
        data = {"home": {"insight_message": "Test"}, "brand": {}}
        data_file = tmp_path / "dashboard_data.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        with patch("src.api.dependencies.DATA_PATH", str(data_file)):
            result = load_dashboard_data()

        assert result == data

    def test_load_file_not_found(self):
        """파일 없는 경우 빈 딕셔너리"""
        with patch("src.api.dependencies.DATA_PATH", "/nonexistent/path/data.json"):
            result = load_dashboard_data()

        assert result == {}

    def test_load_invalid_json(self, tmp_path):
        """잘못된 JSON 파일"""
        data_file = tmp_path / "bad_data.json"
        data_file.write_text("not json {{{", encoding="utf-8")

        with patch("src.api.dependencies.DATA_PATH", str(data_file)):
            result = load_dashboard_data()

        assert result == {}


class TestBuildDataContext:
    """build_data_context 테스트"""

    def test_empty_data(self):
        """데이터가 없는 경우"""
        result = build_data_context({}, QueryType.ANALYSIS, {})
        assert "현재 데이터가 없습니다" in result

    def test_metadata_always_included(self):
        """메타데이터가 항상 포함"""
        data = {
            "metadata": {
                "data_date": "2026-01-15",
                "total_products": 100,
                "laneige_products": 5,
            }
        }
        result = build_data_context(data, QueryType.DATA_QUERY, {})

        assert "2026-01-15" in result
        assert "100" in result
        assert "5" in result

    def test_analysis_includes_kpis(self):
        """분석 질문에 KPI 포함"""
        data = {
            "metadata": {},
            "brand": {
                "kpis": {
                    "sos": 12.5,
                    "top10_count": 3,
                    "avg_rank": 25,
                    "hhi": 0.15,
                },
                "competitors": [],
            },
        }
        result = build_data_context(data, QueryType.ANALYSIS, {})

        assert "SoS" in result
        assert "12.5" in result

    def test_analysis_includes_competitors(self):
        """분석 질문에 경쟁사 포함"""
        data = {
            "metadata": {},
            "brand": {
                "kpis": {},
                "competitors": [
                    {
                        "brand": "COSRX",
                        "sos": 15.0,
                        "avg_rank": 20,
                        "product_count": 8,
                    }
                ],
            },
        }
        result = build_data_context(data, QueryType.ANALYSIS, {})

        assert "COSRX" in result
        assert "경쟁사" in result

    def test_data_query_includes_products(self):
        """데이터 조회 시 제품 포함"""
        data = {
            "metadata": {},
            "products": {
                "B0TEST": {
                    "name": "Lip Sleeping Mask",
                    "rank": 3,
                    "rank_delta": "+2",
                    "rating": 4.5,
                }
            },
        }
        result = build_data_context(data, QueryType.DATA_QUERY, {})

        assert "Lip Sleeping Mask" in result
        assert "제품 현황" in result

    def test_competitor_brand_mentioned(self):
        """경쟁사 브랜드가 언급된 경우 경쟁사 데이터 포함"""
        data = {
            "metadata": {},
            "brand": {
                "kpis": {},
                "competitors": [
                    {
                        "brand": "COSRX",
                        "sos": 15.0,
                        "avg_rank": 20,
                        "product_count": 8,
                    }
                ],
            },
        }
        entities = {"brands": ["COSRX"]}
        result = build_data_context(data, QueryType.DATA_QUERY, entities)

        assert "COSRX" in result

    def test_analysis_includes_action_items(self):
        """분석 질문에 액션 아이템 포함"""
        data = {
            "metadata": {},
            "brand": {"kpis": {}, "competitors": []},
            "home": {
                "action_items": [
                    {
                        "priority": "HIGH",
                        "product_name": "Lip Mask",
                        "signal": "Rank dropped",
                        "action_tag": "Monitor",
                    }
                ]
            },
        }
        result = build_data_context(data, QueryType.ANALYSIS, {})

        assert "액션 아이템" in result
        assert "Lip Mask" in result

    def test_categories_included_for_analysis(self):
        """분석 질문에 카테고리 포함"""
        data = {
            "metadata": {},
            "categories": {
                "lip_care": {
                    "name": "Lip Care",
                    "sos": 15.0,
                    "best_rank": 3,
                    "cpi": 95,
                }
            },
        }
        result = build_data_context(data, QueryType.ANALYSIS, {})

        assert "Lip Care" in result
        assert "카테고리" in result


class TestGenerateDynamicSuggestions:
    """generate_dynamic_suggestions 테스트"""

    def test_basic_suggestions(self):
        """기본 후속 질문 생성"""
        suggestions = generate_dynamic_suggestions(
            QueryType.ANALYSIS, {}, "LANEIGE 순위가 하락했습니다."
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3

    def test_entity_based_suggestions(self):
        """엔티티 기반 제안"""
        entities = {"brands": ["LANEIGE", "COSRX"], "categories": [], "indicators": []}
        suggestions = generate_dynamic_suggestions(QueryType.DATA_QUERY, entities, "")

        assert len(suggestions) > 0

    def test_no_duplicates(self):
        """중복 제거"""
        suggestions = generate_dynamic_suggestions(
            QueryType.ANALYSIS,
            {"brands": ["LANEIGE"], "categories": [], "indicators": ["sos"]},
            "순위가 하락하고 있습니다.",
        )

        assert len(suggestions) == len(set(suggestions))

    def test_max_3_suggestions(self):
        """최대 3개 제안"""
        entities = {
            "brands": ["LANEIGE", "COSRX"],
            "categories": ["Lip Care"],
            "indicators": ["sos", "hhi"],
        }
        suggestions = generate_dynamic_suggestions(
            QueryType.COMBINATION, entities, "순위 하락 경쟁사 가격 트렌드"
        )

        assert len(suggestions) <= 3


class TestExtractResponseKeywords:
    """_extract_response_keywords 테스트"""

    def test_rank_drop_detected(self):
        """순위 하락 키워드 감지"""
        keywords = _extract_response_keywords("LANEIGE 순위가 급락했습니다.")
        assert len(keywords) > 0
        assert any("순위" in k or "하락" in k or "분석" in k for k in keywords)

    def test_rank_rise_detected(self):
        """순위 상승 키워드 감지"""
        keywords = _extract_response_keywords("LANEIGE 순위가 급등했습니다.")
        assert len(keywords) > 0

    def test_competitor_detected(self):
        """경쟁사 키워드 감지"""
        keywords = _extract_response_keywords("경쟁사 COSRX의 영향이 큽니다.")
        assert len(keywords) > 0

    def test_sos_detected(self):
        """SoS 키워드 감지"""
        keywords = _extract_response_keywords("현재 SoS 점유율이 12.5%입니다.")
        assert len(keywords) > 0

    def test_no_keywords(self):
        """키워드 없는 응답"""
        keywords = _extract_response_keywords("일반적인 내용입니다.")
        assert isinstance(keywords, list)

    def test_max_2_keywords(self):
        """최대 2개 키워드"""
        keywords = _extract_response_keywords(
            "순위가 급락하고 경쟁사가 가격 인하하며 리뷰가 좋고 트렌드가 변화"
        )
        assert len(keywords) <= 2


class TestGenerateEntitySuggestions:
    """_generate_entity_suggestions 테스트"""

    def test_brand_suggestions(self):
        """브랜드 기반 제안"""
        suggestions = _generate_entity_suggestions(["LANEIGE"], [], [])
        assert len(suggestions) > 0
        assert any("LANEIGE" in s for s in suggestions)

    def test_multiple_brands_vs(self):
        """여러 브랜드 비교 제안"""
        suggestions = _generate_entity_suggestions(["LANEIGE", "COSRX"], [], [])
        assert any("vs" in s for s in suggestions)

    def test_category_suggestions(self):
        """카테고리 기반 제안"""
        suggestions = _generate_entity_suggestions([], ["Lip Care"], [])
        assert any("Lip Care" in s for s in suggestions)

    def test_indicator_suggestions(self):
        """지표 기반 제안"""
        suggestions = _generate_entity_suggestions([], [], ["sos"])
        assert any("SOS" in s for s in suggestions)

    def test_empty_entities(self):
        """엔티티 없는 경우"""
        suggestions = _generate_entity_suggestions([], [], [])
        assert suggestions == []


class TestGenerateTypeSuggestions:
    """_generate_type_suggestions 테스트"""

    def test_definition_type(self):
        """정의 질의 유형 제안"""
        suggestions = _generate_type_suggestions(QueryType.DEFINITION, [], ["sos"])
        assert len(suggestions) > 0

    def test_interpretation_type(self):
        """해석 질의 유형 제안"""
        suggestions = _generate_type_suggestions(QueryType.INTERPRETATION, [], [])
        assert len(suggestions) > 0

    def test_data_query_type(self):
        """데이터 조회 유형 제안"""
        suggestions = _generate_type_suggestions(QueryType.DATA_QUERY, [], [])
        assert len(suggestions) > 0

    def test_analysis_type(self):
        """분석 유형 제안"""
        suggestions = _generate_type_suggestions(QueryType.ANALYSIS, [], [])
        assert len(suggestions) > 0

    def test_combination_type(self):
        """조합 유형 제안"""
        suggestions = _generate_type_suggestions(QueryType.COMBINATION, [], [])
        assert len(suggestions) > 0

    def test_unknown_type(self):
        """알 수 없는 유형 제안"""
        suggestions = _generate_type_suggestions(QueryType.UNKNOWN, [], [])
        assert len(suggestions) > 0


class TestGetBaseUrl:
    """get_base_url 테스트"""

    @patch.dict(
        "os.environ",
        {"DASHBOARD_URL": "https://custom.example.com/"},
        clear=False,
    )
    def test_dashboard_url_env(self):
        """DASHBOARD_URL 환경변수 사용"""
        result = get_base_url()
        assert result == "https://custom.example.com"

    @patch.dict(
        "os.environ",
        {"RAILWAY_PUBLIC_DOMAIN": "myapp.railway.app"},
        clear=False,
    )
    def test_railway_domain(self):
        """Railway 도메인 사용"""
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("DASHBOARD_URL", None)
            result = get_base_url()
        assert result == "https://myapp.railway.app"

    @patch.dict("os.environ", {"PORT": "3000"}, clear=False)
    def test_localhost_with_port(self):
        """로컬 환경 포트 사용"""
        os.environ.pop("DASHBOARD_URL", None)
        os.environ.pop("RAILWAY_PUBLIC_DOMAIN", None)
        result = get_base_url()
        assert result == "http://localhost:3000"


class TestLogChatInteraction:
    """log_chat_interaction 테스트"""

    def test_log_chat_interaction(self):
        """챗봇 대화 로그 기록"""
        with patch("src.api.dependencies.audit_logger") as mock_logger:
            log_chat_interaction(
                session_id="sess-1",
                user_query="What is SoS?",
                ai_response="SoS stands for Share of Shelf",
                query_type="definition",
                confidence=0.95,
                entities={"indicators": ["sos"]},
                sources=["Strategic Indicators Definition"],
                response_time_ms=150.5,
            )

            mock_logger.info.assert_called_once()
            logged_json = mock_logger.info.call_args[0][0]
            logged_data = json.loads(logged_json)

            assert logged_data["session_id"] == "sess-1"
            assert logged_data["user_query"] == "What is SoS?"
            assert logged_data["confidence"] == 0.95

    def test_log_long_response_truncated(self):
        """긴 응답 truncation"""
        long_response = "A" * 600

        with patch("src.api.dependencies.audit_logger") as mock_logger:
            log_chat_interaction(
                session_id="sess-2",
                user_query="Test",
                ai_response=long_response,
                query_type="analysis",
                confidence=0.8,
                entities={},
                sources=[],
                response_time_ms=200.0,
            )

            logged_json = mock_logger.info.call_args[0][0]
            logged_data = json.loads(logged_json)
            assert logged_data["ai_response"].endswith("...")
            assert len(logged_data["ai_response"]) <= 504  # 500 + "..."


class TestJWTHelpers:
    """JWT 관련 함수 테스트"""

    @patch.dict(
        "os.environ",
        {"JWT_SECRET_KEY": "test-secret-key-for-jwt-12345678"},  # pragma: allowlist secret
        clear=False,
    )
    def test_create_and_verify_token(self):
        """토큰 생성 및 검증"""
        import src.api.dependencies as deps

        original_key = deps.JWT_SECRET_KEY
        deps.JWT_SECRET_KEY = "test-secret-key-for-jwt-12345678"  # pragma: allowlist secret

        try:
            token = deps.create_email_verification_token("test@example.com")
            assert isinstance(token, str)

            result = deps.verify_jwt_email_token(token)
            assert result["valid"] is True
            assert result["email"] == "test@example.com"
        finally:
            deps.JWT_SECRET_KEY = original_key

    def test_create_token_no_secret(self):
        """JWT_SECRET_KEY 없으면 에러"""
        import src.api.dependencies as deps

        original_key = deps.JWT_SECRET_KEY
        deps.JWT_SECRET_KEY = None

        try:
            with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
                deps.create_email_verification_token("test@example.com")
        finally:
            deps.JWT_SECRET_KEY = original_key

    def test_verify_token_no_secret(self):
        """JWT_SECRET_KEY 없으면 invalid 반환"""
        import src.api.dependencies as deps

        original_key = deps.JWT_SECRET_KEY
        deps.JWT_SECRET_KEY = None

        try:
            result = deps.verify_jwt_email_token("some-token")
            assert result["valid"] is False
        finally:
            deps.JWT_SECRET_KEY = original_key

    def test_verify_invalid_token(self):
        """유효하지 않은 토큰"""
        import src.api.dependencies as deps

        original_key = deps.JWT_SECRET_KEY
        deps.JWT_SECRET_KEY = "test-secret-key-for-jwt-12345678"  # pragma: allowlist secret

        try:
            result = deps.verify_jwt_email_token("invalid-token-string")
            assert result["valid"] is False
        finally:
            deps.JWT_SECRET_KEY = original_key

    def test_verify_wrong_purpose_token(self):
        """목적이 다른 토큰"""
        import jwt as pyjwt

        import src.api.dependencies as deps

        original_key = deps.JWT_SECRET_KEY
        deps.JWT_SECRET_KEY = "test-secret-key-for-jwt-12345678"  # pragma: allowlist secret

        try:
            from datetime import UTC

            token = pyjwt.encode(
                {
                    "email": "test@example.com",
                    "purpose": "wrong_purpose",
                    "exp": datetime.now(UTC) + timedelta(minutes=30),
                },
                deps.JWT_SECRET_KEY,
                algorithm="HS256",
            )
            result = deps.verify_jwt_email_token(token)
            assert result["valid"] is False
        finally:
            deps.JWT_SECRET_KEY = original_key


class TestVerifyApiKey:
    """verify_api_key 테스트"""

    @pytest.mark.asyncio
    async def test_verify_valid_key(self):
        """유효한 API Key"""

        import src.api.dependencies as deps

        original_key = deps.API_KEY
        deps.API_KEY = "test-api-key-12345"

        try:
            result = await deps.verify_api_key("test-api-key-12345")
            assert result == "test-api-key-12345"
        finally:
            deps.API_KEY = original_key

    @pytest.mark.asyncio
    async def test_verify_missing_key(self):
        """API Key 없는 경우"""
        from fastapi import HTTPException

        import src.api.dependencies as deps

        with pytest.raises(HTTPException) as exc_info:
            await deps.verify_api_key(None)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_verify_invalid_key(self):
        """잘못된 API Key"""
        from fastapi import HTTPException

        import src.api.dependencies as deps

        original_key = deps.API_KEY
        deps.API_KEY = "correct-key"  # pragma: allowlist secret

        try:
            with pytest.raises(HTTPException) as exc_info:
                await deps.verify_api_key("wrong-key")
            assert exc_info.value.status_code == 403
        finally:
            deps.API_KEY = original_key
