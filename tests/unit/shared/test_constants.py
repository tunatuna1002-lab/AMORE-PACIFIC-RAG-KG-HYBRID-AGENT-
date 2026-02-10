"""Tests for src.shared.constants module."""

from src.shared.constants import (
    CACHE_TTL_MINUTES,
    CHATBOT_TEMPERATURE,
    CRAWL_BATCH_SIZE,
    CRAWL_DELAY_SECONDS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    HHI_CONCENTRATED,
    HHI_MODERATE,
    INSIGHT_TEMPERATURE,
    JSON_TEMPERATURE,
    MAX_CACHE_SIZE,
    MAX_RETRIES,
    MAX_TOKENS_DEFAULT,
    MAX_TOKENS_LONG,
    RANK_SHOCK_THRESHOLD,
    REQUEST_TIMEOUT,
    SOS_DOMINANT,
    SOS_STRONG,
    SOS_WEAK,
    SUGGESTION_MAX_COUNT,
    SUGGESTION_MAX_TOKENS,
    SUGGESTION_TEMPERATURE,
    TOP_100,
    TOP_N_DEFAULT,
    TOP_N_EXTENDED,
    VOLATILITY_WINDOW_DAYS,
)


class TestRankingThresholds:
    def test_top_n_values(self):
        assert TOP_N_DEFAULT == 10
        assert TOP_N_EXTENDED == 20
        assert TOP_100 == 100

    def test_hierarchy(self):
        assert TOP_N_DEFAULT < TOP_N_EXTENDED < TOP_100


class TestMarketMetrics:
    def test_sos_thresholds(self):
        assert SOS_DOMINANT > SOS_STRONG > SOS_WEAK
        assert SOS_DOMINANT == 30.0
        assert SOS_STRONG == 15.0
        assert SOS_WEAK == 5.0

    def test_hhi_thresholds(self):
        assert HHI_CONCENTRATED > HHI_MODERATE
        assert 0 < HHI_MODERATE < HHI_CONCENTRATED <= 1.0


class TestRankAnalysis:
    def test_rank_shock_threshold(self):
        assert RANK_SHOCK_THRESHOLD > 0
        assert isinstance(RANK_SHOCK_THRESHOLD, int)

    def test_volatility_window(self):
        assert VOLATILITY_WINDOW_DAYS > 0


class TestLLMSettings:
    def test_default_model(self):
        assert DEFAULT_MODEL == "gpt-4.1-mini"

    def test_temperatures_range(self):
        for temp in [
            CHATBOT_TEMPERATURE,
            INSIGHT_TEMPERATURE,
            DEFAULT_TEMPERATURE,
            JSON_TEMPERATURE,
        ]:
            assert 0.0 <= temp <= 2.0

    def test_chatbot_lower_than_insight(self):
        assert CHATBOT_TEMPERATURE < INSIGHT_TEMPERATURE

    def test_json_temperature_is_low(self):
        assert JSON_TEMPERATURE <= 0.5

    def test_max_tokens(self):
        assert MAX_TOKENS_DEFAULT > 0
        assert MAX_TOKENS_LONG > MAX_TOKENS_DEFAULT


class TestCacheSettings:
    def test_cache_values(self):
        assert CACHE_TTL_MINUTES > 0
        assert MAX_CACHE_SIZE > 0


class TestAPISettings:
    def test_timeout(self):
        assert REQUEST_TIMEOUT > 0

    def test_retries(self):
        assert MAX_RETRIES >= 1


class TestSuggestionSettings:
    def test_suggestion_values(self):
        assert SUGGESTION_MAX_COUNT > 0
        assert SUGGESTION_MAX_TOKENS > 0
        assert 0.0 <= SUGGESTION_TEMPERATURE <= 2.0


class TestCrawlingConfig:
    def test_crawl_values(self):
        assert CRAWL_BATCH_SIZE > 0
        assert CRAWL_DELAY_SECONDS > 0
