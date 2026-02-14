"""
Centralized Constants
=====================
All magic numbers and hardcoded values extracted to one place.
"""

# ==============================================================================
# TIMEZONE
# ==============================================================================

from datetime import timedelta, timezone

# Korea Standard Time (UTC+9)
KST = timezone(timedelta(hours=9))


# ==============================================================================
# RANKING THRESHOLDS
# ==============================================================================

# Number of top products to analyze in various contexts
TOP_N_DEFAULT = 10  # Standard analysis focus (e.g., "top 10 competitors")
TOP_N_EXTENDED = 20  # Extended analysis for deeper insights
TOP_100 = 100  # Full bestseller list crawled from Amazon


# ==============================================================================
# MARKET METRICS THRESHOLDS
# ==============================================================================

# Share of Shelf (SoS) thresholds - percentage of products in top N
SOS_DOMINANT = 30.0  # SoS > 30% indicates market dominance
SOS_STRONG = 15.0  # SoS > 15% indicates strong market position
SOS_WEAK = 5.0  # SoS < 5% indicates weak market presence

# Herfindahl-Hirschman Index (HHI) thresholds - market concentration
HHI_CONCENTRATED = 0.25  # HHI > 0.25 indicates highly concentrated market
HHI_MODERATE = 0.15  # HHI > 0.15 indicates moderately concentrated market


# ==============================================================================
# RANK ANALYSIS
# ==============================================================================

# Rank change detection
RANK_SHOCK_THRESHOLD = 5  # ±5 rank change is considered significant
VOLATILITY_WINDOW_DAYS = 7  # Rolling window for volatility calculations


# ==============================================================================
# LLM SETTINGS
# ==============================================================================

# Model configuration
DEFAULT_MODEL = "gpt-4.1-mini"  # Primary model for general tasks

# Temperature settings (E2E Audit Action Item - 2026-01-27)
# - Chatbot: 낮은 temperature = 사실적/일관된 답변
# - Insight: 높은 temperature = 창의적 분석/전략 제안
CHATBOT_TEMPERATURE = 0.4  # Conservative for factual Q&A
INSIGHT_TEMPERATURE = 0.6  # Creative for strategic insights
DEFAULT_TEMPERATURE = 0.7  # Legacy compatibility
JSON_TEMPERATURE = 0.3  # Lower temperature for structured output

MAX_TOKENS_DEFAULT = 2000  # Standard response length
MAX_TOKENS_LONG = 4000  # Extended response for complex analysis


# ==============================================================================
# CACHE SETTINGS
# ==============================================================================

# In-memory cache configuration
CACHE_TTL_MINUTES = 30  # Time-to-live for cached data
MAX_CACHE_SIZE = 100  # Maximum number of cached entries


# ==============================================================================
# API SETTINGS
# ==============================================================================

# HTTP client configuration
REQUEST_TIMEOUT = 60  # Timeout for external API calls (seconds)
MAX_RETRIES = 3  # Maximum retry attempts for failed requests


# ==============================================================================
# SUGGESTION GENERATION SETTINGS
# ==============================================================================

# LLM-based suggestion generation
SUGGESTION_TEMPERATURE = 0.7  # Higher for diversity
SUGGESTION_MAX_TOKENS = 150  # Compact output
SUGGESTION_MAX_COUNT = 3  # Maximum suggestions per response


# ==============================================================================
# CRAWLING CONFIGURATION
# ==============================================================================

# Amazon scraper settings
CRAWL_BATCH_SIZE = 100  # Number of products to crawl per category
CRAWL_DELAY_SECONDS = 2  # Delay between requests to avoid rate limiting
