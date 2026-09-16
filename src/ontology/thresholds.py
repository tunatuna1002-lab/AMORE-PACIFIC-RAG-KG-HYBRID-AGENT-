"""
Thresholds - the single source of business thresholds (F4).

``config/thresholds.json`` is load-bearing: every rule in ``src/ontology/rules``, the OWL
restriction classes (``src/ontology/tbox.py``) and the alert judges (``AlertAgent``,
``MetricsAgent``) read their cut-offs from the module-level ``get_thresholds()`` instead of
literals. Defaults equal the values that used to be hard-coded so behaviour is unchanged
until the JSON is edited.

Unit contract: SoS / HHI are FRACTIONS (0-1); CPI is an index (100 = category average);
ranks are positions (1 = best).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "thresholds.json"


@dataclass(frozen=True)
class Thresholds:
    """Business thresholds. All SoS/HHI values are fractions."""

    # ----- market position (rules/market_rules.py, growth_rules.py) -----
    sos_dominance: float = 0.15  # market_dominance_fragmented: SoS >= 15%
    sos_major_player: float = 0.20  # major_player_concentrated: SoS >= 20%
    sos_challenger_min: float = 0.05  # challenger_position: 5% <= SoS < sos_dominance
    sos_niche_max: float = 0.03  # category_entry_opportunity: SoS < 3%
    sos_decline_pp: float = -0.02  # share_erosion: SoS change < -2%p
    hhi_fragmented: float = 0.15  # HHI < 0.15 -> fragmented market
    hhi_concentrated: float = 0.25  # HHI >= 0.25 -> concentrated market
    avg_rank_strong: float = 20  # avg_rank < 20 -> strong presence

    # ----- price / quality (rules/price_rules.py, alert_rules.py) -----
    cpi_value: float = 90  # CPI < 90 -> value position
    cpi_premium: float = 110  # CPI > 110 -> premium (price_quality_gap alert)
    cpi_luxury: float = 150  # CPI > 150 -> luxury premium
    price_premium_pct: float = 20  # price > category avg by 20%
    discount_overlap_ratio: float = 0.8
    discount_dependency_score: float = 61
    rating_gap_advantage: float = 0.05
    rating_trend_up: float = 0.05
    review_count_min: int = 100

    # ----- ranking (ranking.*) -----
    rank_drop: int = 5  # MetricsAgent._check_alerts: rank_change_1d >= 5
    rank_rise: int = 10
    alert_rank_change: int = 10  # AlertAgent.process_metrics: |rank_change| >= 10
    top_n: int = 10  # Top10Product / Top-10 entry
    top_n_leader: int = 3
    rank_stable_abs: int = 3
    rank_volatility_low: float = 3
    rank_volatility_high: float = 5
    streak_long_days: int = 30
    streak_short_days: int = 14
    churn_rate_high: float = 0.2
    competitor_count_many: int = 5
    competitor_count_some: int = 3
    trend_keywords_min: int = 2

    # ----- sentiment (rules/sentiment_rules.py) -----
    sentiment_cluster_min_mentions: int = 2
    sentiment_cluster_any: int = 1

    # ----- IR cross analysis (rules/ir_rules.py) -----
    ir_event_rank_surge: int = 10  # rank improved by 10+ during an IR event
    ir_consecutive_growth_quarters: int = 2
    ir_growth_slowdown_ratio: float = 0.5

    # ----- ontology (tbox.py restriction classes, materializer) -----
    owl_dominant_sos: float = 0.30  # DominantBrand: SoS >= 0.30
    owl_strong_sos: float = 0.15  # StrongBrand: 0.15 <= SoS < 0.30 ; NicheBrand: < 0.15
    new_entrant_days: int = 7  # NewEntrant: first_seen within 7 days of the snapshot

    def __post_init__(self) -> None:
        for name in (
            "sos_dominance",
            "sos_major_player",
            "sos_challenger_min",
            "sos_niche_max",
            "hhi_fragmented",
            "hhi_concentrated",
            "owl_dominant_sos",
            "owl_strong_sos",
        ):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"Thresholds.{name} must be a fraction in [0, 1], got {value!r} "
                    "(percent values are not accepted)"
                )


# JSON section/key -> Thresholds field. Only these keys are read; everything else in
# thresholds.json (category URLs, system config) is owned by other loaders.
_JSON_MAP: dict[str, tuple[str, str]] = {
    "sos_dominance": ("brand_health", "sos_dominance"),
    "sos_major_player": ("brand_health", "sos_major_player"),
    "sos_challenger_min": ("brand_health", "sos_challenger_min"),
    "sos_niche_max": ("brand_health", "sos_niche_max"),
    "sos_decline_pp": ("brand_health", "sos_decline_pp"),
    "hhi_fragmented": ("brand_health", "hhi_fragmented"),
    "hhi_concentrated": ("brand_health", "hhi_concentrated"),
    "avg_rank_strong": ("brand_health", "avg_rank_strong"),
    "cpi_value": ("price_quality", "cpi_value"),
    "cpi_premium": ("price_quality", "cpi_premium_alert"),
    "cpi_luxury": ("price_quality", "cpi_luxury"),
    "price_premium_pct": ("price_quality", "price_premium_pct"),
    "discount_overlap_ratio": ("price_quality", "discount_overlap_ratio"),
    "discount_dependency_score": ("price_quality", "discount_dependency_score"),
    "rating_gap_advantage": ("price_quality", "rating_gap_advantage"),
    "rating_trend_up": ("price_quality", "rating_trend_up"),
    "review_count_min": ("price_quality", "review_count_min"),
    "rank_drop": ("ranking", "significant_drop"),
    "rank_rise": ("ranking", "significant_rise"),
    "alert_rank_change": ("ranking", "alert_rank_change"),
    "top_n": ("ranking", "top_n"),
    "top_n_leader": ("ranking", "top_n_leader"),
    "rank_stable_abs": ("ranking", "rank_stable_abs"),
    "rank_volatility_low": ("ranking", "rank_volatility_low"),
    "rank_volatility_high": ("ranking", "rank_volatility_high"),
    "streak_long_days": ("streak", "monthly_highlight"),
    "streak_short_days": ("streak", "biweekly_highlight"),
    "churn_rate_high": ("monitoring", "churn_rate_high"),
    "competitor_count_many": ("competition", "competitor_count_many"),
    "competitor_count_some": ("competition", "competitor_count_some"),
    "trend_keywords_min": ("monitoring", "trend_keywords_min"),
    "sentiment_cluster_min_mentions": ("sentiment", "cluster_min_mentions"),
    "sentiment_cluster_any": ("sentiment", "cluster_any"),
    "ir_event_rank_surge": ("ir", "event_rank_surge"),
    "ir_consecutive_growth_quarters": ("ir", "consecutive_growth_quarters"),
    "ir_growth_slowdown_ratio": ("ir", "growth_slowdown_ratio"),
    "owl_dominant_sos": ("ontology", "owl_dominant_sos"),
    "owl_strong_sos": ("ontology", "owl_strong_sos"),
    "new_entrant_days": ("monitoring", "new_entrant_days"),
}

_FIELD_TYPES = {f.name: f.type for f in fields(Thresholds)}


def _coerce(name: str, value: Any) -> Any:
    kind = _FIELD_TYPES[name]
    if kind == "int":
        return int(value)
    return float(value)


def load_thresholds(path: str | Path | None = None) -> Thresholds:
    """Read ``config/thresholds.json`` (or ``path``) into a ``Thresholds``.

    Missing keys keep their defaults; a missing/unreadable file yields the defaults.
    Percent-scale SoS values raise ``ValueError`` (unit contract violation).
    """
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        logger.debug("thresholds file not found (%s); using defaults", cfg_path)
        return Thresholds()
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("thresholds file unreadable (%s): %s; using defaults", cfg_path, exc)
        return Thresholds()

    overrides: dict[str, Any] = {}
    for field_name, (section, key) in _JSON_MAP.items():
        section_data = data.get(section)
        if isinstance(section_data, dict) and key in section_data:
            overrides[field_name] = _coerce(field_name, section_data[key])
    return Thresholds(**overrides)


_current: Thresholds | None = None


def get_thresholds() -> Thresholds:
    """Process-wide thresholds (lazily loaded from config/thresholds.json)."""
    global _current
    if _current is None:
        _current = load_thresholds()
    return _current


def set_thresholds(thresholds: Thresholds | None) -> None:
    """Override the process-wide thresholds (tests / experiments). ``None`` -> reload lazily."""
    global _current
    _current = thresholds


def reset_thresholds() -> None:
    """Drop any override so the next ``get_thresholds()`` reloads from config."""
    set_thresholds(None)


def threshold(name: str) -> float:
    """Current value of one threshold field (resolved at call time)."""
    return getattr(get_thresholds(), name)


__all__ = [
    "Thresholds",
    "load_thresholds",
    "get_thresholds",
    "set_thresholds",
    "reset_thresholds",
    "threshold",
]
