"""
Inference context builder - the ONE place that turns metric payloads into rule input (F4-2).

Consumers (``HybridRetriever._build_inference_context`` and
``DashboardExporter._build_inference_context``) are thin wrappers around
``build_inference_context``; they only add source-specific lookups (KG competitors,
sentiment, dashboard action items).

Contract:
- Input SoS values (``summary.laneige_sos_by_category``, ``brand_metrics[].share_of_shelf``)
  are PERCENT (MetricCalculator contract) -> the context carries FRACTIONS
  (``src.shared.units.percent_to_fraction`` is the only conversion point). The unit is never
  guessed from the magnitude.
- No fabricated defaults: a metric that the payload does not carry is absent from the
  context (rules treat absent keys via their own ``ctx_num`` defaults). Only ``has_rank_shock``
  and ``alert_count`` are always present because they are derived from the alerts list.
- Sentiment clusters are normalised to ``{cluster: count}`` (``normalize_sentiment_clusters``).
"""

from __future__ import annotations

from typing import Any

from src.shared.units import percent_to_fraction

TARGET_BRAND = "laneige"

_MARKET_KEYS: tuple[tuple[str, str], ...] = (
    ("hhi", "hhi"),
    ("cpi", "cpi"),
    ("churn_rate", "churn_rate_7d"),
    ("rating_gap", "avg_rating_gap"),
)
_PRODUCT_KEYS: tuple[str, ...] = (
    "current_rank",
    "rank_change_1d",
    "rank_change_7d",
    "rank_volatility",
    "streak_days",
    "asin",
)


def normalize_sentiment_clusters(clusters: Any) -> dict[str, Any]:
    """Normalise sentiment clusters to the dict shape the rules read.

    - dict (``{"Hydration": 2}`` or ``{"Hydration": [...]}``) is copied as-is
    - list/tuple/set (``["Hydration", "Sensory"]``) becomes ``{cluster: count}``
    - None / empty / anything else -> ``{}``
    """
    if not clusters:
        return {}
    if isinstance(clusters, dict):
        return dict(clusters)
    if isinstance(clusters, (list, tuple, set)):
        counts: dict[str, int] = {}
        for cluster in clusters:
            counts[cluster] = counts.get(cluster, 0) + 1
        return counts
    return {}


def _brand_matches(bm: dict[str, Any], brand: str | None) -> bool:
    if brand is None:
        return bool(bm.get("is_laneige"))
    name = str(bm.get("brand_name", "")).lower()
    return name == brand.lower() or (brand.lower() == TARGET_BRAND and bool(bm.get("is_laneige")))


def _pick_product(products: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not products:
        return None
    return min(products, key=lambda p: p.get("current_rank") or 100)


def build_inference_context(
    metrics: dict[str, Any],
    brand: str | None = None,
    *,
    products: list[dict[str, Any]] | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Build the rule-input context from a MetricsAgent-shaped payload.

    Args:
        metrics: ``{"summary", "brand_metrics", "market_metrics", "product_metrics", "alerts"}``
            (all keys optional). SoS values are PERCENT.
        brand: brand the question/insight is about (``None`` -> target brand via ``is_laneige``).
        products: overrides ``metrics["product_metrics"]`` (list of product metric dicts).
        category: category the question is scoped to (selects SoS / market metrics).

    Returns:
        Context dict with FRACTION SoS and no fabricated defaults.
    """
    metrics = metrics or {}
    context: dict[str, Any] = {}

    if brand:
        context["brand"] = brand
        context["is_target"] = brand.lower() == TARGET_BRAND
    if category:
        context["category"] = category

    # --- SoS from summary (percent -> fraction) ---
    summary = metrics.get("summary") or {}
    sos_by_category = summary.get("laneige_sos_by_category") or {}
    if category and category in sos_by_category:
        context["sos"] = percent_to_fraction(sos_by_category[category])
    elif sos_by_category:
        context["sos"] = percent_to_fraction(next(iter(sos_by_category.values())))

    # --- brand metrics (percent -> fraction) ---
    for bm in metrics.get("brand_metrics") or []:
        if not _brand_matches(bm, brand):
            continue
        if category and bm.get("category_id") not in (None, category):
            continue
        if "share_of_shelf" in bm and bm["share_of_shelf"] is not None:
            context["sos"] = percent_to_fraction(bm["share_of_shelf"])
        if bm.get("avg_rank") is not None:
            context["avg_rank"] = bm["avg_rank"]
        if bm.get("product_count") is not None:
            context["product_count"] = bm["product_count"]
        break

    # --- market metrics: only keys that are actually present ---
    for mm in metrics.get("market_metrics") or []:
        if category and mm.get("category_id") != category:
            continue
        for ctx_key, src_key in _MARKET_KEYS:
            if mm.get(src_key) is not None:
                context[ctx_key] = mm[src_key]
        break

    # --- product metrics: best-ranked product ---
    best = _pick_product(products if products is not None else metrics.get("product_metrics") or [])
    if best is not None:
        for key in _PRODUCT_KEYS:
            if best.get(key) is not None:
                context[key] = best[key]

    # --- alerts ---
    alerts = metrics.get("alerts") or []
    context["has_rank_shock"] = any(a.get("type") == "rank_shock" for a in alerts)
    context["alert_count"] = len(alerts)

    return context


__all__ = ["build_inference_context", "normalize_sentiment_clusters", "TARGET_BRAND"]
