"""
Dashboard JSON shape adapters
=============================
`DashboardExporter.export_dashboard_data()` (src/tools/exporters/dashboard_exporter.py)
writes ``data/dashboard_data.json`` with:

- ``products``   : dict keyed by ASIN (LANEIGE-only, ``name`` instead of ``title``, no ``brand``)
- ``categories`` : dict keyed by category id (``name``, ``sos``, ``best_rank``, ``cpi``, ...)
- ``brand``      : ``{"kpis": {...}, "competitors": [{"brand", "sos", "avg_rank", "product_count"}]}``
- ``metadata``   : ``total_products`` / ``laneige_products``

Several routes were written against an older, flatter shape (``products`` as a list
with ``brand``/``title``, top-level ``summary``, ``brand_metrics``, ``category_metrics``,
``category[*].top_products``, ``ai_insights``). The pure functions below accept *both*
shapes and return the flat form, so a route never has to know which one it loaded.

All adapters are side-effect free: they never mutate ``data``.
"""

from __future__ import annotations

from typing import Any

_LANEIGE = "LANEIGE"


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _rank_key(product: dict) -> float:
    rank = product.get("rank")
    try:
        return float(rank)
    except (TypeError, ValueError):
        return float("inf")


def products_as_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return products as a list sorted by rank, tolerant of the dict-keyed-by-ASIN shape.

    Exporter products (dict shape) get the legacy keys filled in: ``brand`` (the exporter
    only emits LANEIGE products), ``title``/``product_name`` (mirrors of ``name``) and
    ``asin`` (from the dict key). Legacy list products only get missing name aliases.
    """
    raw = _as_dict(data).get("products")
    items: list[dict[str, Any]] = []

    if isinstance(raw, dict):
        for asin, product in raw.items():
            if not isinstance(product, dict):
                continue
            p = dict(product)
            p.setdefault("asin", asin)
            p.setdefault("brand", _LANEIGE)
            items.append(p)
    elif isinstance(raw, list):
        items = [dict(p) for p in raw if isinstance(p, dict)]
    else:
        return []

    for p in items:
        name = p.get("name") or p.get("title") or p.get("product_name") or ""
        p.setdefault("name", name)
        p.setdefault("title", name)
        p.setdefault("product_name", name)

    return sorted(items, key=_rank_key)


def categories_as_dict(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return categories keyed by id (exporter ``categories`` dict, legacy ``category`` dict or list)."""
    data = _as_dict(data)
    raw = data.get("categories")
    if raw is None:
        raw = data.get("category")

    if isinstance(raw, dict):
        return {str(k): dict(v) if isinstance(v, dict) else {} for k, v in raw.items()}

    if isinstance(raw, list):
        out: dict[str, dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = item.get("category_id") or item.get("id") or item.get("name")
            if key:
                out[str(key)] = dict(item)
        return out

    return {}


def summary_from(data: dict[str, Any]) -> dict[str, Any]:
    """Top-level summary (``total_products``, ``categories_count``, ``laneige_products``, ``avg_price``)."""
    data = _as_dict(data)
    explicit = _as_dict(data.get("summary"))
    metadata = _as_dict(data.get("metadata"))
    kpis = _as_dict(_as_dict(data.get("brand")).get("kpis"))
    products = products_as_list(data)

    total_products = explicit.get("total_products", metadata.get("total_products"))
    if total_products is None:
        total_products = len(products)

    laneige_products = explicit.get("laneige_products", metadata.get("laneige_products"))
    if laneige_products is None:
        laneige_products = len([p for p in products if str(p.get("brand", "")).upper() == _LANEIGE])

    categories_count = explicit.get("categories_count")
    if categories_count is None:
        categories_count = len(categories_as_dict(data))

    avg_price = explicit.get("avg_price", kpis.get("avg_price"))
    if avg_price is None:
        prices = [float(p["price"]) for p in products if isinstance(p.get("price"), (int, float))]
        avg_price = round(sum(prices) / len(prices), 2) if prices else 0

    return {
        **explicit,
        "total_products": total_products or 0,
        "categories_count": categories_count or 0,
        "laneige_products": laneige_products or 0,
        "avg_price": avg_price or 0,
    }


def brand_metrics_from(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Brand rows (``brand``, ``sos``, ``avg_rank``, ``product_count``) from ``brand_metrics`` or ``brand.competitors``."""
    data = _as_dict(data)
    explicit = data.get("brand_metrics")
    if isinstance(explicit, list):
        return [dict(b) for b in explicit if isinstance(b, dict)]

    competitors = _as_dict(data.get("brand")).get("competitors")
    if not isinstance(competitors, list):
        return []
    return [
        {
            "brand": c.get("brand", "Unknown"),
            "sos": c.get("sos", 0),
            "avg_rank": c.get("avg_rank", 0),
            "product_count": c.get("product_count", 0),
        }
        for c in competitors
        if isinstance(c, dict)
    ]


def category_metrics_from(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Category rows for reports (``category``, ``total_products``, ``laneige_products``, ``avg_price``, ``hhi``, ``cpi``)."""
    data = _as_dict(data)
    explicit = data.get("category_metrics")
    if isinstance(explicit, list):
        return [dict(c) for c in explicit if isinstance(c, dict)]

    rows = []
    for cat_id, cat in categories_as_dict(data).items():
        rows.append(
            {
                "category_id": cat_id,
                "category": cat.get("name") or cat.get("category") or cat_id,
                "total_products": cat.get("product_count", cat.get("total_products", 0)) or 0,
                "laneige_products": cat.get("laneige_count", cat.get("laneige_products", 0)) or 0,
                "avg_price": cat.get("avg_price", 0) or 0,
                "hhi": cat.get("hhi", 0) or 0,
                "cpi": cat.get("cpi", 0) or 0,
                "sos": cat.get("sos", 0) or 0,
                "best_rank": cat.get("best_rank"),
            }
        )
    return rows


def category_top_products(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Products grouped by category id (legacy ``category[*].top_products`` or exporter products)."""
    data = _as_dict(data)
    legacy = data.get("category")
    if isinstance(legacy, dict) and any(
        isinstance(v, dict) and "top_products" in v for v in legacy.values()
    ):
        return {
            str(cat_id): [
                dict(p) for p in _as_dict(cat).get("top_products", []) if isinstance(p, dict)
            ]
            for cat_id, cat in legacy.items()
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for product in products_as_list(data):
        cat_id = product.get("category") or product.get("category_id") or ""
        grouped.setdefault(str(cat_id), []).append(product)
    return grouped


def ai_insights_from(data: dict[str, Any]) -> dict[str, Any]:
    """``ai_insights`` block; falls back to ``home.insight_message`` as a single strategic insight."""
    data = _as_dict(data)
    explicit = data.get("ai_insights")
    if isinstance(explicit, dict):
        return dict(explicit)

    message = _as_dict(data.get("home")).get("insight_message")
    if message:
        return {"strategic_insights": [{"title": "핵심 인사이트", "content": str(message)}]}
    return {}
