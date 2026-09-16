"""
Category names / metadata
=========================
``config/category_hierarchy.json`` is the single source of category ids, names,
levels and parents (F6-2). Routes used to carry literal copies of this map.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HIERARCHY_PATH = Path(__file__).resolve().parents[3] / "config" / "category_hierarchy.json"


@lru_cache(maxsize=4)
def load_category_hierarchy(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """``categories`` block of category_hierarchy.json ({} when missing/corrupt)."""
    hierarchy_path = Path(path) if path else DEFAULT_HIERARCHY_PATH
    try:
        with open(hierarchy_path, encoding="utf-8") as f:
            return json.load(f).get("categories", {})
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"category_hierarchy.json unavailable ({hierarchy_path}): {e}")
        return {}


def category_name(category_id: str, path: str | Path | None = None) -> str:
    """Display name for a category id (the id itself when unknown)."""
    return load_category_hierarchy(path).get(category_id, {}).get("name", category_id)


def monitored_category_meta(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Metadata for the monitored categories, in hierarchy (depth-first) order.

    Each entry: ``{"name", "level", "parent_id", "indent", "order"}`` where
    ``order`` is the DFS position among monitored nodes and ``indent`` is the
    display depth (a level whose parent is *not* monitored is collapsed by one).
    """
    categories = load_category_hierarchy(path)
    meta: dict[str, dict[str, Any]] = {}

    def visit(cat_id: str) -> None:
        cat = categories.get(cat_id)
        if cat is None:
            return
        if cat.get("is_monitored"):
            parent_id = cat.get("parent_id")
            parent_monitored = bool(categories.get(parent_id, {}).get("is_monitored"))
            level = int(cat.get("level", 0))
            indent = level if (parent_id is None or parent_monitored) else max(level - 1, 0)
            meta[cat_id] = {
                "name": cat.get("name", cat_id),
                "level": level,
                "parent_id": parent_id,
                "indent": indent,
                "order": len(meta),
            }
        for child in cat.get("children", []):
            visit(child)

    roots = [cid for cid, c in categories.items() if c.get("parent_id") is None]
    for root in roots:
        visit(root)
    return meta
