"""
Brand helpers (domain)
======================
Single definition of "is this the brand we monitor?" (F6-2). The API routes used
six ad-hoc variants (``== "LANEIGE"``, ``.upper() == ...``, ``"laneige" in x.lower()``,
a three-spelling loop, ...); they all funnel through :func:`is_target_brand` now.

No external dependencies - this module stays importable from every layer.
"""

from __future__ import annotations

from typing import Any

TARGET_BRAND = "LANEIGE"


def normalize_brand(name: Any) -> str:
    """Case-folded, whitespace-trimmed brand name ("" for None/non-strings)."""
    if name is None:
        return ""
    return str(name).strip().casefold()


def is_target_brand(name: Any, target: str = TARGET_BRAND) -> bool:
    """
    True when ``name`` denotes the target brand.

    Matching is case-insensitive and tolerant of decorations ("LANEIGE (라네즈)",
    "Laneige Official"): the normalized target must appear as a whole token/substring
    of the normalized name. Empty or None names never match.
    """
    needle = normalize_brand(target)
    haystack = normalize_brand(name)
    if not needle or not haystack:
        return False
    return needle in haystack


def sql_like_pattern(target: str = TARGET_BRAND) -> str:
    """``LIKE`` pattern for the same match in SQL (``LOWER(col) LIKE ?``)."""
    return f"%{normalize_brand(target)}%"
