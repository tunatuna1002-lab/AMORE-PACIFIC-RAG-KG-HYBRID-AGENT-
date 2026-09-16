"""
Parsing helpers
===============
Small, dependency-free parsers shared by routes/services (single definition, F6-2).
"""

from __future__ import annotations

from typing import Any


def parse_price(value: Any) -> float | None:
    """
    Parse a price-like value ("$24.00", "1,299", 24, 24.0) into a float.

    Returns None for empty / non-numeric input instead of raising, so callers can
    write ``parse_price(v) or 0`` when they need a numeric default.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("$", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
