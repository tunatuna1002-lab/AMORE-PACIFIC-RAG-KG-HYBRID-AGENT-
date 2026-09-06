"""
Unit conversion helpers - the single percent <-> fraction boundary.

Contract:
- ``MetricCalculator`` / ``brand_metrics[].share_of_shelf`` / dashboard KPIs are PERCENT (0-100).
- Ontology, Knowledge Graph metadata and business rules use FRACTION (0-1).
- Every crossing between the two domains goes through this module. The helpers never
  guess the unit from the magnitude of the value: ``percent_to_fraction(0.5)`` is 0.005.
"""

from __future__ import annotations


def percent_to_fraction(value: float | None) -> float | None:
    """Convert a percent value (3.0 == 3%) into a fraction (0.03). ``None`` passes through."""
    if value is None:
        return None
    return float(value) / 100.0


def fraction_to_percent(value: float | None) -> float | None:
    """Convert a fraction (0.03) into a percent value (3.0). ``None`` passes through."""
    if value is None:
        return None
    return float(value) * 100.0


__all__ = ["percent_to_fraction", "fraction_to_percent"]
