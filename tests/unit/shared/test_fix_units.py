"""Unit contract helpers (bug D2): single conversion point between percent and fraction."""

import pytest

from src.shared.units import fraction_to_percent, percent_to_fraction


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3.0, 0.03), (100.0, 1.0), (0.0, 0.0), (12, 0.12)],
)
def test_percent_to_fraction(value, expected):
    assert percent_to_fraction(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.03, 3.0), (1.0, 100.0), (0.0, 0.0)],
)
def test_fraction_to_percent(value, expected):
    assert fraction_to_percent(value) == pytest.approx(expected)


def test_none_passthrough():
    assert percent_to_fraction(None) is None
    assert fraction_to_percent(None) is None


def test_no_heuristics_small_percent_is_not_left_alone():
    # 0.5% is 0.005 as a fraction - the helper must never "guess" the unit.
    assert percent_to_fraction(0.5) == pytest.approx(0.005)
    assert fraction_to_percent(0.5) == pytest.approx(50.0)
