"""
Date range resolution
=====================
Single definition of the "default the missing start/end date" logic that the
analytics/export routes each carried a copy of (F6-2).

The crawler stamps ``snapshot_date`` in KST, so "today" is KST today.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from src.shared.constants import KST

DATE_FMT = "%Y-%m-%d"


def today_kst() -> date:
    """KST 'today' (the crawler's snapshot_date calendar)."""
    return datetime.now(KST).date()


def _coerce_today(today: date | datetime | str | None) -> date:
    if today is None:
        return today_kst()
    if isinstance(today, datetime):
        return today.date()
    if isinstance(today, date):
        return today
    return datetime.strptime(today, DATE_FMT).date()


def resolve_date_range(
    start: str | None,
    end: str | None,
    default_days: int = 30,
    today: date | datetime | str | None = None,
) -> tuple[str, str]:
    """
    Fill in missing ``start``/``end`` (YYYY-MM-DD strings).

    - ``end`` missing → KST today (or ``today`` when given)
    - ``start`` missing → ``end - default_days``
    - explicit values are returned untouched (no reordering / validation here)
    """
    if not end:
        end = _coerce_today(today).strftime(DATE_FMT)
    if not start:
        end_date = datetime.strptime(end, DATE_FMT).date()
        start = (end_date - timedelta(days=default_days)).strftime(DATE_FMT)
    return start, end


def days_in_range(start: str, end: str, fallback: int = 7) -> int:
    """Inclusive day count of a YYYY-MM-DD range; ``fallback`` on unparsable input."""
    try:
        start_dt = datetime.strptime(start, DATE_FMT)
        end_dt = datetime.strptime(end, DATE_FMT)
    except (TypeError, ValueError):
        return fallback
    return (end_dt - start_dt).days + 1
