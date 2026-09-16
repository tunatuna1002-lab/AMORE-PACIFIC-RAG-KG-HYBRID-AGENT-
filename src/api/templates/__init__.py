"""
HTML templates served directly by API routes.
==============================================

These are the few standalone pages the backend returns as ``HTMLResponse``
(email confirmation landing pages). The dashboard SPA lives in ``dashboard/``
and is not rendered here.

Templates use :class:`string.Template` (``$name``) rather than ``str.format``
so that the CSS in the file keeps its literal single braces and the file stays
valid, editable HTML.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from string import Template

_TEMPLATE_DIR = Path(__file__).resolve().parent

__all__ = ["render"]


@cache
def _load(name: str) -> Template:
    """Read a template once and cache the parsed result."""
    return Template((_TEMPLATE_DIR / f"{name}.html").read_text(encoding="utf-8"))


def render(name: str, /, **values: object) -> str:
    """
    Render ``<name>.html`` with ``$placeholder`` substitution.

    Uses ``safe_substitute`` so a stray ``$`` in interpolated content (or a
    placeholder a caller forgot) renders literally instead of raising.

    >>> render("email_confirm_success", email="a@b.c")  # doctest: +SKIP
    '<!DOCTYPE html>...'
    """
    return _load(name).safe_substitute(**values)
