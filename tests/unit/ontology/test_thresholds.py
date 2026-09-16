"""F4-1: thresholds.json is load-bearing.

- ``Thresholds`` defaults equal today's hard-coded values.
- ``load_thresholds`` reads config/thresholds.json (and a tmp file).
- ``set_thresholds`` changes which rules fire without re-importing the rule modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ontology.thresholds import (
    Thresholds,
    get_thresholds,
    load_thresholds,
    reset_thresholds,
    set_thresholds,
)


@pytest.fixture(autouse=True)
def _restore_thresholds():
    yield
    reset_thresholds()


def test_defaults_match_hardcoded_values() -> None:
    t = Thresholds()
    assert (t.sos_dominance, t.sos_major_player, t.sos_challenger_min, t.sos_niche_max) == (
        0.15,
        0.20,
        0.05,
        0.03,
    )
    assert (t.hhi_fragmented, t.hhi_concentrated) == (0.15, 0.25)
    assert (t.cpi_value, t.cpi_premium, t.cpi_luxury) == (90, 110, 150)
    assert (t.rank_drop, t.rank_rise) == (5, 10)
    assert t.alert_rank_change == 10
    assert (t.owl_dominant_sos, t.owl_strong_sos) == (0.30, 0.15)
    assert t.new_entrant_days == 7
    assert t.top_n == 10


def test_thresholds_is_frozen() -> None:
    t = Thresholds()
    with pytest.raises(Exception):
        t.rank_drop = 99  # type: ignore[misc]


def test_load_from_repo_config_matches_defaults() -> None:
    loaded = load_thresholds()
    assert loaded == Thresholds()


def test_load_from_tmp_file_overrides(tmp_path: Path) -> None:
    cfg = {
        "ranking": {"significant_drop": 7, "significant_rise": 12, "alert_rank_change": 7},
        "brand_health": {"hhi_concentrated": 0.4, "sos_dominance": 0.25},
        "price_quality": {"cpi_premium_alert": 120},
    }
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    t = load_thresholds(path)
    assert t.rank_drop == 7
    assert t.rank_rise == 12
    assert t.alert_rank_change == 7
    assert t.hhi_concentrated == 0.4
    assert t.sos_dominance == 0.25
    assert t.cpi_premium == 120
    # untouched keys keep defaults
    assert t.sos_major_player == 0.20


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    assert load_thresholds(tmp_path / "nope.json") == Thresholds()


def test_load_rejects_percent_scale_sos(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"brand_health": {"sos_dominance": 15}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_thresholds(path)


def test_set_and_get_thresholds_roundtrip() -> None:
    custom = Thresholds(rank_drop=7)
    set_thresholds(custom)
    assert get_thresholds() is custom
    reset_thresholds()
    assert get_thresholds() == Thresholds()


def test_set_thresholds_changes_rule_firing() -> None:
    """sos_dominance 0.25 → ``{sos: 0.18}`` no longer fires market_dominance_fragmented."""
    from src.ontology.knowledge_graph import KnowledgeGraph
    from src.ontology.reasoner import OntologyReasoner
    from src.ontology.rules import register_all_rules

    kg = KnowledgeGraph(persist_path=None, auto_load=False, auto_save=False)
    reasoner = OntologyReasoner(kg)
    register_all_rules(reasoner)
    ctx = {"brand": "LANEIGE", "is_target": True, "sos": 0.18, "hhi": 0.10, "competitor_count": 6}

    fired = {r.rule_name for r in reasoner.infer(ctx)}
    assert "market_dominance_fragmented" in fired

    set_thresholds(Thresholds(sos_dominance=0.25))
    fired = {r.rule_name for r in reasoner.infer(ctx)}
    assert "market_dominance_fragmented" not in fired


def test_no_literal_thresholds_left_in_rules() -> None:
    """Rule condition lambdas must read thresholds from ``get_thresholds()``."""
    import re

    rules_dir = Path(__file__).resolve().parents[3] / "src" / "ontology" / "rules"
    offenders: list[str] = []
    pattern = re.compile(r"(StandardConditions\.\w+\(\s*-?\d|check=lambda ctx:.*[<>]=?\s*-?\d)")
    for file in rules_dir.glob("*.py"):
        for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            line = line.split("#", 1)[0]  # ignore trailing comments
            if pattern.search(line):
                # comparisons against 0 (sign checks) are not thresholds
                if re.search(r"[<>]=?\s*0(?!\.)\b", line) and not re.search(
                    r"[<>]=?\s*-?\d*\.\d|[<>]=?\s*-?[1-9]", line
                ):
                    continue
                offenders.append(f"{file.name}:{lineno}: {stripped}")
    assert offenders == []


def test_standard_conditions_accept_threshold_keys() -> None:
    from src.ontology.reasoner import StandardConditions

    cond = StandardConditions.hhi_below("hhi_fragmented")
    assert cond.name == "hhi_below_0.15"  # name pinned by chatbot characterization
    assert cond.check({"hhi": 0.10}) is True
    set_thresholds(Thresholds(hhi_fragmented=0.05))
    assert cond.check({"hhi": 0.10}) is False
