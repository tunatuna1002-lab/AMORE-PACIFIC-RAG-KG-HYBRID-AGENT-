"""
F6-2: "is this LANEIGE?" convergence on :mod:`src.domain.brand`.

Three groups of tests:

1. ``TestNoAdHocBrandLiterals`` - a source scan over ``src/**/*.py`` asserting that no
   module re-implements the target-brand check with a string literal. Deliberate
   exceptions live in :data:`ALLOWLIST` with a reason each.
2. ``TestConvergedSitesStillWork`` - behavioural tests through public entry points for
   the call sites that were converted, including the widening that
   ``is_target_brand`` (substring match) intentionally brings.
3. ``TestBusinessRulesShimGone`` - ``src.ontology.business_rules`` (compat shim) is
   deleted and ``src.ontology.rules`` is the real module.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest

from src.domain.brand import TARGET_BRAND, is_target_brand, normalize_brand, sql_like_pattern

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

# ---------------------------------------------------------------------------
# 1. Source scan
# ---------------------------------------------------------------------------

# Literal brand comparisons that must go through ``src.domain.brand.is_target_brand``:
#   ``== "LANEIGE"`` / ``== 'laneige'`` (covers ``.upper() ==`` / ``.lower() ==``)
#   ``"LANEIGE" ==``
#   ``"laneige" in x.lower()`` / ``"LANEIGE" in x.upper()``
_BRAND_LITERAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"""==\s*["']laneige["']""", re.IGNORECASE),
    re.compile(r"""["']laneige["']\s*==""", re.IGNORECASE),
    re.compile(r"""["']laneige["']\s+in\s""", re.IGNORECASE),
)

# ALLOWLIST: relative POSIX path -> reason it is not converged (yet).
# Every entry is a conscious decision, not an oversight.
ALLOWLIST: dict[str, str] = {
    # The single source of truth itself - it *defines* the literal.
    "src/domain/brand.py": "defines TARGET_BRAND / is_target_brand",
}


def _iter_src_python_files() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _offending_lines(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("#"):  # commentary, not a comparison
            continue
        if any(pattern.search(raw) for pattern in _BRAND_LITERAL_PATTERNS):
            hits.append((lineno, stripped))
    return hits


class TestNoAdHocBrandLiterals:
    def test_no_literal_brand_comparison_outside_domain_brand(self):
        """Every target-brand check funnels through src/domain/brand.py."""
        offenders: dict[str, list[tuple[int, str]]] = {}
        for path in _iter_src_python_files():
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            if rel in ALLOWLIST:
                continue
            hits = _offending_lines(path)
            if hits:
                offenders[rel] = hits

        assert not offenders, (
            "Ad-hoc LANEIGE literal comparisons found. Use "
            "src.domain.brand.is_target_brand(), or add an ALLOWLIST entry with a reason:\n"
            + "\n".join(
                f"  {rel}:{lineno}: {text}"
                for rel, hits in offenders.items()
                for lineno, text in hits
            )
        )

    def test_allowlist_entries_all_exist_and_still_offend(self):
        """The allowlist may not rot: every entry must exist and still contain a hit."""
        for rel, reason in ALLOWLIST.items():
            path = PROJECT_ROOT / rel
            assert path.is_file(), f"stale ALLOWLIST entry (file gone): {rel} ({reason})"
            assert _offending_lines(path), (
                f"stale ALLOWLIST entry (no literal left): {rel} ({reason}) - remove it"
            )

    def test_converged_modules_import_the_domain_helper(self):
        """The files converted by F6-2 import the single helper."""
        converged = [
            "src/agents/metrics_agent.py",
            # CHANGED (Phase 4): the alerts route's brand logic moved into
            # src/application/services/alert_service.py, so the helper is imported there now.
            "src/application/services/alert_service.py",
            "src/core/brain.py",
            "src/tools/intelligence/morning_brief.py",
            "src/tools/exporters/dashboard_exporter.py",
            "src/tools/notifications/email_sender.py",
            "src/ontology/reasoner.py",
            "src/ontology/rules/growth_rules.py",
            "src/ontology/inference_context.py",
        ]
        for rel in converged:
            text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
            assert "from src.domain.brand import" in text, rel
            assert "is_target_brand" in text, rel


# ---------------------------------------------------------------------------
# 2. Behaviour of the converged sites
# ---------------------------------------------------------------------------


class TestDomainBrandHelper:
    def test_substring_match_is_the_intended_widening(self):
        assert is_target_brand("LANEIGE")
        assert is_target_brand("laneige")
        assert is_target_brand("  LANEIGE  ")
        assert is_target_brand("LANEIGE Official")
        assert is_target_brand("LANEIGE (라네즈)")
        assert not is_target_brand("COSRX")
        assert not is_target_brand("")
        assert not is_target_brand(None)

    def test_normalize_and_sql_pattern(self):
        assert normalize_brand("  LaNeIgE ") == "laneige"
        assert normalize_brand(None) == ""
        assert sql_like_pattern() == "%laneige%"
        assert TARGET_BRAND == "LANEIGE"


@pytest.fixture
def thresholds_config_file() -> str:
    config = {
        "ranking": {"top_n_tiers": [3, 5, 10, 20, 50, 100], "significant_drop": 5},
        "brand_health": {"sos_change_up": 1.0, "sos_change_down": -1.0},
        "thresholds": {"significant_rank_drop": 5},
        "categories": {},
        "target_brands": ["LANEIGE"],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        return f.name


def _crawl_payload(brands: list[str]) -> dict:
    return {
        "categories": {
            "lip_care": {
                "rank_records": [
                    {
                        "rank": i + 1,
                        "brand": brand,
                        "title": f"{brand} Lip Product",
                        "product_asin": f"B0000000{i}",
                        "price": 20.0 + i,
                        "rating": 4.5,
                        "review_count": 1000,
                    }
                    for i, brand in enumerate(brands)
                ]
            }
        }
    }


class TestConvergedSitesStillWork:
    async def test_metrics_agent_marks_target_brand_and_its_decorated_spellings(
        self, thresholds_config_file
    ):
        """MetricsAgent.execute: brand_metrics[].is_laneige via is_target_brand."""
        from src.agents.metrics_agent import MetricsAgent

        agent = MetricsAgent(config_path=thresholds_config_file)
        result = await agent.execute(_crawl_payload(["LANEIGE", "LANEIGE Official", "COSRX"]))

        flags = {bm["brand_name"]: bm["is_laneige"] for bm in result["brand_metrics"]}
        assert flags["LANEIGE"] is True
        # Widened by F6: decorated spellings now count as the target brand.
        assert flags["LANEIGE Official"] is True
        assert flags["COSRX"] is False

    async def test_metrics_agent_product_metrics_cover_decorated_spellings(
        self, thresholds_config_file
    ):
        """_is_laneige gates product-level metrics; it must see 'LANEIGE Official' too."""
        from src.agents.metrics_agent import MetricsAgent

        agent = MetricsAgent(config_path=thresholds_config_file)
        result = await agent.execute(_crawl_payload(["LANEIGE Official", "COSRX"]))

        asins = {pm.get("asin") or pm.get("product_asin") for pm in result["product_metrics"]}
        assert len(result["product_metrics"]) == 1, result["product_metrics"]
        assert "B00000000" in asins

    async def test_metrics_agent_matches_on_title_when_brand_is_missing(
        self, thresholds_config_file
    ):
        """_is_laneige still checks the title (unchanged semantics)."""
        from src.agents.metrics_agent import MetricsAgent

        agent = MetricsAgent(config_path=thresholds_config_file)
        payload = _crawl_payload(["Unknown"])
        payload["categories"]["lip_care"]["rank_records"][0]["title"] = "Laneige Lip Sleeping Mask"
        result = await agent.execute(payload)
        assert len(result["product_metrics"]) == 1

    async def test_morning_brief_counts_target_brand_products(self):
        from src.tools.intelligence.morning_brief import MorningBriefData, MorningBriefGenerator

        gen = MorningBriefGenerator()
        brief = MorningBriefData(date="2026-09-16", day_of_week="Wednesday")
        crawl = {
            "category": "lip_care",
            "products": [
                {"rank": 1, "brand": "LANEIGE"},
                {"rank": 2, "brand": "laneige official"},
                {"rank": 3, "brand": "COSRX"},
            ],
        }
        await gen._analyze_crawl_data(brief, crawl)

        assert len(brief.laneige_products) == 2
        assert brief.laneige_top10_count == 2
        assert brief.category_stats["lip_care"]["laneige_count"] == 2
        assert brief.category_stats["lip_care"]["laneige_best_rank"] == 1

    async def test_morning_brief_rank_change_uses_the_same_check(self):
        from src.tools.intelligence.morning_brief import MorningBriefData, MorningBriefGenerator

        gen = MorningBriefGenerator()
        brief = MorningBriefData(date="2026-09-16", day_of_week="Wednesday")
        crawl = {"category": "lip_care", "products": [{"rank": 2, "brand": "LANEIGE"}]}
        previous = {"products": [{"rank": 5, "brand": "LANEIGE Official"}]}

        await gen._analyze_crawl_data(brief, crawl, previous)
        assert brief.laneige_rank_change == pytest.approx(3.0)  # 5 -> 2, 상승

    def test_reasoner_standard_condition_uses_the_domain_helper(self):
        from src.ontology.reasoner import StandardConditions

        cond = StandardConditions.is_target_brand()
        assert cond.evaluate({"brand": "LANEIGE"}) is True
        assert cond.evaluate({"brand": "laneige"}) is True
        assert cond.evaluate({"brand": "LANEIGE Official"}) is True  # widened by F6
        assert cond.evaluate({"is_target": True}) is True
        assert cond.evaluate({"brand": "COSRX"}) is False
        assert cond.evaluate({}) is False

    def test_growth_rule_target_brand_condition(self):
        from src.ontology.rules.growth_rules import RULE_TOP3_ACHIEVEMENT

        cond = next(c for c in RULE_TOP3_ACHIEVEMENT.conditions if c.name == "is_target_brand")
        assert cond.evaluate({"brand": "LANEIGE"}) is True
        assert cond.evaluate({"brand": "LANEIGE Official"}) is True  # widened by F6
        assert cond.evaluate({"brand": "COSRX"}) is False

    def test_inference_context_is_target_flag(self):
        from src.ontology.inference_context import build_inference_context

        assert build_inference_context({}, brand="LANEIGE")["is_target"] is True
        assert build_inference_context({}, brand="LANEIGE Official")["is_target"] is True
        assert build_inference_context({}, brand="COSRX")["is_target"] is False

    def test_inference_context_brand_lookup_stays_exact_for_other_brands(self):
        """
        D24 guard: the *requested* brand is still matched exactly.

        Widening this to a substring match would let a request for "Beauty" pick up
        "Beauty of Joseon" rows, so ``_brand_matches`` keeps normalised equality for
        non-target brands and only routes the target-brand identity check through
        ``is_target_brand``.
        """
        from src.ontology.inference_context import build_inference_context

        metrics = {
            "brand_metrics": [
                {"brand_name": "Beauty of Joseon", "share_of_shelf": 12.0, "avg_rank": 8}
            ]
        }
        ctx = build_inference_context(metrics, brand="Beauty")
        assert "sos" not in ctx, ctx

        ctx_exact = build_inference_context(metrics, brand="beauty of joseon")
        assert ctx_exact["sos"] == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# 3. business_rules shim is gone
# ---------------------------------------------------------------------------


class TestBusinessRulesShimGone:
    def test_business_rules_module_no_longer_exists(self):
        assert not (SRC_ROOT / "ontology" / "business_rules.py").exists()
        with pytest.raises(ModuleNotFoundError):
            import src.ontology.business_rules  # noqa: F401

    def test_rules_package_is_the_real_module(self):
        from src.ontology.rules import ALL_BUSINESS_RULES, register_all_rules

        assert len(ALL_BUSINESS_RULES) > 0

        from src.ontology.reasoner import OntologyReasoner

        reasoner = OntologyReasoner()
        registered = register_all_rules(reasoner)
        assert registered == len(ALL_BUSINESS_RULES)
        assert len(reasoner.rules) >= registered

    def test_ontology_package_lazy_table_resolves_register_all_rules(self):
        import src.ontology as ontology
        from src.ontology.rules import register_all_rules

        assert ontology.register_all_rules is register_all_rules
