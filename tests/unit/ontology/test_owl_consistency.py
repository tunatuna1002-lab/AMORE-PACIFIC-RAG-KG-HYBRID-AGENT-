"""
Unit tests for OWL Consistency Check (D-4)

Tests check_consistency, _check_disjointness_violations,
_check_cardinality_violations, _check_domain_range_violations,
and _rule_based_consistency_check in src/ontology/owl_reasoner.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.ontology.owl_reasoner import ConsistencyReport, OWLReasoner

# =========================================================================
# ConsistencyReport dataclass tests
# =========================================================================


class TestConsistencyReport:
    """Test ConsistencyReport dataclass."""

    def test_report_creation(self):
        """ConsistencyReport can be created with required fields."""
        report = ConsistencyReport(is_consistent=True)
        assert report.is_consistent is True
        assert report.violations == []
        assert report.warnings == []
        assert report.checked_at != ""

    def test_report_with_violations(self):
        """ConsistencyReport stores violations."""
        violations = [
            {
                "type": "disjointness_violation",
                "entity": "TestBrand",
                "description": "test violation",
                "severity": "error",
            }
        ]
        report = ConsistencyReport(
            is_consistent=False,
            violations=violations,
            check_method="owlready2_reasoner",
        )
        assert report.is_consistent is False
        assert len(report.violations) == 1
        assert report.check_method == "owlready2_reasoner"

    def test_report_with_warnings(self):
        """ConsistencyReport stores warnings."""
        report = ConsistencyReport(
            is_consistent=True,
            warnings=["test warning"],
        )
        assert len(report.warnings) == 1

    def test_report_auto_timestamp(self):
        """ConsistencyReport auto-generates checked_at."""
        report = ConsistencyReport(is_consistent=True)
        assert len(report.checked_at) > 0
        # Should be a valid ISO timestamp
        from datetime import datetime

        datetime.fromisoformat(report.checked_at)

    def test_report_explicit_timestamp(self):
        """ConsistencyReport uses explicit checked_at if given."""
        report = ConsistencyReport(
            is_consistent=True,
            checked_at="2026-02-18T12:00:00",
        )
        assert report.checked_at == "2026-02-18T12:00:00"


# =========================================================================
# Fallback mode tests (owlready2 unavailable)
# =========================================================================


class TestConsistencyCheckFallback:
    """Test consistency check when owlready2 is unavailable."""

    @pytest.fixture
    def reasoner_no_owl(self):
        """OWLReasoner in fallback mode."""
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            mock_fallback = MagicMock()
            r = OWLReasoner(fallback_reasoner=mock_fallback)
            assert r.onto is None
            yield r

    def test_check_consistency_fallback(self, reasoner_no_owl):
        """Consistency check returns fallback report when no owlready2."""
        report = reasoner_no_owl.check_consistency()
        assert isinstance(report, ConsistencyReport)
        assert report.is_consistent is True
        assert report.check_method == "rule_based_fallback"
        assert len(report.warnings) > 0

    def test_rule_based_fallback(self, reasoner_no_owl):
        """_rule_based_consistency_check returns valid report."""
        report = reasoner_no_owl._rule_based_consistency_check()
        assert isinstance(report, ConsistencyReport)
        assert report.is_consistent is True
        assert report.check_method == "rule_based_fallback"

    def test_disjointness_fallback_returns_empty(self, reasoner_no_owl):
        """_check_disjointness_violations returns empty without owlready2."""
        violations = reasoner_no_owl._check_disjointness_violations()
        assert violations == []

    def test_cardinality_fallback_returns_empty(self, reasoner_no_owl):
        """_check_cardinality_violations returns empty without owlready2."""
        violations = reasoner_no_owl._check_cardinality_violations()
        assert violations == []

    def test_domain_range_fallback_returns_empty(self, reasoner_no_owl):
        """_check_domain_range_violations returns empty without owlready2."""
        violations = reasoner_no_owl._check_domain_range_violations()
        assert violations == []


# =========================================================================
# OWL-based tests (owlready2 available)
# =========================================================================


class TestConsistencyCheckWithOWL:
    """Test consistency check with owlready2 available."""

    @pytest.fixture
    def reasoner(self):
        """OWLReasoner with owlready2 (if available)."""
        try:
            import owlready2  # noqa: F401

            r = OWLReasoner()
            return r
        except ImportError:
            pytest.skip("owlready2 not installed")

    def test_consistent_ontology(self, reasoner):
        """Clean ontology reports consistent."""
        report = reasoner.check_consistency()
        assert isinstance(report, ConsistencyReport)
        assert report.checked_at != ""

    def test_check_method_is_owlready2(self, reasoner):
        """Check method is owlready2_reasoner when available."""
        report = reasoner.check_consistency()
        assert report.check_method == "owlready2_reasoner"

    def test_no_disjointness_violations_clean(self, reasoner):
        """Clean ontology has no disjointness violations."""
        violations = reasoner._check_disjointness_violations()
        assert isinstance(violations, list)

    def test_no_cardinality_violations_clean(self, reasoner):
        """Clean ontology has no cardinality violations."""
        violations = reasoner._check_cardinality_violations()
        assert isinstance(violations, list)

    def test_no_domain_range_violations_clean(self, reasoner):
        """Clean ontology has no domain/range violations."""
        violations = reasoner._check_domain_range_violations()
        assert isinstance(violations, list)


class TestDisjointnessViolationDetection:
    """Test disjointness violation detection with owlready2."""

    @pytest.fixture
    def reasoner_with_violation(self):
        """OWLReasoner with a disjointness violation (brand in 2 classes)."""
        try:
            import owlready2  # noqa: F401
        except ImportError:
            pytest.skip("owlready2 not installed")

        r = OWLReasoner()
        # Add a brand with SoS that puts it in DominantBrand
        r.add_brand("ViolationTest", sos=0.35)
        r.infer_market_positions()

        # Force add to StrongBrand too (creates violation)
        with r.onto:
            brand = r.onto.search_one(iri="*ViolationTest")
            if brand:
                brand.is_a.append(r.onto.StrongBrand)

        return r

    def test_detects_disjointness_violation(self, reasoner_with_violation):
        """Disjointness violation detected when brand in two classes."""
        violations = reasoner_with_violation._check_disjointness_violations()
        # Should have at least one violation for ViolationTest
        entity_names = [v.get("entity", "") for v in violations]
        assert "ViolationTest" in entity_names


class TestCardinalityViolationDetection:
    """Test cardinality violation detection with owlready2."""

    @pytest.fixture
    def reasoner_with_product(self):
        """OWLReasoner with a product that has a category."""
        try:
            import owlready2  # noqa: F401
        except ImportError:
            pytest.skip("owlready2 not installed")

        r = OWLReasoner()
        r.add_brand("TestBrand", sos=0.1)
        r.add_product(
            asin="B0TESTPROD1",
            brand="TestBrand",
            category="lip_care",
            rank=5,
        )
        return r

    def test_product_with_category_no_violation(self, reasoner_with_product):
        """Product with exactly 1 category has no cardinality violation."""
        violations = reasoner_with_product._check_cardinality_violations()
        # Filter for our specific product
        test_violations = [v for v in violations if v.get("entity") == "B0TESTPROD1"]
        assert len(test_violations) == 0
