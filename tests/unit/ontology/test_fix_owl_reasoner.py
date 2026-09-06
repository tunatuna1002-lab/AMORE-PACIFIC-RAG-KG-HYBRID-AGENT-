"""Bug D2(c): OWLReasoner.add_brand receives fractions; a percent-scale value must fail loudly."""

from unittest.mock import MagicMock

import pytest

from src.ontology.owl_reasoner import OWLReasoner


def test_add_brand_rejects_percent_scale_sos():
    reasoner = OWLReasoner(fallback_reasoner=MagicMock())
    with pytest.raises(ValueError):
        reasoner.add_brand("LANEIGE", sos=3.0)


def test_add_brand_accepts_fraction_boundary():
    reasoner = OWLReasoner(fallback_reasoner=MagicMock())
    # 1.0 (100%) is the upper bound of the fraction domain and must not raise.
    reasoner.add_brand("LANEIGE", sos=1.0)
    reasoner.add_brand("COSRX", sos=0.0)
