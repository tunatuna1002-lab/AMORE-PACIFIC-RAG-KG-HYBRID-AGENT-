"""
Cleanup - single source of truth for shared helpers and Pydantic models
=======================================================================
alerts.py carried its own copy of the JWT helpers and get_app_state_manager;
export.py and market_intelligence.py re-declared models that live in src/api/models.py.
"""

from __future__ import annotations

import inspect


def test_alerts_uses_shared_jwt_and_state_manager_helpers():
    from src.api import dependencies
    from src.api.routes import alerts

    assert alerts.create_email_verification_token is dependencies.create_email_verification_token
    assert alerts.verify_jwt_email_token is dependencies.verify_jwt_email_token
    assert alerts.get_app_state_manager is dependencies.get_app_state_manager

    src = inspect.getsource(alerts)
    assert "def create_email_verification_token" not in src
    assert "def verify_jwt_email_token" not in src
    assert "def get_app_state_manager" not in src
    assert "_state_manager_singleton" not in src


def test_export_models_come_from_api_models():
    from src.api import models
    from src.api.routes import export

    assert export.ExportRequest is models.ExportRequest
    assert export.AnalystReportRequest is models.AnalystReportRequest
    assert export.AsyncExportRequest is models.AsyncExportRequest
    assert models.ExportRequest().include_external_signals is True
    assert models.ExportRequest(include_external_signals=False).include_external_signals is False
    assert "class ExportRequest" not in inspect.getsource(export)
    assert "class AnalystReportRequest" not in inspect.getsource(export)


def test_market_intelligence_models_come_from_api_models():
    from src.api import models
    from src.api.routes import market_intelligence as mi

    assert mi.MarketIntelligenceStatusResponse is models.MarketIntelligenceStatusResponse
    # LayerDataResponse was declared in the route module but never used there; it now
    # lives only in src/api/models.py.
    assert hasattr(models, "LayerDataResponse")
    assert "class MarketIntelligenceStatusResponse" not in inspect.getsource(mi)
    assert "class LayerDataResponse" not in inspect.getsource(mi)
