"""
API characterization fixtures
=============================
Fixtures for pinning the public HTTP surface of `src.api.dashboard_api:app`.
Imported explicitly by the API characterization test modules (not via conftest)
so the module-level environment setup below runs before `src.api.dependencies`
is imported for the first time. Nothing under src/ is modified.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

# --- Process-level environment (must happen BEFORE src.api.dependencies is imported) ---
# `src.api.dependencies.verify_api_key` reads API_KEY from the environment at call time
# (the module-level constant is only a production startup guard). The characterization
# suite pins the *no-API_KEY* (development) behaviour of verify_api_key, so make sure a
# shell-exported API_KEY cannot leak in. .env.test (loaded by the root conftest) does not
# define API_KEY either. Tests that need a configured key use the `configured_api_key` /
# `auth_headers` fixtures below (monkeypatch.setenv).
os.environ.pop("API_KEY", None)
os.environ["AUTO_START_SCHEDULER"] = "false"
os.environ.setdefault(
    "KG_PERSIST_PATH",
    os.path.join(tempfile.gettempdir(), "amore_test_kg", "knowledge_graph.json"),
)
os.environ.pop("RAILWAY_ENVIRONMENT", None)
# The vector retriever gates itself on OPENAI_API_KEY (src/rag/retriever.py:_check_vector_search)
# and caches the verdict in a module global. Without the key, UnifiedBrain.initialize()
# fails fast and deterministically (no ChromaDB / embedding downloads, no network) — the
# chat characterization tests pin that offline fallback.
os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture(scope="session")
def app():
    """The real FastAPI app from the main entry point (routers, middleware, handlers)."""
    from src.api.dashboard_api import app as _app

    return _app


@pytest.fixture
def client(app):
    """
    TestClient WITHOUT the context manager on purpose: entering `with TestClient(app)`
    would run the lifespan, which schedules a background Playwright crawl when no
    "today" data exists. Routes lazily initialise everything they need, so startup is
    not required to exercise them.

    base_url must be http://localhost — TrustedHostMiddleware rejects "testserver".
    """
    from fastapi.testclient import TestClient

    return TestClient(app, base_url="http://localhost")


@pytest.fixture
def lenient_client(app):
    """Same as `client`, but unhandled exceptions surface as the app's 500 JSON body."""
    from fastapi.testclient import TestClient

    return TestClient(app, base_url="http://localhost", raise_server_exceptions=False)


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """
    Run the request from an empty working directory.

    The data layer resolves everything relative to CWD ("./data/dashboard_data.json",
    "./data/amore_data.db", "./data/external_signals", StateManager "./data", ...), so
    chdir-ing to a temp dir both isolates the test and keeps destructive routes
    (e.g. DELETE /api/signals/clear) away from the repository's data/ directory.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Dashboard JSON shaped like src/tools/exporters/dashboard_exporter.py
# ---------------------------------------------------------------------------

DASHBOARD_TOP_LEVEL_KEYS = [
    "metadata",
    "data_source",
    "home",
    "brand",
    "categories",
    "products",
    "charts",
]


def exporter_shaped_dashboard() -> dict:
    """
    Minimal dict mirroring DashboardExporter.export_dashboard_data() output:
    metadata / data_source / home / brand.kpis / brand.competitors /
    categories{id} / products{asin} (LANEIGE-only, no `brand` key) / charts.
    """
    return {
        "metadata": {
            "generated_at": "2026-09-01T22:30:00+09:00",
            "data_date": "2026-09-01",
            "total_products": 2,
            "laneige_products": 1,
            "ontology_enabled": False,
        },
        "data_source": {
            "platform": "Amazon US Best Sellers",
            "collected_at": "2026-09-01T22:30:00+09:00",
            "snapshot_date": "2026-09-01",
            "disclaimer": "snapshot",
            "url": "https://www.amazon.com/gp/bestsellers/beauty",
        },
        "home": {
            "insight_message": "LANEIGE Lip Sleeping Mask #1 in Lip Care",
            "status": {
                "exposure": "Strong",
                "exposure_type": "success",
                "position": "Top 1",
                "warning_count": 0,
            },
            "action_items": [
                {
                    "asin": "B07GFJWPDQ",
                    "product_name": "LANEIGE Lip Sleeping Mask",
                    "brand_variant": "LANEIGE",
                    "rank": 1,
                    "rank_change": 0,
                    "signal": "순위 #1",
                    "signal_detail": "",
                    "action_tag": "Monitor",
                    "priority": "P1",
                }
            ],
        },
        "brand": {
            "kpis": {
                "sos": 12.5,
                "sos_delta": "+2.1%p",
                "top10_count": 1,
                "avg_rank": 1.0,
                "avg_price": 24.0,
                "hhi": 812.5,
            },
            "competitors": [
                {"brand": "LANEIGE", "sos": 50.0, "avg_rank": 1.0, "product_count": 1},
                {"brand": "Burt's Bees", "sos": 50.0, "avg_rank": 2.0, "product_count": 1},
            ],
        },
        "categories": {
            "lip_care": {
                "name": "Lip Care",
                "sos": 50.0,
                "best_rank": 1,
                "cpi": 150.0,
                "new_competitors": 2,
                "product_count": 2,
                "laneige_count": 1,
            }
        },
        "products": {
            "B07GFJWPDQ": {
                "asin": "B07GFJWPDQ",
                "name": "LANEIGE Lip Sleeping Mask",
                "rank": 1,
                "rank_delta": "유지",
                "rating": 4.6,
                "volatility_status": "안정적",
                "price": 24.0,
                "category": "lip_care",
            }
        },
        "charts": {},
    }


@pytest.fixture
def dashboard_file(isolated_cwd):
    """Write a valid exporter-shaped ./data/dashboard_data.json in the isolated CWD."""
    data_dir = isolated_cwd / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "dashboard_data.json"
    path.write_text(json.dumps(exporter_shaped_dashboard(), ensure_ascii=False), encoding="utf-8")
    return path


TEST_API_KEY = "characterization-api-key"  # pragma: allowlist secret


@pytest.fixture
def configured_api_key(monkeypatch):
    """Configure API_KEY in the process environment (read by verify_api_key at call time)."""
    monkeypatch.setenv("API_KEY", TEST_API_KEY)
    return TEST_API_KEY


@pytest.fixture
def auth_headers(configured_api_key):
    return {"X-API-Key": configured_api_key}


@pytest.fixture
def reset_rate_limits():
    """slowapi keeps per-IP counters in process memory; clear them so tests never 429."""
    from src.api.dependencies import limiter

    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def auth_bypass(app):
    """
    Public FastAPI test hook: override the verify_api_key dependency so the chat
    routes can be exercised while the process has no API_KEY configured (which
    would otherwise short-circuit with 503, pinned separately).
    """
    from src.api.dependencies import verify_api_key

    app.dependency_overrides[verify_api_key] = lambda: "characterization-key"
    yield
    app.dependency_overrides.pop(verify_api_key, None)


class _CannedMessage:
    content = "캔 응답: LANEIGE 립 슬리핑 마스크는 Lip Care 1위입니다."


class _CannedChoice:
    message = _CannedMessage()
    finish_reason = "stop"


class _CannedUsage:
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class CannedCompletion:
    """Minimal object shaped like a litellm ModelResponse (see src/shared/llm_client.py)."""

    choices = [_CannedChoice()]
    usage = _CannedUsage()


@pytest.fixture
def patched_llm(monkeypatch):
    """
    No-network LLM: replace litellm.acompletion with a canned coroutine.

    Callers do `from litellm import acompletion` at import time, so the name is
    already bound inside their modules; rebind every loaded module attribute that
    still points at the original function. This is generic (no module is named)
    and is the only way to make the public patch of litellm.acompletion effective.
    """
    import litellm

    original = litellm.acompletion
    calls: list[dict] = []

    async def fake_acompletion(*args, **kwargs):
        calls.append(kwargs)
        return CannedCompletion()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    for module in list(sys.modules.values()):
        if getattr(module, "acompletion", None) is original:
            monkeypatch.setattr(module, "acompletion", fake_acompletion, raising=False)
    return calls
