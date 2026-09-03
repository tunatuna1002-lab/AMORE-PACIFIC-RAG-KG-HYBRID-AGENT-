"""
API characterization fixtures
=============================
Fixtures for pinning the public HTTP surface of `src.api.dashboard_api:app`.
Imported explicitly by the API characterization test modules (not via conftest)
so the module-level environment setup below runs before `src.api.dependencies`
is imported for the first time. Nothing under src/ is modified.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

# --- Process-level environment (must happen BEFORE src.api.dependencies is imported) ---
# `src.api.dependencies.API_KEY` is read once at import time. The characterization
# suite pins the *no-API_KEY* (development) behaviour of verify_api_key, so make sure
# a shell-exported API_KEY cannot leak in. .env.test (loaded by the root conftest) does
# not define API_KEY either.
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
