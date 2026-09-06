"""
Fix D22 - slowapi requires the starlette Request parameter to be named `request`
=================================================================================
`@limiter.limit(...)` inspects the handler signature for a parameter named `request`
that is a starlette Request. Handlers that named the pydantic body `request` and the
starlette object `http_request` failed on every call with a 500.
"""

from __future__ import annotations

import inspect

from starlette.requests import Request


def _endpoints(router):
    for route in router.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None:
            yield route, endpoint


def test_export_rate_limited_handlers_name_the_starlette_request_parameter_request():
    from src.api.routes import export

    for route, endpoint in _endpoints(export.router):
        params = inspect.signature(endpoint).parameters
        assert "http_request" not in params, route.path
        assert "request" in params, route.path
        assert params["request"].annotation is Request, route.path


def test_no_app_handler_uses_request_for_a_non_starlette_parameter(app):
    for route, endpoint in _endpoints(app.router):
        params = inspect.signature(endpoint).parameters
        if "request" in params:
            annotation = params["request"].annotation
            assert annotation is Request or annotation is inspect.Parameter.empty, route.path


def test_export_docx_no_longer_500s_on_the_limiter(
    lenient_client, dashboard_file, auth_headers, reset_rate_limits
):
    r = lenient_client.post(
        "/api/export/docx", json={"include_external_signals": False}, headers=auth_headers
    )
    assert r.status_code == 200, r.text


def test_export_async_start_accepts_payload_named_body(
    client, isolated_cwd, auth_headers, reset_rate_limits, fresh_job_queue
):
    r = client.post(
        "/api/export/async/start", json={"job_type": "export_docx"}, headers=auth_headers
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "pending"
