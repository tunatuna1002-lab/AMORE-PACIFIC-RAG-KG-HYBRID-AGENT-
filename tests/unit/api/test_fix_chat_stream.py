"""
Fix D23 - /api/v4/chat/stream always answers as an event stream
================================================================
When the brain fails to initialise (offline: no OPENAI_API_KEY) the route used to
raise HTTPException(500) with a JSON body. Clients that already opened an
EventSource expect `text/event-stream` and an `error` frame.
"""

from __future__ import annotations

import json


def _frames(body: str) -> list[dict]:
    out = []
    for line in body.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: ") :]))
    return out


def test_stream_emits_sse_error_frame_when_brain_fails(
    client, isolated_cwd, auth_bypass, patched_llm
):
    r = client.post("/api/v4/chat/stream", json={"message": "안녕"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers.get("cache-control") == "no-cache"

    frames = _frames(r.text)
    assert frames, r.text
    assert frames[-1]["type"] == "error"
    assert frames[-1]["content"].startswith("Vector search is required but not available")
    assert patched_llm == []


def test_stream_still_rejects_blank_message_before_streaming(client, isolated_cwd, auth_bypass):
    r = client.post("/api/v4/chat/stream", json={"message": " "})
    assert r.status_code == 400
    assert r.json() == {"detail": "Message is required"}
