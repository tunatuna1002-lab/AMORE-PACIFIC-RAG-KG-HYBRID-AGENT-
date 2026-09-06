"""
Shared fixtures for the API unit tests.

The app/client/auth/LLM fakes live in tests/characterization/_api_fixtures.py so the
characterization suite and the unit-level fix tests exercise the *same* FastAPI app
(`src.api.dashboard_api:app`) through the same TestClient configuration
(base_url=http://localhost, no lifespan). Importing the fixture functions here
registers them for every module under tests/unit/api/.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any

import pytest

from tests.characterization._api_fixtures import (  # noqa: F401
    app,
    auth_bypass,
    auth_headers,
    client,
    configured_api_key,
    dashboard_file,
    exporter_shaped_dashboard,
    isolated_cwd,
    lenient_client,
    patched_llm,
    reset_rate_limits,
)


@pytest.fixture
def fresh_job_queue(isolated_cwd, monkeypatch):  # noqa: F811
    """A JobQueue bound to the isolated cwd (get_job_queue() is a process-wide singleton)."""
    from src.tools.utilities import job_queue as jq

    queue = jq.JobQueue(db_path=str(isolated_cwd / "data" / "job_queue.db"))
    monkeypatch.setattr(jq, "_job_queue", queue)
    return queue


class FakeCursor:
    def __init__(self, rows: list[Any]):
        self._rows = rows

    def fetchall(self) -> list[Any]:
        return list(self._rows)

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None


class FakeConn:
    """Minimal sqlite3.Connection stand-in that records the thread of each execute()."""

    def __init__(self, storage: FakeSqliteStorage):
        self._storage = storage

    def execute(self, query: str, params: tuple = ()) -> FakeCursor:
        self._storage.executed.append((query, params, threading.get_ident()))
        return FakeCursor(self._storage.rows_for(query))

    def commit(self) -> None:
        pass


class FakeSqliteStorage:
    """Stand-in for src.tools.storage.sqlite_storage.SQLiteStorage used by the routes.

    Routes call: await initialize(), and the *synchronous* get_connection() context
    manager. `executed` records (query, params, thread_ident) per execute() call.
    """

    def __init__(self, rows: list[Any] | None = None):
        self.rows = rows or []
        self.executed: list[tuple[str, tuple, int]] = []
        self.get_connection_thread_idents: list[int] = []

    def rows_for(self, query: str) -> list[Any]:
        return self.rows

    async def initialize(self) -> None:
        return None

    @contextmanager
    def get_connection(self):
        self.get_connection_thread_idents.append(threading.get_ident())
        yield FakeConn(self)

    async def get_unsent_alerts(self, limit: int = 50) -> list[dict]:
        return []

    async def get_deals_summary(self, days: int = 7) -> dict:
        return {}
