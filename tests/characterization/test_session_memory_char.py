"""
Session / conversation memory characterization
==============================================
Pins the public module functions in src/api/dependencies.py
(add_to_memory / get_conversation_history / cleanup_expired_sessions) and
src/memory/context.py (ContextManager).

The root conftest clears `conversation_memory` / `session_last_activity` after
every test, so each test starts from an empty store.
"""

from datetime import datetime, timedelta

import pytest

from src.api import dependencies as deps
from src.api.dependencies import (
    MAX_MEMORY_TURNS,
    SESSION_TTL_HOURS,
    add_to_memory,
    cleanup_expired_sessions,
    get_conversation_history,
)

# ---------------------------------------------------------------------------
# add_to_memory / get_conversation_history
# ---------------------------------------------------------------------------


def test_constants_are_pinned():
    assert MAX_MEMORY_TURNS == 10
    assert SESSION_TTL_HOURS == 1
    assert deps.MAX_SESSIONS == 1000


def test_history_contains_added_user_message():
    add_to_memory("s1", "user", "hi")
    history = get_conversation_history("s1")
    assert "hi" in history
    assert history == "[사용자]: hi"


def test_history_labels_roles_and_preserves_order():
    add_to_memory("s1", "user", "질문")
    add_to_memory("s1", "assistant", "답변")
    assert get_conversation_history("s1") == "[사용자]: 질문\n[AI]: 답변"


def test_non_user_roles_are_all_labelled_ai():
    # Any role other than "user" renders as "AI" (system, tool, ...).
    add_to_memory("s1", "system", "sys")
    add_to_memory("s1", "tool", "tool-out")
    assert get_conversation_history("s1") == "[AI]: sys\n[AI]: tool-out"


def test_unknown_session_returns_empty_string():
    assert get_conversation_history("never-seen") == ""


def test_long_content_is_truncated_to_150_chars_with_ellipsis():
    add_to_memory("s1", "user", "x" * 151)
    add_to_memory("s1", "user", "y" * 150)
    lines = get_conversation_history("s1").split("\n")
    assert lines[0] == "[사용자]: " + "x" * 150 + "..."
    assert lines[1] == "[사용자]: " + "y" * 150  # exactly 150 is not truncated


def test_history_default_limit_is_last_five_turns():
    for i in range(7):
        add_to_memory("s1", "user", f"m{i}")
    lines = get_conversation_history("s1").split("\n")
    assert lines == [f"[사용자]: m{i}" for i in range(2, 7)]
    assert len(get_conversation_history("s1", limit=3).split("\n")) == 3


def test_memory_is_trimmed_to_twice_max_memory_turns():
    total = MAX_MEMORY_TURNS * 2 + 5
    for i in range(total):
        add_to_memory("s1", "user", f"m{i}")

    stored = deps.conversation_memory["s1"]
    assert len(stored) == MAX_MEMORY_TURNS * 2
    assert stored[0]["content"] == "m5"  # oldest 5 dropped
    assert stored[-1]["content"] == f"m{total - 1}"
    assert set(stored[0].keys()) == {"role", "content", "timestamp"}

    lines = get_conversation_history("s1", limit=1000).split("\n")
    assert len(lines) == MAX_MEMORY_TURNS * 2
    assert lines[0] == "[사용자]: m5"


def test_sessions_are_isolated():
    add_to_memory("a", "user", "for-a")
    add_to_memory("b", "user", "for-b")
    assert get_conversation_history("a") == "[사용자]: for-a"
    assert get_conversation_history("b") == "[사용자]: for-b"


# ---------------------------------------------------------------------------
# cleanup_expired_sessions
# ---------------------------------------------------------------------------


@pytest.fixture
def frozen_clock(monkeypatch):
    """
    There is no injectable clock in src/api/dependencies.py — `datetime.now()` is called
    directly — so this is the ONE allowed internal patch: replace the module's `datetime`
    name with a subclass whose now() returns a controllable instant.
    """
    state = {"now": datetime(2026, 9, 3, 12, 0, 0)}

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401 - mirrors datetime.now signature
            return state["now"]

    monkeypatch.setattr(deps, "datetime", _FrozenDateTime)

    def advance(**kwargs):
        state["now"] = state["now"] + timedelta(**kwargs)

    return advance


def test_cleanup_removes_sessions_older_than_ttl(frozen_clock):
    add_to_memory("old", "user", "stale")
    frozen_clock(hours=SESSION_TTL_HOURS, seconds=1)  # strictly older than TTL
    add_to_memory("fresh", "user", "recent")

    removed = cleanup_expired_sessions()

    assert removed == 1
    assert get_conversation_history("old") == ""
    assert "old" not in deps.conversation_memory
    assert "old" not in deps.session_last_activity
    assert get_conversation_history("fresh") == "[사용자]: recent"


def test_cleanup_keeps_session_exactly_at_ttl_boundary(frozen_clock):
    add_to_memory("edge", "user", "x")
    frozen_clock(hours=SESSION_TTL_HOURS)  # == TTL, comparison is strict ">"
    assert cleanup_expired_sessions() == 0
    assert get_conversation_history("edge") == "[사용자]: x"


def test_cleanup_on_empty_store_returns_zero():
    assert cleanup_expired_sessions() == 0


def test_add_to_memory_does_not_evict_expired_sessions_on_every_call(frozen_clock):
    """
    PINS CURRENT BEHAVIOR: add_to_memory only triggers cleanup when the session count
    is > MAX_SESSIONS or a multiple of 100 (0 included). With a single expired session
    present, adding a second session leaves the expired one in place.
    """
    add_to_memory("old", "user", "stale")
    frozen_clock(hours=2)
    add_to_memory("new", "user", "fresh")  # len(session_last_activity) == 1 -> no cleanup
    assert "old" in deps.conversation_memory
    assert get_conversation_history("old") == "[사용자]: stale"


def test_add_to_memory_refreshes_last_activity(frozen_clock):
    add_to_memory("s1", "user", "one")
    frozen_clock(minutes=50)
    add_to_memory("s1", "user", "two")  # touch resets the TTL window
    frozen_clock(minutes=50)
    assert cleanup_expired_sessions() == 0
    assert get_conversation_history("s1") == "[사용자]: one\n[사용자]: two"


# ---------------------------------------------------------------------------
# src/memory/context.py ContextManager
# ---------------------------------------------------------------------------


@pytest.fixture
def context_manager(tmp_path):
    from src.memory.context import ContextManager

    return ContextManager(context_dir=str(tmp_path / "ctx"))


def test_context_manager_creates_context_dir(tmp_path, context_manager):
    assert (tmp_path / "ctx").is_dir()


def test_context_manager_history_limit_one_returns_last_turn_dict(context_manager):
    context_manager.add_user_message("질문입니다")
    context_manager.add_assistant_message("답변입니다", metadata={"confidence": 0.9})

    history = context_manager.get_conversation_history(limit=1)
    assert isinstance(history, list)
    assert len(history) == 1
    turn = history[0]
    assert set(turn.keys()) == {"role", "content", "timestamp", "metadata"}
    assert turn["role"] == "assistant"
    assert turn["content"] == "답변입니다"
    assert turn["metadata"] == {"confidence": 0.9}
    assert datetime.fromisoformat(turn["timestamp"])  # ISO-8601 timestamp


def test_context_manager_history_default_limit_and_order(context_manager):
    context_manager.add_user_message("q")
    context_manager.add_assistant_message("a")
    history = context_manager.get_conversation_history()
    assert [(t["role"], t["content"]) for t in history] == [("user", "q"), ("assistant", "a")]
    assert history[0]["metadata"] == {}  # None metadata becomes {}


def test_context_manager_empty_history_and_summary(context_manager):
    assert context_manager.get_conversation_history(limit=1) == []
    assert context_manager.get_conversation_summary() == "이전 대화 없음"


def test_context_manager_summary_uses_last_five_turns_with_100_char_truncation(context_manager):
    for i in range(6):
        context_manager.add_user_message(f"m{i}")
    context_manager.add_assistant_message("z" * 101)
    summary = context_manager.get_conversation_summary().split("\n")
    assert summary == [
        "[사용자]: m2",
        "[사용자]: m3",
        "[사용자]: m4",
        "[사용자]: m5",
        "[어시스턴트]: " + "z" * 100 + "...",
    ]


def test_context_manager_trims_to_100_turns(context_manager):
    for i in range(105):
        context_manager.add_user_message(f"m{i}")
    history = context_manager.get_conversation_history(limit=1000)
    assert len(history) == 100
    assert history[0]["content"] == "m5"
    assert history[-1]["content"] == "m104"
