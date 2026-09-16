"""
Session memory (API)
====================
ONE conversation memory for the API (F7): a module-level
``src.memory.conversation_memory.ConversationMemory`` configured with the API's
limits (turns per session, session TTL, max sessions). The former ad-hoc dicts in
``dependencies.py`` are gone; ``add_to_memory`` / ``get_conversation_history`` /
``cleanup_expired_sessions`` keep their signatures on top of it.

Expired sessions are evicted on every ``add_to_memory``.
"""

from __future__ import annotations

from src.memory.conversation_memory import ConversationMemory

MAX_MEMORY_TURNS = 10  # user+assistant pairs kept per session (stored: 2x)
SESSION_TTL_HOURS = 1
MAX_SESSIONS = 1000
HISTORY_TURNS_FOR_BRAIN = 6  # last N turns handed to the brain with each query
HISTORY_PREVIEW_CHARS = 150

conversation_memory = ConversationMemory(
    max_recent_turns=HISTORY_TURNS_FOR_BRAIN,
    max_sessions=MAX_SESSIONS,
    max_turns_per_session=MAX_MEMORY_TURNS * 2,
    ttl_hours=SESSION_TTL_HOURS,
)

# Test-reset hook (the root conftest clears it between tests). Same dict object the
# memory uses for TTL bookkeeping - do not rebind.
session_last_activity = conversation_memory._last_activity


def get_memory() -> ConversationMemory:
    """The API-wide conversation memory."""
    return conversation_memory


def cleanup_expired_sessions() -> int:
    """만료된 세션 정리 (제거된 세션 수)"""
    return conversation_memory.cleanup_expired_sessions()


def add_to_memory(session_id: str, role: str, content: str) -> None:
    """대화 메모리에 추가 (만료 세션은 매 호출 시 정리)"""
    conversation_memory.add_turn(session_id, role, content)


def clear_session(session_id: str) -> bool:
    """세션 대화 기록 삭제. 세션이 존재했으면 True."""
    existed = conversation_memory.has_session(session_id)
    conversation_memory.clear_session(session_id)
    return existed


def get_recent_turns(
    session_id: str, limit: int = HISTORY_TURNS_FOR_BRAIN
) -> list[dict[str, str]]:
    """최근 ``limit``개 턴 (``{"role", "content"}``) - 브레인/LLM 전달용"""
    return conversation_memory.get_recent_turns(session_id, limit=limit)


def get_conversation_history(session_id: str, limit: int = 5) -> str:
    """대화 기록 조회 (프롬프트용 텍스트: ``[사용자]: ...`` / ``[AI]: ...``)"""
    history = conversation_memory.get_recent_turns(session_id, limit=limit)
    if not history:
        return ""

    lines = []
    for turn in history:
        role = "사용자" if turn["role"] == "user" else "AI"
        content = turn["content"]
        if len(content) > HISTORY_PREVIEW_CHARS:
            content = content[:HISTORY_PREVIEW_CHARS] + "..."
        lines.append(f"[{role}]: {content}")

    return "\n".join(lines)
