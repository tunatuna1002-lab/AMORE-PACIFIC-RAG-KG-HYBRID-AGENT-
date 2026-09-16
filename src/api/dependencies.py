"""
API Dependencies (facade)
=========================
Backwards-compatible re-export of the dependency modules under ``src/api/deps``:

- ``deps.auth``        verify_api_key, limiter, JWT helpers
- ``deps.session``     one ConversationMemory: add_to_memory / get_conversation_history / ...
- ``deps.audit``       log_chat_interaction (file handler attached lazily)
- ``deps.data``        load_dashboard_data (DashboardDataService), build_data_context
- ``deps.suggestions`` generate_dynamic_suggestions
- ``deps.providers``   get_rag_context, get_app_state_manager, get_sheets_writer, ...

Importing this module has no side effects beyond the production startup guards
(API_KEY / JWT_SECRET_KEY presence checks in ``deps.auth``).
"""

from __future__ import annotations

from typing import Any

from src.api.deps import providers as _providers
from src.api.deps.audit import (
    AUDIT_LOG_DIR,
    audit_logger,
    get_audit_logger,
    log_chat_interaction,
    setup_audit_logger,
)
from src.api.deps.auth import (
    API_KEY,
    EMAIL_VERIFICATION_EXPIRES_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    api_key_header,
    create_email_verification_token,
    get_configured_api_key,
    limiter,
    verify_api_key,
    verify_jwt_email_token,
)
from src.api.deps.data import (
    DOCS_PATH,
    build_data_context,
    dashboard_data_path,
    get_dashboard_data,
    get_data_service,
    load_dashboard_data,
    resolve_data_dir,
)
from src.api.deps.providers import (
    get_app_state_manager,
    get_base_url,
    get_doc_retriever,
    get_market_intelligence,
    get_rag_context,
    get_rag_router,
    get_sheets_writer,
)
from src.api.deps.session import (
    HISTORY_TURNS_FOR_BRAIN,
    MAX_MEMORY_TURNS,
    MAX_SESSIONS,
    SESSION_TTL_HOURS,
    add_to_memory,
    cleanup_expired_sessions,
    clear_session,
    conversation_memory,
    get_conversation_history,
    get_memory,
    get_recent_turns,
    session_last_activity,
)
from src.api.deps.suggestions import (
    _extract_response_keywords,
    _generate_entity_suggestions,
    _generate_type_suggestions,
    generate_dynamic_suggestions,
)

__all__ = [
    "API_KEY",
    "AUDIT_LOG_DIR",
    "DOCS_PATH",
    "EMAIL_VERIFICATION_EXPIRES_MINUTES",
    "HISTORY_TURNS_FOR_BRAIN",
    "JWT_ALGORITHM",
    "JWT_SECRET_KEY",
    "MAX_MEMORY_TURNS",
    "MAX_SESSIONS",
    "SESSION_TTL_HOURS",
    "add_to_memory",
    "api_key_header",
    "audit_logger",
    "build_data_context",
    "cleanup_expired_sessions",
    "clear_session",
    "conversation_memory",
    "create_email_verification_token",
    "dashboard_data_path",
    "generate_dynamic_suggestions",
    "get_app_state_manager",
    "get_audit_logger",
    "get_base_url",
    "get_configured_api_key",
    "get_conversation_history",
    "get_dashboard_data",
    "get_data_service",
    "get_doc_retriever",
    "get_market_intelligence",
    "get_memory",
    "get_rag_context",
    "get_rag_router",
    "get_recent_turns",
    "get_sheets_writer",
    "limiter",
    "load_dashboard_data",
    "log_chat_interaction",
    "resolve_data_dir",
    "session_last_activity",
    "setup_audit_logger",
    "verify_api_key",
    "verify_jwt_email_token",
    "_extract_response_keywords",
    "_generate_entity_suggestions",
    "_generate_type_suggestions",
]


def __getattr__(name: str) -> Any:
    """Legacy attributes kept lazy: RAG objects and the resolved data paths."""
    if name in ("rag_router", "doc_retriever"):
        return getattr(_providers, name)
    if name == "RESOLVED_DATA_DIR":
        return str(resolve_data_dir())
    if name == "DATA_PATH":
        return str(dashboard_data_path())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
