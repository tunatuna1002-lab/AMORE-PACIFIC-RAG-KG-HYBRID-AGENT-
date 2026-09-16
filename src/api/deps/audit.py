"""
Audit trail
===========
Chat interaction audit log. The file handler is attached lazily on first use
(importing this module no longer creates ``./logs`` or opens a file).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

AUDIT_LOG_DIR = "./logs"
AUDIT_LOGGER_NAME = "audit_trail"

audit_logger = logging.getLogger(AUDIT_LOGGER_NAME)
_handler_installed = False


def setup_audit_logger() -> logging.Logger:
    """Audit Trail 로거 설정 (파일 핸들러 부착; 호출할 때마다 오늘 날짜 파일로 재설정)"""
    global _handler_installed
    Path(AUDIT_LOG_DIR).mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = Path(AUDIT_LOG_DIR) / f"chatbot_audit_{today}.log"

    audit_logger.setLevel(logging.INFO)
    for handler in list(audit_logger.handlers):
        audit_logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    audit_logger.addHandler(file_handler)
    _handler_installed = True
    return audit_logger


def get_audit_logger() -> logging.Logger:
    """Audit logger, installing the file handler on first use."""
    if not _handler_installed:
        try:
            setup_audit_logger()
        except OSError as e:  # read-only FS etc. - keep serving requests
            logging.getLogger(__name__).warning(f"Audit log file unavailable: {e}")
    return audit_logger


def log_chat_interaction(
    session_id: str,
    user_query: str,
    ai_response: str,
    query_type: str,
    confidence: float,
    entities: dict,
    sources: list[str],
    response_time_ms: float,
):
    """챗봇 대화 Audit Trail 기록"""
    audit_entry = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "ai_response": ai_response[:500] + "..." if len(ai_response) > 500 else ai_response,
        "query_type": query_type,
        "confidence": round(confidence, 4),
        "entities": entities,
        "sources": sources,
        "response_time_ms": round(response_time_ms, 2),
    }
    get_audit_logger().info(json.dumps(audit_entry, ensure_ascii=False))
