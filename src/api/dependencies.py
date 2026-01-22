"""
API Dependencies
================
공통 의존성 모듈 (인증, 레이트리밋, 세션 관리 등)
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
from pathlib import Path
import json

from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

# ============= API Key 인증 =============

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logging.warning("⚠️ API_KEY 환경변수가 설정되지 않았습니다.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """API Key 검증 (민감한 엔드포인트용)"""
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API Key가 필요합니다. 헤더에 X-API-Key를 추가하세요."
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="유효하지 않은 API Key입니다."
        )
    return api_key


# ============= Rate Limiter =============

limiter = Limiter(key_func=get_remote_address)


# ============= Session Management =============

conversation_memory: Dict[str, List[Dict[str, str]]] = defaultdict(list)
session_last_activity: Dict[str, datetime] = {}
MAX_MEMORY_TURNS = 10
SESSION_TTL_HOURS = 1
MAX_SESSIONS = 1000


def cleanup_expired_sessions() -> int:
    """만료된 세션 정리"""
    now = datetime.now()
    expired = [
        sid for sid, last_time in session_last_activity.items()
        if (now - last_time).total_seconds() > SESSION_TTL_HOURS * 3600
    ]
    for sid in expired:
        if sid in conversation_memory:
            del conversation_memory[sid]
        if sid in session_last_activity:
            del session_last_activity[sid]
    return len(expired)


def get_conversation_history(session_id: str, limit: int = 5) -> str:
    """대화 기록 조회"""
    history = conversation_memory.get(session_id, [])[-limit:]
    if not history:
        return ""

    lines = []
    for turn in history:
        role = "사용자" if turn["role"] == "user" else "AI"
        content = turn["content"][:150] + "..." if len(turn["content"]) > 150 else turn["content"]
        lines.append(f"[{role}]: {content}")

    return "\n".join(lines)


def add_to_memory(session_id: str, role: str, content: str) -> None:
    """대화 메모리에 추가"""
    now = datetime.now()

    if len(session_last_activity) > MAX_SESSIONS or len(session_last_activity) % 100 == 0:
        cleanup_expired_sessions()

    session_last_activity[session_id] = now
    conversation_memory[session_id].append({
        "role": role,
        "content": content,
        "timestamp": now.isoformat()
    })

    if len(conversation_memory[session_id]) > MAX_MEMORY_TURNS * 2:
        conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_TURNS * 2:]


# ============= Audit Trail Logger =============

AUDIT_LOG_DIR = "./logs"


def setup_audit_logger():
    """Audit Trail 로거 설정"""
    Path(AUDIT_LOG_DIR).mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = Path(AUDIT_LOG_DIR) / f"chatbot_audit_{today}.log"

    audit_logger = logging.getLogger("audit_trail")
    audit_logger.setLevel(logging.INFO)
    audit_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    audit_logger.addHandler(file_handler)

    return audit_logger


audit_logger = setup_audit_logger()


def log_chat_interaction(
    session_id: str,
    user_query: str,
    ai_response: str,
    query_type: str,
    confidence: float,
    entities: Dict,
    sources: List[str],
    response_time_ms: float
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
        "response_time_ms": round(response_time_ms, 2)
    }
    audit_logger.info(json.dumps(audit_entry, ensure_ascii=False))


# ============= Data Helpers =============

DATA_PATH = "./data/dashboard_data.json"


def load_dashboard_data() -> Dict:
    """대시보드 데이터 로드"""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
