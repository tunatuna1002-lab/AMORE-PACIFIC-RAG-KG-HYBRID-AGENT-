"""
Health Check Routes
===================
헬스체크 및 루트 엔드포인트 (/,  /dashboard, /api/health, /api/health/deep)
"""

import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from src.api.dependencies import limiter
from src.core.brain import get_initialized_brain

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

# Dashboard read-only token (서버 API_KEY 노출 방지)
DASHBOARD_READ_TOKEN = os.getenv("DASHBOARD_READ_TOKEN", "")

DASHBOARD_HTML_PATH = Path("./dashboard/amore_unified_dashboard_v4.html")

# 대시보드 HTML은 프로세스 수명 동안 변하지 않으므로 모듈 레벨에서 1회만 읽어 캐시한다.
_DASHBOARD_HTML_CACHE: dict[str, str] = {}


def _read_html_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_dashboard_html(path: Path) -> str:
    """대시보드 HTML을 읽어 모듈 레벨 캐시에 보관 (경로별 1회 읽기)"""
    key = str(path.resolve())
    html = _DASHBOARD_HTML_CACHE.get(key)
    if html is None:
        html = _read_html_file(path)
        _DASHBOARD_HTML_CACHE[key] = html
    return html


@router.get("/")
@limiter.limit("60/minute")
async def root(request: Request):
    """헬스 체크"""
    return {
        "status": "ok",
        "message": "AMORE Dashboard API v2.0 (RAG + Ontology)",
        "features": ["chatbot", "rag", "ontology", "memory", "docx_export"],
    }


@router.get("/dashboard")
@limiter.limit("60/minute")
async def serve_dashboard(request: Request):
    """
    대시보드 HTML 페이지 서빙 (API 키 자동 주입)

    서버의 API_KEY를 HTML에 자동으로 주입하여
    프론트엔드에서 별도 설정 없이 인증된 API 호출 가능
    """
    dashboard_path = DASHBOARD_HTML_PATH
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")

    html_content = _load_dashboard_html(dashboard_path)

    # 대시보드 전용 읽기 토큰만 주입 (서버 API_KEY는 절대 노출하지 않음)
    if DASHBOARD_READ_TOKEN:
        api_key_script = (
            f'<script>window.DASHBOARD_API_KEY = "{DASHBOARD_READ_TOKEN}";</script>\n</head>'
        )
        html_content = html_content.replace("</head>", api_key_script)

    return HTMLResponse(content=html_content, media_type="text/html")


@router.get("/api/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """기본 헬스 체크 엔드포인트 (Railway healthcheck용)"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.get("/api/health/deep")
@limiter.limit("60/minute")
async def deep_health_check(request: Request):
    """
    Deep Health Check - 모든 서브시스템 상태 확인

    Returns:
        - database: SQLite 연결 상태
        - knowledge_graph: KG 로드 상태 및 트리플 수
        - llm: OpenAI API 연결 상태
        - scheduler: 자율 스케줄러 상태
        - memory: 시스템 메모리 사용량
        - disk: 디스크 사용량 (Railway Volume)
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "warnings": [],
    }

    # 1. SQLite 연결 확인
    try:
        db_path = (
            Path("/data/amore_data.db") if Path("/data").exists() else Path("data/amore_data.db")
        )
        if db_path.exists():
            conn = sqlite3.connect(str(db_path), timeout=5)
            cursor = conn.execute("SELECT COUNT(*) FROM raw_data")
            count = cursor.fetchone()[0]
            conn.close()
            health["checks"]["database"] = {
                "status": "healthy",
                "records": count,
                "path": str(db_path),
            }
        else:
            health["checks"]["database"] = {"status": "missing", "path": str(db_path)}
            health["warnings"].append("SQLite database not found")
    except Exception as e:
        health["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"

    # 2. Knowledge Graph 상태
    try:
        from src.ontology.knowledge_graph import get_knowledge_graph

        kg = get_knowledge_graph()
        triple_count = len(kg.triples) if kg.triples else 0
        health["checks"]["knowledge_graph"] = {
            "status": "healthy" if triple_count > 0 else "empty",
            "triples": triple_count,
            "max_triples": kg.max_triples,
        }
        if triple_count == 0:
            health["warnings"].append("Knowledge Graph is empty")
    except Exception as e:
        health["checks"]["knowledge_graph"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"

    # 3. LLM API 연결 (OpenAI)
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key and api_key.startswith("sk-"):
            health["checks"]["llm"] = {
                "status": "configured",
                "provider": "openai",
                "key_prefix": api_key[:10] + "...",
            }
        else:
            health["checks"]["llm"] = {"status": "not_configured"}
            health["warnings"].append("OPENAI_API_KEY not properly configured")
    except Exception as e:
        health["checks"]["llm"] = {"status": "error", "error": str(e)}

    # 4. 스케줄러 상태
    try:
        brain = await get_initialized_brain()
        scheduler_running = brain.scheduler.is_running if brain.scheduler else False
        health["checks"]["scheduler"] = {
            "status": "running" if scheduler_running else "stopped",
            "mode": brain.mode.value if brain.mode else "unknown",
        }
    except Exception as e:
        health["checks"]["scheduler"] = {"status": "error", "error": str(e)}

    # 5. 메모리 사용량
    try:
        import psutil

        memory = psutil.virtual_memory()
        health["checks"]["memory"] = {
            "status": "healthy" if memory.percent < 90 else "warning",
            "used_percent": round(memory.percent, 1),
            "available_gb": round(memory.available / (1024**3), 2),
        }
        if memory.percent > 90:
            health["warnings"].append(f"High memory usage: {memory.percent}%")
            health["status"] = "degraded"
    except ImportError:
        health["checks"]["memory"] = {"status": "unknown", "note": "psutil not installed"}

    # 6. 디스크 사용량 (Railway Volume)
    try:
        import shutil

        data_path = Path("/data") if Path("/data").exists() else Path("data")
        if data_path.exists():
            total, used, free = shutil.disk_usage(data_path)
            used_percent = (used / total) * 100
            health["checks"]["disk"] = {
                "status": "healthy" if used_percent < 90 else "warning",
                "used_percent": round(used_percent, 1),
                "free_gb": round(free / (1024**3), 2),
                "path": str(data_path),
            }
            if used_percent > 90:
                health["warnings"].append(f"Low disk space: {100 - used_percent:.1f}% free")
                health["status"] = "degraded"
    except Exception as e:
        health["checks"]["disk"] = {"status": "error", "error": str(e)}

    # 최종 상태 결정
    unhealthy_checks = [k for k, v in health["checks"].items() if v.get("status") == "unhealthy"]
    if unhealthy_checks:
        health["status"] = "unhealthy"

    return health
